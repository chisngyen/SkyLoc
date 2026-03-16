# ============================================================================
# EXP02 — UAV-AVL Baseline: DINOv2 + Reference-Map Tile Matching
# ============================================================================
# Self-contained script for Kaggle H100.
# Dataset: hunhtrungkit/uav-avl
# Backbone: DINOv2 ViT-B/14
# Loss: InfoNCE with GPS-distance-based positive/negative assignment
# Metrics: R@1/5/10, PDM@K @5m/@10m/@25m, Mean/Median error, Inference timing
# ============================================================================

# === SETUP ===
import subprocess, sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

for pkg in ["timm", "faiss-gpu", "rasterio", "scipy"]:
    try:
        __import__(pkg.replace("-", "_").split("==")[0])
    except ImportError:
        install(pkg)

# === IMPORTS ===
import os
import json
import math
import time
import random
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("[WARN] rasterio not available — will use PIL fallback for .tif")

warnings.filterwarnings("ignore")

# === CONFIG ===
class CFG:
    # Paths (Kaggle)
    DATASET_ROOT = "/kaggle/input/datasets/hunhtrungkit/uav-avl/Data"
    OUTPUT_DIR   = "/kaggle/working"

    # Reference map tiling
    TILE_SIZE    = 256    # pixels
    TILE_STRIDE  = 128    # overlap for denser coverage
    REF_MAP_DIR  = "Reference_map/QZ_Town"
    UAV_DIR      = "UAV_image/QZ_Town"
    META_FILE    = "metadata/QZ_Town.json"

    # Model
    BACKBONE     = "dinov2_vitb14"
    EMBED_DIM    = 512
    IMG_SIZE     = 384

    # Training
    EPOCHS       = 20
    BATCH_SIZE   = 32
    LR           = 1e-4
    WEIGHT_DECAY = 0.03
    WARMUP_EP    = 3
    UNFREEZE_BLOCKS = 4
    TEMPERATURE  = 0.1
    NUM_WORKERS  = 4
    SEED         = 42

    # GPS matching thresholds (meters)
    POS_RADIUS   = 10.0   # positive: tile center within Xm of UAV GPS
    NEG_RADIUS   = 50.0   # hard negative: tile center between pos and neg radius

    # Eval thresholds
    EVAL_THRESHOLDS_M = [5, 10, 25]

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(CFG.SEED)

# ============================================================================
# 1. REFERENCE MAP TILING
# ============================================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_tif_as_pil(path):
    """Load a .tif file as PIL RGB image."""
    if HAS_RASTERIO:
        with rasterio.open(path) as src:
            # Read first 3 bands as RGB
            bands = min(3, src.count)
            data = src.read(list(range(1, bands + 1)))  # (C, H, W)
            if bands == 1:
                data = np.repeat(data, 3, axis=0)
            data = np.transpose(data, (1, 2, 0))  # (H, W, C)
            # Normalize to uint8 if needed
            if data.dtype != np.uint8:
                dmin, dmax = data.min(), data.max()
                if dmax > dmin:
                    data = ((data - dmin) / (dmax - dmin) * 255).astype(np.uint8)
                else:
                    data = np.zeros_like(data, dtype=np.uint8)
            return Image.fromarray(data)
    else:
        return Image.open(path).convert("RGB")


def get_tif_geotransform(path):
    """Get geotransform from a GeoTIFF → pixel-to-GPS mapping."""
    if HAS_RASTERIO:
        with rasterio.open(path) as src:
            return src.transform, src.width, src.height
    return None, None, None


def tile_reference_map(tif_path, tile_size, stride):
    """Slice a reference map .tif into tiles with GPS coordinates.
    Returns list of (tile_image_pil, center_lat, center_lon, tile_id).
    """
    transform, width, height = get_tif_geotransform(tif_path)

    if transform is None:
        # Fallback: load entire image and tile without GPS
        img = load_tif_as_pil(tif_path)
        width, height = img.size
        tiles = []
        tid = 0
        for y in range(0, height - tile_size + 1, stride):
            for x in range(0, width - tile_size + 1, stride):
                tile = img.crop((x, y, x + tile_size, y + tile_size))
                tiles.append((tile, 0.0, 0.0, f"tile_{tid:06d}"))
                tid += 1
        return tiles

    # With rasterio: extract tiles with GPS
    tiles = []
    tid = 0
    with rasterio.open(tif_path) as src:
        for y in range(0, height - tile_size + 1, stride):
            for x in range(0, width - tile_size + 1, stride):
                window = Window(x, y, tile_size, tile_size)
                data = src.read(list(range(1, min(4, src.count + 1))), window=window)
                if data.shape[0] == 1:
                    data = np.repeat(data, 3, axis=0)
                data = np.transpose(data, (1, 2, 0))
                if data.dtype != np.uint8:
                    dmin, dmax = data.min(), data.max()
                    if dmax > dmin:
                        data = ((data - dmin) / (dmax - dmin) * 255).astype(np.uint8)
                    else:
                        data = np.zeros_like(data, dtype=np.uint8)

                tile_img = Image.fromarray(data)

                # Center pixel → GPS
                cx, cy = x + tile_size // 2, y + tile_size // 2
                lon, lat = src.transform * (cx, cy)

                tiles.append((tile_img, lat, lon, f"tile_{tid:06d}"))
                tid += 1

    return tiles


# ============================================================================
# 2. DATA
# ============================================================================

def parse_smart_pos(filepath):
    """Parse smart_pos.txt → list of (img_filename, lat, lon, alt)."""
    positions = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                fname = parts[0]
                lat, lon = float(parts[1]), float(parts[2])
                alt = float(parts[3]) if len(parts) > 3 else 0.0
                positions.append((fname, lat, lon, alt))
    return positions


def discover_uav_data(uav_root):
    """Discover all UAV images with their GPS from all sub-flights.
    Returns list of (img_path, lat, lon).
    """
    all_samples = []
    for flight in sorted(os.listdir(uav_root)):
        flight_dir = os.path.join(uav_root, flight)
        if not os.path.isdir(flight_dir):
            continue

        # Look for GPS file
        gps_file = None
        for name in ["smart_pos.txt", "smart_pos_photoscan.txt"]:
            p = os.path.join(flight_dir, name)
            if os.path.exists(p):
                gps_file = p
                break

        if gps_file:
            positions = parse_smart_pos(gps_file)
            gps_dict = {fname: (lat, lon) for fname, lat, lon, _ in positions}
        else:
            gps_dict = {}

        # Collect images
        for f in sorted(os.listdir(flight_dir)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(flight_dir, f)
                if f in gps_dict:
                    lat, lon = gps_dict[f]
                    all_samples.append((img_path, lat, lon))
                else:
                    # No GPS for this image — skip for training
                    pass

    return all_samples


class UAVAVLTileDataset(Dataset):
    """Dataset for reference-map tiles (gallery)."""

    def __init__(self, tiles, transform):
        self.tiles = tiles  # list of (pil_img, lat, lon, tile_id)
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        img, lat, lon, tid = self.tiles[idx]
        img = self.transform(img)
        return img, tid, lat, lon


class UAVAVLQueryDataset(Dataset):
    """Dataset for UAV query images."""

    def __init__(self, samples, transform):
        self.samples = samples  # list of (img_path, lat, lon)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lat, lon = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, lat, lon


class UAVAVLTrainDataset(Dataset):
    """Training dataset: pairs UAV images with positive reference tiles."""

    def __init__(self, uav_samples, tiles, transform_uav, transform_tile,
                 pos_radius=CFG.POS_RADIUS):
        self.uav_samples = uav_samples
        self.tiles = tiles
        self.transform_uav  = transform_uav
        self.transform_tile = transform_tile

        # Build tile GPS array for fast matching
        self.tile_gps = np.array([(t[1], t[2]) for t in tiles])

        # Pre-compute positive tile indices for each UAV image
        self.pairs = []
        for uav_idx, (_, ulat, ulon) in enumerate(uav_samples):
            dists = np.array([
                haversine(ulat, ulon, tlat, tlon)
                for tlat, tlon in self.tile_gps
            ])
            pos_indices = np.where(dists <= pos_radius)[0]
            if len(pos_indices) > 0:
                self.pairs.append((uav_idx, pos_indices))

        print(f"[Train] {len(self.pairs)} UAV images with ≥1 positive tile "
              f"(radius={pos_radius}m)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        uav_idx, pos_tile_indices = self.pairs[idx]
        tile_idx = random.choice(pos_tile_indices)

        # UAV image
        img_path, _, _ = self.uav_samples[uav_idx]
        uav_img = Image.open(img_path).convert("RGB")
        uav_img = self.transform_uav(uav_img)

        # Tile image
        tile_img, _, _, _ = self.tiles[tile_idx]
        tile_img = self.transform_tile(tile_img)

        return uav_img, tile_img, idx


# ============================================================================
# 3. MODEL (same as exp01)
# ============================================================================

class GeoModel(nn.Module):
    def __init__(self, backbone_name=CFG.BACKBONE, embed_dim=CFG.EMBED_DIM):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", backbone_name, pretrained=True
        )
        feat_dim = self.backbone.embed_dim
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, embed_dim),
        )
        self._freeze_backbone()

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n):
        for p in self.backbone.norm.parameters():
            p.requires_grad = True
        total_blocks = len(self.backbone.blocks)
        for i in range(total_blocks - n, total_blocks):
            for p in self.backbone.blocks[i].parameters():
                p.requires_grad = True
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[Model] Unfroze last {n} blocks. "
              f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M")

    def forward(self, x):
        feat = self.backbone(x)
        feat = self.proj(feat)
        feat = F.normalize(feat, p=2, dim=-1)
        return feat


class SymmetricInfoNCE(nn.Module):
    def __init__(self, temperature=CFG.TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, feat_a, feat_b):
        logits = feat_a @ feat_b.T / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels))


# ============================================================================
# 4. EVALUATION
# ============================================================================

def extract_features_tiles(model, dataloader, device):
    """Extract features for reference tiles. Returns feats, lats, lons, tids, time_s."""
    model.eval()
    all_feats, all_lats, all_lons, all_tids = [], [], [], []

    # Warmup GPU
    if device == "cuda":
        dummy = torch.randn(1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE, device=device)
        with torch.no_grad():
            _ = model(dummy)
        torch.cuda.synchronize()

    t_start = time.time()
    with torch.no_grad():
        for imgs, tids, lats, lons in tqdm(dataloader, desc="Gallery features", leave=False):
            imgs = imgs.to(device)
            feats = model(imgs)
            all_feats.append(feats.cpu().numpy())
            all_tids.extend(tids)
            all_lats.extend(lats.numpy())
            all_lons.extend(lons.numpy())
    if device == "cuda":
        torch.cuda.synchronize()
    t_total = time.time() - t_start
    return np.concatenate(all_feats), np.array(all_lats), np.array(all_lons), all_tids, t_total


def extract_features_queries(model, dataloader, device):
    """Extract features for UAV queries. Returns feats, lats, lons, time_s."""
    model.eval()
    all_feats, all_lats, all_lons = [], [], []
    t_start = time.time()
    with torch.no_grad():
        for imgs, lats, lons in tqdm(dataloader, desc="Query features", leave=False):
            imgs = imgs.to(device)
            feats = model(imgs)
            all_feats.append(feats.cpu().numpy())
            all_lats.extend(lats.numpy())
            all_lons.extend(lons.numpy())
    if device == "cuda":
        torch.cuda.synchronize()
    t_total = time.time() - t_start
    return np.concatenate(all_feats), np.array(all_lats), np.array(all_lons), t_total


def evaluate_avl(model, query_loader, gallery_loader, device,
                 thresholds_m=CFG.EVAL_THRESHOLDS_M):
    """Evaluate: R@K + PDM@K + localization error + inference timing."""
    q_feats, q_lats, q_lons, q_time = extract_features_queries(model, query_loader, device)
    g_feats, g_lats, g_lons, _, g_time = extract_features_tiles(model, gallery_loader, device)

    n_queries = len(q_lats)
    n_gallery = len(g_lats)

    # Cosine similarity + retrieval timing
    t_retrieval = time.time()
    sim = q_feats @ g_feats.T
    top1_indices = np.argmax(sim, axis=1)
    t_retrieval = time.time() - t_retrieval

    # --- Localization errors ---
    errors = []
    for i in range(n_queries):
        g_idx = top1_indices[i]
        dist = haversine(q_lats[i], q_lons[i], g_lats[g_idx], g_lons[g_idx])
        errors.append(dist)
    errors = np.array(errors)

    metrics = {
        "mean_error_m": float(np.mean(errors)),
        "median_error_m": float(np.median(errors)),
    }

    # --- R@K (tile-level retrieval recall) ---
    # A match is correct if the retrieved tile is within POS_RADIUS of the query GPS
    for k in [1, 5, 10]:
        top_k_idx = np.argsort(-sim, axis=1)[:, :k]
        correct = 0
        for i in range(n_queries):
            for j in range(k):
                g_idx = top_k_idx[i, j]
                d = haversine(q_lats[i], q_lons[i], g_lats[g_idx], g_lons[g_idx])
                if d <= CFG.POS_RADIUS:
                    correct += 1
                    break
        metrics[f"R@{k}"] = correct / n_queries

    # --- PDM@K at various thresholds ---
    for k in [1, 5, 10]:
        if k == 1:
            top_k_indices = top1_indices.reshape(-1, 1)
        else:
            top_k_indices = np.argsort(-sim, axis=1)[:, :k]

        for thresh in thresholds_m:
            correct = 0
            for i in range(n_queries):
                min_dist = float('inf')
                for j in range(k):
                    g_idx = top_k_indices[i, j]
                    d = haversine(q_lats[i], q_lons[i], g_lats[g_idx], g_lons[g_idx])
                    min_dist = min(min_dist, d)
                if min_dist <= thresh:
                    correct += 1
            metrics[f"PDM@{k}_@{thresh}m"] = correct / n_queries

    # --- Inference timing ---
    metrics["infer_query_total_s"] = round(q_time, 2)
    metrics["infer_gallery_total_s"] = round(g_time, 2)
    metrics["infer_per_query_ms"] = round(q_time / max(n_queries, 1) * 1000, 2)
    metrics["infer_per_gallery_ms"] = round(g_time / max(n_gallery, 1) * 1000, 2)
    metrics["retrieval_time_s"] = round(t_retrieval, 2)
    metrics["n_queries"] = n_queries
    metrics["n_gallery"] = n_gallery

    return metrics


# ============================================================================
# 5. TRAINING
# ============================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0
    count = 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CFG.EPOCHS}", leave=False)
    for uav_imgs, tile_imgs, _ in pbar:
        uav_imgs  = uav_imgs.to(device)
        tile_imgs = tile_imgs.to(device)
        optimizer.zero_grad()
        with autocast():
            feat_u = model(uav_imgs)
            feat_t = model(tile_imgs)
            loss = criterion(feat_u, feat_t)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * uav_imgs.size(0)
        count += uav_imgs.size(0)
        pbar.set_postfix(loss=f"{total_loss/count:.4f}")
    return total_loss / count


# ============================================================================
# 6. MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  EXP02 — UAV-AVL Baseline (Reference-Map Tile Matching)")
    print(f"  Backbone: {CFG.BACKBONE} | Embed: {CFG.EMBED_DIM}")
    print(f"  Device: {CFG.DEVICE}")
    print("=" * 60)

    root = CFG.DATASET_ROOT
    ref_dir = os.path.join(root, CFG.REF_MAP_DIR)
    uav_dir = os.path.join(root, CFG.UAV_DIR)

    assert os.path.exists(ref_dir), f"Missing: {ref_dir}"
    assert os.path.exists(uav_dir), f"Missing: {uav_dir}"

    # --- Build reference tile gallery ---
    print("\n[1/4] Tiling reference maps...")
    tif_files = sorted([
        os.path.join(ref_dir, f) for f in os.listdir(ref_dir)
        if f.lower().endswith('.tif')
    ])
    print(f"  Found {len(tif_files)} .tif reference maps")

    all_tiles = []
    for tif_path in tqdm(tif_files, desc="Tiling"):
        tiles = tile_reference_map(tif_path, CFG.TILE_SIZE, CFG.TILE_STRIDE)
        all_tiles.extend(tiles)
    print(f"  ✓ {len(all_tiles)} tiles generated ({CFG.TILE_SIZE}px, stride={CFG.TILE_STRIDE})")

    # --- Discover UAV images ---
    print("\n[2/4] Discovering UAV images...")
    uav_samples = discover_uav_data(uav_dir)
    print(f"  ✓ {len(uav_samples)} UAV images with GPS")

    # --- Split train/val (80/20 by image index) ---
    n_total = len(uav_samples)
    n_train = int(n_total * 0.8)
    random.shuffle(uav_samples)
    train_uav = uav_samples[:n_train]
    val_uav   = uav_samples[n_train:]
    print(f"  Train: {len(train_uav)} | Val: {len(val_uav)}")

    # --- Transforms ---
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    train_uav_tf = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        normalize,
    ])
    train_tile_tf = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(45),
        transforms.ToTensor(),
        normalize,
    ])
    test_tf = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    # --- Datasets ---
    train_dataset = UAVAVLTrainDataset(
        train_uav, all_tiles, train_uav_tf, train_tile_tf,
        pos_radius=CFG.POS_RADIUS
    )
    val_query_ds   = UAVAVLQueryDataset(val_uav, test_tf)
    val_gallery_ds = UAVAVLTileDataset(all_tiles, test_tf)

    train_loader = DataLoader(
        train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True,
        num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    val_q_loader = DataLoader(
        val_query_ds, batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
        num_workers=CFG.NUM_WORKERS, pin_memory=True,
    )
    val_g_loader = DataLoader(
        val_gallery_ds, batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
        num_workers=CFG.NUM_WORKERS, pin_memory=True,
    )

    # --- Model ---
    print("\n[3/4] Building model...")
    model = GeoModel().to(CFG.DEVICE)
    criterion = SymmetricInfoNCE()
    scaler = GradScaler()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.EPOCHS, eta_min=1e-6,
    )

    # --- Training ---
    print("\n[4/4] Training...")
    best_pdm = 0.0
    history = []

    for epoch in range(CFG.EPOCHS):
        t0 = time.time()

        if epoch == CFG.WARMUP_EP:
            model.unfreeze_last_n_blocks(CFG.UNFREEZE_BLOCKS)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=CFG.LR * 0.1, weight_decay=CFG.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=CFG.EPOCHS - epoch, eta_min=1e-6,
            )

        loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, CFG.DEVICE, epoch)
        scheduler.step()
        elapsed = time.time() - t0

        log = {"epoch": epoch + 1, "loss": round(loss, 5), "time_s": round(elapsed, 1)}

        # Evaluate every 5 epochs or last epoch
        if (epoch + 1) % 5 == 0 or epoch == CFG.EPOCHS - 1:
            metrics = evaluate_avl(model, val_q_loader, val_g_loader, CFG.DEVICE)
            log.update(metrics)

            pdm1_10 = metrics.get("PDM@1_@10m", 0.0)
            if pdm1_10 > best_pdm:
                best_pdm = pdm1_10
                torch.save(model.state_dict(),
                           os.path.join(CFG.OUTPUT_DIR, "best_uavavl_baseline.pth"))
                log["best"] = True

            print(f"Epoch {epoch+1:3d} | loss={loss:.4f} | "
                  f"R@1={metrics.get('R@1',0):.3f} R@5={metrics.get('R@5',0):.3f} | "
                  f"PDM@1_5m={metrics.get('PDM@1_@5m',0):.3f} "
                  f"PDM@1_10m={metrics.get('PDM@1_@10m',0):.3f} "
                  f"PDM@1_25m={metrics.get('PDM@1_@25m',0):.3f} | "
                  f"err={metrics['mean_error_m']:.1f}m | "
                  f"infer={metrics['infer_per_query_ms']:.1f}ms/q | {elapsed:.0f}s")
        else:
            print(f"Epoch {epoch+1:3d} | loss={loss:.4f} | {elapsed:.0f}s")

        history.append(log)

    # --- Final ---
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION (best model)")
    print("=" * 60)
    model.load_state_dict(
        torch.load(os.path.join(CFG.OUTPUT_DIR, "best_uavavl_baseline.pth"))
    )
    final_metrics = evaluate_avl(model, val_q_loader, val_g_loader, CFG.DEVICE)
    for k, v in sorted(final_metrics.items()):
        print(f"  {k}: {v:.4f}")

    results = {
        "experiment": "exp02_uavavl_baseline",
        "backbone": CFG.BACKBONE,
        "embed_dim": CFG.EMBED_DIM,
        "tile_size": CFG.TILE_SIZE,
        "tile_stride": CFG.TILE_STRIDE,
        "n_tiles": len(all_tiles),
        "n_uav_images": len(uav_samples),
        "epochs": CFG.EPOCHS,
        "best_PDM@1_10m": best_pdm,
        "final_metrics": final_metrics,
        "history": history,
    }

    out_path = os.path.join(CFG.OUTPUT_DIR, "results_exp02_uavavl_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
