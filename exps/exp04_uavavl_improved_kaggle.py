# ============================================================================
# EXP04 — UAV-AVL Improved: DINOv2 + GeM + Denser Tiling + More Epochs
# ============================================================================
# Improvements over EXP02:
#   1. GeM pooling on patch tokens
#   2. TILE_STRIDE=64 for denser gallery coverage
#   3. 50 epochs (was 20)
#   4. IMG_SIZE=518 (DINOv2 native)
#   5. Unfreeze 6 blocks (was 4)
#   6. Better augmentation
# Dataset: hunhtrungkit/uav-avl
# Metrics: R@1/5/10, PDM@K @5/10/25m, Mean/Median error, Inference timing
# ============================================================================

import subprocess, sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

for pkg in ["timm", "rasterio", "scipy", "pyproj"]:
    try:
        __import__(pkg.replace("-", "_").split("==")[0])
    except ImportError:
        install(pkg)

import os, json, math, time, random, warnings
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
    print("[WARN] rasterio not available")

warnings.filterwarnings("ignore")

# === CONFIG ===
class CFG:
    DATASET_ROOT = "/kaggle/input/datasets/hunhtrungkit/uav-avl/Data"
    OUTPUT_DIR   = "/kaggle/working"

    TILE_SIZE    = 256
    TILE_STRIDE  = 64       # denser tiling (was 128)
    REF_MAP_DIR  = "Reference_map/QZ_Town"
    UAV_DIR      = "UAV_image/QZ_Town"

    BACKBONE     = "dinov2_vitb14"
    EMBED_DIM    = 512
    IMG_SIZE     = 518       # DINOv2 native (518/14=37)

    EPOCHS       = 50        # more epochs (was 20)
    BATCH_SIZE   = 24        # smaller for 518px
    LR           = 1e-4
    WEIGHT_DECAY = 0.03
    WARMUP_EP    = 5
    UNFREEZE_BLOCKS = 6
    TEMPERATURE  = 0.1
    LABEL_SMOOTH = 0.1
    NUM_WORKERS  = 4
    SEED         = 42

    POS_RADIUS   = 25.0
    NEG_RADIUS   = 100.0
    EVAL_THRESHOLDS_M = [5, 10, 25]
    EVAL_EVERY   = 5

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(CFG.SEED)

# ============================================================================
# 1. REFERENCE MAP TILING (with CRS reprojection)
# ============================================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def load_tif_as_pil(path):
    if HAS_RASTERIO:
        with rasterio.open(path) as src:
            bands = min(3, src.count)
            data = src.read(list(range(1, bands + 1)))
            if bands == 1:
                data = np.repeat(data, 3, axis=0)
            data = np.transpose(data, (1, 2, 0))
            if data.dtype != np.uint8:
                dmin, dmax = data.min(), data.max()
                if dmax > dmin:
                    data = ((data - dmin) / (dmax - dmin) * 255).astype(np.uint8)
                else:
                    data = np.zeros_like(data, dtype=np.uint8)
            return Image.fromarray(data)
    return Image.open(path).convert("RGB")


def get_tif_geotransform(path):
    if HAS_RASTERIO:
        with rasterio.open(path) as src:
            return src.transform, src.width, src.height
    return None, None, None


def tile_reference_map(tif_path, tile_size, stride):
    """Tile reference map with CRS→WGS84 reprojection."""
    transform_affine, width, height = get_tif_geotransform(tif_path)

    if transform_affine is None:
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

    # CRS reprojection
    try:
        from pyproj import Transformer
        with rasterio.open(tif_path) as src:
            src_crs = src.crs
        if src_crs and not src_crs.is_geographic:
            transformer = Transformer.from_crs(src_crs, "EPSG:4326", always_xy=True)
            print(f"    [CRS] {src_crs} → WGS84")
        else:
            transformer = None
    except Exception:
        transformer = None

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
                cx, cy = x + tile_size // 2, y + tile_size // 2
                native_x, native_y = src.transform * (cx, cy)

                if transformer is not None:
                    lon, lat = transformer.transform(native_x, native_y)
                else:
                    lon, lat = native_x, native_y

                tiles.append((tile_img, lat, lon, f"tile_{tid:06d}"))
                tid += 1

    if tiles:
        lats = [t[1] for t in tiles]
        lons = [t[2] for t in tiles]
        print(f"    Tile GPS: lat=[{min(lats):.6f}, {max(lats):.6f}] "
              f"lon=[{min(lons):.6f}, {max(lons):.6f}]")
    return tiles


# ============================================================================
# 2. DATA
# ============================================================================

def parse_smart_pos(filepath):
    positions = []
    with open(filepath, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 4:
                positions.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return positions


def discover_uav_data(uav_root):
    all_samples = []
    for flight in sorted(os.listdir(uav_root)):
        flight_dir = os.path.join(uav_root, flight)
        if not os.path.isdir(flight_dir):
            continue
        gps_file = None
        for name in ["smart_pos.txt", "smart_pos_photoscan.txt"]:
            p = os.path.join(flight_dir, name)
            if os.path.exists(p):
                gps_file = p
                break
        gps_dict = {}
        if gps_file:
            gps_dict = {fname: (lat, lon) for fname, lat, lon, _ in parse_smart_pos(gps_file)}
        for f in sorted(os.listdir(flight_dir)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                if f in gps_dict:
                    img_path = os.path.join(flight_dir, f)
                    lat, lon = gps_dict[f]
                    all_samples.append((img_path, lat, lon))
    return all_samples


class UAVAVLTileDataset(Dataset):
    def __init__(self, tiles, transform):
        self.tiles = tiles
        self.transform = transform

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        img, lat, lon, tid = self.tiles[idx]
        return self.transform(img), tid, torch.tensor(lat, dtype=torch.float32), torch.tensor(lon, dtype=torch.float32)


class UAVAVLQueryDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, lat, lon = self.samples[idx]
        img = self.transform(Image.open(img_path).convert("RGB"))
        return img, torch.tensor(lat, dtype=torch.float32), torch.tensor(lon, dtype=torch.float32)


class UAVAVLTrainDataset(Dataset):
    def __init__(self, uav_samples, tiles, transform_uav, transform_tile, pos_radius=CFG.POS_RADIUS):
        self.uav_samples = uav_samples
        self.tiles = tiles
        self.transform_uav  = transform_uav
        self.transform_tile = transform_tile
        self.tile_gps = np.array([(t[1], t[2]) for t in tiles])

        self.pairs = []
        for uav_idx, (_, ulat, ulon) in enumerate(uav_samples):
            dists = np.array([haversine(ulat, ulon, tlat, tlon) for tlat, tlon in self.tile_gps])
            pos_indices = np.where(dists <= pos_radius)[0]
            if len(pos_indices) > 0:
                self.pairs.append((uav_idx, pos_indices))

        # Fallback: if no matches, use nearest tile
        if len(self.pairs) == 0:
            print(f"[Train][WARN] No matches within {pos_radius}m. Falling back to nearest-tile.")
            for uav_idx, (_, ulat, ulon) in enumerate(uav_samples):
                dists = np.array([haversine(ulat, ulon, tlat, tlon) for tlat, tlon in self.tile_gps])
                nearest = np.argsort(dists)[:5]
                self.pairs.append((uav_idx, nearest))

        print(f"[Train] {len(self.pairs)} UAV images with training tiles (radius={pos_radius}m)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        uav_idx, pos_tile_indices = self.pairs[idx]
        tile_idx = random.choice(pos_tile_indices)
        img_path, _, _ = self.uav_samples[uav_idx]
        uav_img = self.transform_uav(Image.open(img_path).convert("RGB"))
        tile_img, _, _, _ = self.tiles[tile_idx]
        tile_img = self.transform_tile(tile_img)
        return uav_img, tile_img, idx


# ============================================================================
# 3. MODEL — DINOv2 + GeM Pooling
# ============================================================================

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        x = x.clamp(min=self.eps).pow(self.p)
        x = x.mean(dim=1)
        return x.pow(1.0 / self.p)


class GeoModel(nn.Module):
    def __init__(self, backbone_name=CFG.BACKBONE, embed_dim=CFG.EMBED_DIM):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name, pretrained=True)
        feat_dim = self.backbone.embed_dim
        self.gem = GeM(p=3.0)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim),
            nn.GELU(),
            nn.Dropout(0.1),
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
        print(f"[Model] Unfroze {n} blocks. Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M")

    def forward(self, x):
        x = self.backbone.prepare_tokens_with_masks(x)
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)
        cls_token = x[:, 0]
        patch_tokens = x[:, 1:]
        gem_feat = self.gem(patch_tokens)
        combined = torch.cat([cls_token, gem_feat], dim=1)
        feat = self.proj(combined)
        return F.normalize(feat, p=2, dim=-1)


class SymmetricInfoNCE(nn.Module):
    def __init__(self, temperature=CFG.TEMPERATURE, label_smoothing=CFG.LABEL_SMOOTH):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing

    def forward(self, feat_a, feat_b):
        logits = feat_a @ feat_b.T / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
                     + F.cross_entropy(logits.T, labels, label_smoothing=self.label_smoothing))


# ============================================================================
# 4. EVALUATION
# ============================================================================

def extract_features_tiles(model, dataloader, device):
    model.eval()
    all_feats, all_lats, all_lons, all_tids = [], [], [], []
    if device == "cuda":
        dummy = torch.randn(1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE, device=device)
        with torch.no_grad():
            _ = model(dummy)
        torch.cuda.synchronize()
    t_start = time.time()
    with torch.no_grad():
        for imgs, tids, lats, lons in tqdm(dataloader, desc="Gallery feats", leave=False):
            feats = model(imgs.to(device))
            all_feats.append(feats.cpu().numpy())
            all_tids.extend(tids)
            all_lats.extend(lats.numpy())
            all_lons.extend(lons.numpy())
    if device == "cuda":
        torch.cuda.synchronize()
    return np.concatenate(all_feats), np.array(all_lats), np.array(all_lons), all_tids, time.time() - t_start


def extract_features_queries(model, dataloader, device):
    model.eval()
    all_feats, all_lats, all_lons = [], [], []
    t_start = time.time()
    with torch.no_grad():
        for imgs, lats, lons in tqdm(dataloader, desc="Query feats", leave=False):
            feats = model(imgs.to(device))
            all_feats.append(feats.cpu().numpy())
            all_lats.extend(lats.numpy())
            all_lons.extend(lons.numpy())
    if device == "cuda":
        torch.cuda.synchronize()
    return np.concatenate(all_feats), np.array(all_lats), np.array(all_lons), time.time() - t_start


def evaluate_avl(model, query_loader, gallery_loader, device,
                 thresholds_m=CFG.EVAL_THRESHOLDS_M):
    q_feats, q_lats, q_lons, q_time = extract_features_queries(model, query_loader, device)
    g_feats, g_lats, g_lons, _, g_time = extract_features_tiles(model, gallery_loader, device)

    n_q, n_g = len(q_lats), len(g_lats)
    t_ret = time.time()
    sim = q_feats @ g_feats.T
    top1_indices = np.argmax(sim, axis=1)
    t_ret = time.time() - t_ret

    # Localization errors
    errors = np.array([haversine(q_lats[i], q_lons[i], g_lats[top1_indices[i]], g_lons[top1_indices[i]])
                       for i in range(n_q)])

    metrics = {"mean_error_m": float(np.mean(errors)), "median_error_m": float(np.median(errors))}

    # R@K (tile within POS_RADIUS)
    for k in [1, 5, 10]:
        top_k = np.argsort(-sim, axis=1)[:, :k]
        correct = 0
        for i in range(n_q):
            for j in range(k):
                if haversine(q_lats[i], q_lons[i], g_lats[top_k[i, j]], g_lons[top_k[i, j]]) <= CFG.POS_RADIUS:
                    correct += 1
                    break
        metrics[f"R@{k}"] = correct / n_q

    # PDM@K
    for k in [1, 5, 10]:
        top_k = np.argsort(-sim, axis=1)[:, :k] if k > 1 else top1_indices.reshape(-1, 1)
        for thresh in thresholds_m:
            correct = sum(1 for i in range(n_q)
                         if min(haversine(q_lats[i], q_lons[i], g_lats[top_k[i, j]], g_lons[top_k[i, j]])
                                for j in range(k)) <= thresh)
            metrics[f"PDM@{k}_@{thresh}m"] = correct / n_q

    # Timing
    metrics.update({
        "infer_query_total_s": round(q_time, 2), "infer_gallery_total_s": round(g_time, 2),
        "infer_per_query_ms": round(q_time / max(n_q, 1) * 1000, 2),
        "infer_per_gallery_ms": round(g_time / max(n_g, 1) * 1000, 2),
        "retrieval_time_s": round(t_ret, 2), "n_queries": n_q, "n_gallery": n_g,
    })
    return metrics


# ============================================================================
# 5. TRAINING
# ============================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss, count = 0, 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CFG.EPOCHS}", leave=False)
    for uav_imgs, tile_imgs, _ in pbar:
        uav_imgs, tile_imgs = uav_imgs.to(device), tile_imgs.to(device)
        optimizer.zero_grad()
        with autocast():
            loss = criterion(model(uav_imgs), model(tile_imgs))
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
    print("  EXP04 — UAV-AVL Improved (GeM + 518px + Dense Tiling)")
    print(f"  Backbone: {CFG.BACKBONE} | IMG: {CFG.IMG_SIZE} | Stride: {CFG.TILE_STRIDE}")
    print(f"  Device: {CFG.DEVICE}")
    print("=" * 60)

    root = CFG.DATASET_ROOT
    ref_dir = os.path.join(root, CFG.REF_MAP_DIR)
    uav_dir = os.path.join(root, CFG.UAV_DIR)
    assert os.path.exists(ref_dir), f"Missing: {ref_dir}"
    assert os.path.exists(uav_dir), f"Missing: {uav_dir}"

    # Tile reference maps
    print("\n[1/4] Tiling reference maps...")
    tif_files = sorted([os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.lower().endswith('.tif')])
    print(f"  Found {len(tif_files)} .tif files")
    all_tiles = []
    for tif_path in tqdm(tif_files, desc="Tiling"):
        all_tiles.extend(tile_reference_map(tif_path, CFG.TILE_SIZE, CFG.TILE_STRIDE))
    print(f"  ✓ {len(all_tiles)} tiles (stride={CFG.TILE_STRIDE})")

    # Discover UAV images
    print("\n[2/4] Discovering UAV images...")
    uav_samples = discover_uav_data(uav_dir)
    print(f"  ✓ {len(uav_samples)} UAV images with GPS")
    if uav_samples:
        ulats, ulons = [s[1] for s in uav_samples], [s[2] for s in uav_samples]
        print(f"  UAV GPS: lat=[{min(ulats):.6f}, {max(ulats):.6f}] lon=[{min(ulons):.6f}, {max(ulons):.6f}]")

    # Split
    n_train = int(len(uav_samples) * 0.8)
    random.shuffle(uav_samples)
    train_uav, val_uav = uav_samples[:n_train], uav_samples[n_train:]
    print(f"  Train: {len(train_uav)} | Val: {len(val_uav)}")

    # Transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_uav_tf = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
        transforms.ToTensor(), normalize,
    ])
    train_tile_tf = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(90),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(), normalize,
    ])
    test_tf = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=Image.BICUBIC),
        transforms.ToTensor(), normalize,
    ])

    # Datasets
    train_ds = UAVAVLTrainDataset(train_uav, all_tiles, train_uav_tf, train_tile_tf)
    val_q_ds = UAVAVLQueryDataset(val_uav, test_tf)
    val_g_ds = UAVAVLTileDataset(all_tiles, test_tf)

    train_loader = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_q_loader = DataLoader(val_q_ds, batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)
    val_g_loader = DataLoader(val_g_ds, batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True)

    # Model
    print("\n[3/4] Building model...")
    model = GeoModel().to(CFG.DEVICE)
    criterion = SymmetricInfoNCE()
    scaler = GradScaler()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS, eta_min=1e-6)

    # Training
    print("\n[4/4] Training...")
    best_pdm = 0.0
    history = []

    for epoch in range(CFG.EPOCHS):
        t0 = time.time()
        if epoch == CFG.WARMUP_EP:
            model.unfreeze_last_n_blocks(CFG.UNFREEZE_BLOCKS)
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                          lr=CFG.LR * 0.1, weight_decay=CFG.WEIGHT_DECAY)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS - epoch, eta_min=1e-6)

        loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, CFG.DEVICE, epoch)
        scheduler.step()
        elapsed = time.time() - t0
        log = {"epoch": epoch + 1, "loss": round(loss, 5), "time_s": round(elapsed, 1)}

        if (epoch + 1) % CFG.EVAL_EVERY == 0 or epoch == CFG.EPOCHS - 1:
            metrics = evaluate_avl(model, val_q_loader, val_g_loader, CFG.DEVICE)
            log.update(metrics)
            pdm = metrics.get("PDM@1_@10m", 0)
            if pdm > best_pdm:
                best_pdm = pdm
                torch.save(model.state_dict(), os.path.join(CFG.OUTPUT_DIR, "best_exp04.pth"))
                log["best"] = True
            print(f"Epoch {epoch+1:3d} | loss={loss:.4f} | "
                  f"R@1={metrics.get('R@1',0):.3f} | "
                  f"PDM@1_10m={pdm:.3f} PDM@10_25m={metrics.get('PDM@10_@25m',0):.3f} | "
                  f"err={metrics['mean_error_m']:.1f}m | {elapsed:.0f}s")
        else:
            print(f"Epoch {epoch+1:3d} | loss={loss:.4f} | {elapsed:.0f}s")
        history.append(log)

    # Final
    print("\n" + "=" * 60 + "\n  FINAL EVALUATION\n" + "=" * 60)
    model.load_state_dict(torch.load(os.path.join(CFG.OUTPUT_DIR, "best_exp04.pth")))
    final = evaluate_avl(model, val_q_loader, val_g_loader, CFG.DEVICE)
    for k, v in sorted(final.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    results = {
        "experiment": "exp04_uavavl_improved",
        "improvements": "GeM pooling, IMG_SIZE=518, stride=64, label_smoothing, 6 blocks, 50 epochs",
        "backbone": CFG.BACKBONE, "embed_dim": CFG.EMBED_DIM, "img_size": CFG.IMG_SIZE,
        "tile_size": CFG.TILE_SIZE, "tile_stride": CFG.TILE_STRIDE, "n_tiles": len(all_tiles),
        "epochs": CFG.EPOCHS, "best_PDM@1_10m": best_pdm,
        "final_metrics": final, "history": history,
    }
    out_path = os.path.join(CFG.OUTPUT_DIR, "results_exp04_uavavl_improved.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
