# ============================================================================
# EXP01 — DenseUAV Baseline: DINOv2 + Sample4Geo-style Contrastive Learning
# ============================================================================
# Self-contained script for Kaggle H100.
# Dataset: chisboiz/denseuav
# Backbone: DINOv2 ViT-B/14
# Loss: Symmetric InfoNCE with GPS-based hard-negative mining
# Metrics: R@1, R@5, R@10, R@1%, AP, SDM@1, SDM@5, SDM@10, MA@5m/10m/25m
# ============================================================================

# === SETUP (Auto-install dependencies) ===
import subprocess, sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

for pkg in ["timm", "scipy"]:
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
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore")

# === CONFIG ===
class CFG:
    # Paths (Kaggle)
    DATASET_ROOT = "/kaggle/input/datasets/chisboiz/denseuav/DenseUAV"
    OUTPUT_DIR   = "/kaggle/working"

    # Model
    BACKBONE     = "dinov2_vitb14"
    EMBED_DIM    = 512
    IMG_SIZE     = 392       # must be divisible by 14 (DINOv2 patch_size)

    # Training
    EPOCHS       = 40
    BATCH_SIZE   = 64
    LR           = 1e-4
    WEIGHT_DECAY = 0.03
    WARMUP_EP    = 5          # freeze backbone for first N epochs
    UNFREEZE_BLOCKS = 4       # unfreeze last N transformer blocks
    TEMPERATURE  = 0.1
    NUM_WORKERS  = 4
    SEED         = 42

    # Hard-negative mining
    HARD_NEG_K   = 10         # number of geo-nearest negatives per sample
    MINING_START = 5          # start hard mining after N epochs

    # Eval
    EVAL_EVERY   = 5          # evaluate every N epochs

    # Device
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

seed_everything(CFG.SEED)

# ============================================================================
# 1. DATA
# ============================================================================

def _parse_coord(s):
    """Parse a coordinate string like 'N36.123' or 'E120.456' or plain '36.123'."""
    s = s.strip()
    sign = 1.0
    if s[0] in ('S', 's', 'W', 'w'):
        sign = -1.0
        s = s[1:]
    elif s[0] in ('N', 'n', 'E', 'e'):
        s = s[1:]
    return sign * float(s)


def parse_gps_file(path):
    """Parse DenseUAV GPS file → dict[folder_id] = (lat, lon).
    Handles formats: '0 N36.xxx E120.xxx' or '0 36.xxx 120.xxx'
    """
    gps = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                folder_id = parts[0].zfill(6)
                lat = _parse_coord(parts[1])
                lon = _parse_coord(parts[2])
                gps[folder_id] = (lat, lon)
    return gps


def get_image_paths(root_dir):
    """Return sorted list of (folder_id, [image_paths])."""
    folders = {}
    for folder in sorted(os.listdir(root_dir)):
        folder_path = os.path.join(root_dir, folder)
        if os.path.isdir(folder_path):
            imgs = sorted([
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))
            ])
            if imgs:
                folders[folder] = imgs
    return folders


class DenseUAVTrainDataset(Dataset):
    """Returns (drone_img, satellite_img, idx) — matched pairs by folder ID."""

    def __init__(self, drone_root, sat_root, transform_drone, transform_sat):
        self.drone_folders = get_image_paths(drone_root)
        self.sat_folders   = get_image_paths(sat_root)

        # Only keep IDs that exist in both
        common_ids = sorted(set(self.drone_folders.keys()) & set(self.sat_folders.keys()))
        self.ids = common_ids
        print(f"[Train] {len(self.ids)} matched locations")

        self.transform_drone = transform_drone
        self.transform_sat   = transform_sat

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fid = self.ids[idx]
        # Random drone image from this location
        drone_path = random.choice(self.drone_folders[fid])
        # Random satellite image from this location
        sat_path   = random.choice(self.sat_folders[fid])

        drone_img = Image.open(drone_path).convert("RGB")
        sat_img   = Image.open(sat_path).convert("RGB")

        drone_img = self.transform_drone(drone_img)
        sat_img   = self.transform_sat(sat_img)

        return drone_img, sat_img, idx


class DenseUAVTestDataset(Dataset):
    """Returns (img, folder_id, img_index) for feature extraction."""

    def __init__(self, root_dir, transform, view_type="drone"):
        self.root_dir = root_dir
        self.transform = transform
        self.view_type = view_type

        self.samples = []  # (img_path, folder_id)
        folders = get_image_paths(root_dir)
        for fid in sorted(folders.keys()):
            for img_path in folders[fid]:
                self.samples.append((img_path, fid))

        print(f"[Test-{view_type}] {len(self.samples)} images from "
              f"{len(folders)} locations")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, fid = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, fid


def build_transforms():
    """Standard transforms following DenseUAV / Sample4Geo papers."""
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    train_drone = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        transforms.ToTensor(),
        normalize,
    ])

    train_sat = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=90, fill=0),  # satellite rotation aug
        transforms.ToTensor(),
        normalize,
    ])

    test_tf = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    return train_drone, train_sat, test_tf


# ============================================================================
# 2. MODEL
# ============================================================================

class GeoModel(nn.Module):
    """DINOv2 backbone + projection head for cross-view matching."""

    def __init__(self, backbone_name=CFG.BACKBONE, embed_dim=CFG.EMBED_DIM):
        super().__init__()

        # Load DINOv2 from torch hub
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", backbone_name, pretrained=True
        )
        feat_dim = self.backbone.embed_dim  # 768 for ViT-B

        # Projection head
        self.proj = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, embed_dim),
        )

        # Initially freeze backbone
        self._freeze_backbone()

    def _freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n):
        """Unfreeze the last N transformer blocks + norm."""
        # Unfreeze norm
        for p in self.backbone.norm.parameters():
            p.requires_grad = True

        # Unfreeze last n blocks
        total_blocks = len(self.backbone.blocks)
        for i in range(total_blocks - n, total_blocks):
            for p in self.backbone.blocks[i].parameters():
                p.requires_grad = True

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"[Model] Unfroze last {n} blocks. "
              f"Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M")

    def forward(self, x):
        # DINOv2 cls token
        feat = self.backbone(x)  # (B, feat_dim)
        feat = self.proj(feat)   # (B, embed_dim)
        feat = F.normalize(feat, p=2, dim=-1)
        return feat


# ============================================================================
# 3. LOSS
# ============================================================================

class SymmetricInfoNCE(nn.Module):
    """Symmetric InfoNCE loss (Sample4Geo-style).
    Given drone features D and satellite features S of a batch:
      L = 0.5 * (InfoNCE(D→S) + InfoNCE(S→D))
    """

    def __init__(self, temperature=CFG.TEMPERATURE):
        super().__init__()
        self.temperature = temperature

    def forward(self, feat_d, feat_s):
        # Cosine similarity matrix
        logits = feat_d @ feat_s.T / self.temperature  # (B, B)
        labels = torch.arange(logits.size(0), device=logits.device)

        loss_d2s = F.cross_entropy(logits, labels)
        loss_s2d = F.cross_entropy(logits.T, labels)

        return 0.5 * (loss_d2s + loss_s2d)


# ============================================================================
# 4. EVALUATION
# ============================================================================

def extract_features(model, dataloader, device):
    """Extract features for all images in dataloader. Returns feats, fids, total_time_s."""
    model.eval()
    all_feats = []
    all_fids  = []

    # Warmup GPU
    if device == "cuda":
        dummy = torch.randn(1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE, device=device)
        with torch.no_grad():
            _ = model(dummy)
        torch.cuda.synchronize()

    t_start = time.time()
    with torch.no_grad():
        for imgs, fids in tqdm(dataloader, desc="Extracting features", leave=False):
            imgs = imgs.to(device)
            feats = model(imgs)
            all_feats.append(feats.cpu().numpy())
            all_fids.extend(fids)
    if device == "cuda":
        torch.cuda.synchronize()
    t_total = time.time() - t_start

    all_feats = np.concatenate(all_feats, axis=0)
    return all_feats, all_fids, t_total


def compute_recalls(query_feats, gallery_feats, query_fids, gallery_fids, k_list=[1, 5, 10]):
    """Compute Recall@K: a query is correct if any of top-K gallery items share fid."""
    # Group gallery images by folder id
    gallery_fid_to_indices = defaultdict(list)
    for i, fid in enumerate(gallery_fids):
        gallery_fid_to_indices[fid].append(i)

    # Cosine similarity (features are already L2-norm)
    sim_matrix = query_feats @ gallery_feats.T  # (Q, G)

    recalls = {}
    max_k = max(k_list)

    # For each query, get top-K gallery indices
    top_indices = np.argsort(-sim_matrix, axis=1)[:, :max_k]

    for k in k_list:
        correct = 0
        for q_idx in range(len(query_fids)):
            q_fid = query_fids[q_idx]
            top_k_gids = [gallery_fids[g_idx] for g_idx in top_indices[q_idx, :k]]
            if q_fid in top_k_gids:
                correct += 1
        recalls[f"R@{k}"] = correct / len(query_fids)

    # R@1%
    one_pct = max(1, len(gallery_fids) // 100)
    correct = 0
    top_1pct = np.argsort(-sim_matrix, axis=1)[:, :one_pct]
    for q_idx in range(len(query_fids)):
        q_fid = query_fids[q_idx]
        top_gids = [gallery_fids[g_idx] for g_idx in top_1pct[q_idx]]
        if q_fid in top_gids:
            correct += 1
    recalls["R@1%"] = correct / len(query_fids)

    # AP (Average Precision across all queries)
    aps = []
    for q_idx in range(len(query_fids)):
        q_fid = query_fids[q_idx]
        sorted_gids = [gallery_fids[g_idx] for g_idx in np.argsort(-sim_matrix[q_idx])]
        n_relevant = len(gallery_fid_to_indices[q_fid])
        if n_relevant == 0:
            continue
        hits = 0
        precision_sum = 0.0
        for rank, gid in enumerate(sorted_gids, 1):
            if gid == q_fid:
                hits += 1
                precision_sum += hits / rank
        aps.append(precision_sum / n_relevant)
    recalls["AP"] = np.mean(aps) if aps else 0.0

    return recalls


def compute_ma(query_feats, gallery_feats, query_fids, gallery_fids,
               gps_dict, thresholds_m=[5, 10, 25]):
    """Compute MA@Xm (Meter-level Accuracy).
    For each query, find top-1 gallery match, compute GPS distance,
    report % of queries within X meters.
    """
    sim_matrix = query_feats @ gallery_feats.T

    # Per-location gallery features (average)
    unique_g_fids = sorted(set(gallery_fids))
    loc_sums = defaultdict(lambda: np.zeros(gallery_feats.shape[1]))
    loc_counts = defaultdict(int)
    for i, fid in enumerate(gallery_fids):
        loc_sums[fid] += gallery_feats[i]
        loc_counts[fid] += 1
    loc_feats = {}
    for fid in unique_g_fids:
        f = loc_sums[fid] / loc_counts[fid]
        loc_feats[fid] = f / (np.linalg.norm(f) + 1e-8)
    loc_gallery = np.stack([loc_feats[fid] for fid in unique_g_fids])

    # Per-location query features (average)
    unique_q_fids = sorted(set(query_fids))
    q_sums = defaultdict(lambda: np.zeros(query_feats.shape[1]))
    q_counts = defaultdict(int)
    for i, fid in enumerate(query_fids):
        q_sums[fid] += query_feats[i]
        q_counts[fid] += 1
    q_loc_feats = {}
    for fid in unique_q_fids:
        f = q_sums[fid] / q_counts[fid]
        q_loc_feats[fid] = f / (np.linalg.norm(f) + 1e-8)
    q_gallery = np.stack([q_loc_feats[fid] for fid in unique_q_fids])

    loc_sim = q_gallery @ loc_gallery.T
    top1_loc = np.argmax(loc_sim, axis=1)

    errors = []
    for q_idx, q_fid in enumerate(unique_q_fids):
        if q_fid not in gps_dict:
            continue
        g_fid = unique_g_fids[top1_loc[q_idx]]
        if g_fid not in gps_dict:
            continue
        q_lat, q_lon = gps_dict[q_fid]
        g_lat, g_lon = gps_dict[g_fid]
        errors.append(haversine(q_lat, q_lon, g_lat, g_lon))

    errors = np.array(errors)
    ma = {}
    for t in thresholds_m:
        ma[f"MA@{t}m"] = float(np.mean(errors <= t)) if len(errors) > 0 else 0.0
    if len(errors) > 0:
        ma["mean_loc_error_m"] = float(np.mean(errors))
        ma["median_loc_error_m"] = float(np.median(errors))
    return ma


def compute_sdm(query_feats, gallery_feats, query_fids, gallery_fids, gps_dict, k_list=[1, 5, 10]):
    """Compute SDM@K (Spatial Distance Metric) using GPS ground truth.
    SDM@K measures how close the top-K retrieved locations are to the query's true GPS position.
    """
    sim_matrix = query_feats @ gallery_feats.T

    # Unique gallery folder ids (for per-location aggregation)
    unique_g_fids = sorted(set(gallery_fids))
    fid_to_first_idx = {}
    for i, fid in enumerate(gallery_fids):
        if fid not in fid_to_first_idx:
            fid_to_first_idx[fid] = i

    # Per-location gallery features (average all images of same location)
    loc_feats = {}
    loc_counts = defaultdict(int)
    loc_sums = defaultdict(lambda: np.zeros(gallery_feats.shape[1]))
    for i, fid in enumerate(gallery_fids):
        loc_sums[fid] += gallery_feats[i]
        loc_counts[fid] += 1
    for fid in unique_g_fids:
        f = loc_sums[fid] / loc_counts[fid]
        loc_feats[fid] = f / (np.linalg.norm(f) + 1e-8)

    # Build location-level gallery matrix
    loc_gallery = np.stack([loc_feats[fid] for fid in unique_g_fids])

    # Query: average features per query location
    unique_q_fids = sorted(set(query_fids))
    q_loc_feats = {}
    q_counts = defaultdict(int)
    q_sums   = defaultdict(lambda: np.zeros(query_feats.shape[1]))
    for i, fid in enumerate(query_fids):
        q_sums[fid] += query_feats[i]
        q_counts[fid] += 1
    for fid in unique_q_fids:
        f = q_sums[fid] / q_counts[fid]
        q_loc_feats[fid] = f / (np.linalg.norm(f) + 1e-8)

    q_gallery = np.stack([q_loc_feats[fid] for fid in unique_q_fids])

    # Location-level similarity
    loc_sim = q_gallery @ loc_gallery.T

    sdm = {}
    for k in k_list:
        distances = []
        for q_idx, q_fid in enumerate(unique_q_fids):
            if q_fid not in gps_dict:
                continue
            q_lat, q_lon = gps_dict[q_fid]

            top_k_loc_idx = np.argsort(-loc_sim[q_idx])[:k]
            min_dist = float('inf')
            for loc_idx in top_k_loc_idx:
                g_fid = unique_g_fids[loc_idx]
                if g_fid in gps_dict:
                    g_lat, g_lon = gps_dict[g_fid]
                    # Haversine distance in meters
                    dist = haversine(q_lat, q_lon, g_lat, g_lon)
                    min_dist = min(min_dist, dist)
            if min_dist < float('inf'):
                distances.append(min_dist)

        if distances:
            mean_dist = np.mean(distances)
            # SDM@K = exp(-mean_dist / sigma), sigma chosen so that smaller dist = higher score
            sigma = 100.0  # 100m baseline
            sdm[f"SDM@{k}"] = float(np.exp(-mean_dist / sigma))
            sdm[f"MeanDist@{k}_m"] = float(mean_dist)
        else:
            sdm[f"SDM@{k}"] = 0.0
            sdm[f"MeanDist@{k}_m"] = float('inf')

    return sdm


def haversine(lat1, lon1, lat2, lon2):
    """Haversine distance in meters."""
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def evaluate(model, test_query_loader, test_gallery_loader, gps_dict, device):
    """Full evaluation: R@K + SDM@K + MA@Xm + inference timing."""
    q_feats, q_fids, q_time = extract_features(model, test_query_loader, device)
    g_feats, g_fids, g_time = extract_features(model, test_gallery_loader, device)

    n_queries = len(q_fids)
    n_gallery = len(g_fids)

    # Retrieval timing
    t_retrieval = time.time()
    recalls = compute_recalls(q_feats, g_feats, q_fids, g_fids, k_list=[1, 5, 10])
    t_retrieval = time.time() - t_retrieval

    sdm = compute_sdm(q_feats, g_feats, q_fids, g_fids, gps_dict, k_list=[1, 5, 10])
    ma  = compute_ma(q_feats, g_feats, q_fids, g_fids, gps_dict, thresholds_m=[5, 10, 25])

    # Inference timing summary
    timing = {
        "infer_query_total_s": round(q_time, 2),
        "infer_gallery_total_s": round(g_time, 2),
        "infer_per_query_ms": round(q_time / max(n_queries, 1) * 1000, 2),
        "infer_per_gallery_ms": round(g_time / max(n_gallery, 1) * 1000, 2),
        "retrieval_time_s": round(t_retrieval, 2),
        "n_queries": n_queries,
        "n_gallery": n_gallery,
    }

    metrics = {**recalls, **sdm, **ma, **timing}
    return metrics


# ============================================================================
# 5. TRAINING
# ============================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss = 0
    count = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CFG.EPOCHS}", leave=False)
    for drone_imgs, sat_imgs, indices in pbar:
        drone_imgs = drone_imgs.to(device)
        sat_imgs   = sat_imgs.to(device)

        optimizer.zero_grad()
        with autocast():
            feat_d = model(drone_imgs)
            feat_s = model(sat_imgs)
            loss = criterion(feat_d, feat_s)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * drone_imgs.size(0)
        count += drone_imgs.size(0)
        pbar.set_postfix(loss=f"{total_loss/count:.4f}")

    return total_loss / count


# ============================================================================
# 6. MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  EXP01 — DenseUAV Baseline")
    print(f"  Backbone: {CFG.BACKBONE} | Embed: {CFG.EMBED_DIM}")
    print(f"  Device: {CFG.DEVICE}")
    print("=" * 60)

    # --- Paths ---
    root = CFG.DATASET_ROOT
    train_drone_dir  = os.path.join(root, "train", "drone")
    train_sat_dir    = os.path.join(root, "train", "satellite")
    test_query_dir   = os.path.join(root, "test", "query_drone")
    test_gallery_dir = os.path.join(root, "test", "gallery_satellite")
    gps_test_file    = os.path.join(root, "Dense_GPS_test.txt")
    gps_all_file     = os.path.join(root, "Dense_GPS_ALL.txt")

    # Verify paths
    for p, name in [(train_drone_dir, "train/drone"),
                     (train_sat_dir, "train/satellite"),
                     (test_query_dir, "test/query_drone"),
                     (test_gallery_dir, "test/gallery_satellite")]:
        assert os.path.exists(p), f"Missing: {p}"
        print(f"  ✓ {name}: {len(os.listdir(p))} folders")

    # --- GPS ---
    gps_dict = {}
    for gps_file in [gps_test_file, gps_all_file]:
        if os.path.exists(gps_file):
            gps_dict.update(parse_gps_file(gps_file))
            print(f"  ✓ GPS loaded from {os.path.basename(gps_file)}: {len(gps_dict)} entries")

    # --- Transforms ---
    train_drone_tf, train_sat_tf, test_tf = build_transforms()

    # --- Datasets ---
    train_dataset = DenseUAVTrainDataset(
        train_drone_dir, train_sat_dir, train_drone_tf, train_sat_tf
    )
    test_query_dataset   = DenseUAVTestDataset(test_query_dir, test_tf, view_type="query_drone")
    test_gallery_dataset = DenseUAVTestDataset(test_gallery_dir, test_tf, view_type="gallery_sat")

    train_loader = DataLoader(
        train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True,
        num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True,
    )
    test_query_loader = DataLoader(
        test_query_dataset, batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
        num_workers=CFG.NUM_WORKERS, pin_memory=True,
    )
    test_gallery_loader = DataLoader(
        test_gallery_dataset, batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
        num_workers=CFG.NUM_WORKERS, pin_memory=True,
    )

    # --- Model ---
    model = GeoModel().to(CFG.DEVICE)
    criterion = SymmetricInfoNCE()
    scaler = GradScaler()

    # Optimizer (only proj head initially since backbone frozen)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=CFG.EPOCHS, eta_min=1e-6
    )

    # --- Training Loop ---
    best_r1 = 0.0
    history = []

    for epoch in range(CFG.EPOCHS):
        t0 = time.time()

        # Unfreeze backbone after warmup
        if epoch == CFG.WARMUP_EP:
            model.unfreeze_last_n_blocks(CFG.UNFREEZE_BLOCKS)
            optimizer = torch.optim.AdamW(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=CFG.LR * 0.1,  # lower LR for backbone
                weight_decay=CFG.WEIGHT_DECAY,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=CFG.EPOCHS - epoch, eta_min=1e-6
            )

        loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, CFG.DEVICE, epoch)
        scheduler.step()
        elapsed = time.time() - t0

        log = {
            "epoch": epoch + 1,
            "loss": round(loss, 5),
            "lr": round(optimizer.param_groups[0]["lr"], 7),
            "time_s": round(elapsed, 1),
        }

        # Evaluate periodically
        if (epoch + 1) % CFG.EVAL_EVERY == 0 or epoch == CFG.EPOCHS - 1:
            metrics = evaluate(model, test_query_loader, test_gallery_loader, gps_dict, CFG.DEVICE)
            log.update(metrics)

            r1 = metrics["R@1"]
            if r1 > best_r1:
                best_r1 = r1
                torch.save(model.state_dict(),
                           os.path.join(CFG.OUTPUT_DIR, "best_denseuav_baseline.pth"))
                log["best"] = True

            print(f"Epoch {epoch+1:3d} | loss={loss:.4f} | "
                  f"R@1={metrics['R@1']:.4f} R@5={metrics['R@5']:.4f} R@10={metrics['R@10']:.4f} | "
                  f"AP={metrics['AP']:.4f} | SDM@1={metrics['SDM@1']:.4f} | "
                  f"MA@10m={metrics.get('MA@10m',0):.4f} | "
                  f"infer={metrics['infer_per_query_ms']:.1f}ms/q | {elapsed:.0f}s")
        else:
            print(f"Epoch {epoch+1:3d} | loss={loss:.4f} | lr={log['lr']:.2e} | {elapsed:.0f}s")

        history.append(log)

    # --- Final Evaluation ---
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION (best model)")
    print("=" * 60)
    model.load_state_dict(
        torch.load(os.path.join(CFG.OUTPUT_DIR, "best_denseuav_baseline.pth"))
    )
    final_metrics = evaluate(model, test_query_loader, test_gallery_loader, gps_dict, CFG.DEVICE)

    for k, v in sorted(final_metrics.items()):
        print(f"  {k}: {v:.4f}")

    # --- Save Results ---
    results = {
        "experiment": "exp01_denseuav_baseline",
        "backbone": CFG.BACKBONE,
        "embed_dim": CFG.EMBED_DIM,
        "img_size": CFG.IMG_SIZE,
        "epochs": CFG.EPOCHS,
        "batch_size": CFG.BATCH_SIZE,
        "best_R@1": best_r1,
        "final_metrics": final_metrics,
        "history": history,
    }

    out_path = os.path.join(CFG.OUTPUT_DIR, "results_exp01_denseuav_baseline.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
