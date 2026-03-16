# ============================================================================
# EXP03 — DenseUAV Improved: DINOv2 + GeM Pooling + Native Resolution
# ============================================================================
# Improvements over EXP01:
#   1. Fix GPS key normalization (SDM/MA metrics now work)
#   2. GeM pooling on patch tokens (instead of CLS-only)
#   3. IMG_SIZE=518 (DINOv2 native resolution, 518/14=37)
#   4. 60 epochs, unfreeze 6 blocks
#   5. Label smoothing in InfoNCE
# Dataset: chisboiz/denseuav
# Metrics: R@1/5/10, R@1%, AP, SDM@1/5/10, MA@5m/10m/25m
# ============================================================================

import subprocess, sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

for pkg in ["timm", "scipy"]:
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

warnings.filterwarnings("ignore")

# === CONFIG ===
class CFG:
    DATASET_ROOT = "/kaggle/input/datasets/chisboiz/denseuav/DenseUAV"
    OUTPUT_DIR   = "/kaggle/working"

    BACKBONE     = "dinov2_vitb14"
    EMBED_DIM    = 512
    IMG_SIZE     = 518       # DINOv2 native resolution (518/14=37)

    EPOCHS       = 60
    BATCH_SIZE   = 32        # smaller BS for larger images
    LR           = 1e-4
    WEIGHT_DECAY = 0.03
    WARMUP_EP    = 5
    UNFREEZE_BLOCKS = 6      # unfreeze more blocks
    TEMPERATURE  = 0.1
    LABEL_SMOOTH = 0.1       # label smoothing
    NUM_WORKERS  = 4
    SEED         = 42

    EVAL_EVERY   = 5
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
# 1. DATA (with GPS key normalization fix)
# ============================================================================

def normalize_fid(fid):
    """Normalize folder ID — strip leading zeros for consistent matching."""
    try:
        return str(int(fid))
    except ValueError:
        return fid


def _parse_coord(s):
    s = s.strip()
    sign = 1.0
    if s[0] in ('S', 's', 'W', 'w'):
        sign = -1.0
        s = s[1:]
    elif s[0] in ('N', 'n', 'E', 'e'):
        s = s[1:]
    return sign * float(s)


def parse_gps_file(path):
    """Parse DenseUAV GPS file → dict[normalized_folder_id] = (lat, lon)."""
    gps = {}
    with open(path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 3:
                folder_id = normalize_fid(parts[0])
                lat = _parse_coord(parts[1])
                lon = _parse_coord(parts[2])
                gps[folder_id] = (lat, lon)
    return gps


def get_image_paths(root_dir):
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
                # Use normalized folder ID
                folders[normalize_fid(folder)] = imgs
    return folders


class DenseUAVTrainDataset(Dataset):
    def __init__(self, drone_root, sat_root, transform_drone, transform_sat):
        self.drone_folders = get_image_paths(drone_root)
        self.sat_folders   = get_image_paths(sat_root)
        common_ids = sorted(set(self.drone_folders.keys()) & set(self.sat_folders.keys()))
        self.ids = common_ids
        print(f"[Train] {len(self.ids)} matched locations")
        self.transform_drone = transform_drone
        self.transform_sat   = transform_sat

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        fid = self.ids[idx]
        drone_path = random.choice(self.drone_folders[fid])
        sat_path   = random.choice(self.sat_folders[fid])
        drone_img = self.transform_drone(Image.open(drone_path).convert("RGB"))
        sat_img   = self.transform_sat(Image.open(sat_path).convert("RGB"))
        return drone_img, sat_img, idx


class DenseUAVTestDataset(Dataset):
    def __init__(self, root_dir, transform, view_type="drone"):
        self.transform = transform
        self.samples = []
        folders = get_image_paths(root_dir)
        for fid in sorted(folders.keys()):
            for img_path in folders[fid]:
                self.samples.append((img_path, fid))
        print(f"[Test-{view_type}] {len(self.samples)} images from {len(folders)} locations")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, fid = self.samples[idx]
        img = self.transform(Image.open(img_path).convert("RGB"))
        return img, fid


def build_transforms():
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_drone = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
        transforms.ToTensor(),
        normalize,
    ])
    train_sat = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=180, fill=0),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
# 2. MODEL — DINOv2 + GeM Pooling
# ============================================================================

class GeM(nn.Module):
    """Generalized Mean Pooling on patch tokens."""
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        # x: (B, N, C) — patch tokens
        x = x.clamp(min=self.eps).pow(self.p)
        x = x.mean(dim=1)
        return x.pow(1.0 / self.p)


class GeoModel(nn.Module):
    """DINOv2 + GeM pooling + projection head."""

    def __init__(self, backbone_name=CFG.BACKBONE, embed_dim=CFG.EMBED_DIM):
        super().__init__()
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", backbone_name, pretrained=True
        )
        feat_dim = self.backbone.embed_dim  # 768

        # GeM pooling for patch tokens
        self.gem = GeM(p=3.0)

        # Projection: CLS(768) + GeM(768) → concat(1536) → embed_dim
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
        print(f"[Model] Unfroze last {n} blocks. Trainable: {trainable/1e6:.1f}M / {total/1e6:.1f}M")

    def forward(self, x):
        # Forward through backbone manually to get both CLS and patch tokens
        x = self.backbone.prepare_tokens_with_masks(x)
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)

        cls_token = x[:, 0]      # (B, 768)
        patch_tokens = x[:, 1:]  # (B, N, 768)

        gem_feat = self.gem(patch_tokens)  # (B, 768)
        combined = torch.cat([cls_token, gem_feat], dim=1)  # (B, 1536)

        feat = self.proj(combined)
        feat = F.normalize(feat, p=2, dim=-1)
        return feat


# ============================================================================
# 3. LOSS — InfoNCE with label smoothing
# ============================================================================

class SymmetricInfoNCE(nn.Module):
    def __init__(self, temperature=CFG.TEMPERATURE, label_smoothing=CFG.LABEL_SMOOTH):
        super().__init__()
        self.temperature = temperature
        self.label_smoothing = label_smoothing

    def forward(self, feat_d, feat_s):
        logits = feat_d @ feat_s.T / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss_d2s = F.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        loss_s2d = F.cross_entropy(logits.T, labels, label_smoothing=self.label_smoothing)
        return 0.5 * (loss_d2s + loss_s2d)


# ============================================================================
# 4. EVALUATION (with GPS key fix)
# ============================================================================

def extract_features(model, dataloader, device):
    model.eval()
    all_feats, all_fids = [], []
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
    return np.concatenate(all_feats), all_fids, t_total


def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def compute_recalls(query_feats, gallery_feats, query_fids, gallery_fids, k_list=[1, 5, 10]):
    gallery_fid_to_indices = defaultdict(list)
    for i, fid in enumerate(gallery_fids):
        gallery_fid_to_indices[fid].append(i)

    sim_matrix = query_feats @ gallery_feats.T
    recalls = {}
    max_k = max(k_list)
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
    top_1pct = np.argsort(-sim_matrix, axis=1)[:, :one_pct]
    correct = sum(1 for q_idx in range(len(query_fids))
                  if query_fids[q_idx] in [gallery_fids[g] for g in top_1pct[q_idx]])
    recalls["R@1%"] = correct / len(query_fids)

    # AP
    aps = []
    for q_idx in range(len(query_fids)):
        q_fid = query_fids[q_idx]
        sorted_gids = [gallery_fids[g] for g in np.argsort(-sim_matrix[q_idx])]
        n_relevant = len(gallery_fid_to_indices[q_fid])
        if n_relevant == 0:
            continue
        hits, precision_sum = 0, 0.0
        for rank, gid in enumerate(sorted_gids, 1):
            if gid == q_fid:
                hits += 1
                precision_sum += hits / rank
        aps.append(precision_sum / n_relevant)
    recalls["AP"] = np.mean(aps) if aps else 0.0
    return recalls


def compute_sdm(query_feats, gallery_feats, query_fids, gallery_fids, gps_dict, k_list=[1, 5, 10]):
    # Aggregate per-location features
    def agg_by_fid(feats, fids):
        unique_fids = sorted(set(fids))
        sums = defaultdict(lambda: np.zeros(feats.shape[1]))
        counts = defaultdict(int)
        for i, fid in enumerate(fids):
            sums[fid] += feats[i]
            counts[fid] += 1
        mat = []
        for fid in unique_fids:
            f = sums[fid] / counts[fid]
            mat.append(f / (np.linalg.norm(f) + 1e-8))
        return np.stack(mat), unique_fids

    q_mat, q_fids_u = agg_by_fid(query_feats, query_fids)
    g_mat, g_fids_u = agg_by_fid(gallery_feats, gallery_fids)
    loc_sim = q_mat @ g_mat.T

    sdm = {}
    for k in k_list:
        distances = []
        for q_idx, q_fid in enumerate(q_fids_u):
            if q_fid not in gps_dict:
                continue
            q_lat, q_lon = gps_dict[q_fid]
            top_k = np.argsort(-loc_sim[q_idx])[:k]
            min_dist = float('inf')
            for loc_idx in top_k:
                g_fid = g_fids_u[loc_idx]
                if g_fid in gps_dict:
                    g_lat, g_lon = gps_dict[g_fid]
                    min_dist = min(min_dist, haversine(q_lat, q_lon, g_lat, g_lon))
            if min_dist < float('inf'):
                distances.append(min_dist)
        if distances:
            mean_dist = np.mean(distances)
            sdm[f"SDM@{k}"] = float(np.exp(-mean_dist / 100.0))
            sdm[f"MeanDist@{k}_m"] = float(mean_dist)
        else:
            sdm[f"SDM@{k}"] = 0.0
            sdm[f"MeanDist@{k}_m"] = float('inf')
    return sdm


def compute_ma(query_feats, gallery_feats, query_fids, gallery_fids,
               gps_dict, thresholds_m=[5, 10, 25]):
    def agg_by_fid(feats, fids):
        unique_fids = sorted(set(fids))
        sums = defaultdict(lambda: np.zeros(feats.shape[1]))
        counts = defaultdict(int)
        for i, fid in enumerate(fids):
            sums[fid] += feats[i]
            counts[fid] += 1
        mat = []
        for fid in unique_fids:
            f = sums[fid] / counts[fid]
            mat.append(f / (np.linalg.norm(f) + 1e-8))
        return np.stack(mat), unique_fids

    q_mat, q_fids_u = agg_by_fid(query_feats, query_fids)
    g_mat, g_fids_u = agg_by_fid(gallery_feats, gallery_fids)
    loc_sim = q_mat @ g_mat.T
    top1 = np.argmax(loc_sim, axis=1)

    errors = []
    for q_idx, q_fid in enumerate(q_fids_u):
        if q_fid not in gps_dict:
            continue
        g_fid = g_fids_u[top1[q_idx]]
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
    # Debug: report how many locations had GPS matches
    ma["gps_matched_locs"] = len(errors)
    return ma


def evaluate(model, test_query_loader, test_gallery_loader, gps_dict, device):
    q_feats, q_fids, q_time = extract_features(model, test_query_loader, device)
    g_feats, g_fids, g_time = extract_features(model, test_gallery_loader, device)

    n_q, n_g = len(q_fids), len(g_fids)

    t_ret = time.time()
    recalls = compute_recalls(q_feats, g_feats, q_fids, g_fids)
    t_ret = time.time() - t_ret

    sdm = compute_sdm(q_feats, g_feats, q_fids, g_fids, gps_dict)
    ma  = compute_ma(q_feats, g_feats, q_fids, g_fids, gps_dict)

    timing = {
        "infer_query_total_s": round(q_time, 2),
        "infer_gallery_total_s": round(g_time, 2),
        "infer_per_query_ms": round(q_time / max(n_q, 1) * 1000, 2),
        "infer_per_gallery_ms": round(g_time / max(n_g, 1) * 1000, 2),
        "retrieval_time_s": round(t_ret, 2),
        "n_queries": n_q, "n_gallery": n_g,
    }
    return {**recalls, **sdm, **ma, **timing}


# ============================================================================
# 5. TRAINING
# ============================================================================

def train_one_epoch(model, dataloader, optimizer, criterion, scaler, device, epoch):
    model.train()
    total_loss, count = 0, 0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CFG.EPOCHS}", leave=False)
    for drone_imgs, sat_imgs, _ in pbar:
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
    print("  EXP03 — DenseUAV Improved (GeM + 518px + Label Smooth)")
    print(f"  Backbone: {CFG.BACKBONE} | Embed: {CFG.EMBED_DIM} | IMG: {CFG.IMG_SIZE}")
    print(f"  Device: {CFG.DEVICE}")
    print("=" * 60)

    root = CFG.DATASET_ROOT
    train_drone_dir  = os.path.join(root, "train", "drone")
    train_sat_dir    = os.path.join(root, "train", "satellite")
    test_query_dir   = os.path.join(root, "test", "query_drone")
    test_gallery_dir = os.path.join(root, "test", "gallery_satellite")

    for p, name in [(train_drone_dir, "train/drone"),
                     (train_sat_dir, "train/satellite"),
                     (test_query_dir, "test/query_drone"),
                     (test_gallery_dir, "test/gallery_satellite")]:
        assert os.path.exists(p), f"Missing: {p}"
        print(f"  ✓ {name}: {len(os.listdir(p))} folders")

    # GPS (try multiple file names)
    gps_dict = {}
    for gps_name in ["Dense_GPS_test.txt", "Dense_GPS_ALL.txt",
                      "Dense_GPS_all.txt", "dense_gps_test.txt"]:
        gps_file = os.path.join(root, gps_name)
        if os.path.exists(gps_file):
            gps_dict.update(parse_gps_file(gps_file))
            print(f"  ✓ GPS loaded from {gps_name}: {len(gps_dict)} entries")

    # Debug: check GPS key overlap with test folders
    test_fids = set(normalize_fid(f) for f in os.listdir(test_query_dir) if os.path.isdir(os.path.join(test_query_dir, f)))
    gps_overlap = test_fids & set(gps_dict.keys())
    print(f"  GPS ∩ test_query: {len(gps_overlap)}/{len(test_fids)} locations")
    if len(gps_overlap) == 0 and len(gps_dict) > 0:
        # Debug: show sample keys
        sample_gps = list(gps_dict.keys())[:3]
        sample_test = list(test_fids)[:3]
        print(f"  [WARN] GPS keys sample: {sample_gps}")
        print(f"  [WARN] Test folder sample: {sample_test}")

    train_drone_tf, train_sat_tf, test_tf = build_transforms()

    train_dataset = DenseUAVTrainDataset(train_drone_dir, train_sat_dir, train_drone_tf, train_sat_tf)
    test_query_ds = DenseUAVTestDataset(test_query_dir, test_tf, "query_drone")
    test_gallery_ds = DenseUAVTestDataset(test_gallery_dir, test_tf, "gallery_sat")

    train_loader = DataLoader(train_dataset, batch_size=CFG.BATCH_SIZE, shuffle=True,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    test_q_loader = DataLoader(test_query_ds, batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
                               num_workers=CFG.NUM_WORKERS, pin_memory=True)
    test_g_loader = DataLoader(test_gallery_ds, batch_size=CFG.BATCH_SIZE * 2, shuffle=False,
                               num_workers=CFG.NUM_WORKERS, pin_memory=True)

    model = GeoModel().to(CFG.DEVICE)
    criterion = SymmetricInfoNCE()
    scaler = GradScaler()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.EPOCHS, eta_min=1e-6)

    best_r1 = 0.0
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
                optimizer, T_max=CFG.EPOCHS - epoch, eta_min=1e-6
            )

        loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler, CFG.DEVICE, epoch)
        scheduler.step()
        elapsed = time.time() - t0

        log = {"epoch": epoch + 1, "loss": round(loss, 5),
               "lr": round(optimizer.param_groups[0]["lr"], 7), "time_s": round(elapsed, 1)}

        if (epoch + 1) % CFG.EVAL_EVERY == 0 or epoch == CFG.EPOCHS - 1:
            metrics = evaluate(model, test_q_loader, test_g_loader, gps_dict, CFG.DEVICE)
            log.update(metrics)

            r1 = metrics["R@1"]
            if r1 > best_r1:
                best_r1 = r1
                torch.save(model.state_dict(), os.path.join(CFG.OUTPUT_DIR, "best_exp03.pth"))
                log["best"] = True

            print(f"Epoch {epoch+1:3d} | loss={loss:.4f} | "
                  f"R@1={metrics['R@1']:.4f} R@5={metrics['R@5']:.4f} R@10={metrics['R@10']:.4f} | "
                  f"AP={metrics['AP']:.4f} | SDM@1={metrics.get('SDM@1',0):.4f} | "
                  f"MA@10m={metrics.get('MA@10m',0):.4f} | "
                  f"infer={metrics['infer_per_query_ms']:.1f}ms/q | {elapsed:.0f}s")
        else:
            print(f"Epoch {epoch+1:3d} | loss={loss:.4f} | lr={log['lr']:.2e} | {elapsed:.0f}s")

        history.append(log)

    # Final eval with best model
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION (best model)")
    print("=" * 60)
    model.load_state_dict(torch.load(os.path.join(CFG.OUTPUT_DIR, "best_exp03.pth")))
    final_metrics = evaluate(model, test_q_loader, test_g_loader, gps_dict, CFG.DEVICE)
    for k, v in sorted(final_metrics.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    results = {
        "experiment": "exp03_denseuav_improved",
        "improvements": "GeM pooling, IMG_SIZE=518, label_smoothing=0.1, 6 blocks unfrozen, 60 epochs",
        "backbone": CFG.BACKBONE, "embed_dim": CFG.EMBED_DIM, "img_size": CFG.IMG_SIZE,
        "epochs": CFG.EPOCHS, "batch_size": CFG.BATCH_SIZE,
        "best_R@1": best_r1, "final_metrics": final_metrics, "history": history,
    }

    out_path = os.path.join(CFG.OUTPUT_DIR, "results_exp03_denseuav_improved.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
