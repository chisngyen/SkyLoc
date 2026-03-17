# ============================================================================
# EXP05 — TG-AVL: Temporal-Geometric Absolute Visual Localization
# ============================================================================
# Novelty: Trajectory-aware drone localization using temporal transformer
#   on top of DINOv2+GeM backbone (frozen from exp03).
#
# Architecture:
#   K drone views → DINOv2+GeM (frozen) → K×512 per-frame embeds
#       → TemporalEncoder (2-layer transformer, [AGG] token)
#       → 512-d trajectory embedding ↔ satellite embedding
#
# Training:
#   Phase 1 (ep 1-20): Train backbone (single-image pairs, like exp03)
#   Phase 2 (ep 21-50): Freeze backbone, train temporal module (trajectory)
#
# Evaluation:
#   - Single-image R@1 (fair comparison with exp03)
#   - Trajectory R@1 (novelty — K views aggregated)
#
# Dataset: chisboiz/denseuav
# ============================================================================

import subprocess, sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

for pkg in ["timm", "scipy"]:
    try:
        __import__(pkg.replace("-", "_").split("==")[0])
    except ImportError:
        install(pkg)

import os, json, math, time, random, warnings, glob
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
    IMG_SIZE     = 518       # DINOv2 native (518/14=37)

    # Temporal
    TRAJ_K       = 3         # views per trajectory (DenseUAV has ~3 views/loc)
    TEMP_LAYERS  = 2         # transformer encoder layers
    TEMP_HEADS   = 8
    TEMP_DROPOUT = 0.1

    # Phase 1: backbone warmup (single-image)
    PHASE1_EPOCHS = 20
    PHASE1_BS     = 32
    PHASE1_LR     = 1e-4
    UNFREEZE_BLOCKS = 6

    # Phase 2: temporal module training (trajectory)
    PHASE2_EPOCHS = 30
    PHASE2_BS     = 16       # smaller BS for trajectory (K images per sample)
    PHASE2_LR     = 5e-4     # higher LR for small module

    WARMUP_EP    = 5
    WEIGHT_DECAY = 0.03
    TEMPERATURE  = 0.1
    LABEL_SMOOTH = 0.1
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
# 1. DATA
# ============================================================================

def normalize_fid(fid):
    """Strip leading zeros for consistent matching."""
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


def auto_discover_gps(root):
    """Auto-discover and parse GPS files anywhere under root."""
    gps_dict = {}
    # Search for any file with 'gps' or 'GPS' in name
    patterns = ["**/[Gg][Pp][Ss]*", "**/*[Gg][Pp][Ss]*", "**/Dense_GPS*",
                "**/metadata/*.txt", "**/*.gps"]
    found_files = set()
    for pat in patterns:
        found_files.update(glob.glob(os.path.join(root, pat), recursive=True))

    for gps_file in sorted(found_files):
        if not os.path.isfile(gps_file):
            continue
        try:
            count = 0
            with open(gps_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        fid_raw = parts[0]
                        try:
                            lat = _parse_coord(parts[1])
                            lon = _parse_coord(parts[2])
                        except (ValueError, IndexError):
                            continue
                        # Store with BOTH normalized and original keys
                        norm_fid = normalize_fid(fid_raw)
                        gps_dict[norm_fid] = (lat, lon)
                        gps_dict[fid_raw] = (lat, lon)
                        count += 1
            if count > 0:
                print(f"  ✓ GPS: {count} entries from {os.path.relpath(gps_file, root)}")
        except Exception as e:
            print(f"  [WARN] Failed to parse {gps_file}: {e}")

    if not gps_dict:
        print("  [WARN] No GPS files found. SDM/MA metrics will be skipped.")
    return gps_dict


def get_image_paths(root_dir):
    """Get folder_id → [image_paths] mapping."""
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
                folders[normalize_fid(folder)] = imgs
    return folders


# --- Phase 1 Dataset: Single-image pairs ---
class SinglePairDataset(Dataset):
    def __init__(self, drone_root, sat_root, tf_drone, tf_sat):
        self.drone_folders = get_image_paths(drone_root)
        self.sat_folders = get_image_paths(sat_root)
        self.ids = sorted(set(self.drone_folders) & set(self.sat_folders))
        print(f"[Phase1-Train] {len(self.ids)} matched locations")
        self.tf_d, self.tf_s = tf_drone, tf_sat

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        fid = self.ids[idx]
        d = self.tf_d(Image.open(random.choice(self.drone_folders[fid])).convert("RGB"))
        s = self.tf_s(Image.open(random.choice(self.sat_folders[fid])).convert("RGB"))
        return d, s, idx


# --- Phase 2 Dataset: Trajectory (K views) ---
class TrajectoryDataset(Dataset):
    """Returns K drone views per location + 1 satellite image."""

    def __init__(self, drone_root, sat_root, tf_drone, tf_sat, K=CFG.TRAJ_K):
        self.drone_folders = get_image_paths(drone_root)
        self.sat_folders = get_image_paths(sat_root)
        self.ids = sorted(set(self.drone_folders) & set(self.sat_folders))
        self.K = K
        self.tf_d, self.tf_s = tf_drone, tf_sat

        # Stats on views per location
        views = [len(self.drone_folders[fid]) for fid in self.ids]
        print(f"[Phase2-Train] {len(self.ids)} locations, "
              f"views/loc: min={min(views)}, max={max(views)}, "
              f"mean={np.mean(views):.1f}, K={K}")

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        fid = self.ids[idx]
        drone_paths = self.drone_folders[fid]

        # Select K views (with replacement if fewer than K)
        if len(drone_paths) >= self.K:
            selected = random.sample(drone_paths, self.K)
        else:
            selected = [random.choice(drone_paths) for _ in range(self.K)]

        drone_imgs = torch.stack([
            self.tf_d(Image.open(p).convert("RGB")) for p in selected
        ])  # (K, C, H, W)

        sat_img = self.tf_s(Image.open(random.choice(self.sat_folders[fid])).convert("RGB"))
        return drone_imgs, sat_img, idx


# --- Test datasets ---
class TestDataset(Dataset):
    def __init__(self, root_dir, transform, name="test"):
        self.transform = transform
        self.samples = []
        folders = get_image_paths(root_dir)
        for fid in sorted(folders.keys()):
            for img_path in folders[fid]:
                self.samples.append((img_path, fid))
        print(f"[{name}] {len(self.samples)} images, {len(folders)} locations")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, fid = self.samples[idx]
        return self.transform(Image.open(path).convert("RGB")), fid


class TrajectoryTestDataset(Dataset):
    """Returns all drone views per location as a trajectory."""

    def __init__(self, root_dir, transform, K=CFG.TRAJ_K, name="traj_test"):
        self.transform = transform
        self.K = K
        folders = get_image_paths(root_dir)
        self.location_ids = sorted(folders.keys())
        self.location_imgs = {fid: folders[fid] for fid in self.location_ids}
        print(f"[{name}] {len(self.location_ids)} locations, K={K}")

    def __len__(self): return len(self.location_ids)

    def __getitem__(self, idx):
        fid = self.location_ids[idx]
        paths = self.location_imgs[fid]
        # Use first K images (deterministic for eval)
        selected = paths[:self.K]
        while len(selected) < self.K:
            selected.append(selected[-1])  # pad by repeating last
        imgs = torch.stack([self.transform(Image.open(p).convert("RGB")) for p in selected])
        return imgs, fid


def build_transforms():
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_d = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
        transforms.ToTensor(), norm,
    ])
    train_s = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=180, fill=0),
        transforms.ColorJitter(0.2, 0.2),
        transforms.ToTensor(), norm,
    ])
    test_tf = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=Image.BICUBIC),
        transforms.ToTensor(), norm,
    ])
    return train_d, train_s, test_tf


# ============================================================================
# 2. MODEL
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


class GeoBackbone(nn.Module):
    """DINOv2 + GeM + projection (same arch as exp03)."""

    def __init__(self, backbone_name=CFG.BACKBONE, embed_dim=CFG.EMBED_DIM):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", backbone_name, pretrained=True)
        feat_dim = self.backbone.embed_dim
        self.gem = GeM(p=3.0)
        self.proj = nn.Sequential(
            nn.Linear(feat_dim * 2, feat_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(feat_dim, embed_dim),
        )
        self._freeze_all()

    def _freeze_all(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_last_n_blocks(self, n):
        for p in self.backbone.norm.parameters():
            p.requires_grad = True
        blocks = self.backbone.blocks
        for i in range(len(blocks) - n, len(blocks)):
            for p in blocks[i].parameters():
                p.requires_grad = True
        # Projection head always trainable
        for p in self.proj.parameters():
            p.requires_grad = True
        for p in self.gem.parameters():
            p.requires_grad = True
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        a = sum(p.numel() for p in self.parameters())
        print(f"[Backbone] Unfroze {n} blocks: {t/1e6:.1f}M / {a/1e6:.1f}M trainable")

    def freeze_all(self):
        """Freeze everything for Phase 2."""
        for p in self.parameters():
            p.requires_grad = False
        print("[Backbone] Fully frozen for Phase 2")

    def forward(self, x):
        x = self.backbone.prepare_tokens_with_masks(x)
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)
        cls_tok = x[:, 0]
        pat_tok = x[:, 1:]
        combined = torch.cat([cls_tok, self.gem(pat_tok)], dim=1)
        return F.normalize(self.proj(combined), p=2, dim=-1)


class TemporalEncoder(nn.Module):
    """Transformer encoder that aggregates K per-frame embeddings into one.

    Uses a learnable [AGG] token prepended to the sequence.
    Output = [AGG] token after self-attention.
    """

    def __init__(self, embed_dim=CFG.EMBED_DIM, n_layers=CFG.TEMP_LAYERS,
                 n_heads=CFG.TEMP_HEADS, dropout=CFG.TEMP_DROPOUT,
                 max_len=16):
        super().__init__()
        self.embed_dim = embed_dim

        # Learnable aggregation token
        self.agg_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)

        # Learned positional encoding (max_len + 1 for [AGG])
        self.pos_embed = nn.Parameter(torch.randn(1, max_len + 1, embed_dim) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_heads, dim_feedforward=embed_dim * 4,
            dropout=dropout, activation='gelu', batch_first=True,
            norm_first=True,  # Pre-norm for stability
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # LayerNorm at output
        self.norm = nn.LayerNorm(embed_dim)

        n_params = sum(p.numel() for p in self.parameters())
        print(f"[TemporalEncoder] {n_params/1e6:.2f}M params, {n_layers} layers, {n_heads} heads")

    def forward(self, x):
        """
        Args:
            x: (B, K, D) — K per-frame embeddings
        Returns:
            (B, D) — aggregated trajectory embedding
        """
        B, K, D = x.shape

        # Prepend [AGG] token
        agg = self.agg_token.expand(B, -1, -1)  # (B, 1, D)
        x = torch.cat([agg, x], dim=1)  # (B, K+1, D)

        # Add positional encoding
        x = x + self.pos_embed[:, :K + 1, :]

        # Transformer self-attention
        x = self.encoder(x)

        # Extract [AGG] token output
        agg_out = self.norm(x[:, 0])  # (B, D)
        return F.normalize(agg_out, p=2, dim=-1)


class TGAVLModel(nn.Module):
    """Full TG-AVL model: GeoBackbone + TemporalEncoder."""

    def __init__(self):
        super().__init__()
        self.backbone = GeoBackbone()
        self.temporal = TemporalEncoder()

    def forward_single(self, x):
        """Single image → embedding (B, D)."""
        return self.backbone(x)

    def forward_trajectory(self, x):
        """Trajectory → embedding (B, D).
        Args: x: (B, K, C, H, W)
        """
        B, K, C, H, W = x.shape
        # Extract per-frame features
        frames = x.reshape(B * K, C, H, W)
        with torch.no_grad() if not any(p.requires_grad for p in self.backbone.parameters()) else torch.enable_grad():
            feats = self.backbone(frames)  # (B*K, D)
        feats = feats.reshape(B, K, -1)  # (B, K, D)

        # Temporal aggregation
        return self.temporal(feats)


# ============================================================================
# 3. LOSS
# ============================================================================

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

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlam = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlam/2)**2
    return 2 * R * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def extract_features_single(model, dataloader, device):
    """Extract per-image features using backbone only."""
    model.eval()
    feats, fids = [], []
    t0 = time.time()
    with torch.no_grad():
        for imgs, fid_batch in tqdm(dataloader, desc="Single feats", leave=False):
            f = model.forward_single(imgs.to(device))
            feats.append(f.cpu().numpy())
            fids.extend(fid_batch)
    return np.concatenate(feats), fids, time.time() - t0


def extract_features_trajectory(model, dataloader, device):
    """Extract per-location trajectory features."""
    model.eval()
    feats, fids = [], []
    t0 = time.time()
    with torch.no_grad():
        for traj_imgs, fid_batch in tqdm(dataloader, desc="Traj feats", leave=False):
            f = model.forward_trajectory(traj_imgs.to(device))
            feats.append(f.cpu().numpy())
            fids.extend(fid_batch)
    return np.concatenate(feats), fids, time.time() - t0


def compute_recalls(q_feats, g_feats, q_fids, g_fids):
    sim = q_feats @ g_feats.T
    recalls = {}
    g_fid_to_idx = defaultdict(list)
    for i, fid in enumerate(g_fids):
        g_fid_to_idx[fid].append(i)

    max_k = 10
    top = np.argsort(-sim, axis=1)[:, :max_k]
    for k in [1, 5, 10]:
        correct = sum(1 for qi in range(len(q_fids))
                      if q_fids[qi] in [g_fids[top[qi, j]] for j in range(k)])
        recalls[f"R@{k}"] = correct / len(q_fids)

    # R@1%
    one_pct = max(1, len(g_fids) // 100)
    top_1p = np.argsort(-sim, axis=1)[:, :one_pct]
    correct = sum(1 for qi in range(len(q_fids))
                  if q_fids[qi] in [g_fids[top_1p[qi, j]] for j in range(min(one_pct, top_1p.shape[1]))])
    recalls["R@1%"] = correct / len(q_fids)

    # AP
    aps = []
    for qi in range(len(q_fids)):
        q_fid = q_fids[qi]
        n_rel = len(g_fid_to_idx[q_fid])
        if n_rel == 0: continue
        ranked = [g_fids[j] for j in np.argsort(-sim[qi])]
        hits, psum = 0, 0.0
        for rank, gid in enumerate(ranked, 1):
            if gid == q_fid:
                hits += 1
                psum += hits / rank
        aps.append(psum / n_rel)
    recalls["AP"] = np.mean(aps) if aps else 0.0
    return recalls


def compute_gps_metrics(q_feats, g_feats, q_fids, g_fids, gps_dict):
    """Compute SDM and MA metrics using GPS coordinates."""
    if not gps_dict:
        return {"SDM@1": 0.0, "MA@10m": 0.0, "gps_matched_locs": 0}

    # Aggregate features by location
    def agg(feats, fids):
        uf = sorted(set(fids))
        sums = defaultdict(lambda: np.zeros(feats.shape[1]))
        cnts = defaultdict(int)
        for i, fid in enumerate(fids):
            sums[fid] += feats[i]
            cnts[fid] += 1
        mat = []
        for fid in uf:
            v = sums[fid] / cnts[fid]
            mat.append(v / (np.linalg.norm(v) + 1e-8))
        return np.stack(mat), uf

    q_m, q_u = agg(q_feats, q_fids)
    g_m, g_u = agg(g_feats, g_fids)
    sim = q_m @ g_m.T
    top1 = np.argmax(sim, axis=1)

    errors = []
    for qi, qf in enumerate(q_u):
        if qf not in gps_dict: continue
        gf = g_u[top1[qi]]
        if gf not in gps_dict: continue
        qlat, qlon = gps_dict[qf]
        glat, glon = gps_dict[gf]
        errors.append(haversine(qlat, qlon, glat, glon))

    metrics = {"gps_matched_locs": len(errors)}
    if errors:
        errors = np.array(errors)
        for t in [5, 10, 25]:
            metrics[f"MA@{t}m"] = float(np.mean(errors <= t))
        metrics["mean_loc_error_m"] = float(np.mean(errors))
        metrics["median_loc_error_m"] = float(np.median(errors))
        for k in [1, 5, 10]:
            topk = np.argsort(-sim, axis=1)[:, :k]
            dists = []
            for qi, qf in enumerate(q_u):
                if qf not in gps_dict: continue
                qlat, qlon = gps_dict[qf]
                md = float('inf')
                for j in range(k):
                    gf = g_u[topk[qi, j]]
                    if gf in gps_dict:
                        glat, glon = gps_dict[gf]
                        md = min(md, haversine(qlat, qlon, glat, glon))
                if md < float('inf'):
                    dists.append(md)
            if dists:
                metrics[f"SDM@{k}"] = float(np.exp(-np.mean(dists) / 100.0))
                metrics[f"MeanDist@{k}_m"] = float(np.mean(dists))
    return metrics


def evaluate_full(model, test_q_loader, test_g_loader, traj_q_loader, gps_dict, device):
    """Evaluate both single-image and trajectory modes."""
    # GPU warmup
    if device == "cuda":
        dummy = torch.randn(1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE, device=device)
        with torch.no_grad(): model.forward_single(dummy)
        torch.cuda.synchronize()

    # Single-image evaluation
    q_feats, q_fids, q_t = extract_features_single(model, test_q_loader, device)
    g_feats, g_fids, g_t = extract_features_single(model, test_g_loader, device)

    single_recalls = compute_recalls(q_feats, g_feats, q_fids, g_fids)
    single_gps = compute_gps_metrics(q_feats, g_feats, q_fids, g_fids, gps_dict)

    # Trajectory evaluation (aggregate K views per query location)
    tq_feats, tq_fids, tq_t = extract_features_trajectory(model, traj_q_loader, device)
    # Gallery: aggregate single-image features by location
    g_agg = defaultdict(list)
    for i, fid in enumerate(g_fids):
        g_agg[fid].append(g_feats[i])
    g_locs = sorted(g_agg.keys())
    g_loc_feats = np.stack([np.mean(g_agg[f], axis=0) for f in g_locs])
    g_loc_feats = g_loc_feats / (np.linalg.norm(g_loc_feats, axis=1, keepdims=True) + 1e-8)

    traj_sim = tq_feats @ g_loc_feats.T
    traj_recalls = {}
    for k in [1, 5, 10]:
        topk = np.argsort(-traj_sim, axis=1)[:, :k]
        correct = sum(1 for qi in range(len(tq_fids))
                      if tq_fids[qi] in [g_locs[topk[qi, j]] for j in range(k)])
        traj_recalls[f"traj_R@{k}"] = correct / len(tq_fids)

    # Timing
    timing = {
        "infer_query_total_s": round(q_t, 2),
        "infer_gallery_total_s": round(g_t, 2),
        "infer_per_query_ms": round(q_t / max(len(q_fids), 1) * 1000, 2),
        "infer_per_gallery_ms": round(g_t / max(len(g_fids), 1) * 1000, 2),
        "infer_traj_total_s": round(tq_t, 2),
        "n_queries": len(q_fids), "n_gallery": len(g_fids),
        "n_traj_queries": len(tq_fids),
    }
    return {**single_recalls, **single_gps, **traj_recalls, **timing}


# ============================================================================
# 5. TRAINING
# ============================================================================

def train_phase1_epoch(model, loader, optimizer, criterion, scaler, device, epoch, total_ep):
    """Phase 1: single-image pair training (backbone warmup)."""
    model.train()
    total_loss, cnt = 0, 0
    pbar = tqdm(loader, desc=f"P1 Ep {epoch+1}/{total_ep}", leave=False)
    for drone, sat, _ in pbar:
        drone, sat = drone.to(device), sat.to(device)
        optimizer.zero_grad()
        with autocast():
            loss = criterion(model.forward_single(drone), model.forward_single(sat))
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * drone.size(0)
        cnt += drone.size(0)
        pbar.set_postfix(loss=f"{total_loss/cnt:.4f}")
    return total_loss / cnt


def train_phase2_epoch(model, loader, optimizer, criterion, scaler, device, epoch, total_ep):
    """Phase 2: trajectory training (temporal module)."""
    model.train()
    # Only temporal module is trainable, backbone is frozen
    total_loss, cnt = 0, 0
    pbar = tqdm(loader, desc=f"P2 Ep {epoch+1}/{total_ep}", leave=False)
    for traj_drone, sat, _ in pbar:
        traj_drone, sat = traj_drone.to(device), sat.to(device)
        optimizer.zero_grad()
        with autocast():
            traj_feat = model.forward_trajectory(traj_drone)
            sat_feat = model.forward_single(sat)
            loss = criterion(traj_feat, sat_feat)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item() * sat.size(0)
        cnt += sat.size(0)
        pbar.set_postfix(loss=f"{total_loss/cnt:.4f}")
    return total_loss / cnt


# ============================================================================
# 6. MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  EXP05 — TG-AVL: Temporal-Geometric AVL")
    print(f"  Backbone: {CFG.BACKBONE} | K={CFG.TRAJ_K} views/trajectory")
    print(f"  Phase1: {CFG.PHASE1_EPOCHS}ep (backbone) | Phase2: {CFG.PHASE2_EPOCHS}ep (temporal)")
    print(f"  Device: {CFG.DEVICE}")
    print("=" * 60)

    root = CFG.DATASET_ROOT
    dirs = {
        "train_drone":  os.path.join(root, "train", "drone"),
        "train_sat":    os.path.join(root, "train", "satellite"),
        "test_query":   os.path.join(root, "test", "query_drone"),
        "test_gallery": os.path.join(root, "test", "gallery_satellite"),
    }
    for name, p in dirs.items():
        assert os.path.exists(p), f"Missing: {p}"
        print(f"  ✓ {name}: {len(os.listdir(p))} folders")

    # GPS auto-discovery
    print("\n[GPS] Auto-discovering GPS files...")
    gps_dict = auto_discover_gps(root)
    # Debug overlap
    test_fids = set(normalize_fid(f) for f in os.listdir(dirs["test_query"])
                    if os.path.isdir(os.path.join(dirs["test_query"], f)))
    overlap = test_fids & set(gps_dict.keys())
    print(f"  GPS ∩ test: {len(overlap)}/{len(test_fids)} locations")
    if len(overlap) == 0 and gps_dict:
        sg = list(gps_dict.keys())[:5]
        st = list(test_fids)[:5]
        print(f"  [DEBUG] GPS keys: {sg}")
        print(f"  [DEBUG] Test folders: {st}")

    # Transforms
    tf_d, tf_s, tf_test = build_transforms()

    # Phase 1 datasets
    p1_train = SinglePairDataset(dirs["train_drone"], dirs["train_sat"], tf_d, tf_s)
    p1_loader = DataLoader(p1_train, batch_size=CFG.PHASE1_BS, shuffle=True,
                           num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)

    # Phase 2 datasets
    p2_train = TrajectoryDataset(dirs["train_drone"], dirs["train_sat"], tf_d, tf_s)
    p2_loader = DataLoader(p2_train, batch_size=CFG.PHASE2_BS, shuffle=True,
                           num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)

    # Test datasets
    test_q_ds = TestDataset(dirs["test_query"], tf_test, "test_query")
    test_g_ds = TestDataset(dirs["test_gallery"], tf_test, "test_gallery")
    traj_q_ds = TrajectoryTestDataset(dirs["test_query"], tf_test, name="traj_query")

    test_q_ld = DataLoader(test_q_ds, batch_size=64, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    test_g_ld = DataLoader(test_g_ds, batch_size=64, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    traj_q_ld = DataLoader(traj_q_ds, batch_size=32, num_workers=CFG.NUM_WORKERS, pin_memory=True)

    # Model
    print("\n[Model] Building TG-AVL...")
    model = TGAVLModel().to(CFG.DEVICE)
    criterion = SymmetricInfoNCE()
    scaler = GradScaler()
    history = []
    best_r1, best_traj_r1 = 0.0, 0.0

    # ========== PHASE 1: Backbone warmup ==========
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Backbone warmup ({CFG.PHASE1_EPOCHS} epochs)")
    print(f"{'='*60}")

    # Unfreeze backbone blocks + proj
    model.backbone.unfreeze_last_n_blocks(CFG.UNFREEZE_BLOCKS)
    # Temporal module trainable too but its loss contribution is 0 in Phase 1

    p1_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(p1_params, lr=CFG.PHASE1_LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.PHASE1_EPOCHS, eta_min=1e-6)

    for epoch in range(CFG.PHASE1_EPOCHS):
        t0 = time.time()
        loss = train_phase1_epoch(model, p1_loader, optimizer, criterion, scaler, CFG.DEVICE,
                                   epoch, CFG.PHASE1_EPOCHS)
        scheduler.step()
        elapsed = time.time() - t0
        log = {"phase": 1, "epoch": epoch + 1, "loss": round(loss, 5), "time_s": round(elapsed, 1)}

        if (epoch + 1) % CFG.EVAL_EVERY == 0 or epoch == CFG.PHASE1_EPOCHS - 1:
            metrics = evaluate_full(model, test_q_ld, test_g_ld, traj_q_ld, gps_dict, CFG.DEVICE)
            log.update(metrics)
            r1 = metrics["R@1"]
            if r1 > best_r1:
                best_r1 = r1
                torch.save(model.state_dict(), os.path.join(CFG.OUTPUT_DIR, "best_exp05_p1.pth"))
                log["best_p1"] = True
            print(f"P1 Ep {epoch+1:3d} | loss={loss:.4f} | "
                  f"R@1={r1:.4f} traj_R@1={metrics.get('traj_R@1',0):.4f} | "
                  f"AP={metrics['AP']:.4f} | {elapsed:.0f}s")
        else:
            print(f"P1 Ep {epoch+1:3d} | loss={loss:.4f} | {elapsed:.0f}s")
        history.append(log)

    # ========== PHASE 2: Temporal module training ==========
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Temporal module ({CFG.PHASE2_EPOCHS} epochs)")
    print(f"{'='*60}")

    # Freeze backbone entirely
    model.backbone.freeze_all()

    # Only temporal module parameters
    p2_params = filter(lambda p: p.requires_grad, model.temporal.parameters())
    optimizer = torch.optim.AdamW(p2_params, lr=CFG.PHASE2_LR, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.PHASE2_EPOCHS, eta_min=1e-5)

    for epoch in range(CFG.PHASE2_EPOCHS):
        t0 = time.time()
        loss = train_phase2_epoch(model, p2_loader, optimizer, criterion, scaler, CFG.DEVICE,
                                   epoch, CFG.PHASE2_EPOCHS)
        scheduler.step()
        elapsed = time.time() - t0
        global_ep = CFG.PHASE1_EPOCHS + epoch + 1
        log = {"phase": 2, "epoch": global_ep, "loss": round(loss, 5), "time_s": round(elapsed, 1)}

        if (epoch + 1) % CFG.EVAL_EVERY == 0 or epoch == CFG.PHASE2_EPOCHS - 1:
            metrics = evaluate_full(model, test_q_ld, test_g_ld, traj_q_ld, gps_dict, CFG.DEVICE)
            log.update(metrics)
            tr1 = metrics.get("traj_R@1", 0)
            if tr1 > best_traj_r1:
                best_traj_r1 = tr1
                torch.save(model.state_dict(), os.path.join(CFG.OUTPUT_DIR, "best_exp05.pth"))
                log["best"] = True
            print(f"P2 Ep {epoch+1:3d} | loss={loss:.4f} | "
                  f"R@1={metrics['R@1']:.4f} traj_R@1={tr1:.4f} | "
                  f"AP={metrics['AP']:.4f} | {elapsed:.0f}s")
        else:
            print(f"P2 Ep {epoch+1:3d} | loss={loss:.4f} | {elapsed:.0f}s")
        history.append(log)

    # ========== FINAL EVALUATION ==========
    print(f"\n{'='*60}")
    print("  FINAL EVALUATION (best trajectory model)")
    print(f"{'='*60}")
    model.load_state_dict(torch.load(os.path.join(CFG.OUTPUT_DIR, "best_exp05.pth")))
    final = evaluate_full(model, test_q_ld, test_g_ld, traj_q_ld, gps_dict, CFG.DEVICE)
    for k, v in sorted(final.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save results
    results = {
        "experiment": "exp05_tgavl",
        "description": "TG-AVL: Temporal-Geometric AVL with DINOv2+GeM+TemporalTransformer",
        "backbone": CFG.BACKBONE, "embed_dim": CFG.EMBED_DIM, "img_size": CFG.IMG_SIZE,
        "traj_K": CFG.TRAJ_K, "temp_layers": CFG.TEMP_LAYERS, "temp_heads": CFG.TEMP_HEADS,
        "phase1_epochs": CFG.PHASE1_EPOCHS, "phase2_epochs": CFG.PHASE2_EPOCHS,
        "best_single_R@1": best_r1, "best_traj_R@1": best_traj_r1,
        "final_metrics": final, "history": history,
    }
    out_path = os.path.join(CFG.OUTPUT_DIR, "results_exp05_tgavl.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {out_path}")
    print(f"  Best single R@1: {best_r1:.4f}")
    print(f"  Best traj  R@1: {best_traj_r1:.4f}")
    delta = best_traj_r1 - best_r1
    print(f"  Δ (traj - single): {'+' if delta > 0 else ''}{delta:.4f} "
          f"{'✅ Temporal helps!' if delta > 0 else '⚠️ No improvement'}")


if __name__ == "__main__":
    main()
