# ============================================================================
# EXP07 — TG-AVL v2: Improved Temporal Module
# ============================================================================
# Fixes from EXP05 failure analysis:
#   1. Phase 1: 40 epochs (was 20) → stronger backbone foundation
#   2. Attention-Weighted Pooling instead of [AGG] token
#   3. Alignment loss: penalize divergence between traj and single embeds
#   4. Joint evaluation: single + trajectory + late fusion comparison
#   5. K=5 views (was 3) for richer context
#
# Architecture:
#   K drone views → DINOv2+GeM (frozen after P1) → K×512 per-frame embeds
#       → AttentionPool (learns which view matters most)
#       → 512-d trajectory embedding ↔ satellite embedding
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
    IMG_SIZE     = 518

    # Temporal
    TRAJ_K       = 5         # more views (was 3 in exp05)
    TEMP_LAYERS  = 2
    TEMP_HEADS   = 8
    TEMP_DROPOUT = 0.1

    # Phase 1: backbone warmup (LONGER than exp05)
    PHASE1_EPOCHS = 40       # was 20 in exp05
    PHASE1_BS     = 32
    PHASE1_LR     = 1e-4
    UNFREEZE_BLOCKS = 6

    # Phase 2: temporal module training
    PHASE2_EPOCHS = 30
    PHASE2_BS     = 12       # smaller BS (K=5 images per sample)
    PHASE2_LR     = 3e-4

    # Alignment loss weight
    ALIGN_WEIGHT  = 0.3      # penalize traj-single divergence

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
# 1. DATA (reuse from EXP05)
# ============================================================================

def normalize_fid(fid):
    try:
        return str(int(fid))
    except ValueError:
        return fid


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
                folders[normalize_fid(folder)] = imgs
    return folders


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


class TrajectoryDataset(Dataset):
    def __init__(self, drone_root, sat_root, tf_drone, tf_sat, K=CFG.TRAJ_K):
        self.drone_folders = get_image_paths(drone_root)
        self.sat_folders = get_image_paths(sat_root)
        self.ids = sorted(set(self.drone_folders) & set(self.sat_folders))
        self.K = K
        self.tf_d, self.tf_s = tf_drone, tf_sat
        views = [len(self.drone_folders[fid]) for fid in self.ids]
        print(f"[Phase2-Train] {len(self.ids)} locations, "
              f"views/loc: min={min(views)}, max={max(views)}, mean={np.mean(views):.1f}, K={K}")

    def __len__(self): return len(self.ids)

    def __getitem__(self, idx):
        fid = self.ids[idx]
        drone_paths = self.drone_folders[fid]
        if len(drone_paths) >= self.K:
            selected = random.sample(drone_paths, self.K)
        else:
            selected = [random.choice(drone_paths) for _ in range(self.K)]
        drone_imgs = torch.stack([self.tf_d(Image.open(p).convert("RGB")) for p in selected])
        sat_img = self.tf_s(Image.open(random.choice(self.sat_folders[fid])).convert("RGB"))
        return drone_imgs, sat_img, idx


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
        selected = paths[:self.K]
        while len(selected) < self.K:
            selected.append(selected[-1])
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
        for p in self.proj.parameters():
            p.requires_grad = True
        for p in self.gem.parameters():
            p.requires_grad = True
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        a = sum(p.numel() for p in self.parameters())
        print(f"[Backbone] Unfroze {n} blocks: {t/1e6:.1f}M / {a/1e6:.1f}M trainable")

    def freeze_all(self):
        for p in self.parameters():
            p.requires_grad = False
        print("[Backbone] Fully frozen")

    def forward(self, x):
        x = self.backbone.prepare_tokens_with_masks(x)
        for blk in self.backbone.blocks:
            x = blk(x)
        x = self.backbone.norm(x)
        cls_tok = x[:, 0]
        pat_tok = x[:, 1:]
        combined = torch.cat([cls_tok, self.gem(pat_tok)], dim=1)
        return F.normalize(self.proj(combined), p=2, dim=-1)


class AttentionPooling(nn.Module):
    """Attention-weighted pooling over K views.
    Learns which view is most informative for matching.
    More interpretable and effective than [AGG] token."""

    def __init__(self, embed_dim=CFG.EMBED_DIM, n_heads=4):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=0.1)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        n_params = sum(p.numel() for p in self.parameters())
        print(f"[AttentionPooling] {n_params/1e3:.1f}K params, {n_heads} heads")

    def forward(self, x):
        """Args: x: (B, K, D). Returns: (B, D)"""
        B = x.shape[0]
        q = self.query.expand(B, -1, -1)  # (B, 1, D)

        # Cross-attention: query attends to all K views
        attn_out, attn_weights = self.attn(q, x, x)  # (B, 1, D), (B, 1, K)
        attn_out = self.norm1(attn_out.squeeze(1))  # (B, D)

        # FFN
        out = attn_out + self.ffn(attn_out)
        out = self.norm2(out)
        return F.normalize(out, p=2, dim=-1), attn_weights.squeeze(1)  # (B, D), (B, K)


class TGAVLv2Model(nn.Module):
    """TG-AVL v2: GeoBackbone + AttentionPooling (improved from EXP05)."""

    def __init__(self):
        super().__init__()
        self.backbone = GeoBackbone()
        self.temporal = AttentionPooling()

    def forward_single(self, x):
        return self.backbone(x)

    def forward_trajectory(self, x, return_attn=False):
        """Args: x: (B, K, C, H, W). Returns: (B, D) and optionally attention weights."""
        B, K, C, H, W = x.shape
        frames = x.reshape(B * K, C, H, W)
        with torch.no_grad() if not any(p.requires_grad for p in self.backbone.parameters()) else torch.enable_grad():
            feats = self.backbone(frames)
        feats = feats.reshape(B, K, -1)
        traj_feat, attn_w = self.temporal(feats)
        if return_attn:
            return traj_feat, attn_w
        return traj_feat

    def forward_trajectory_with_singles(self, x):
        """Returns both trajectory embed AND per-view single embeds."""
        B, K, C, H, W = x.shape
        frames = x.reshape(B * K, C, H, W)
        with torch.no_grad():
            single_feats = self.backbone(frames)  # (B*K, D)
        single_feats_grouped = single_feats.reshape(B, K, -1)  # (B, K, D)
        traj_feat, attn_w = self.temporal(single_feats_grouped)
        return traj_feat, single_feats_grouped, attn_w


# ============================================================================
# 3. LOSSES
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


class AlignmentLoss(nn.Module):
    """Penalize divergence between trajectory and best single-view embed."""
    def forward(self, traj_feat, single_feats):
        """
        Args:
            traj_feat: (B, D)
            single_feats: (B, K, D)
        """
        # Cosine similarity of traj with each single view
        sim = torch.bmm(single_feats, traj_feat.unsqueeze(-1)).squeeze(-1)  # (B, K)
        # Best single view per sample
        best_single = single_feats[torch.arange(sim.size(0)), sim.argmax(dim=1)]  # (B, D)
        # MSE between traj embed and best single-view embed
        return F.mse_loss(traj_feat, best_single)


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

    top = np.argsort(-sim, axis=1)[:, :10]
    for k in [1, 5, 10]:
        correct = sum(1 for qi in range(len(q_fids))
                      if q_fids[qi] in [g_fids[top[qi, j]] for j in range(k)])
        recalls[f"R@{k}"] = correct / len(q_fids)

    one_pct = max(1, len(g_fids) // 100)
    top_1p = np.argsort(-sim, axis=1)[:, :one_pct]
    correct = sum(1 for qi in range(len(q_fids))
                  if q_fids[qi] in [g_fids[top_1p[qi, j]] for j in range(min(one_pct, top_1p.shape[1]))])
    recalls["R@1%"] = correct / len(q_fids)

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


def evaluate_full(model, test_q_ld, test_g_ld, traj_q_ld, device):
    if device == "cuda":
        dummy = torch.randn(1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE, device=device)
        with torch.no_grad(): model.forward_single(dummy)
        torch.cuda.synchronize()

    q_feats, q_fids, q_t = extract_features_single(model, test_q_ld, device)
    g_feats, g_fids, g_t = extract_features_single(model, test_g_ld, device)

    single_recalls = compute_recalls(q_feats, g_feats, q_fids, g_fids)

    # Trajectory evaluation
    tq_feats, tq_fids, tq_t = extract_features_trajectory(model, traj_q_ld, device)
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

    # Late fusion (max-score, using single features grouped by location)
    q_by_loc = defaultdict(list)
    for i, fid in enumerate(q_fids):
        q_by_loc[fid].append(q_feats[i])
    q_loc_ids = sorted(q_by_loc.keys())

    late_sim = np.zeros((len(q_loc_ids), len(g_locs)))
    for qi, fid in enumerate(q_loc_ids):
        views = np.stack(q_by_loc[fid])
        view_sims = views @ g_loc_feats.T  # (K, N)
        late_sim[qi] = np.max(view_sims, axis=0)

    late_recalls = {}
    for k in [1, 5, 10]:
        topk = np.argsort(-late_sim, axis=1)[:, :k]
        correct = sum(1 for qi in range(len(q_loc_ids))
                      if q_loc_ids[qi] in [g_locs[topk[qi, j]] for j in range(k)])
        late_recalls[f"late_R@{k}"] = correct / len(q_loc_ids)

    timing = {
        "infer_query_total_s": round(q_t, 2),
        "infer_gallery_total_s": round(g_t, 2),
        "infer_per_query_ms": round(q_t / max(len(q_fids), 1) * 1000, 2),
        "infer_per_gallery_ms": round(g_t / max(len(g_fids), 1) * 1000, 2),
        "infer_traj_total_s": round(tq_t, 2),
        "n_queries": len(q_fids), "n_gallery": len(g_fids),
        "n_traj_queries": len(tq_fids),
    }
    return {**single_recalls, **traj_recalls, **late_recalls, **timing}


# ============================================================================
# 5. TRAINING
# ============================================================================

def train_phase1_epoch(model, loader, optimizer, criterion, scaler, device, epoch, total_ep):
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


def train_phase2_epoch(model, loader, optimizer, criterion, align_loss_fn, scaler, device, epoch, total_ep):
    """Phase 2: trajectory + alignment loss."""
    model.train()
    total_loss, total_match, total_align, cnt = 0, 0, 0, 0
    pbar = tqdm(loader, desc=f"P2 Ep {epoch+1}/{total_ep}", leave=False)
    for traj_drone, sat, _ in pbar:
        traj_drone, sat = traj_drone.to(device), sat.to(device)
        optimizer.zero_grad()
        with autocast():
            traj_feat, single_feats, attn_w = model.forward_trajectory_with_singles(traj_drone)
            sat_feat = model.forward_single(sat)

            # Main matching loss
            match_loss = criterion(traj_feat, sat_feat)

            # Alignment loss: keep trajectory embed close to best single view
            align = align_loss_fn(traj_feat.float(), single_feats.float())

            loss = match_loss + CFG.ALIGN_WEIGHT * align

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * sat.size(0)
        total_match += match_loss.item() * sat.size(0)
        total_align += align.item() * sat.size(0)
        cnt += sat.size(0)
        pbar.set_postfix(loss=f"{total_loss/cnt:.4f}", m=f"{total_match/cnt:.3f}", a=f"{total_align/cnt:.3f}")
    return total_loss / cnt, total_match / cnt, total_align / cnt


# ============================================================================
# 6. MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  EXP07 — TG-AVL v2: Improved Temporal Module")
    print(f"  Backbone: {CFG.BACKBONE} | K={CFG.TRAJ_K} views/trajectory")
    print(f"  Phase1: {CFG.PHASE1_EPOCHS}ep | Phase2: {CFG.PHASE2_EPOCHS}ep")
    print(f"  Improvements: AttentionPooling, AlignmentLoss, K=5, 40ep warmup")
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

    tf_d, tf_s, tf_test = build_transforms()

    # Datasets
    p1_train = SinglePairDataset(dirs["train_drone"], dirs["train_sat"], tf_d, tf_s)
    p1_loader = DataLoader(p1_train, batch_size=CFG.PHASE1_BS, shuffle=True,
                           num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)

    p2_train = TrajectoryDataset(dirs["train_drone"], dirs["train_sat"], tf_d, tf_s)
    p2_loader = DataLoader(p2_train, batch_size=CFG.PHASE2_BS, shuffle=True,
                           num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)

    test_q_ds = TestDataset(dirs["test_query"], tf_test, "test_query")
    test_g_ds = TestDataset(dirs["test_gallery"], tf_test, "test_gallery")
    traj_q_ds = TrajectoryTestDataset(dirs["test_query"], tf_test, name="traj_query")

    test_q_ld = DataLoader(test_q_ds, batch_size=64, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    test_g_ld = DataLoader(test_g_ds, batch_size=64, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    traj_q_ld = DataLoader(traj_q_ds, batch_size=16, num_workers=CFG.NUM_WORKERS, pin_memory=True)

    # Model
    print("\n[Model] Building TG-AVL v2...")
    model = TGAVLv2Model().to(CFG.DEVICE)
    criterion = SymmetricInfoNCE()
    align_loss_fn = AlignmentLoss()
    scaler = GradScaler()
    history = []
    best_r1, best_traj_r1 = 0.0, 0.0

    # ========== PHASE 1: Backbone warmup (40 epochs) ==========
    print(f"\n{'='*60}")
    print(f"  PHASE 1: Backbone warmup ({CFG.PHASE1_EPOCHS} epochs)")
    print(f"{'='*60}")

    model.backbone.unfreeze_last_n_blocks(CFG.UNFREEZE_BLOCKS)
    p1_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(p1_params, lr=CFG.PHASE1_LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.PHASE1_EPOCHS, eta_min=1e-6)

    for epoch in range(CFG.PHASE1_EPOCHS):
        t0 = time.time()
        loss = train_phase1_epoch(model, p1_loader, optimizer, criterion, scaler,
                                   CFG.DEVICE, epoch, CFG.PHASE1_EPOCHS)
        scheduler.step()
        elapsed = time.time() - t0
        log = {"phase": 1, "epoch": epoch + 1, "loss": round(loss, 5), "time_s": round(elapsed, 1)}

        if (epoch + 1) % CFG.EVAL_EVERY == 0 or epoch == CFG.PHASE1_EPOCHS - 1:
            metrics = evaluate_full(model, test_q_ld, test_g_ld, traj_q_ld, CFG.DEVICE)
            log.update(metrics)
            r1 = metrics["R@1"]
            if r1 > best_r1:
                best_r1 = r1
                torch.save(model.state_dict(), os.path.join(CFG.OUTPUT_DIR, "best_exp07_p1.pth"))
                log["best_p1"] = True
            print(f"P1 Ep {epoch+1:3d} | loss={loss:.4f} | "
                  f"R@1={r1:.4f} late_R@1={metrics.get('late_R@1',0):.4f} | "
                  f"AP={metrics['AP']:.4f} | {elapsed:.0f}s")
        else:
            print(f"P1 Ep {epoch+1:3d} | loss={loss:.4f} | {elapsed:.0f}s")
        history.append(log)

    # ========== PHASE 2: Temporal module (30 epochs) ==========
    print(f"\n{'='*60}")
    print(f"  PHASE 2: Temporal module ({CFG.PHASE2_EPOCHS} epochs)")
    print(f"  Using AttentionPooling + AlignmentLoss (weight={CFG.ALIGN_WEIGHT})")
    print(f"{'='*60}")

    # Freeze backbone, train only temporal
    model.backbone.freeze_all()
    for p in model.temporal.parameters():
        p.requires_grad = True

    p2_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = torch.optim.AdamW(p2_params, lr=CFG.PHASE2_LR, weight_decay=CFG.WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CFG.PHASE2_EPOCHS, eta_min=1e-6)

    for epoch in range(CFG.PHASE2_EPOCHS):
        t0 = time.time()
        loss, m_loss, a_loss = train_phase2_epoch(
            model, p2_loader, optimizer, criterion, align_loss_fn, scaler,
            CFG.DEVICE, epoch, CFG.PHASE2_EPOCHS)
        scheduler.step()
        elapsed = time.time() - t0
        log = {"phase": 2, "epoch": CFG.PHASE1_EPOCHS + epoch + 1,
               "loss": round(loss, 5), "match_loss": round(m_loss, 5),
               "align_loss": round(a_loss, 5), "time_s": round(elapsed, 1)}

        if (epoch + 1) % CFG.EVAL_EVERY == 0 or epoch == CFG.PHASE2_EPOCHS - 1:
            metrics = evaluate_full(model, test_q_ld, test_g_ld, traj_q_ld, CFG.DEVICE)
            log.update(metrics)
            tr1 = metrics.get("traj_R@1", 0)
            lr1 = metrics.get("late_R@1", 0)
            if tr1 > best_traj_r1:
                best_traj_r1 = tr1
                torch.save(model.state_dict(), os.path.join(CFG.OUTPUT_DIR, "best_exp07.pth"))
                log["best"] = True
            print(f"P2 Ep {epoch+1:3d} | loss={loss:.4f} (match={m_loss:.4f} align={a_loss:.4f}) | "
                  f"R@1={metrics['R@1']:.4f} traj_R@1={tr1:.4f} late_R@1={lr1:.4f} | {elapsed:.0f}s")
        else:
            print(f"P2 Ep {epoch+1:3d} | loss={loss:.4f} (match={m_loss:.4f} align={a_loss:.4f}) | {elapsed:.0f}s")
        history.append(log)

    # ========== FINAL EVALUATION ==========
    print("\n" + "=" * 60)
    print("  FINAL EVALUATION")
    print("=" * 60)

    model.load_state_dict(torch.load(os.path.join(CFG.OUTPUT_DIR, "best_exp07.pth")))
    final = evaluate_full(model, test_q_ld, test_g_ld, traj_q_ld, CFG.DEVICE)

    print(f"\n  Single R@1:   {final['R@1']:.4f}")
    print(f"  Traj R@1:     {final.get('traj_R@1', 0):.4f}")
    print(f"  Late R@1:     {final.get('late_R@1', 0):.4f}")
    print(f"  Best overall: {max(final['R@1'], final.get('traj_R@1', 0), final.get('late_R@1', 0)):.4f}")

    for k, v in sorted(final.items()):
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    results = {
        "experiment": "exp07_tgavl_v2",
        "description": "TG-AVL v2: AttentionPooling + AlignmentLoss + 40ep warmup + K=5",
        "backbone": CFG.BACKBONE, "embed_dim": CFG.EMBED_DIM, "img_size": CFG.IMG_SIZE,
        "traj_K": CFG.TRAJ_K, "align_weight": CFG.ALIGN_WEIGHT,
        "phase1_epochs": CFG.PHASE1_EPOCHS, "phase2_epochs": CFG.PHASE2_EPOCHS,
        "best_single_R@1": best_r1, "best_traj_R@1": best_traj_r1,
        "final_metrics": final, "history": history,
    }
    out_path = os.path.join(CFG.OUTPUT_DIR, "results_exp07_tgavl_v2.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
