# ============================================================================
# EXP09 — ViT-B + Multi-Scale TTA + Re-ranking (DenseUAV)
# ============================================================================
# Same small ViT-B backbone as EXP08, push accuracy via smarter inference:
#   1. Multi-scale TTA: extract features at 336, 448, 518 → average
#   2. Horizontal flip TTA: original + flipped → average
#   3. QE Re-ranking: Query Expansion with top-K gallery matches
#   4. Late fusion + temporal (from EXP08)
#   5. Load pretrained EXP08 backbone if available (skip training)
#
# Expected: late_mean_R@1 > 85% (vs 81.1% EXP08)
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

    BACKBONE     = "dinov2_vitb14"  # SAME small model
    EMBED_DIM    = 512
    IMG_SIZE     = 518

    # Multi-scale TTA
    TTA_SCALES   = [336, 448, 518]   # 3 scales
    TTA_FLIP     = True              # + horizontal flip
    QE_TOP_K     = 5                 # Query Expansion: average with top-5 gallery

    # Temporal
    TRAJ_K       = 5
    TEMP_HEADS   = 4

    # Training (same as EXP08)
    EPOCHS       = 60
    BATCH_SIZE   = 32
    LR           = 1e-4
    UNFREEZE_BLOCKS = 6

    # Phase 2
    PHASE2_EPOCHS = 30
    PHASE2_BS     = 12
    PHASE2_LR     = 3e-4
    ALIGN_WEIGHT  = 0.3

    WARMUP_EP    = 5
    WEIGHT_DECAY = 0.03
    TEMPERATURE  = 0.1
    LABEL_SMOOTH = 0.1
    NUM_WORKERS  = 4
    SEED         = 42
    EVAL_EVERY   = 10  # less frequent eval (training is same as EXP08)

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
    try: return str(int(fid))
    except ValueError: return fid

def get_image_paths(root_dir):
    folders = {}
    for folder in sorted(os.listdir(root_dir)):
        fp = os.path.join(root_dir, folder)
        if os.path.isdir(fp):
            imgs = sorted([os.path.join(fp, f) for f in os.listdir(fp)
                          if f.lower().endswith(('.jpg', '.jpeg', '.png', '.tif'))])
            if imgs: folders[normalize_fid(folder)] = imgs
    return folders

class SinglePairDataset(Dataset):
    def __init__(self, drone_root, sat_root, tf_d, tf_s):
        self.d = get_image_paths(drone_root)
        self.s = get_image_paths(sat_root)
        self.ids = sorted(set(self.d) & set(self.s))
        print(f"[Train] {len(self.ids)} locs")
        self.tf_d, self.tf_s = tf_d, tf_s
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        fid = self.ids[idx]
        return (self.tf_d(Image.open(random.choice(self.d[fid])).convert("RGB")),
                self.tf_s(Image.open(random.choice(self.s[fid])).convert("RGB")), idx)

class TrajectoryDataset(Dataset):
    def __init__(self, drone_root, sat_root, tf_d, tf_s, K=CFG.TRAJ_K):
        self.d = get_image_paths(drone_root)
        self.s = get_image_paths(sat_root)
        self.ids = sorted(set(self.d) & set(self.s))
        self.K, self.tf_d, self.tf_s = K, tf_d, tf_s
        print(f"[P2-Train] {len(self.ids)} locs, K={K}")
    def __len__(self): return len(self.ids)
    def __getitem__(self, idx):
        fid = self.ids[idx]
        paths = self.d[fid]
        sel = random.sample(paths, min(self.K, len(paths)))
        while len(sel) < self.K: sel.append(random.choice(paths))
        return (torch.stack([self.tf_d(Image.open(p).convert("RGB")) for p in sel]),
                self.tf_s(Image.open(random.choice(self.s[fid])).convert("RGB")), idx)

class RawTestDataset(Dataset):
    """Returns raw PIL images + fids for multi-scale TTA."""
    def __init__(self, root_dir, name="test"):
        self.samples = []
        folders = get_image_paths(root_dir)
        for fid in sorted(folders):
            for p in folders[fid]: self.samples.append((p, fid))
        print(f"[{name}] {len(self.samples)} imgs, {len(folders)} locs")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        return self.samples[idx]  # (path, fid)

class TestDataset(Dataset):
    def __init__(self, root_dir, transform, name="test"):
        self.tf = transform
        self.samples = []
        folders = get_image_paths(root_dir)
        for fid in sorted(folders):
            for p in folders[fid]: self.samples.append((p, fid))
        print(f"[{name}] {len(self.samples)} imgs, {len(folders)} locs")
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        p, fid = self.samples[idx]
        return self.tf(Image.open(p).convert("RGB")), fid

class TrajectoryTestDataset(Dataset):
    def __init__(self, root_dir, transform, K=CFG.TRAJ_K):
        self.tf, self.K = transform, K
        folders = get_image_paths(root_dir)
        self.locs = sorted(folders.keys())
        self.imgs = {f: folders[f] for f in self.locs}
        print(f"[TrajTest] {len(self.locs)} locs, K={K}")
    def __len__(self): return len(self.locs)
    def __getitem__(self, idx):
        fid = self.locs[idx]
        paths = self.imgs[fid][:self.K]
        while len(paths) < self.K: paths.append(paths[-1])
        return torch.stack([self.tf(Image.open(p).convert("RGB")) for p in paths]), fid

def build_transforms(img_size=CFG.IMG_SIZE):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    train_d = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
        transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
        transforms.ToTensor(), norm])
    train_s = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=180, fill=0),
        transforms.ColorJitter(0.2, 0.2), transforms.ToTensor(), norm])
    test = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(), norm])
    return train_d, train_s, test

def build_test_transform(img_size):
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    return transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=Image.BICUBIC),
        transforms.ToTensor(), norm])


# ============================================================================
# 2. MODEL (same ViT-B as EXP08)
# ============================================================================

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0 / self.p)

class GeoBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", CFG.BACKBONE, pretrained=True)
        fd = self.backbone.embed_dim
        self.gem = GeM()
        self.proj = nn.Sequential(
            nn.Linear(fd * 2, fd), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(fd, CFG.EMBED_DIM))
        self._freeze_all()

    def _freeze_all(self):
        for p in self.backbone.parameters(): p.requires_grad = False

    def unfreeze_last_n_blocks(self, n):
        for p in self.backbone.norm.parameters(): p.requires_grad = True
        for i in range(len(self.backbone.blocks) - n, len(self.backbone.blocks)):
            for p in self.backbone.blocks[i].parameters(): p.requires_grad = True
        for p in list(self.proj.parameters()) + list(self.gem.parameters()): p.requires_grad = True
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        a = sum(p.numel() for p in self.parameters())
        print(f"[Backbone] Unfroze {n} blocks: {t/1e6:.1f}M / {a/1e6:.1f}M trainable")

    def freeze_all(self):
        for p in self.parameters(): p.requires_grad = False

    def forward(self, x):
        x = self.backbone.prepare_tokens_with_masks(x)
        for blk in self.backbone.blocks: x = blk(x)
        x = self.backbone.norm(x)
        c = torch.cat([x[:, 0], self.gem(x[:, 1:])], dim=1)
        return F.normalize(self.proj(c), p=2, dim=-1)

class AttentionPooling(nn.Module):
    def __init__(self, embed_dim=CFG.EMBED_DIM, n_heads=CFG.TEMP_HEADS):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=0.1)
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(nn.Linear(embed_dim, embed_dim*2), nn.GELU(),
                                 nn.Dropout(0.1), nn.Linear(embed_dim*2, embed_dim))
    def forward(self, x):
        B = x.shape[0]
        out, w = self.attn(self.query.expand(B,-1,-1), x, x)
        out = self.norm1(out.squeeze(1))
        out = self.norm2(out + self.ffn(out))
        return F.normalize(out, p=2, dim=-1), w.squeeze(1)

class FinalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = GeoBackbone()
        self.temporal = AttentionPooling()
    def forward_single(self, x): return self.backbone(x)
    def forward_trajectory(self, x):
        B, K, C, H, W = x.shape
        with torch.no_grad() if not any(p.requires_grad for p in self.backbone.parameters()) else torch.enable_grad():
            feats = self.backbone(x.reshape(B*K, C, H, W)).reshape(B, K, -1)
        return self.temporal(feats)[0]
    def forward_traj_with_singles(self, x):
        B, K, C, H, W = x.shape
        with torch.no_grad():
            sf = self.backbone(x.reshape(B*K, C, H, W)).reshape(B, K, -1)
        tf, w = self.temporal(sf)
        return tf, sf, w


# ============================================================================
# 3. LOSSES
# ============================================================================

class SymmetricInfoNCE(nn.Module):
    def forward(self, a, b):
        logits = a @ b.T / CFG.TEMPERATURE
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (F.cross_entropy(logits, labels, label_smoothing=CFG.LABEL_SMOOTH)
                     + F.cross_entropy(logits.T, labels, label_smoothing=CFG.LABEL_SMOOTH))

class AlignmentLoss(nn.Module):
    def forward(self, traj, singles):
        sim = torch.bmm(singles, traj.unsqueeze(-1)).squeeze(-1)
        best = singles[torch.arange(sim.size(0)), sim.argmax(dim=1)]
        return F.mse_loss(traj, best)


# ============================================================================
# 4. MULTI-SCALE TTA EXTRACTION
# ============================================================================

def extract_multiscale_tta(model, img_paths_fids, device, scales=CFG.TTA_SCALES, flip=CFG.TTA_FLIP):
    """Extract features with multi-scale + flip TTA. Returns averaged L2-normed features."""
    model.eval()
    norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    all_feats, all_fids = [], []

    t0 = time.time()
    with torch.no_grad():
        for path, fid in tqdm(img_paths_fids, desc="MS-TTA feats", leave=False):
            img = Image.open(path).convert("RGB")
            feat_accum = None
            n_views = 0

            for scale in scales:
                tf = transforms.Compose([
                    transforms.Resize((scale, scale), interpolation=Image.BICUBIC),
                    transforms.ToTensor(), norm])
                x = tf(img).unsqueeze(0).to(device)
                f = model.forward_single(x).cpu()
                feat_accum = f if feat_accum is None else feat_accum + f
                n_views += 1

                if flip:
                    x_flip = torch.flip(x, dims=[3])
                    f_flip = model.forward_single(x_flip).cpu()
                    feat_accum += f_flip
                    n_views += 1

            avg_feat = feat_accum / n_views
            avg_feat = F.normalize(avg_feat, p=2, dim=-1)
            all_feats.append(avg_feat.numpy())
            all_fids.append(fid)

    return np.concatenate(all_feats), all_fids, time.time() - t0


def query_expansion(q_feats, g_feats, top_k=CFG.QE_TOP_K):
    """Average Query Expansion: replace query with avg(query + top-K gallery)."""
    sim = q_feats @ g_feats.T
    topk_idx = np.argsort(-sim, axis=1)[:, :top_k]
    expanded = np.copy(q_feats)
    for i in range(len(q_feats)):
        neighbors = g_feats[topk_idx[i]]
        expanded[i] = np.mean(np.vstack([q_feats[i:i+1], neighbors]), axis=0)
    expanded = expanded / (np.linalg.norm(expanded, axis=1, keepdims=True) + 1e-8)
    return expanded


# ============================================================================
# 5. EVALUATION
# ============================================================================

def compute_recalls(qf, gf, qi, gi):
    sim = qf @ gf.T
    r = {}
    top = np.argsort(-sim, axis=1)[:, :10]
    for k in [1,5,10]:
        r[f"R@{k}"] = sum(qi[i] in [gi[top[i,j]] for j in range(k)] for i in range(len(qi))) / len(qi)
    pct = max(1, len(gi)//100)
    tp = np.argsort(-sim, axis=1)[:, :pct]
    r["R@1%"] = sum(qi[i] in [gi[tp[i,j]] for j in range(min(pct,tp.shape[1]))] for i in range(len(qi))) / len(qi)
    g2i = defaultdict(list)
    for i, f in enumerate(gi): g2i[f].append(i)
    aps = []
    for i in range(len(qi)):
        nr = len(g2i[qi[i]])
        if nr == 0: continue
        ranked = [gi[j] for j in np.argsort(-sim[i])]
        h, ps = 0, 0.0
        for rk, g in enumerate(ranked, 1):
            if g == qi[i]: h += 1; ps += h/rk
        aps.append(ps/nr)
    r["AP"] = np.mean(aps) if aps else 0
    return r

def extract_single(model, loader, device):
    model.eval()
    feats, fids = [], []
    t0 = time.time()
    with torch.no_grad():
        for imgs, f in tqdm(loader, leave=False):
            feats.append(model.forward_single(imgs.to(device)).cpu().numpy())
            fids.extend(f)
    return np.concatenate(feats), fids, time.time() - t0

def extract_traj(model, loader, device):
    model.eval()
    feats, fids = [], []
    t0 = time.time()
    with torch.no_grad():
        for imgs, f in tqdm(loader, leave=False):
            feats.append(model.forward_trajectory(imgs.to(device)).cpu().numpy())
            fids.extend(f)
    return np.concatenate(feats), fids, time.time() - t0

def evaluate_comprehensive(model, test_q_dir, test_g_dir, device):
    """Full evaluation: baseline + TTA + QE + late fusion + temporal."""
    results = {}

    # === 1. Baseline (single-scale, no TTA) ===
    _, _, tf_test = build_transforms()
    tql = DataLoader(TestDataset(test_q_dir, tf_test, "q_base"), batch_size=64, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    tgl = DataLoader(TestDataset(test_g_dir, tf_test, "g_base"), batch_size=64, num_workers=CFG.NUM_WORKERS, pin_memory=True)

    if device == "cuda":
        with torch.no_grad(): model.forward_single(torch.randn(1,3,CFG.IMG_SIZE,CFG.IMG_SIZE,device=device))
        torch.cuda.synchronize()

    qf, qi, qt = extract_single(model, tql, device)
    gf, gi, gt = extract_single(model, tgl, device)
    base_r = compute_recalls(qf, gf, qi, gi)
    results["baseline"] = {**base_r, "infer_q_s": round(qt,2), "infer_g_s": round(gt,2)}
    print(f"  Baseline:  R@1={base_r['R@1']:.4f} R@5={base_r['R@5']:.4f} AP={base_r['AP']:.4f}")

    # === 2. Multi-Scale TTA ===
    raw_q = RawTestDataset(test_q_dir, "q_tta")
    raw_g = RawTestDataset(test_g_dir, "g_tta")
    qf_ms, qi_ms, qt_ms = extract_multiscale_tta(model, raw_q.samples, device)
    gf_ms, gi_ms, gt_ms = extract_multiscale_tta(model, raw_g.samples, device)
    tta_r = compute_recalls(qf_ms, gf_ms, qi_ms, gi_ms)
    results["multiscale_tta"] = {**tta_r, "scales": CFG.TTA_SCALES, "flip": CFG.TTA_FLIP,
                                  "infer_q_s": round(qt_ms,2), "infer_g_s": round(gt_ms,2)}
    print(f"  MS-TTA:    R@1={tta_r['R@1']:.4f} R@5={tta_r['R@5']:.4f} AP={tta_r['AP']:.4f}")

    # === 3. TTA + Query Expansion ===
    qf_qe = query_expansion(qf_ms, gf_ms)
    qe_r = compute_recalls(qf_qe, gf_ms, qi_ms, gi_ms)
    results["tta_qe"] = {**qe_r, "qe_top_k": CFG.QE_TOP_K}
    print(f"  TTA+QE:    R@1={qe_r['R@1']:.4f} R@5={qe_r['R@5']:.4f} AP={qe_r['AP']:.4f}")

    # === 4. Late Fusion (on TTA features) ===
    ga = defaultdict(list)
    for i, f in enumerate(gi_ms): ga[f].append(gf_ms[i])
    gl = sorted(ga.keys())
    glf = np.stack([np.mean(ga[f], axis=0) for f in gl])
    glf = glf / (np.linalg.norm(glf, axis=1, keepdims=True) + 1e-8)

    qbl = defaultdict(list)
    for i, f in enumerate(qi_ms): qbl[f].append(qf_ms[i])
    qli = sorted(qbl.keys())

    for method in ["max", "mean"]:
        ls = np.zeros((len(qli), len(gl)))
        for i, f in enumerate(qli):
            vs = np.stack(qbl[f]) @ glf.T
            ls[i] = np.max(vs, axis=0) if method=="max" else np.mean(vs, axis=0)
        lr = {}
        for k in [1,5,10]:
            tk = np.argsort(-ls, axis=1)[:, :k]
            lr[f"R@{k}"] = sum(qli[i] in [gl[tk[i,j]] for j in range(k)] for i in range(len(qli))) / len(qli)
        results[f"late_{method}_tta"] = lr
        print(f"  Late-{method}-TTA: R@1={lr['R@1']:.4f} R@5={lr['R@5']:.4f}")

    # === 5. Late Fusion + QE ===
    for method in ["max", "mean"]:
        ls = np.zeros((len(qli), len(gl)))
        for i, f in enumerate(qli):
            views_qe = []
            for vi, vf in enumerate(qbl[f]):
                # QE each view
                sim_v = vf @ gf_ms.T
                topk_v = np.argsort(-sim_v)[:CFG.QE_TOP_K]
                expanded_v = np.mean(np.vstack([vf.reshape(1,-1), gf_ms[topk_v]]), axis=0)
                expanded_v = expanded_v / (np.linalg.norm(expanded_v) + 1e-8)
                views_qe.append(expanded_v)
            vs = np.stack(views_qe) @ glf.T
            ls[i] = np.max(vs, axis=0) if method=="max" else np.mean(vs, axis=0)
        lr = {}
        for k in [1,5,10]:
            tk = np.argsort(-ls, axis=1)[:, :k]
            lr[f"R@{k}"] = sum(qli[i] in [gl[tk[i,j]] for j in range(k)] for i in range(len(qli))) / len(qli)
        results[f"late_{method}_tta_qe"] = lr
        print(f"  Late-{method}-TTA-QE: R@1={lr['R@1']:.4f} R@5={lr['R@5']:.4f}")

    # === 6. Temporal (AttentionPooling) ===
    trql = DataLoader(TrajectoryTestDataset(test_q_dir, tf_test), batch_size=16, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    tqf, tqi, tqt = extract_traj(model, trql, device)
    # Use base gallery (location-level)
    ga_base = defaultdict(list)
    for i, f in enumerate(gi): ga_base[f].append(gf[i])
    gl_base = sorted(ga_base.keys())
    glf_base = np.stack([np.mean(ga_base[f], axis=0) for f in gl_base])
    glf_base = glf_base / (np.linalg.norm(glf_base, axis=1, keepdims=True) + 1e-8)

    tsim = tqf @ glf_base.T
    tr = {}
    for k in [1,5,10]:
        tk = np.argsort(-tsim, axis=1)[:, :k]
        tr[f"R@{k}"] = sum(tqi[i] in [gl_base[tk[i,j]] for j in range(k)] for i in range(len(tqi))) / len(tqi)
    results["temporal"] = {**tr, "infer_traj_s": round(tqt,2)}
    print(f"  Temporal:  R@1={tr['R@1']:.4f} R@5={tr['R@5']:.4f}")

    results["n_queries"] = len(qi)
    results["n_gallery"] = len(gi)
    results["n_traj"] = len(tqi)
    results["n_q_locs"] = len(qli)
    results["n_g_locs"] = len(gl)
    return results


# ============================================================================
# 6. TRAINING (same as EXP08)
# ============================================================================

def train_p1(model, loader, opt, crit, scaler, dev, ep, total):
    model.train(); tl, c = 0, 0
    for d, s, _ in tqdm(loader, desc=f"P1 {ep+1}/{total}", leave=False):
        d, s = d.to(dev), s.to(dev); opt.zero_grad()
        with autocast(): loss = crit(model.forward_single(d), model.forward_single(s))
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tl += loss.item()*d.size(0); c += d.size(0)
    return tl/c

def train_p2(model, loader, opt, crit, align, scaler, dev, ep, total):
    model.train(); tl, tm, ta, c = 0, 0, 0, 0
    for td, s, _ in tqdm(loader, desc=f"P2 {ep+1}/{total}", leave=False):
        td, s = td.to(dev), s.to(dev); opt.zero_grad()
        with autocast():
            tf, sf, _ = model.forward_traj_with_singles(td)
            ml = crit(tf, model.forward_single(s))
            al = align(tf.float(), sf.float())
            loss = ml + CFG.ALIGN_WEIGHT * al
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tl += loss.item()*s.size(0); tm += ml.item()*s.size(0)
        ta += al.item()*s.size(0); c += s.size(0)
    return tl/c, tm/c, ta/c


# ============================================================================
# 7. MAIN
# ============================================================================

def main():
    print("="*60)
    print(f"  EXP09 — ViT-B + Multi-Scale TTA + Re-ranking")
    print(f"  Backbone: {CFG.BACKBONE} | TTA scales: {CFG.TTA_SCALES}")
    print(f"  QE top-K: {CFG.QE_TOP_K} | ViT-B (small params!)")
    print("="*60)

    root = CFG.DATASET_ROOT
    dirs = {
        "td": os.path.join(root, "train", "drone"),
        "ts": os.path.join(root, "train", "satellite"),
        "tq": os.path.join(root, "test", "query_drone"),
        "tg": os.path.join(root, "test", "gallery_satellite"),
    }
    for n, p in dirs.items():
        assert os.path.exists(p), f"Missing: {p}"
        print(f"  ✓ {n}: {len(os.listdir(p))} folders")

    # Try load pretrained EXP08
    model = FinalModel().to(CFG.DEVICE)
    crit, align_fn, scaler = SymmetricInfoNCE(), AlignmentLoss(), GradScaler()
    history = []
    best_r1, best_tr1 = 0.0, 0.0

    pretrained = None
    for name in ["best_exp08.pth", "best_exp08_p1.pth", "backbone_exp08.pth"]:
        p = os.path.join(CFG.OUTPUT_DIR, name)
        if os.path.exists(p):
            pretrained = p; break

    if pretrained:
        print(f"\n[SKIP TRAINING] Loading pretrained: {pretrained}")
        state = torch.load(pretrained, map_location=CFG.DEVICE)
        model.load_state_dict(state, strict=False)
        print("  ✓ Loaded pretrained weights")
    else:
        # Full training (same as EXP08)
        print(f"\n[TRAINING] No pretrained found, training from scratch")
        tfd, tfs, _ = build_transforms()
        p1d = SinglePairDataset(dirs["td"], dirs["ts"], tfd, tfs)
        p1l = DataLoader(p1d, batch_size=CFG.BATCH_SIZE, shuffle=True, num_workers=CFG.NUM_WORKERS,
                         pin_memory=True, drop_last=True)
        p2d = TrajectoryDataset(dirs["td"], dirs["ts"], tfd, tfs)
        p2l = DataLoader(p2d, batch_size=CFG.PHASE2_BS, shuffle=True, num_workers=CFG.NUM_WORKERS,
                         pin_memory=True, drop_last=True)

        # Phase 1
        print(f"\n{'='*60}\n  PHASE 1: Backbone ({CFG.EPOCHS} epochs)\n{'='*60}")
        model.backbone.unfreeze_last_n_blocks(CFG.UNFREEZE_BLOCKS)
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.EPOCHS, eta_min=1e-6)

        for ep in range(CFG.EPOCHS):
            t0 = time.time()
            loss = train_p1(model, p1l, opt, crit, scaler, CFG.DEVICE, ep, CFG.EPOCHS)
            sched.step(); el = time.time() - t0
            log = {"phase":1, "epoch":ep+1, "loss":round(loss,5), "time_s":round(el,1)}
            if (ep+1) % CFG.EVAL_EVERY == 0 or ep == CFG.EPOCHS-1:
                _, _, tf_t = build_transforms()
                tql_e = DataLoader(TestDataset(dirs["tq"], tf_t, "q"), batch_size=64, num_workers=CFG.NUM_WORKERS, pin_memory=True)
                tgl_e = DataLoader(TestDataset(dirs["tg"], tf_t, "g"), batch_size=64, num_workers=CFG.NUM_WORKERS, pin_memory=True)
                qf, qi, _ = extract_single(model, tql_e, CFG.DEVICE)
                gf, gi, _ = extract_single(model, tgl_e, CFG.DEVICE)
                r = compute_recalls(qf, gf, qi, gi)
                log.update(r)
                if r["R@1"] > best_r1:
                    best_r1 = r["R@1"]
                    torch.save(model.state_dict(), os.path.join(CFG.OUTPUT_DIR, "best_exp09_p1.pth"))
                    log["best_p1"] = True
                print(f"P1 {ep+1:3d} | loss={loss:.4f} | R@1={r['R@1']:.4f} | {el:.0f}s")
            else:
                print(f"P1 {ep+1:3d} | loss={loss:.4f} | {el:.0f}s")
            history.append(log)

        # Phase 2
        print(f"\n{'='*60}\n  PHASE 2: Temporal ({CFG.PHASE2_EPOCHS} epochs)\n{'='*60}")
        model.backbone.freeze_all()
        for p in model.temporal.parameters(): p.requires_grad = True
        opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr=CFG.PHASE2_LR, weight_decay=CFG.WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.PHASE2_EPOCHS, eta_min=1e-6)

        for ep in range(CFG.PHASE2_EPOCHS):
            t0 = time.time()
            loss, ml, al = train_p2(model, p2l, opt, crit, align_fn, scaler, CFG.DEVICE, ep, CFG.PHASE2_EPOCHS)
            sched.step(); el = time.time() - t0
            log = {"phase":2, "epoch":CFG.EPOCHS+ep+1, "loss":round(loss,5), "time_s":round(el,1)}
            print(f"P2 {ep+1:3d} | loss={loss:.4f} | {el:.0f}s")
            history.append(log)

        torch.save(model.state_dict(), os.path.join(CFG.OUTPUT_DIR, "best_exp09.pth"))

    # === COMPREHENSIVE EVALUATION ===
    print(f"\n{'='*60}\n  COMPREHENSIVE EVALUATION\n{'='*60}")
    results = evaluate_comprehensive(model, dirs["tq"], dirs["tg"], CFG.DEVICE)
    results["history"] = history

    out = {"experiment": "exp09_tta_qe", "backbone": CFG.BACKBONE,
           "embed_dim": CFG.EMBED_DIM, "tta_scales": CFG.TTA_SCALES,
           "qe_top_k": CFG.QE_TOP_K, "pretrained_used": pretrained is not None,
           **results}
    with open(os.path.join(CFG.OUTPUT_DIR, "results_exp09_tta_qe.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n✓ Saved results_exp09_tta_qe.json")

if __name__ == "__main__":
    main()
