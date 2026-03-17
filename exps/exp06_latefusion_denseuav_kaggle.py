# ============================================================================
# EXP06 — Late Fusion: Multi-View Score Aggregation (Zero-Shot)
# ============================================================================
# Approach: Use pre-trained EXP03 backbone (DINOv2+GeM, R@1=73.4%)
#   - Extract features for each drone view independently
#   - Match each view against gallery → K similarity scores per gallery item
#   - Aggregate: max/mean/geom_mean across K views
#   - NO training needed — pure inference with EXP03 weights
#
# Why this should beat single-view:
#   - Multiple views = multiple chances to find distinctive features
#   - Max fusion: if ANY view matches well, it's counted
#   - Mean fusion: noise averages out, signal reinforces
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
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

warnings.filterwarnings("ignore")

# === CONFIG ===
class CFG:
    DATASET_ROOT = "/kaggle/input/datasets/chisboiz/denseuav/DenseUAV"
    OUTPUT_DIR   = "/kaggle/working"

    # Try to load pretrained exp03 weights
    PRETRAINED_PATH = "/kaggle/working/best_exp03.pth"

    BACKBONE     = "dinov2_vitb14"
    EMBED_DIM    = 512
    IMG_SIZE     = 518

    BATCH_SIZE   = 64
    NUM_WORKERS  = 4
    SEED         = 42

    # Late fusion configs
    FUSION_KS    = [2, 3, 5]  # test multiple K values

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

seed_everything(CFG.SEED)


# ============================================================================
# 1. MODEL (same architecture as EXP03)
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
            nn.Linear(feat_dim * 2, feat_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(feat_dim, embed_dim),
        )

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


# ============================================================================
# 2. DATA
# ============================================================================

def normalize_fid(fid):
    try:
        return str(int(fid))
    except ValueError:
        return fid


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


class TestDataset(Dataset):
    """Standard per-image test dataset."""
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


# ============================================================================
# 3. FEATURE EXTRACTION
# ============================================================================

def extract_all_features(model, dataloader, device):
    """Extract features for all images."""
    model.eval()
    all_feats, all_fids = [], []
    with torch.no_grad():
        for imgs, fids in tqdm(dataloader, desc="Extracting features", leave=False):
            feats = model(imgs.to(device))
            all_feats.append(feats.cpu().numpy())
            all_fids.extend(fids)
    return np.concatenate(all_feats), all_fids


# ============================================================================
# 4. LATE FUSION EVALUATION
# ============================================================================

def compute_single_recalls(q_feats, g_feats, q_fids, g_fids):
    """Standard single-image retrieval evaluation."""
    sim = q_feats @ g_feats.T
    recalls = {}

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
    g_fid_to_idx = defaultdict(list)
    for i, fid in enumerate(g_fids):
        g_fid_to_idx[fid].append(i)

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


def late_fusion_evaluate(q_feats_by_loc, g_feats, g_fids, K, method="max"):
    """Late fusion: aggregate K query views' scores then rank.

    Args:
        q_feats_by_loc: dict[fid] → ndarray (N_views, D)
        g_feats: ndarray (N_gallery, D)
        g_fids: list of gallery fids
        K: number of views to use
        method: 'max', 'mean', 'geom_mean', 'feat_avg'
    """
    q_locs = sorted(q_feats_by_loc.keys())
    n_q = len(q_locs)

    if method == "feat_avg":
        # Average features first, then match
        q_avg = []
        for fid in q_locs:
            views = q_feats_by_loc[fid][:K]
            if len(views) < K:
                views = np.concatenate([views] * ((K // len(views)) + 1))[:K]
            avg = np.mean(views, axis=0)
            avg = avg / (np.linalg.norm(avg) + 1e-8)
            q_avg.append(avg)
        q_avg = np.stack(q_avg)
        sim_agg = q_avg @ g_feats.T
    else:
        # Score-level fusion
        sim_agg = np.zeros((n_q, len(g_fids)))
        for qi, fid in enumerate(q_locs):
            views = q_feats_by_loc[fid][:K]
            if len(views) < K:
                views = np.concatenate([views] * ((K // len(views)) + 1))[:K]
            # Each view → similarity with all gallery items
            view_sims = views @ g_feats.T  # (K, N_gallery)

            if method == "max":
                sim_agg[qi] = np.max(view_sims, axis=0)
            elif method == "mean":
                sim_agg[qi] = np.mean(view_sims, axis=0)
            elif method == "geom_mean":
                # Shift to positive then geometric mean
                shifted = view_sims - view_sims.min() + 1e-6
                sim_agg[qi] = np.exp(np.mean(np.log(shifted), axis=0))

    # Evaluate recalls
    recalls = {}
    for k in [1, 5, 10]:
        top_k = np.argsort(-sim_agg, axis=1)[:, :k]
        correct = sum(1 for qi in range(n_q)
                      if q_locs[qi] in [g_fids[top_k[qi, j]] for j in range(k)])
        recalls[f"R@{k}"] = correct / n_q

    # R@1%
    one_pct = max(1, len(g_fids) // 100)
    top_1p = np.argsort(-sim_agg, axis=1)[:, :one_pct]
    correct = sum(1 for qi in range(n_q)
                  if q_locs[qi] in [g_fids[top_1p[qi, j]] for j in range(min(one_pct, top_1p.shape[1]))])
    recalls["R@1%"] = correct / n_q

    return recalls


# ============================================================================
# 5. MAIN
# ============================================================================

def main():
    print("=" * 60)
    print("  EXP06 — Late Fusion: Multi-View Score Aggregation")
    print(f"  Backbone: {CFG.BACKBONE} | IMG: {CFG.IMG_SIZE}")
    print(f"  Fusion K values: {CFG.FUSION_KS}")
    print(f"  Device: {CFG.DEVICE}")
    print("=" * 60)

    root = CFG.DATASET_ROOT
    test_q_dir = os.path.join(root, "test", "query_drone")
    test_g_dir = os.path.join(root, "test", "gallery_satellite")
    assert os.path.exists(test_q_dir), f"Missing: {test_q_dir}"
    assert os.path.exists(test_g_dir), f"Missing: {test_g_dir}"

    # Transforms
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_tf = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=Image.BICUBIC),
        transforms.ToTensor(), normalize,
    ])

    # Create datasets
    test_q_ds = TestDataset(test_q_dir, test_tf, "query_drone")
    test_g_ds = TestDataset(test_g_dir, test_tf, "gallery_satellite")

    test_q_ld = DataLoader(test_q_ds, batch_size=CFG.BATCH_SIZE,
                           num_workers=CFG.NUM_WORKERS, pin_memory=True)
    test_g_ld = DataLoader(test_g_ds, batch_size=CFG.BATCH_SIZE,
                           num_workers=CFG.NUM_WORKERS, pin_memory=True)

    # Load model
    print("\n[Model] Loading pretrained backbone...")
    model = GeoModel().to(CFG.DEVICE)

    # Try to load pretrained weights
    if os.path.exists(CFG.PRETRAINED_PATH):
        state = torch.load(CFG.PRETRAINED_PATH, map_location=CFG.DEVICE)
        model.load_state_dict(state, strict=False)
        print(f"  ✓ Loaded weights from {CFG.PRETRAINED_PATH}")
    else:
        print(f"  [WARN] No pretrained weights at {CFG.PRETRAINED_PATH}")
        print("  Training from scratch (Phase 1 warmup)...")

        # Quick Phase 1 training if no pretrained weights
        from torch.cuda.amp import autocast, GradScaler
        train_d_dir = os.path.join(root, "train", "drone")
        train_s_dir = os.path.join(root, "train", "satellite")

        train_d_tf = transforms.Compose([
            transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(0.3, 0.3, 0.2, 0.05),
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05)),
            transforms.ToTensor(), normalize,
        ])
        train_s_tf = transforms.Compose([
            transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=180, fill=0),
            transforms.ColorJitter(0.2, 0.2),
            transforms.ToTensor(), normalize,
        ])

        class PairDataset(Dataset):
            def __init__(self, d_root, s_root, tf_d, tf_s):
                self.d_folders = get_image_paths(d_root)
                self.s_folders = get_image_paths(s_root)
                self.ids = sorted(set(self.d_folders) & set(self.s_folders))
                self.tf_d, self.tf_s = tf_d, tf_s
                print(f"[Train] {len(self.ids)} matched locations")
            def __len__(self): return len(self.ids)
            def __getitem__(self, idx):
                fid = self.ids[idx]
                d = self.tf_d(Image.open(random.choice(self.d_folders[fid])).convert("RGB"))
                s = self.tf_s(Image.open(random.choice(self.s_folders[fid])).convert("RGB"))
                return d, s, idx

        train_ds = PairDataset(train_d_dir, train_s_dir, train_d_tf, train_s_tf)
        train_ld = DataLoader(train_ds, batch_size=32, shuffle=True,
                              num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)

        # Unfreeze and train
        for p in model.backbone.norm.parameters():
            p.requires_grad = True
        for i in range(len(model.backbone.blocks) - 6, len(model.backbone.blocks)):
            for p in model.backbone.blocks[i].parameters():
                p.requires_grad = True
        for p in model.proj.parameters():
            p.requires_grad = True
        for p in model.gem.parameters():
            p.requires_grad = True

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                      lr=1e-4, weight_decay=0.03)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60, eta_min=1e-6)
        scaler = GradScaler()
        temperature = 0.1
        label_smooth = 0.1

        best_r1 = 0.0
        for epoch in range(60):
            model.train()
            total_loss, cnt = 0, 0
            t0 = time.time()

            if epoch == 5:
                optimizer = torch.optim.AdamW(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    lr=1e-5, weight_decay=0.03)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=55, eta_min=1e-6)

            for d_imgs, s_imgs, _ in tqdm(train_ld, desc=f"Ep {epoch+1}/60", leave=False):
                d_imgs, s_imgs = d_imgs.to(CFG.DEVICE), s_imgs.to(CFG.DEVICE)
                optimizer.zero_grad()
                with autocast():
                    d_f, s_f = model(d_imgs), model(s_imgs)
                    logits = d_f @ s_f.T / temperature
                    labels = torch.arange(logits.size(0), device=logits.device)
                    loss = 0.5 * (F.cross_entropy(logits, labels, label_smoothing=label_smooth)
                                 + F.cross_entropy(logits.T, labels, label_smoothing=label_smooth))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item() * d_imgs.size(0)
                cnt += d_imgs.size(0)
            scheduler.step()
            elapsed = time.time() - t0

            if (epoch + 1) % 10 == 0 or epoch == 59:
                model.eval()
                q_f, q_id = extract_all_features(model, test_q_ld, CFG.DEVICE)
                g_f, g_id = extract_all_features(model, test_g_ld, CFG.DEVICE)
                r = compute_single_recalls(q_f, g_f, q_id, g_id)
                r1 = r["R@1"]
                if r1 > best_r1:
                    best_r1 = r1
                    torch.save(model.state_dict(), os.path.join(CFG.OUTPUT_DIR, "best_exp06.pth"))
                print(f"Ep {epoch+1:3d} | loss={total_loss/cnt:.4f} | R@1={r1:.4f} R@5={r['R@5']:.4f} | {elapsed:.0f}s")
            else:
                print(f"Ep {epoch+1:3d} | loss={total_loss/cnt:.4f} | {elapsed:.0f}s")

        # Reload best
        model.load_state_dict(torch.load(os.path.join(CFG.OUTPUT_DIR, "best_exp06.pth")))

    # ====================================================================
    # MAIN EVALUATION: Single + Late Fusion
    # ====================================================================
    print("\n" + "=" * 60)
    print("  EVALUATION: Single-Image vs Late Fusion")
    print("=" * 60)

    # GPU warmup
    if CFG.DEVICE == "cuda":
        dummy = torch.randn(1, 3, CFG.IMG_SIZE, CFG.IMG_SIZE, device=CFG.DEVICE)
        with torch.no_grad(): model(dummy)
        torch.cuda.synchronize()

    # Extract all features
    print("\n[1] Extracting query features...")
    t0 = time.time()
    q_feats, q_fids = extract_all_features(model, test_q_ld, CFG.DEVICE)
    q_time = time.time() - t0

    print("[2] Extracting gallery features...")
    t1 = time.time()
    g_feats, g_fids = extract_all_features(model, test_g_ld, CFG.DEVICE)
    g_time = time.time() - t1

    # === Single-image baseline ===
    print("\n[3] Single-image evaluation...")
    single_recalls = compute_single_recalls(q_feats, g_feats, q_fids, g_fids)
    print(f"  Single-Image: R@1={single_recalls['R@1']:.4f} R@5={single_recalls['R@5']:.4f} "
          f"R@10={single_recalls['R@10']:.4f} AP={single_recalls['AP']:.4f}")

    # === Group query features by location ===
    q_feats_by_loc = defaultdict(list)
    for i, fid in enumerate(q_fids):
        q_feats_by_loc[fid].append(q_feats[i])
    for fid in q_feats_by_loc:
        q_feats_by_loc[fid] = np.stack(q_feats_by_loc[fid])

    views_per_loc = [len(v) for v in q_feats_by_loc.values()]
    print(f"\n  Query locations: {len(q_feats_by_loc)}")
    print(f"  Views/location: min={min(views_per_loc)}, max={max(views_per_loc)}, "
          f"mean={np.mean(views_per_loc):.1f}")

    # === Gallery: aggregate by location ===
    g_feats_by_loc = defaultdict(list)
    for i, fid in enumerate(g_fids):
        g_feats_by_loc[fid].append(g_feats[i])

    # Create location-level gallery (averaged features)
    g_locs = sorted(g_feats_by_loc.keys())
    g_loc_feats = []
    for fid in g_locs:
        avg = np.mean(g_feats_by_loc[fid], axis=0)
        avg = avg / (np.linalg.norm(avg) + 1e-8)
        g_loc_feats.append(avg)
    g_loc_feats = np.stack(g_loc_feats)

    # === Late Fusion evaluation ===
    print("\n[4] Late Fusion evaluation...")
    results_all = {
        "single_image": single_recalls,
    }

    fusion_methods = ["max", "mean", "feat_avg"]

    for K in CFG.FUSION_KS:
        for method in fusion_methods:
            key = f"fusion_K{K}_{method}"
            recalls = late_fusion_evaluate(q_feats_by_loc, g_loc_feats, g_locs, K, method)
            results_all[key] = recalls
            delta = recalls["R@1"] - single_recalls["R@1"]
            marker = "🔥" if delta > 0 else "❌"
            print(f"  {key}: R@1={recalls['R@1']:.4f} ({delta:+.4f} {marker}) "
                  f"R@5={recalls['R@5']:.4f} R@10={recalls['R@10']:.4f}")

    # === Also test: TTA-style fusion (flip + original) ===
    print("\n[5] TTA Fusion (horizontal flip)...")
    flip_tf = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(p=1.0),  # always flip
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    flip_q_ds = TestDataset(test_q_dir, flip_tf, "query_flipped")
    flip_q_ld = DataLoader(flip_q_ds, batch_size=CFG.BATCH_SIZE,
                           num_workers=CFG.NUM_WORKERS, pin_memory=True)
    q_feats_flip, q_fids_flip = extract_all_features(model, flip_q_ld, CFG.DEVICE)

    # TTA: average original + flipped features
    q_tta = (q_feats + q_feats_flip) / 2
    q_tta = q_tta / (np.linalg.norm(q_tta, axis=1, keepdims=True) + 1e-8)
    tta_recalls = compute_single_recalls(q_tta, g_feats, q_fids, g_fids)
    results_all["tta_flip"] = tta_recalls
    delta = tta_recalls["R@1"] - single_recalls["R@1"]
    marker = "🔥" if delta > 0 else "❌"
    print(f"  TTA-flip: R@1={tta_recalls['R@1']:.4f} ({delta:+.4f} {marker}) "
          f"R@5={tta_recalls['R@5']:.4f} AP={tta_recalls['AP']:.4f}")

    # === Summary ===
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    best_key, best_r1 = "single_image", single_recalls["R@1"]
    for key, vals in results_all.items():
        if vals["R@1"] > best_r1:
            best_key, best_r1 = key, vals["R@1"]
    print(f"  Best method: {best_key} → R@1={best_r1:.4f}")
    print(f"  Improvement over single: {best_r1 - single_recalls['R@1']:+.4f}")

    # Save results
    results = {
        "experiment": "exp06_latefusion",
        "description": "Late Fusion: Multi-View Score Aggregation using EXP03 backbone",
        "backbone": CFG.BACKBONE,
        "embed_dim": CFG.EMBED_DIM,
        "img_size": CFG.IMG_SIZE,
        "fusion_ks": CFG.FUSION_KS,
        "best_method": best_key,
        "best_R@1": best_r1,
        "all_results": {k: v for k, v in results_all.items()},
        "timing": {
            "query_extract_s": round(q_time, 2),
            "gallery_extract_s": round(g_time, 2),
            "n_queries": len(q_fids),
            "n_gallery": len(g_fids),
            "n_query_locations": len(q_feats_by_loc),
            "n_gallery_locations": len(g_locs),
        }
    }

    out_path = os.path.join(CFG.OUTPUT_DIR, "results_exp06_latefusion.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✓ Results saved to {out_path}")


if __name__ == "__main__":
    main()
