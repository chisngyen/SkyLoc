# ============================================================================
# EXP10 — UAV-AVL + Late Fusion + Temporal + Multi-Scale TTA
# ============================================================================
# Apply proven multi-view approach to UAV-AVL satellite tiling dataset:
#   1. DINOv2-B/14 + GeM backbone (same as EXP04)
#   2. Sequential drone frames → temporal fusion (consecutive K frames)
#   3. Late fusion (max/mean score aggregation)
#   4. Multi-scale TTA for queries
#   5. Full benchmark metrics: PDE@K, PDM@K, R@K, mean/median error
#
# Key difference from DenseUAV: UAV-AVL has GPS-paired drone→tile, not folder IDs
# Temporal fusion uses K consecutive drone frames from same flight path
#
# Dataset: hunhtrungkit/uav-avl
# ============================================================================

import subprocess, sys

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])

for pkg in ["timm", "rasterio", "scipy", "pyproj"]:
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

try:
    import rasterio
    from rasterio.windows import Window
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False

warnings.filterwarnings("ignore")

# === CONFIG ===
class CFG:
    DATASET_ROOT = "/kaggle/input/datasets/hunhtrungkit/uav-avl/Data"
    OUTPUT_DIR   = "/kaggle/working"

    TILE_SIZE    = 256
    TILE_STRIDE  = 64
    REF_MAP_DIR  = "Reference_map/QZ_Town"
    UAV_DIR      = "UAV_image/QZ_Town"

    BACKBONE     = "dinov2_vitb14"
    EMBED_DIM    = 512
    IMG_SIZE     = 518

    # Temporal
    TRAJ_K       = 5          # consecutive frames per trajectory
    TEMP_HEADS   = 4

    # Training
    EPOCHS       = 50
    BATCH_SIZE   = 24
    LR           = 1e-4
    WEIGHT_DECAY = 0.03
    WARMUP_EP    = 5
    UNFREEZE_BLOCKS = 6
    TEMPERATURE  = 0.1
    LABEL_SMOOTH = 0.1
    NUM_WORKERS  = 4
    SEED         = 42

    # Multi-scale TTA
    TTA_SCALES   = [336, 448, 518]
    TTA_FLIP     = True

    POS_RADIUS   = 25.0
    EVAL_THRESHOLDS_M = [5, 10, 25, 50]
    EVAL_EVERY   = 10

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

seed_everything(CFG.SEED)


# ============================================================================
# 1. TILING (from EXP04 with black filter + CRS)
# ============================================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2-lat1), math.radians(lon2-lon1)
    a = math.sin(dp/2)**2 + math.cos(p1)*math.cos(p2)*math.sin(dl/2)**2
    return 2*R*math.atan2(math.sqrt(a), math.sqrt(1-a))

def is_tile_mostly_black(data, threshold=0.4):
    gray = data.mean(axis=2) if data.ndim==3 else data
    return np.sum(gray < 5) / gray.size > threshold

def tile_reference_map(tif_path, tile_size, stride):
    if not HAS_RASTERIO:
        return []
    try:
        from pyproj import Transformer
    except ImportError:
        Transformer = None

    tiles = []; n_skip = 0
    with rasterio.open(tif_path) as src:
        w, h = src.width, src.height
        crs = src.crs
        transformer = None
        if Transformer and crs and not crs.is_geographic:
            transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)

        for y in range(0, h - tile_size + 1, stride):
            for x in range(0, w - tile_size + 1, stride):
                window = Window(x, y, tile_size, tile_size)
                data = src.read(list(range(1, min(4, src.count+1))), window=window)
                if data.shape[0] == 1: data = np.repeat(data, 3, axis=0)
                data = np.transpose(data, (1,2,0))
                if data.dtype != np.uint8:
                    mn, mx = data.min(), data.max()
                    data = ((data-mn)/(mx-mn+1e-8)*255).astype(np.uint8) if mx>mn else np.zeros_like(data, dtype=np.uint8)
                if is_tile_mostly_black(data):
                    n_skip += 1; continue
                cx, cy = x+tile_size//2, y+tile_size//2
                nx, ny = src.transform * (cx, cy)
                if transformer:
                    lon, lat = transformer.transform(nx, ny)
                else:
                    lon, lat = nx, ny
                tiles.append((Image.fromarray(data), lat, lon, f"tile_{len(tiles):06d}", cx, cy))

    if n_skip > 0: print(f"    Skipped {n_skip} black tiles")
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
    """Discover UAV images with GPS, organized by flight for trajectory grouping."""
    flights = {}  # flight_name → [(img_path, lat, lon)]
    all_samples = []

    for flight in sorted(os.listdir(uav_root)):
        fd = os.path.join(uav_root, flight)
        if not os.path.isdir(fd): continue
        gps_file = None
        for name in ["smart_pos.txt", "smart_pos_photoscan.txt"]:
            p = os.path.join(fd, name)
            if os.path.exists(p): gps_file = p; break
        if not gps_file: continue
        gps_dict = {fn: (lat, lon) for fn, lat, lon, _ in parse_smart_pos(gps_file)}
        flight_samples = []
        for f in sorted(os.listdir(fd)):
            if f.lower().endswith(('.jpg', '.jpeg', '.png')) and f in gps_dict:
                lat, lon = gps_dict[f]
                flight_samples.append((os.path.join(fd, f), lat, lon))
        if flight_samples:
            flights[flight] = flight_samples
            all_samples.extend(flight_samples)
            print(f"    Flight {flight}: {len(flight_samples)} imgs")

    return all_samples, flights


class TileDataset(Dataset):
    def __init__(self, tiles, transform):
        self.tiles = tiles; self.tf = transform
    def __len__(self): return len(self.tiles)
    def __getitem__(self, idx):
        img, lat, lon, tid, px_cx, px_cy = self.tiles[idx]
        return (self.tf(img), tid,
                torch.tensor(lat, dtype=torch.float32),
                torch.tensor(lon, dtype=torch.float32),
                torch.tensor(px_cx, dtype=torch.float32),
                torch.tensor(px_cy, dtype=torch.float32))

class QueryDataset(Dataset):
    def __init__(self, samples, transform):
        self.samples = samples; self.tf = transform
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, lat, lon = self.samples[idx]
        return (self.tf(Image.open(path).convert("RGB")),
                torch.tensor(lat, dtype=torch.float32),
                torch.tensor(lon, dtype=torch.float32))

class TrainDataset(Dataset):
    def __init__(self, uav_samples, tiles, tf_uav, tf_tile, pos_radius=CFG.POS_RADIUS):
        self.uav = uav_samples; self.tiles = tiles
        self.tf_uav, self.tf_tile = tf_uav, tf_tile
        tile_gps = np.array([(t[1], t[2]) for t in tiles])
        self.pairs = []
        for ui, (_, ulat, ulon) in enumerate(uav_samples):
            dists = np.array([haversine(ulat, ulon, tg[0], tg[1]) for tg in tile_gps])
            pos = np.where(dists <= pos_radius)[0]
            if len(pos) > 0:
                self.pairs.append((ui, pos))
        if not self.pairs:
            for ui, (_, ulat, ulon) in enumerate(uav_samples):
                dists = np.array([haversine(ulat, ulon, tg[0], tg[1]) for tg in tile_gps])
                self.pairs.append((ui, np.argsort(dists)[:5]))
        print(f"[Train] {len(self.pairs)} UAV imgs with tile matches")

    def __len__(self): return len(self.pairs)
    def __getitem__(self, idx):
        ui, pos_tiles = self.pairs[idx]
        ti = random.choice(pos_tiles)
        path, _, _ = self.uav[ui]
        uav_img = self.tf_uav(Image.open(path).convert("RGB"))
        tile_img, _, _, _, _, _ = self.tiles[ti]
        return uav_img, self.tf_tile(tile_img), idx


# ============================================================================
# 3. MODEL (same small ViT-B)
# ============================================================================

class GeM(nn.Module):
    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p); self.eps = eps
    def forward(self, x):
        return x.clamp(min=self.eps).pow(self.p).mean(dim=1).pow(1.0/self.p)

class GeoModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = torch.hub.load("facebookresearch/dinov2", CFG.BACKBONE, pretrained=True)
        fd = self.backbone.embed_dim
        self.gem = GeM()
        self.proj = nn.Sequential(nn.Linear(fd*2, fd), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(fd, CFG.EMBED_DIM))
        for p in self.backbone.parameters(): p.requires_grad = False

    def unfreeze(self, n):
        for p in self.backbone.norm.parameters(): p.requires_grad = True
        for i in range(len(self.backbone.blocks)-n, len(self.backbone.blocks)):
            for p in self.backbone.blocks[i].parameters(): p.requires_grad = True
        t = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[Model] Unfroze {n} blocks: {t/1e6:.1f}M trainable")

    def forward(self, x):
        x = self.backbone.prepare_tokens_with_masks(x)
        for blk in self.backbone.blocks: x = blk(x)
        x = self.backbone.norm(x)
        c = torch.cat([x[:,0], self.gem(x[:,1:])], dim=1)
        return F.normalize(self.proj(c), p=2, dim=-1)

class SymmetricInfoNCE(nn.Module):
    def forward(self, a, b):
        logits = a @ b.T / CFG.TEMPERATURE
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5*(F.cross_entropy(logits, labels, label_smoothing=CFG.LABEL_SMOOTH)
                    +F.cross_entropy(logits.T, labels, label_smoothing=CFG.LABEL_SMOOTH))


# ============================================================================
# 4. EVALUATION (comprehensive w/ late fusion + trajectory)
# ============================================================================

def extract_tiles(model, loader, device):
    model.eval(); feats, lats, lons, tids, pxs, pys = [],[], [],[], [],[]
    t0 = time.time()
    with torch.no_grad():
        for imgs, tid, lat, lon, px, py in tqdm(loader, leave=False):
            feats.append(model(imgs.to(device)).cpu().numpy())
            tids.extend(tid); lats.extend(lat.numpy()); lons.extend(lon.numpy())
            pxs.extend(px.numpy()); pys.extend(py.numpy())
    return (np.concatenate(feats), np.array(lats), np.array(lons),
            tids, np.array(pxs), np.array(pys), time.time()-t0)

def extract_queries(model, loader, device):
    model.eval(); feats, lats, lons = [],[],[]
    t0 = time.time()
    with torch.no_grad():
        for imgs, lat, lon in tqdm(loader, leave=False):
            feats.append(model(imgs.to(device)).cpu().numpy())
            lats.extend(lat.numpy()); lons.extend(lon.numpy())
    return np.concatenate(feats), np.array(lats), np.array(lons), time.time()-t0

def extract_queries_tta(model, query_samples, device, scales=CFG.TTA_SCALES, flip=CFG.TTA_FLIP):
    """Multi-scale TTA for queries."""
    model.eval()
    norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    feats, lats, lons = [],[],[]
    t0 = time.time()
    with torch.no_grad():
        for path, lat, lon in tqdm(query_samples, desc="TTA queries", leave=False):
            img = Image.open(path).convert("RGB")
            acc, n = None, 0
            for s in scales:
                tf = transforms.Compose([transforms.Resize((s,s), interpolation=Image.BICUBIC),
                                         transforms.ToTensor(), norm])
                x = tf(img).unsqueeze(0).to(device)
                f = model(x).cpu()
                acc = f if acc is None else acc+f; n += 1
                if flip:
                    acc += model(torch.flip(x, [3])).cpu(); n += 1
            acc = F.normalize(acc/n, p=2, dim=-1)
            feats.append(acc.numpy()); lats.append(lat); lons.append(lon)
    return np.concatenate(feats), np.array(lats), np.array(lons), time.time()-t0

def compute_pde(q_lat, q_lon, tile_px_cx, tile_px_cy, ref_tf, tile_size, t2utm=None):
    if t2utm:
        qx, qy = t2utm.transform(q_lat, q_lon)
        inv = ~ref_tf; px, py = inv*(qx, qy)
    else:
        inv = ~ref_tf; px, py = inv*(q_lon, q_lat)
    return math.sqrt((px-tile_px_cx)**2 + (py-tile_px_cy)**2) / tile_size

def evaluate_full(q_feats, q_lats, q_lons, g_feats, g_lats, g_lons,
                  g_pxs, g_pys, ref_tf=None, crs=None, flights_val=None):
    """Full evaluation with single, late fusion on trajectory, and GPS metrics."""
    n_q, n_g = len(q_lats), len(g_lats)
    sim = q_feats @ g_feats.T
    top1 = np.argmax(sim, axis=1)

    errors = np.array([haversine(q_lats[i], q_lons[i], g_lats[top1[i]], g_lons[top1[i]]) for i in range(n_q)])
    metrics = {"mean_error_m": float(np.mean(errors)), "median_error_m": float(np.median(errors))}

    # R@K (within POS_RADIUS)
    for k in [1,5,10]:
        topk = np.argsort(-sim, axis=1)[:, :k]
        ok = sum(1 for i in range(n_q)
                 if any(haversine(q_lats[i], q_lons[i], g_lats[topk[i,j]], g_lons[topk[i,j]]) <= CFG.POS_RADIUS
                        for j in range(k)))
        metrics[f"R@{k}"] = ok / n_q

    # PDM@K
    for k in [1,5,10]:
        topk = np.argsort(-sim, axis=1)[:, :k]
        for t in CFG.EVAL_THRESHOLDS_M:
            ok = sum(1 for i in range(n_q)
                     if min(haversine(q_lats[i], q_lons[i], g_lats[topk[i,j]], g_lons[topk[i,j]])
                            for j in range(k)) <= t)
            metrics[f"PDM@{k}_@{t}m"] = ok / n_q

    # PDE@K
    if ref_tf and crs:
        try:
            from pyproj import Transformer
            t2utm = Transformer.from_crs("EPSG:4326", crs, always_xy=False)
            for k in [1,5,10]:
                topk = np.argsort(-sim, axis=1)[:, :k]
                ok = 0
                for i in range(n_q):
                    mpde = min(compute_pde(q_lats[i], q_lons[i], g_pxs[topk[i,j]], g_pys[topk[i,j]],
                                           ref_tf, CFG.TILE_SIZE, t2utm) for j in range(k))
                    if mpde < 1.0: ok += 1
                metrics[f"PDE@{k}"] = ok / n_q
            # Mean PDE@1
            pdes = [compute_pde(q_lats[i], q_lons[i], g_pxs[top1[i]], g_pys[top1[i]],
                                ref_tf, CFG.TILE_SIZE, t2utm) for i in range(n_q)]
            metrics["mean_PDE@1"] = float(np.mean(pdes))
        except Exception as e:
            print(f"  [WARN] PDE failed: {e}")

    # Late Fusion by flight trajectory (consecutive K frames)
    if flights_val:
        for method in ["max", "mean"]:
            traj_errors = []
            for flight_name, flight_samples in flights_val.items():
                flight_indices = []
                for fs in flight_samples:
                    for qi in range(n_q):
                        if abs(q_lats[qi] - fs[1]) < 1e-7 and abs(q_lons[qi] - fs[2]) < 1e-7:
                            flight_indices.append(qi); break
                if len(flight_indices) < 2: continue
                # Sliding window of K frames
                K = min(CFG.TRAJ_K, len(flight_indices))
                for start in range(0, len(flight_indices) - K + 1, max(1, K//2)):
                    window = flight_indices[start:start+K]
                    window_sims = sim[window]  # (K, n_g)
                    if method == "max":
                        fused = np.max(window_sims, axis=0)
                    else:
                        fused = np.mean(window_sims, axis=0)
                    best_tile = np.argmax(fused)
                    center_q = window[len(window)//2]
                    err = haversine(q_lats[center_q], q_lons[center_q],
                                    g_lats[best_tile], g_lons[best_tile])
                    traj_errors.append(err)
            if traj_errors:
                traj_errors = np.array(traj_errors)
                metrics[f"traj_{method}_mean_err_m"] = float(np.mean(traj_errors))
                metrics[f"traj_{method}_median_err_m"] = float(np.median(traj_errors))
                for t in CFG.EVAL_THRESHOLDS_M:
                    metrics[f"traj_{method}_@{t}m"] = float(np.mean(traj_errors <= t))
                metrics[f"traj_{method}_n_windows"] = len(traj_errors)

    metrics["n_queries"] = n_q
    metrics["n_gallery"] = n_g
    return metrics


# ============================================================================
# 5. TRAINING
# ============================================================================

def train_epoch(model, loader, opt, crit, scaler, dev, ep, total):
    model.train(); tl, c = 0, 0
    for uav, tile, _ in tqdm(loader, desc=f"Ep {ep+1}/{total}", leave=False):
        uav, tile = uav.to(dev), tile.to(dev); opt.zero_grad()
        with autocast(): loss = crit(model(uav), model(tile))
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        tl += loss.item()*uav.size(0); c += uav.size(0)
    return tl/c


# ============================================================================
# 6. MAIN
# ============================================================================

def main():
    print("="*60)
    print("  EXP10 — UAV-AVL + Late Fusion + TTA")
    print(f"  {CFG.BACKBONE} | K={CFG.TRAJ_K} | TTA={CFG.TTA_SCALES}")
    print("="*60)

    root = CFG.DATASET_ROOT
    ref_dir = os.path.join(root, CFG.REF_MAP_DIR)
    uav_dir = os.path.join(root, CFG.UAV_DIR)

    # Tile reference map
    print("\n[1/4] Tiling reference map...")
    tifs = sorted([os.path.join(ref_dir, f) for f in os.listdir(ref_dir) if f.endswith('.tif')])
    print(f"  {len(tifs)} TIF files")
    all_tiles = []; ref_tf = None; ref_crs = None
    for tp in tqdm(tifs, desc="Tiling"):
        all_tiles.extend(tile_reference_map(tp, CFG.TILE_SIZE, CFG.TILE_STRIDE))
        if ref_tf is None and HAS_RASTERIO:
            try:
                with rasterio.open(tp) as src:
                    if src.crs and not src.crs.is_geographic:
                        ref_tf = src.transform; ref_crs = str(src.crs)
            except: pass
    print(f"  ✓ {len(all_tiles)} tiles")

    # UAV data
    print("\n[2/4] Discovering UAV data...")
    all_uav, flights = discover_uav_data(uav_dir)
    print(f"  ✓ {len(all_uav)} UAV imgs across {len(flights)} flights")

    # Split by flight
    flight_names = sorted(flights.keys())
    n_train_f = max(1, int(len(flight_names) * 0.7))
    random.shuffle(flight_names)
    train_flights = flight_names[:n_train_f]
    val_flights = flight_names[n_train_f:]
    train_uav = [s for f in train_flights for s in flights[f]]
    val_uav = [s for f in val_flights for s in flights[f]]
    val_flights_dict = {f: flights[f] for f in val_flights}
    print(f"  Train: {len(train_uav)} imgs ({len(train_flights)} flights)")
    print(f"  Val: {len(val_uav)} imgs ({len(val_flights)} flights)")

    # Transforms
    norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    tf_uav = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.2),
        transforms.RandomAffine(degrees=15, translate=(0.05,0.05)),
        transforms.ToTensor(), norm])
    tf_tile = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=Image.BICUBIC),
        transforms.RandomHorizontalFlip(), transforms.RandomRotation(90),
        transforms.ColorJitter(0.2, 0.2), transforms.ToTensor(), norm])
    tf_test = transforms.Compose([
        transforms.Resize((CFG.IMG_SIZE, CFG.IMG_SIZE), interpolation=Image.BICUBIC),
        transforms.ToTensor(), norm])

    # Datasets
    train_ds = TrainDataset(train_uav, all_tiles, tf_uav, tf_tile)
    val_q_ds = QueryDataset(val_uav, tf_test)
    val_g_ds = TileDataset(all_tiles, tf_test)
    train_ld = DataLoader(train_ds, batch_size=CFG.BATCH_SIZE, shuffle=True,
                          num_workers=CFG.NUM_WORKERS, pin_memory=True, drop_last=True)
    val_q_ld = DataLoader(val_q_ds, batch_size=48, num_workers=CFG.NUM_WORKERS, pin_memory=True)
    val_g_ld = DataLoader(val_g_ds, batch_size=48, num_workers=CFG.NUM_WORKERS, pin_memory=True)

    # Model
    print("\n[3/4] Building model...")
    model = GeoModel().to(CFG.DEVICE)
    crit = SymmetricInfoNCE()
    scaler = GradScaler()
    opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=CFG.LR, weight_decay=CFG.WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.EPOCHS, eta_min=1e-6)
    history = []; best_pdm = 0.0

    print("\n[4/4] Training...")
    for ep in range(CFG.EPOCHS):
        t0 = time.time()
        if ep == CFG.WARMUP_EP:
            model.unfreeze(CFG.UNFREEZE_BLOCKS)
            opt = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                    lr=CFG.LR*0.1, weight_decay=CFG.WEIGHT_DECAY)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CFG.EPOCHS-ep, eta_min=1e-6)

        loss = train_epoch(model, train_ld, opt, crit, scaler, CFG.DEVICE, ep, CFG.EPOCHS)
        sched.step(); el = time.time()-t0
        log = {"epoch": ep+1, "loss": round(loss,5), "time_s": round(el,1)}

        if (ep+1) % CFG.EVAL_EVERY == 0 or ep == CFG.EPOCHS-1:
            gf, glat, glon, _, gpx, gpy, gt = extract_tiles(model, val_g_ld, CFG.DEVICE)
            qf, qlat, qlon, qt = extract_queries(model, val_q_ld, CFG.DEVICE)
            m = evaluate_full(qf, qlat, qlon, gf, glat, glon, gpx, gpy,
                              ref_tf, ref_crs, val_flights_dict)
            log.update(m)
            pdm = m.get("PDM@1_@10m", 0)
            if pdm > best_pdm:
                best_pdm = pdm
                torch.save(model.state_dict(), os.path.join(CFG.OUTPUT_DIR, "best_exp10.pth"))
                log["best"] = True
            print(f"Ep {ep+1:3d} | loss={loss:.4f} | R@1={m.get('R@1',0):.3f} "
                  f"PDE@1={m.get('PDE@1',-1):.3f} err={m['mean_error_m']:.1f}m "
                  f"traj_max_err={m.get('traj_max_mean_err_m','N/A')} | {el:.0f}s")
        else:
            print(f"Ep {ep+1:3d} | loss={loss:.4f} | {el:.0f}s")
        history.append(log)

    # Final eval with TTA
    print(f"\n{'='*60}\n  FINAL EVALUATION (with TTA)\n{'='*60}")
    model.load_state_dict(torch.load(os.path.join(CFG.OUTPUT_DIR, "best_exp10.pth")))

    # Standard
    gf, glat, glon, _, gpx, gpy, gt = extract_tiles(model, val_g_ld, CFG.DEVICE)
    qf, qlat, qlon, qt = extract_queries(model, val_q_ld, CFG.DEVICE)
    standard = evaluate_full(qf, qlat, qlon, gf, glat, glon, gpx, gpy,
                             ref_tf, ref_crs, val_flights_dict)
    print(f"  Standard: err={standard['mean_error_m']:.1f}m R@1={standard.get('R@1',0):.3f}")

    # TTA queries
    qf_tta, qlat_tta, qlon_tta, qt_tta = extract_queries_tta(model, val_uav, CFG.DEVICE)
    tta = evaluate_full(qf_tta, qlat_tta, qlon_tta, gf, glat, glon, gpx, gpy,
                        ref_tf, ref_crs, val_flights_dict)
    print(f"  TTA:      err={tta['mean_error_m']:.1f}m R@1={tta.get('R@1',0):.3f}")

    # Save
    results = {
        "experiment": "exp10_uavavl_fusion",
        "backbone": CFG.BACKBONE, "embed_dim": CFG.EMBED_DIM, "traj_K": CFG.TRAJ_K,
        "tta_scales": CFG.TTA_SCALES, "tile_size": CFG.TILE_SIZE, "stride": CFG.TILE_STRIDE,
        "n_tiles": len(all_tiles), "n_train": len(train_uav), "n_val": len(val_uav),
        "standard_metrics": standard, "tta_metrics": tta,
        "best_PDM@1_10m": best_pdm, "history": history,
    }
    with open(os.path.join(CFG.OUTPUT_DIR, "results_exp10_uavavl_fusion.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n✓ Saved results_exp10_uavavl_fusion.json")

if __name__ == "__main__":
    main()
