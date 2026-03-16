# =============================================================================
# EXP54: Zero-Shot Weather Robustness Evaluation
# =============================================================================
# Copy to Kaggle notebook → Run All → Done
#
# Pure inference — NO training.
# Loads any SPDGeo-DPEA-MAR checkpoint and evaluates on all 10 weather
# conditions using pre-generated test images (same protocol as EXP49-53).
#
# Designed for:
#   1. EXP35 checkpoint (baseline / "our model, no weather training")
#   2. EXP52 best checkpoint (finetune + online aug)
#   3. EXP53 best checkpoint (scratch + online aug)
#   → Change CFG.CHECKPOINT and CFG.EXPERIMENT_NAME accordingly
#
# Kaggle Data Sources:
#   1. SUES-200 original:  satellite gallery + clean drone test
#   2. Weather synthetic:  weather-augmented drone TEST images (for eval)
#   3. Model checkpoint:   any SPDGeo-DPEA-MAR .pth file
# =============================================================================

import subprocess, sys
for _p in ["timm", "tqdm", "thop"]:
    try: __import__(_p)
    except ImportError: subprocess.run(f"pip install -q {_p}", shell=True, capture_output=True)

import os, json, time, datetime, gc
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================
class Config:
    # --- Which checkpoint to evaluate ---
    # Switch between EXP35, EXP52, EXP53 best by changing these two fields
    EXPERIMENT_NAME = "EXP54_ZeroShot_EXP35"
    CHECKPOINT      = "/kaggle/input/models/minh2duy/exp35-spdgeo-dpea-m-unfreeze-blocks-4-6/pytorch/default/1/exp35_dpea_ga_best.pth"

    # For EXP52 best:
    # EXPERIMENT_NAME = "EXP54_ZeroShot_EXP52"
    # CHECKPOINT      = "/kaggle/working/exp52_weather_finetune_online_best.pth"

    # For EXP53 best:
    # EXPERIMENT_NAME = "EXP54_ZeroShot_EXP53"
    # CHECKPOINT      = "/kaggle/working/exp53_weather_scratch_online_best.pth"

    # --- Paths ---
    SUES_ROOT       = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
    WEATHER_ROOT    = "/kaggle/input/datasets/minh2duy/sues200-weather/weather_synthetic"
    OUTPUT_DIR      = "/kaggle/working"

    # --- Dataset structure ---
    DRONE_DIR       = "drone-view"
    SAT_DIR         = "satellite-view"
    ALTITUDES       = ["150", "200", "250", "300"]
    ALT_TO_IDX      = {"150": 0, "200": 1, "250": 2, "300": 3}
    NUM_ALTITUDES   = 4
    TEST_LOCS       = list(range(121, 201))   # 80 test locations
    NUM_CLASSES     = 120

    # --- Model architecture (must match checkpoint) ---
    IMG_SIZE        = 336
    N_PARTS         = 8
    PART_DIM        = 256
    EMBED_DIM       = 512
    TEACHER_DIM     = 768
    CLUSTER_TEMP    = 0.07
    UNFREEZE_BLOCKS = 6
    MASK_RATIO      = 0.30

    # --- Weather conditions ---
    WEATHER_NAMES   = ["normal", "fog", "rain", "snow", "dark", "light",
                       "fog_rain", "fog_snow", "rain_snow", "wind"]

    # --- Inference ---
    USE_AMP         = True
    BATCH_SIZE      = 64
    NUM_WORKERS     = 2
    SEED            = 42


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# MODEL DEFINITION (identical to EXP35 / EXP52 / EXP53)
# =============================================================================
class DINOv2Backbone(nn.Module):
    def __init__(self, unfreeze_blocks=4):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        self.feature_dim = 384; self.patch_size = 14
        for p in self.model.parameters(): p.requires_grad = False
        for blk in self.model.blocks[-unfreeze_blocks:]:
            for p in blk.parameters(): p.requires_grad = True
        for p in self.model.norm.parameters(): p.requires_grad = True
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  DINOv2 ViT-S/14: {frozen/1e6:.1f}M frozen, {trainable/1e6:.1f}M trainable")

    def forward(self, x):
        features = self.model.forward_features(x)
        patch_tokens = features['x_norm_patchtokens']
        cls_token = features['x_norm_clstoken']
        H = x.shape[2] // self.patch_size; W = x.shape[3] // self.patch_size
        return patch_tokens, cls_token, (H, W)


class DeepAltitudeFiLM(nn.Module):
    def __init__(self, num_altitudes=4, feat_dim=256):
        super().__init__()
        self.num_altitudes = num_altitudes
        self.gamma = nn.Parameter(torch.ones(num_altitudes, feat_dim))
        self.beta  = nn.Parameter(torch.zeros(num_altitudes, feat_dim))

    def forward(self, feat, alt_idx=None):
        if alt_idx is None:
            gamma = self.gamma.mean(dim=0, keepdim=True)
            beta  = self.beta.mean(dim=0, keepdim=True)
            return feat * gamma.unsqueeze(0) + beta.unsqueeze(0)
        else:
            gamma = self.gamma[alt_idx]
            beta  = self.beta[alt_idx]
            return feat * gamma.unsqueeze(1) + beta.unsqueeze(1)


class AltitudeAwarePartDiscovery(nn.Module):
    def __init__(self, feat_dim=384, n_parts=8, part_dim=256, temperature=0.07,
                 num_altitudes=4):
        super().__init__()
        self.n_parts = n_parts; self.temperature = temperature
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, part_dim), nn.LayerNorm(part_dim), nn.GELU()
        )
        self.altitude_film = DeepAltitudeFiLM(num_altitudes, part_dim)
        self.prototypes = nn.Parameter(torch.randn(n_parts, part_dim) * 0.02)
        self.refine = nn.Sequential(
            nn.LayerNorm(part_dim), nn.Linear(part_dim, part_dim * 2),
            nn.GELU(), nn.Linear(part_dim * 2, part_dim)
        )
        self.salience_head = nn.Sequential(
            nn.Linear(part_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, patch_features, spatial_hw, alt_idx=None):
        B, N, _ = patch_features.shape; H, W = spatial_hw
        feat = self.feat_proj(patch_features)
        feat = self.altitude_film(feat, alt_idx)
        feat_norm = F.normalize(feat, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum('bnd,kd->bnk', feat_norm, proto_norm) / self.temperature
        assign = F.softmax(sim, dim=-1)
        assign_t = assign.transpose(1, 2)
        mass = assign_t.sum(-1, keepdim=True).clamp(min=1e-6)
        part_feat = torch.bmm(assign_t, feat) / mass
        part_feat = part_feat + self.refine(part_feat)
        device = feat.device
        gy = torch.arange(H, device=device).float() / max(H - 1, 1)
        gx = torch.arange(W, device=device).float() / max(W - 1, 1)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        part_pos = torch.bmm(assign_t, coords.unsqueeze(0).expand(B, -1, -1)) / mass
        salience = self.salience_head(part_feat).squeeze(-1)
        return {'part_features': part_feat, 'part_positions': part_pos,
                'assignment': assign, 'salience': salience,
                'projected_patches': feat}


class PartAwarePooling(nn.Module):
    def __init__(self, part_dim=256, embed_dim=512):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(part_dim, part_dim // 2), nn.Tanh(),
                                  nn.Linear(part_dim // 2, 1))
        self.proj = nn.Sequential(nn.Linear(part_dim * 3, embed_dim), nn.LayerNorm(embed_dim),
                                  nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, part_features, salience=None):
        B, K, D = part_features.shape
        aw = self.attn(part_features)
        if salience is not None: aw = aw + salience.unsqueeze(-1).log().clamp(-10)
        aw = F.softmax(aw, dim=1)
        attn_pool = (aw * part_features).sum(1)
        mean_pool = part_features.mean(1); max_pool = part_features.max(1)[0]
        combined = torch.cat([attn_pool, mean_pool, max_pool], dim=-1)
        return F.normalize(self.proj(combined), dim=-1)


class DynamicFusionGate(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
        )
        nn.init.constant_(self.gate[-1].bias, 0.85)

    def forward(self, part_emb, cls_emb):
        combined = torch.cat([part_emb, cls_emb], dim=-1)
        alpha = torch.sigmoid(self.gate(combined))
        fused = alpha * part_emb + (1 - alpha) * cls_emb
        return F.normalize(fused, dim=-1)


class MaskedPartReconstruction(nn.Module):
    def __init__(self, part_dim=256, mask_ratio=0.30):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.decoder = nn.Sequential(
            nn.Linear(part_dim, part_dim * 2), nn.GELU(),
            nn.Linear(part_dim * 2, part_dim), nn.LayerNorm(part_dim),
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, part_dim) * 0.02)

    def forward(self, projected_patches, part_features, assignment):
        B, K, D = part_features.shape
        n_mask = max(1, int(K * self.mask_ratio))
        mask_idx = torch.randperm(K, device=part_features.device)[:n_mask]
        masked = part_features.clone()
        masked[:, mask_idx] = self.mask_token.expand(B, n_mask, -1)
        reconstructed = self.decoder(masked)
        return F.mse_loss(reconstructed[:, mask_idx], part_features[:, mask_idx].detach())


class AltitudePredictionHead(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, embedding, alt_target):
        pred = self.head(embedding).squeeze(-1)
        return F.mse_loss(pred, alt_target)


class SPDGeoDPEAMARModel(nn.Module):
    def __init__(self, num_classes, cfg=CFG):
        super().__init__()
        self.backbone  = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc = AltitudeAwarePartDiscovery(
            384, cfg.N_PARTS, cfg.PART_DIM, cfg.CLUSTER_TEMP, cfg.NUM_ALTITUDES
        )
        self.pool      = PartAwarePooling(cfg.PART_DIM, cfg.EMBED_DIM)
        self.fusion_gate = DynamicFusionGate(cfg.EMBED_DIM)
        self.bottleneck = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM),
                                        nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.classifier     = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.cls_proj       = nn.Sequential(nn.Linear(384, cfg.EMBED_DIM),
                                            nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.cls_classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.teacher_proj   = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.TEACHER_DIM),
                                            nn.LayerNorm(cfg.TEACHER_DIM))
        self.mask_recon = MaskedPartReconstruction(cfg.PART_DIM, cfg.MASK_RATIO)
        self.alt_pred   = AltitudePredictionHead(cfg.EMBED_DIM)

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPDGeo-DPEA-MAR: {total/1e6:.1f}M trainable parameters")

    def extract_embedding(self, x, alt_idx=None):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx)
        emb = self.pool(parts['part_features'], parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb)


# =============================================================================
# DATASET — weather test queries (pre-generated)
# =============================================================================
class WeatherQueryDataset(Dataset):
    def __init__(self, weather_root, weather_name, transform=None):
        self.transform = transform
        self.weather_name = weather_name
        self.samples = []

        weather_test_dir = os.path.join(weather_root, "sues200_weather_test")
        if not os.path.isdir(weather_test_dir):
            return

        test_locs = [f"{l:04d}" for l in CFG.TEST_LOCS]
        loc_to_idx = {l: i for i, l in enumerate(test_locs)}

        for class_dir_name in sorted(os.listdir(weather_test_dir)):
            class_path = os.path.join(weather_test_dir, class_dir_name)
            if not os.path.isdir(class_path):
                continue
            parts = class_dir_name.split("_")
            if len(parts) != 2:
                continue
            loc_id, alt = parts[0], parts[1]
            if loc_id not in loc_to_idx:
                continue
            label_idx = loc_to_idx[loc_id]
            for fname in sorted(os.listdir(class_path)):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                stem = os.path.splitext(fname)[0]
                if '-' not in stem:
                    continue
                w = stem.rsplit('-', 1)[-1]
                if w == weather_name:
                    self.samples.append((os.path.join(class_path, fname), label_idx, alt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, alt = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        alt_idx = CFG.ALT_TO_IDX.get(alt, 0)
        return {'drone': img, 'label': label, 'altitude': int(alt), 'alt_idx': alt_idx}


# =============================================================================
# HELPERS
# =============================================================================
def get_test_transform():
    sz = CFG.IMG_SIZE
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])


def build_satellite_gallery(model, device, transform):
    sat_dir = os.path.join(CFG.SUES_ROOT, CFG.SAT_DIR)
    test_locs = [f"{l:04d}" for l in CFG.TEST_LOCS]
    loc_to_idx = {l: i for i, l in enumerate(test_locs)}
    all_locs = [f"{l:04d}" for l in range(1, 201)]

    sat_imgs, sat_labels = [], []
    distractor_cnt = 0
    for loc in all_locs:
        sp = os.path.join(sat_dir, loc, "0.png")
        if not os.path.exists(sp):
            continue
        sat_imgs.append(transform(Image.open(sp).convert('RGB')))
        if loc in loc_to_idx:
            sat_labels.append(loc_to_idx[loc])
        else:
            sat_labels.append(-1000 - distractor_cnt)
            distractor_cnt += 1

    sat_feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sat_imgs), CFG.BATCH_SIZE):
            batch = torch.stack(sat_imgs[i:i + CFG.BATCH_SIZE]).to(device)
            with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
                feats = model.extract_embedding(batch, alt_idx=None)
            sat_feats.append(feats.cpu())

    sat_feats = torch.cat(sat_feats)
    sat_labels = torch.tensor(sat_labels)
    print(f"  Satellite gallery: {len(sat_feats)} images ({distractor_cnt} distractors)")
    return sat_feats, sat_labels


@torch.no_grad()
def evaluate_retrieval(model, query_ds, sat_feats, sat_labels, device):
    model.eval()
    loader = DataLoader(query_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True)

    drone_feats, drone_labels, drone_alts = [], [], []
    for batch in loader:
        imgs = batch['drone'].to(device)
        alt_idx = batch['alt_idx'].to(device)
        with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
            feats = model.extract_embedding(imgs, alt_idx=alt_idx)
        drone_feats.append(feats.cpu())
        drone_labels.append(batch['label'])
        drone_alts.append(batch['altitude'])

    drone_feats = torch.cat(drone_feats)
    drone_labels = torch.cat(drone_labels)
    drone_alts = torch.cat(drone_alts)

    sim = drone_feats @ sat_feats.T
    _, rank = sim.sort(1, descending=True)
    N = drone_feats.size(0)

    r1 = r5 = r10 = ap = 0
    for i in range(N):
        matches = torch.where(sat_labels[rank[i]] == drone_labels[i])[0]
        if len(matches) == 0:
            continue
        first = matches[0].item()
        if first < 1:  r1 += 1
        if first < 5:  r5 += 1
        if first < 10: r10 += 1
        ap += sum((j + 1) / (p.item() + 1) for j, p in enumerate(matches)) / len(matches)

    overall = {
        'R@1': r1 / N * 100, 'R@5': r5 / N * 100,
        'R@10': r10 / N * 100, 'mAP': ap / N * 100,
        'num_queries': N,
    }

    per_alt = {}
    for alt in sorted(drone_alts.unique().tolist()):
        mask = drone_alts == alt
        af = drone_feats[mask]; al = drone_labels[mask]
        s = af @ sat_feats.T; _, rk = s.sort(1, descending=True)
        n = af.size(0); a1 = a5 = a10 = aap = 0
        for i in range(n):
            m = torch.where(sat_labels[rk[i]] == al[i])[0]
            if len(m) == 0: continue
            f = m[0].item()
            if f < 1:  a1 += 1
            if f < 5:  a5 += 1
            if f < 10: a10 += 1
            aap += sum((j + 1) / (p.item() + 1) for j, p in enumerate(m)) / len(m)
        per_alt[int(alt)] = {'R@1': a1/n*100, 'R@5': a5/n*100,
                             'R@10': a10/n*100, 'mAP': aap/n*100, 'n': n}

    return overall, per_alt


# =============================================================================
# SATELLITE→DRONE EVALUATION (S2D)
# =============================================================================
def build_drone_gallery_s2d(model, weather_name, device, transform):
    """
    Builds drone gallery for Satellite→Drone (S2D) retrieval.
    Gallery = weather-augmented test drones (labeled 0..79) +
              clean train drones as distractors (labeled -1000...).
    Uses CFG.WEATHER_ROOT / CFG.SUES_ROOT directly.
    """
    test_locs  = [f"{l:04d}" for l in CFG.TEST_LOCS]
    loc_to_idx = {l: i for i, l in enumerate(test_locs)}
    train_locs = [f"{l:04d}" for l in range(1, CFG.NUM_CLASSES + 1)]

    drone_imgs, drone_lbls, drone_alt_idxs = [], [], []
    distractor_cnt = 0

    # 1. Weather-augmented TEST drone images (labeled)
    weather_test_dir = os.path.join(CFG.WEATHER_ROOT, "sues200_weather_test")
    if os.path.isdir(weather_test_dir):
        for class_dir_name in sorted(os.listdir(weather_test_dir)):
            class_path = os.path.join(weather_test_dir, class_dir_name)
            if not os.path.isdir(class_path):
                continue
            parts_split = class_dir_name.split("_")
            if len(parts_split) != 2:
                continue
            loc_id, alt = parts_split[0], parts_split[1]
            if loc_id not in loc_to_idx:
                continue
            label_idx = loc_to_idx[loc_id]
            for fname in sorted(os.listdir(class_path)):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                stem = os.path.splitext(fname)[0]
                if '-' not in stem:
                    continue
                w = stem.rsplit('-', 1)[-1]
                if w != weather_name:
                    continue
                try:
                    img = transform(Image.open(os.path.join(class_path, fname)).convert('RGB'))
                    drone_imgs.append(img)
                    drone_lbls.append(label_idx)
                    drone_alt_idxs.append(CFG.ALT_TO_IDX.get(alt, 0))
                except Exception:
                    pass

    # 2. Clean TRAIN drone images (distractors)
    drone_dir = os.path.join(CFG.SUES_ROOT, CFG.DRONE_DIR)
    for loc in train_locs:
        for alt in CFG.ALTITUDES:
            alt_dir = os.path.join(drone_dir, loc, alt)
            if not os.path.isdir(alt_dir):
                continue
            imgs_in_dir = sorted(f for f in os.listdir(alt_dir)
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png')))
            if not imgs_in_dir:
                continue
            try:
                img = transform(Image.open(os.path.join(alt_dir, imgs_in_dir[0])).convert('RGB'))
                drone_imgs.append(img)
                drone_lbls.append(-1000 - distractor_cnt)
                drone_alt_idxs.append(CFG.ALT_TO_IDX.get(alt, 0))
                distractor_cnt += 1
            except Exception:
                pass

    # Embed all drone images
    drone_feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(drone_imgs), CFG.BATCH_SIZE):
            bimgs = torch.stack(drone_imgs[i:i + CFG.BATCH_SIZE]).to(device)
            balts = torch.tensor(drone_alt_idxs[i:i + CFG.BATCH_SIZE], device=device)
            with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
                feats = model.extract_embedding(bimgs, alt_idx=balts)
            drone_feats.append(feats.cpu())

    drone_feats = torch.cat(drone_feats)
    drone_lbls  = torch.tensor(drone_lbls)
    test_cnt    = len(drone_feats) - distractor_cnt
    print(f"  Drone gallery S2D ({weather_name}): {len(drone_feats)} "
          f"(test={test_cnt}, distractors={distractor_cnt})")
    return drone_feats, drone_lbls


@torch.no_grad()
def evaluate_s2d(sat_feats, sat_labels, drone_feats, drone_labels):
    """
    Satellite→Drone retrieval.
    Queries  = test satellite images (sat_labels >= 0, 80 queries).
    Gallery  = weather-augmented test drones + clean distractor train drones.
    Each query has up to 4 matches (one per altitude of same location).
    """
    q_mask   = sat_labels >= 0
    q_feats  = sat_feats[q_mask]
    q_labels = sat_labels[q_mask]
    sim = q_feats @ drone_feats.T
    _, rank = sim.sort(1, descending=True)
    N = q_feats.size(0)
    r1 = r5 = r10 = ap = 0
    for i in range(N):
        matches = torch.where(drone_labels[rank[i]] == q_labels[i])[0]
        if len(matches) == 0:
            continue
        first = matches[0].item()
        if first < 1:  r1 += 1
        if first < 5:  r5 += 1
        if first < 10: r10 += 1
        ap += sum((j + 1) / (p.item() + 1) for j, p in enumerate(matches)) / len(matches)
    return {
        'R@1': r1 / N * 100, 'R@5': r5 / N * 100,
        'R@10': r10 / N * 100, 'mAP': ap / N * 100,
        'num_queries': N,
    }


def print_s2d_weather_table(all_s2d_results, normal_s2d):
    """Print Satellite→Drone per-weather results table."""
    print(f"\n{'='*85}")
    print(f"  WEATHER ROBUSTNESS — Satellite→Drone Retrieval (SUES-200 Test)")
    print(f"{'='*85}")
    print(f"  {'Weather':>12s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}  {'\u0394R@1':>7s}  {'\u0394mAP':>7s}  {'#Q':>5s}")
    print(f"  {'-'*78}")
    nr1 = normal_s2d['R@1']; nmap = normal_s2d['mAP']
    for w_name in CFG.WEATHER_NAMES:
        r = all_s2d_results[w_name]
        dr1 = r['R@1'] - nr1; dmap = r['mAP'] - nmap
        print(f"  {w_name:>12s}  {r['R@1']:6.2f}%  {r['R@5']:6.2f}%  {r['R@10']:6.2f}%  "
              f"{r['mAP']:6.2f}%  {dr1:+6.2f}%  {dmap:+6.2f}%  "
              f"{r['num_queries']:>5d}")
    print(f"  {'-'*78}")
    adv = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    avg_r1  = np.mean([all_s2d_results[w]['R@1']  for w in CFG.WEATHER_NAMES])
    avg_r5  = np.mean([all_s2d_results[w]['R@5']  for w in CFG.WEATHER_NAMES])
    avg_r10 = np.mean([all_s2d_results[w]['R@10'] for w in CFG.WEATHER_NAMES])
    avg_map = np.mean([all_s2d_results[w]['mAP']  for w in CFG.WEATHER_NAMES])
    avg_adv_r1  = np.mean([all_s2d_results[w]['R@1']  for w in adv])
    avg_adv_map = np.mean([all_s2d_results[w]['mAP']  for w in adv])
    print(f"  {'Avg(all)':>12s}  {avg_r1:6.2f}%  {avg_r5:6.2f}%  {avg_r10:6.2f}%  {avg_map:6.2f}%")
    print(f"  {'Avg(adverse)':>12s}  {avg_adv_r1:6.2f}%  {'':>7s}  {'':>7s}  {avg_adv_map:6.2f}%  "
          f"{avg_adv_r1-nr1:+5.2f}%  {avg_adv_map-nmap:+5.2f}%")
    print(f"{'='*85}")


def measure_model_complexity(model, device):
    sz = CFG.IMG_SIZE
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.eval()
    dummy = torch.randn(1, 3, sz, sz).to(device)
    gflops = None
    try:
        from thop import profile as thop_profile
        class _Wrap(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): return self.m.extract_embedding(x)
        macs, _ = thop_profile(_Wrap(model).to(device), inputs=(dummy,), verbose=False)
        gflops = macs / 1e9
    except Exception:
        pass
    model.eval()
    with torch.no_grad():
        for _ in range(10): model.extract_embedding(dummy)
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100): model.extract_embedding(dummy)
        if device.type == 'cuda': torch.cuda.synchronize()
        ms_query = (time.time() - t0) / 100 * 1000
    return {
        'total_params_M': total_params / 1e6,
        'trainable_params_M': trainable_params / 1e6,
        'gflops': gflops,
        'ms_per_query': ms_query,
        'img_size': sz,
    }


# =============================================================================
# PRINT / TABLE FUNCTIONS
# =============================================================================
def print_complexity(stats):
    print(f"\n  ┌─ Model Complexity ─────────────────────────────┐")
    print(f"  │  Total params:     {stats['total_params_M']:>8.2f}M                │")
    print(f"  │  Trainable params: {stats['trainable_params_M']:>8.2f}M                │")
    if stats['gflops'] is not None:
        print(f"  │  GFLOPs:           {stats['gflops']:>8.2f}                 │")
    else:
        print(f"  │  GFLOPs:           N/A (pip install thop)    │")
    print(f"  │  Inference:        {stats['ms_per_query']:>8.1f} ms/query         │")
    print(f"  │  Image size:       {stats['img_size']:>4d}×{stats['img_size']:<4d}               │")
    print(f"  └─────────────────────────────────────────────────┘")


def print_weather_table(all_results, normal_results):
    print(f"\n{'='*85}")
    print(f"  WEATHER ROBUSTNESS — Drone→Satellite Retrieval (SUES-200 Test)")
    print(f"{'='*85}")
    print(f"  {'Weather':>12s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}  {'ΔR@1':>7s}  {'ΔmAP':>7s}  {'#Q':>5s}")
    print(f"  {'-'*78}")
    nr1 = normal_results['R@1']; nmap = normal_results['mAP']
    for w_name in CFG.WEATHER_NAMES:
        r = all_results[w_name]['overall']
        dr1 = r['R@1'] - nr1; dmap = r['mAP'] - nmap
        sign_r1 = '+' if dr1 >= 0 else ''; sign_map = '+' if dmap >= 0 else ''
        print(f"  {w_name:>12s}  {r['R@1']:6.2f}%  {r['R@5']:6.2f}%  {r['R@10']:6.2f}%  "
              f"{r['mAP']:6.2f}%  {sign_r1}{dr1:5.2f}%  {sign_map}{dmap:5.2f}%  "
              f"{r['num_queries']:>5d}")
    print(f"  {'-'*78}")
    avg_r1  = np.mean([all_results[w]['overall']['R@1']  for w in CFG.WEATHER_NAMES])
    avg_r5  = np.mean([all_results[w]['overall']['R@5']  for w in CFG.WEATHER_NAMES])
    avg_r10 = np.mean([all_results[w]['overall']['R@10'] for w in CFG.WEATHER_NAMES])
    avg_map = np.mean([all_results[w]['overall']['mAP']  for w in CFG.WEATHER_NAMES])
    adv_weathers = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    avg_adv_r1  = np.mean([all_results[w]['overall']['R@1'] for w in adv_weathers])
    avg_adv_map = np.mean([all_results[w]['overall']['mAP'] for w in adv_weathers])
    print(f"  {'Avg(all)':>12s}  {avg_r1:6.2f}%  {avg_r5:6.2f}%  {avg_r10:6.2f}%  {avg_map:6.2f}%")
    print(f"  {'Avg(adverse)':>12s}  {avg_adv_r1:6.2f}%  {'':>7s}  {'':>7s}  {avg_adv_map:6.2f}%  "
          f"{avg_adv_r1-nr1:+5.2f}%  {avg_adv_map-nmap:+5.2f}%")
    print(f"{'='*85}")


def print_cross_table(all_results):
    print(f"\n{'='*75}")
    print(f"  R@1 (%) — Weather × Altitude")
    print(f"{'='*75}")
    header = f"  {'Weather':>12s}"
    for alt in CFG.ALTITUDES:
        header += f"  {alt+'m':>7s}"
    header += f"  {'Avg':>7s}"
    print(header)
    print(f"  {'-'*60}")
    for w in CFG.WEATHER_NAMES:
        row = f"  {w:>12s}"
        vals = []
        for alt in CFG.ALTITUDES:
            v = all_results[w]['per_alt'].get(int(alt), {}).get('R@1', 0)
            row += f"  {v:6.2f}%"; vals.append(v)
        row += f"  {np.mean(vals):6.2f}%"
        print(row)
    print(f"{'='*75}")


# =============================================================================
# KNOWN RESULTS FROM PREVIOUS EXPERIMENTS
# =============================================================================
EXP49_RESULTS = {   # zero-shot (EXP35 checkpoint, old eval script)
    'normal':    {'R@1': 95.94, 'mAP': 97.64},
    'fog':       {'R@1': 86.88, 'mAP': 92.68},
    'rain':      {'R@1': 43.75, 'mAP': 56.99},
    'snow':      {'R@1': 40.62, 'mAP': 54.48},
    'dark':      {'R@1': 66.88, 'mAP': 76.74},
    'light':     {'R@1': 83.44, 'mAP': 89.97},
    'fog_rain':  {'R@1': 42.19, 'mAP': 54.92},
    'fog_snow':  {'R@1': 22.19, 'mAP': 35.01},
    'rain_snow': {'R@1': 34.69, 'mAP': 46.85},
    'wind':      {'R@1': 81.25, 'mAP': 88.76},
}
EXP50_RESULTS = {   # finetune + pre-generated weather
    'normal':    {'R@1': 94.38, 'mAP': 96.78},
    'fog':       {'R@1': 90.94, 'mAP': 95.00},
    'rain':      {'R@1': 85.31, 'mAP': 91.07},
    'snow':      {'R@1': 88.12, 'mAP': 92.86},
    'dark':      {'R@1': 80.00, 'mAP': 86.58},
    'light':     {'R@1': 86.25, 'mAP': 91.42},
    'fog_rain':  {'R@1': 78.12, 'mAP': 86.61},
    'fog_snow':  {'R@1': 73.44, 'mAP': 82.36},
    'rain_snow': {'R@1': 84.69, 'mAP': 90.36},
    'wind':      {'R@1': 85.62, 'mAP': 90.78},
}
EXP51_RESULTS = {   # scratch + pre-generated weather
    'normal':    {'R@1': 90.94, 'mAP': 95.18},
    'fog':       {'R@1': 89.69, 'mAP': 94.37},
    'rain':      {'R@1': 86.56, 'mAP': 92.09},
    'snow':      {'R@1': 89.69, 'mAP': 93.66},
    'dark':      {'R@1': 80.31, 'mAP': 87.74},
    'light':     {'R@1': 86.88, 'mAP': 91.98},
    'fog_rain':  {'R@1': 81.25, 'mAP': 88.60},
    'fog_snow':  {'R@1': 75.00, 'mAP': 84.63},
    'rain_snow': {'R@1': 86.25, 'mAP': 91.27},
    'wind':      {'R@1': 88.12, 'mAP': 92.98},
}
# Fill these in after EXP52/53 complete (set to None to skip in comparison)
EXP52_RESULTS = {   # finetune + online aug (WeatherPrompt-style)
    'normal':    {'R@1': 95.00, 'mAP': 96.71},
    'fog':       {'R@1': 90.00, 'mAP': 94.30},
    'rain':      {'R@1': 89.69, 'mAP': 93.05},
    'snow':      {'R@1': 87.50, 'mAP': 92.05},
    'dark':      {'R@1': 78.75, 'mAP': 85.07},
    'light':     {'R@1': 84.69, 'mAP': 90.14},
    'fog_rain':  {'R@1': 81.88, 'mAP': 88.64},
    'fog_snow':  {'R@1': 69.38, 'mAP': 79.21},
    'rain_snow': {'R@1': 84.69, 'mAP': 90.06},
    'wind':      {'R@1': 88.75, 'mAP': 92.93},
}
EXP53_RESULTS = {   # scratch  + online aug (WeatherPrompt-style)
    'normal':    {'R@1': 92.81, 'mAP': 95.81},
    'fog':       {'R@1': 91.88, 'mAP': 94.94},
    'rain':      {'R@1': 90.31, 'mAP': 94.18},
    'snow':      {'R@1': 87.19, 'mAP': 92.19},
    'dark':      {'R@1': 82.50, 'mAP': 87.90},
    'light':     {'R@1': 86.25, 'mAP': 91.25},
    'fog_rain':  {'R@1': 85.62, 'mAP': 91.19},
    'fog_snow':  {'R@1': 79.38, 'mAP': 86.89},
    'rain_snow': {'R@1': 84.69, 'mAP': 90.76},
    'wind':      {'R@1': 87.50, 'mAP': 92.33},
}


def print_comparison(all_results):
    """Compare EXP54 against all known prior experiments."""
    # Build list of experiments that have real data
    exps = [
        ("EXP49", EXP49_RESULTS, "zero-shot"),
        ("EXP50", EXP50_RESULTS, "ft-pregen"),
        ("EXP51", EXP51_RESULTS, "scratch-pregen"),
    ]
    if EXP52_RESULTS is not None:
        exps.append(("EXP52", EXP52_RESULTS, "ft-online"))
    if EXP53_RESULTS is not None:
        exps.append(("EXP53", EXP53_RESULTS, "scratch-online"))

    exp_name_short = CFG.EXPERIMENT_NAME.split("_")[-1]   # e.g. "EXP35"
    col_width = max(8, len(exp_name_short) + 2)

    header  = f"\n{'='*120}\n"
    header += f"  COMPARISON (R@1%): "
    header += " vs ".join(f"{e[0]}({e[2]})" for e in exps)
    header += f" vs {CFG.EXPERIMENT_NAME}\n"
    header += f"{'='*120}"
    print(header)

    # Column headers
    row = f"  {'Weather':>12s}"
    for exp_id, _, _ in exps:
        row += f"  {exp_id:>8s}"
    row += f"  {exp_name_short:>8s}"
    print(row)
    print(f"  {'-'*80}")

    for w in CFG.WEATHER_NAMES:
        row = f"  {w:>12s}"
        for _, res, _ in exps:
            row += f"  {res[w]['R@1']:7.2f}%"
        e54_r1 = all_results[w]['overall']['R@1']
        row += f"  {e54_r1:7.2f}%"
        print(row)

    print(f"  {'-'*80}")
    adv = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    row = f"  {'Avg(adverse)':>12s}"
    for _, res, _ in exps:
        row += f"  {np.mean([res[w]['R@1'] for w in adv]):7.2f}%"
    row += f"  {np.mean([all_results[w]['overall']['R@1'] for w in adv]):7.2f}%"
    print(row)

    row = f"  {'Avg(all)':>12s}"
    for _, res, _ in exps:
        row += f"  {np.mean([res[w]['R@1'] for w in CFG.WEATHER_NAMES]):7.2f}%"
    row += f"  {np.mean([all_results[w]['overall']['R@1'] for w in CFG.WEATHER_NAMES]):7.2f}%"
    print(row)
    print(f"{'='*120}")


def generate_latex_table(all_results, normal_results):
    nr1 = normal_results['R@1']; nmap = normal_results['mAP']
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Zero-shot weather robustness evaluation ({CFG.EXPERIMENT_NAME}).}}",
        rf"\label{{tab:{CFG.EXPERIMENT_NAME.lower()}}}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l|cccc|cc}",
        r"\toprule",
        r"Weather & R@1 & R@5 & R@10 & mAP & $\Delta$R@1 & $\Delta$mAP \\",
        r"\midrule",
    ]
    for w in CFG.WEATHER_NAMES:
        r = all_results[w]['overall']
        dr1 = r['R@1'] - nr1; dmap = r['mAP'] - nmap
        label = w.replace('_', r'\_')
        if w == 'normal':
            lines.append(f"{label} & {r['R@1']:.2f} & {r['R@5']:.2f} & {r['R@10']:.2f} & "
                         f"{r['mAP']:.2f} & --- & --- \\\\")
        else:
            lines.append(f"{label} & {r['R@1']:.2f} & {r['R@5']:.2f} & {r['R@10']:.2f} & "
                         f"{r['mAP']:.2f} & {dr1:+.2f} & {dmap:+.2f} \\\\")
    adv = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    avg_r1  = np.mean([all_results[w]['overall']['R@1']  for w in adv])
    avg_r5  = np.mean([all_results[w]['overall']['R@5']  for w in adv])
    avg_r10 = np.mean([all_results[w]['overall']['R@10'] for w in adv])
    avg_map = np.mean([all_results[w]['overall']['mAP']  for w in adv])
    lines += [
        r"\midrule",
        f"Avg(adverse) & {avg_r1:.2f} & {avg_r5:.2f} & {avg_r10:.2f} & "
        f"{avg_map:.2f} & {avg_r1-nr1:+.2f} & {avg_map-nmap:+.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]
    return '\n'.join(lines)


# =============================================================================
# MAIN
# =============================================================================
def main():
    import random
    random.seed(CFG.SEED); np.random.seed(CFG.SEED)
    torch.manual_seed(CFG.SEED); torch.cuda.manual_seed_all(CFG.SEED)
    torch.backends.cudnn.benchmark = True

    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    start_time = datetime.datetime.now()

    print("=" * 75)
    print(f"  {CFG.EXPERIMENT_NAME}")
    print(f"  Zero-Shot Weather Robustness Evaluation (no training)")
    print(f"  Checkpoint: {CFG.CHECKPOINT}")
    print(f"  SUES-200:   {CFG.SUES_ROOT}")
    print(f"  Device:     {DEVICE}")
    print("=" * 75)

    # ---- 1. Load Model ----
    print("\n[1/3] Loading model …")
    if not os.path.exists(CFG.CHECKPOINT):
        print(f"  ERROR: Checkpoint not found at {CFG.CHECKPOINT}")
        return

    model = SPDGeoDPEAMARModel(CFG.NUM_CLASSES).to(DEVICE)
    ckpt = torch.load(CFG.CHECKPOINT, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys  ({len(missing)}): {missing[:5]} …")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]} …")

    ckpt_epoch = ckpt.get('epoch', '?')
    ckpt_r1    = ckpt.get('metrics', {}).get('R@1', ckpt.get('best_r1', '?'))
    print(f"  Loaded  epoch={ckpt_epoch}  pretrained R@1={ckpt_r1}")

    model.eval()

    complexity = measure_model_complexity(model, DEVICE)
    print_complexity(complexity)

    # ---- 2. Build Satellite Gallery ----
    print("\n[2/3] Building satellite gallery …")
    tf = get_test_transform()
    sat_feats, sat_labels = build_satellite_gallery(model, DEVICE, tf)

    # ---- 3. Evaluate All Weather Conditions ----
    print("\n[3/3] Evaluating all 10 weather conditions …")
    all_results = OrderedDict()
    for w_name in CFG.WEATHER_NAMES:
        query_ds = WeatherQueryDataset(CFG.WEATHER_ROOT, w_name, transform=tf)
        if len(query_ds) == 0:
            print(f"  [SKIP] {w_name}: no images found in {CFG.WEATHER_ROOT}")
            all_results[w_name] = {
                'overall': {'R@1': 0, 'R@5': 0, 'R@10': 0, 'mAP': 0, 'num_queries': 0},
                'per_alt': {}
            }
            continue
        overall, per_alt = evaluate_retrieval(model, query_ds, sat_feats, sat_labels, DEVICE)
        all_results[w_name] = {'overall': overall, 'per_alt': per_alt}
        print(f"  {w_name:>12s}  R@1={overall['R@1']:.2f}%  R@5={overall['R@5']:.2f}%  "
              f"R@10={overall['R@10']:.2f}%  mAP={overall['mAP']:.2f}%  ({len(query_ds)} queries)")

    # ---- Print Tables ----
    normal_results = all_results['normal']['overall']
    print_weather_table(all_results, normal_results)
    print_cross_table(all_results)
    print_comparison(all_results)

    # ---- S2D Evaluation (Satellite→Drone) ----
    print(f"\n{'='*75}")
    print(f"  [S2D] Satellite→Drone Evaluation …")
    all_s2d_results = OrderedDict()
    for w_name in CFG.WEATHER_NAMES:
        drone_feats_s2d, drone_lbls_s2d = build_drone_gallery_s2d(
            model, w_name, DEVICE, tf)
        s2d_res = evaluate_s2d(sat_feats, sat_labels, drone_feats_s2d, drone_lbls_s2d)
        all_s2d_results[w_name] = s2d_res
        print(f"    {w_name:>12s}  R@1={s2d_res['R@1']:.2f}%  R@5={s2d_res['R@5']:.2f}%  "
              f"R@10={s2d_res['R@10']:.2f}%  mAP={s2d_res['mAP']:.2f}%")
        del drone_feats_s2d, drone_lbls_s2d
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    normal_s2d = all_s2d_results['normal']
    print_s2d_weather_table(all_s2d_results, normal_s2d)

    # LaTeX
    latex = generate_latex_table(all_results, normal_results)
    print(f"\n{'='*75}\n  LaTeX Table:\n{'='*75}")
    print(latex)

    # ---- Save JSON ----
    end_time = datetime.datetime.now()
    adv_weathers = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    summary = {
        'experiment': CFG.EXPERIMENT_NAME,
        'checkpoint': CFG.CHECKPOINT,
        'ckpt_epoch': ckpt_epoch,
        'dataset': 'SUES-200',
        'direction': 'drone → satellite (D2S) + satellite → drone (S2D) — zero-shot inference',
        'timestamp': {
            'start': start_time.isoformat(),
            'end': end_time.isoformat(),
            'duration_s': (end_time - start_time).total_seconds(),
        },
        'model_complexity': complexity,
        'satellite_gallery_size': int(len(sat_feats)),
        'weather_results': {
            w: {
                'overall': all_results[w]['overall'],
                'per_altitude': {str(k): v for k, v in all_results[w]['per_alt'].items()},
            }
            for w in CFG.WEATHER_NAMES
        },
        'summary_metrics': {
            'normal_R@1': normal_results['R@1'],
            'normal_mAP': normal_results['mAP'],
            'avg_all_R@1':     float(np.mean([all_results[w]['overall']['R@1'] for w in CFG.WEATHER_NAMES])),
            'avg_all_mAP':     float(np.mean([all_results[w]['overall']['mAP'] for w in CFG.WEATHER_NAMES])),
            'avg_adverse_R@1': float(np.mean([all_results[w]['overall']['R@1'] for w in adv_weathers])),
            'avg_adverse_mAP': float(np.mean([all_results[w]['overall']['mAP'] for w in adv_weathers])),
            'vs_exp49': {w: float(all_results[w]['overall']['R@1'] - EXP49_RESULTS[w]['R@1'])
                         for w in CFG.WEATHER_NAMES},
            'vs_exp50': {w: float(all_results[w]['overall']['R@1'] - EXP50_RESULTS[w]['R@1'])
                         for w in CFG.WEATHER_NAMES},
        },
        'latex_table': latex,
        's2d_weather_results': {w: all_s2d_results[w] for w in CFG.WEATHER_NAMES},
        's2d_summary_metrics': {
            'normal_R@1': normal_s2d['R@1'],
            'normal_mAP': normal_s2d['mAP'],
            'avg_all_R@1':     float(np.mean([all_s2d_results[w]['R@1'] for w in CFG.WEATHER_NAMES])),
            'avg_all_mAP':     float(np.mean([all_s2d_results[w]['mAP'] for w in CFG.WEATHER_NAMES])),
            'avg_adverse_R@1': float(np.mean([all_s2d_results[w]['R@1'] for w in adv_weathers])),
            'avg_adverse_mAP': float(np.mean([all_s2d_results[w]['mAP'] for w in adv_weathers])),
        },
    }

    json_name = CFG.EXPERIMENT_NAME.lower() + ".json"
    json_path = os.path.join(CFG.OUTPUT_DIR, json_name)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {json_path}")

    # ---- Final Summary ----
    sm = summary['summary_metrics']
    print(f"\n{'='*75}")
    print(f"  {CFG.EXPERIMENT_NAME} COMPLETE  (zero-shot, no training)")
    print(f"{'='*75}")
    print(f"  Checkpoint:         epoch={ckpt_epoch}")
    print(f"  Normal:             R@1={sm['normal_R@1']:.2f}%  mAP={sm['normal_mAP']:.2f}%")
    print(f"  Avg (all weather):  R@1={sm['avg_all_R@1']:.2f}%  mAP={sm['avg_all_mAP']:.2f}%")
    print(f"  Avg (adverse):      R@1={sm['avg_adverse_R@1']:.2f}%  mAP={sm['avg_adverse_mAP']:.2f}%")
    e49_adv = np.mean([EXP49_RESULTS[w]['R@1'] for w in adv_weathers])
    print(f"\n  vs EXP49 (zero-shot, old eval)  avg adverse R@1: "
          f"{e49_adv:.2f}% → {sm['avg_adverse_R@1']:.2f}% "
          f"({sm['avg_adverse_R@1'] - e49_adv:+.2f}%)")
    e50_adv = np.mean([EXP50_RESULTS[w]['R@1'] for w in adv_weathers])
    print(f"  vs EXP50 (ft+pregen)            avg adverse R@1: "
          f"{e50_adv:.2f}% → {sm['avg_adverse_R@1']:.2f}% "
          f"({sm['avg_adverse_R@1'] - e50_adv:+.2f}%)")
    print(f"  Eval duration:      {(end_time - start_time).total_seconds():.1f}s")
    s2d_sm = summary['s2d_summary_metrics']
    print(f"\n  S2D — Satellite→Drone:")
    print(f"  Normal:             R@1={normal_s2d['R@1']:.2f}%  mAP={normal_s2d['mAP']:.2f}%")
    print(f"  Avg (all weather):  R@1={s2d_sm['avg_all_R@1']:.2f}%  mAP={s2d_sm['avg_all_mAP']:.2f}%")
    print(f"  Avg (adverse):      R@1={s2d_sm['avg_adverse_R@1']:.2f}%  mAP={s2d_sm['avg_adverse_mAP']:.2f}%")
    print(f"{'='*75}")


main()
# =============================================================================
# EXP54: Zero-Shot Weather Robustness Evaluation
# =============================================================================
# Copy to Kaggle notebook → Run All → Done
#
# Pure inference — NO training.
# Loads any SPDGeo-DPEA-MAR checkpoint and evaluates on all 10 weather
# conditions using pre-generated test images (same protocol as EXP49-53).
#
# Designed for:
#   1. EXP35 checkpoint (baseline / "our model, no weather training")
#   2. EXP52 best checkpoint (finetune + online aug)
#   3. EXP53 best checkpoint (scratch + online aug)
#   → Change CFG.CHECKPOINT and CFG.EXPERIMENT_NAME accordingly
#
# Kaggle Data Sources:
#   1. SUES-200 original:  satellite gallery + clean drone test
#   2. Weather synthetic:  weather-augmented drone TEST images (for eval)
#   3. Model checkpoint:   any SPDGeo-DPEA-MAR .pth file
# =============================================================================

import subprocess, sys
for _p in ["timm", "tqdm", "thop"]:
    try: __import__(_p)
    except ImportError: subprocess.run(f"pip install -q {_p}", shell=True, capture_output=True)

import os, json, time, datetime, gc
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================
class Config:
    # --- Which checkpoint to evaluate ---
    # Switch between EXP35, EXP52, EXP53 best by changing these two fields
    EXPERIMENT_NAME = "EXP54_ZeroShot_EXP35"
    CHECKPOINT      = "/kaggle/input/models/minh2duy/exp35-spdgeo-dpea-m-unfreeze-blocks-4-6/pytorch/default/1/exp35_dpea_ga_best.pth"

    # For EXP52 best:
    # EXPERIMENT_NAME = "EXP54_ZeroShot_EXP52"
    # CHECKPOINT      = "/kaggle/working/exp52_weather_finetune_online_best.pth"

    # For EXP53 best:
    # EXPERIMENT_NAME = "EXP54_ZeroShot_EXP53"
    # CHECKPOINT      = "/kaggle/working/exp53_weather_scratch_online_best.pth"

    # --- Paths ---
    SUES_ROOT       = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
    WEATHER_ROOT    = "/kaggle/input/datasets/minh2duy/sues200-weather/weather_synthetic"
    OUTPUT_DIR      = "/kaggle/working"

    # --- Dataset structure ---
    DRONE_DIR       = "drone-view"
    SAT_DIR         = "satellite-view"
    ALTITUDES       = ["150", "200", "250", "300"]
    ALT_TO_IDX      = {"150": 0, "200": 1, "250": 2, "300": 3}
    NUM_ALTITUDES   = 4
    TEST_LOCS       = list(range(121, 201))   # 80 test locations
    NUM_CLASSES     = 120

    # --- Model architecture (must match checkpoint) ---
    IMG_SIZE        = 336
    N_PARTS         = 8
    PART_DIM        = 256
    EMBED_DIM       = 512
    TEACHER_DIM     = 768
    CLUSTER_TEMP    = 0.07
    UNFREEZE_BLOCKS = 6
    MASK_RATIO      = 0.30

    # --- Weather conditions ---
    WEATHER_NAMES   = ["normal", "fog", "rain", "snow", "dark", "light",
                       "fog_rain", "fog_snow", "rain_snow", "wind"]

    # --- Inference ---
    USE_AMP         = True
    BATCH_SIZE      = 64
    NUM_WORKERS     = 2
    SEED            = 42


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# MODEL DEFINITION (identical to EXP35 / EXP52 / EXP53)
# =============================================================================
class DINOv2Backbone(nn.Module):
    def __init__(self, unfreeze_blocks=4):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        self.feature_dim = 384; self.patch_size = 14
        for p in self.model.parameters(): p.requires_grad = False
        for blk in self.model.blocks[-unfreeze_blocks:]:
            for p in blk.parameters(): p.requires_grad = True
        for p in self.model.norm.parameters(): p.requires_grad = True
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  DINOv2 ViT-S/14: {frozen/1e6:.1f}M frozen, {trainable/1e6:.1f}M trainable")

    def forward(self, x):
        features = self.model.forward_features(x)
        patch_tokens = features['x_norm_patchtokens']
        cls_token = features['x_norm_clstoken']
        H = x.shape[2] // self.patch_size; W = x.shape[3] // self.patch_size
        return patch_tokens, cls_token, (H, W)


class DeepAltitudeFiLM(nn.Module):
    def __init__(self, num_altitudes=4, feat_dim=256):
        super().__init__()
        self.num_altitudes = num_altitudes
        self.gamma = nn.Parameter(torch.ones(num_altitudes, feat_dim))
        self.beta  = nn.Parameter(torch.zeros(num_altitudes, feat_dim))

    def forward(self, feat, alt_idx=None):
        if alt_idx is None:
            gamma = self.gamma.mean(dim=0, keepdim=True)
            beta  = self.beta.mean(dim=0, keepdim=True)
            return feat * gamma.unsqueeze(0) + beta.unsqueeze(0)
        else:
            gamma = self.gamma[alt_idx]
            beta  = self.beta[alt_idx]
            return feat * gamma.unsqueeze(1) + beta.unsqueeze(1)


class AltitudeAwarePartDiscovery(nn.Module):
    def __init__(self, feat_dim=384, n_parts=8, part_dim=256, temperature=0.07,
                 num_altitudes=4):
        super().__init__()
        self.n_parts = n_parts; self.temperature = temperature
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, part_dim), nn.LayerNorm(part_dim), nn.GELU()
        )
        self.altitude_film = DeepAltitudeFiLM(num_altitudes, part_dim)
        self.prototypes = nn.Parameter(torch.randn(n_parts, part_dim) * 0.02)
        self.refine = nn.Sequential(
            nn.LayerNorm(part_dim), nn.Linear(part_dim, part_dim * 2),
            nn.GELU(), nn.Linear(part_dim * 2, part_dim)
        )
        self.salience_head = nn.Sequential(
            nn.Linear(part_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, patch_features, spatial_hw, alt_idx=None):
        B, N, _ = patch_features.shape; H, W = spatial_hw
        feat = self.feat_proj(patch_features)
        feat = self.altitude_film(feat, alt_idx)
        feat_norm = F.normalize(feat, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum('bnd,kd->bnk', feat_norm, proto_norm) / self.temperature
        assign = F.softmax(sim, dim=-1)
        assign_t = assign.transpose(1, 2)
        mass = assign_t.sum(-1, keepdim=True).clamp(min=1e-6)
        part_feat = torch.bmm(assign_t, feat) / mass
        part_feat = part_feat + self.refine(part_feat)
        device = feat.device
        gy = torch.arange(H, device=device).float() / max(H - 1, 1)
        gx = torch.arange(W, device=device).float() / max(W - 1, 1)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        part_pos = torch.bmm(assign_t, coords.unsqueeze(0).expand(B, -1, -1)) / mass
        salience = self.salience_head(part_feat).squeeze(-1)
        return {'part_features': part_feat, 'part_positions': part_pos,
                'assignment': assign, 'salience': salience,
                'projected_patches': feat}


class PartAwarePooling(nn.Module):
    def __init__(self, part_dim=256, embed_dim=512):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(part_dim, part_dim // 2), nn.Tanh(),
                                  nn.Linear(part_dim // 2, 1))
        self.proj = nn.Sequential(nn.Linear(part_dim * 3, embed_dim), nn.LayerNorm(embed_dim),
                                  nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, part_features, salience=None):
        B, K, D = part_features.shape
        aw = self.attn(part_features)
        if salience is not None: aw = aw + salience.unsqueeze(-1).log().clamp(-10)
        aw = F.softmax(aw, dim=1)
        attn_pool = (aw * part_features).sum(1)
        mean_pool = part_features.mean(1); max_pool = part_features.max(1)[0]
        combined = torch.cat([attn_pool, mean_pool, max_pool], dim=-1)
        return F.normalize(self.proj(combined), dim=-1)


class DynamicFusionGate(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
        )
        nn.init.constant_(self.gate[-1].bias, 0.85)

    def forward(self, part_emb, cls_emb):
        combined = torch.cat([part_emb, cls_emb], dim=-1)
        alpha = torch.sigmoid(self.gate(combined))
        fused = alpha * part_emb + (1 - alpha) * cls_emb
        return F.normalize(fused, dim=-1)


class MaskedPartReconstruction(nn.Module):
    def __init__(self, part_dim=256, mask_ratio=0.30):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.decoder = nn.Sequential(
            nn.Linear(part_dim, part_dim * 2), nn.GELU(),
            nn.Linear(part_dim * 2, part_dim), nn.LayerNorm(part_dim),
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, part_dim) * 0.02)

    def forward(self, projected_patches, part_features, assignment):
        B, K, D = part_features.shape
        n_mask = max(1, int(K * self.mask_ratio))
        mask_idx = torch.randperm(K, device=part_features.device)[:n_mask]
        masked = part_features.clone()
        masked[:, mask_idx] = self.mask_token.expand(B, n_mask, -1)
        reconstructed = self.decoder(masked)
        return F.mse_loss(reconstructed[:, mask_idx], part_features[:, mask_idx].detach())


class AltitudePredictionHead(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, embedding, alt_target):
        pred = self.head(embedding).squeeze(-1)
        return F.mse_loss(pred, alt_target)


class SPDGeoDPEAMARModel(nn.Module):
    def __init__(self, num_classes, cfg=CFG):
        super().__init__()
        self.backbone  = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc = AltitudeAwarePartDiscovery(
            384, cfg.N_PARTS, cfg.PART_DIM, cfg.CLUSTER_TEMP, cfg.NUM_ALTITUDES
        )
        self.pool      = PartAwarePooling(cfg.PART_DIM, cfg.EMBED_DIM)
        self.fusion_gate = DynamicFusionGate(cfg.EMBED_DIM)
        self.bottleneck = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM),
                                        nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.classifier     = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.cls_proj       = nn.Sequential(nn.Linear(384, cfg.EMBED_DIM),
                                            nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.cls_classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.teacher_proj   = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.TEACHER_DIM),
                                            nn.LayerNorm(cfg.TEACHER_DIM))
        self.mask_recon = MaskedPartReconstruction(cfg.PART_DIM, cfg.MASK_RATIO)
        self.alt_pred   = AltitudePredictionHead(cfg.EMBED_DIM)

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPDGeo-DPEA-MAR: {total/1e6:.1f}M trainable parameters")

    def extract_embedding(self, x, alt_idx=None):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx)
        emb = self.pool(parts['part_features'], parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb)


# =============================================================================
# DATASET — weather test queries (pre-generated)
# =============================================================================
class WeatherQueryDataset(Dataset):
    def __init__(self, weather_root, weather_name, transform=None):
        self.transform = transform
        self.weather_name = weather_name
        self.samples = []

        weather_test_dir = os.path.join(weather_root, "sues200_weather_test")
        if not os.path.isdir(weather_test_dir):
            return

        test_locs = [f"{l:04d}" for l in CFG.TEST_LOCS]
        loc_to_idx = {l: i for i, l in enumerate(test_locs)}

        for class_dir_name in sorted(os.listdir(weather_test_dir)):
            class_path = os.path.join(weather_test_dir, class_dir_name)
            if not os.path.isdir(class_path):
                continue
            parts = class_dir_name.split("_")
            if len(parts) != 2:
                continue
            loc_id, alt = parts[0], parts[1]
            if loc_id not in loc_to_idx:
                continue
            label_idx = loc_to_idx[loc_id]
            for fname in sorted(os.listdir(class_path)):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                stem = os.path.splitext(fname)[0]
                if '-' not in stem:
                    continue
                w = stem.rsplit('-', 1)[-1]
                if w == weather_name:
                    self.samples.append((os.path.join(class_path, fname), label_idx, alt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, alt = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        alt_idx = CFG.ALT_TO_IDX.get(alt, 0)
        return {'drone': img, 'label': label, 'altitude': int(alt), 'alt_idx': alt_idx}


# =============================================================================
# HELPERS
# =============================================================================
def get_test_transform():
    sz = CFG.IMG_SIZE
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])


def build_satellite_gallery(model, device, transform):
    sat_dir = os.path.join(CFG.SUES_ROOT, CFG.SAT_DIR)
    test_locs = [f"{l:04d}" for l in CFG.TEST_LOCS]
    loc_to_idx = {l: i for i, l in enumerate(test_locs)}
    all_locs = [f"{l:04d}" for l in range(1, 201)]

    sat_imgs, sat_labels = [], []
    distractor_cnt = 0
    for loc in all_locs:
        sp = os.path.join(sat_dir, loc, "0.png")
        if not os.path.exists(sp):
            continue
        sat_imgs.append(transform(Image.open(sp).convert('RGB')))
        if loc in loc_to_idx:
            sat_labels.append(loc_to_idx[loc])
        else:
            sat_labels.append(-1000 - distractor_cnt)
            distractor_cnt += 1

    sat_feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sat_imgs), CFG.BATCH_SIZE):
            batch = torch.stack(sat_imgs[i:i + CFG.BATCH_SIZE]).to(device)
            with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
                feats = model.extract_embedding(batch, alt_idx=None)
            sat_feats.append(feats.cpu())

    sat_feats = torch.cat(sat_feats)
    sat_labels = torch.tensor(sat_labels)
    print(f"  Satellite gallery: {len(sat_feats)} images ({distractor_cnt} distractors)")
    return sat_feats, sat_labels


@torch.no_grad()
def evaluate_retrieval(model, query_ds, sat_feats, sat_labels, device):
    model.eval()
    loader = DataLoader(query_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True)

    drone_feats, drone_labels, drone_alts = [], [], []
    for batch in loader:
        imgs = batch['drone'].to(device)
        alt_idx = batch['alt_idx'].to(device)
        with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
            feats = model.extract_embedding(imgs, alt_idx=alt_idx)
        drone_feats.append(feats.cpu())
        drone_labels.append(batch['label'])
        drone_alts.append(batch['altitude'])

    drone_feats = torch.cat(drone_feats)
    drone_labels = torch.cat(drone_labels)
    drone_alts = torch.cat(drone_alts)

    sim = drone_feats @ sat_feats.T
    _, rank = sim.sort(1, descending=True)
    N = drone_feats.size(0)

    r1 = r5 = r10 = ap = 0
    for i in range(N):
        matches = torch.where(sat_labels[rank[i]] == drone_labels[i])[0]
        if len(matches) == 0:
            continue
        first = matches[0].item()
        if first < 1:  r1 += 1
        if first < 5:  r5 += 1
        if first < 10: r10 += 1
        ap += sum((j + 1) / (p.item() + 1) for j, p in enumerate(matches)) / len(matches)

    overall = {
        'R@1': r1 / N * 100, 'R@5': r5 / N * 100,
        'R@10': r10 / N * 100, 'mAP': ap / N * 100,
        'num_queries': N,
    }

    per_alt = {}
    for alt in sorted(drone_alts.unique().tolist()):
        mask = drone_alts == alt
        af = drone_feats[mask]; al = drone_labels[mask]
        s = af @ sat_feats.T; _, rk = s.sort(1, descending=True)
        n = af.size(0); a1 = a5 = a10 = aap = 0
        for i in range(n):
            m = torch.where(sat_labels[rk[i]] == al[i])[0]
            if len(m) == 0: continue
            f = m[0].item()
            if f < 1:  a1 += 1
            if f < 5:  a5 += 1
            if f < 10: a10 += 1
            aap += sum((j + 1) / (p.item() + 1) for j, p in enumerate(m)) / len(m)
        per_alt[int(alt)] = {'R@1': a1/n*100, 'R@5': a5/n*100,
                             'R@10': a10/n*100, 'mAP': aap/n*100, 'n': n}

    return overall, per_alt


# =============================================================================
# SATELLITE→DRONE EVALUATION (S2D)
# =============================================================================
def build_drone_gallery_s2d(model, weather_name, device, transform):
    """
    Builds drone gallery for Satellite→Drone (S2D) retrieval.
    Gallery = weather-augmented test drones (labeled 0..79) +
              clean train drones as distractors (labeled -1000...).
    Uses CFG.WEATHER_ROOT / CFG.SUES_ROOT directly.
    """
    test_locs  = [f"{l:04d}" for l in CFG.TEST_LOCS]
    loc_to_idx = {l: i for i, l in enumerate(test_locs)}
    train_locs = [f"{l:04d}" for l in range(1, CFG.NUM_CLASSES + 1)]

    drone_imgs, drone_lbls, drone_alt_idxs = [], [], []
    distractor_cnt = 0

    # 1. Weather-augmented TEST drone images (labeled)
    weather_test_dir = os.path.join(CFG.WEATHER_ROOT, "sues200_weather_test")
    if os.path.isdir(weather_test_dir):
        for class_dir_name in sorted(os.listdir(weather_test_dir)):
            class_path = os.path.join(weather_test_dir, class_dir_name)
            if not os.path.isdir(class_path):
                continue
            parts_split = class_dir_name.split("_")
            if len(parts_split) != 2:
                continue
            loc_id, alt = parts_split[0], parts_split[1]
            if loc_id not in loc_to_idx:
                continue
            label_idx = loc_to_idx[loc_id]
            for fname in sorted(os.listdir(class_path)):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                stem = os.path.splitext(fname)[0]
                if '-' not in stem:
                    continue
                w = stem.rsplit('-', 1)[-1]
                if w != weather_name:
                    continue
                try:
                    img = transform(Image.open(os.path.join(class_path, fname)).convert('RGB'))
                    drone_imgs.append(img)
                    drone_lbls.append(label_idx)
                    drone_alt_idxs.append(CFG.ALT_TO_IDX.get(alt, 0))
                except Exception:
                    pass

    # 2. Clean TRAIN drone images (distractors)
    drone_dir = os.path.join(CFG.SUES_ROOT, CFG.DRONE_DIR)
    for loc in train_locs:
        for alt in CFG.ALTITUDES:
            alt_dir = os.path.join(drone_dir, loc, alt)
            if not os.path.isdir(alt_dir):
                continue
            imgs_in_dir = sorted(f for f in os.listdir(alt_dir)
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png')))
            if not imgs_in_dir:
                continue
            try:
                img = transform(Image.open(os.path.join(alt_dir, imgs_in_dir[0])).convert('RGB'))
                drone_imgs.append(img)
                drone_lbls.append(-1000 - distractor_cnt)
                drone_alt_idxs.append(CFG.ALT_TO_IDX.get(alt, 0))
                distractor_cnt += 1
            except Exception:
                pass

    # Embed all drone images
    drone_feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(drone_imgs), CFG.BATCH_SIZE):
            bimgs = torch.stack(drone_imgs[i:i + CFG.BATCH_SIZE]).to(device)
            balts = torch.tensor(drone_alt_idxs[i:i + CFG.BATCH_SIZE], device=device)
            with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
                feats = model.extract_embedding(bimgs, alt_idx=balts)
            drone_feats.append(feats.cpu())

    drone_feats = torch.cat(drone_feats)
    drone_lbls  = torch.tensor(drone_lbls)
    test_cnt    = len(drone_feats) - distractor_cnt
    print(f"  Drone gallery S2D ({weather_name}): {len(drone_feats)} "
          f"(test={test_cnt}, distractors={distractor_cnt})")
    return drone_feats, drone_lbls


@torch.no_grad()
def evaluate_s2d(sat_feats, sat_labels, drone_feats, drone_labels):
    """
    Satellite→Drone retrieval.
    Queries  = test satellite images (sat_labels >= 0, 80 queries).
    Gallery  = weather-augmented test drones + clean distractor train drones.
    Each query has up to 4 matches (one per altitude of same location).
    """
    q_mask   = sat_labels >= 0
    q_feats  = sat_feats[q_mask]
    q_labels = sat_labels[q_mask]
    sim = q_feats @ drone_feats.T
    _, rank = sim.sort(1, descending=True)
    N = q_feats.size(0)
    r1 = r5 = r10 = ap = 0
    for i in range(N):
        matches = torch.where(drone_labels[rank[i]] == q_labels[i])[0]
        if len(matches) == 0:
            continue
        first = matches[0].item()
        if first < 1:  r1 += 1
        if first < 5:  r5 += 1
        if first < 10: r10 += 1
        ap += sum((j + 1) / (p.item() + 1) for j, p in enumerate(matches)) / len(matches)
    return {
        'R@1': r1 / N * 100, 'R@5': r5 / N * 100,
        'R@10': r10 / N * 100, 'mAP': ap / N * 100,
        'num_queries': N,
    }


def print_s2d_weather_table(all_s2d_results, normal_s2d):
    """Print Satellite→Drone per-weather results table."""
    print(f"\n{'='*85}")
    print(f"  WEATHER ROBUSTNESS — Satellite→Drone Retrieval (SUES-200 Test)")
    print(f"{'='*85}")
    print(f"  {'Weather':>12s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}  {'\u0394R@1':>7s}  {'\u0394mAP':>7s}  {'#Q':>5s}")
    print(f"  {'-'*78}")
    nr1 = normal_s2d['R@1']; nmap = normal_s2d['mAP']
    for w_name in CFG.WEATHER_NAMES:
        r = all_s2d_results[w_name]
        dr1 = r['R@1'] - nr1; dmap = r['mAP'] - nmap
        print(f"  {w_name:>12s}  {r['R@1']:6.2f}%  {r['R@5']:6.2f}%  {r['R@10']:6.2f}%  "
              f"{r['mAP']:6.2f}%  {dr1:+6.2f}%  {dmap:+6.2f}%  "
              f"{r['num_queries']:>5d}")
    print(f"  {'-'*78}")
    adv = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    avg_r1  = np.mean([all_s2d_results[w]['R@1']  for w in CFG.WEATHER_NAMES])
    avg_r5  = np.mean([all_s2d_results[w]['R@5']  for w in CFG.WEATHER_NAMES])
    avg_r10 = np.mean([all_s2d_results[w]['R@10'] for w in CFG.WEATHER_NAMES])
    avg_map = np.mean([all_s2d_results[w]['mAP']  for w in CFG.WEATHER_NAMES])
    avg_adv_r1  = np.mean([all_s2d_results[w]['R@1']  for w in adv])
    avg_adv_map = np.mean([all_s2d_results[w]['mAP']  for w in adv])
    print(f"  {'Avg(all)':>12s}  {avg_r1:6.2f}%  {avg_r5:6.2f}%  {avg_r10:6.2f}%  {avg_map:6.2f}%")
    print(f"  {'Avg(adverse)':>12s}  {avg_adv_r1:6.2f}%  {'':>7s}  {'':>7s}  {avg_adv_map:6.2f}%  "
          f"{avg_adv_r1-nr1:+5.2f}%  {avg_adv_map-nmap:+5.2f}%")
    print(f"{'='*85}")


def measure_model_complexity(model, device):
    sz = CFG.IMG_SIZE
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.eval()
    dummy = torch.randn(1, 3, sz, sz).to(device)
    gflops = None
    try:
        from thop import profile as thop_profile
        class _Wrap(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): return self.m.extract_embedding(x)
        macs, _ = thop_profile(_Wrap(model).to(device), inputs=(dummy,), verbose=False)
        gflops = macs / 1e9
    except Exception:
        pass
    model.eval()
    with torch.no_grad():
        for _ in range(10): model.extract_embedding(dummy)
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100): model.extract_embedding(dummy)
        if device.type == 'cuda': torch.cuda.synchronize()
        ms_query = (time.time() - t0) / 100 * 1000
    return {
        'total_params_M': total_params / 1e6,
        'trainable_params_M': trainable_params / 1e6,
        'gflops': gflops,
        'ms_per_query': ms_query,
        'img_size': sz,
    }


# =============================================================================
# PRINT / TABLE FUNCTIONS
# =============================================================================
def print_complexity(stats):
    print(f"\n  ┌─ Model Complexity ─────────────────────────────┐")
    print(f"  │  Total params:     {stats['total_params_M']:>8.2f}M                │")
    print(f"  │  Trainable params: {stats['trainable_params_M']:>8.2f}M                │")
    if stats['gflops'] is not None:
        print(f"  │  GFLOPs:           {stats['gflops']:>8.2f}                 │")
    else:
        print(f"  │  GFLOPs:           N/A (pip install thop)    │")
    print(f"  │  Inference:        {stats['ms_per_query']:>8.1f} ms/query         │")
    print(f"  │  Image size:       {stats['img_size']:>4d}×{stats['img_size']:<4d}               │")
    print(f"  └─────────────────────────────────────────────────┘")


def print_weather_table(all_results, normal_results):
    print(f"\n{'='*85}")
    print(f"  WEATHER ROBUSTNESS — Drone→Satellite Retrieval (SUES-200 Test)")
    print(f"{'='*85}")
    print(f"  {'Weather':>12s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}  {'ΔR@1':>7s}  {'ΔmAP':>7s}  {'#Q':>5s}")
    print(f"  {'-'*78}")
    nr1 = normal_results['R@1']; nmap = normal_results['mAP']
    for w_name in CFG.WEATHER_NAMES:
        r = all_results[w_name]['overall']
        dr1 = r['R@1'] - nr1; dmap = r['mAP'] - nmap
        sign_r1 = '+' if dr1 >= 0 else ''; sign_map = '+' if dmap >= 0 else ''
        print(f"  {w_name:>12s}  {r['R@1']:6.2f}%  {r['R@5']:6.2f}%  {r['R@10']:6.2f}%  "
              f"{r['mAP']:6.2f}%  {sign_r1}{dr1:5.2f}%  {sign_map}{dmap:5.2f}%  "
              f"{r['num_queries']:>5d}")
    print(f"  {'-'*78}")
    avg_r1  = np.mean([all_results[w]['overall']['R@1']  for w in CFG.WEATHER_NAMES])
    avg_r5  = np.mean([all_results[w]['overall']['R@5']  for w in CFG.WEATHER_NAMES])
    avg_r10 = np.mean([all_results[w]['overall']['R@10'] for w in CFG.WEATHER_NAMES])
    avg_map = np.mean([all_results[w]['overall']['mAP']  for w in CFG.WEATHER_NAMES])
    adv_weathers = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    avg_adv_r1  = np.mean([all_results[w]['overall']['R@1'] for w in adv_weathers])
    avg_adv_map = np.mean([all_results[w]['overall']['mAP'] for w in adv_weathers])
    print(f"  {'Avg(all)':>12s}  {avg_r1:6.2f}%  {avg_r5:6.2f}%  {avg_r10:6.2f}%  {avg_map:6.2f}%")
    print(f"  {'Avg(adverse)':>12s}  {avg_adv_r1:6.2f}%  {'':>7s}  {'':>7s}  {avg_adv_map:6.2f}%  "
          f"{avg_adv_r1-nr1:+5.2f}%  {avg_adv_map-nmap:+5.2f}%")
    print(f"{'='*85}")


def print_cross_table(all_results):
    print(f"\n{'='*75}")
    print(f"  R@1 (%) — Weather × Altitude")
    print(f"{'='*75}")
    header = f"  {'Weather':>12s}"
    for alt in CFG.ALTITUDES:
        header += f"  {alt+'m':>7s}"
    header += f"  {'Avg':>7s}"
    print(header)
    print(f"  {'-'*60}")
    for w in CFG.WEATHER_NAMES:
        row = f"  {w:>12s}"
        vals = []
        for alt in CFG.ALTITUDES:
            v = all_results[w]['per_alt'].get(int(alt), {}).get('R@1', 0)
            row += f"  {v:6.2f}%"; vals.append(v)
        row += f"  {np.mean(vals):6.2f}%"
        print(row)
    print(f"{'='*75}")


# =============================================================================
# KNOWN RESULTS FROM PREVIOUS EXPERIMENTS
# =============================================================================
EXP49_RESULTS = {   # zero-shot (EXP35 checkpoint, old eval script)
    'normal':    {'R@1': 95.94, 'mAP': 97.64},
    'fog':       {'R@1': 86.88, 'mAP': 92.68},
    'rain':      {'R@1': 43.75, 'mAP': 56.99},
    'snow':      {'R@1': 40.62, 'mAP': 54.48},
    'dark':      {'R@1': 66.88, 'mAP': 76.74},
    'light':     {'R@1': 83.44, 'mAP': 89.97},
    'fog_rain':  {'R@1': 42.19, 'mAP': 54.92},
    'fog_snow':  {'R@1': 22.19, 'mAP': 35.01},
    'rain_snow': {'R@1': 34.69, 'mAP': 46.85},
    'wind':      {'R@1': 81.25, 'mAP': 88.76},
}
EXP50_RESULTS = {   # finetune + pre-generated weather
    'normal':    {'R@1': 94.38, 'mAP': 96.78},
    'fog':       {'R@1': 90.94, 'mAP': 95.00},
    'rain':      {'R@1': 85.31, 'mAP': 91.07},
    'snow':      {'R@1': 88.12, 'mAP': 92.86},
    'dark':      {'R@1': 80.00, 'mAP': 86.58},
    'light':     {'R@1': 86.25, 'mAP': 91.42},
    'fog_rain':  {'R@1': 78.12, 'mAP': 86.61},
    'fog_snow':  {'R@1': 73.44, 'mAP': 82.36},
    'rain_snow': {'R@1': 84.69, 'mAP': 90.36},
    'wind':      {'R@1': 85.62, 'mAP': 90.78},
}
EXP51_RESULTS = {   # scratch + pre-generated weather
    'normal':    {'R@1': 90.94, 'mAP': 95.18},
    'fog':       {'R@1': 89.69, 'mAP': 94.37},
    'rain':      {'R@1': 86.56, 'mAP': 92.09},
    'snow':      {'R@1': 89.69, 'mAP': 93.66},
    'dark':      {'R@1': 80.31, 'mAP': 87.74},
    'light':     {'R@1': 86.88, 'mAP': 91.98},
    'fog_rain':  {'R@1': 81.25, 'mAP': 88.60},
    'fog_snow':  {'R@1': 75.00, 'mAP': 84.63},
    'rain_snow': {'R@1': 86.25, 'mAP': 91.27},
    'wind':      {'R@1': 88.12, 'mAP': 92.98},
}
# Fill these in after EXP52/53 complete (set to None to skip in comparison)
EXP52_RESULTS = {   # finetune + online aug (WeatherPrompt-style)
    'normal':    {'R@1': 95.00, 'mAP': 96.71},
    'fog':       {'R@1': 90.00, 'mAP': 94.30},
    'rain':      {'R@1': 89.69, 'mAP': 93.05},
    'snow':      {'R@1': 87.50, 'mAP': 92.05},
    'dark':      {'R@1': 78.75, 'mAP': 85.07},
    'light':     {'R@1': 84.69, 'mAP': 90.14},
    'fog_rain':  {'R@1': 81.88, 'mAP': 88.64},
    'fog_snow':  {'R@1': 69.38, 'mAP': 79.21},
    'rain_snow': {'R@1': 84.69, 'mAP': 90.06},
    'wind':      {'R@1': 88.75, 'mAP': 92.93},
}
EXP53_RESULTS = {   # scratch  + online aug (WeatherPrompt-style)
    'normal':    {'R@1': 92.81, 'mAP': 95.81},
    'fog':       {'R@1': 91.88, 'mAP': 94.94},
    'rain':      {'R@1': 90.31, 'mAP': 94.18},
    'snow':      {'R@1': 87.19, 'mAP': 92.19},
    'dark':      {'R@1': 82.50, 'mAP': 87.90},
    'light':     {'R@1': 86.25, 'mAP': 91.25},
    'fog_rain':  {'R@1': 85.62, 'mAP': 91.19},
    'fog_snow':  {'R@1': 79.38, 'mAP': 86.89},
    'rain_snow': {'R@1': 84.69, 'mAP': 90.76},
    'wind':      {'R@1': 87.50, 'mAP': 92.33},
}


def print_comparison(all_results):
    """Compare EXP54 against all known prior experiments."""
    # Build list of experiments that have real data
    exps = [
        ("EXP49", EXP49_RESULTS, "zero-shot"),
        ("EXP50", EXP50_RESULTS, "ft-pregen"),
        ("EXP51", EXP51_RESULTS, "scratch-pregen"),
    ]
    if EXP52_RESULTS is not None:
        exps.append(("EXP52", EXP52_RESULTS, "ft-online"))
    if EXP53_RESULTS is not None:
        exps.append(("EXP53", EXP53_RESULTS, "scratch-online"))

    exp_name_short = CFG.EXPERIMENT_NAME.split("_")[-1]   # e.g. "EXP35"
    col_width = max(8, len(exp_name_short) + 2)

    header  = f"\n{'='*120}\n"
    header += f"  COMPARISON (R@1%): "
    header += " vs ".join(f"{e[0]}({e[2]})" for e in exps)
    header += f" vs {CFG.EXPERIMENT_NAME}\n"
    header += f"{'='*120}"
    print(header)

    # Column headers
    row = f"  {'Weather':>12s}"
    for exp_id, _, _ in exps:
        row += f"  {exp_id:>8s}"
    row += f"  {exp_name_short:>8s}"
    print(row)
    print(f"  {'-'*80}")

    for w in CFG.WEATHER_NAMES:
        row = f"  {w:>12s}"
        for _, res, _ in exps:
            row += f"  {res[w]['R@1']:7.2f}%"
        e54_r1 = all_results[w]['overall']['R@1']
        row += f"  {e54_r1:7.2f}%"
        print(row)

    print(f"  {'-'*80}")
    adv = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    row = f"  {'Avg(adverse)':>12s}"
    for _, res, _ in exps:
        row += f"  {np.mean([res[w]['R@1'] for w in adv]):7.2f}%"
    row += f"  {np.mean([all_results[w]['overall']['R@1'] for w in adv]):7.2f}%"
    print(row)

    row = f"  {'Avg(all)':>12s}"
    for _, res, _ in exps:
        row += f"  {np.mean([res[w]['R@1'] for w in CFG.WEATHER_NAMES]):7.2f}%"
    row += f"  {np.mean([all_results[w]['overall']['R@1'] for w in CFG.WEATHER_NAMES]):7.2f}%"
    print(row)
    print(f"{'='*120}")


def generate_latex_table(all_results, normal_results):
    nr1 = normal_results['R@1']; nmap = normal_results['mAP']
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Zero-shot weather robustness evaluation ({CFG.EXPERIMENT_NAME}).}}",
        rf"\label{{tab:{CFG.EXPERIMENT_NAME.lower()}}}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l|cccc|cc}",
        r"\toprule",
        r"Weather & R@1 & R@5 & R@10 & mAP & $\Delta$R@1 & $\Delta$mAP \\",
        r"\midrule",
    ]
    for w in CFG.WEATHER_NAMES:
        r = all_results[w]['overall']
        dr1 = r['R@1'] - nr1; dmap = r['mAP'] - nmap
        label = w.replace('_', r'\_')
        if w == 'normal':
            lines.append(f"{label} & {r['R@1']:.2f} & {r['R@5']:.2f} & {r['R@10']:.2f} & "
                         f"{r['mAP']:.2f} & --- & --- \\\\")
        else:
            lines.append(f"{label} & {r['R@1']:.2f} & {r['R@5']:.2f} & {r['R@10']:.2f} & "
                         f"{r['mAP']:.2f} & {dr1:+.2f} & {dmap:+.2f} \\\\")
    adv = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    avg_r1  = np.mean([all_results[w]['overall']['R@1']  for w in adv])
    avg_r5  = np.mean([all_results[w]['overall']['R@5']  for w in adv])
    avg_r10 = np.mean([all_results[w]['overall']['R@10'] for w in adv])
    avg_map = np.mean([all_results[w]['overall']['mAP']  for w in adv])
    lines += [
        r"\midrule",
        f"Avg(adverse) & {avg_r1:.2f} & {avg_r5:.2f} & {avg_r10:.2f} & "
        f"{avg_map:.2f} & {avg_r1-nr1:+.2f} & {avg_map-nmap:+.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]
    return '\n'.join(lines)


# =============================================================================
# MAIN
# =============================================================================
def main():
    import random
    random.seed(CFG.SEED); np.random.seed(CFG.SEED)
    torch.manual_seed(CFG.SEED); torch.cuda.manual_seed_all(CFG.SEED)
    torch.backends.cudnn.benchmark = True

    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    start_time = datetime.datetime.now()

    print("=" * 75)
    print(f"  {CFG.EXPERIMENT_NAME}")
    print(f"  Zero-Shot Weather Robustness Evaluation (no training)")
    print(f"  Checkpoint: {CFG.CHECKPOINT}")
    print(f"  SUES-200:   {CFG.SUES_ROOT}")
    print(f"  Device:     {DEVICE}")
    print("=" * 75)

    # ---- 1. Load Model ----
    print("\n[1/3] Loading model …")
    if not os.path.exists(CFG.CHECKPOINT):
        print(f"  ERROR: Checkpoint not found at {CFG.CHECKPOINT}")
        return

    model = SPDGeoDPEAMARModel(CFG.NUM_CLASSES).to(DEVICE)
    ckpt = torch.load(CFG.CHECKPOINT, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys  ({len(missing)}): {missing[:5]} …")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]} …")

    ckpt_epoch = ckpt.get('epoch', '?')
    ckpt_r1    = ckpt.get('metrics', {}).get('R@1', ckpt.get('best_r1', '?'))
    print(f"  Loaded  epoch={ckpt_epoch}  pretrained R@1={ckpt_r1}")

    model.eval()

    complexity = measure_model_complexity(model, DEVICE)
    print_complexity(complexity)

    # ---- 2. Build Satellite Gallery ----
    print("\n[2/3] Building satellite gallery …")
    tf = get_test_transform()
    sat_feats, sat_labels = build_satellite_gallery(model, DEVICE, tf)

    # ---- 3. Evaluate All Weather Conditions ----
    print("\n[3/3] Evaluating all 10 weather conditions …")
    all_results = OrderedDict()
    for w_name in CFG.WEATHER_NAMES:
        query_ds = WeatherQueryDataset(CFG.WEATHER_ROOT, w_name, transform=tf)
        if len(query_ds) == 0:
            print(f"  [SKIP] {w_name}: no images found in {CFG.WEATHER_ROOT}")
            all_results[w_name] = {
                'overall': {'R@1': 0, 'R@5': 0, 'R@10': 0, 'mAP': 0, 'num_queries': 0},
                'per_alt': {}
            }
            continue
        overall, per_alt = evaluate_retrieval(model, query_ds, sat_feats, sat_labels, DEVICE)
        all_results[w_name] = {'overall': overall, 'per_alt': per_alt}
        print(f"  {w_name:>12s}  R@1={overall['R@1']:.2f}%  R@5={overall['R@5']:.2f}%  "
              f"R@10={overall['R@10']:.2f}%  mAP={overall['mAP']:.2f}%  ({len(query_ds)} queries)")

    # ---- Print Tables ----
    normal_results = all_results['normal']['overall']
    print_weather_table(all_results, normal_results)
    print_cross_table(all_results)
    print_comparison(all_results)

    # ---- S2D Evaluation (Satellite→Drone) ----
    print(f"\n{'='*75}")
    print(f"  [S2D] Satellite→Drone Evaluation …")
    all_s2d_results = OrderedDict()
    for w_name in CFG.WEATHER_NAMES:
        drone_feats_s2d, drone_lbls_s2d = build_drone_gallery_s2d(
            model, w_name, DEVICE, tf)
        s2d_res = evaluate_s2d(sat_feats, sat_labels, drone_feats_s2d, drone_lbls_s2d)
        all_s2d_results[w_name] = s2d_res
        print(f"    {w_name:>12s}  R@1={s2d_res['R@1']:.2f}%  R@5={s2d_res['R@5']:.2f}%  "
              f"R@10={s2d_res['R@10']:.2f}%  mAP={s2d_res['mAP']:.2f}%")
        del drone_feats_s2d, drone_lbls_s2d
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    normal_s2d = all_s2d_results['normal']
    print_s2d_weather_table(all_s2d_results, normal_s2d)

    # LaTeX
    latex = generate_latex_table(all_results, normal_results)
    print(f"\n{'='*75}\n  LaTeX Table:\n{'='*75}")
    print(latex)

    # ---- Save JSON ----
    end_time = datetime.datetime.now()
    adv_weathers = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    summary = {
        'experiment': CFG.EXPERIMENT_NAME,
        'checkpoint': CFG.CHECKPOINT,
        'ckpt_epoch': ckpt_epoch,
        'dataset': 'SUES-200',
        'direction': 'drone → satellite (D2S) + satellite → drone (S2D) — zero-shot inference',
        'timestamp': {
            'start': start_time.isoformat(),
            'end': end_time.isoformat(),
            'duration_s': (end_time - start_time).total_seconds(),
        },
        'model_complexity': complexity,
        'satellite_gallery_size': int(len(sat_feats)),
        'weather_results': {
            w: {
                'overall': all_results[w]['overall'],
                'per_altitude': {str(k): v for k, v in all_results[w]['per_alt'].items()},
            }
            for w in CFG.WEATHER_NAMES
        },
        'summary_metrics': {
            'normal_R@1': normal_results['R@1'],
            'normal_mAP': normal_results['mAP'],
            'avg_all_R@1':     float(np.mean([all_results[w]['overall']['R@1'] for w in CFG.WEATHER_NAMES])),
            'avg_all_mAP':     float(np.mean([all_results[w]['overall']['mAP'] for w in CFG.WEATHER_NAMES])),
            'avg_adverse_R@1': float(np.mean([all_results[w]['overall']['R@1'] for w in adv_weathers])),
            'avg_adverse_mAP': float(np.mean([all_results[w]['overall']['mAP'] for w in adv_weathers])),
            'vs_exp49': {w: float(all_results[w]['overall']['R@1'] - EXP49_RESULTS[w]['R@1'])
                         for w in CFG.WEATHER_NAMES},
            'vs_exp50': {w: float(all_results[w]['overall']['R@1'] - EXP50_RESULTS[w]['R@1'])
                         for w in CFG.WEATHER_NAMES},
        },
        'latex_table': latex,
        's2d_weather_results': {w: all_s2d_results[w] for w in CFG.WEATHER_NAMES},
        's2d_summary_metrics': {
            'normal_R@1': normal_s2d['R@1'],
            'normal_mAP': normal_s2d['mAP'],
            'avg_all_R@1':     float(np.mean([all_s2d_results[w]['R@1'] for w in CFG.WEATHER_NAMES])),
            'avg_all_mAP':     float(np.mean([all_s2d_results[w]['mAP'] for w in CFG.WEATHER_NAMES])),
            'avg_adverse_R@1': float(np.mean([all_s2d_results[w]['R@1'] for w in adv_weathers])),
            'avg_adverse_mAP': float(np.mean([all_s2d_results[w]['mAP'] for w in adv_weathers])),
        },
    }

    json_name = CFG.EXPERIMENT_NAME.lower() + ".json"
    json_path = os.path.join(CFG.OUTPUT_DIR, json_name)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {json_path}")

    # ---- Final Summary ----
    sm = summary['summary_metrics']
    print(f"\n{'='*75}")
    print(f"  {CFG.EXPERIMENT_NAME} COMPLETE  (zero-shot, no training)")
    print(f"{'='*75}")
    print(f"  Checkpoint:         epoch={ckpt_epoch}")
    print(f"  Normal:             R@1={sm['normal_R@1']:.2f}%  mAP={sm['normal_mAP']:.2f}%")
    print(f"  Avg (all weather):  R@1={sm['avg_all_R@1']:.2f}%  mAP={sm['avg_all_mAP']:.2f}%")
    print(f"  Avg (adverse):      R@1={sm['avg_adverse_R@1']:.2f}%  mAP={sm['avg_adverse_mAP']:.2f}%")
    e49_adv = np.mean([EXP49_RESULTS[w]['R@1'] for w in adv_weathers])
    print(f"\n  vs EXP49 (zero-shot, old eval)  avg adverse R@1: "
          f"{e49_adv:.2f}% → {sm['avg_adverse_R@1']:.2f}% "
          f"({sm['avg_adverse_R@1'] - e49_adv:+.2f}%)")
    e50_adv = np.mean([EXP50_RESULTS[w]['R@1'] for w in adv_weathers])
    print(f"  vs EXP50 (ft+pregen)            avg adverse R@1: "
          f"{e50_adv:.2f}% → {sm['avg_adverse_R@1']:.2f}% "
          f"({sm['avg_adverse_R@1'] - e50_adv:+.2f}%)")
    print(f"  Eval duration:      {(end_time - start_time).total_seconds():.1f}s")
    s2d_sm = summary['s2d_summary_metrics']
    print(f"\n  S2D — Satellite→Drone:")
    print(f"  Normal:             R@1={normal_s2d['R@1']:.2f}%  mAP={normal_s2d['mAP']:.2f}%")
    print(f"  Avg (all weather):  R@1={s2d_sm['avg_all_R@1']:.2f}%  mAP={s2d_sm['avg_all_mAP']:.2f}%")
    print(f"  Avg (adverse):      R@1={s2d_sm['avg_adverse_R@1']:.2f}%  mAP={s2d_sm['avg_adverse_mAP']:.2f}%")
    print(f"{'='*75}")


main()
# =============================================================================
# EXP54: Zero-Shot Weather Robustness Evaluation
# =============================================================================
# Copy to Kaggle notebook → Run All → Done
#
# Pure inference — NO training.
# Loads any SPDGeo-DPEA-MAR checkpoint and evaluates on all 10 weather
# conditions using pre-generated test images (same protocol as EXP49-53).
#
# Designed for:
#   1. EXP35 checkpoint (baseline / "our model, no weather training")
#   2. EXP52 best checkpoint (finetune + online aug)
#   3. EXP53 best checkpoint (scratch + online aug)
#   → Change CFG.CHECKPOINT and CFG.EXPERIMENT_NAME accordingly
#
# Kaggle Data Sources:
#   1. SUES-200 original:  satellite gallery + clean drone test
#   2. Weather synthetic:  weather-augmented drone TEST images (for eval)
#   3. Model checkpoint:   any SPDGeo-DPEA-MAR .pth file
# =============================================================================

import subprocess, sys
for _p in ["timm", "tqdm", "thop"]:
    try: __import__(_p)
    except ImportError: subprocess.run(f"pip install -q {_p}", shell=True, capture_output=True)

import os, json, time, datetime, gc
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================
class Config:
    # --- Which checkpoint to evaluate ---
    # Switch between EXP35, EXP52, EXP53 best by changing these two fields
    EXPERIMENT_NAME = "EXP54_ZeroShot_EXP35"
    CHECKPOINT      = "/kaggle/input/models/minh2duy/exp35-spdgeo-dpea-m-unfreeze-blocks-4-6/pytorch/default/1/exp35_dpea_ga_best.pth"

    # For EXP52 best:
    # EXPERIMENT_NAME = "EXP54_ZeroShot_EXP52"
    # CHECKPOINT      = "/kaggle/working/exp52_weather_finetune_online_best.pth"

    # For EXP53 best:
    # EXPERIMENT_NAME = "EXP54_ZeroShot_EXP53"
    # CHECKPOINT      = "/kaggle/working/exp53_weather_scratch_online_best.pth"

    # --- Paths ---
    SUES_ROOT       = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
    WEATHER_ROOT    = "/kaggle/input/datasets/minh2duy/sues200-weather/weather_synthetic"
    OUTPUT_DIR      = "/kaggle/working"

    # --- Dataset structure ---
    DRONE_DIR       = "drone-view"
    SAT_DIR         = "satellite-view"
    ALTITUDES       = ["150", "200", "250", "300"]
    ALT_TO_IDX      = {"150": 0, "200": 1, "250": 2, "300": 3}
    NUM_ALTITUDES   = 4
    TEST_LOCS       = list(range(121, 201))   # 80 test locations
    NUM_CLASSES     = 120

    # --- Model architecture (must match checkpoint) ---
    IMG_SIZE        = 336
    N_PARTS         = 8
    PART_DIM        = 256
    EMBED_DIM       = 512
    TEACHER_DIM     = 768
    CLUSTER_TEMP    = 0.07
    UNFREEZE_BLOCKS = 6
    MASK_RATIO      = 0.30

    # --- Weather conditions ---
    WEATHER_NAMES   = ["normal", "fog", "rain", "snow", "dark", "light",
                       "fog_rain", "fog_snow", "rain_snow", "wind"]

    # --- Inference ---
    USE_AMP         = True
    BATCH_SIZE      = 64
    NUM_WORKERS     = 2
    SEED            = 42


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# MODEL DEFINITION (identical to EXP35 / EXP52 / EXP53)
# =============================================================================
class DINOv2Backbone(nn.Module):
    def __init__(self, unfreeze_blocks=4):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        self.feature_dim = 384; self.patch_size = 14
        for p in self.model.parameters(): p.requires_grad = False
        for blk in self.model.blocks[-unfreeze_blocks:]:
            for p in blk.parameters(): p.requires_grad = True
        for p in self.model.norm.parameters(): p.requires_grad = True
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  DINOv2 ViT-S/14: {frozen/1e6:.1f}M frozen, {trainable/1e6:.1f}M trainable")

    def forward(self, x):
        features = self.model.forward_features(x)
        patch_tokens = features['x_norm_patchtokens']
        cls_token = features['x_norm_clstoken']
        H = x.shape[2] // self.patch_size; W = x.shape[3] // self.patch_size
        return patch_tokens, cls_token, (H, W)


class DeepAltitudeFiLM(nn.Module):
    def __init__(self, num_altitudes=4, feat_dim=256):
        super().__init__()
        self.num_altitudes = num_altitudes
        self.gamma = nn.Parameter(torch.ones(num_altitudes, feat_dim))
        self.beta  = nn.Parameter(torch.zeros(num_altitudes, feat_dim))

    def forward(self, feat, alt_idx=None):
        if alt_idx is None:
            gamma = self.gamma.mean(dim=0, keepdim=True)
            beta  = self.beta.mean(dim=0, keepdim=True)
            return feat * gamma.unsqueeze(0) + beta.unsqueeze(0)
        else:
            gamma = self.gamma[alt_idx]
            beta  = self.beta[alt_idx]
            return feat * gamma.unsqueeze(1) + beta.unsqueeze(1)


class AltitudeAwarePartDiscovery(nn.Module):
    def __init__(self, feat_dim=384, n_parts=8, part_dim=256, temperature=0.07,
                 num_altitudes=4):
        super().__init__()
        self.n_parts = n_parts; self.temperature = temperature
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, part_dim), nn.LayerNorm(part_dim), nn.GELU()
        )
        self.altitude_film = DeepAltitudeFiLM(num_altitudes, part_dim)
        self.prototypes = nn.Parameter(torch.randn(n_parts, part_dim) * 0.02)
        self.refine = nn.Sequential(
            nn.LayerNorm(part_dim), nn.Linear(part_dim, part_dim * 2),
            nn.GELU(), nn.Linear(part_dim * 2, part_dim)
        )
        self.salience_head = nn.Sequential(
            nn.Linear(part_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, patch_features, spatial_hw, alt_idx=None):
        B, N, _ = patch_features.shape; H, W = spatial_hw
        feat = self.feat_proj(patch_features)
        feat = self.altitude_film(feat, alt_idx)
        feat_norm = F.normalize(feat, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum('bnd,kd->bnk', feat_norm, proto_norm) / self.temperature
        assign = F.softmax(sim, dim=-1)
        assign_t = assign.transpose(1, 2)
        mass = assign_t.sum(-1, keepdim=True).clamp(min=1e-6)
        part_feat = torch.bmm(assign_t, feat) / mass
        part_feat = part_feat + self.refine(part_feat)
        device = feat.device
        gy = torch.arange(H, device=device).float() / max(H - 1, 1)
        gx = torch.arange(W, device=device).float() / max(W - 1, 1)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        part_pos = torch.bmm(assign_t, coords.unsqueeze(0).expand(B, -1, -1)) / mass
        salience = self.salience_head(part_feat).squeeze(-1)
        return {'part_features': part_feat, 'part_positions': part_pos,
                'assignment': assign, 'salience': salience,
                'projected_patches': feat}


class PartAwarePooling(nn.Module):
    def __init__(self, part_dim=256, embed_dim=512):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(part_dim, part_dim // 2), nn.Tanh(),
                                  nn.Linear(part_dim // 2, 1))
        self.proj = nn.Sequential(nn.Linear(part_dim * 3, embed_dim), nn.LayerNorm(embed_dim),
                                  nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, part_features, salience=None):
        B, K, D = part_features.shape
        aw = self.attn(part_features)
        if salience is not None: aw = aw + salience.unsqueeze(-1).log().clamp(-10)
        aw = F.softmax(aw, dim=1)
        attn_pool = (aw * part_features).sum(1)
        mean_pool = part_features.mean(1); max_pool = part_features.max(1)[0]
        combined = torch.cat([attn_pool, mean_pool, max_pool], dim=-1)
        return F.normalize(self.proj(combined), dim=-1)


class DynamicFusionGate(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
        )
        nn.init.constant_(self.gate[-1].bias, 0.85)

    def forward(self, part_emb, cls_emb):
        combined = torch.cat([part_emb, cls_emb], dim=-1)
        alpha = torch.sigmoid(self.gate(combined))
        fused = alpha * part_emb + (1 - alpha) * cls_emb
        return F.normalize(fused, dim=-1)


class MaskedPartReconstruction(nn.Module):
    def __init__(self, part_dim=256, mask_ratio=0.30):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.decoder = nn.Sequential(
            nn.Linear(part_dim, part_dim * 2), nn.GELU(),
            nn.Linear(part_dim * 2, part_dim), nn.LayerNorm(part_dim),
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, part_dim) * 0.02)

    def forward(self, projected_patches, part_features, assignment):
        B, K, D = part_features.shape
        n_mask = max(1, int(K * self.mask_ratio))
        mask_idx = torch.randperm(K, device=part_features.device)[:n_mask]
        masked = part_features.clone()
        masked[:, mask_idx] = self.mask_token.expand(B, n_mask, -1)
        reconstructed = self.decoder(masked)
        return F.mse_loss(reconstructed[:, mask_idx], part_features[:, mask_idx].detach())


class AltitudePredictionHead(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, embedding, alt_target):
        pred = self.head(embedding).squeeze(-1)
        return F.mse_loss(pred, alt_target)


class SPDGeoDPEAMARModel(nn.Module):
    def __init__(self, num_classes, cfg=CFG):
        super().__init__()
        self.backbone  = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc = AltitudeAwarePartDiscovery(
            384, cfg.N_PARTS, cfg.PART_DIM, cfg.CLUSTER_TEMP, cfg.NUM_ALTITUDES
        )
        self.pool      = PartAwarePooling(cfg.PART_DIM, cfg.EMBED_DIM)
        self.fusion_gate = DynamicFusionGate(cfg.EMBED_DIM)
        self.bottleneck = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM),
                                        nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.classifier     = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.cls_proj       = nn.Sequential(nn.Linear(384, cfg.EMBED_DIM),
                                            nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.cls_classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.teacher_proj   = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.TEACHER_DIM),
                                            nn.LayerNorm(cfg.TEACHER_DIM))
        self.mask_recon = MaskedPartReconstruction(cfg.PART_DIM, cfg.MASK_RATIO)
        self.alt_pred   = AltitudePredictionHead(cfg.EMBED_DIM)

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPDGeo-DPEA-MAR: {total/1e6:.1f}M trainable parameters")

    def extract_embedding(self, x, alt_idx=None):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx)
        emb = self.pool(parts['part_features'], parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb)


# =============================================================================
# DATASET — weather test queries (pre-generated)
# =============================================================================
class WeatherQueryDataset(Dataset):
    def __init__(self, weather_root, weather_name, transform=None):
        self.transform = transform
        self.weather_name = weather_name
        self.samples = []

        weather_test_dir = os.path.join(weather_root, "sues200_weather_test")
        if not os.path.isdir(weather_test_dir):
            return

        test_locs = [f"{l:04d}" for l in CFG.TEST_LOCS]
        loc_to_idx = {l: i for i, l in enumerate(test_locs)}

        for class_dir_name in sorted(os.listdir(weather_test_dir)):
            class_path = os.path.join(weather_test_dir, class_dir_name)
            if not os.path.isdir(class_path):
                continue
            parts = class_dir_name.split("_")
            if len(parts) != 2:
                continue
            loc_id, alt = parts[0], parts[1]
            if loc_id not in loc_to_idx:
                continue
            label_idx = loc_to_idx[loc_id]
            for fname in sorted(os.listdir(class_path)):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                stem = os.path.splitext(fname)[0]
                if '-' not in stem:
                    continue
                w = stem.rsplit('-', 1)[-1]
                if w == weather_name:
                    self.samples.append((os.path.join(class_path, fname), label_idx, alt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, alt = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        alt_idx = CFG.ALT_TO_IDX.get(alt, 0)
        return {'drone': img, 'label': label, 'altitude': int(alt), 'alt_idx': alt_idx}


# =============================================================================
# HELPERS
# =============================================================================
def get_test_transform():
    sz = CFG.IMG_SIZE
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])


def build_satellite_gallery(model, device, transform):
    sat_dir = os.path.join(CFG.SUES_ROOT, CFG.SAT_DIR)
    test_locs = [f"{l:04d}" for l in CFG.TEST_LOCS]
    loc_to_idx = {l: i for i, l in enumerate(test_locs)}
    all_locs = [f"{l:04d}" for l in range(1, 201)]

    sat_imgs, sat_labels = [], []
    distractor_cnt = 0
    for loc in all_locs:
        sp = os.path.join(sat_dir, loc, "0.png")
        if not os.path.exists(sp):
            continue
        sat_imgs.append(transform(Image.open(sp).convert('RGB')))
        if loc in loc_to_idx:
            sat_labels.append(loc_to_idx[loc])
        else:
            sat_labels.append(-1000 - distractor_cnt)
            distractor_cnt += 1

    sat_feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sat_imgs), CFG.BATCH_SIZE):
            batch = torch.stack(sat_imgs[i:i + CFG.BATCH_SIZE]).to(device)
            with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
                feats = model.extract_embedding(batch, alt_idx=None)
            sat_feats.append(feats.cpu())

    sat_feats = torch.cat(sat_feats)
    sat_labels = torch.tensor(sat_labels)
    print(f"  Satellite gallery: {len(sat_feats)} images ({distractor_cnt} distractors)")
    return sat_feats, sat_labels


@torch.no_grad()
def evaluate_retrieval(model, query_ds, sat_feats, sat_labels, device):
    model.eval()
    loader = DataLoader(query_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True)

    drone_feats, drone_labels, drone_alts = [], [], []
    for batch in loader:
        imgs = batch['drone'].to(device)
        alt_idx = batch['alt_idx'].to(device)
        with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
            feats = model.extract_embedding(imgs, alt_idx=alt_idx)
        drone_feats.append(feats.cpu())
        drone_labels.append(batch['label'])
        drone_alts.append(batch['altitude'])

    drone_feats = torch.cat(drone_feats)
    drone_labels = torch.cat(drone_labels)
    drone_alts = torch.cat(drone_alts)

    sim = drone_feats @ sat_feats.T
    _, rank = sim.sort(1, descending=True)
    N = drone_feats.size(0)

    r1 = r5 = r10 = ap = 0
    for i in range(N):
        matches = torch.where(sat_labels[rank[i]] == drone_labels[i])[0]
        if len(matches) == 0:
            continue
        first = matches[0].item()
        if first < 1:  r1 += 1
        if first < 5:  r5 += 1
        if first < 10: r10 += 1
        ap += sum((j + 1) / (p.item() + 1) for j, p in enumerate(matches)) / len(matches)

    overall = {
        'R@1': r1 / N * 100, 'R@5': r5 / N * 100,
        'R@10': r10 / N * 100, 'mAP': ap / N * 100,
        'num_queries': N,
    }

    per_alt = {}
    for alt in sorted(drone_alts.unique().tolist()):
        mask = drone_alts == alt
        af = drone_feats[mask]; al = drone_labels[mask]
        s = af @ sat_feats.T; _, rk = s.sort(1, descending=True)
        n = af.size(0); a1 = a5 = a10 = aap = 0
        for i in range(n):
            m = torch.where(sat_labels[rk[i]] == al[i])[0]
            if len(m) == 0: continue
            f = m[0].item()
            if f < 1:  a1 += 1
            if f < 5:  a5 += 1
            if f < 10: a10 += 1
            aap += sum((j + 1) / (p.item() + 1) for j, p in enumerate(m)) / len(m)
        per_alt[int(alt)] = {'R@1': a1/n*100, 'R@5': a5/n*100,
                             'R@10': a10/n*100, 'mAP': aap/n*100, 'n': n}

    return overall, per_alt


# =============================================================================
# SATELLITE→DRONE EVALUATION (S2D)
# =============================================================================
def build_drone_gallery_s2d(model, weather_name, device, transform):
    """
    Builds drone gallery for Satellite→Drone (S2D) retrieval.
    Gallery = weather-augmented test drones (labeled 0..79) +
              clean train drones as distractors (labeled -1000...).
    Uses CFG.WEATHER_ROOT / CFG.SUES_ROOT directly.
    """
    test_locs  = [f"{l:04d}" for l in CFG.TEST_LOCS]
    loc_to_idx = {l: i for i, l in enumerate(test_locs)}
    train_locs = [f"{l:04d}" for l in range(1, CFG.NUM_CLASSES + 1)]

    drone_imgs, drone_lbls, drone_alt_idxs = [], [], []
    distractor_cnt = 0

    # 1. Weather-augmented TEST drone images (labeled)
    weather_test_dir = os.path.join(CFG.WEATHER_ROOT, "sues200_weather_test")
    if os.path.isdir(weather_test_dir):
        for class_dir_name in sorted(os.listdir(weather_test_dir)):
            class_path = os.path.join(weather_test_dir, class_dir_name)
            if not os.path.isdir(class_path):
                continue
            parts_split = class_dir_name.split("_")
            if len(parts_split) != 2:
                continue
            loc_id, alt = parts_split[0], parts_split[1]
            if loc_id not in loc_to_idx:
                continue
            label_idx = loc_to_idx[loc_id]
            for fname in sorted(os.listdir(class_path)):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                stem = os.path.splitext(fname)[0]
                if '-' not in stem:
                    continue
                w = stem.rsplit('-', 1)[-1]
                if w != weather_name:
                    continue
                try:
                    img = transform(Image.open(os.path.join(class_path, fname)).convert('RGB'))
                    drone_imgs.append(img)
                    drone_lbls.append(label_idx)
                    drone_alt_idxs.append(CFG.ALT_TO_IDX.get(alt, 0))
                except Exception:
                    pass

    # 2. Clean TRAIN drone images (distractors)
    drone_dir = os.path.join(CFG.SUES_ROOT, CFG.DRONE_DIR)
    for loc in train_locs:
        for alt in CFG.ALTITUDES:
            alt_dir = os.path.join(drone_dir, loc, alt)
            if not os.path.isdir(alt_dir):
                continue
            imgs_in_dir = sorted(f for f in os.listdir(alt_dir)
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png')))
            if not imgs_in_dir:
                continue
            try:
                img = transform(Image.open(os.path.join(alt_dir, imgs_in_dir[0])).convert('RGB'))
                drone_imgs.append(img)
                drone_lbls.append(-1000 - distractor_cnt)
                drone_alt_idxs.append(CFG.ALT_TO_IDX.get(alt, 0))
                distractor_cnt += 1
            except Exception:
                pass

    # Embed all drone images
    drone_feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(drone_imgs), CFG.BATCH_SIZE):
            bimgs = torch.stack(drone_imgs[i:i + CFG.BATCH_SIZE]).to(device)
            balts = torch.tensor(drone_alt_idxs[i:i + CFG.BATCH_SIZE], device=device)
            with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
                feats = model.extract_embedding(bimgs, alt_idx=balts)
            drone_feats.append(feats.cpu())

    drone_feats = torch.cat(drone_feats)
    drone_lbls  = torch.tensor(drone_lbls)
    test_cnt    = len(drone_feats) - distractor_cnt
    print(f"  Drone gallery S2D ({weather_name}): {len(drone_feats)} "
          f"(test={test_cnt}, distractors={distractor_cnt})")
    return drone_feats, drone_lbls


@torch.no_grad()
def evaluate_s2d(sat_feats, sat_labels, drone_feats, drone_labels):
    """
    Satellite→Drone retrieval.
    Queries  = test satellite images (sat_labels >= 0, 80 queries).
    Gallery  = weather-augmented test drones + clean distractor train drones.
    Each query has up to 4 matches (one per altitude of same location).
    """
    q_mask   = sat_labels >= 0
    q_feats  = sat_feats[q_mask]
    q_labels = sat_labels[q_mask]
    sim = q_feats @ drone_feats.T
    _, rank = sim.sort(1, descending=True)
    N = q_feats.size(0)
    r1 = r5 = r10 = ap = 0
    for i in range(N):
        matches = torch.where(drone_labels[rank[i]] == q_labels[i])[0]
        if len(matches) == 0:
            continue
        first = matches[0].item()
        if first < 1:  r1 += 1
        if first < 5:  r5 += 1
        if first < 10: r10 += 1
        ap += sum((j + 1) / (p.item() + 1) for j, p in enumerate(matches)) / len(matches)
    return {
        'R@1': r1 / N * 100, 'R@5': r5 / N * 100,
        'R@10': r10 / N * 100, 'mAP': ap / N * 100,
        'num_queries': N,
    }


def print_s2d_weather_table(all_s2d_results, normal_s2d):
    """Print Satellite→Drone per-weather results table."""
    print(f"\n{'='*85}")
    print(f"  WEATHER ROBUSTNESS — Satellite→Drone Retrieval (SUES-200 Test)")
    print(f"{'='*85}")
    print(f"  {'Weather':>12s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}  {'\u0394R@1':>7s}  {'\u0394mAP':>7s}  {'#Q':>5s}")
    print(f"  {'-'*78}")
    nr1 = normal_s2d['R@1']; nmap = normal_s2d['mAP']
    for w_name in CFG.WEATHER_NAMES:
        r = all_s2d_results[w_name]
        dr1 = r['R@1'] - nr1; dmap = r['mAP'] - nmap
        print(f"  {w_name:>12s}  {r['R@1']:6.2f}%  {r['R@5']:6.2f}%  {r['R@10']:6.2f}%  "
              f"{r['mAP']:6.2f}%  {dr1:+6.2f}%  {dmap:+6.2f}%  "
              f"{r['num_queries']:>5d}")
    print(f"  {'-'*78}")
    adv = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    avg_r1  = np.mean([all_s2d_results[w]['R@1']  for w in CFG.WEATHER_NAMES])
    avg_r5  = np.mean([all_s2d_results[w]['R@5']  for w in CFG.WEATHER_NAMES])
    avg_r10 = np.mean([all_s2d_results[w]['R@10'] for w in CFG.WEATHER_NAMES])
    avg_map = np.mean([all_s2d_results[w]['mAP']  for w in CFG.WEATHER_NAMES])
    avg_adv_r1  = np.mean([all_s2d_results[w]['R@1']  for w in adv])
    avg_adv_map = np.mean([all_s2d_results[w]['mAP']  for w in adv])
    print(f"  {'Avg(all)':>12s}  {avg_r1:6.2f}%  {avg_r5:6.2f}%  {avg_r10:6.2f}%  {avg_map:6.2f}%")
    print(f"  {'Avg(adverse)':>12s}  {avg_adv_r1:6.2f}%  {'':>7s}  {'':>7s}  {avg_adv_map:6.2f}%  "
          f"{avg_adv_r1-nr1:+5.2f}%  {avg_adv_map-nmap:+5.2f}%")
    print(f"{'='*85}")


def measure_model_complexity(model, device):
    sz = CFG.IMG_SIZE
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.eval()
    dummy = torch.randn(1, 3, sz, sz).to(device)
    gflops = None
    try:
        from thop import profile as thop_profile
        class _Wrap(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): return self.m.extract_embedding(x)
        macs, _ = thop_profile(_Wrap(model).to(device), inputs=(dummy,), verbose=False)
        gflops = macs / 1e9
    except Exception:
        pass
    model.eval()
    with torch.no_grad():
        for _ in range(10): model.extract_embedding(dummy)
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100): model.extract_embedding(dummy)
        if device.type == 'cuda': torch.cuda.synchronize()
        ms_query = (time.time() - t0) / 100 * 1000
    return {
        'total_params_M': total_params / 1e6,
        'trainable_params_M': trainable_params / 1e6,
        'gflops': gflops,
        'ms_per_query': ms_query,
        'img_size': sz,
    }


# =============================================================================
# PRINT / TABLE FUNCTIONS
# =============================================================================
def print_complexity(stats):
    print(f"\n  ┌─ Model Complexity ─────────────────────────────┐")
    print(f"  │  Total params:     {stats['total_params_M']:>8.2f}M                │")
    print(f"  │  Trainable params: {stats['trainable_params_M']:>8.2f}M                │")
    if stats['gflops'] is not None:
        print(f"  │  GFLOPs:           {stats['gflops']:>8.2f}                 │")
    else:
        print(f"  │  GFLOPs:           N/A (pip install thop)    │")
    print(f"  │  Inference:        {stats['ms_per_query']:>8.1f} ms/query         │")
    print(f"  │  Image size:       {stats['img_size']:>4d}×{stats['img_size']:<4d}               │")
    print(f"  └─────────────────────────────────────────────────┘")


def print_weather_table(all_results, normal_results):
    print(f"\n{'='*85}")
    print(f"  WEATHER ROBUSTNESS — Drone→Satellite Retrieval (SUES-200 Test)")
    print(f"{'='*85}")
    print(f"  {'Weather':>12s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}  {'ΔR@1':>7s}  {'ΔmAP':>7s}  {'#Q':>5s}")
    print(f"  {'-'*78}")
    nr1 = normal_results['R@1']; nmap = normal_results['mAP']
    for w_name in CFG.WEATHER_NAMES:
        r = all_results[w_name]['overall']
        dr1 = r['R@1'] - nr1; dmap = r['mAP'] - nmap
        sign_r1 = '+' if dr1 >= 0 else ''; sign_map = '+' if dmap >= 0 else ''
        print(f"  {w_name:>12s}  {r['R@1']:6.2f}%  {r['R@5']:6.2f}%  {r['R@10']:6.2f}%  "
              f"{r['mAP']:6.2f}%  {sign_r1}{dr1:5.2f}%  {sign_map}{dmap:5.2f}%  "
              f"{r['num_queries']:>5d}")
    print(f"  {'-'*78}")
    avg_r1  = np.mean([all_results[w]['overall']['R@1']  for w in CFG.WEATHER_NAMES])
    avg_r5  = np.mean([all_results[w]['overall']['R@5']  for w in CFG.WEATHER_NAMES])
    avg_r10 = np.mean([all_results[w]['overall']['R@10'] for w in CFG.WEATHER_NAMES])
    avg_map = np.mean([all_results[w]['overall']['mAP']  for w in CFG.WEATHER_NAMES])
    adv_weathers = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    avg_adv_r1  = np.mean([all_results[w]['overall']['R@1'] for w in adv_weathers])
    avg_adv_map = np.mean([all_results[w]['overall']['mAP'] for w in adv_weathers])
    print(f"  {'Avg(all)':>12s}  {avg_r1:6.2f}%  {avg_r5:6.2f}%  {avg_r10:6.2f}%  {avg_map:6.2f}%")
    print(f"  {'Avg(adverse)':>12s}  {avg_adv_r1:6.2f}%  {'':>7s}  {'':>7s}  {avg_adv_map:6.2f}%  "
          f"{avg_adv_r1-nr1:+5.2f}%  {avg_adv_map-nmap:+5.2f}%")
    print(f"{'='*85}")


def print_cross_table(all_results):
    print(f"\n{'='*75}")
    print(f"  R@1 (%) — Weather × Altitude")
    print(f"{'='*75}")
    header = f"  {'Weather':>12s}"
    for alt in CFG.ALTITUDES:
        header += f"  {alt+'m':>7s}"
    header += f"  {'Avg':>7s}"
    print(header)
    print(f"  {'-'*60}")
    for w in CFG.WEATHER_NAMES:
        row = f"  {w:>12s}"
        vals = []
        for alt in CFG.ALTITUDES:
            v = all_results[w]['per_alt'].get(int(alt), {}).get('R@1', 0)
            row += f"  {v:6.2f}%"; vals.append(v)
        row += f"  {np.mean(vals):6.2f}%"
        print(row)
    print(f"{'='*75}")


# =============================================================================
# KNOWN RESULTS FROM PREVIOUS EXPERIMENTS
# =============================================================================
EXP49_RESULTS = {   # zero-shot (EXP35 checkpoint, old eval script)
    'normal':    {'R@1': 95.94, 'mAP': 97.64},
    'fog':       {'R@1': 86.88, 'mAP': 92.68},
    'rain':      {'R@1': 43.75, 'mAP': 56.99},
    'snow':      {'R@1': 40.62, 'mAP': 54.48},
    'dark':      {'R@1': 66.88, 'mAP': 76.74},
    'light':     {'R@1': 83.44, 'mAP': 89.97},
    'fog_rain':  {'R@1': 42.19, 'mAP': 54.92},
    'fog_snow':  {'R@1': 22.19, 'mAP': 35.01},
    'rain_snow': {'R@1': 34.69, 'mAP': 46.85},
    'wind':      {'R@1': 81.25, 'mAP': 88.76},
}
EXP50_RESULTS = {   # finetune + pre-generated weather
    'normal':    {'R@1': 94.38, 'mAP': 96.78},
    'fog':       {'R@1': 90.94, 'mAP': 95.00},
    'rain':      {'R@1': 85.31, 'mAP': 91.07},
    'snow':      {'R@1': 88.12, 'mAP': 92.86},
    'dark':      {'R@1': 80.00, 'mAP': 86.58},
    'light':     {'R@1': 86.25, 'mAP': 91.42},
    'fog_rain':  {'R@1': 78.12, 'mAP': 86.61},
    'fog_snow':  {'R@1': 73.44, 'mAP': 82.36},
    'rain_snow': {'R@1': 84.69, 'mAP': 90.36},
    'wind':      {'R@1': 85.62, 'mAP': 90.78},
}
EXP51_RESULTS = {   # scratch + pre-generated weather
    'normal':    {'R@1': 90.94, 'mAP': 95.18},
    'fog':       {'R@1': 89.69, 'mAP': 94.37},
    'rain':      {'R@1': 86.56, 'mAP': 92.09},
    'snow':      {'R@1': 89.69, 'mAP': 93.66},
    'dark':      {'R@1': 80.31, 'mAP': 87.74},
    'light':     {'R@1': 86.88, 'mAP': 91.98},
    'fog_rain':  {'R@1': 81.25, 'mAP': 88.60},
    'fog_snow':  {'R@1': 75.00, 'mAP': 84.63},
    'rain_snow': {'R@1': 86.25, 'mAP': 91.27},
    'wind':      {'R@1': 88.12, 'mAP': 92.98},
}
# Fill these in after EXP52/53 complete (set to None to skip in comparison)
EXP52_RESULTS = {   # finetune + online aug (WeatherPrompt-style)
    'normal':    {'R@1': 95.00, 'mAP': 96.71},
    'fog':       {'R@1': 90.00, 'mAP': 94.30},
    'rain':      {'R@1': 89.69, 'mAP': 93.05},
    'snow':      {'R@1': 87.50, 'mAP': 92.05},
    'dark':      {'R@1': 78.75, 'mAP': 85.07},
    'light':     {'R@1': 84.69, 'mAP': 90.14},
    'fog_rain':  {'R@1': 81.88, 'mAP': 88.64},
    'fog_snow':  {'R@1': 69.38, 'mAP': 79.21},
    'rain_snow': {'R@1': 84.69, 'mAP': 90.06},
    'wind':      {'R@1': 88.75, 'mAP': 92.93},
}
EXP53_RESULTS = {   # scratch  + online aug (WeatherPrompt-style)
    'normal':    {'R@1': 92.81, 'mAP': 95.81},
    'fog':       {'R@1': 91.88, 'mAP': 94.94},
    'rain':      {'R@1': 90.31, 'mAP': 94.18},
    'snow':      {'R@1': 87.19, 'mAP': 92.19},
    'dark':      {'R@1': 82.50, 'mAP': 87.90},
    'light':     {'R@1': 86.25, 'mAP': 91.25},
    'fog_rain':  {'R@1': 85.62, 'mAP': 91.19},
    'fog_snow':  {'R@1': 79.38, 'mAP': 86.89},
    'rain_snow': {'R@1': 84.69, 'mAP': 90.76},
    'wind':      {'R@1': 87.50, 'mAP': 92.33},
}


def print_comparison(all_results):
    """Compare EXP54 against all known prior experiments."""
    # Build list of experiments that have real data
    exps = [
        ("EXP49", EXP49_RESULTS, "zero-shot"),
        ("EXP50", EXP50_RESULTS, "ft-pregen"),
        ("EXP51", EXP51_RESULTS, "scratch-pregen"),
    ]
    if EXP52_RESULTS is not None:
        exps.append(("EXP52", EXP52_RESULTS, "ft-online"))
    if EXP53_RESULTS is not None:
        exps.append(("EXP53", EXP53_RESULTS, "scratch-online"))

    exp_name_short = CFG.EXPERIMENT_NAME.split("_")[-1]   # e.g. "EXP35"
    col_width = max(8, len(exp_name_short) + 2)

    header  = f"\n{'='*120}\n"
    header += f"  COMPARISON (R@1%): "
    header += " vs ".join(f"{e[0]}({e[2]})" for e in exps)
    header += f" vs {CFG.EXPERIMENT_NAME}\n"
    header += f"{'='*120}"
    print(header)

    # Column headers
    row = f"  {'Weather':>12s}"
    for exp_id, _, _ in exps:
        row += f"  {exp_id:>8s}"
    row += f"  {exp_name_short:>8s}"
    print(row)
    print(f"  {'-'*80}")

    for w in CFG.WEATHER_NAMES:
        row = f"  {w:>12s}"
        for _, res, _ in exps:
            row += f"  {res[w]['R@1']:7.2f}%"
        e54_r1 = all_results[w]['overall']['R@1']
        row += f"  {e54_r1:7.2f}%"
        print(row)

    print(f"  {'-'*80}")
    adv = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    row = f"  {'Avg(adverse)':>12s}"
    for _, res, _ in exps:
        row += f"  {np.mean([res[w]['R@1'] for w in adv]):7.2f}%"
    row += f"  {np.mean([all_results[w]['overall']['R@1'] for w in adv]):7.2f}%"
    print(row)

    row = f"  {'Avg(all)':>12s}"
    for _, res, _ in exps:
        row += f"  {np.mean([res[w]['R@1'] for w in CFG.WEATHER_NAMES]):7.2f}%"
    row += f"  {np.mean([all_results[w]['overall']['R@1'] for w in CFG.WEATHER_NAMES]):7.2f}%"
    print(row)
    print(f"{'='*120}")


def generate_latex_table(all_results, normal_results):
    nr1 = normal_results['R@1']; nmap = normal_results['mAP']
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Zero-shot weather robustness evaluation ({CFG.EXPERIMENT_NAME}).}}",
        rf"\label{{tab:{CFG.EXPERIMENT_NAME.lower()}}}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l|cccc|cc}",
        r"\toprule",
        r"Weather & R@1 & R@5 & R@10 & mAP & $\Delta$R@1 & $\Delta$mAP \\",
        r"\midrule",
    ]
    for w in CFG.WEATHER_NAMES:
        r = all_results[w]['overall']
        dr1 = r['R@1'] - nr1; dmap = r['mAP'] - nmap
        label = w.replace('_', r'\_')
        if w == 'normal':
            lines.append(f"{label} & {r['R@1']:.2f} & {r['R@5']:.2f} & {r['R@10']:.2f} & "
                         f"{r['mAP']:.2f} & --- & --- \\\\")
        else:
            lines.append(f"{label} & {r['R@1']:.2f} & {r['R@5']:.2f} & {r['R@10']:.2f} & "
                         f"{r['mAP']:.2f} & {dr1:+.2f} & {dmap:+.2f} \\\\")
    adv = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    avg_r1  = np.mean([all_results[w]['overall']['R@1']  for w in adv])
    avg_r5  = np.mean([all_results[w]['overall']['R@5']  for w in adv])
    avg_r10 = np.mean([all_results[w]['overall']['R@10'] for w in adv])
    avg_map = np.mean([all_results[w]['overall']['mAP']  for w in adv])
    lines += [
        r"\midrule",
        f"Avg(adverse) & {avg_r1:.2f} & {avg_r5:.2f} & {avg_r10:.2f} & "
        f"{avg_map:.2f} & {avg_r1-nr1:+.2f} & {avg_map-nmap:+.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]
    return '\n'.join(lines)


# =============================================================================
# MAIN
# =============================================================================
def main():
    import random
    random.seed(CFG.SEED); np.random.seed(CFG.SEED)
    torch.manual_seed(CFG.SEED); torch.cuda.manual_seed_all(CFG.SEED)
    torch.backends.cudnn.benchmark = True

    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    start_time = datetime.datetime.now()

    print("=" * 75)
    print(f"  {CFG.EXPERIMENT_NAME}")
    print(f"  Zero-Shot Weather Robustness Evaluation (no training)")
    print(f"  Checkpoint: {CFG.CHECKPOINT}")
    print(f"  SUES-200:   {CFG.SUES_ROOT}")
    print(f"  Device:     {DEVICE}")
    print("=" * 75)

    # ---- 1. Load Model ----
    print("\n[1/3] Loading model …")
    if not os.path.exists(CFG.CHECKPOINT):
        print(f"  ERROR: Checkpoint not found at {CFG.CHECKPOINT}")
        return

    model = SPDGeoDPEAMARModel(CFG.NUM_CLASSES).to(DEVICE)
    ckpt = torch.load(CFG.CHECKPOINT, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys  ({len(missing)}): {missing[:5]} …")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]} …")

    ckpt_epoch = ckpt.get('epoch', '?')
    ckpt_r1    = ckpt.get('metrics', {}).get('R@1', ckpt.get('best_r1', '?'))
    print(f"  Loaded  epoch={ckpt_epoch}  pretrained R@1={ckpt_r1}")

    model.eval()

    complexity = measure_model_complexity(model, DEVICE)
    print_complexity(complexity)

    # ---- 2. Build Satellite Gallery ----
    print("\n[2/3] Building satellite gallery …")
    tf = get_test_transform()
    sat_feats, sat_labels = build_satellite_gallery(model, DEVICE, tf)

    # ---- 3. Evaluate All Weather Conditions ----
    print("\n[3/3] Evaluating all 10 weather conditions …")
    all_results = OrderedDict()
    for w_name in CFG.WEATHER_NAMES:
        query_ds = WeatherQueryDataset(CFG.WEATHER_ROOT, w_name, transform=tf)
        if len(query_ds) == 0:
            print(f"  [SKIP] {w_name}: no images found in {CFG.WEATHER_ROOT}")
            all_results[w_name] = {
                'overall': {'R@1': 0, 'R@5': 0, 'R@10': 0, 'mAP': 0, 'num_queries': 0},
                'per_alt': {}
            }
            continue
        overall, per_alt = evaluate_retrieval(model, query_ds, sat_feats, sat_labels, DEVICE)
        all_results[w_name] = {'overall': overall, 'per_alt': per_alt}
        print(f"  {w_name:>12s}  R@1={overall['R@1']:.2f}%  R@5={overall['R@5']:.2f}%  "
              f"R@10={overall['R@10']:.2f}%  mAP={overall['mAP']:.2f}%  ({len(query_ds)} queries)")

    # ---- Print Tables ----
    normal_results = all_results['normal']['overall']
    print_weather_table(all_results, normal_results)
    print_cross_table(all_results)
    print_comparison(all_results)

    # ---- S2D Evaluation (Satellite→Drone) ----
    print(f"\n{'='*75}")
    print(f"  [S2D] Satellite→Drone Evaluation …")
    all_s2d_results = OrderedDict()
    for w_name in CFG.WEATHER_NAMES:
        drone_feats_s2d, drone_lbls_s2d = build_drone_gallery_s2d(
            model, w_name, DEVICE, tf)
        s2d_res = evaluate_s2d(sat_feats, sat_labels, drone_feats_s2d, drone_lbls_s2d)
        all_s2d_results[w_name] = s2d_res
        print(f"    {w_name:>12s}  R@1={s2d_res['R@1']:.2f}%  R@5={s2d_res['R@5']:.2f}%  "
              f"R@10={s2d_res['R@10']:.2f}%  mAP={s2d_res['mAP']:.2f}%")
        del drone_feats_s2d, drone_lbls_s2d
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    normal_s2d = all_s2d_results['normal']
    print_s2d_weather_table(all_s2d_results, normal_s2d)

    # LaTeX
    latex = generate_latex_table(all_results, normal_results)
    print(f"\n{'='*75}\n  LaTeX Table:\n{'='*75}")
    print(latex)

    # ---- Save JSON ----
    end_time = datetime.datetime.now()
    adv_weathers = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    summary = {
        'experiment': CFG.EXPERIMENT_NAME,
        'checkpoint': CFG.CHECKPOINT,
        'ckpt_epoch': ckpt_epoch,
        'dataset': 'SUES-200',
        'direction': 'drone → satellite (D2S) + satellite → drone (S2D) — zero-shot inference',
        'timestamp': {
            'start': start_time.isoformat(),
            'end': end_time.isoformat(),
            'duration_s': (end_time - start_time).total_seconds(),
        },
        'model_complexity': complexity,
        'satellite_gallery_size': int(len(sat_feats)),
        'weather_results': {
            w: {
                'overall': all_results[w]['overall'],
                'per_altitude': {str(k): v for k, v in all_results[w]['per_alt'].items()},
            }
            for w in CFG.WEATHER_NAMES
        },
        'summary_metrics': {
            'normal_R@1': normal_results['R@1'],
            'normal_mAP': normal_results['mAP'],
            'avg_all_R@1':     float(np.mean([all_results[w]['overall']['R@1'] for w in CFG.WEATHER_NAMES])),
            'avg_all_mAP':     float(np.mean([all_results[w]['overall']['mAP'] for w in CFG.WEATHER_NAMES])),
            'avg_adverse_R@1': float(np.mean([all_results[w]['overall']['R@1'] for w in adv_weathers])),
            'avg_adverse_mAP': float(np.mean([all_results[w]['overall']['mAP'] for w in adv_weathers])),
            'vs_exp49': {w: float(all_results[w]['overall']['R@1'] - EXP49_RESULTS[w]['R@1'])
                         for w in CFG.WEATHER_NAMES},
            'vs_exp50': {w: float(all_results[w]['overall']['R@1'] - EXP50_RESULTS[w]['R@1'])
                         for w in CFG.WEATHER_NAMES},
        },
        'latex_table': latex,
        's2d_weather_results': {w: all_s2d_results[w] for w in CFG.WEATHER_NAMES},
        's2d_summary_metrics': {
            'normal_R@1': normal_s2d['R@1'],
            'normal_mAP': normal_s2d['mAP'],
            'avg_all_R@1':     float(np.mean([all_s2d_results[w]['R@1'] for w in CFG.WEATHER_NAMES])),
            'avg_all_mAP':     float(np.mean([all_s2d_results[w]['mAP'] for w in CFG.WEATHER_NAMES])),
            'avg_adverse_R@1': float(np.mean([all_s2d_results[w]['R@1'] for w in adv_weathers])),
            'avg_adverse_mAP': float(np.mean([all_s2d_results[w]['mAP'] for w in adv_weathers])),
        },
    }

    json_name = CFG.EXPERIMENT_NAME.lower() + ".json"
    json_path = os.path.join(CFG.OUTPUT_DIR, json_name)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {json_path}")

    # ---- Final Summary ----
    sm = summary['summary_metrics']
    print(f"\n{'='*75}")
    print(f"  {CFG.EXPERIMENT_NAME} COMPLETE  (zero-shot, no training)")
    print(f"{'='*75}")
    print(f"  Checkpoint:         epoch={ckpt_epoch}")
    print(f"  Normal:             R@1={sm['normal_R@1']:.2f}%  mAP={sm['normal_mAP']:.2f}%")
    print(f"  Avg (all weather):  R@1={sm['avg_all_R@1']:.2f}%  mAP={sm['avg_all_mAP']:.2f}%")
    print(f"  Avg (adverse):      R@1={sm['avg_adverse_R@1']:.2f}%  mAP={sm['avg_adverse_mAP']:.2f}%")
    e49_adv = np.mean([EXP49_RESULTS[w]['R@1'] for w in adv_weathers])
    print(f"\n  vs EXP49 (zero-shot, old eval)  avg adverse R@1: "
          f"{e49_adv:.2f}% → {sm['avg_adverse_R@1']:.2f}% "
          f"({sm['avg_adverse_R@1'] - e49_adv:+.2f}%)")
    e50_adv = np.mean([EXP50_RESULTS[w]['R@1'] for w in adv_weathers])
    print(f"  vs EXP50 (ft+pregen)            avg adverse R@1: "
          f"{e50_adv:.2f}% → {sm['avg_adverse_R@1']:.2f}% "
          f"({sm['avg_adverse_R@1'] - e50_adv:+.2f}%)")
    print(f"  Eval duration:      {(end_time - start_time).total_seconds():.1f}s")
    s2d_sm = summary['s2d_summary_metrics']
    print(f"\n  S2D — Satellite→Drone:")
    print(f"  Normal:             R@1={normal_s2d['R@1']:.2f}%  mAP={normal_s2d['mAP']:.2f}%")
    print(f"  Avg (all weather):  R@1={s2d_sm['avg_all_R@1']:.2f}%  mAP={s2d_sm['avg_all_mAP']:.2f}%")
    print(f"  Avg (adverse):      R@1={s2d_sm['avg_adverse_R@1']:.2f}%  mAP={s2d_sm['avg_adverse_mAP']:.2f}%")
    print(f"{'='*75}")


main()
# =============================================================================
# EXP54: Zero-Shot Weather Robustness Evaluation
# =============================================================================
# Copy to Kaggle notebook → Run All → Done
#
# Pure inference — NO training.
# Loads any SPDGeo-DPEA-MAR checkpoint and evaluates on all 10 weather
# conditions using pre-generated test images (same protocol as EXP49-53).
#
# Designed for:
#   1. EXP35 checkpoint (baseline / "our model, no weather training")
#   2. EXP52 best checkpoint (finetune + online aug)
#   3. EXP53 best checkpoint (scratch + online aug)
#   → Change CFG.CHECKPOINT and CFG.EXPERIMENT_NAME accordingly
#
# Kaggle Data Sources:
#   1. SUES-200 original:  satellite gallery + clean drone test
#   2. Weather synthetic:  weather-augmented drone TEST images (for eval)
#   3. Model checkpoint:   any SPDGeo-DPEA-MAR .pth file
# =============================================================================

import subprocess, sys
for _p in ["timm", "tqdm", "thop"]:
    try: __import__(_p)
    except ImportError: subprocess.run(f"pip install -q {_p}", shell=True, capture_output=True)

import os, json, time, datetime, gc
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm


# =============================================================================
# CONFIG
# =============================================================================
class Config:
    # --- Which checkpoint to evaluate ---
    # Switch between EXP35, EXP52, EXP53 best by changing these two fields
    EXPERIMENT_NAME = "EXP54_ZeroShot_EXP35"
    CHECKPOINT      = "/kaggle/input/models/minh2duy/exp35-spdgeo-dpea-m-unfreeze-blocks-4-6/pytorch/default/1/exp35_dpea_ga_best.pth"

    # For EXP52 best:
    # EXPERIMENT_NAME = "EXP54_ZeroShot_EXP52"
    # CHECKPOINT      = "/kaggle/working/exp52_weather_finetune_online_best.pth"

    # For EXP53 best:
    # EXPERIMENT_NAME = "EXP54_ZeroShot_EXP53"
    # CHECKPOINT      = "/kaggle/working/exp53_weather_scratch_online_best.pth"

    # --- Paths ---
    SUES_ROOT       = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
    WEATHER_ROOT    = "/kaggle/input/datasets/minh2duy/sues200-weather/weather_synthetic"
    OUTPUT_DIR      = "/kaggle/working"

    # --- Dataset structure ---
    DRONE_DIR       = "drone-view"
    SAT_DIR         = "satellite-view"
    ALTITUDES       = ["150", "200", "250", "300"]
    ALT_TO_IDX      = {"150": 0, "200": 1, "250": 2, "300": 3}
    NUM_ALTITUDES   = 4
    TEST_LOCS       = list(range(121, 201))   # 80 test locations
    NUM_CLASSES     = 120

    # --- Model architecture (must match checkpoint) ---
    IMG_SIZE        = 336
    N_PARTS         = 8
    PART_DIM        = 256
    EMBED_DIM       = 512
    TEACHER_DIM     = 768
    CLUSTER_TEMP    = 0.07
    UNFREEZE_BLOCKS = 6
    MASK_RATIO      = 0.30

    # --- Weather conditions ---
    WEATHER_NAMES   = ["normal", "fog", "rain", "snow", "dark", "light",
                       "fog_rain", "fog_snow", "rain_snow", "wind"]

    # --- Inference ---
    USE_AMP         = True
    BATCH_SIZE      = 64
    NUM_WORKERS     = 2
    SEED            = 42


CFG = Config()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =============================================================================
# MODEL DEFINITION (identical to EXP35 / EXP52 / EXP53)
# =============================================================================
class DINOv2Backbone(nn.Module):
    def __init__(self, unfreeze_blocks=4):
        super().__init__()
        self.model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        self.feature_dim = 384; self.patch_size = 14
        for p in self.model.parameters(): p.requires_grad = False
        for blk in self.model.blocks[-unfreeze_blocks:]:
            for p in blk.parameters(): p.requires_grad = True
        for p in self.model.norm.parameters(): p.requires_grad = True
        frozen = sum(p.numel() for p in self.model.parameters() if not p.requires_grad)
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  DINOv2 ViT-S/14: {frozen/1e6:.1f}M frozen, {trainable/1e6:.1f}M trainable")

    def forward(self, x):
        features = self.model.forward_features(x)
        patch_tokens = features['x_norm_patchtokens']
        cls_token = features['x_norm_clstoken']
        H = x.shape[2] // self.patch_size; W = x.shape[3] // self.patch_size
        return patch_tokens, cls_token, (H, W)


class DeepAltitudeFiLM(nn.Module):
    def __init__(self, num_altitudes=4, feat_dim=256):
        super().__init__()
        self.num_altitudes = num_altitudes
        self.gamma = nn.Parameter(torch.ones(num_altitudes, feat_dim))
        self.beta  = nn.Parameter(torch.zeros(num_altitudes, feat_dim))

    def forward(self, feat, alt_idx=None):
        if alt_idx is None:
            gamma = self.gamma.mean(dim=0, keepdim=True)
            beta  = self.beta.mean(dim=0, keepdim=True)
            return feat * gamma.unsqueeze(0) + beta.unsqueeze(0)
        else:
            gamma = self.gamma[alt_idx]
            beta  = self.beta[alt_idx]
            return feat * gamma.unsqueeze(1) + beta.unsqueeze(1)


class AltitudeAwarePartDiscovery(nn.Module):
    def __init__(self, feat_dim=384, n_parts=8, part_dim=256, temperature=0.07,
                 num_altitudes=4):
        super().__init__()
        self.n_parts = n_parts; self.temperature = temperature
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, part_dim), nn.LayerNorm(part_dim), nn.GELU()
        )
        self.altitude_film = DeepAltitudeFiLM(num_altitudes, part_dim)
        self.prototypes = nn.Parameter(torch.randn(n_parts, part_dim) * 0.02)
        self.refine = nn.Sequential(
            nn.LayerNorm(part_dim), nn.Linear(part_dim, part_dim * 2),
            nn.GELU(), nn.Linear(part_dim * 2, part_dim)
        )
        self.salience_head = nn.Sequential(
            nn.Linear(part_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, patch_features, spatial_hw, alt_idx=None):
        B, N, _ = patch_features.shape; H, W = spatial_hw
        feat = self.feat_proj(patch_features)
        feat = self.altitude_film(feat, alt_idx)
        feat_norm = F.normalize(feat, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum('bnd,kd->bnk', feat_norm, proto_norm) / self.temperature
        assign = F.softmax(sim, dim=-1)
        assign_t = assign.transpose(1, 2)
        mass = assign_t.sum(-1, keepdim=True).clamp(min=1e-6)
        part_feat = torch.bmm(assign_t, feat) / mass
        part_feat = part_feat + self.refine(part_feat)
        device = feat.device
        gy = torch.arange(H, device=device).float() / max(H - 1, 1)
        gx = torch.arange(W, device=device).float() / max(W - 1, 1)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)
        part_pos = torch.bmm(assign_t, coords.unsqueeze(0).expand(B, -1, -1)) / mass
        salience = self.salience_head(part_feat).squeeze(-1)
        return {'part_features': part_feat, 'part_positions': part_pos,
                'assignment': assign, 'salience': salience,
                'projected_patches': feat}


class PartAwarePooling(nn.Module):
    def __init__(self, part_dim=256, embed_dim=512):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(part_dim, part_dim // 2), nn.Tanh(),
                                  nn.Linear(part_dim // 2, 1))
        self.proj = nn.Sequential(nn.Linear(part_dim * 3, embed_dim), nn.LayerNorm(embed_dim),
                                  nn.GELU(), nn.Linear(embed_dim, embed_dim))

    def forward(self, part_features, salience=None):
        B, K, D = part_features.shape
        aw = self.attn(part_features)
        if salience is not None: aw = aw + salience.unsqueeze(-1).log().clamp(-10)
        aw = F.softmax(aw, dim=1)
        attn_pool = (aw * part_features).sum(1)
        mean_pool = part_features.mean(1); max_pool = part_features.max(1)[0]
        combined = torch.cat([attn_pool, mean_pool, max_pool], dim=-1)
        return F.normalize(self.proj(combined), dim=-1)


class DynamicFusionGate(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
        )
        nn.init.constant_(self.gate[-1].bias, 0.85)

    def forward(self, part_emb, cls_emb):
        combined = torch.cat([part_emb, cls_emb], dim=-1)
        alpha = torch.sigmoid(self.gate(combined))
        fused = alpha * part_emb + (1 - alpha) * cls_emb
        return F.normalize(fused, dim=-1)


class MaskedPartReconstruction(nn.Module):
    def __init__(self, part_dim=256, mask_ratio=0.30):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.decoder = nn.Sequential(
            nn.Linear(part_dim, part_dim * 2), nn.GELU(),
            nn.Linear(part_dim * 2, part_dim), nn.LayerNorm(part_dim),
        )
        self.mask_token = nn.Parameter(torch.randn(1, 1, part_dim) * 0.02)

    def forward(self, projected_patches, part_features, assignment):
        B, K, D = part_features.shape
        n_mask = max(1, int(K * self.mask_ratio))
        mask_idx = torch.randperm(K, device=part_features.device)[:n_mask]
        masked = part_features.clone()
        masked[:, mask_idx] = self.mask_token.expand(B, n_mask, -1)
        reconstructed = self.decoder(masked)
        return F.mse_loss(reconstructed[:, mask_idx], part_features[:, mask_idx].detach())


class AltitudePredictionHead(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128), nn.ReLU(inplace=True),
            nn.Dropout(0.1), nn.Linear(128, 1), nn.Sigmoid(),
        )

    def forward(self, embedding, alt_target):
        pred = self.head(embedding).squeeze(-1)
        return F.mse_loss(pred, alt_target)


class SPDGeoDPEAMARModel(nn.Module):
    def __init__(self, num_classes, cfg=CFG):
        super().__init__()
        self.backbone  = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc = AltitudeAwarePartDiscovery(
            384, cfg.N_PARTS, cfg.PART_DIM, cfg.CLUSTER_TEMP, cfg.NUM_ALTITUDES
        )
        self.pool      = PartAwarePooling(cfg.PART_DIM, cfg.EMBED_DIM)
        self.fusion_gate = DynamicFusionGate(cfg.EMBED_DIM)
        self.bottleneck = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.EMBED_DIM),
                                        nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.classifier     = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.cls_proj       = nn.Sequential(nn.Linear(384, cfg.EMBED_DIM),
                                            nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))
        self.cls_classifier = nn.Linear(cfg.EMBED_DIM, num_classes)
        self.teacher_proj   = nn.Sequential(nn.Linear(cfg.EMBED_DIM, cfg.TEACHER_DIM),
                                            nn.LayerNorm(cfg.TEACHER_DIM))
        self.mask_recon = MaskedPartReconstruction(cfg.PART_DIM, cfg.MASK_RATIO)
        self.alt_pred   = AltitudePredictionHead(cfg.EMBED_DIM)

        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"  SPDGeo-DPEA-MAR: {total/1e6:.1f}M trainable parameters")

    def extract_embedding(self, x, alt_idx=None):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx)
        emb = self.pool(parts['part_features'], parts['salience'])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(emb, cls_emb)


# =============================================================================
# DATASET — weather test queries (pre-generated)
# =============================================================================
class WeatherQueryDataset(Dataset):
    def __init__(self, weather_root, weather_name, transform=None):
        self.transform = transform
        self.weather_name = weather_name
        self.samples = []

        weather_test_dir = os.path.join(weather_root, "sues200_weather_test")
        if not os.path.isdir(weather_test_dir):
            return

        test_locs = [f"{l:04d}" for l in CFG.TEST_LOCS]
        loc_to_idx = {l: i for i, l in enumerate(test_locs)}

        for class_dir_name in sorted(os.listdir(weather_test_dir)):
            class_path = os.path.join(weather_test_dir, class_dir_name)
            if not os.path.isdir(class_path):
                continue
            parts = class_dir_name.split("_")
            if len(parts) != 2:
                continue
            loc_id, alt = parts[0], parts[1]
            if loc_id not in loc_to_idx:
                continue
            label_idx = loc_to_idx[loc_id]
            for fname in sorted(os.listdir(class_path)):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                stem = os.path.splitext(fname)[0]
                if '-' not in stem:
                    continue
                w = stem.rsplit('-', 1)[-1]
                if w == weather_name:
                    self.samples.append((os.path.join(class_path, fname), label_idx, alt))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, alt = self.samples[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        alt_idx = CFG.ALT_TO_IDX.get(alt, 0)
        return {'drone': img, 'label': label, 'altitude': int(alt), 'alt_idx': alt_idx}


# =============================================================================
# HELPERS
# =============================================================================
def get_test_transform():
    sz = CFG.IMG_SIZE
    return transforms.Compose([
        transforms.Resize((sz, sz)),
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])


def build_satellite_gallery(model, device, transform):
    sat_dir = os.path.join(CFG.SUES_ROOT, CFG.SAT_DIR)
    test_locs = [f"{l:04d}" for l in CFG.TEST_LOCS]
    loc_to_idx = {l: i for i, l in enumerate(test_locs)}
    all_locs = [f"{l:04d}" for l in range(1, 201)]

    sat_imgs, sat_labels = [], []
    distractor_cnt = 0
    for loc in all_locs:
        sp = os.path.join(sat_dir, loc, "0.png")
        if not os.path.exists(sp):
            continue
        sat_imgs.append(transform(Image.open(sp).convert('RGB')))
        if loc in loc_to_idx:
            sat_labels.append(loc_to_idx[loc])
        else:
            sat_labels.append(-1000 - distractor_cnt)
            distractor_cnt += 1

    sat_feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(sat_imgs), CFG.BATCH_SIZE):
            batch = torch.stack(sat_imgs[i:i + CFG.BATCH_SIZE]).to(device)
            with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
                feats = model.extract_embedding(batch, alt_idx=None)
            sat_feats.append(feats.cpu())

    sat_feats = torch.cat(sat_feats)
    sat_labels = torch.tensor(sat_labels)
    print(f"  Satellite gallery: {len(sat_feats)} images ({distractor_cnt} distractors)")
    return sat_feats, sat_labels


@torch.no_grad()
def evaluate_retrieval(model, query_ds, sat_feats, sat_labels, device):
    model.eval()
    loader = DataLoader(query_ds, batch_size=CFG.BATCH_SIZE, shuffle=False,
                        num_workers=CFG.NUM_WORKERS, pin_memory=True)

    drone_feats, drone_labels, drone_alts = [], [], []
    for batch in loader:
        imgs = batch['drone'].to(device)
        alt_idx = batch['alt_idx'].to(device)
        with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
            feats = model.extract_embedding(imgs, alt_idx=alt_idx)
        drone_feats.append(feats.cpu())
        drone_labels.append(batch['label'])
        drone_alts.append(batch['altitude'])

    drone_feats = torch.cat(drone_feats)
    drone_labels = torch.cat(drone_labels)
    drone_alts = torch.cat(drone_alts)

    sim = drone_feats @ sat_feats.T
    _, rank = sim.sort(1, descending=True)
    N = drone_feats.size(0)

    r1 = r5 = r10 = ap = 0
    for i in range(N):
        matches = torch.where(sat_labels[rank[i]] == drone_labels[i])[0]
        if len(matches) == 0:
            continue
        first = matches[0].item()
        if first < 1:  r1 += 1
        if first < 5:  r5 += 1
        if first < 10: r10 += 1
        ap += sum((j + 1) / (p.item() + 1) for j, p in enumerate(matches)) / len(matches)

    overall = {
        'R@1': r1 / N * 100, 'R@5': r5 / N * 100,
        'R@10': r10 / N * 100, 'mAP': ap / N * 100,
        'num_queries': N,
    }

    per_alt = {}
    for alt in sorted(drone_alts.unique().tolist()):
        mask = drone_alts == alt
        af = drone_feats[mask]; al = drone_labels[mask]
        s = af @ sat_feats.T; _, rk = s.sort(1, descending=True)
        n = af.size(0); a1 = a5 = a10 = aap = 0
        for i in range(n):
            m = torch.where(sat_labels[rk[i]] == al[i])[0]
            if len(m) == 0: continue
            f = m[0].item()
            if f < 1:  a1 += 1
            if f < 5:  a5 += 1
            if f < 10: a10 += 1
            aap += sum((j + 1) / (p.item() + 1) for j, p in enumerate(m)) / len(m)
        per_alt[int(alt)] = {'R@1': a1/n*100, 'R@5': a5/n*100,
                             'R@10': a10/n*100, 'mAP': aap/n*100, 'n': n}

    return overall, per_alt


# =============================================================================
# SATELLITE→DRONE EVALUATION (S2D)
# =============================================================================
def build_drone_gallery_s2d(model, weather_name, device, transform):
    """
    Builds drone gallery for Satellite→Drone (S2D) retrieval.
    Gallery = weather-augmented test drones (labeled 0..79) +
              clean train drones as distractors (labeled -1000...).
    Uses CFG.WEATHER_ROOT / CFG.SUES_ROOT directly.
    """
    test_locs  = [f"{l:04d}" for l in CFG.TEST_LOCS]
    loc_to_idx = {l: i for i, l in enumerate(test_locs)}
    train_locs = [f"{l:04d}" for l in range(1, CFG.NUM_CLASSES + 1)]

    drone_imgs, drone_lbls, drone_alt_idxs = [], [], []
    distractor_cnt = 0

    # 1. Weather-augmented TEST drone images (labeled)
    weather_test_dir = os.path.join(CFG.WEATHER_ROOT, "sues200_weather_test")
    if os.path.isdir(weather_test_dir):
        for class_dir_name in sorted(os.listdir(weather_test_dir)):
            class_path = os.path.join(weather_test_dir, class_dir_name)
            if not os.path.isdir(class_path):
                continue
            parts_split = class_dir_name.split("_")
            if len(parts_split) != 2:
                continue
            loc_id, alt = parts_split[0], parts_split[1]
            if loc_id not in loc_to_idx:
                continue
            label_idx = loc_to_idx[loc_id]
            for fname in sorted(os.listdir(class_path)):
                if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                stem = os.path.splitext(fname)[0]
                if '-' not in stem:
                    continue
                w = stem.rsplit('-', 1)[-1]
                if w != weather_name:
                    continue
                try:
                    img = transform(Image.open(os.path.join(class_path, fname)).convert('RGB'))
                    drone_imgs.append(img)
                    drone_lbls.append(label_idx)
                    drone_alt_idxs.append(CFG.ALT_TO_IDX.get(alt, 0))
                except Exception:
                    pass

    # 2. Clean TRAIN drone images (distractors)
    drone_dir = os.path.join(CFG.SUES_ROOT, CFG.DRONE_DIR)
    for loc in train_locs:
        for alt in CFG.ALTITUDES:
            alt_dir = os.path.join(drone_dir, loc, alt)
            if not os.path.isdir(alt_dir):
                continue
            imgs_in_dir = sorted(f for f in os.listdir(alt_dir)
                                 if f.lower().endswith(('.jpg', '.jpeg', '.png')))
            if not imgs_in_dir:
                continue
            try:
                img = transform(Image.open(os.path.join(alt_dir, imgs_in_dir[0])).convert('RGB'))
                drone_imgs.append(img)
                drone_lbls.append(-1000 - distractor_cnt)
                drone_alt_idxs.append(CFG.ALT_TO_IDX.get(alt, 0))
                distractor_cnt += 1
            except Exception:
                pass

    # Embed all drone images
    drone_feats = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(drone_imgs), CFG.BATCH_SIZE):
            bimgs = torch.stack(drone_imgs[i:i + CFG.BATCH_SIZE]).to(device)
            balts = torch.tensor(drone_alt_idxs[i:i + CFG.BATCH_SIZE], device=device)
            with torch.amp.autocast('cuda', enabled=CFG.USE_AMP):
                feats = model.extract_embedding(bimgs, alt_idx=balts)
            drone_feats.append(feats.cpu())

    drone_feats = torch.cat(drone_feats)
    drone_lbls  = torch.tensor(drone_lbls)
    test_cnt    = len(drone_feats) - distractor_cnt
    print(f"  Drone gallery S2D ({weather_name}): {len(drone_feats)} "
          f"(test={test_cnt}, distractors={distractor_cnt})")
    return drone_feats, drone_lbls


@torch.no_grad()
def evaluate_s2d(sat_feats, sat_labels, drone_feats, drone_labels):
    """
    Satellite→Drone retrieval.
    Queries  = test satellite images (sat_labels >= 0, 80 queries).
    Gallery  = weather-augmented test drones + clean distractor train drones.
    Each query has up to 4 matches (one per altitude of same location).
    """
    q_mask   = sat_labels >= 0
    q_feats  = sat_feats[q_mask]
    q_labels = sat_labels[q_mask]
    sim = q_feats @ drone_feats.T
    _, rank = sim.sort(1, descending=True)
    N = q_feats.size(0)
    r1 = r5 = r10 = ap = 0
    for i in range(N):
        matches = torch.where(drone_labels[rank[i]] == q_labels[i])[0]
        if len(matches) == 0:
            continue
        first = matches[0].item()
        if first < 1:  r1 += 1
        if first < 5:  r5 += 1
        if first < 10: r10 += 1
        ap += sum((j + 1) / (p.item() + 1) for j, p in enumerate(matches)) / len(matches)
    return {
        'R@1': r1 / N * 100, 'R@5': r5 / N * 100,
        'R@10': r10 / N * 100, 'mAP': ap / N * 100,
        'num_queries': N,
    }


def print_s2d_weather_table(all_s2d_results, normal_s2d):
    """Print Satellite→Drone per-weather results table."""
    print(f"\n{'='*85}")
    print(f"  WEATHER ROBUSTNESS — Satellite→Drone Retrieval (SUES-200 Test)")
    print(f"{'='*85}")
    print(f"  {'Weather':>12s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}  {'\u0394R@1':>7s}  {'\u0394mAP':>7s}  {'#Q':>5s}")
    print(f"  {'-'*78}")
    nr1 = normal_s2d['R@1']; nmap = normal_s2d['mAP']
    for w_name in CFG.WEATHER_NAMES:
        r = all_s2d_results[w_name]
        dr1 = r['R@1'] - nr1; dmap = r['mAP'] - nmap
        print(f"  {w_name:>12s}  {r['R@1']:6.2f}%  {r['R@5']:6.2f}%  {r['R@10']:6.2f}%  "
              f"{r['mAP']:6.2f}%  {dr1:+6.2f}%  {dmap:+6.2f}%  "
              f"{r['num_queries']:>5d}")
    print(f"  {'-'*78}")
    adv = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    avg_r1  = np.mean([all_s2d_results[w]['R@1']  for w in CFG.WEATHER_NAMES])
    avg_r5  = np.mean([all_s2d_results[w]['R@5']  for w in CFG.WEATHER_NAMES])
    avg_r10 = np.mean([all_s2d_results[w]['R@10'] for w in CFG.WEATHER_NAMES])
    avg_map = np.mean([all_s2d_results[w]['mAP']  for w in CFG.WEATHER_NAMES])
    avg_adv_r1  = np.mean([all_s2d_results[w]['R@1']  for w in adv])
    avg_adv_map = np.mean([all_s2d_results[w]['mAP']  for w in adv])
    print(f"  {'Avg(all)':>12s}  {avg_r1:6.2f}%  {avg_r5:6.2f}%  {avg_r10:6.2f}%  {avg_map:6.2f}%")
    print(f"  {'Avg(adverse)':>12s}  {avg_adv_r1:6.2f}%  {'':>7s}  {'':>7s}  {avg_adv_map:6.2f}%  "
          f"{avg_adv_r1-nr1:+5.2f}%  {avg_adv_map-nmap:+5.2f}%")
    print(f"{'='*85}")


def measure_model_complexity(model, device):
    sz = CFG.IMG_SIZE
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.eval()
    dummy = torch.randn(1, 3, sz, sz).to(device)
    gflops = None
    try:
        from thop import profile as thop_profile
        class _Wrap(nn.Module):
            def __init__(self, m): super().__init__(); self.m = m
            def forward(self, x): return self.m.extract_embedding(x)
        macs, _ = thop_profile(_Wrap(model).to(device), inputs=(dummy,), verbose=False)
        gflops = macs / 1e9
    except Exception:
        pass
    model.eval()
    with torch.no_grad():
        for _ in range(10): model.extract_embedding(dummy)
        if device.type == 'cuda': torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(100): model.extract_embedding(dummy)
        if device.type == 'cuda': torch.cuda.synchronize()
        ms_query = (time.time() - t0) / 100 * 1000
    return {
        'total_params_M': total_params / 1e6,
        'trainable_params_M': trainable_params / 1e6,
        'gflops': gflops,
        'ms_per_query': ms_query,
        'img_size': sz,
    }


# =============================================================================
# PRINT / TABLE FUNCTIONS
# =============================================================================
def print_complexity(stats):
    print(f"\n  ┌─ Model Complexity ─────────────────────────────┐")
    print(f"  │  Total params:     {stats['total_params_M']:>8.2f}M                │")
    print(f"  │  Trainable params: {stats['trainable_params_M']:>8.2f}M                │")
    if stats['gflops'] is not None:
        print(f"  │  GFLOPs:           {stats['gflops']:>8.2f}                 │")
    else:
        print(f"  │  GFLOPs:           N/A (pip install thop)    │")
    print(f"  │  Inference:        {stats['ms_per_query']:>8.1f} ms/query         │")
    print(f"  │  Image size:       {stats['img_size']:>4d}×{stats['img_size']:<4d}               │")
    print(f"  └─────────────────────────────────────────────────┘")


def print_weather_table(all_results, normal_results):
    print(f"\n{'='*85}")
    print(f"  WEATHER ROBUSTNESS — Drone→Satellite Retrieval (SUES-200 Test)")
    print(f"{'='*85}")
    print(f"  {'Weather':>12s}  {'R@1':>7s}  {'R@5':>7s}  {'R@10':>7s}  {'mAP':>7s}  {'ΔR@1':>7s}  {'ΔmAP':>7s}  {'#Q':>5s}")
    print(f"  {'-'*78}")
    nr1 = normal_results['R@1']; nmap = normal_results['mAP']
    for w_name in CFG.WEATHER_NAMES:
        r = all_results[w_name]['overall']
        dr1 = r['R@1'] - nr1; dmap = r['mAP'] - nmap
        sign_r1 = '+' if dr1 >= 0 else ''; sign_map = '+' if dmap >= 0 else ''
        print(f"  {w_name:>12s}  {r['R@1']:6.2f}%  {r['R@5']:6.2f}%  {r['R@10']:6.2f}%  "
              f"{r['mAP']:6.2f}%  {sign_r1}{dr1:5.2f}%  {sign_map}{dmap:5.2f}%  "
              f"{r['num_queries']:>5d}")
    print(f"  {'-'*78}")
    avg_r1  = np.mean([all_results[w]['overall']['R@1']  for w in CFG.WEATHER_NAMES])
    avg_r5  = np.mean([all_results[w]['overall']['R@5']  for w in CFG.WEATHER_NAMES])
    avg_r10 = np.mean([all_results[w]['overall']['R@10'] for w in CFG.WEATHER_NAMES])
    avg_map = np.mean([all_results[w]['overall']['mAP']  for w in CFG.WEATHER_NAMES])
    adv_weathers = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    avg_adv_r1  = np.mean([all_results[w]['overall']['R@1'] for w in adv_weathers])
    avg_adv_map = np.mean([all_results[w]['overall']['mAP'] for w in adv_weathers])
    print(f"  {'Avg(all)':>12s}  {avg_r1:6.2f}%  {avg_r5:6.2f}%  {avg_r10:6.2f}%  {avg_map:6.2f}%")
    print(f"  {'Avg(adverse)':>12s}  {avg_adv_r1:6.2f}%  {'':>7s}  {'':>7s}  {avg_adv_map:6.2f}%  "
          f"{avg_adv_r1-nr1:+5.2f}%  {avg_adv_map-nmap:+5.2f}%")
    print(f"{'='*85}")


def print_cross_table(all_results):
    print(f"\n{'='*75}")
    print(f"  R@1 (%) — Weather × Altitude")
    print(f"{'='*75}")
    header = f"  {'Weather':>12s}"
    for alt in CFG.ALTITUDES:
        header += f"  {alt+'m':>7s}"
    header += f"  {'Avg':>7s}"
    print(header)
    print(f"  {'-'*60}")
    for w in CFG.WEATHER_NAMES:
        row = f"  {w:>12s}"
        vals = []
        for alt in CFG.ALTITUDES:
            v = all_results[w]['per_alt'].get(int(alt), {}).get('R@1', 0)
            row += f"  {v:6.2f}%"; vals.append(v)
        row += f"  {np.mean(vals):6.2f}%"
        print(row)
    print(f"{'='*75}")


# =============================================================================
# KNOWN RESULTS FROM PREVIOUS EXPERIMENTS
# =============================================================================
EXP49_RESULTS = {   # zero-shot (EXP35 checkpoint, old eval script)
    'normal':    {'R@1': 95.94, 'mAP': 97.64},
    'fog':       {'R@1': 86.88, 'mAP': 92.68},
    'rain':      {'R@1': 43.75, 'mAP': 56.99},
    'snow':      {'R@1': 40.62, 'mAP': 54.48},
    'dark':      {'R@1': 66.88, 'mAP': 76.74},
    'light':     {'R@1': 83.44, 'mAP': 89.97},
    'fog_rain':  {'R@1': 42.19, 'mAP': 54.92},
    'fog_snow':  {'R@1': 22.19, 'mAP': 35.01},
    'rain_snow': {'R@1': 34.69, 'mAP': 46.85},
    'wind':      {'R@1': 81.25, 'mAP': 88.76},
}
EXP50_RESULTS = {   # finetune + pre-generated weather
    'normal':    {'R@1': 94.38, 'mAP': 96.78},
    'fog':       {'R@1': 90.94, 'mAP': 95.00},
    'rain':      {'R@1': 85.31, 'mAP': 91.07},
    'snow':      {'R@1': 88.12, 'mAP': 92.86},
    'dark':      {'R@1': 80.00, 'mAP': 86.58},
    'light':     {'R@1': 86.25, 'mAP': 91.42},
    'fog_rain':  {'R@1': 78.12, 'mAP': 86.61},
    'fog_snow':  {'R@1': 73.44, 'mAP': 82.36},
    'rain_snow': {'R@1': 84.69, 'mAP': 90.36},
    'wind':      {'R@1': 85.62, 'mAP': 90.78},
}
EXP51_RESULTS = {   # scratch + pre-generated weather
    'normal':    {'R@1': 90.94, 'mAP': 95.18},
    'fog':       {'R@1': 89.69, 'mAP': 94.37},
    'rain':      {'R@1': 86.56, 'mAP': 92.09},
    'snow':      {'R@1': 89.69, 'mAP': 93.66},
    'dark':      {'R@1': 80.31, 'mAP': 87.74},
    'light':     {'R@1': 86.88, 'mAP': 91.98},
    'fog_rain':  {'R@1': 81.25, 'mAP': 88.60},
    'fog_snow':  {'R@1': 75.00, 'mAP': 84.63},
    'rain_snow': {'R@1': 86.25, 'mAP': 91.27},
    'wind':      {'R@1': 88.12, 'mAP': 92.98},
}
# Fill these in after EXP52/53 complete (set to None to skip in comparison)
EXP52_RESULTS = {   # finetune + online aug (WeatherPrompt-style)
    'normal':    {'R@1': 95.00, 'mAP': 96.71},
    'fog':       {'R@1': 90.00, 'mAP': 94.30},
    'rain':      {'R@1': 89.69, 'mAP': 93.05},
    'snow':      {'R@1': 87.50, 'mAP': 92.05},
    'dark':      {'R@1': 78.75, 'mAP': 85.07},
    'light':     {'R@1': 84.69, 'mAP': 90.14},
    'fog_rain':  {'R@1': 81.88, 'mAP': 88.64},
    'fog_snow':  {'R@1': 69.38, 'mAP': 79.21},
    'rain_snow': {'R@1': 84.69, 'mAP': 90.06},
    'wind':      {'R@1': 88.75, 'mAP': 92.93},
}
EXP53_RESULTS = {   # scratch  + online aug (WeatherPrompt-style)
    'normal':    {'R@1': 92.81, 'mAP': 95.81},
    'fog':       {'R@1': 91.88, 'mAP': 94.94},
    'rain':      {'R@1': 90.31, 'mAP': 94.18},
    'snow':      {'R@1': 87.19, 'mAP': 92.19},
    'dark':      {'R@1': 82.50, 'mAP': 87.90},
    'light':     {'R@1': 86.25, 'mAP': 91.25},
    'fog_rain':  {'R@1': 85.62, 'mAP': 91.19},
    'fog_snow':  {'R@1': 79.38, 'mAP': 86.89},
    'rain_snow': {'R@1': 84.69, 'mAP': 90.76},
    'wind':      {'R@1': 87.50, 'mAP': 92.33},
}


def print_comparison(all_results):
    """Compare EXP54 against all known prior experiments."""
    # Build list of experiments that have real data
    exps = [
        ("EXP49", EXP49_RESULTS, "zero-shot"),
        ("EXP50", EXP50_RESULTS, "ft-pregen"),
        ("EXP51", EXP51_RESULTS, "scratch-pregen"),
    ]
    if EXP52_RESULTS is not None:
        exps.append(("EXP52", EXP52_RESULTS, "ft-online"))
    if EXP53_RESULTS is not None:
        exps.append(("EXP53", EXP53_RESULTS, "scratch-online"))

    exp_name_short = CFG.EXPERIMENT_NAME.split("_")[-1]   # e.g. "EXP35"
    col_width = max(8, len(exp_name_short) + 2)

    header  = f"\n{'='*120}\n"
    header += f"  COMPARISON (R@1%): "
    header += " vs ".join(f"{e[0]}({e[2]})" for e in exps)
    header += f" vs {CFG.EXPERIMENT_NAME}\n"
    header += f"{'='*120}"
    print(header)

    # Column headers
    row = f"  {'Weather':>12s}"
    for exp_id, _, _ in exps:
        row += f"  {exp_id:>8s}"
    row += f"  {exp_name_short:>8s}"
    print(row)
    print(f"  {'-'*80}")

    for w in CFG.WEATHER_NAMES:
        row = f"  {w:>12s}"
        for _, res, _ in exps:
            row += f"  {res[w]['R@1']:7.2f}%"
        e54_r1 = all_results[w]['overall']['R@1']
        row += f"  {e54_r1:7.2f}%"
        print(row)

    print(f"  {'-'*80}")
    adv = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    row = f"  {'Avg(adverse)':>12s}"
    for _, res, _ in exps:
        row += f"  {np.mean([res[w]['R@1'] for w in adv]):7.2f}%"
    row += f"  {np.mean([all_results[w]['overall']['R@1'] for w in adv]):7.2f}%"
    print(row)

    row = f"  {'Avg(all)':>12s}"
    for _, res, _ in exps:
        row += f"  {np.mean([res[w]['R@1'] for w in CFG.WEATHER_NAMES]):7.2f}%"
    row += f"  {np.mean([all_results[w]['overall']['R@1'] for w in CFG.WEATHER_NAMES]):7.2f}%"
    print(row)
    print(f"{'='*120}")


def generate_latex_table(all_results, normal_results):
    nr1 = normal_results['R@1']; nmap = normal_results['mAP']
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        rf"\caption{{Zero-shot weather robustness evaluation ({CFG.EXPERIMENT_NAME}).}}",
        rf"\label{{tab:{CFG.EXPERIMENT_NAME.lower()}}}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{l|cccc|cc}",
        r"\toprule",
        r"Weather & R@1 & R@5 & R@10 & mAP & $\Delta$R@1 & $\Delta$mAP \\",
        r"\midrule",
    ]
    for w in CFG.WEATHER_NAMES:
        r = all_results[w]['overall']
        dr1 = r['R@1'] - nr1; dmap = r['mAP'] - nmap
        label = w.replace('_', r'\_')
        if w == 'normal':
            lines.append(f"{label} & {r['R@1']:.2f} & {r['R@5']:.2f} & {r['R@10']:.2f} & "
                         f"{r['mAP']:.2f} & --- & --- \\\\")
        else:
            lines.append(f"{label} & {r['R@1']:.2f} & {r['R@5']:.2f} & {r['R@10']:.2f} & "
                         f"{r['mAP']:.2f} & {dr1:+.2f} & {dmap:+.2f} \\\\")
    adv = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    avg_r1  = np.mean([all_results[w]['overall']['R@1']  for w in adv])
    avg_r5  = np.mean([all_results[w]['overall']['R@5']  for w in adv])
    avg_r10 = np.mean([all_results[w]['overall']['R@10'] for w in adv])
    avg_map = np.mean([all_results[w]['overall']['mAP']  for w in adv])
    lines += [
        r"\midrule",
        f"Avg(adverse) & {avg_r1:.2f} & {avg_r5:.2f} & {avg_r10:.2f} & "
        f"{avg_map:.2f} & {avg_r1-nr1:+.2f} & {avg_map-nmap:+.2f} \\\\",
        r"\bottomrule",
        r"\end{tabular}}",
        r"\end{table}",
    ]
    return '\n'.join(lines)


# =============================================================================
# MAIN
# =============================================================================
def main():
    import random
    random.seed(CFG.SEED); np.random.seed(CFG.SEED)
    torch.manual_seed(CFG.SEED); torch.cuda.manual_seed_all(CFG.SEED)
    torch.backends.cudnn.benchmark = True

    os.makedirs(CFG.OUTPUT_DIR, exist_ok=True)
    start_time = datetime.datetime.now()

    print("=" * 75)
    print(f"  {CFG.EXPERIMENT_NAME}")
    print(f"  Zero-Shot Weather Robustness Evaluation (no training)")
    print(f"  Checkpoint: {CFG.CHECKPOINT}")
    print(f"  SUES-200:   {CFG.SUES_ROOT}")
    print(f"  Device:     {DEVICE}")
    print("=" * 75)

    # ---- 1. Load Model ----
    print("\n[1/3] Loading model …")
    if not os.path.exists(CFG.CHECKPOINT):
        print(f"  ERROR: Checkpoint not found at {CFG.CHECKPOINT}")
        return

    model = SPDGeoDPEAMARModel(CFG.NUM_CLASSES).to(DEVICE)
    ckpt = torch.load(CFG.CHECKPOINT, map_location=DEVICE, weights_only=False)
    state_dict = ckpt.get('model_state_dict', ckpt)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys  ({len(missing)}): {missing[:5]} …")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:5]} …")

    ckpt_epoch = ckpt.get('epoch', '?')
    ckpt_r1    = ckpt.get('metrics', {}).get('R@1', ckpt.get('best_r1', '?'))
    print(f"  Loaded  epoch={ckpt_epoch}  pretrained R@1={ckpt_r1}")

    model.eval()

    complexity = measure_model_complexity(model, DEVICE)
    print_complexity(complexity)

    # ---- 2. Build Satellite Gallery ----
    print("\n[2/3] Building satellite gallery …")
    tf = get_test_transform()
    sat_feats, sat_labels = build_satellite_gallery(model, DEVICE, tf)

    # ---- 3. Evaluate All Weather Conditions ----
    print("\n[3/3] Evaluating all 10 weather conditions …")
    all_results = OrderedDict()
    for w_name in CFG.WEATHER_NAMES:
        query_ds = WeatherQueryDataset(CFG.WEATHER_ROOT, w_name, transform=tf)
        if len(query_ds) == 0:
            print(f"  [SKIP] {w_name}: no images found in {CFG.WEATHER_ROOT}")
            all_results[w_name] = {
                'overall': {'R@1': 0, 'R@5': 0, 'R@10': 0, 'mAP': 0, 'num_queries': 0},
                'per_alt': {}
            }
            continue
        overall, per_alt = evaluate_retrieval(model, query_ds, sat_feats, sat_labels, DEVICE)
        all_results[w_name] = {'overall': overall, 'per_alt': per_alt}
        print(f"  {w_name:>12s}  R@1={overall['R@1']:.2f}%  R@5={overall['R@5']:.2f}%  "
              f"R@10={overall['R@10']:.2f}%  mAP={overall['mAP']:.2f}%  ({len(query_ds)} queries)")

    # ---- Print Tables ----
    normal_results = all_results['normal']['overall']
    print_weather_table(all_results, normal_results)
    print_cross_table(all_results)
    print_comparison(all_results)

    # ---- S2D Evaluation (Satellite→Drone) ----
    print(f"\n{'='*75}")
    print(f"  [S2D] Satellite→Drone Evaluation …")
    all_s2d_results = OrderedDict()
    for w_name in CFG.WEATHER_NAMES:
        drone_feats_s2d, drone_lbls_s2d = build_drone_gallery_s2d(
            model, w_name, DEVICE, tf)
        s2d_res = evaluate_s2d(sat_feats, sat_labels, drone_feats_s2d, drone_lbls_s2d)
        all_s2d_results[w_name] = s2d_res
        print(f"    {w_name:>12s}  R@1={s2d_res['R@1']:.2f}%  R@5={s2d_res['R@5']:.2f}%  "
              f"R@10={s2d_res['R@10']:.2f}%  mAP={s2d_res['mAP']:.2f}%")
        del drone_feats_s2d, drone_lbls_s2d
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    normal_s2d = all_s2d_results['normal']
    print_s2d_weather_table(all_s2d_results, normal_s2d)

    # LaTeX
    latex = generate_latex_table(all_results, normal_results)
    print(f"\n{'='*75}\n  LaTeX Table:\n{'='*75}")
    print(latex)

    # ---- Save JSON ----
    end_time = datetime.datetime.now()
    adv_weathers = [w for w in CFG.WEATHER_NAMES if w != 'normal']
    summary = {
        'experiment': CFG.EXPERIMENT_NAME,
        'checkpoint': CFG.CHECKPOINT,
        'ckpt_epoch': ckpt_epoch,
        'dataset': 'SUES-200',
        'direction': 'drone → satellite (D2S) + satellite → drone (S2D) — zero-shot inference',
        'timestamp': {
            'start': start_time.isoformat(),
            'end': end_time.isoformat(),
            'duration_s': (end_time - start_time).total_seconds(),
        },
        'model_complexity': complexity,
        'satellite_gallery_size': int(len(sat_feats)),
        'weather_results': {
            w: {
                'overall': all_results[w]['overall'],
                'per_altitude': {str(k): v for k, v in all_results[w]['per_alt'].items()},
            }
            for w in CFG.WEATHER_NAMES
        },
        'summary_metrics': {
            'normal_R@1': normal_results['R@1'],
            'normal_mAP': normal_results['mAP'],
            'avg_all_R@1':     float(np.mean([all_results[w]['overall']['R@1'] for w in CFG.WEATHER_NAMES])),
            'avg_all_mAP':     float(np.mean([all_results[w]['overall']['mAP'] for w in CFG.WEATHER_NAMES])),
            'avg_adverse_R@1': float(np.mean([all_results[w]['overall']['R@1'] for w in adv_weathers])),
            'avg_adverse_mAP': float(np.mean([all_results[w]['overall']['mAP'] for w in adv_weathers])),
            'vs_exp49': {w: float(all_results[w]['overall']['R@1'] - EXP49_RESULTS[w]['R@1'])
                         for w in CFG.WEATHER_NAMES},
            'vs_exp50': {w: float(all_results[w]['overall']['R@1'] - EXP50_RESULTS[w]['R@1'])
                         for w in CFG.WEATHER_NAMES},
        },
        'latex_table': latex,
        's2d_weather_results': {w: all_s2d_results[w] for w in CFG.WEATHER_NAMES},
        's2d_summary_metrics': {
            'normal_R@1': normal_s2d['R@1'],
            'normal_mAP': normal_s2d['mAP'],
            'avg_all_R@1':     float(np.mean([all_s2d_results[w]['R@1'] for w in CFG.WEATHER_NAMES])),
            'avg_all_mAP':     float(np.mean([all_s2d_results[w]['mAP'] for w in CFG.WEATHER_NAMES])),
            'avg_adverse_R@1': float(np.mean([all_s2d_results[w]['R@1'] for w in adv_weathers])),
            'avg_adverse_mAP': float(np.mean([all_s2d_results[w]['mAP'] for w in adv_weathers])),
        },
    }

    json_name = CFG.EXPERIMENT_NAME.lower() + ".json"
    json_path = os.path.join(CFG.OUTPUT_DIR, json_name)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {json_path}")

    # ---- Final Summary ----
    sm = summary['summary_metrics']
    print(f"\n{'='*75}")
    print(f"  {CFG.EXPERIMENT_NAME} COMPLETE  (zero-shot, no training)")
    print(f"{'='*75}")
    print(f"  Checkpoint:         epoch={ckpt_epoch}")
    print(f"  Normal:             R@1={sm['normal_R@1']:.2f}%  mAP={sm['normal_mAP']:.2f}%")
    print(f"  Avg (all weather):  R@1={sm['avg_all_R@1']:.2f}%  mAP={sm['avg_all_mAP']:.2f}%")
    print(f"  Avg (adverse):      R@1={sm['avg_adverse_R@1']:.2f}%  mAP={sm['avg_adverse_mAP']:.2f}%")
    e49_adv = np.mean([EXP49_RESULTS[w]['R@1'] for w in adv_weathers])
    print(f"\n  vs EXP49 (zero-shot, old eval)  avg adverse R@1: "
          f"{e49_adv:.2f}% → {sm['avg_adverse_R@1']:.2f}% "
          f"({sm['avg_adverse_R@1'] - e49_adv:+.2f}%)")
    e50_adv = np.mean([EXP50_RESULTS[w]['R@1'] for w in adv_weathers])
    print(f"  vs EXP50 (ft+pregen)            avg adverse R@1: "
          f"{e50_adv:.2f}% → {sm['avg_adverse_R@1']:.2f}% "
          f"({sm['avg_adverse_R@1'] - e50_adv:+.2f}%)")
    print(f"  Eval duration:      {(end_time - start_time).total_seconds():.1f}s")
    s2d_sm = summary['s2d_summary_metrics']
    print(f"\n  S2D — Satellite→Drone:")
    print(f"  Normal:             R@1={normal_s2d['R@1']:.2f}%  mAP={normal_s2d['mAP']:.2f}%")
    print(f"  Avg (all weather):  R@1={s2d_sm['avg_all_R@1']:.2f}%  mAP={s2d_sm['avg_all_mAP']:.2f}%")
    print(f"  Avg (adverse):      R@1={s2d_sm['avg_adverse_R@1']:.2f}%  mAP={s2d_sm['avg_adverse_mAP']:.2f}%")
    print(f"{'='*75}")


main()
