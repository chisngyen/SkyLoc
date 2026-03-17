#!/usr/bin/env python3
"""
exp05_standalone_kaggle.py
==========================
Fully self-contained UAV-AVL localization baseline.
Copy this single file to a Kaggle notebook cell and run — no other files needed.

Pipeline
--------
  Step 1 : SPDGeo retrieval   (DINOv2-S + part discovery + fusion gate)
  Step 2 : RoMa dense matching (loaded directly from HuggingFace via `romatch`)
  Step 3 : PnP + DSM → GPS

Kaggle datasets required
------------------------
  • hunhtrungkit/uav-avl  →  /kaggle/input/datasets/hunhtrungkit/uav-avl
  • SPDGeo checkpoint     →  set SPDGEO_CKPT (path or env var)

Internet must be ON in the Kaggle notebook (for romatch + DINOv2 auto-download).
"""

# =============================================================================
# 0.  INSTALL DEPENDENCIES  (runs first, before any other import)
# =============================================================================
import subprocess, sys

def _pip(*pkgs):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *pkgs])

print("[INSTALL] Installing packages …")
_pip("romatch", "pyproj", "scikit-image", "tqdm", "Pillow", "opencv-python-headless")
print("[INSTALL] Done.\n")

# =============================================================================
# 1.  STANDARD IMPORTS
# =============================================================================
import glob, json, math, os, pickle, time, warnings
from dataclasses import dataclass
from math import sqrt
from typing import Optional

import cv2
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torchvision import transforms

from pyproj import Transformer
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# RoMa matching — auto-downloads weights on first call (from GitHub / HuggingFace)
from romatch import roma_outdoor

# =============================================================================
# 2.  CONFIGURATION  (edit these for your Kaggle setup)
# =============================================================================

# ── Paths ────────────────────────────────────────────────────────────────────
_IN = "/kaggle/input"
DATASET_BASE = f"{_IN}/datasets/hunhtrungkit/uav-avl"
SAVE_DIR     = "/kaggle/working/results"

# SPDGeo checkpoint — set env var SPDGEO_CKPT or edit the fallback path below
SPDGEO_CKPT = os.environ.get(
    "SPDGEO_CKPT",
    (f"{_IN}/models/minh2duy/"
     "exp35-spdgeo-dpea-m-unfreeze-blocks-4-6/pytorch/default/1/exp35_dpea_ga_best.pth"),
)

# ── Runtime ──────────────────────────────────────────────────────────────────
DEVICE       = "cuda" if torch.cuda.is_available() else "cpu"
REF_TYPE     = "HIGH"       # HIGH (aerial) | LOW (satellite)
POSE_PRIORI  = "yp"         # yp (yaw+pitch prior) | p | unknown
STRATEGY     = "Topn_opt"   # Top1 | Topn_opt | Inliners
RESIZE_RATIO = 0.2           # downscale during matching (saves VRAM)
TEST_INTERVAL = 1            # use every Nth UAV image; raise to speed up

# ── Retrieval ────────────────────────────────────────────────────────────────
RETRIEVAL_SIZE       = 384   # resize gallery tiles / query before embedding
RETRIEVAL_COVER      = 50    # % overlap between adjacent gallery tiles
RETRIEVAL_TOPN       = 5     # re-rank this many candidates after matching
RETRIEVAL_BATCH_SIZE = 128
FEAT_NORM            = True
SHOW_RESULT          = False  # save retrieval visualisation PNG?

# ── Region: QZ_Town  (inlined from Regions_params/QZ_Town.yaml) ──────────────
REGION     = "QZ_Town"
UTM_SYSTEM = "50N"
UAV_PLACES = ["QZ_SongCity", "Qingzhou_3_2", "QingZhou_2024"]

# Map file names (relative to Data/Reference_map/QZ_Town/)
REF_FNAME  = {"HIGH": "result_roi.tif",    "LOW": "satellite_roi.tif"}
DSM_FNAME  = {"HIGH": "dsm_roi.tif",       "LOW": "QZ_town_DSM30_reproject_resize_roi.tif"}

# Aerial (HIGH) reference map parameters
HIGH_REF_resolution = 0.0609957
HIGH_REF_initialX   = 631556.341       # UTM Easting  of pixel (0,0)
HIGH_REF_initialY   = 4061562.708      # UTM Northing of pixel (0,0)
HIGH_DSM_resolution = 0.937272
HIGH_REF_COORDINATE = [2143.3682, 2301.1853]   # [col, row] of alignment point in REF
HIGH_DSM_COORDINATE = [140.1837,  150.3175]    # [col, row] of alignment point in DSM

# Satellite (LOW) reference map parameters
LOW_REF_resolution  = 0.26004821
LOW_REF_initialX    = 631556.196
LOW_REF_initialY    = 4061567.017
LOW_DSM_resolution  = 30.0
LOW_REF_COORDINATE  = [408.4728, 431.1906]
LOW_DSM_COORDINATE  = [3.8365,   4.2391]


def _cfg(key: str):
    """Quick accessor for the active REF_TYPE config values."""
    return globals()[f"{REF_TYPE}_{key}"]


# =============================================================================
# 3.  SPDGeo MODEL  (DINOv2-S backbone + altitude-aware part discovery + gate)
# =============================================================================

@dataclass
class SPDGeoConfig:
    IMG_SIZE:        int   = 336
    EMBED_DIM:       int   = 512
    PART_DIM:        int   = 256
    N_PARTS:         int   = 8
    CLUSTER_TEMP:    float = 0.07
    NUM_ALTITUDES:   int   = 4
    UNFREEZE_BLOCKS: int   = 6


class _DINOv2Backbone(nn.Module):
    def __init__(self, unfreeze_blocks: int = 6):
        super().__init__()
        self.model = torch.hub.load(
            "facebookresearch/dinov2", "dinov2_vits14", pretrained=True
        )
        self.patch_size = 14
        for p in self.model.parameters():
            p.requires_grad = False
        if unfreeze_blocks > 0:
            for blk in self.model.blocks[-unfreeze_blocks:]:
                for p in blk.parameters():
                    p.requires_grad = True
            for p in self.model.norm.parameters():
                p.requires_grad = True

    def forward(self, x):
        f = self.model.forward_features(x)
        H = x.shape[2] // self.patch_size
        W = x.shape[3] // self.patch_size
        return f["x_norm_patchtokens"], f["x_norm_clstoken"], (H, W)


class _AltitudeFiLM(nn.Module):
    def __init__(self, n: int = 4, d: int = 256):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(n, d))
        self.beta  = nn.Parameter(torch.zeros(n, d))

    def forward(self, feat, alt_idx=None):
        if alt_idx is None:
            g, b = self.gamma.mean(0, keepdim=True), self.beta.mean(0, keepdim=True)
            return feat * g.unsqueeze(0) + b.unsqueeze(0)
        return feat * self.gamma[alt_idx].unsqueeze(1) + self.beta[alt_idx].unsqueeze(1)


class _PartDiscovery(nn.Module):
    def __init__(self, feat_dim=384, n_parts=8, part_dim=256,
                 temperature=0.07, num_altitudes=4):
        super().__init__()
        self.n_parts = n_parts
        self.temperature = temperature
        self.feat_proj = nn.Sequential(
            nn.Linear(feat_dim, part_dim), nn.LayerNorm(part_dim), nn.GELU()
        )
        self.film = _AltitudeFiLM(num_altitudes, part_dim)
        self.prototypes = nn.Parameter(torch.randn(n_parts, part_dim) * 0.02)
        self.refine = nn.Sequential(
            nn.LayerNorm(part_dim),
            nn.Linear(part_dim, part_dim * 2), nn.GELU(),
            nn.Linear(part_dim * 2, part_dim),
        )
        self.salience = nn.Sequential(
            nn.Linear(part_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, patches, hw, alt_idx=None):
        feat = self.film(self.feat_proj(patches), alt_idx)
        sim  = torch.einsum("bnd,kd->bnk",
                            F.normalize(feat, dim=-1),
                            F.normalize(self.prototypes, dim=-1)) / self.temperature
        assign = F.softmax(sim, dim=-1).transpose(1, 2)      # (B, K, N)
        mass   = assign.sum(-1, keepdim=True).clamp(min=1e-6)
        part   = torch.bmm(assign, feat) / mass               # (B, K, D)
        part   = part + self.refine(part)
        return {"part_features": part, "salience": self.salience(part).squeeze(-1)}


class _PartPooling(nn.Module):
    def __init__(self, part_dim=256, embed_dim=512):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(part_dim, part_dim // 2), nn.Tanh(), nn.Linear(part_dim // 2, 1)
        )
        self.proj = nn.Sequential(
            nn.Linear(part_dim * 3, embed_dim), nn.LayerNorm(embed_dim), nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, parts, salience=None):
        aw = self.attn(parts)
        if salience is not None:
            aw = aw + salience.unsqueeze(-1).log().clamp(-10)
        aw = F.softmax(aw, dim=1)
        combined = torch.cat(
            [(aw * parts).sum(1), parts.mean(1), parts.max(1)[0]], dim=-1
        )
        return F.normalize(self.proj(combined), dim=-1)


class _FusionGate(nn.Module):
    def __init__(self, embed_dim=512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2), nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
        )
        nn.init.constant_(self.gate[-1].bias, 0.85)

    def forward(self, part_emb, cls_emb):
        alpha = torch.sigmoid(self.gate(torch.cat([part_emb, cls_emb], dim=-1)))
        return F.normalize(alpha * part_emb + (1 - alpha) * cls_emb, dim=-1)


class SPDGeoModel(nn.Module):
    def __init__(self, cfg: SPDGeoConfig = SPDGeoConfig()):
        super().__init__()
        self.cfg       = cfg
        self.backbone  = _DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc = _PartDiscovery(384, cfg.N_PARTS, cfg.PART_DIM,
                                        cfg.CLUSTER_TEMP, cfg.NUM_ALTITUDES)
        self.pool      = _PartPooling(cfg.PART_DIM, cfg.EMBED_DIM)
        self.gate      = _FusionGate(cfg.EMBED_DIM)
        self.cls_proj  = nn.Sequential(
            nn.Linear(384, cfg.EMBED_DIM),
            nn.BatchNorm1d(cfg.EMBED_DIM),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, alt_idx=None):
        patches, cls_tok, hw = self.backbone(x)
        parts    = self.part_disc(patches, hw, alt_idx)
        part_emb = self.pool(parts["part_features"], parts["salience"])
        cls_emb  = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.gate(part_emb, cls_emb)


def _spdgeo_transform(img_size: int = 336):
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def load_spdgeo(checkpoint_path: Optional[str] = None, device: str = "cuda"):
    """Load SPDGeo model + transform.  Falls back to random init if ckpt not found."""
    cfg   = SPDGeoConfig()
    model = SPDGeoModel(cfg)
    ckpt  = checkpoint_path or os.environ.get("SPDGEO_CKPT")
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[SPDGeo] Loaded: {ckpt}")
        if missing:     print(f"[SPDGeo] Missing   : {len(missing)} keys")
        if unexpected:  print(f"[SPDGeo] Unexpected: {len(unexpected)} keys")
    else:
        if ckpt:
            print(f"[SPDGeo][WARN] Checkpoint not found ({ckpt}), using random init")
        else:
            print("[SPDGeo][WARN] No checkpoint set, using random init")
    model.to(device).eval()
    return model, _spdgeo_transform(cfg.IMG_SIZE)


# =============================================================================
# 4.  COORDINATE UTILITIES
# =============================================================================

def _utm_zone(utm_sys: str = UTM_SYSTEM) -> str:
    return utm_sys[:2]      # "50N" → "50"


def deg2utm(lon: float, lat: float) -> tuple:
    """WGS-84 (lon, lat) → UTM (easting, northing)."""
    epsg = f"epsg:326{_utm_zone()}"
    t = Transformer.from_crs("epsg:4326", epsg, always_xy=True)
    return t.transform(lon, lat)


def utm2deg(x: float, y: float) -> tuple:
    """UTM (easting, northing) → WGS-84 (lon, lat)."""
    epsg = f"epsg:326{_utm_zone()}"
    t = Transformer.from_crs(epsg, "epsg:4326", always_xy=True)
    return t.transform(x, y)


def _rvec2rot(rvec):
    return cv2.Rodrigues(rvec)[0]


def _rot2euler(dcm):
    r1 = np.arctan2(-dcm[1, 0], dcm[1, 1])
    r2 = np.arcsin(np.clip(dcm[1, 2], -1, 1))
    r3 = np.arctan2(-dcm[0, 2], dcm[2, 2])
    return np.array([-r1 - np.pi, -r2, r3 + np.pi]) * 180.0 / np.pi


def dump_rotate_image(img, degree: float):
    """Rotate image by *degree* and return (rotated_img, affine_matrix)."""
    rad  = degree / 180.0 * np.pi
    h, w = img.shape[:2]
    hNew = int(w * abs(np.sin(rad)) + h * abs(np.cos(rad)))
    wNew = int(h * abs(np.sin(rad)) + w * abs(np.cos(rad)))
    M    = cv2.getRotationMatrix2D((w // 2, h // 2), degree, 1.0)
    M[0, 2] += (wNew - w) // 2
    M[1, 2] += (hNew - h) // 2
    return cv2.warpAffine(img, M, (wNew, hNew), borderValue=(0, 0, 0)), M


# =============================================================================
# 5.  CAMERA & POSE UTILITIES
# =============================================================================

def compute_camera_matrix(meta: dict) -> np.ndarray:
    W, H  = meta["width"], meta["height"]
    diag  = sqrt(W ** 2 + H ** 2)
    f_px  = meta["focal_len"] / (meta["cam_size"] / diag)
    return np.array([[f_px, 0, W / 2.0],
                     [0, f_px, H / 2.0],
                     [0,    0,       1]])


def resolution_and_size(meta: dict, pose_priori: str = "yp"):
    """Return (drone_ground_resolution_m, square_crop_size_px)."""
    W, H = meta["width"], meta["height"]
    pitch = -30.0 if pose_priori == "unknown" else meta["pitch"]
    res   = (2 * meta["rel_alt"] / np.sin(-np.pi * pitch / 180.0)
             * meta["cam_size"] / 2.0 / meta["focal_len"] / sqrt(W ** 2 + H ** 2))
    side  = min(W, H)
    return res, np.array([side, side])


def estimate_drone_pose(utm_x, utm_y, dsm_z, match_pts_px, K):
    """
    Solve PnP (P3P + RANSAC) and return:
      BLH dict, cam_angle, n_inliers, inliers_array
    """
    pts3d = np.column_stack((utm_x, utm_y, dsm_z)).astype(np.float64)
    pts2d = np.array(match_pts_px, dtype=np.float64)
    dist  = np.zeros(4, dtype=np.float32)
    ok, rvec, tvec, inliers = cv2.solvePnPRansac(
        pts3d, pts2d, K, dist, flags=cv2.SOLVEPNP_P3P
    )
    if not ok or inliers is None:
        return {"B": None, "L": None, "H": None}, None, 0, []
    R  = _rvec2rot(rvec)
    X0 = -R.T @ tvec
    lon, lat = utm2deg(float(X0[0]), float(X0[1]))
    Rx90 = np.array([[1, 0, 0],
                     [0,  np.cos(-np.pi / 2), np.sin(-np.pi / 2)],
                     [0, -np.sin(-np.pi / 2), np.cos(-np.pi / 2)]])
    return ({"B": float(lat), "L": float(lon), "H": float(X0[2])},
            _rot2euler(Rx90 @ R), len(inliers), inliers)


# =============================================================================
# 6.  DATA LOADING
# =============================================================================

def find_data_root(base: str = DATASET_BASE) -> str:
    for candidate in [os.path.join(base, "Data"), base]:
        if os.path.isdir(os.path.join(candidate, "UAV_image")):
            return candidate
    raise FileNotFoundError(
        f"Cannot find UAV_image/ under '{base}'.\n"
        "Make sure the dataset 'hunhtrungkit/uav-avl' is added to this notebook."
    )


def read_metadata(json_path: str) -> list:
    """Load metadata JSON; handles both list and {\"root\": [...]} formats."""
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and isinstance(data.get("root"), list):
        data = data["root"]
    assert isinstance(data, list), f"Unexpected metadata format in {json_path}"
    return data


def query_metadata(meta_list: list, img_path: str) -> Optional[dict]:
    """
    Find the metadata entry for a given image path.
    Tries exact 'name' match first, then falls back to basename match.
    """
    key = f"{os.path.dirname(img_path)}/{os.path.basename(img_path)}"
    for e in meta_list:
        if e.get("name") == key:
            return e
    base = os.path.basename(img_path).lower()
    for e in meta_list:
        if os.path.basename(str(e.get("name", ""))).lower() == base:
            return e
    return None


def get_jpg_files(folder: str) -> list:
    if not os.path.isdir(folder):
        return []
    return glob.glob(os.path.join(folder, "*.JPG"))


def crop_center_pil(img_path: str, W: int, H: int) -> Image.Image:
    """Read image and crop a W×H region centred on the image."""
    img = cv2.imread(img_path)
    oh, ow = img.shape[:2]
    l = max(0, ow // 2 - W // 2)
    r = min(ow, ow // 2 + W // 2)
    t = max(0, oh // 2 - H // 2)
    b = min(oh, oh // 2 + H // 2)
    return Image.fromarray(img[t:b, l:r])


# =============================================================================
# 7.  GALLERY CREATION
# =============================================================================

def compute_gallery_centers(ref_map: np.ndarray,
                             block_size: list,
                             step_size: list) -> np.ndarray:
    """
    Slide a window over ref_map and collect centre coordinates of non-black tiles.
    Returns (N, 2) array of [row_mid, col_mid].
    """
    h, w = ref_map.shape[:2]
    small = cv2.resize(ref_map, (w // 10, h // 10),
                       interpolation=cv2.INTER_NEAREST)
    thresh = block_size[0] * block_size[1] / 100.0 / 5.0 * 2.0

    n_row = len(range(0, h, step_size[0]))
    n_col = len(range(0, w, step_size[1]))
    mids  = []
    for i in range(n_row):
        for j in range(n_col):
            sx = min(i * step_size[0], h - block_size[0] - 1)
            sy = min(j * step_size[1], w - block_size[1] - 1)
            ex, ey = sx + block_size[0], sy + block_size[1]
            tile = small[sx // 10:ex // 10, sy // 10:ey // 10]
            if np.sum(tile[:, :, 0] > 0) >= thresh:
                mids.append([(sx + ex) / 2.0, (sy + ey) / 2.0])

    return np.array(mids)   # shape (N, 2): [:, 0]=row_mid, [:, 1]=col_mid


# =============================================================================
# 8.  RETRIEVAL
# =============================================================================

def view_center_px(meta: dict, initial_x: float, initial_y: float,
                   ref_res: float, matRot: np.ndarray):
    """
    Project the UAV's estimated view-centre (ground point directly below the
    yaw-offset footprint) onto the *rotated* reference map in pixel coords.
    Returns (col_px, row_px).
    """
    pitch   = -meta["pitch"] / 180.0 * np.pi
    yaw     = -meta["yaw"]   / 180.0 * np.pi
    utm_x, utm_y = deg2utm(meta["lon"], meta["lat"])
    dy = meta["rel_alt"] / np.tan(pitch) * np.cos(yaw)
    dx = -meta["rel_alt"] / np.tan(pitch) * np.sin(yaw)
    # Ground point in original ref_map pixels (col, row)
    col0 = int((utm_x + dx - initial_x) / ref_res)
    row0 = int((initial_y - (utm_y + dy)) / ref_res)
    # Rotate to the rotated map
    c = np.array([col0, row0, 1.0]) @ matRot.T
    return float(c[0]), float(c[1])     # (col_px, row_px) on rotated map


def _extract_features(spdgeo_model, transform, ref_map, mids,
                       block_size, uav_tensor, device):
    """
    Extract L2-normalised SPDGeo embeddings for all gallery tiles and the query.
    Returns: (gallery_feats [N, D], query_feat [1, D])
    """
    spdgeo_model.eval()
    gallery = []
    for bs in tqdm(range(0, len(mids), RETRIEVAL_BATCH_SIZE), desc="Gallery"):
        be    = min(bs + RETRIEVAL_BATCH_SIZE, len(mids))
        batch = []
        for mx, my in mids[bs:be]:
            tile = ref_map[
                int(mx - block_size[0] / 2):int(mx + block_size[0] / 2),
                int(my - block_size[1] / 2):int(my + block_size[1] / 2),
            ]
            tile = cv2.resize(tile, (RETRIEVAL_SIZE, RETRIEVAL_SIZE))
            batch.append(transform(Image.fromarray(tile)))
        if not batch:
            continue
        imgs = torch.stack(batch).to(device)
        with torch.no_grad(), autocast():
            feat = spdgeo_model(imgs)
            if FEAT_NORM:
                feat = F.normalize(feat, dim=-1)
        gallery.append(feat.float())

    gf = torch.cat(gallery, dim=0) if gallery else torch.zeros(0, device=device)

    with torch.no_grad(), autocast():
        qf = spdgeo_model(uav_tensor.unsqueeze(0).to(device))
        if FEAT_NORM:
            qf = F.normalize(qf, dim=-1)

    return gf, qf.float()


def run_retrieval(ref_map, uav_img_path, meta, ref_res,
                  matRot, spdgeo_model, spdgeo_transform, device):
    """
    Perform image-level retrieval.

    Returns
    -------
    order, start_row_list, start_col_list, pde_list,
    cut_H (= block_size[1]), cut_W (= block_size[0]), finescale, retrieval_time_per_img
    """
    initial_x = _cfg("REF_initialX")
    initial_y = _cfg("REF_initialY")

    drone_res, drone_size = resolution_and_size(meta, POSE_PRIORI)
    finescale = drone_res / ref_res
    view_m    = drone_size * drone_res                      # footprint in metres

    bh = math.ceil(view_m[0] / ref_res)
    bh += bh % 2                                           # ensure even
    bw = math.ceil(view_m[1] / ref_res)
    bw += bw % 2
    block_size = [bh, bw]
    step_size  = [int(bh * (100 - RETRIEVAL_COVER) / 100),
                  int(bw * (100 - RETRIEVAL_COVER) / 100)]

    center_col, center_row = view_center_px(
        meta, initial_x, initial_y, ref_res, matRot
    )
    mids = compute_gallery_centers(ref_map, block_size, step_size)  # (N, 2): [row, col]

    # Prepare query image
    uav_pil = crop_center_pil(uav_img_path, int(drone_size[0]), int(drone_size[1]))
    uav_pil = uav_pil.resize((RETRIEVAL_SIZE, RETRIEVAL_SIZE))
    uav_t   = spdgeo_transform(uav_pil)

    t0 = time.time()
    gf, qf = _extract_features(spdgeo_model, spdgeo_transform,
                                ref_map, mids, block_size, uav_t, device)
    t1 = time.time()
    retrieval_time = (t1 - t0) / max(len(mids) + 1, 1)

    scores = (gf @ qf.T).squeeze(-1).cpu().numpy()
    order  = np.argsort(scores)[::-1].copy()

    start_rows, start_cols, pde_list = [], [], []
    for idx in order:
        mx, my = mids[idx, 0], mids[idx, 1]       # row_mid, col_mid
        start_rows.append(max(0, int(mx - block_size[0] / 2)))
        start_cols.append(max(0, int(my - block_size[1] / 2)))
        # PDE: normalised distance between view centre and tile centre
        d = sqrt((center_col - my) ** 2 + (center_row - mx) ** 2)
        pde_list.append(d / block_size[0])

    # Return signature mirrors the original Baseline.py unpacking:
    #   order, refLocX(=start_row), refLocY(=start_col), PDE,
    #   cut_H(=block_size[1]), cut_W(=block_size[0]), finescale, time
    return (order, start_rows, start_cols, pde_list,
            block_size[1], block_size[0], finescale, retrieval_time)


# =============================================================================
# 9.  MATCHING  (RoMa via romatch)
# =============================================================================

def roma_match_wrapper(uav_bgr: np.ndarray,
                       ref_bgr: np.ndarray,
                       roma_model,
                       device: str = "cuda"):
    """
    Dense match two BGR images with RoMa (loaded from romatch pip package).
    Returns (sen_pts, ref_pts) as lists of [col, row] pixel coordinates.
    """
    H_A, W_A = uav_bgr.shape[:2]
    H_B, W_B = ref_bgr.shape[:2]
    # romatch expects PIL images (RGB), but BGR vs RGB only affects colours,
    # not keypoint positions — so fromarray on BGR is fine for localisation.
    im_a = Image.fromarray(uav_bgr)
    im_b = Image.fromarray(ref_bgr)

    warp, certainty = roma_model.match(im_a, im_b, device=device)
    matches, certainty = roma_model.sample(warp, certainty, num=3000)
    kp_a, kp_b = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)

    sen_pts = kp_a.cpu().numpy().tolist()
    ref_pts = kp_b.cpu().numpy().tolist()
    return sen_pts, ref_pts


# =============================================================================
# 10.  MATCHING + PnP LOOP
# =============================================================================

def match_and_pnp(uav_bgr, finescale, K,
                  ref_map, dsm_map,
                  start_rows, start_cols, cut_H, cut_W,
                  roma_model, matRot, device):
    """
    For each top-N retrieval candidate:
      1. Crop fineRef tile from rotated ref_map
      2. Run RoMa dense matching
      3. Convert pixel matches → UTM 3D points via DSM elevation
      4. Solve PnP → GPS (lat, lon)

    Returns
    -------
    BLH_list, inliers_list, match_times, pnp_times
    """
    rev_rot = cv2.invertAffineTransform(matRot) if POSE_PRIORI == "yp" else None

    initial_x = _cfg("REF_initialX")
    initial_y = _cfg("REF_initialY")
    ref_res   = _cfg("REF_resolution")
    dsm_res   = _cfg("DSM_resolution")
    ref_coor  = _cfg("REF_COORDINATE")   # [col, row] of alignment pt in REF
    dsm_coor  = _cfg("DSM_COORDINATE")   # [col, row] of alignment pt in DSM

    dsm_ratio = dsm_res / ref_res
    # DSM alignment offset in REF-pixel units (same formula as Benchmark/utils.py)
    dsm_off_col = int(dsm_coor[0] * dsm_ratio - ref_coor[0])
    dsm_off_row = int(dsm_coor[1] * dsm_ratio - ref_coor[1])

    if RESIZE_RATIO < 1:
        uav_img = cv2.resize(uav_bgr, None, fx=RESIZE_RATIO, fy=RESIZE_RATIO)
    else:
        uav_img = uav_bgr

    top_n = {"Top1": 1, "Topn_opt": min(RETRIEVAL_TOPN, len(start_rows))}.get(
        STRATEGY, len(start_rows)
    )

    BLH_list, inliers_list, match_times, pnp_times = [], [], [], []

    for i in range(top_n):
        sr, sc = start_rows[i], start_cols[i]
        # Crop gallery tile: [start_row : start_row+cut_W, start_col : start_col+cut_H]
        fineRef = ref_map[sr:sr + cut_W, sc:sc + cut_H]
        fineRef = cv2.resize(fineRef, None,
                             fx=RESIZE_RATIO / finescale,
                             fy=RESIZE_RATIO / finescale)

        t0 = time.time()
        sen_pts, ref_pts = roma_match_wrapper(uav_img, fineRef, roma_model, device)
        t1 = time.time()

        if len(ref_pts) >= 5:
            # ref_pts are (col, row) in resized fineRef → convert to absolute ref_map coords
            ref_coord = (np.array(ref_pts) / RESIZE_RATIO * finescale
                         + np.array([sc, sr]))   # + [start_col, start_row]

            # Un-rotate from rotated map back to original map
            if rev_rot is not None:
                rc1 = np.hstack([ref_coord, np.ones((len(ref_coord), 1))])
                ref_coord = rc1 @ rev_rot.T       # (N, 2): [col, row] in original map

            # Pixel coordinates → UTM
            #   ref_coord[:, 0] = col  →  Easting  (UTM_X)
            #   ref_coord[:, 1] = row  →  Northing (UTM_Y, note: Y decreases with row)
            utm_x = ref_coord[:, 0] * ref_res + initial_x
            utm_y = initial_y - ref_coord[:, 1] * ref_res

            # Look up DSM elevation
            #   dsm_x (row in DSM) uses ref_coord[:, 1] + row_offset
            #   dsm_y (col in DSM) uses ref_coord[:, 0] + col_offset
            dsm_row = (ref_coord[:, 1] + dsm_off_row + 1) / dsm_ratio - 1
            dsm_col = (ref_coord[:, 0] + dsm_off_col + 1) / dsm_ratio - 1
            dri = np.clip(dsm_row.astype(int), 0, dsm_map.shape[0] - 1)
            dci = np.clip(dsm_col.astype(int), 0, dsm_map.shape[1] - 1)
            dsm_z = dsm_map[dri, dci]

            # UAV image pixel coordinates (divided by resize_ratio = original scale)
            uav_pts = np.array(sen_pts) / RESIZE_RATIO

            blh, _, n_inliers, _ = estimate_drone_pose(utm_x, utm_y, dsm_z, uav_pts, K)
        else:
            blh, n_inliers = {"B": None, "L": None, "H": None}, 0

        t2 = time.time()
        BLH_list.append(blh)
        inliers_list.append(n_inliers)
        match_times.append(t1 - t0)
        pnp_times.append(t2 - t1)

    return BLH_list, inliers_list, match_times, pnp_times


# =============================================================================
# 11.  ERROR CALCULATION
# =============================================================================

def pos2error(true_meta: dict, BLH_list: list, inliers_list: list):
    """
    Pick the prediction with the most PnP inliers and compute its localization
    error in metres (horizontal only).

    Returns (pred_loc_dict, best_error_m, all_errors_list)
    """
    best_idx = int(np.argmax(inliers_list))
    errors   = []
    for blh in BLH_list:
        if blh["B"] is not None and blh["L"] is not None:
            dlat = (blh["B"] - true_meta["lat"]) * 110_000
            dlon = (blh["L"] - true_meta["lon"]) * 110_000 * math.cos(
                math.radians(true_meta["lat"])
            )
            errors.append(sqrt(dlat ** 2 + dlon ** 2))
        else:
            errors.append(10_000.0)   # sentinel for failed prediction

    pred_loc = {"lat": BLH_list[best_idx]["B"], "lon": BLH_list[best_idx]["L"]}
    return pred_loc, errors[best_idx], errors


# =============================================================================
# 12.  MAIN PIPELINE
# =============================================================================

def main():
    # ── Resolve data root ───────────────────────────────────────────────────
    data_root = find_data_root()
    uav_root  = os.path.join(data_root, "UAV_image",     REGION)
    meta_json = os.path.join(data_root, "metadata",      f"{REGION}.json")
    ref_path  = os.path.join(data_root, "Reference_map", REGION, REF_FNAME[REF_TYPE])
    dsm_path  = os.path.join(data_root, "Reference_map", REGION, DSM_FNAME[REF_TYPE])
    ref_res   = _cfg("REF_resolution")

    print(f"[INFO] data_root : {data_root}")
    print(f"[INFO] ref_map   : {ref_path}")
    print(f"[INFO] dsm       : {dsm_path}")

    # ── Load reference + DSM ────────────────────────────────────────────────
    ref_map = cv2.imread(ref_path)[:, :, :3].astype(np.uint8)
    dsm_map = cv2.imread(dsm_path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    print(f"[INFO] ref_map {ref_map.shape}  dsm_map {dsm_map.shape}")

    # ── Load metadata ───────────────────────────────────────────────────────
    metadata = read_metadata(meta_json)
    print(f"[INFO] metadata entries : {len(metadata)}")

    # ── Collect UAV image list ──────────────────────────────────────────────
    all_imgs = []
    for place in UAV_PLACES:
        all_imgs += get_jpg_files(os.path.join(uav_root, place))
    all_imgs = sorted(all_imgs)[::TEST_INTERVAL]
    print(f"[INFO] images to test   : {len(all_imgs)}\n")

    # ── Load models ─────────────────────────────────────────────────────────
    print(f"[MODEL] Loading SPDGeo (device={DEVICE}) …")
    spdgeo_model, spdgeo_transform = load_spdgeo(SPDGEO_CKPT, device=DEVICE)

    print(f"[MODEL] Loading RoMa from HuggingFace/GitHub …")
    roma_model = roma_outdoor(device=DEVICE)   # auto-downloads weights
    print()

    # ── Main loop ────────────────────────────────────────────────────────────
    results = []
    n_total = len(all_imgs)

    for idx, img_path in enumerate(tqdm(all_imgs, desc="UAV-AVL", unit="img")):
        place    = os.path.basename(os.path.dirname(img_path))
        img_stem = os.path.splitext(os.path.basename(img_path))[0]
        pkl_path = os.path.join(
            SAVE_DIR, REGION, place,
            f"{REF_TYPE}-SPDGeo-Roma-{POSE_PRIORI}",
            f"VG_{img_stem}.pkl",
        )

        if os.path.exists(pkl_path):
            print(f"[SKIP] {place}/{img_stem} (cached)")
            continue

        # Find metadata for this image
        meta = query_metadata(metadata, img_path)
        if meta is None:
            print(f"[WARN] No metadata for {img_stem}, skipping")
            continue

        K       = compute_camera_matrix(meta)
        uav_bgr = cv2.imread(img_path)

        # Rotate reference map by yaw prior to roughly align with UAV view
        ref_rotated, matRot = dump_rotate_image(ref_map, meta["yaw"])

        t_start = time.time()

        # ── Step 1: Image-level Retrieval ──────────────────────────────────
        (order, start_rows, start_cols, pde_list,
         cut_H, cut_W, finescale, ret_time) = run_retrieval(
            ref_rotated, img_path, meta, ref_res,
            matRot, spdgeo_model, spdgeo_transform, DEVICE,
        )

        # ── Steps 2 & 3: Dense Matching + PnP ─────────────────────────────
        BLH_list, inliers_list, match_times, pnp_times = match_and_pnp(
            uav_bgr, finescale, K,
            ref_rotated, dsm_map,
            start_rows, start_cols, cut_H, cut_W,
            roma_model, matRot, DEVICE,
        )

        # ── Step 4: Select best prediction & compute error ─────────────────
        pred_loc, pred_error, error_list = pos2error(meta, BLH_list, inliers_list)

        total_t = time.time() - t_start
        print(
            f"[{idx+1}/{n_total}] {place}/{img_stem} "
            f"→ error={pred_error:.1f}m  inliers={inliers_list}  t={total_t:.1f}s"
        )

        row = {
            "image":      img_path,
            "place":      place,
            "pred_lat":   pred_loc["lat"],
            "pred_lon":   pred_loc["lon"],
            "true_lat":   meta["lat"],
            "true_lon":   meta["lon"],
            "error_m":    pred_error,
            "inliers":    inliers_list,
            "error_list": error_list,
        }
        results.append(row)

        # Cache to disk
        os.makedirs(os.path.dirname(pkl_path), exist_ok=True)
        with open(pkl_path, "wb") as f:
            pickle.dump(row, f)

    # ── Summary ──────────────────────────────────────────────────────────────
    valid = [r for r in results if r["error_m"] < 9_000]
    print(f"\n{'='*60}")
    print(f"  RESULTS  ({len(valid)}/{len(results)} images with valid prediction)")
    if valid:
        errs = [r["error_m"] for r in valid]
        print(f"  Mean error  : {np.mean(errs):.2f} m")
        print(f"  Median      : {np.median(errs):.2f} m")
        print(f"  Under  1 m  : {sum(1 for e in errs if e < 1)}")
        print(f"  Under  5 m  : {sum(1 for e in errs if e < 5)}")
        print(f"  Under 10 m  : {sum(1 for e in errs if e < 10)}")
        print(f"  Results in  : {SAVE_DIR}")
    print(f"{'='*60}\n")

    return results


if __name__ == "__main__":
    main()
