"""
EXP02 (Benchmark-style) — UAV-AVL QZ_Town Aerial Baseline (1-file runner)
----------------------------------------------------------------------------
Goal: replicate Benchmark/Baseline.py behavior for Kaggle demo data layout:

Data/
  Reference_map/QZ_Town/{result_roi.tif,dsm_roi.tif,satellite_roi.tif,...}
  UAV_image/QZ_Town/{QZ_SongCity,Qingzhou_3_2,QingZhou_2024}/*.JPG
  metadata/QZ_Town.json   (dict with key "root": list[dict])

Pipeline (same as Benchmark demo):
  CAMP retrieval -> RoMa matching -> DSM sampling -> solvePnPRansac (P3P) -> error metrics.

Notes:
  - This file avoids YAML by hardcoding the QZ_Town constants.
  - It still uses Benchmark's implementation for retrieval/matching/PnP to stay identical.
"""

import os
import sys
import json
import time
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

warnings.filterwarnings("ignore")


# -----------------------------------------------------------------------------
# Optional lightweight dependency bootstrap (Kaggle-friendly)
# -----------------------------------------------------------------------------
def _pip_install(pkg: str) -> None:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


def _ensure_import(mod: str, pkg: str | None = None) -> None:
    try:
        __import__(mod)
    except Exception:
        _pip_install(pkg or mod)


for _mod, _pkg in [
    ("cv2", "opencv-python-headless"),
    ("timm", "timm"),
    ("pyproj", "pyproj"),
    ("skimage", "scikit-image"),
    ("torchvision", "torchvision"),
]:
    _ensure_import(_mod, _pkg)

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402


# -----------------------------------------------------------------------------
# Benchmark imports (kept identical; we just wire config + metadata ourselves)
# -----------------------------------------------------------------------------
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent
BENCH_ROOT = REPO_ROOT / "Benchmark"

if not BENCH_ROOT.exists():
    raise RuntimeError(
        f"Benchmark folder not found at: {BENCH_ROOT}. "
        "Copy the Benchmark directory alongside this script."
    )

# Make `import utils`, `Retrieval_Models...`, `Matching_Models...` resolve like Benchmark demo
sys.path.insert(0, str(BENCH_ROOT))

# Benchmark code assumes relative paths like ./Matching_Models/..., ./Retrieval_Models/...
os.chdir(BENCH_ROOT)

from utils import (  # noqa: E402
    retrieval_init,
    matching_init,
    load_config_parameters_new,
    retrieval_all,
    Match2Pos_all,
    pos2error,
    computeCameraMatrix,
    read_data_from_file,
    get_jpg_files,
    img_name,
)


# -----------------------------------------------------------------------------
# Hardcoded QZ_Town constants (from Benchmark/Regions_params/QZ_Town.yaml)
# -----------------------------------------------------------------------------
class CFG:
    # Kaggle dataset root (points to .../Data)
    DATASET_ROOT = "/kaggle/input/datasets/hunhtrungkit/uav-avl/Data"

    REGION = "QZ_Town"
    REF_TYPE = "HIGH"  # aerial map (result_roi.tif + dsm_roi.tif)

    # UAV places
    PLACES = ["QZ_SongCity", "Qingzhou_3_2", "QingZhou_2024"]
    TEST_INTERVAL = 20  # same as Benchmark/config.yaml default

    # Retrieval/matching settings (mirror Benchmark/config.yaml)
    DEVICE = "cuda"
    RETRIEVAL_METHODS = ["CAMP"]
    MATCHING_METHODS = ["Roma"]
    RETRIEVAL_COVER = 50
    RETRIEVAL_TOPN = 5
    RETRIEVAL_IMG = "/1_Retrieval.png"
    RETRIEVAL_IMG_NUM = 5
    RETRIEVAL_FEATURE_NORM = True
    SHOW_RETRIEVAL_RESULT = False
    BATCH_SIZE = 128

    # Region geo params
    QZ_Town_UTM_SYSTEM = "50N"

    # Aerial (HIGH)
    QZ_Town_HIGH_REF_initialX = 631556.341
    QZ_Town_HIGH_REF_initialY = 4061562.708
    QZ_Town_HIGH_REF_resolution = 0.0609957
    QZ_Town_HIGH_DSM_resolution = 0.937272
    QZ_Town_HIGH_REF_COORDINATE = [2143.3682, 2301.1853]
    QZ_Town_HIGH_DSM_COORDINATE = [140.1837, 150.3175]

    # Output
    OUTPUT_DIR = "/kaggle/working"


@dataclass
class Opt:
    save_dir: str = CFG.OUTPUT_DIR
    device: str = CFG.DEVICE
    pose_priori: str = "yp"
    strategy: str = "Topn_opt"
    PnP_method: str = "P3P"
    Ref_type: str = CFG.REF_TYPE
    resize_ratio: float = 0.2


def _path_in_data(*parts: str) -> str:
    return os.path.join(CFG.DATASET_ROOT, *parts)


def build_region_config() -> dict:
    region = CFG.REGION
    ref_dir = _path_in_data("Reference_map", region)
    uav_dir = _path_in_data("UAV_image", region)
    meta_path = _path_in_data("metadata", f"{region}.json")

    cfg = {
        # base config keys used by Benchmark/utils.py
        "TEST_INTERVAL": CFG.TEST_INTERVAL,
        "DEVICE": CFG.DEVICE,
        "RETRIEVAL_METHODS": CFG.RETRIEVAL_METHODS,
        "MATCHING_METHODS": CFG.MATCHING_METHODS,
        "RETRIEVAL_COVER": CFG.RETRIEVAL_COVER,
        "RETRIEVAL_TOPN": CFG.RETRIEVAL_TOPN,
        "RETRIEVAL_IMG": CFG.RETRIEVAL_IMG,
        "RETRIEVAL_IMG_NUM": CFG.RETRIEVAL_IMG_NUM,
        "RETRIEVAL_FEATURE_NORM": CFG.RETRIEVAL_FEATURE_NORM,
        "SHOW_RETRIEVAL_RESULT": CFG.SHOW_RETRIEVAL_RESULT,
        "BATCH_SIZE": CFG.BATCH_SIZE,
        # region keys
        f"{region}_UTM_SYSTEM": CFG.QZ_Town_UTM_SYSTEM,
        f"{region}_UAV_PATH": uav_dir + os.sep,
        f"{region}_UAV_PLACES": list(CFG.PLACES),
        # paths (HIGH = aerial)
        f"{region}_HIGH_REF_PATH": os.path.join(ref_dir, "result_roi.tif"),
        f"{region}_HIGH_DSM_PATH": os.path.join(ref_dir, "dsm_roi.tif"),
        f"{region}_HIGH_REF_initialX": CFG.QZ_Town_HIGH_REF_initialX,
        f"{region}_HIGH_REF_initialY": CFG.QZ_Town_HIGH_REF_initialY,
        f"{region}_HIGH_REF_resolution": CFG.QZ_Town_HIGH_REF_resolution,
        f"{region}_HIGH_DSM_resolution": CFG.QZ_Town_HIGH_DSM_resolution,
        f"{region}_HIGH_REF_COORDINATE": list(CFG.QZ_Town_HIGH_REF_COORDINATE),
        f"{region}_HIGH_DSM_COORDINATE": list(CFG.QZ_Town_HIGH_DSM_COORDINATE),
        # metadata path
        "__META_JSON__": meta_path,
    }
    return cfg


def load_metadata_index(meta_json_path: str) -> dict:
    """
    Kaggle QZ_Town.json is {"root": [ {name, lat, lon, ...}, ... ]}.
    Build a robust lookup by multiple normalized keys.
    """
    data = json.load(open(meta_json_path, "r", encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("root"), list):
        entries = data["root"]
    elif isinstance(data, list):
        entries = data
    else:
        raise ValueError(f"Unexpected metadata schema in {meta_json_path}")

    idx: dict[str, dict] = {}
    for e in entries:
        if not isinstance(e, dict):
            continue
        name = e.get("name")
        if not name:
            continue
        name = str(name).replace("\\", "/")
        variants = set()
        variants.add(name)
        variants.add(name.lstrip("./"))
        if name.lower().startswith("./data/"):
            variants.add(name[2:])  # "Data/.."
            variants.add(name[7:])  # after "./Data/"
        if name.lower().startswith("data/"):
            variants.add(name[5:])
        for k in variants:
            idx[k] = e
    return idx


def meta_key_from_uav_path(uav_path: str) -> str:
    # Benchmark queries JSON by: f'{os.path.dirname(uav_path)}/{basename}'
    # Kaggle metadata stores: "./Data/UAV_image/...."
    uav_path = os.path.normpath(uav_path)
    rel = os.path.relpath(uav_path, CFG.DATASET_ROOT).replace("\\", "/")
    return f"./Data/{rel}"


def list_uav_images(region_cfg: dict) -> list[str]:
    uav_root = region_cfg[f"{CFG.REGION}_UAV_PATH"]
    uav_imgs: list[str] = []
    for place in region_cfg[f"{CFG.REGION}_UAV_PLACES"]:
        uav_imgs.extend(get_jpg_files(os.path.join(uav_root, place)))
    uav_imgs = uav_imgs[0 :: int(region_cfg["TEST_INTERVAL"])]
    return uav_imgs


def main():
    print("=" * 60)
    print("  EXP02 — UAV-AVL Benchmark-style Baseline (QZ_Town Aerial)")
    print("  Pipeline: CAMP retrieval → RoMa matching → DSM+PnP")
    print("=" * 60)

    opt = Opt()
    region_cfg = build_region_config()

    # Check data paths
    meta_json = region_cfg["__META_JSON__"]
    assert os.path.exists(meta_json), f"Missing metadata: {meta_json}"
    assert os.path.exists(region_cfg[f"{CFG.REGION}_HIGH_REF_PATH"]), "Missing HIGH ref map"
    assert os.path.exists(region_cfg[f"{CFG.REGION}_HIGH_DSM_PATH"]), "Missing HIGH DSM map"

    # Metadata index (fixes Benchmark's query_data_from_file limitation with {"root":...})
    meta_idx = load_metadata_index(meta_json)

    # Init models
    method_dict: dict = {}
    method_dict["retrieval_method"] = CFG.RETRIEVAL_METHODS[0]
    method_dict = retrieval_init(method_dict, region_cfg)
    method_dict["matching_method"] = CFG.MATCHING_METHODS[0]
    method_dict = matching_init(method_dict)

    # Load maps
    ref_map0, dsm_map0, save_path0, ref_resolution = load_config_parameters_new(region_cfg, opt, CFG.REGION)

    # UAV list
    uav_list = list_uav_images(region_cfg)
    print(f"[Data] UAV images: {len(uav_list)} (interval={region_cfg['TEST_INTERVAL']})")

    os.makedirs(opt.save_dir, exist_ok=True)

    per_image = []
    errors_m = []
    thresholds = [5, 10, 25]
    below = {t: 0 for t in thresholds}

    for i, uav_path in enumerate(tqdm(uav_list, desc=CFG.REGION, unit="image")):
        place = os.path.basename(os.path.dirname(uav_path))

        meta_key = meta_key_from_uav_path(uav_path)
        truePos = meta_idx.get(meta_key) or meta_idx.get(meta_key.lstrip("./")) or meta_idx.get(meta_key.replace("./Data/", "Data/"))
        if truePos is None:
            # Fallback: try matching by suffix
            suffix = "/".join(meta_key.split("/")[-4:])  # UAV_image/QZ_Town/<place>/<file>
            candidates = [v for k, v in meta_idx.items() if k.endswith(suffix)]
            truePos = candidates[0] if candidates else None
        if truePos is None:
            continue

        save_path = os.path.join(save_path0, CFG.REGION, place, str(i + 1))
        os.makedirs(save_path, exist_ok=True)

        K = computeCameraMatrix(truePos)
        uav_image = cv2.imread(uav_path)

        # Rotate ref map using yaw (Benchmark does this inside Baseline.py)
        # We call Benchmark's dumpRotateImage through retrieval_all/Match2Pos_all which expects matRotation,
        # so we re-import it here lazily to avoid shadowing.
        from utils import dumpRotateImage  # noqa: E402
        ref_map, matRotation = dumpRotateImage(ref_map0, truePos["yaw"])

        t0 = time.time()
        IR_order, refLocX, refLocY, PDE_list, cut_H, cut_W, fineScale, retrieval_time = retrieval_all(
            ref_map, uav_path, truePos, ref_resolution, matRotation, save_path, opt, CFG.REGION, region_cfg, method_dict
        )
        BLH_list, inliers_list, match_time, pnp_time = Match2Pos_all(
            opt, CFG.REGION, region_cfg, uav_image, fineScale, K,
            ref_map, dsm_map0, refLocY, refLocX, cut_H, cut_W,
            save_path, method_dict, matRotation
        )
        pred_loc, pred_error, location_error_list = pos2error(truePos, BLH_list, inliers_list)
        total_time = time.time() - t0

        errors_m.append(float(pred_error))
        for t in thresholds:
            if pred_error <= t:
                below[t] += 1

        per_image.append(
            {
                "uav_path": uav_path,
                "place": place,
                "true": {"lat": truePos.get("lat"), "lon": truePos.get("lon")},
                "pred": pred_loc,
                "pred_error_m": float(pred_error),
                "inliers": inliers_list,
                "retrieval_time_s_per_img": float(retrieval_time),
                "match_time_s": match_time,
                "pnp_time_s": pnp_time,
                "total_time_s": float(total_time),
                "topn_refLocX": refLocX[: CFG.RETRIEVAL_TOPN] if isinstance(refLocX, list) else refLocX,
                "topn_refLocY": refLocY[: CFG.RETRIEVAL_TOPN] if isinstance(refLocY, list) else refLocY,
                "PDE_topn": PDE_list[: CFG.RETRIEVAL_TOPN] if isinstance(PDE_list, list) else PDE_list,
            }
        )

    if len(errors_m) == 0:
        raise RuntimeError("No images evaluated (metadata lookup likely failed).")

    errors_np = np.array(errors_m, dtype=np.float32)
    summary = {
        "n_images": int(len(errors_m)),
        "mean_error_m": float(np.mean(errors_np)),
        "median_error_m": float(np.median(errors_np)),
        **{f"acc_<=_{t}m": float(below[t] / len(errors_m)) for t in thresholds},
    }

    out = {
        "experiment": "exp02_uavavl_benchmarkstyle_qztown_aerial",
        "region": CFG.REGION,
        "ref_type": CFG.REF_TYPE,
        "dataset_root": CFG.DATASET_ROOT,
        "config": {k: v for k, v in region_cfg.items() if not k.startswith("__")},
        "summary": summary,
        "per_image": per_image,
    }

    out_path = os.path.join(opt.save_dir, "results_exp02_uavavl_benchmarkstyle_qztown.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 60)
    print("SUMMARY")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"\n✓ Saved: {out_path}")


if __name__ == "__main__":
    main()

