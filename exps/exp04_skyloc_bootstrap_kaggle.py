"""
exp04_skyloc_bootstrap_kaggle.py
================================
One-file Kaggle runner:

- Clones YOUR SkyLoc repo (which already vendors `Benchmark/`)
- Links Kaggle dataset `hunhtrungkit/uav-avl` into `Benchmark/Data`
- Installs dependencies
- (Optional) downloads RoMa + DINOv2 weights
- Runs `Benchmark/Baseline.py` using retrieval method `SPDGeo` (patched in repo config)

Important:
  - Provide SPDGeo checkpoint via env var `SPDGEO_CKPT` (recommended) or edit below.
  - RoMa weights are required for full baseline (matching+PnP).
"""

import os
import sys
import shutil
import subprocess
import textwrap
from pathlib import Path


SKYLOC_REPO_URL = os.environ.get("SKYLOC_REPO_URL", "https://github.com/chisngyen/SkyLoc.git")

WORK = Path("/kaggle/working")
REPO_DIR = WORK / "SkyLoc"
BENCH_DIR = REPO_DIR / "Benchmark"

DATA_ROOT = Path("/kaggle/input/datasets/hunhtrungkit/uav-avl/Data")

# Optional weights download (public)
ROMA_URL = "https://huggingface.co/datasets/Parskatt/storage/resolve/main/roma_outdoor.pth"
DINO_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"

ROMA_DST = BENCH_DIR / "Matching_Models/RoMa/ckpt/roma_outdoor.pth"
DINO_DST = BENCH_DIR / "Matching_Models/RoMa/ckpt/dinov2_vitl14_pretrain.pth"


def run(cmd, cwd=None):
    print("\n$", " ".join(str(c) for c in cmd))
    subprocess.check_call([str(c) for c in cmd], cwd=cwd)


def pip_install(*pkgs):
    run([sys.executable, "-m", "pip", "install", "-q", *pkgs])


def wget(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["wget", "-q", "--show-progress", "-O", str(dest), url])


def section(title: str):
    print("\n" + "=" * 60)
    print(" ", title)
    print("=" * 60)


def main():
    section("Step 1/6 — Clone SkyLoc repo")
    if REPO_DIR.exists():
        print(f"[SKIP] {REPO_DIR} exists")
    else:
        run(["git", "clone", "--depth", "1", SKYLOC_REPO_URL, str(REPO_DIR)])
    assert BENCH_DIR.exists(), f"Missing Benchmark folder at: {BENCH_DIR}"

    section("Step 2/6 — Install dependencies")
    req = BENCH_DIR / "requriements.txt"
    if req.exists():
        pip_install("-r", str(req))
    else:
        pip_install(
            "torch", "torchvision",
            "timm",
            "opencv-python-headless",
            "pyproj",
            "scikit-image",
            "matplotlib",
            "tqdm",
            "Pillow",
            "pyyaml",
        )

    section("Step 3/6 — Link UAV-AVL dataset into Benchmark/Data")
    assert DATA_ROOT.exists(), (
        f"[ERROR] Dataset not found at {DATA_ROOT}\n"
        "Add dataset `hunhtrungkit/uav-avl` to this notebook."
    )
    data_link = BENCH_DIR / "Data"
    if data_link.exists() or data_link.is_symlink():
        print(f"[SKIP] {data_link} exists")
    else:
        data_link.symlink_to(DATA_ROOT)
        print(f"[OK] Linked {data_link} -> {DATA_ROOT}")

    section("Step 4/6 — Ensure weights (RoMa + DINOv2)")
    if not ROMA_DST.exists():
        print("[DL] RoMa weights...")
        wget(ROMA_URL, ROMA_DST)
    else:
        print("[SKIP] RoMa weights present")
    if not DINO_DST.exists():
        print("[DL] DINOv2 ViT-L/14 weights...")
        wget(DINO_URL, DINO_DST)
    else:
        print("[SKIP] DINOv2 weights present")

    section("Step 5/6 — SPDGeo checkpoint check")
    spdgeo_ckpt = os.environ.get("SPDGEO_CKPT")
    if not spdgeo_ckpt:
        print(textwrap.dedent("""
        [WARN] SPDGEO_CKPT is not set.
        - SPDGeo retrieval will run with default init (not recommended).
        - Set env var SPDGEO_CKPT to a .pth checkpoint path, e.g.:
          /kaggle/input/models/.../exp35_dpea_ga_best.pth
        """))
    else:
        print(f"[OK] SPDGEO_CKPT={spdgeo_ckpt}")

    section("Step 6/6 — Run Benchmark/Baseline.py")
    # Run from Benchmark folder to match relative imports
    run(
        [
            sys.executable,
            "Baseline.py",
            "--yaml",
            "config.yaml",
            "--save_dir",
            str(WORK / "results"),
            "--Ref_type",
            "HIGH",
            "--pose_priori",
            "yp",
            "--strategy",
            "Topn_opt",
            "--PnP_method",
            "P3P",
            "--resize_ratio",
            "0.2",
        ],
        cwd=str(BENCH_DIR),
    )
    print("\n[DONE] Results in /kaggle/working/results")


if __name__ == "__main__":
    main()

