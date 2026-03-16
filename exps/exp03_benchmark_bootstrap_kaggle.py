"""
exp03_benchmark_bootstrap_kaggle.py
=====================================
1 file chạy độc lập trên Kaggle. Tự động:
  1. Clone https://github.com/UAV-AVL/Benchmark
  2. Cài toàn bộ dependencies
  3. Download RoMa + DINOv2 weights (HuggingFace / fbaipublicfiles — public)
  4. CAMP weights: tìm trong Kaggle Dataset input; nếu không có thì hướng dẫn rõ
  5. Symlink Data/ -> Kaggle dataset (/kaggle/input/.../Data)
  6. Patch utils.py để đọc metadata JSON dạng {"root":[...]} của Kaggle
  7. Patch QZ_Town.yaml: thay backslash -> "/" (Linux compat), fix absolute paths
  8. Chạy python Baseline.py --yaml config.yaml (y chang Benchmark gốc)

Cách dùng trên Kaggle:
  - Add Dataset: hunhtrungkit/uav-avl
  - (Optional) Add Dataset chứa CAMP weights (xem hướng dẫn bên dưới)
  - Chạy file này
"""

# ============================================================================
# SECTION 0: Bootstrap pre-imports (only stdlib + pip bootstrap)
# ============================================================================
import os
import sys
import json
import shutil
import subprocess
import textwrap
from pathlib import Path

# ============================================================================
# SECTION 1: CONFIG
# ============================================================================

BENCH_REPO_URL = "https://github.com/UAV-AVL/Benchmark.git"
WORK           = Path("/kaggle/working")
BENCH_ROOT     = WORK / "Benchmark"

# Kaggle dataset path (hunhtrungkit/uav-avl)
DATA_ROOT = Path("/kaggle/input/datasets/hunhtrungkit/uav-avl/Data")

# Baseline args (mirror Benchmark/config.yaml defaults)
BASELINE_ARGS = {
    "--yaml":         "config.yaml",
    "--save_dir":     str(WORK / "results"),
    "--Ref_type":     "HIGH",          # aerial map
    "--pose_priori":  "yp",            # use yaw+pitch prior
    "--strategy":     "Topn_opt",
    "--PnP_method":   "P3P",
    "--resize_ratio": "0.2",
}

# Weight paths inside BENCH_ROOT
CAMP_WEIGHT_REL  = Path("Retrieval_Models/CAMP/weights/weights_0.9446_for_U1652.pth")
ROMA_WEIGHT_REL  = Path("Matching_Models/RoMa/ckpt/roma_outdoor.pth")
DINO_WEIGHT_REL  = Path("Matching_Models/RoMa/ckpt/dinov2_vitl14_pretrain.pth")

# Public download URLs
ROMA_URL = "https://huggingface.co/datasets/Parskatt/storage/resolve/main/roma_outdoor.pth"
DINO_URL = "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth"

# Where users might have uploaded CAMP weights as a Kaggle Dataset
CAMP_KAGGLE_SEARCH_PATHS = [
    Path("/kaggle/input/uav-avl-weights/weights_0.9446_for_U1652.pth"),
    Path("/kaggle/input/camp-weights/weights_0.9446_for_U1652.pth"),
    Path("/kaggle/input/uav-avl-camp/weights_0.9446_for_U1652.pth"),
    Path("/kaggle/input/benchmark-weights/weights_0.9446_for_U1652.pth"),
    Path("/kaggle/input/camp-u1652/weights_0.9446_for_U1652.pth"),
]

# ============================================================================
# SECTION 2: HELPERS
# ============================================================================

def run(cmd: list[str], cwd: str | None = None) -> None:
    cmd_str = " ".join(str(c) for c in cmd)
    print(f"\n$ {cmd_str}")
    subprocess.check_call([str(c) for c in cmd], cwd=cwd)


def pip_install(*packages: str) -> None:
    run([sys.executable, "-m", "pip", "install", "-q", *packages])


def wget(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    run(["wget", "-q", "--show-progress", "-O", str(dest), url])


def section(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")


# ============================================================================
# SECTION 3: STEP 1 — Install minimal bootstrap deps
# ============================================================================

section("Step 1/8 — Installing bootstrap dependencies")
pip_install("gdown", "pyyaml")
import gdown  # noqa: E402
import yaml   # noqa: E402


# ============================================================================
# SECTION 4: STEP 2 — Clone Benchmark repo
# ============================================================================

section("Step 2/8 — Cloning UAV-AVL/Benchmark")
if BENCH_ROOT.exists():
    print(f"[SKIP] Already cloned at {BENCH_ROOT}")
else:
    run(["git", "clone", "--depth", "1", BENCH_REPO_URL, str(BENCH_ROOT)])
    print(f"[OK] Cloned to {BENCH_ROOT}")


# ============================================================================
# SECTION 5: STEP 3 — Install Benchmark requirements
# ============================================================================

section("Step 3/8 — Installing Benchmark requirements")

# Benchmark typo: "requriements.txt"
req_file = BENCH_ROOT / "requriements.txt"
if req_file.exists():
    pip_install("-r", str(req_file))
else:
    # fallback: known deps
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
print("[OK] Dependencies installed")


# ============================================================================
# SECTION 6: STEP 4 — Symlink Data/
# ============================================================================

section("Step 4/8 — Linking dataset")

assert DATA_ROOT.exists(), (
    f"\n[ERROR] Dataset not found at: {DATA_ROOT}\n"
    "Please add dataset 'hunhtrungkit/uav-avl' to this Kaggle notebook."
)

data_link = BENCH_ROOT / "Data"
if data_link.exists() or data_link.is_symlink():
    print(f"[SKIP] {data_link} already exists")
else:
    data_link.symlink_to(DATA_ROOT)
    print(f"[OK] Symlinked: {data_link} -> {DATA_ROOT}")


# ============================================================================
# SECTION 7: STEP 5 — Download weights
# ============================================================================

section("Step 5/8 — Downloading model weights")

# ---- 5a: RoMa outdoor weights (HuggingFace, public) ----
roma_path = BENCH_ROOT / ROMA_WEIGHT_REL
if roma_path.exists():
    print(f"[SKIP] RoMa weights already at {roma_path}")
else:
    print("[DL] Downloading RoMa outdoor weights (~700 MB)...")
    wget(ROMA_URL, roma_path)
    print(f"[OK] RoMa weights saved to {roma_path}")

# ---- 5b: DINOv2 ViT-L/14 weights (fbaipublicfiles, public) ----
dino_path = BENCH_ROOT / DINO_WEIGHT_REL
if dino_path.exists():
    print(f"[SKIP] DINOv2 weights already at {dino_path}")
else:
    print("[DL] Downloading DINOv2 ViT-L/14 weights (~1.1 GB)...")
    wget(DINO_URL, dino_path)
    print(f"[OK] DINOv2 weights saved to {dino_path}")

# ---- 5c: CAMP weights (need manual upload or Google Drive) ----
camp_path = BENCH_ROOT / CAMP_WEIGHT_REL
camp_path.parent.mkdir(parents=True, exist_ok=True)

if camp_path.exists():
    print(f"[SKIP] CAMP weights already at {camp_path}")
else:
    # Search known Kaggle Dataset input paths first
    found = False
    for p in CAMP_KAGGLE_SEARCH_PATHS:
        if p.exists():
            shutil.copy(str(p), str(camp_path))
            print(f"[OK] Copied CAMP weights from {p}")
            found = True
            break

    if not found:
        # Try Google Drive (same ID the README lists for both dataset + weights)
        CAMP_GDRIVE_ID = "1GmBOD_5tB9GyHdLmDlXY6--RAsCJbLQf"
        print(f"[DL] Trying Google Drive ID: {CAMP_GDRIVE_ID} ...")
        try:
            gdown.download(
                f"https://drive.google.com/uc?id={CAMP_GDRIVE_ID}",
                str(camp_path),
                quiet=False,
            )
            if camp_path.exists() and camp_path.stat().st_size > 1_000_000:
                print(f"[OK] CAMP weights downloaded to {camp_path}")
                found = True
            else:
                camp_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"[WARN] gdown failed: {e}")

    if not found:
        print(textwrap.dedent(f"""
        ============================================================
        [ACTION REQUIRED] CAMP weights not found!

        Please do ONE of the following:

        Option A — Upload weights as a Kaggle Dataset:
          1. Download weights from:
             https://drive.google.com/file/d/1GmBOD_5tB9GyHdLmDlXY6--RAsCJbLQf
          2. Create a new Kaggle Dataset named "uav-avl-weights"
          3. Upload file: weights_0.9446_for_U1652.pth
          4. Add this dataset to the notebook (it will appear at
             /kaggle/input/uav-avl-weights/weights_0.9446_for_U1652.pth)
          5. Rerun this script

        Option B — Add to notebook files manually:
          Place the weights at:
          {camp_path}

        The script will continue but retrieval will FAIL without this file.
        ============================================================
        """))


# ============================================================================
# SECTION 8: STEP 6 — Patch utils.py to handle {"root":[...]} JSON (Kaggle format)
# ============================================================================

section("Step 6/8 — Patching utils.py for Kaggle metadata JSON format")

utils_py = BENCH_ROOT / "utils.py"
utils_text = utils_py.read_text(encoding="utf-8")

OLD_READ = '''def read_data_from_file(file_name):
    """Read UAV data from a file"""
    if not os.path.exists(file_name):
        print("File does not exist!")
        return []
    with open(file_name, 'r') as f:
        data = json.load(f)

    return data'''

NEW_READ = '''def read_data_from_file(file_name):
    """Read UAV data from a file.
    Supports both plain list[dict] and Kaggle-style {"root": list[dict]}.
    """
    if not os.path.exists(file_name):
        print("File does not exist!")
        return []
    with open(file_name, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Kaggle metadata wraps entries under "root" key
    if isinstance(data, dict) and isinstance(data.get("root"), list):
        data = data["root"]
    return data'''

if OLD_READ in utils_text:
    utils_text = utils_text.replace(OLD_READ, NEW_READ)
    utils_py.write_text(utils_text, encoding="utf-8")
    print("[OK] utils.py patched: read_data_from_file now handles {'root':[...]} format")
elif NEW_READ in utils_text:
    print("[SKIP] utils.py already patched")
else:
    print("[WARN] Could not find expected read_data_from_file signature — check manually")


# ============================================================================
# SECTION 9: STEP 7 — Patch QZ_Town.yaml: backslash -> "/" + fix paths
# ============================================================================

section("Step 7/8 — Patching Regions_params/QZ_Town.yaml (Linux path compat)")

qz_yaml = BENCH_ROOT / "Regions_params" / "QZ_Town.yaml"
with open(qz_yaml, "r", encoding="utf-8") as f:
    qz_cfg = yaml.safe_load(f)

# Fix backslash paths to forward slashes (Windows YAML on Linux Kaggle)
patched_lines = []
for line in qz_yaml.read_text(encoding="utf-8").splitlines():
    # Replace backslash separators in path-looking values only
    if "PATH" in line and "\\" in line:
        line = line.replace("\\", "/")
    patched_lines.append(line)

patched_yaml = "\n".join(patched_lines) + "\n"
qz_yaml.write_text(patched_yaml, encoding="utf-8")
print(f"[OK] {qz_yaml} patched (backslash -> /)")

# Verify the key paths will resolve from BENCH_ROOT cwd
with open(qz_yaml, "r", encoding="utf-8") as f:
    qz_check = yaml.safe_load(f)

for key in ["QZ_Town_HIGH_REF_PATH", "QZ_Town_HIGH_DSM_PATH"]:
    rel_path = qz_check.get(key, "").lstrip("./")
    abs_path = BENCH_ROOT / rel_path
    status = "[OK]" if abs_path.exists() else "[WARN] NOT FOUND"
    print(f"  {status} {key}: {abs_path}")


# ============================================================================
# SECTION 10: STEP 8 — Run Baseline.py
# ============================================================================

section("Step 8/8 — Running Benchmark Baseline")

# Final weight check
missing_weights = []
for rel, name in [(CAMP_WEIGHT_REL, "CAMP"), (ROMA_WEIGHT_REL, "RoMa"), (DINO_WEIGHT_REL, "DINOv2")]:
    p = BENCH_ROOT / rel
    if not p.exists():
        missing_weights.append((name, p))

if missing_weights:
    print("\n[ERROR] Missing weights, cannot run:")
    for name, p in missing_weights:
        print(f"  - {name}: {p}")
    raise SystemExit(
        "\nPlease provide the missing weights (see instructions above) and rerun."
    )

# Build cmd
cmd = [sys.executable, "Baseline.py"]
for k, v in BASELINE_ARGS.items():
    cmd += [k, v]

print("\nCommand:", " ".join(str(c) for c in cmd))
print(f"Working dir: {BENCH_ROOT}\n")

os.makedirs(BASELINE_ARGS["--save_dir"], exist_ok=True)

# Run from BENCH_ROOT so all relative imports/paths resolve correctly
subprocess.check_call([str(c) for c in cmd], cwd=str(BENCH_ROOT))

print(f"\n[DONE] Results saved to: {BASELINE_ARGS['--save_dir']}")
