import os
from pathlib import Path
from collections import Counter

IMAGE_EXTS = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}


def _snapshot(directory):
    """Return a hashable 'fingerprint' of a directory's direct contents:
    (num_subdirs, frozenset_of_file_extensions, num_files).
    Used to decide whether a batch of sibling folders look alike."""
    try:
        entries = list(os.scandir(directory))
    except PermissionError:
        return None
    subdirs = sum(1 for e in entries if e.is_dir())
    exts = frozenset(
        Path(e.name).suffix.lower() for e in entries if e.is_file()
    )
    n_files = sum(1 for e in entries if e.is_file())
    return (subdirs, exts, n_files)


def _count_images(directory):
    """Count image files directly inside *directory*."""
    try:
        return sum(
            1 for e in os.scandir(directory)
            if e.is_file() and Path(e.name).suffix.lower() in IMAGE_EXTS
        )
    except PermissionError:
        return 0


def print_tree(directory, prefix="", max_depth=None, _depth=0):
    """
    Print a directory tree that is friendly for huge datasets:

    * Image files (.png/.jpg/…) are never listed individually – only counted.
    * When a directory contains many sub-folders with a *similar* internal
      structure (e.g. folders 0001-2000 each holding ~50 .jpg), they are
      collapsed into ONE summary line instead of being printed one-by-one.
    """
    if max_depth is not None and _depth > max_depth:
        return

    try:
        entries = sorted(os.scandir(directory), key=lambda e: e.name)
    except PermissionError:
        print(prefix + "└── [Permission Denied]")
        return

    dirs  = [e for e in entries if e.is_dir()]
    files = [e for e in entries if e.is_file()]

    images      = [f for f in files if Path(f.name).suffix.lower() in IMAGE_EXTS]
    other_files = [f for f in files if Path(f.name).suffix.lower() not in IMAGE_EXTS]

    # --- Collapse similar sub-folders -----------------------------------------
    # Group dirs by their internal "snapshot". If a snapshot appears for ≥ 5
    # dirs we collapse them into a single summary line.
    COLLAPSE_THRESHOLD = 5
    snapshot_to_dirs = {}
    for d in dirs:
        snap = _snapshot(d.path)
        snapshot_to_dirs.setdefault(snap, []).append(d)

    # Split into groups that will be collapsed vs. printed individually
    collapsed_groups = []   # list of (snapshot, [dir_entries])
    individual_dirs  = []
    for snap, group in snapshot_to_dirs.items():
        if len(group) >= COLLAPSE_THRESHOLD:
            collapsed_groups.append((snap, group))
        else:
            individual_dirs.extend(group)
    individual_dirs.sort(key=lambda e: e.name)

    all_items = []  # list of (label_str, recurse_path_or_None)

    # 1) Collapsed groups
    for snap, group in collapsed_groups:
        group.sort(key=lambda e: e.name)
        first, last = group[0].name, group[-1].name
        avg_imgs = sum(_count_images(d.path) for d in group) // max(len(group), 1)
        ext_str = ", ".join(sorted(snap[1])) if snap and snap[1] else "empty"
        all_items.append((
            f"[{len(group)} folders: {first}/ … {last}/]  "
            f"(each ~{avg_imgs} images, extensions: {ext_str})",
            None,
        ))

    # 2) Individual directories
    for d in individual_dirs:
        all_items.append((f"{d.name}/", d.path))

    # 3) Non-image files (print names)
    for f in other_files:
        all_items.append((f.name, None))

    # 4) Image summary
    if images:
        ext_counter = Counter(Path(f.name).suffix.lower() for f in images)
        parts = ", ".join(f"{cnt} {ext}" for ext, cnt in ext_counter.most_common())
        all_items.append((f"[{len(images)} images: {parts}]", None))

    # --- Render ---------------------------------------------------------------
    for idx, (label, recurse) in enumerate(all_items):
        is_last = idx == len(all_items) - 1
        connector  = "└── " if is_last else "├── "
        extension  = "    " if is_last else "│   "
        print(f"{prefix}{connector}{label}")
        if recurse is not None:
            print_tree(recurse, prefix + extension,
                       max_depth=max_depth, _depth=_depth + 1)


def main():
    datasets = [
        "/kaggle/input/datasets/hunhtrungkit/uav-avl/Data",
        "/kaggle/input/datasets/chisboiz/denseuav/DenseUAV",
    ]

    for dataset_path in datasets:
        print(f"\n{'=' * 60}")
        print(f"  Dataset: {dataset_path}")
        print(f"{'=' * 60}")

        if os.path.exists(dataset_path):
            print(f"{dataset_path}/")
            print_tree(dataset_path)
        else:
            print(f"  [NOT FOUND] {dataset_path}")
            print("  → Make sure this Kaggle dataset is attached to the notebook.")
        print()


if __name__ == "__main__":
    main()
