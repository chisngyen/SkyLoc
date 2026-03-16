# SkyLoc — Absolute Visual Localization for UAVs 🛸

> *From map-matching to meter-level absolute positioning*

Research codebase for cross-view geo-localization and absolute visual localization (AVL) of UAVs in GNSS-denied environments. Targeting NeurIPS submission.

## 🎯 Direction

**Temporal-Geometric Absolute Localization (TG-AVL)** — combining temporal trajectory modeling with geometric feature disentanglement for robust, trajectory-aware drone localization.

## 📁 Structure

```
SkyLoc/
├── exps/                    # Self-contained Kaggle experiment scripts
│   ├── exp01_denseuav_baseline_kaggle.py   # DenseUAV baseline (DINOv2 + InfoNCE)
│   └── exp02_uavavl_baseline_kaggle.py     # UAV-AVL baseline (tile matching)
├── papers/                  # Reference papers (PDF)
├── explore_datasets.py      # Dataset structure explorer for Kaggle
└── README.md
```

## 📊 Datasets

| Dataset | Type | Scale | Metrics |
|---------|------|-------|---------|
| [DenseUAV](https://github.com/Dmmm1997/DenseUAV) | Cross-view retrieval | 3033 locations | R@K, SDM@K, MA@Xm |
| [UAV-AVL / AnyVisLoc](https://github.com/AnyVisLoc) | Ref-map localization | 487 UAV images | PDM@K, R@K |

## 🚀 Quick Start (Kaggle H100)

1. Attach datasets: `chisboiz/denseuav` and `hunhtrungkit/uav-avl`
2. Copy any `exps/exp*.py` into a Kaggle notebook cell
3. Run — results saved to `/kaggle/working/results_*.json`

## 🏗️ Baseline Architecture

- **Backbone**: DINOv2 ViT-B/14 (frozen → progressive unfreeze)
- **Loss**: Symmetric InfoNCE with GPS-based hard-negative mining
- **Training**: AdamW + CosineAnnealing, mixed-precision (AMP)

## 📝 License

MIT
