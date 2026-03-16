### EXP02 — UAV-AVL Baseline (Reference-Map Tile Matching)

**Run date**: _Kaggle session (latest log provided)_  
**Script**: `exp02_uavavl_baseline_kaggle.py`  
**Dataset**: `hunhtrungkit/uav-avl` — region `QZ_Town`  
**Backbone**: `DINOv2 ViT-B/14` (`dinov2_vitb14`)  
**Image size**: `392 × 392` (patch-friendly for ViT-B/14)  
**Loss**: Symmetric InfoNCE (temperature = 0.1)  
**Training pairs**: 269 UAV images with ≥1 training tile (radius 10m or nearest-tile fallback)  
**Hardware**: Kaggle GPU (`cuda`, H100 according to comments)

#### Hyperparameters

- **Epochs**: 20  
- **Batch size**: 32  
- **Optimizer**: AdamW (`lr=1e-4`, `weight_decay=0.03`)  
- **Scheduler**: CosineAnnealingLR (`T_max=20`, `eta_min=1e-6`)  
- **Warmup / unfreeze**: Freeze DINOv2 backbone, unfreeze last 4 blocks at epoch 3 with `lr=1e-5`  
- **POS_RADIUS** (tile match): 10 m  
- **Eval thresholds (PDM)**: 5 m, 10 m, 25 m  

#### Validation performance (best checkpoint)

_Final evaluation after training, loading `best_uavavl_baseline.pth` (selected by highest `PDM@1_@10m`)._

- **Retrieval metrics**
  - **R@1**: 0.0580  
  - **R@5**: 0.1739  
  - **R@10**: 0.3188  

- **PDM@K (Position-Dependent Match)**
  - **PDM@1 @5 m**: 0.0000  
  - **PDM@1 @10 m**: 0.0580  
  - **PDM@1 @25 m**: 0.1594  
  - **PDM@5 @5 m**: 0.0435  
  - **PDM@5 @10 m**: 0.1739  
  - **PDM@5 @25 m**: 0.4783  
  - **PDM@10 @5 m**: 0.1014  
  - **PDM@10 @10 m**: 0.3188  
  - **PDM@10 @25 m**: 0.6377  

- **Localization error**
  - **Mean error**: 67.64 m  
  - **Median error**: 47.66 m  

- **Dataset stats**
  - **# Gallery tiles**: 4658  
  - **# Query UAV images (val)**: 69  

- **Inference timing**
  - **Query feature extraction**: 22.79 s total → 330.30 ms / query  
  - **Gallery feature extraction**: 222.90 s total → 47.85 ms / tile  
  - **Retrieval computation**: ~0.00 s (matrix similarity on pre-computed features)

#### Training dynamics (loss)

- Epoch 1 → 5: loss 3.50 → 2.84 (rapid initial convergence)  
- Epoch 10: loss 1.92  
- Epoch 15: loss 1.54  
- Epoch 20: loss 1.40  

Loss consistently decreases, with modest but non-zero gains in retrieval (R@K) and PDM metrics after CRS and tiling fixes.

