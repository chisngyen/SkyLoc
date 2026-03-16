import os
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms


@dataclass
class SPDGeoConfig:
    """
    Minimal config mirroring `exps/Geo_Localization.py` embedding extractor.
    Defaults are set to match the exported checkpoints used in this repo.
    """

    IMG_SIZE: int = 336
    EMBED_DIM: int = 512
    PART_DIM: int = 256
    N_PARTS: int = 8
    CLUSTER_TEMP: float = 0.07
    NUM_ALTITUDES: int = 4
    UNFREEZE_BLOCKS: int = 6


class DINOv2Backbone(nn.Module):
    def __init__(self, unfreeze_blocks: int = 6):
        super().__init__()
        self.model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
        self.feature_dim = 384
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
        features = self.model.forward_features(x)
        patch_tokens = features["x_norm_patchtokens"]
        cls_token = features["x_norm_clstoken"]
        H = x.shape[2] // self.patch_size
        W = x.shape[3] // self.patch_size
        return patch_tokens, cls_token, (H, W)


class DeepAltitudeFiLM(nn.Module):
    def __init__(self, num_altitudes: int = 4, feat_dim: int = 256):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(num_altitudes, feat_dim))
        self.beta = nn.Parameter(torch.zeros(num_altitudes, feat_dim))

    def forward(self, feat, alt_idx=None):
        if alt_idx is None:
            gamma = self.gamma.mean(dim=0, keepdim=True)
            beta = self.beta.mean(dim=0, keepdim=True)
            return feat * gamma.unsqueeze(0) + beta.unsqueeze(0)
        gamma = self.gamma[alt_idx]
        beta = self.beta[alt_idx]
        return feat * gamma.unsqueeze(1) + beta.unsqueeze(1)


class AltitudeAwarePartDiscovery(nn.Module):
    def __init__(
        self,
        feat_dim: int = 384,
        n_parts: int = 8,
        part_dim: int = 256,
        temperature: float = 0.07,
        num_altitudes: int = 4,
    ):
        super().__init__()
        self.n_parts = n_parts
        self.temperature = temperature
        self.feat_proj = nn.Sequential(nn.Linear(feat_dim, part_dim), nn.LayerNorm(part_dim), nn.GELU())
        self.altitude_film = DeepAltitudeFiLM(num_altitudes, part_dim)
        self.prototypes = nn.Parameter(torch.randn(n_parts, part_dim) * 0.02)
        self.refine = nn.Sequential(
            nn.LayerNorm(part_dim),
            nn.Linear(part_dim, part_dim * 2),
            nn.GELU(),
            nn.Linear(part_dim * 2, part_dim),
        )
        self.salience_head = nn.Sequential(nn.Linear(part_dim, 64), nn.GELU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, patch_features, spatial_hw, alt_idx=None):
        B, N, _ = patch_features.shape
        H, W = spatial_hw

        feat = self.feat_proj(patch_features)
        feat = self.altitude_film(feat, alt_idx)

        feat_norm = F.normalize(feat, dim=-1)
        proto_norm = F.normalize(self.prototypes, dim=-1)
        sim = torch.einsum("bnd,kd->bnk", feat_norm, proto_norm) / self.temperature
        assign = F.softmax(sim, dim=-1)

        assign_t = assign.transpose(1, 2)
        mass = assign_t.sum(-1, keepdim=True).clamp(min=1e-6)
        part_feat = torch.bmm(assign_t, feat) / mass
        part_feat = part_feat + self.refine(part_feat)

        # salience
        salience = self.salience_head(part_feat).squeeze(-1)
        return {"part_features": part_feat, "salience": salience}


class PartAwarePooling(nn.Module):
    def __init__(self, part_dim: int = 256, embed_dim: int = 512):
        super().__init__()
        self.attn = nn.Sequential(nn.Linear(part_dim, part_dim // 2), nn.Tanh(), nn.Linear(part_dim // 2, 1))
        self.proj = nn.Sequential(
            nn.Linear(part_dim * 3, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def forward(self, part_features, salience=None):
        aw = self.attn(part_features)
        if salience is not None:
            aw = aw + salience.unsqueeze(-1).log().clamp(-10)
        aw = F.softmax(aw, dim=1)
        attn_pool = (aw * part_features).sum(1)
        mean_pool = part_features.mean(1)
        max_pool = part_features.max(1)[0]
        combined = torch.cat([attn_pool, mean_pool, max_pool], dim=-1)
        return F.normalize(self.proj(combined), dim=-1)


class DynamicFusionGate(nn.Module):
    def __init__(self, embed_dim: int = 512):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(embed_dim // 2, 1),
        )
        nn.init.constant_(self.gate[-1].bias, 0.85)

    def forward(self, part_emb, cls_emb):
        alpha = torch.sigmoid(self.gate(torch.cat([part_emb, cls_emb], dim=-1)))
        fused = alpha * part_emb + (1 - alpha) * cls_emb
        return F.normalize(fused, dim=-1)


class SPDGeoDPEAMARModel(nn.Module):
    def __init__(self, cfg: SPDGeoConfig = SPDGeoConfig()):
        super().__init__()
        self.cfg = cfg
        self.backbone = DINOv2Backbone(cfg.UNFREEZE_BLOCKS)
        self.part_disc = AltitudeAwarePartDiscovery(
            feat_dim=384,
            n_parts=cfg.N_PARTS,
            part_dim=cfg.PART_DIM,
            temperature=cfg.CLUSTER_TEMP,
            num_altitudes=cfg.NUM_ALTITUDES,
        )
        self.pool = PartAwarePooling(cfg.PART_DIM, cfg.EMBED_DIM)
        self.fusion_gate = DynamicFusionGate(cfg.EMBED_DIM)
        self.cls_proj = nn.Sequential(nn.Linear(384, cfg.EMBED_DIM), nn.BatchNorm1d(cfg.EMBED_DIM), nn.ReLU(inplace=True))

    def extract_embedding(self, x, alt_idx=None):
        patches, cls_tok, hw = self.backbone(x)
        parts = self.part_disc(patches, hw, alt_idx=alt_idx)
        part_emb = self.pool(parts["part_features"], parts["salience"])
        cls_emb = F.normalize(self.cls_proj(cls_tok), dim=-1)
        return self.fusion_gate(part_emb, cls_emb)

    def forward(self, x):
        return self.extract_embedding(x, alt_idx=None)


def get_spdgeo_transforms(img_size: int = 336):
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def get_SPDGeo_model(checkpoint_path: str | None = None, device: str | None = None):
    """
    Returns (model, transform) in the same style as other Benchmark retrieval loaders.

    checkpoint_path:
      - if provided: load weights from that path
      - else: try env var SPDGEO_CKPT
      - else: return randomly initialized heads on top of DINOv2 backbone (not recommended)
    """
    cfg = SPDGeoConfig()
    model = SPDGeoDPEAMARModel(cfg)
    ckpt = checkpoint_path or os.environ.get("SPDGEO_CKPT")
    if ckpt and os.path.exists(ckpt):
        state = torch.load(ckpt, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        print(f"[SPDGeo] Loaded checkpoint: {ckpt}")
        if missing:
            print(f"[SPDGeo][WARN] Missing keys: {len(missing)}")
        if unexpected:
            print(f"[SPDGeo][WARN] Unexpected keys: {len(unexpected)}")
    else:
        if ckpt:
            print(f"[SPDGeo][WARN] Checkpoint not found: {ckpt} (using default init)")
        else:
            print("[SPDGeo][WARN] No checkpoint provided (SPDGEO_CKPT not set); using default init")

    if device:
        model = model.to(device)

    return model, get_spdgeo_transforms(cfg.IMG_SIZE)

