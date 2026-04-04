"""
model.py — EfficientNet-B0 + Temporal Transformer for engagement detection.

Architecture:
  1. CNN backbone (EfficientNet-B0) extracts a feature vector per frame.
  2. A learned linear projection reduces that vector to d_model dimensions.
  3. Learned positional embeddings are added to encode temporal position.
  4. A Transformer encoder attends across all frames in the window.
  5. Mean pooling over time collapses the sequence to one vector.
  6. Four independent sigmoid regression heads output [0,1] scores for
     boredom, engagement, confusion, and frustration.

Design rationale:
  - EfficientNet-B0 is chosen over larger variants because it needs to be
    exported and run in real-time on a demo laptop CPU at inference time.
    Its 1280-dim features are rich enough; the Transformer compensates for
    the lighter backbone with temporal context.
  - Mean pooling instead of a [CLS] token simplifies ONNX export and is
    slightly more stable under short sequences.
  - Four separate heads (vs. one shared head) let each dimension learn an
    independent decision boundary, which matters because DAiSEE dimensions
    are only weakly correlated.
"""

import torch
import torch.nn as nn
import timm

from config import CFG, ModelConfig, LABEL_NAMES


class EngagementModel(nn.Module):

    def __init__(self, cfg: ModelConfig = CFG.model):
        super().__init__()
        self.cfg    = cfg
        self.labels = LABEL_NAMES

        # -----------------------------------------------------------
        # 1. CNN backbone — spatial feature extractor per frame
        # -----------------------------------------------------------
        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=0,          # remove classification head
            global_pool="avg",      # global average pool → flat vector
        )
        feat_dim = self.backbone.num_features   # 1280 for EfficientNet-B0

        # -----------------------------------------------------------
        # 2. Feature projection into Transformer space
        # -----------------------------------------------------------
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, cfg.d_model),
            nn.LayerNorm(cfg.d_model),
        )

        # -----------------------------------------------------------
        # 3. Learnable positional embeddings
        # -----------------------------------------------------------
        self.pos_embed = nn.Parameter(
            torch.randn(1, cfg.seq_len, cfg.d_model) * 0.02
        )

        # -----------------------------------------------------------
        # 4. Temporal Transformer encoder
        # -----------------------------------------------------------
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_model * 4,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,     # (batch, seq, features) convention
            norm_first=True,      # Pre-LN: more stable gradient flow
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=cfg.n_layers,
        )

        # -----------------------------------------------------------
        # 5. Independent regression head per output dimension
        # -----------------------------------------------------------
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(cfg.d_model),
                nn.Dropout(cfg.dropout),
                nn.Linear(cfg.d_model, cfg.d_model // 2),
                nn.GELU(),
                nn.Linear(cfg.d_model // 2, 1),
                nn.Sigmoid(),
            )
            for _ in range(cfg.num_outputs)
        ])

        self._init_weights()

    # ------------------------------------------------------------------

    def _init_weights(self):
        """Xavier initialisation for projection and head layers."""
        for module in [self.input_proj, *self.heads]:
            for layer in module.modules():
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, seq_len, C, H, W)  float32

        Returns
        -------
        scores : (batch, 4)  float32 in [0, 1]
                 columns: boredom, engagement, confusion, frustration
        """
        B, T, C, H, W = x.shape

        # --- Backbone: process all frames in parallel ---
        x_flat = x.view(B * T, C, H, W)
        feats  = self.backbone(x_flat)           # (B*T, feat_dim)

        # --- Project + add positional embeddings ---
        feats = self.input_proj(feats)           # (B*T, d_model)
        feats = feats.view(B, T, -1)             # (B, T, d_model)
        feats = feats + self.pos_embed[:, :T, :] # broadcast over batch

        # --- Temporal Transformer ---
        ctx = self.transformer(feats)            # (B, T, d_model)

        # --- Mean pool over time window ---
        ctx = ctx.mean(dim=1)                    # (B, d_model)

        # --- Four independent regression heads ---
        scores = torch.cat([h(ctx) for h in self.heads], dim=-1)  # (B, 4)

        return scores

    # ------------------------------------------------------------------

    def predict_dict(self, x: torch.Tensor) -> list[dict]:
        """
        Convenience wrapper: returns a list of dicts (one per batch item).
        Useful for logging and downstream consumers.
        """
        with torch.no_grad():
            scores = self.forward(x).cpu().numpy()

        return [
            {name: float(scores[i, j]) for j, name in enumerate(self.labels)}
            for i in range(scores.shape[0])
        ]


# ---------------------------------------------------------------------------
# Model factory + parameter count helper
# ---------------------------------------------------------------------------

def build_model(cfg: ModelConfig = CFG.model, device: str = "cuda") -> EngagementModel:
    model = EngagementModel(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model: {cfg.backbone} + Transformer({cfg.n_layers}L, {cfg.n_heads}H, d={cfg.d_model})")
    print(f"Trainable parameters: {n_params / 1e6:.2f}M")
    return model


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model  = build_model(device=device)
    model.eval()

    # Simulate one batch
    dummy = torch.randn(4, CFG.model.seq_len, 3, 112, 112).to(device)

    with torch.no_grad():
        out = model(dummy)

    print(f"Input  shape : {dummy.shape}")    # (4, 10, 3, 112, 112)
    print(f"Output shape : {out.shape}")      # (4, 4)
    print(f"Output range : [{out.min():.3f}, {out.max():.3f}]")  # should be in [0,1]
    print(f"Sample output: {dict(zip(model.labels, out[0].tolist()))}")
