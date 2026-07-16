"""
Corrected DINOv3 vegetation segmentation model.

Why this replaces your cell 25 `DinoV3Vegetation`:
  - `timm.create_model("vit_small_patch16_224", ...)` is a PLAIN ViT. DINOv3
    is architecturally different (RoPE instead of learned pos_embed, plus
    4 register tokens). Loading the DINOv3 checkpoint into that plain ViT
    with strict=False silently drops the mismatched keys.
  - This version loads the model through `transformers`, using Meta's own
    HF conversion, so the architecture and weights are guaranteed to match.

One tradeoff: this downloads a (separate, HF-format) copy of the same
weights the first time you run it — a few hundred MB, one-time, then cached
locally by the `transformers` library. If you'd rather reuse your existing
.pth exactly as-is, see the note at the bottom of this file for the
torch.hub alternative instead.

Install once:
    pip install transformers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

HF_MODEL_ID = "facebook/dinov3-vits16-pretrain-lvd1689m"

# ViT-S/16: hidden size 384, patch size 16, 1 CLS + 4 register tokens
HIDDEN_DIM = 384
PATCH_SIZE = 16
NUM_PREFIX_TOKENS = 5  # 1 CLS + 4 register tokens — NOT just 1


class DinoV3Vegetation(nn.Module):
    def __init__(self, freeze_encoder: bool = True):
        super().__init__()

        self.encoder = AutoModel.from_pretrained(HF_MODEL_ID)

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.decoder = nn.Sequential(
            nn.Conv2d(HIDDEN_DIM, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1),
        )

    def set_encoder_trainable(self, trainable: bool):
        for p in self.encoder.parameters():
            p.requires_grad = trainable

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ViT encoder only, return the (B, 384, h, w) patch feature map."""
        B, C, H, W = x.shape
        h, w = H // PATCH_SIZE, W // PATCH_SIZE

        out = self.encoder(pixel_values=x)
        tokens = out.last_hidden_state  # (B, num_prefix + h*w, 384)

        patch_tokens = tokens[:, NUM_PREFIX_TOKENS:, :]
        assert patch_tokens.shape[1] == h * w, (
            f"Expected {h * w} patch tokens, got {patch_tokens.shape[1]}. "
            "Input size must be a multiple of 16."
        )

        feat = patch_tokens.transpose(1, 2).reshape(B, HIDDEN_DIM, h, w)
        return feat

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        feat = self.encode(x)
        logits = self.decoder(feat)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        return logits

    def forward_from_features(self, feat: torch.Tensor, out_size) -> torch.Tensor:
        """Decoder-only forward pass, for training on cached encoder features."""
        logits = self.decoder(feat)
        logits = F.interpolate(logits, size=out_size, mode="bilinear", align_corners=False)
        return logits


# ---------------------------------------------------------------------------
# Alternative: keep using your original local .pth file, no re-download.
# Requires cloning the official repo once:
#     git clone https://github.com/facebookresearch/dinov3
# Then:
#
#   import torch
#   REPO_DIR = r"C:\path\to\dinov3"
#   encoder = torch.hub.load(
#       REPO_DIR, "dinov3_vits16", source="local",
#       weights=r"C:\Users\13519\first_version_change_detection\dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
#   )
#
# This constructs the REAL DINOv3 architecture (not a timm stand-in) and
# loads your exact existing checkpoint file into it directly.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    model = DinoV3Vegetation()
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        y = model(x)
    print("Output shape:", y.shape)  # expect (1, 1, 512, 512)
