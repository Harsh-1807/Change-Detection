import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------------------
# DINOv3 local repo
# --------------------------------------------------

DINO_REPO = r"C:\Users\13519\first_version_change_detection\dinov3"

if DINO_REPO not in sys.path:
    sys.path.append(DINO_REPO)

# --------------------------------------------------
# Local checkpoint
# --------------------------------------------------

CKPT_PATH = r"C:\Users\13519\first_version_change_detection\dinov3_vits16_pretrain_lvd1689m-08c60483.pth"

# --------------------------------------------------
# Parameters
# --------------------------------------------------

HIDDEN_DIM = 384
PATCH_SIZE = 16

# --------------------------------------------------
# Model
# --------------------------------------------------

class DinoV3Vegetation(nn.Module):

    def __init__(self, freeze_encoder=True):

        super().__init__()

        self.encoder = torch.hub.load(
            DINO_REPO,
            "dinov3_vits16",
            source="local",
            weights=CKPT_PATH
        )

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.decoder = nn.Sequential(

            nn.Conv2d(
                HIDDEN_DIM,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                256,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                256,
                128,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                128,
                64,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                64,
                1,
                kernel_size=1
            )
        )

    def set_encoder_trainable(self, trainable):

        for p in self.encoder.parameters():
            p.requires_grad = trainable

    def encode(self, x):

        B, C, H, W = x.shape

        h = H // PATCH_SIZE
        w = W // PATCH_SIZE

        outputs = self.encoder.forward_features(x)

        # DINOv3 repo returns normalized patch tokens
        patch_tokens = outputs["x_norm_patchtokens"]

        feat = (
            patch_tokens
            .transpose(1, 2)
            .reshape(B, HIDDEN_DIM, h, w)
        )

        return feat

    def forward(self, x):

        H = x.shape[2]
        W = x.shape[3]

        feat = self.encode(x)

        logits = self.decoder(feat)

        logits = F.interpolate(
            logits,
            size=(H, W),
            mode="bilinear",
            align_corners=False
        )

        return logits
    
    def forward_from_features(self, feat, out_size):

        logits = self.decoder(feat)

        logits = F.interpolate(
            logits,
            size=out_size,
            mode="bilinear",
            align_corners=False
        )

        return logits


# --------------------------------------------------
# Quick test
# --------------------------------------------------
if __name__ == "__main__":

    model = DinoV3Vegetation(
        freeze_encoder=True
    )

    x = torch.randn(
        1,
        3,
        512,
        512
    )

    with torch.no_grad():
        y = model(x)

    print(y.shape)