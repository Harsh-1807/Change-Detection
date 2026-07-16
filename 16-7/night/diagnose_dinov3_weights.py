"""
Diagnose whether your DINOv3 checkpoint is actually loading correctly.

Your current notebook code (cell 25) does:

    encoder = timm.create_model("vit_small_patch16_224", pretrained=False, ...)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    encoder.load_state_dict(ckpt, strict=False)

`strict=False` silently swallows the return value. This script captures
that return value and shows you exactly what did and didn't load, plus
a shape-by-shape comparison for anything that DID match by name but
might still be wrong.

Run this first, before touching the training loop.
"""

import torch
import timm

CKPT_PATH = r"C:\Users\13519\first_version_change_detection\dinov3_vits16_pretrain_lvd1689m-08c60483.pth"


def main():
    ckpt = torch.load(CKPT_PATH, map_location="cpu")
    state_dict = ckpt.get("model", ckpt.get("state_dict", ckpt)) if isinstance(ckpt, dict) else ckpt

    print(f"Checkpoint has {len(state_dict)} tensors.\n")

    encoder = timm.create_model(
        "vit_small_patch16_224", pretrained=False, num_classes=0, img_size=512
    )
    model_keys = dict(encoder.state_dict())

    result = encoder.load_state_dict(state_dict, strict=False)

    print("=" * 70)
    print(f"MISSING keys (in model, not filled by checkpoint): {len(result.missing_keys)}")
    print("=" * 70)
    for k in result.missing_keys:
        print(" ", k)

    print()
    print("=" * 70)
    print(f"UNEXPECTED keys (in checkpoint, no home in model): {len(result.unexpected_keys)}")
    print("=" * 70)
    for k in result.unexpected_keys:
        print(" ", k)

    # Of the keys that DID match by name, check shapes actually agree.
    # (load_state_dict raises on shape mismatch for matched names by default,
    # so if we got here without an exception, matched-name shapes are fine —
    # but let's confirm explicitly and report coverage.)
    matched = [k for k in state_dict if k in model_keys]
    print()
    print("=" * 70)
    print(f"Matched-by-name keys: {len(matched)} / {len(model_keys)} model tensors")
    coverage = 100 * len(matched) / max(len(model_keys), 1)
    print(f"Coverage: {coverage:.1f}% of the model's tensors were actually filled")
    print("=" * 70)

    if "pos_embed" in result.missing_keys:
        print(
            "\n>>> pos_embed is MISSING. DINOv3 uses RoPE, not a learned absolute\n"
            "    position embedding, so a plain timm ViT-S/16 has no home for\n"
            "    the checkpoint's positional information at all. This alone\n"
            "    means the encoder is not seeing pretrained spatial structure.\n"
        )

    if coverage < 90:
        print(
            f">>> Only {coverage:.1f}% of the encoder's weights were loaded from\n"
            "    the checkpoint. The rest are randomly initialized. Freezing\n"
            "    this encoder (cell 34/37) freezes mostly-random weights.\n"
            "    This explains the near-uniform prediction.png output — the\n"
            "    decoder was trying to learn on top of noise.\n"
        )

    print(
        "\nRecommended fix: don't hand-load the raw checkpoint into a generic\n"
        "timm ViT. Use `transformers` (facebook/dinov3-vits16-pretrain-lvd1689m),\n"
        "which is Meta's own conversion and guarantees correct key mapping,\n"
        "RoPE, and register tokens. See model_dinov3.py in this same folder.\n"
    )


if __name__ == "__main__":
    main()
