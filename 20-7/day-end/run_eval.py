"""
Main evaluation script.
Run: python run_eval.py
"""

import json
from pathlib import Path

import torch
import numpy as np
from eval_config import CACHE_ROOT, RGB_ROOT, MODEL_PATH, OUT_ROOT, THRESHOLD
from eval_utils import evaluate_tile, print_stats
from eval_plots import plot_kde_simple, create_dashboard, save_mosaic_plot
from eval_mosaic import build_mosaic
from eval_visualize import visualize_sample

from dataset import VegetationDataset
from model_dinov3 import DinoV3Vegetation


def load_model(model_path):
    """Load trained model."""
    model = DinoV3Vegetation(freeze_encoder=True)
    checkpoint = torch.load(model_path, map_location="cpu")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def run_evaluation(sample_files, model, threshold):
    """Evaluate all tiles and return categorized metrics."""
    all_metrics = []
    nonempty_metrics = []
    empty_metrics = []

    print("\n" + "=" * 70)
    print("PHASE 1: Evaluating ALL validation tiles")
    print("=" * 70)

    for idx, sample_file in enumerate(sample_files):
        sample = torch.load(sample_file, weights_only=True)
        result = evaluate_tile(sample, model, threshold)

        metrics = {
            "idx": idx,
            "filename": sample_file.name,
            "gt_sum": result["mask_sum"],
            "gt_empty": result["is_empty"],
            "pred_sum": result["pred_sum"],
            "pred_empty": result["pred_empty"],
            "dice": result["dice"],
            "iou": result["iou"],
            "precision": result["precision"],
            "recall": result["recall"],
            "f1": result["f1"],
            "pcc": result["pcc"],
            "pred_min": result["pred_min"],
            "pred_max": result["pred_max"],
            "pred_mean": result["pred_mean"],
            "pred_std": result["pred_std"],
        }

        all_metrics.append(metrics)

        if result["is_empty"]:
            empty_metrics.append(metrics)
        else:
            nonempty_metrics.append(metrics)

        if (idx + 1) % 100 == 0 or idx == len(sample_files) - 1:
            print(f"  Processed {idx + 1}/{len(sample_files)} tiles...")

    return all_metrics, nonempty_metrics, empty_metrics


def save_metrics(all_metrics, nonempty_metrics, empty_metrics, out_root):
    """Save metrics to JSON files."""
    # Convert numpy types to Python native types for JSON serialization
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_for_json(v) for v in obj]
        elif isinstance(obj, (np.bool_, np.integer)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    all_metrics_clean = clean_for_json(all_metrics)
    nonempty_clean = clean_for_json(nonempty_metrics)
    empty_clean = clean_for_json(empty_metrics)

    with open(out_root / "all_metrics.json", "w") as f:
        json.dump(all_metrics_clean, f, indent=2)
    with open(out_root / "nonempty_metrics.json", "w") as f:
        json.dump(nonempty_clean, f, indent=2)
    with open(out_root / "empty_metrics.json", "w") as f:
        json.dump(empty_clean, f, indent=2)

    print(f"\nSaved metrics: {len(all_metrics)} total | {len(nonempty_metrics)} non-empty | {len(empty_metrics)} empty")


def generate_kde_plots(nonempty_metrics, out_root):
    """Generate all KDE plots."""
    print("\n" + "=" * 70)
    print("PHASE 3: Creating Clean KDE Plots")
    print("=" * 70)

    plot_kde_simple([m["dice"] for m in nonempty_metrics], "Dice",
                    out_root / "kde_dice.png", "#2ecc71")
    plot_kde_simple([m["iou"] for m in nonempty_metrics], "IoU",
                    out_root / "kde_iou.png", "#3498db")

    pcc_valid = [m["pcc"] for m in nonempty_metrics if m["pcc"] is not None]
    if pcc_valid:
        plot_kde_simple(pcc_valid, "PCC", out_root / "kde_pcc.png", "#9b59b6")

    plot_kde_simple([m["recall"] for m in nonempty_metrics], "Recall",
                    out_root / "kde_recall.png", "#e74c3c")
    plot_kde_simple([m["precision"] for m in nonempty_metrics], "Precision",
                    out_root / "kde_precision.png", "#f39c12")


def generate_mosaic(sample_files, model, threshold, out_root):
    """Generate full mosaic overview."""
    print("\n" + "=" * 70)
    print("PHASE 5: Creating Full Mosaic Overview")
    print("=" * 70)

    mosaic_gt, mosaic_pred, mosaic_diff = build_mosaic(sample_files, model, threshold)
    save_mosaic_plot(mosaic_gt, mosaic_pred, mosaic_diff, len(sample_files),
                     out_root / "full_mosaic_overview.png")


def generate_best_worst(nonempty_metrics, sample_files, rgb_ds, model, threshold, out_root):
    """Visualize best and worst non-empty cases."""
    print("\n" + "=" * 70)
    print("PHASE 6: Visualizing Best/Worst Non-Empty Cases")
    print("=" * 70)

    nonempty_sorted = sorted(nonempty_metrics, key=lambda x: x["dice"])
    worst_10 = nonempty_sorted[:10]
    best_10 = nonempty_sorted[-10:]

    for m in worst_10:
        visualize_sample(m["idx"], sample_files, rgb_ds, model, threshold, out_root, "worst_nonempty")

    for m in best_10:
        visualize_sample(m["idx"], sample_files, rgb_ds, model, threshold, out_root, "best_nonempty")

    print(f"  Visualized {len(worst_10)} worst and {len(best_10)} best non-empty cases")


def main():
    """Main entry point."""
    OUT_ROOT.mkdir(exist_ok=True)

    model = load_model(MODEL_PATH)
    rgb_ds = VegetationDataset(RGB_ROOT)
    sample_files = sorted(CACHE_ROOT.glob("*.pt"))

    if not sample_files:
        raise RuntimeError(f"No cached samples found in {CACHE_ROOT}")

    # Phase 1: Evaluate
    all_metrics, nonempty_metrics, empty_metrics = run_evaluation(sample_files, model, THRESHOLD)

    # Phase 2: Save & Summarize
    save_metrics(all_metrics, nonempty_metrics, empty_metrics, OUT_ROOT)

    print("\n" + "=" * 70)
    print("PHASE 2: Summary Statistics")
    print("=" * 70)
    print_stats(all_metrics, "ALL TILES")
    print_stats(nonempty_metrics, "NON-EMPTY TILES")
    print_stats(empty_metrics, "EMPTY TILES")

    # Phase 3: KDE Plots
    generate_kde_plots(nonempty_metrics, OUT_ROOT)

    # Phase 4: Dashboard
    print("\n" + "=" * 70)
    print("PHASE 4: Creating Overview Dashboard")
    print("=" * 70)
    create_dashboard(nonempty_metrics, empty_metrics, all_metrics,
                     OUT_ROOT / "overview_dashboard.png", THRESHOLD)

    # Phase 5: Mosaic
    generate_mosaic(sample_files, model, THRESHOLD, OUT_ROOT)

    # Phase 6: Best/Worst
    generate_best_worst(nonempty_metrics, sample_files, rgb_ds, model, THRESHOLD, OUT_ROOT)

    print("\n" + "=" * 70)
    print("DONE — All outputs in:", OUT_ROOT)
    print("=" * 70)


if __name__ == "__main__":
    main()