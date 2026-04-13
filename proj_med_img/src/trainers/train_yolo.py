"""
src/trainers/train_yolo.py

Main training functions for Ultralytics YOLO (v9 / v11 / v26) models.
- Takes model name + augmentation dictionary
- Trains the model with Ultralytics
- Saves epoch-wise metrics (perfect for matplotlib plotting)
- Saves final metrics summary (including precision, recall, F1, mAP@50, mAP@55 … mAP@95)
- Organizes outputs exactly as you requested:
    runs/<model_name>/
    ├── raw/<aug_name>/
    │   ├── results.csv              ← full epoch-by-epoch metrics (load with pandas)
    │   └── metrics_summary.csv      ← clean final metrics table
    └── plots/<aug_name>/
            └── *.png                ← all Ultralytics plots (F1_curve, P_curve, R_curve,
                                       PR_curve, confusion_matrix, etc.)
"""

from pathlib import Path
import shutil
import hashlib
import pandas as pd
from typing import Dict, Any, Optional
from ultralytics import YOLO
import torch

def _generate_aug_name(aug_dict: Dict[str, Any]) -> str:
    """Generate a short unique name from the augmentation dictionary (if no name provided)."""
    # Sort items so same config always gives same name
    aug_key = str(sorted(aug_dict.items()))
    short_hash = hashlib.md5(aug_key.encode()).hexdigest()[:8]
    return f"aug_{short_hash}"


def train_yolo(
    model_name: str,
    aug_dict: Dict[str, Any],
    data_yaml: str,
    aug_name: Optional[str] = None,
    epochs: int = 100,
    imgsz: int = 1080,
    batch: int = 16,
    device: str = "0",
    patience: int = 50,
    **kwargs: Any,
) -> None:
    """
    Train a single Ultralytics YOLO model with given augmentations.

    Args:
        model_name: e.g. "yolov9c", "yolo11m", "yolo26m", "yolov9e", etc.
        aug_dict: Dictionary of Ultralytics augmentation parameters
                  (hsv_h, degrees, mosaic, mixup, etc.)
        data_yaml: Path to your dataset YAML file.
        aug_name: name for the augmentation setting.
                  If None, a hash-based name is generated automatically.
        epochs, imgsz, batch, device, patience: Standard training parameters.
        **kwargs: Any extra Ultralytics train() arguments you want to pass.
    """
    # Generate aug_name if not provided
    if aug_name is None:
        aug_name = _generate_aug_name(aug_dict)

    # --------------------- Directory setup (exactly as you requested) ---------------------
    base_dir = Path("runs") / model_name
    raw_dir = base_dir / "raw" / aug_name
    plots_dir = base_dir / "plots" / aug_name

    raw_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    print(f"Starting training: {model_name} | Augmentation: {aug_name}")

    # --------------------- Ultralytics training ---------------------
    # Use project + name so Ultralytics saves everything under runs/<model_name>/<aug_name>/
    ultra_project = str(base_dir)
    ultra_name = aug_name

    model = YOLO(f"{model_name}.pt")  # automatically downloads pretrained weights

    train_args = {
        "data": data_yaml,
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device if torch.cuda.is_available() else "cpu",
        "project": ultra_project,
        "name": ultra_name,
        "exist_ok": True,
        "patience": patience,
        "verbose": True,
        **aug_dict,          # ← all your augmentations unpacked here
        **kwargs
    }

    # Train
    model.train(**train_args)

    # Ultralytics run folder (where it saved results.csv and plots/)
    ultra_run_dir = base_dir / aug_name

    # --------------------- Save raw metrics (epoch-wise + summary) ---------------------
    # 1. Copy the full epoch-wise results.csv (perfect for matplotlib line plots)
    results_csv_src = ultra_run_dir / "results.csv"
    if results_csv_src.exists():
        shutil.copy(results_csv_src, raw_dir / "results.csv")
        print(f"Saved epoch-wise metrics → {raw_dir / 'results.csv'}")

    # 2. Load best weights and run final validation to extract ALL important metrics
    best_pt = ultra_run_dir / "weights" / "best.pt"
    if best_pt.exists():
        best_model = YOLO(str(best_pt))
        val_results = best_model.val(
            data=data_yaml,
            imgsz=imgsz,
            device=device,
            plots=False,      # we already have plots from training
            save_json=False,
        )

        # Extract key metrics (works for both single-class and multi-class datasets)
        box = val_results.box

        # Overall (mean across classes if multi-class)
        precision = float(box.p.mean()) if hasattr(box.p, "mean") else float(box.p)
        recall = float(box.r.mean()) if hasattr(box.r, "mean") else float(box.r)
        f1 = float((2 * box.p * box.r / (box.p + box.r + 1e-8)).mean()) \
            if hasattr(box.p, "mean") else float(2 * box.p * box.r / (box.p + box.r + 1e-8))

        metrics_data = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "map50": float(box.map50),
            "map50-95": float(box.map),
        }

        # Add individual mAP@50, mAP@55, ..., mAP@95 (10 values)
        for i, mAP_value in enumerate(box.maps):
            iou_threshold = 50 + i * 5
            metrics_data[f"map{iou_threshold}"] = float(mAP_value)

        # Save clean summary CSV (super easy to load & plot)
        summary_df = pd.DataFrame(list(metrics_data.items()), columns=["metric", "value"])
        summary_df.to_csv(raw_dir / "metrics_summary.csv", index=False)

        print(f"Saved final metrics summary → {raw_dir / 'metrics_summary.csv'}")
        print(f"      → mAP50-95 = {box.map:.4f} | Precision = {precision:.4f} | Recall = {recall:.4f}")

    # --------------------- Save plots ---------------------
    ultra_plots_src = ultra_run_dir / "plots"
    if ultra_plots_src.exists():
        for plot_file in ultra_plots_src.glob("*.png"):
            shutil.copy(plot_file, plots_dir / plot_file.name)
        print(f"   Saved all plots → {plots_dir} (F1_curve, P_curve, R_curve, PR_curve, etc.)")

    print(f"   Training finished successfully!\n"
          f"   Model: {model_name}\n"
          f"   Augmentation: {aug_name}\n"
          f"   Raw data: {raw_dir}\n"
          f"   Plots: {plots_dir}\n")