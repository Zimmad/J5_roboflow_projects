from pathlib import Path
import shutil
import hashlib
import pandas as pd
from typing import Dict, Any, Optional
from ultralytics import YOLO
import torch
import mlflow
import optuna  # for type hint only


def _suggest_augmentations(trial: optuna.Trial) -> Dict[str, Any]:
    aug = {}

    #I use brightness only as the color (hue and saturation) does nothing useful in grey images.
    aug["hsv_v"] = trial.suggest_float("hsv_v", 0.0, 0.9, step=0.09)

    # Geometric Transformations ===
    aug["degrees"] = trial.suggest_float("degrees", 0.0, 45.0, step=5.0)
    aug["translate"] = trial.suggest_float("translate", 0.0, 0.3, step=0.03)
    aug["scale"] = trial.suggest_float("scale", 0.0, 0.525, step=0.075)
    aug["shear"] = trial.suggest_float("shear", 0.0, 15.0, step=2.0)
    aug["perspective"] = trial.suggest_float("perspective", 0.0, 0.001, step=0.0001)
    aug["flipud"] = trial.suggest_float("flipud", 0.0, 0.75, step=0.15)   # probability
    aug["fliplr"] = trial.suggest_float("fliplr", 0.0, 0.75, step=0.15)
    
    #=== Mosaic only transformation

    aug["mosaic"] = trial.suggest_categorical("mosaic", [0.0, 1.0])

    # mixup/copy_paste 
    aug["mixup"] = trial.suggest_float("mixup", 0.0, 0.2, step=0.05)
    aug["copy_paste"] = trial.suggest_float("copy_paste", 0.0, 0.2, step=0.05)

    return aug


def _generate_aug_name(aug_dict: Dict[str, Any]) -> str:
    """Generate readable + unique name from augmentation dict."""
    key_items = [f"{k}_{v}" for k, v in sorted(aug_dict.items())]
    aug_key = "_".join(key_items)
    short_hash = hashlib.md5(aug_key.encode()).hexdigest()[:6]
    return f"aug_{short_hash}"


def train_yolo(
    model_name: str,
    data_yaml: str,
    aug_dict: Optional[Dict[str, Any]] = None,
    trial: Optional[optuna.Trial] = None,
    aug_name: Optional[str] = None,
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "0",
    patience: int = 50,
    experiment_name: str = "yolo_augmentation_sweep",
    
    **kwargs: Any,
) -> Dict[str, float]:
    
    mlflow_uri = "http://127.0.0.1:5000"
    mlflow.set_tracking_uri(mlflow_uri)
    
    
    """If trial is provided use the optuna for auto matic hyper parameters. If aug_dict is provided then use the static mode. """
    # using optuna for automaitc hyperparameters ===
    if trial is not None:
        aug_dict = _suggest_augmentations(trial)
        # Prune bad trial early 
        # trial.report(intermediate_value, step)
        # if trial.should_prune(): 
        #     print(f"Trial {trial.number} pruned at epoch {epoch} (mAP50-95: {map50_95:.4f})")
        #     raise optuna.exceptions.TrialPruned()

    if aug_dict is None:
        aug_dict = {}  # static augmentations

    if aug_name is None:
        aug_name = _generate_aug_name(aug_dict)

    # ======= Tracking the experiments with MLflow
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=f"{model_name}_{aug_name}"):
        mlflow.log_params({
            "model": model_name,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            **aug_dict
        })

        #  Directory Setup
        project_root = Path(__file__).resolve().parents[2]
        base_dir = project_root / "runs" / model_name
        raw_dir = base_dir / "raw" / aug_name
        plots_dir = base_dir / "plots" / aug_name

        raw_dir.mkdir(parents=True, exist_ok=True)
        plots_dir.mkdir(parents=True, exist_ok=True)

        print(f"Training: {model_name} | Aug: {aug_name}")

        # ====================== Ultralytics Training ======================
        model = YOLO(f"{model_name}.pt")

        train_args = {
            "data": data_yaml,
            "epochs": epochs,
            "imgsz": imgsz,
            "batch": batch,
            "device": device if torch.cuda.is_available() else "cpu",
            "project": str(base_dir),
            "name": aug_name,
            "exist_ok": True,
            "patience": patience,
            "verbose": False,
            "save_json": False,
            "save_hybrid": False,
            "save": True,           # Set to False to minimize saving (recommended)
            "save_period": -1,      # -1 = save only final, don't save every N epochs
            **aug_dict,
            **kwargs
        }

        model.train(**train_args)


        # weights_dir = ultra_run_dir / "weights"
        # last_pt = weights_dir / "last.pt"

        # if last_pt.exists():
        #     last_pt.unlink()  # delete last.pt

        ultra_run_dir = base_dir / aug_name

        # This saves the Results & Logs to MLflow 
        # Copy epoch-wise results
        results_csv_src = ultra_run_dir / "results.csv"
        if results_csv_src.exists():
            shutil.copy(results_csv_src, raw_dir / "results.csv")
            mlflow.log_artifact(str(raw_dir / "results.csv"))

        # Final validation on best weights
        best_pt = ultra_run_dir / "weights" / "best.pt"
        metrics_summary = {}

        if best_pt.exists():
            best_model = YOLO(str(best_pt))
            val_results = best_model.val(data=data_yaml, imgsz=imgsz, 
                                       device=device, plots=False)

            box = val_results.box
            precision = float(box.p.mean()) if hasattr(box.p, "mean") else float(box.p)
            recall = float(box.r.mean()) if hasattr(box.r, "mean") else float(box.r)
            f1 = float((2 * precision * recall) / (precision + recall + 1e-8))

            metrics_summary = {
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "map50": float(box.map50),
                "map50-95": float(box.map),
            }

            # Log all mAP@xx
            for i, mAP_value in enumerate(box.maps):
                metrics_summary[f"map{50 + i*5}"] = float(mAP_value)

            # Save summary
            summary_df = pd.DataFrame(list(metrics_summary.items()), columns=["metric", "value"])
            summary_df.to_csv(raw_dir / "metrics_summary.csv", index=False)
            mlflow.log_artifact(str(raw_dir / "metrics_summary.csv"))

            # Loging the best metrics to MLflow
            mlflow.log_metrics(metrics_summary)

            print(f"Finished → mAP50-95: {box.map:.4f} | F1: {f1:.4f}")

        # Copy the plots
        ultra_plots_src = ultra_run_dir / "plots"
        if ultra_plots_src.exists():
            for p in ultra_plots_src.glob("*.png"):
                shutil.copy(p, plots_dir / p.name)
            mlflow.log_artifacts(str(plots_dir), artifact_path="plots")

        return metrics_summary