"""
sweep_runner.py
Minimal Optuna-based augmentation hyperparameter sweep for YOLO models.
"""

import optuna
from optuna.samplers import TPESampler
from pathlib import Path
import logging
from ultralytics.utils import LOGGER
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.trainers.optuna_train_yolo import train_yolo


def setup_logging_for_run(log_file: Path):
    """Add file handler for training logs."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    LOGGER.addHandler(file_handler)
    return file_handler


def objective(trial, model_name: str, data_yaml: str, train_args: dict):
    # Suggest augmentations
    aug_dict = {
    #I use brightness only as the color (hue and saturation) does nothing useful in grey images. 
    "hsv_v": trial.suggest_float("hsv_v", 0.0, 0.9, step=0.09),

    #  geomatric transformation
    "degrees": trial.suggest_float("degrees", 0.0, 45.0, step=5.0),
    "translate": trial.suggest_float("translate", 0.0, 0.3, step=0.03),
    "scale": trial.suggest_float("scale", 0.0, 0.525, step=0.075),
    "shear": trial.suggest_float("shear", 0.0, 15.0, step=2.0),
    "perspective": trial.suggest_float("perspective", 0.0, 0.001, step=0.0001),

    # Flips 
    "flipud": trial.suggest_float("fliplr", 0.0, 0.75, step=0.15),
    "fliplr": trial.suggest_float("fliplr", 0.0, 0.75, step=0.15),

    # Mosaic
    "mosaic": trial.suggest_categorical("mosaic", [0.0, 1.0]),

    # Mixup/Copy-Paste 
    "mixup": trial.suggest_float("mixup", 0.0, 0.2, step=0.05),
    "copy_paste": trial.suggest_float("copy_paste", 0.0, 0.2, step=0.05),
}


    aug_name = f"trial_{trial.number:03d}"

    # Logging for this trial
    base_dir = Path("runs") / model_name
    log_dir = base_dir / "logs" / aug_name
    training_log = log_dir / "training.log"
    
    handler = setup_logging_for_run(training_log)

    try:
        metrics = train_yolo(
            model_name=model_name,
            data_yaml=data_yaml,
            aug_dict=aug_dict,      
            trial=trial,            # usingOptuna
            aug_name=aug_name,
            **train_args
        )
        
        map50 = metrics.get("map50", 0.0)
        precision = metrics.get("precision", 0.0)
        
        # Weighting the score for best results. This weighting can be the best middle ground between map and precision
        score = 0.6 * map50 + 0.4 * precision
        return score
        
        
                
    except Exception as e:
        print(f"Trial {trial.number} failed: {e}")
        raise optuna.exceptions.TrialPruned()
    finally:
        try:
            LOGGER.removeHandler(handler)
            handler.close()
        except:
            pass


def run_augmentation_sweep():
    data_yaml = "datasets/SVS-1/data.yaml" # Dataset configuration .yaml file
    
    model_names = ["yolov9e", "yolo11x", "yolo26x"]
    
    train_args = {
        "epochs": 2,
        "imgsz": 640,
        "batch": 4,
        "patience": 50,
        "device": "0",
        "lr0": 0.001,
        "lrf": 0.01,
        "seed": 42,
    }

    n_trials = 50   # Number of trials

    for model_name in model_names:
        print(f"\n{'='*90}")
        print(f"Starting Optuna Sweep for Model: {model_name}")
        print(f"Dataset: {data_yaml}")
        print(f"Trials: {n_trials}")
        print(f"{'='*90}\n")

        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(seed=42, multivariate=True),
            study_name=f"{model_name}_aug_optuna"
        )

        study.optimize(
            lambda trial: objective(trial, model_name, data_yaml, train_args),
            n_trials=n_trials,
            show_progress_bar=True
        )
        
        # Filter for completed trials only
        completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        
        if completed_trials:

        # Final Summary
            print(f"  \nBest Trial for {model_name}:")
            print(f"  Value (mAP50-95): {study.best_value:.5f}")
            print(f"  Best Params: {study.best_params}")
            
            # Save best parameters
            best_params_path = Path(f"runs/{model_name}/best_optuna_params.txt")
            best_params_path.parent.mkdir(parents=True, exist_ok=True)
            best_params_path.write_text(str(study.best_params))
            
        else:
            print(f"  No completed trials.  All {len(study.trials)} trials were failed.")

    print("\n All of the Optuna sweeps completed! ===")


if __name__ == "__main__":
    run_augmentation_sweep()