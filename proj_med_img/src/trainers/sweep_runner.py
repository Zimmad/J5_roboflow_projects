"""
sweep_runner.py
Runs a full augmentation sweep for Ultralytics YOLO on your SVS dataset.
Trains on 01_aug_baseline through 09_aug_baseline.
Organizes outputs for easy comparison and paper writing.
"""

from pathlib import Path
import logging
from ultralytics.utils import LOGGER
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from src.utils.load_augmentations import load_augmentations
from src.trainers.train_yolo import train_yolo   

def setup_logging_for_run(log_file: Path):
    """Add file handler to capture full training console output."""
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    file_handler = logging.FileHandler(log_file, mode='w', encoding='utf-8')
    formatter = logging.Formatter('%(asctime)s | %(message)s')
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    LOGGER.addHandler(file_handler)
    return file_handler  # return so we can remove it later if needed

def run_augmentation_sweep():
    # ================== Configuration ==================
    model_name = "yolov9c"          # "yolo11m", "yolov8s", "yolov26m", etc.
    data_yaml = "datasets/SVS-1/data.yaml"   # dataset YAML
    
    model_names = [   "yolov9e",  "yolo11x" ,"yolo26x"  ] 
    aug_files = [
        "01_aug_baseline", "02_aug_baseline", "03_aug_baseline",
        "04_aug_baseline", "05_aug_baseline", "06_aug_baseline",
        "07_aug_baseline", "08_aug_baseline", "09_aug_baseline"
    ]
    
    train_args = {
        "epochs": 300,           
        "imgsz": 640,            
        "batch": 4,             
        "patience": 50,
        "device": "0",         
        "lr0": 0.001,
        "lrf": 0.01,
        "seed": 42,              
    }
    
    
    for m_name in model_names: 
        model_name = m_name
        
        print(f"Starting augmentation sweep for model: {model_name}")
        print(f"Dataset: {data_yaml}")
        print(f"Total runs: {len(aug_files)}\n")
        
        for aug_file in aug_files:
            try:
                # Load augmentation dict from your YAML files
                aug_dict = load_augmentations(aug_file)
                print(f"\n{'='*80}")
                print(f"Running: {aug_file}")
                print(f"Augmentations: {aug_dict}")
                print(f"{'='*80}")
                
                # Setup extra logging for this run (for paper)
                base_dir = Path("runs") / model_name
                log_dir = base_dir / "logs" / aug_file
                training_log = log_dir / "training.log"
                
                # Add file logger
                handler = setup_logging_for_run(training_log)
                
                # Train the model
                train_yolo(
                    model_name=model_name,
                    aug_dict=aug_dict,
                    data_yaml=data_yaml,
                    aug_name=aug_file,         
                    **train_args
                )
                
                # Cleanup handler
                LOGGER.removeHandler(handler)
                handler.close()
                
                print(f" Completed: {aug_file}")
                print(f"    Raw metrics + log: runs/{model_name}/raw/{aug_file}/")
                print(f"    Plots: runs/{model_name}/plots/{aug_file}/")
                print(f"    Full training log: runs/{model_name}/logs/{aug_file}/training.log\n")
                
            except Exception as e:
                print(f"Failed on {aug_file}: {e}")
                # Try to remove handler if it was added
                try:
                    LOGGER.removeHandler(handler)
                    handler.close()
                except:
                    pass
                continue
    
    print("\n Augmentation sweep completed successfully!")
    print(f"All results organized under: runs/{model_name}/")
    print("You can now compare metrics_summary.csv files across the 9 runs for your paper.")

if __name__ == "__main__":
    run_augmentation_sweep()