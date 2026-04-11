"""
Helper to load YOLO augmentation configurations from YAML files.
Supports configs/augmentations/*.yaml
"""

from pathlib import Path
import yaml
from typing import Dict, Any, Optional


def load_augmentations(aug_name: str, augmentations_dir: str = "/home/std-25-353/DEV/robofl/proj_med_img/configs/augmentations") -> Dict[str,Any]:
                    #    augmentations_dir: str =  "../../configs/augmentations/") -> Dict[str, Any]:
    """
    Load augmentation parameters from a YAML file and return them as a dict
    ready to be passed to Ultralytics YOLO .train().
    
    Args:
        aug_name (str): Name of the augmentation config without .yaml extension.
                        Examples: "baseline", "heavy", "no_mosaic", "medical_aug1"
        augmentations_dir (str): Path to the folder containing the yaml files.
    
    Returns:
        Dict of augmentation parameters (only the ones defined in the yaml).
    
    Example YAML content (configs/augmentations/medical_heavy.yaml):
        hsv_h: 0.015
        hsv_s: 0.7
        hsv_v: 0.4
        degrees: 15.0
        translate: 0.2
        scale: 0.6
        shear: 5.0
        flipud: 0.0
        fliplr: 0.5
        mosaic: 1.0
        mixup: 0.15
        copy_paste: 0.1
        erasing: 0.4
    """
    aug_path = Path(augmentations_dir) / f"{aug_name}.yaml"
    
    if not aug_path.exists():
        raise FileNotFoundError(f"Augmentation config not found: {aug_path}")
    
    with open(aug_path, "r") as f:
        aug_dict = yaml.safe_load(f)
    
    if not isinstance(aug_dict, dict):
        raise ValueError(f"Invalid augmentation YAML format in {aug_path}")
    
    # Optional: Filter only valid Ultralytics augmentation keys (safety)
    valid_keys = {
        "hsv_h", "hsv_s", "hsv_v", "degrees", "translate", "scale", "shear",
        "perspective", "flipud", "fliplr", "mosaic", "mixup", "copy_paste",
        "copy_paste_mode", "erasing", "auto_augment", "cutmix", "bgr"
        # Add more if needed in future (e.g. "close_mosaic", etc.)
    }
    
    filtered_aug = {k: v for k, v in aug_dict.items() if k in valid_keys}
    
    print(f"Loaded augmentation config: {aug_name} "
          f"({len(filtered_aug)} parameters)")
    
    return filtered_aug


def get_augmentation_config(aug_name: str, 
                            base_config: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Load augmentations and optionally merge with a base/default config.
    Useful if you want fallback values.
    """
    aug_dict = load_augmentations(aug_name)
    
    if base_config:
        # Merge: yaml values override base
        base = base_config.copy()
        base.update(aug_dict)
        return base
    
    return aug_dict


