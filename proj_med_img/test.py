from src.utils.load_augmentations import load_augmentations
from src.utils.training import train_yolo

# Quick test / usage example (run this file directly)
if __name__ == "__main__":
    try:
        
        
        aug_list = ["aug_baseline",  "aug_default",  "aug_light",  "medical_heavy",  "medical_light"]
        for aug_file_namme in aug_list:
            aug_from_list = load_augmentations(aug_file_namme)
            print(f"Loaded from {aug_file_namme}: {aug_from_list} \n")
        
      
        example_aug = load_augmentations("medical_heavy")
        
    except Exception as e:
        print("Error:", e)
        
        
        
    # example_aug = {
    #     "hsv_h": 0.015,
    #     "hsv_s": 0.7,
    #     "hsv_v": 0.4,
    #     "degrees": 15.0,
    #     "translate": 0.2,
    #     "scale": 0.6,
    #     "shear": 5.0,
    #     "flipud": 0.0,
    #     "fliplr": 0.5,
    #     "mosaic": 1.0,
    #     "mixup": 0.15,
    #     "copy_paste": 0.1,
    # }

    train_yolo(
        model_name="yolov9c",           # or "yolov9c", "yolo26m", etc.
        aug_dict=example_aug,
        data_yaml="datasets/SVS-1/data.yaml",   # ← change to your YAML
        aug_name=  "aug_baseline", #"aug_default", #"medical_heavy",       #
        epochs=300,                      # small number for quick test
        imgsz=1080,
        batch=16,
    )