import argparse
from pathlib import Path
from typing import List, Optional


def verify_yolo_dataset(
    dataset_path: str | Path,
    splits: List[str] = ["train", "val", "test"],
    image_extensions: Optional[set] = None,
    verbose: bool = True,
) -> bool:
    """
    Verify that a dataset follows the official YOLO (Ultralytics) format.
    
    Expected structure for each split:
        dataset/
        ├── train/
        │   ├── images/   ← image files
        │   └── labels/   ← corresponding .txt files
        ├── val/
        └── test/
    
    Label format (each line in .txt):
        class_id x_center y_center width height
        - class_id: integer >= 0
        - x, y, w, h: floats in [0, 1] (normalized to image size)
    
    Every image MUST have a corresponding .txt label file (empty file = no objects).
    """
    dataset_path = Path(dataset_path).resolve()
    if not dataset_path.exists():
        print(f"Dataset root not found: {dataset_path}")
        return False

    if image_extensions is None:
        # Medical images often use these formats
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

    print(f"\n🔍 Verifying dataset: {dataset_path.name}")
    print(f"   Path: {dataset_path}")
    print("=" * 80)

    overall_ok = True

    for split in splits:
        images_dir = dataset_path / split / "images"
        labels_dir = dataset_path / split / "labels"

        print(f"\n📂 Checking split: {split}")

        # 1. Directory existence
        if not images_dir.exists():
            print(f"   Missing directory: {images_dir.relative_to(dataset_path)}")
            overall_ok = False
            continue
        if not labels_dir.exists():
            print(f"   Missing directory: {labels_dir.relative_to(dataset_path)}")
            overall_ok = False
            continue

        # 2. Collect files
        image_files = [
            f for f in images_dir.iterdir()
            if f.is_file() and f.suffix.lower() in image_extensions
        ]
        label_files = {f.stem: f for f in labels_dir.glob("*.txt") if f.is_file()}

        print(f"   Found {len(image_files)} images and {len(label_files)} label files")

        # 3. Check for orphan labels (labels without matching image)
        image_stems = {img.stem for img in image_files}
        orphan_labels = set(label_files.keys()) - image_stems
        if orphan_labels:
            print(f"   {len(orphan_labels)} orphan label files (no matching image):")
            for name in sorted(list(orphan_labels)[:5]):  # show first 5
                print(f"      • {name}.txt")
            if len(orphan_labels) > 5:
                print(f"      ... and {len(orphan_labels)-5} more")
            overall_ok = False

        # 4. Check every image has a label file + validate label content
        missing_labels = 0
        invalid_labels_count = 0

        for img in image_files:
            label_path = labels_dir / f"{img.stem}.txt"

            if not label_path.exists():
                if verbose:
                    print(f"   Missing label for image: {img.name}")
                missing_labels += 1
                overall_ok = False
                continue

            # Parse and validate label file
            with open(label_path, "r", encoding="utf-8") as f:
                lines = [line.strip() for line in f.readlines() if line.strip()]

            for line_no, line in enumerate(lines, 1):
                parts = line.split()
                if len(parts) != 5:
                    if verbose:
                        print(f"   Invalid format in {label_path.name} (line {line_no}): "
                              f"expected 5 values, got {len(parts)}")
                    invalid_labels_count += 1
                    overall_ok = False
                    continue

                try:
                    cls_id = int(parts[0])
                    x, y, w, h = map(float, parts[1:])

                    if cls_id < 0:
                        if verbose:
                            print(f"   Negative class_id in {label_path.name} (line {line_no})")
                        invalid_labels_count += 1
                        overall_ok = False

                    if not (0 <= x <= 1 and 0 <= y <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                        if verbose:
                            print(f"   Bounding box values not in [0,1] in {label_path.name} (line {line_no})")
                        invalid_labels_count += 1
                        overall_ok = False

                except ValueError:
                    if verbose:
                        print(f"   Non-numeric values in {label_path.name} (line {line_no})")
                    invalid_labels_count += 1
                    overall_ok = False

        if missing_labels > 0:
            print(f"   {missing_labels} images are missing label files")
        if invalid_labels_count > 0:
            print(f"   {invalid_labels_count} invalid label entries found")

    print("\n" + "=" * 80)
    if overall_ok:
        print(f"{dataset_path.name} PASSED all YOLO format checks!")
    else:
        print(f"{dataset_path.name} FAILED verification. Please fix the issues above.")
    
    return overall_ok


def main():
    parser = argparse.ArgumentParser(
        description="Verify YOLO dataset format for medical imaging project",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--datasets-root",
        type=str,
        default="datasets",
        help="Root folder containing all datasets (e.g. SVS-1, Syrinx-2)",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["SVS-1", "Syrinx-2"],
        help="List of dataset folders to verify",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "val", "test"],
        help="Dataset splits to check",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output (only show summary)",
    )

    args = parser.parse_args()

    root = Path(args.datasets_root).resolve()
    if not root.exists():
        print(f"Datasets root not found: {root}")
        return

    print("OLO Dataset Verification Tool")
    print(f"   Root: {root}")
    print(f"   Datasets to check: {', '.join(args.datasets)}")
    print(f"   Splits: {', '.join(args.splits)}")
    print("-" * 80)

    all_datasets_ok = True
    for ds_name in args.datasets:
        ds_path = root / ds_name
        if not ds_path.exists():
            print(f"Dataset folder not found: {ds_name} (skipping)")
            all_datasets_ok = False
            continue

        ok = verify_yolo_dataset(
            dataset_path=ds_path,
            splits=args.splits,
            verbose=not args.quiet,
        )
        if not ok:
            all_datasets_ok = False

    print("\n" + "=" * 80)
    if all_datasets_ok:
        print("ALL DATASETS ARE IN CORRECT YOLO FORMAT!")
        print("You can now safely run training with YOLOv9 / YOLOv11 / YOLOv26.")
    else:
        print("Some datasets have issues. Fix them before training.")


if __name__ == "__main__":
    main()