"""
Helper script to verify the MVTec dataset is correctly placed.

MVTec AD must be downloaded manually (it requires accepting a license agreement):
    https://www.mvtec.com/company/research/datasets/mvtec-ad

After downloading and extracting, verify the structure with this script:
    python scripts/download_mvtec.py --data_dir data/raw

Expected structure:
    data/raw/
    ├── bottle/
    │   ├── train/good/
    │   └── test/{good,broken_large,broken_small,contamination}/
    ├── cable/
    └── ...
"""

import os
import argparse


MVTEC_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def verify_dataset(data_dir: str):
    if not os.path.exists(data_dir):
        print(f"ERROR: {data_dir} does not exist. Create it and download MVTec AD there.")
        return

    found = []
    missing = []

    for category in MVTEC_CATEGORIES:
        cat_path = os.path.join(data_dir, category)
        train_good = os.path.join(cat_path, "train", "good")
        test_dir = os.path.join(cat_path, "test")

        if os.path.exists(train_good) and os.path.exists(test_dir):
            n_train = len(os.listdir(train_good))
            n_test = sum(
                len(os.listdir(os.path.join(test_dir, d)))
                for d in os.listdir(test_dir)
                if os.path.isdir(os.path.join(test_dir, d))
            )
            found.append(f"  ✓ {category:<15} — {n_train} train, {n_test} test images")
        else:
            missing.append(f"  ✗ {category}")

    if found:
        print("Found categories:")
        for line in found:
            print(line)

    if missing:
        print("\nMissing categories:")
        for line in missing:
            print(line)

    print(f"\nTotal: {len(found)}/{len(MVTEC_CATEGORIES)} categories ready")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="data/raw")
    args = parser.parse_args()
    verify_dataset(args.data_dir)
