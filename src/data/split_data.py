"""
src/data/split_data.py
Split dataset into train/val/test sets using symbolic links (no extra disk space).

This script reads from either the full dataset (data/Drowsy/, data/Non Drowsy/)
or the subsampled dataset (data/subsampled/Drowsy/, data/subsampled/Non Drowsy/),
then randomly splits images into train/val/test folders inside data/processed/.
All splits are created as symlinks pointing to the original images.
"""

import os
import random
import shutil
import argparse
from pathlib import Path

def create_symlink(src, dst):
    """Create a symbolic link from src to dst, overwriting if exists."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)

def split_data(source_root, output_root, split_ratios=(0.7, 0.15, 0.15), random_seed=42):
    """
    Split images from source_root/Drowsy and source_root/Non Drowsy into train/val/test.
    
    Args:
        source_root (str): Path containing 'Drowsy' and 'Non Drowsy' subfolders.
        output_root (str): Destination where 'train/', 'val/', 'test/' will be created.
        split_ratios (tuple): (train_ratio, val_ratio, test_ratio). Must sum to 1.
        random_seed (int): Seed for reproducibility.
    """
    random.seed(random_seed)
    source_root = Path(source_root)
    output_root = Path(output_root)
    
    class_names = ["Drowsy", "Non Drowsy"]
    split_names = ["train", "val", "test"]
    
    # Validate split ratios
    assert abs(sum(split_ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    for class_name in class_names:
        src_class_dir = source_root / class_name
        if not src_class_dir.exists():
            print(f"Warning: {src_class_dir} does not exist. Skipping.")
            continue
        
        # Get all image files
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        all_images = [f for f in src_class_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        print(f"Found {len(all_images)} images in {class_name}")
        
        # Shuffle and split
        random.shuffle(all_images)
        total = len(all_images)
        train_end = int(split_ratios[0] * total)
        val_end = train_end + int(split_ratios[1] * total)
        
        train_images = all_images[:train_end]
        val_images = all_images[train_end:val_end]
        test_images = all_images[val_end:]
        
        splits = {
            "train": train_images,
            "val": val_images,
            "test": test_images
        }
        
        # Create symlinks for each split
        for split_name in split_names:
            dst_split_dir = output_root / split_name / class_name
            dst_split_dir.mkdir(parents=True, exist_ok=True)
            
            for img_path in splits[split_name]:
                dst_path = dst_split_dir / img_path.name
                create_symlink(img_path.resolve(), dst_path)
            
            print(f"  {split_name}: {len(splits[split_name])} symlinks created in {dst_split_dir}")
    
    # Save split info as a simple text file for reference
    info_file = output_root / "split_info.txt"
    with open(info_file, "w") as f:
        f.write(f"Source root: {source_root.resolve()}\n")
        f.write(f"Split ratios: train={split_ratios[0]}, val={split_ratios[1]}, test={split_ratios[2]}\n")
        f.write(f"Random seed: {random_seed}\n")
    print(f"Split info saved to {info_file}")

def main():
    parser = argparse.ArgumentParser(description="Create train/val/test splits using symlinks.")
    parser.add_argument("--source", type=str, choices=["full", "sampled"], default="sampled",
                        help="Which dataset to split: 'full' (original) or 'sampled' (subsampled). Default: sampled")
    parser.add_argument("--dest", type=str, choices=["sampled", "full"], default="sampled",
                        help="Destination folder name inside data/processed/. Typically same as source. Default: sampled")
    parser.add_argument("--train_ratio", type=float, default=0.7, help="Training set ratio (default: 0.7)")
    parser.add_argument("--val_ratio", type=float, default=0.15, help="Validation set ratio (default: 0.15)")
    parser.add_argument("--test_ratio", type=float, default=0.15, help="Test set ratio (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    # Determine source root based on --source argument
    if args.source == "full":
        source_root = "data"
    else:  # sampled
        source_root = "data/subsampled"
    
    # Destination root inside data/processed/
    output_root = f"data/processed/{args.dest}"
    
    split_ratios = (args.train_ratio, args.val_ratio, args.test_ratio)
    
    print(f"Splitting from {source_root} to {output_root}")
    print(f"Split ratios: train={args.train_ratio}, val={args.val_ratio}, test={args.test_ratio}")
    
    split_data(source_root, output_root, split_ratios, args.seed)
    print("Split creation complete.")

if __name__ == "__main__":
    main()