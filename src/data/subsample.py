"""
src/data/subsample.py
Randomly subsample images from the original dataset using symbolic links.

This script reads all images from data/Drowsy/ and data/Non Drowsy/,
randomly selects a fixed number per class, and creates symlinks in
data/subsampled/Drowsy/ and data/subsampled/Non Drowsy/.

No images are copied, so disk space usage is minimal.
"""

import os
import random
import shutil
import argparse
from pathlib import Path

def create_symlink(src, dst):
    """
    Create a symbolic link from src to dst.
    Ensures the parent directory of dst exists.
    """
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    # If a symlink already exists, remove it first
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    os.symlink(src, dst)

def subsample_dataset(raw_root, output_root, samples_per_class=5000, random_seed=42):
    """
    Randomly sample images from each class and create symlinks.

    Args:
        raw_root (str): Path to folder containing 'Drowsy' and 'Non Drowsy' subfolders.
        output_root (str): Path where 'Drowsy' and 'Non Drowsy' symlink folders will be created.
        samples_per_class (int): Number of images to select from each class.
        random_seed (int): Seed for reproducible random selection.
    """
    random.seed(random_seed)
    
    raw_root = Path(raw_root)
    output_root = Path(output_root)
    
    class_names = ["Drowsy", "Non Drowsy"]
    
    for class_name in class_names:
        src_class_dir = raw_root / class_name
        if not src_class_dir.exists():
            print(f"Warning: {src_class_dir} does not exist. Skipping.")
            continue
        
        # Get all image files (common extensions)
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}
        all_images = [f for f in src_class_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        print(f"Found {len(all_images)} images in {class_name}")
        
        # Sample without replacement
        if len(all_images) < samples_per_class:
            print(f"Warning: {class_name} has only {len(all_images)} images, which is less than {samples_per_class}. Using all images.")
            selected_images = all_images
        else:
            selected_images = random.sample(all_images, samples_per_class)
        
        # Create symlinks in output_root/class_name/
        dst_class_dir = output_root / class_name
        dst_class_dir.mkdir(parents=True, exist_ok=True)
        
        for img_path in selected_images:
            dst_path = dst_class_dir / img_path.name
            create_symlink(img_path.resolve(), dst_path)
        
        print(f"Created {len(selected_images)} symlinks in {dst_class_dir}")

def main():
    parser = argparse.ArgumentParser(description="Subsample dataset using symlinks (no extra disk space).")
    parser.add_argument("--samples_per_class", type=int, default=5000,
                        help="Number of images to sample from each class (default: 5000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--raw_root", type=str, default="data",
                        help="Root folder containing Drowsy/ and Non Drowsy/ (default: data)")
    parser.add_argument("--output_root", type=str, default="data/subsampled",
                        help="Destination folder for symlinks (default: data/subsampled)")
    
    args = parser.parse_args()
    
    # Convert to absolute paths relative to project root
    # The script is usually run from project root, so we use relative paths directly.
    raw_root = args.raw_root
    output_root = args.output_root
    
    print(f"Subsampling from {raw_root} to {output_root}")
    print(f"Samples per class: {args.samples_per_class}")
    print(f"Random seed: {args.seed}")
    
    subsample_dataset(raw_root, output_root, args.samples_per_class, args.seed)
    print("Done. Subsampled symlinks created.")

if __name__ == "__main__":
    main()