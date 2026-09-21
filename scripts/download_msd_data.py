#!/usr/bin/env python3
"""
scripts/download_msd_data.py
============================
Automated downloader for the Medical Segmentation Decathlon (MSD)
Task01_BrainTumour dataset (484 multi-modal 3D MRI scans).
"""

import os
import argparse
from monai.apps import download_and_extract

# Official MONAI MSD Task01 BrainTumour URL and MD5 checksum
RESOURCE = "https://msd-for-monai.s3-us-west-2.amazonaws.com/Task01_BrainTumour.tar"
MD5 = "24147396bfab72023f0544dda53403b1"


def download_dataset(root_dir: str = "data"):
    os.makedirs(root_dir, exist_ok=True)
    tar_path = os.path.join(root_dir, "Task01_BrainTumour.tar")
    dataset_dir = os.path.join(root_dir, "Task01_BrainTumour")

    if os.path.exists(dataset_dir) and os.path.exists(os.path.join(dataset_dir, "imagesTr")):
        images_tr = os.path.join(dataset_dir, "imagesTr")
        count = len([f for f in os.listdir(images_tr) if f.endswith(".nii.gz")])
        print(f"Dataset already exists at: {dataset_dir} ({count} patient scans)")
        return dataset_dir

    print(f"Downloading MSD Task01_BrainTumour (484 patients, 7.09 GB)...")
    
    # Download with aria2c or curl (with resume support)
    cmd = (
        f"aria2c -x 8 -s 8 -k 1M -c -d '{root_dir}' -o 'Task01_BrainTumour.tar' '{RESOURCE}' "
        f"|| curl -C - -L -o '{tar_path}' '{RESOURCE}'"
    )
    res = os.system(cmd)
    if res != 0:
        raise RuntimeError("Download failed!")

    print(f"Extracting {tar_path} into {root_dir}...")
    os.system(f"tar -xf '{tar_path}' -C '{root_dir}'")
    print(f"Dataset successfully extracted to: {dataset_dir}")
    return dataset_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download MSD Task01 Brain Tumour Dataset")
    parser.add_argument("--root-dir", type=str, default="data", help="Target root directory")
    args = parser.parse_args()
    download_dataset(args.root_dir)
