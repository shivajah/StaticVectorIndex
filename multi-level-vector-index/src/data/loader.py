import os
import numpy as np
import requests
import h5py

def download_dataset(url, cache_path):
    """Generic function to download any dataset from URL."""
    if not os.path.exists(cache_path):
        print(f"Downloading dataset from {url}...")
        headers = {"User-Agent": "Mozilla/5.0"}
        response = requests.get(url, headers=headers)
        if response.status_code == 200:
            with open(cache_path, "wb") as f:
                f.write(response.content)
        else:
            raise Exception(f"Failed to download dataset: HTTP{response.status_code}")

def load_dataset(cache_path):
    """Generic function to load dataset from HDF5 file."""
    with h5py.File(cache_path, "r") as f:
        xb = f["train"][:].astype(np.float32)
        xq = f["test"][:].astype(np.float32)
        gt = f["neighbors"][:]
    return xb, xq, gt
