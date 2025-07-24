import os
import tempfile
import time
import numpy as np
import faiss
import h5py
import requests
# Constants
DATA_URL ="http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5"
CACHE = os.path.join(tempfile.gettempdir(),"fashion-mnist-784-euclidean.hdf5")
k = 10
# Step 1: Download preprocessed GloVe dataset if not cached
if not os.path.exists(CACHE):
    print("Downloading fashion-mnist-784 (~500 MB)…")
    headers = {"User-Agent":"Mozilla/5.0"}
    response = requests.get(DATA_URL, headers=headers)
    if response.status_code == 200:
        with open(CACHE,"wb") as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download dataset: HTTP{response.status_code}")
# Step 2: Load data from HDF5 file
with h5py.File(CACHE,"r") as f:
    xb = f["train"][:].astype(np.float32)
    xq = f["test"][:].astype(np.float32)
    gt = f["neighbors"][:]
# Step 3: Normalize vectors for cosine similarity (by converting to inner-product)
# Remove normalization for Euclidean distance
# xb /= np.linalg.norm(xb, axis=1, keepdims=True)
# xq /= np.linalg.norm(xq, axis=1, keepdims=True)
d = xb.shape[1]
# Step 4: Build IVF index using Euclidean distance
nlist = 100
quantizer = faiss.IndexFlatL2(d)
index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
# Step 5: Train and add vectors to the index
index.train(xb)
index.add(xb)
# Step 6: Perform search and measure time
index.nprobe = 10
start_time = time.time()
D, I = index.search(xq, k)
elapsed_time = time.time() - start_time
qps = len(xq) / elapsed_time
# Step 7: Compute recall
recall = (I == gt[:, :k]).sum() / (gt.shape[0] * k)
# Step 8: Output results
print("Nearest neighbor indices (first 10 queries):")
print(I[:10])
print("\nDistances to nearest neighbors (first 10 queries):")
print(D[:10])
print(f"\nRecall@{k}:{recall:.4f}")
print(f"Queries per second:{qps:.2f} QPS")