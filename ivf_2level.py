import os
import tempfile
import time
import numpy as np
import faiss
import h5py
import requests
# Constants
DATA_URL ="http://ann-benchmarks.com/glove-100-angular.hdf5"
CACHE = os.path.join(tempfile.gettempdir(),"glove-100-angular.hdf5")
k = 10
# Step 1: Download preprocessed GloVe dataset if not cached
if not os.path.exists(CACHE):
    print("Downloading GloVe-100 (~500 MB)…")
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
xb /= np.linalg.norm(xb, axis=1, keepdims=True)
xq /= np.linalg.norm(xq, axis=1, keepdims=True)
d = xb.shape[1]
# Step 4: Build first-level IVF index
nlist_lvl1 = 100
quantizer_lvl1 = faiss.IndexFlatIP(d)
index_lvl1 = faiss.IndexIVFFlat(quantizer_lvl1, d, nlist_lvl1, faiss.METRIC_INNER_PRODUCT)
index_lvl1.train(xb)
index_lvl1.add(xb)
# Step 5: Build second-level IVF index trained on same dataset
quantizer_lvl2 = faiss.IndexFlatIP(d)
index_lvl2 = faiss.IndexIVFFlat(quantizer_lvl2, d, 10, faiss.METRIC_INNER_PRODUCT)
index_lvl2.train(xb)
index_lvl2.add(xb)
# Step 6: Perform hierarchical IVF search
index_lvl2.nprobe = 3
start_time = time.time()
_, coarse_ids = index_lvl2.search(xq, 1)# find best coarse region
I = []
D = []
for i, _ in enumerate(coarse_ids):
    index_lvl1.nprobe = 10
    dists, ids = index_lvl1.search(xq[i:i+1], k)
    D.append(dists[0])
    I.append(ids[0])
D = np.array(D)
I = np.array(I)
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