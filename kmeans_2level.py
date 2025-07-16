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

print("Step 1: Download preprocessed GloVe dataset if not cached")
if not os.path.exists(CACHE):
    print("Downloading GloVe-100 (~500 MB)…")
    headers = {"User-Agent":"Mozilla/5.0"}
    response = requests.get(DATA_URL, headers=headers)
    if response.status_code == 200:
        with open(CACHE,"wb") as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download dataset: HTTP{response.status_code}")

print("Step 2: Load data from HDF5 file")
with h5py.File(CACHE,"r") as f:
    xb = f["train"][:].astype(np.float32)
    xq = f["test"][:].astype(np.float32)
    gt = f["neighbors"][:]

print("Step 3: Normalize vectors for cosine similarity")
xb /= np.linalg.norm(xb, axis=1, keepdims=True)
xq /= np.linalg.norm(xq, axis=1, keepdims=True)
d = xb.shape[1]

print("Step 4: KMeans with 100 clusters (inner level)")
train_sample = xb[np.random.choice(len(xb), size=min(50000, len(xb)), replace=False)]
inner_kmeans = faiss.Kmeans(d, 100, niter=20, verbose=True, spherical=True)
inner_kmeans.train(train_sample)
_, xb_inner_assignments = inner_kmeans.index.search(xb, 1)
_, xq_inner_assignments = inner_kmeans.index.search(xq, 10)

print("Step 5: KMeans with 10 clusters (outer level) on inner centroids")
inner_centroids = inner_kmeans.centroids
outer_kmeans = faiss.Kmeans(d, 10, niter=20, verbose=True, spherical=True)
outer_kmeans.train(inner_centroids)
_, inner_to_outer = outer_kmeans.index.search(inner_centroids, 1)
_, xq_outer_assignments = outer_kmeans.index.search(xq, 3)

print("Step 6: Pre-group vectors into outer and inner clusters")
from collections import defaultdict
outer_to_inner_to_points = defaultdict(lambda: defaultdict(list))
for idx, inner_id in enumerate(xb_inner_assignments[:, 0]):
    outer_id = inner_to_outer[inner_id][0]
    outer_to_inner_to_points[outer_id][inner_id].append(idx)

print("Step 7: Search using 2-level KMeans with nprobe")
start_time = time.time()
I = []
D = []
for i, x in enumerate(xq):
    outer_ids = xq_outer_assignments[i]  # top-3 outer clusters
    best_ids = []
    best_dists = []

    for outer_id in outer_ids:
        if outer_id not in outer_to_inner_to_points:
            continue

        # Compute distances from query to inner centroids in this outer cluster
        inner_ids = list(outer_to_inner_to_points[outer_id].keys())
        inner_centroids_subset = inner_kmeans.centroids[inner_ids]
        index_ip = faiss.IndexFlatIP(d)
        index_ip.add(inner_centroids_subset)
        _, inner_ranks_local = index_ip.search(x.reshape(1, -1), 10)
        selected_inner_ids = [inner_ids[j] for j in inner_ranks_local[0] if j < len(inner_ids)]

        for inner_id in selected_inner_ids:
            idxs = outer_to_inner_to_points[outer_id][inner_id]
            if not idxs:
                continue
            candidates = xb[idxs]
            dists = np.dot(x.reshape(1, -1), candidates.T).flatten()
            topk = np.argsort(-dists)[:min(k, len(dists))]
            best_ids.append(np.array(idxs)[topk])
            best_dists.append(dists[topk])

    if best_ids:
        all_ids = np.concatenate(best_ids)
        all_dists = np.concatenate(best_dists)
        topk = np.argsort(-all_dists)[:k]
        I.append(all_ids[topk])
        D.append(all_dists[topk])
    else:
        dists = np.dot(x.reshape(1, -1), xb.T).flatten()
        idx = np.argsort(-dists)[:k]
        I.append(idx)
        D.append(dists[idx])

D = np.array(D)
I = np.array(I)
elapsed_time = time.time() - start_time
qps = len(xq) / elapsed_time

print("Step 8: Compute recall")
recall = (I == gt[:, :k]).sum() / (gt.shape[0] * k)

print("Step 9: Output results")
print("Nearest neighbor indices (first 10 queries):")
print(I[:10])
print("\nDistances to nearest neighbors (first 10 queries):")
print(D[:10])
print(f"\nRecall@{k}:{recall:.4f}")
print(f"Queries per second:{qps:.2f} QPS")