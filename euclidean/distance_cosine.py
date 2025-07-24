import os
import tempfile
import time
import numpy as np
import faiss
import h5py
import requests
from collections import defaultdict
import bisect

# This is the result for it: Recall@10:0.9309
# Queries per second:12.49 QPS
# Total vectors skipped due to early stopping: 81868380

# Constants
DATA_URL = "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5"
CACHE = os.path.join(tempfile.gettempdir(), "fashion-mnist-784-euclidean.hdf5")
k = 10

print("Step 1: Download Fashion-MNIST dataset if not cached")
if not os.path.exists(CACHE):
    print("Downloading Fashion-MNIST (~300 MB)…")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(DATA_URL, headers=headers)
    if response.status_code == 200:
        with open(CACHE, "wb") as f:
            f.write(response.content)
    else:
        raise Exception(f"Failed to download dataset: HTTP{response.status_code}")

print("Step 2: Load data from HDF5 file")
with h5py.File(CACHE, "r") as f:
    xb = f["train"][:].astype(np.float32)
    xq = f["test"][:].astype(np.float32)
    gt = f["neighbors"][:]

d = xb.shape[1]

print("Step 3: KMeans with 100 clusters (inner level, Euclidean)")
train_sample = xb[np.random.choice(len(xb), size=min(50000, len(xb)), replace=False)]
inner_kmeans = faiss.Kmeans(d, 100, niter=20, verbose=True, spherical=False)
inner_kmeans.train(train_sample)
_, xb_inner_assignments = inner_kmeans.index.search(xb, 1)
_, xq_inner_assignments = inner_kmeans.index.search(xq, 10)

print("Step 4: KMeans with 10 clusters (outer level, Euclidean)")
inner_centroids = inner_kmeans.centroids
outer_kmeans = faiss.Kmeans(d, 10, niter=20, verbose=True, spherical=False)
outer_kmeans.train(inner_centroids)
_, inner_to_outer = outer_kmeans.index.search(inner_centroids, 1)
_, xq_outer_assignments = outer_kmeans.index.search(xq, 3)

print("Step 5: Pre-group vectors and store dist + cos to centroid (sorted)")
outer_to_inner_to_points = defaultdict(lambda: defaultdict(list))
vector_metadata = defaultdict(dict)
for idx, inner_id in enumerate(xb_inner_assignments[:, 0]):
    outer_id = inner_to_outer[inner_id][0]
    centroid = inner_centroids[inner_id]
    vec = xb[idx]
    dist = np.linalg.norm(vec - centroid)
    cos_sim = np.dot(vec, centroid) / (np.linalg.norm(vec) * np.linalg.norm(centroid) + 1e-8)
    vector_metadata[outer_id].setdefault(inner_id, []).append((dist, cos_sim, idx))

# Sort each list by Euclidean distance to centroid
for outer_id in vector_metadata:
    for inner_id in vector_metadata[outer_id]:
        vector_metadata[outer_id][inner_id].sort()
        outer_to_inner_to_points[outer_id][inner_id] = [t[2] for t in vector_metadata[outer_id][inner_id]]

print("Step 6: Search using 2-level KMeans with lower and upper bounds")
start_time = time.time()
I = []
D = []
skip_counter = 0
for i, x in enumerate(xq):
    outer_ids = xq_outer_assignments[i]
    best_heap = []  # (distance, idx)
    tau = float("inf")

    for outer_id in outer_ids:
        if outer_id not in outer_to_inner_to_points:
            continue

        inner_ids = list(outer_to_inner_to_points[outer_id].keys())
        inner_centroids_subset = inner_kmeans.centroids[inner_ids]
        index_l2 = faiss.IndexFlatL2(d)
        index_l2.add(inner_centroids_subset)
        _, inner_ranks_local = index_l2.search(x.reshape(1, -1), 10)
        selected_inner_ids = [inner_ids[j] for j in inner_ranks_local[0] if j < len(inner_ids)]

        for inner_id in selected_inner_ids:
            idxs_meta = vector_metadata[outer_id][inner_id]
            if not idxs_meta:
                continue
            centroid = inner_centroids[inner_id]
            d_qc = np.linalg.norm(x - centroid)
            distances = [meta[0] for meta in idxs_meta]
            start = bisect.bisect_left(distances, d_qc)

            left = start - 1
            right = start
            while left >= 0 or right < len(idxs_meta):
                for direction in (left, right):
                    if direction < 0 or direction >= len(idxs_meta):
                        continue
                    dist_ic, cos_theta, idx = idxs_meta[direction]
                    lower_bound = abs(d_qc - dist_ic)
                    upper_bound = d_qc + dist_ic
                    if lower_bound > tau:
                        skip_counter += 1
                        continue
                    est_dist = np.sqrt(max(0.0, d_qc ** 2 + dist_ic ** 2 - 2 * d_qc * dist_ic * cos_theta))
                    if est_dist > tau:
                        skip_counter += 1
                        continue
                    actual_dist = np.linalg.norm(x - xb[idx])
                    best_heap.append((actual_dist, idx))
                    if len(best_heap) > k:
                        best_heap.sort()
                        best_heap = best_heap[:k]
                        tau = best_heap[-1][0]
                left -= 1
                right += 1

    if best_heap:
        best_heap.sort()
        I.append([idx for _, idx in best_heap])
        D.append([dist for dist, _ in best_heap])
    else:
        dists = np.linalg.norm(xb - x.reshape(1, -1), axis=1)
        idx = np.argsort(dists)[:k]
        I.append(idx)
        D.append(dists[idx])

D = np.array(D)
I = np.array(I)
elapsed_time = time.time() - start_time
qps = len(xq) / elapsed_time

print("Step 7: Compute recall")
recall = (I == gt[:, :k]).sum() / (gt.shape[0] * k)

print("Step 8: Output results")
print("Nearest neighbor indices (first 10 queries):")
print(I[:10])
print("\nDistances to nearest neighbors (first 10 queries):")
print(D[:10])
print(f"\nRecall@{k}:{recall:.4f}")
print(f"Queries per second:{qps:.2f} QPS")
print(f"Total vectors skipped due to early stopping: {skip_counter}")