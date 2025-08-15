from collections import defaultdict
import numpy as np

class MultiLevelIndex:
    def __init__(self, lowest_level_clusters):
        self.lowest_level_clusters = lowest_level_clusters
        self.levels = []
        self.build_index()

    def build_index(self):
        upper_level_clusters = max(10, self.lowest_level_clusters * 20)
        while upper_level_clusters >= 10:
            self.levels.append(upper_level_clusters)
            upper_level_clusters //= 20
        self.levels.append(10)  # Ensure at least one level with 10 clusters

    def add_clusters(self, level, clusters):
        if level < len(self.levels):
            self.levels[level] = clusters

    def get_centroids(self, level):
        if level < len(self.levels):
            return self.levels[level]
        return None

    def get_number_of_levels(self):
        return len(self.levels)

def search_strategy_nprobe(x, outer_ids, inner_kmeans, vector_metadata, xb, k, d, N_PROBE=1):
    best_heap = []
    tau = float("inf")
    inner_probed = 0
    seen_indices = set()
    outer_idx = 0
    total_outer = len(outer_ids)

    while inner_probed < N_PROBE and outer_idx < total_outer:
        outer_id = outer_ids[outer_idx]
        outer_idx += 1
        if outer_id not in vector_metadata:
            continue
        inner_ids = list(vector_metadata[outer_id].keys())
        if not inner_ids:
            continue
        inner_centroids_subset = inner_kmeans.centroids[inner_ids]
        index_l2 = faiss.IndexFlatL2(d)
        index_l2.add(inner_centroids_subset)
        _, inner_ranks_local = index_l2.search(x.reshape(1, -1), N_PROBE)
        selected_inner_ids = [inner_ids[j] for j in inner_ranks_local[0] if j < len(inner_ids)]
        
        for inner_id in selected_inner_ids:
            idxs_meta = vector_metadata[outer_id][inner_id]
            if not idxs_meta:
                continue
            centroid = inner_kmeans.centroids[inner_id]
            d_qc = np.linalg.norm(x - centroid)
            for dist_ic, cos_theta, idx2 in idxs_meta:
                if idx2 in seen_indices:
                    continue
                lower_bound = abs(d_qc - dist_ic)
                if lower_bound > tau:
                    continue
                est_dist = np.sqrt(max(0.0, d_qc ** 2 + dist_ic ** 2 - 2 * d_qc * dist_ic * cos_theta))
                if est_dist > tau:
                    continue
                actual_dist = np.linalg.norm(x - xb[idx2])
                best_heap.append((actual_dist, idx2))
                seen_indices.add(idx2)
                if len(best_heap) > k:
                    best_heap.sort()
                    best_heap = best_heap[:k]
                    tau = best_heap[-1][0]
            inner_probed += 1
            if inner_probed >= N_PROBE:
                break
    return best_heap

def search_strategy_tshirt(x, outer_ids, inner_kmeans, vector_metadata, xb, k, d, tshirt_size="small"):
    tshirt_settings = {
        "small": 0.10,
        "medium": 0.20,
        "large": 0.30
    }
    best_heap = []
    tau = float("inf")
    probed_inner_ids = set()
    seen_indices = set()
    pct = tshirt_settings[tshirt_size]
    n_outer_probe = max(1, int(np.ceil(len(outer_ids) * pct)))
    outer_ids = outer_ids[:n_outer_probe]

    for outer_id in outer_ids:
        if outer_id not in vector_metadata:
            continue
        inner_ids = list(vector_metadata[outer_id].keys())
        if not inner_ids:
            continue
        n_inner_probe = max(1, int(np.ceil(len(inner_ids) * pct)))
        inner_ids_to_probe = [iid for iid in inner_ids if iid not in probed_inner_ids][:n_inner_probe]
        if not inner_ids_to_probe:
            continue
        inner_centroids_subset = inner_kmeans.centroids[inner_ids_to_probe]
        index_l2 = faiss.IndexFlatL2(d)
        index_l2.add(inner_centroids_subset)
        _, inner_ranks_local = index_l2.search(x.reshape(1, -1), len(inner_ids_to_probe))
        selected_inner_ids = [inner_ids_to_probe[j] for j in inner_ranks_local[0] if j < len(inner_ids_to_probe)]
        
        for inner_id in selected_inner_ids:
            probed_inner_ids.add(inner_id)
            idxs_meta = vector_metadata[outer_id][inner_id]
            if not idxs_meta:
                continue
            centroid = inner_kmeans.centroids[inner_id]
            d_qc = np.linalg.norm(x - centroid)
            for dist_ic, cos_theta, idx2 in idxs_meta:
                if idx2 in seen_indices:
                    continue
                lower_bound = abs(d_qc - dist_ic)
                if lower_bound > tau:
                    continue
                est_dist = np.sqrt(max(0.0, d_qc ** 2 + dist_ic ** 2 - 2 * d_qc * dist_ic * cos_theta))
                if est_dist > tau:
                    continue
                actual_dist = np.linalg.norm(x - xb[idx2])
                best_heap.append((actual_dist, idx2))
                seen_indices.add(idx2)
                if len(best_heap) > k:
                    best_heap.sort()
                    best_heap = best_heap[:k]
                    tau = best_heap[-1][0]
    return best_heap

def search_query(x, outer_ids, inner_kmeans, vector_metadata, xb, k, d, N_PROBE=1, probe_strategy="nprobe", tshirt_size="small"):
    if probe_strategy == "nprobe":
        return search_strategy_nprobe(x, outer_ids, inner_kmeans, vector_metadata, xb, k, d, N_PROBE=N_PROBE)
    else:
        return search_strategy_tshirt(x, outer_ids, inner_kmeans, vector_metadata, xb, k, d, tshirt_size=tshirt_size)