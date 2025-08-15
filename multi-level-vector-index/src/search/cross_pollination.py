# --- Multi-Level Vector Index Search ---
import time
import numpy as np
import bisect
from clustering.multi_level_index import MultiLevelIndex
from data.loader import load_dataset
from utils.helpers import pad_to_k
from experiments.evaluator import plot_evaluation_results

def evaluate_cross_pollination(
    xb, xq, gt, multi_level_index, k, d,
    N_PROBE=1, min_cross=1, max_cross=5, probe_strategy="nprobe", tshirt_size="small"
):
    """Evaluate cross-pollination performance across different N_CROSS values."""
    recalls = []
    qps_list = []
    cross_range = range(min_cross, max_cross + 1)
    
    for N_CROSS in cross_range:
        print(f"Evaluating N_CROSS = {N_CROSS}")
        
        # Build metadata for this N_CROSS value
        vector_metadata = multi_level_index.build_metadata(xb, N_CROSS)
        
        # Perform search across all queries
        I, D, recall, qps = search_all_queries_cross_pollination(
            xq, multi_level_index, xb, k, d, gt,
            N_PROBE=N_PROBE, probe_strategy=probe_strategy, tshirt_size=tshirt_size
        )
        
        recalls.append(recall)
        qps_list.append(qps)
        print(f"N_CROSS={N_CROSS}: recall={recall:.4f}, qps={qps:.2f}")
        
    return cross_range, recalls, qps_list

def search_all_queries_cross_pollination(
    xq, multi_level_index, xb, k, d, gt,
    N_PROBE=1, probe_strategy="nprobe", tshirt_size="small"
):
    """Search all queries using the multi-level index with cross-pollination."""
    I = []
    D = []
    start_time = time.time()
    
    for x in xq:
        best_heap = multi_level_index.search(
            x, k, N_PROBE=N_PROBE, probe_strategy=probe_strategy, tshirt_size=tshirt_size
        )
        
        if best_heap:
            best_heap.sort()
            idxs = [idx for _, idx in best_heap]
            dists = [dist for dist, _ in best_heap]
            I.append(pad_to_k(idxs, k, -1))
            D.append(pad_to_k(dists, k, float('inf')))
        else:
            # Fallback to brute force if no results
            dists = np.linalg.norm(xb - x.reshape(1, -1), axis=1)
            idx = np.argsort(dists)[:k]
            I.append(idx)
            D.append(dists[idx])
            
    D = np.array(D)
    I = np.array(I)
    elapsed_time = time.time() - start_time
    qps = len(xq) / elapsed_time
    
    # Calculate recall
    recall = np.mean([
        len(set(I[i]) & set(gt[i, :k])) / k
        for i in range(gt.shape[0])
    ])
    
    return I, D, recall, qps

def run_cross_pollination_experiment(
    dataset_path,
    n_inner_clusters=400,
    probe_strategy="nprobe",
    N_PROBE=2,
    min_cross=1,
    max_cross=6,
    tshirt_size="small"
):
    """Run the complete cross-pollination experiment."""
    # Load data
    xb, xq, gt = load_dataset(dataset_path)
    d = xb.shape[1]
    k = 10

    print(f"Dataset loaded: {xb.shape[0]} vectors, {d} dimensions")
    print(f"Query set: {xq.shape[0]} queries")

    # Build Multi-Level Index
    print(f"Building multi-level index with {n_inner_clusters} lowest-level clusters...")
    multi_level_index = MultiLevelIndex(n_inner_clusters)
    multi_level_index.build_index(xb)
    
    print("Index structure:")
    print(multi_level_index.get_level_info())

    # Evaluate cross-pollination
    print(f"\nEvaluating cross-pollination with probe_strategy='{probe_strategy}'")
    cross_range, recalls, qps_list = evaluate_cross_pollination(
        xb, xq, gt, multi_level_index, k, d,
        N_PROBE=N_PROBE, min_cross=min_cross, max_cross=max_cross, 
        probe_strategy=probe_strategy, tshirt_size=tshirt_size
    )

    # Plot results
    print("\nGenerating evaluation plots...")
    plot_evaluation_results(cross_range, recalls, qps_list)
    
    return cross_range, recalls, qps_list