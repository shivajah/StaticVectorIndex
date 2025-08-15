from collections import defaultdict
import numpy as np
import time

def evaluate_multi_level_index(
    xb, xq, gt, multi_level_index, k, N_PROBE=1, min_cross=1, max_cross=5
):
    recalls = []
    qps_list = []
    cross_range = range(min_cross, max_cross + 1)

    for N_CROSS in cross_range:
        print(f"Evaluating N_CROSS = {N_CROSS}")
        vector_metadata = multi_level_index.cross_pollinate_metadata(xb, N_CROSS)
        I, D, recall, qps = multi_level_index.search_all_queries(
            xq, vector_metadata, xb, k, N_PROBE=N_PROBE
        )
        recalls.append(recall)
        qps_list.append(qps)
        print(f"N_CROSS={N_CROSS}: recall={recall:.4f}, qps={qps:.2f}")

    return cross_range, recalls, qps_list

def plot_evaluation_results(cross_range, recalls, qps_list):
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots()
    color = 'tab:blue'
    ax1.set_xlabel('N_CROSS (number of clusters each vector is inserted into)')
    ax1.set_ylabel('Recall', color=color)
    ax1.plot(cross_range, recalls, marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('QPS', color=color)
    ax2.plot(cross_range, qps_list, marker='x', color=color)
    ax2.tick_params(axis='y', labelcolor=color)
    plt.title('Recall and QPS vs N_CROSS (Multi-Level Index Evaluation)')
    plt.show()