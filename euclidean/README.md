## Requirements

- Python 3.7+
- [FAISS](https://github.com/facebookresearch/faiss)
- NumPy
- h5py
- requests
- matplotlib

Install dependencies (if needed):

```bash
pip install faiss-cpu numpy h5py requests matplotlib
```

## Usage

### 1. Select Dataset

At the top of the notebook, choose one of the supported datasets:

```python
selected_dataset = "fashion-mnist"  # or "gist", "sift"
```

### 2. Download Dataset

The notebook will automatically download the dataset if it is not present in your system's temp folder.

### 3. Run Experiment

The main experiment is run with:

```python
run_cross_pollination_experiment(
    dataset_path=cache_path,
    n_inner_clusters=100,      # Number of inner clusters for KMeans
    probe_strategy="nprobe",   # Search strategy ("nprobe" recommended)
    N_PROBE=5,                 # Number of inner clusters to probe per query
    min_cross=4,               # Minimum cross-pollination (clusters per vector)
    max_cross=5,               # Maximum cross-pollination
    tshirt_size="small"        # T-shirt size for cluster probing (not used for nprobe)
)
```

You can adjust these parameters to control clustering granularity and search accuracy/speed.

### 4. Experiment Output

- **Recall**: Fraction of true neighbors found.
- **QPS**: Queries per second (speed).
- **Plots**: Recall and QPS vs. cross-pollination parameter.

### 5. How It Works

- **Clustering**: Data is clustered hierarchically (outer and inner clusters).
- **Cross-Pollination**: Each vector is assigned to multiple clusters to improve recall.
- **Search**: For each query, the closest outer clusters are selected, then inner clusters are probed. The search scans vectors in these clusters and maintains a heap of the k closest results.

### 6. Customization

- Change `n_inner_clusters`, `N_PROBE`, `min_cross`, `max_cross` for different trade-offs.
- You can use your own dataset by adapting the data loading functions.

## Troubleshooting

- If FAISS is not installed, use `pip install faiss-cpu`.
- Large datasets may require significant RAM.
- For best performance, run on a machine with sufficient memory and CPU.

## References

- [FAISS Documentation](https://faiss.ai/)
- [ANN Benchmarks](http://ann-benchmarks.com/)
