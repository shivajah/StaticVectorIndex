import os
import tempfile
from data.loader import download_dataset
from search.cross_pollination import run_cross_pollination_experiment

def main():
    # User input for the number of lowest-level clusters
    n_lowest_level_clusters = int(input("Enter the number of lowest-level clusters: "))
    
    print("\nUsing hierarchical search...")

    # Load dataset
    selected_dataset = "fashion-mnist"
    DATA_URLS = {
        "fashion-mnist": "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
        "gist": "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
        "sift": "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
    }
    url = DATA_URLS[selected_dataset]
    cache_path = os.path.join(tempfile.gettempdir(), url.split('/')[-1])
    download_dataset(url, cache_path)

    # Run hierarchical search experiment
    print(f"\n=== Running hierarchical search experiment for {selected_dataset} ===")
    run_cross_pollination_experiment(
        dataset_path=cache_path,
        n_inner_clusters=n_lowest_level_clusters,
        probe_strategy="nprobe",
        N_PROBE=2,
        min_cross=1,
        max_cross=6,
        tshirt_size="small",
        use_hierarchical=True
    )

if __name__ == "__main__":
    main()