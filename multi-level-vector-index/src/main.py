import os
import tempfile
from data.loader import download_dataset
from search.cross_pollination import run_cross_pollination_experiment

def main():
    # User input for the number of lowest-level clusters
    n_lowest_level_clusters = int(input("Enter the number of lowest-level clusters: "))
    
    print("\nUsing hierarchical search...")

    # Dataset selection
    DATA_URLS = {
        "fashion-mnist": "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
        "gist": "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
        "sift": "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
    }
    
    # Display available datasets
    print("\nAvailable datasets:")
    for i, dataset_name in enumerate(DATA_URLS.keys(), 1):
        print(f"{i}. {dataset_name}")
    
    # Get user selection
    while True:
        try:
            choice = int(input("\nSelect dataset (1-3): "))
            if 1 <= choice <= len(DATA_URLS):
                selected_dataset = list(DATA_URLS.keys())[choice - 1]
                break
            else:
                print("Invalid choice. Please select 1, 2, or 3.")
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    print(f"Selected dataset: {selected_dataset}")
    url = DATA_URLS[selected_dataset]
    cache_path = os.path.join(tempfile.gettempdir(), url.split('/')[-1])
    download_dataset(url, cache_path)

    # Run hierarchical search experiment
    print(f"\n=== Running hierarchical search experiment for {selected_dataset} ===")
    run_cross_pollination_experiment(
        dataset_path=cache_path,
        n_inner_clusters=n_lowest_level_clusters,
        n_probe=1,
        min_cross=1,
        max_cross=6,
        probe_strategy="nprobe",
        tshirt_size="small"
    )

if __name__ == "__main__":
    main()