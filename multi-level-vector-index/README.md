## Integrated Multi-Level Index Search Prototype

### Installation

To install the required dependencies, run:

```shell
pip install -r requirements.txt
```

### Usage

1. Clone the repository:

```shell
git clone <repository-url>
cd multi-level-vector-index
```

2. Run the main application:

```shell
python src/main.py
```

3. Follow the prompts to specify the number of lowest-level clusters and initiate the clustering and search processes.

### Directory Structure

- `src/`: Contains the main application code.
  - `data/`: Data loading and preprocessing utilities.
  - `clustering/`: Clustering logic, including multi-level index management.
  - `search/`: Search functionalities and strategies.
  - `utils/`: Helper functions for various tasks.
  - `experiments/`: Evaluation scripts for performance metrics.
- `notebooks/`: Jupyter notebooks for experimentation and reference. [Currently not in use]
- `tests/`: Unit tests for ensuring code correctness.
- `requirements.txt`: Lists the project dependencies.
- `setup.py`: Setup script for package installation.

### Core Components

1. **KMeansBuilder** (`kmeans_builder.py`)
   - Builds hierarchical k-means clustering
   - Stores clustering models for each level
2. **HierarchyMetadata** (`hierarchy_metadata.py`)
   - Manages Index structure metadata and mappings
   - Implements top-down hierarchical search
3. **MultiLevelIndex** (`multi_level_index.py`)
   - Main interface for the hierarchical index
   - Provides  API for building and searching

## Basic Usage

```bash
python src/main.py
```

## Parameter Setup

### 1. Number of Lowest-Level Clusters (`n_lowest_level_clusters`)

```python
n_lowest_level_clusters = 2000  # Example value
```

It is set as program input and it determines structure of the multi-level vector index. This parameter controls the granularity of the finest clustering level (Level 0).

The system automatically builds a multi-level hierarchy using a **divide-by-20 rule** (for prototype use):

```
Example with 2000 lowest-level clusters:
Level 0 (finest):   2000 clusters  ← User specified
Level 1:            100 clusters   ← 2000 / 20 = 100
Level 2 (coarsest): 10 clusters    ← 100 / 20 = 5 → stops at 10 (minimum)
```

### 2. Other Parameters

You can customize the advanced parameters by modifying the `src/main.py` file:

```python
# Advanced parameter configuration
def main():
    # ... existing code ...
    
    # Advanced configuration options
    PROBE_STRATEGY = "nprobe"  # or "tshirt"
    N_PROBE = 2               # Number of clusters to probe per level
    MIN_CROSS = 1             # Minimum N_CROSS value to test
    MAX_CROSS = 6             # Maximum N_CROSS value to test  
    TSHIRT_SIZE = "medium"    # "small", "medium", or "large"
    
    run_cross_pollination_experiment(
        dataset_path=cache_path,
        n_inner_clusters=n_lowest_level_clusters,
        probe_strategy=PROBE_STRATEGY,
        N_PROBE=N_PROBE,
        min_cross=MIN_CROSS,
        max_cross=MAX_CROSS,
        tshirt_size=TSHIRT_SIZE,
        use_hierarchical=use_hierarchical
    )
```

#### A. Probe Strategy (`probe_strategy`)

```python
probe_strategy = "nprobe"  # or "tshirt"
```

**Options**:

- **`"nprobe"`**: Uses `N_PROBE` parameter to determine search breadth
- **`"tshirt"`**: Uses predefined size categories (small/medium/large)

#### B. N_PROBE (`N_PROBE`)

```python
N_PROBE = 2  # Number of clusters to probe per level
```

**What it does**: Controls how many clusters to examine at each hierarchy level
**Typical values**: 1-5

**Trade-offs**:

- **Higher N_PROBE**: Better recall, slower search
- **Lower N_PROBE**: Faster search, potentially lower recall

#### C. N_CROSS Range (`min_cross`, `max_cross`)

```python
min_cross = 1  # Start testing from N_CROSS=1
max_cross = 6  # Test up to N_CROSS=6
```

**What it does**: Sets the range of N_CROSS values to test
**N_CROSS meaning**: Number of nearest clusters each vector is assigned to

**Impact**:

- **N_CROSS=1**: Each vector assigned to 1 cluster (fastest, lowest recall)
- **N_CROSS=3**: Each vector assigned to 3 clusters (balanced)
- **N_CROSS=6**: Each vector assigned to 6 clusters (slowest, highest recall)

#### D. T-Shirt Sizing (`tshirt_size`)

```python
tshirt_size = "medium"  # "small", "medium", "large"
```

**Only used when `probe_strategy="tshirt"`**

**Mapping**:

- **`"small"`**: Conservative search (faster, lower recall)
- **`"medium"`**: Balanced search
- **`"large"`**: Aggressive search (slower, higher recall)

#### E. Use Different Dataset 

To use different datasets, modify the `DATA_URLS` dictionary:

```python
# main.py
# Load dataset
    selected_dataset = "fashion-mnist"
    DATA_URLS = {
        "fashion-mnist": "http://ann-benchmarks.com/fashion-mnist-784-euclidean.hdf5",
        "gist": "http://ann-benchmarks.com/gist-960-euclidean.hdf5",
        "sift": "http://ann-benchmarks.com/sift-128-euclidean.hdf5"
    }
```



## Example Configurations

```python
def fast_search_config():
    n_lowest_level_clusters = 2000
    
    run_cross_pollination_experiment(
        dataset_path=cache_path,
        n_inner_clusters=n_lowest_level_clusters,
        probe_strategy="nprobe",
        N_PROBE=1,           # Search fewer clusters
        min_cross=1,         # Test minimal cross-pollination
        max_cross=3,         # Don't go too high
        tshirt_size="small",
        use_hierarchical=True
    )
```



#### TBD: Integrate Product Quantization.
