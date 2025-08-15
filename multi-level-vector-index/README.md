## Usage

- Multi-level clustering using KMeans.
- User-defined parameters for the number of lowest-level clusters.
- Automatic adjustment of upper-level clusters to ensure fewer than 10 clusters at the root level.
- Efficient search strategies adapted from existing implementations.
- Comprehensive evaluation metrics for performance assessment.

## Installation

To install the required dependencies, run:

```
pip install -r requirements.txt
```

## Usage

1. Clone the repository:

```
git clone <repository-url>
cd multi-level-vector-index
```

2. Run the main application:

```
python src/main.py
```

3. Follow the prompts to specify the number of lowest-level clusters and initiate the clustering and search processes.

## Directory Structure

- `src/`: Contains the main application code.
  - `data/`: Data loading and preprocessing utilities.
  - `clustering/`: Clustering logic, including multi-level index management.
  - `search/`: Search functionalities and strategies.
  - `utils/`: Helper functions for various tasks.
  - `experiments/`: Evaluation scripts for performance metrics.
- `notebooks/`: Jupyter notebooks for experimentation and reference.
- `tests/`: Unit tests for ensuring code correctness.
- `requirements.txt`: Lists the project dependencies.
- `setup.py`: Setup script for package installation.
