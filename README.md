# Connexus
An Integrated Multi-scale Framework for Implementing Economically Feasible Pathways to Landscape Connectivity in Working Agricultural Systems
## Overview

Connexus is a comprehensive framework for optimizing agricultural landscapes by balancing economic profitability with ecological connectivity. The project integrates:

- **Economic Intensification (EI)**: Optimizes farm-level interventions (margin strips, habitat conversion) using Pyomo-based mathematical programming
- **Ecological Connectivity (EC)**: Maximizes landscape connectivity through graph-based optimization and repositioning algorithms
- **Policy Analysis**: Evaluates government interventions (subsidies, payments, mandates) using Bayesian optimization

## Features

- **Multi-objective Optimization**: Balances farm profitability with landscape connectivity
- **Policy Evaluation**: Bayesian optimization for finding optimal policy parameters
- **Spatial Analysis**: Advanced geospatial processing and visualization
- **Modular Architecture**: Separate modules for EI, EC, and policy analysis
- **Comprehensive Visualization**: Generates plots, maps, and analysis outputs

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)
- Git

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Connexus
```

### Step 2: Create a Virtual Environment

```bash
# Create virtual environment
python3 -m venv connexus_env

# Activate virtual environment
source connexus_env/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install required packages
pip install -r requirements.txt
```

### Step 4: Install Optional Solvers (Recommended)

For optimal performance, install additional optimization solvers:

**CBC (Mixed Integer Linear Programming):**
```bash
# On macOS with Homebrew
brew install cbc

# On Ubuntu/Debian
sudo apt-get install coinor-cbc

# On Windows, download from https://github.com/coin-or/Cbc/releases
```

**IPOPT (Nonlinear Programming):**
```bash
# On macOS with Homebrew
brew install ipopt

# On Ubuntu/Debian
sudo apt-get install coinor-ipopt

# On Windows, download from https://github.com/coin-or/Ipopt/releases
```

### Step 5: Verify Installation

```bash
# Test basic imports
python -c "import numpy, pandas, geopandas, pyomo, networkx, matplotlib; print('All core dependencies installed successfully!')"
```

## Project Structure

```
Connexus/
├── src/
│   ├── bo_policy.py              # Bayesian optimization for policy analysis
│   ├── config.py                 # Configuration and parameters
│   ├── ec_analysis.py            # Ecological connectivity analysis
│   ├── eco_intensification.py    # Economic intensification optimization
│   ├── ei_analysis.py            # Economic intensification analysis
│   ├── ei_policy.py              # Policy-integrated economic intensification
│   ├── graph_connectivity.py     # Graph-based connectivity optimization
│   ├── plot_utils.py             # Visualization utilities
│   └── utils/
│       ├── preprocess.py         # Data preprocessing utilities
│       ├── real_farms_config.py  # Real farm configuration
│       ├── synthetic.py          # Synthetic data generation
│       ├── tools.py              # General utility functions
│       └── utils.py              # Core utility functions
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Usage

### Basic Economic Intensification

```python
from src.config import Config
from src.eco_intensification import main_run_pyomo

# Load configuration
cfg = Config()

# Run optimization for a single farm
result_gdf = main_run_pyomo(
    cfg, 
    geojson_path="path/to/farm.geojson",
    image_path="output/visualization.png",
    output_json="output/results.geojson",
    neighbor_dist=1500,
    exit_tol=1e-6,
    penalty_coef=1e5
)
```

### Ecological Connectivity Analysis

```python
from src.graph_connectivity import optimize_landscape_connectivity

# Run connectivity optimization
chosen_pieces, optim_val, conn_val, conn_val_repos, plots, baseline_npv, optimized_npvs, chosen_repos = optimize_landscape_connectivity(
    geojson_path="path/to/landscape.geojson",
    boundary_seg_count=10,
    interior_cell_count=10,
    adjacency_dist=0.0,
    connectivity_metric='IIC',
    al_factor=1e-9,
    max_loss_ratio=0.1,
    neib_dist=1500,
    exit_tol=1e-6,
    reposition=True,
    params=cfg.params,
    conn_dir="output/connectivity",
    margin_weight=0.5,
    mode="optimized",
    plot=True
)
```

### Policy Analysis with Bayesian Optimization

```python
from src.bo_policy import evaluate_policy_for_bo_avg_config_obj

# Define policy parameters
policy_params = {
    'subsidy': {'adj_hab_factor_margin': 0.3, 'adj_hab_factor_habitat': 0.5},
    'payment': {'hab_per_ha': 75},
    'mandate': {'min_total_hab_area': 5.0}
}

# Run policy evaluation
objective_value = evaluate_policy_for_bo_avg_config_obj(
    policy_param_list=[0.3, 0.5, 75, 5.0],
    param_order=['adj_hab_factor_margin', 'adj_hab_factor_habitat', 'hab_per_ha', 'min_total_hab_area'],
    base_cfg=cfg,
    # ... other parameters
)
```

## Configuration

The main configuration is handled in `src/config.py`. Key parameters include:

- **Economic Parameters**: Crop prices, costs, discount rates
- **Ecological Parameters**: Connectivity metrics, adjacency distances
- **Optimization Parameters**: Solver tolerances, penalty coefficients
- **Policy Parameters**: Subsidy rates, payment levels, mandate thresholds

## Outputs

The framework generates various outputs:

- **Optimization Results**: GeoJSON files with intervention recommendations
- **Visualizations**: Maps showing optimal interventions and connectivity
- **Analysis Plots**: Performance comparisons and sensitivity analyses
- **Policy Evaluations**: Bayesian optimization results and parameter correlations
