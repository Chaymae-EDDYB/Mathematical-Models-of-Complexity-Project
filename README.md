# Stochastic Analysis of Rumor Spreading Dynamics with Stifling Mechanisms

## Project Overview

This project implements and analyzes the **Daley-Kendall (DK) model** of rumor spreading with stifling mechanisms on a spatially explicit 2D grid population using **Agent-Based Modeling (ABM)**.

Our approach introduces psychological realism through the concept of **stifling** – where people stop spreading a rumor because they realize everyone around them already knows it.

### Key Research Question
> How does the "Stifling Probability" (the chance a spreader stops spreading when they meet someone who already knows) affect the final reach of a rumor?

## Project Structure

```
project/
├── rumor_model.py              # Core DK model implementation (Mesa framework)
├── analysis.py                 # Comprehensive analysis suite (stochastic, speed, sensitivity)
├── theoretical_validation.py   # Comparison with analytical DK ODE equations
├── animations.py               # Animated GIF visualizations
├── run_all.py                  # Main runner script (quick tests & full analysis)
├── run_full_1000.py            # 1,000 iteration runner
├── run_10k.py                  # 10,000 iteration runner (publication quality)
├── requirements.txt            # Python dependencies
│
├── results_1000iter/           # 1,000 iteration results
│
└── results_10k/                # 10,000 iteration results (FINAL)

# Legacy files (initial development)
├── rumour-spread.py            # Original model prototype
├── rumour-spread-analysis.py   # Combined analysis script
├── rumour-speed.py             # Speed analysis prototype
├── rumour-speed-log.py         # Log-scale plotting
├── stochastics.py              # Stochastic analysis prototype
├── sensitivity.py              # Sensitivity analysis prototype
└── spatial.py                  # Spatial visualization prototype
```

## Installation

```bash
# Using conda 
conda activate your_environment

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Full 1,000 Iteration Run
```bash
python run_full_1000.py
```

### Full 10,000 Iterations Run
```bash
sbatch submit_10k.sh
```

### Run Specific Analysis
```bash
python run_all.py --analysis stochastic
python run_all.py --analysis sensitivity
python run_all.py --analysis theory
```

### Visualization Files
| File | Description |
|------|-------------|
| `stochastic_reach_ci.png` | Bar chart with 95% confidence intervals |
| `stochastic_distribution.png` | Violin plots showing result distributions |
| `stochastic_penetration.png` | Population penetration rate curve |
| `sensitivity_heatmap.png` | Phase diagram (transmission vs stifle) |
| `speed_lifespan.png` | Rumor duration (linear & log scale) |
| `spatial_comparison.png` | Final grid states at different stifle levels |
| `spatial_times_heard.png` | Heatmap of "times heard" per cell |
| `temporal_evolution.png` | State evolution over time |
| `temporal_spreaders.png` | Spreader population curves |
| `rumor_spreading.gif` | Animation of single simulation |
| `stifle_comparison.gif` | Side-by-side animation comparison |


## Authors
Chaymae ED-Dyb,
Oumkalthoum M'HAMDI, 
Zineb MIFTAH

January 2026
