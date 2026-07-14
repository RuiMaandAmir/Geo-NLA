# GeoNLA Demo

> Experimental validation of Neural Network Layered Architecture as an Analogy for Geographic Systems

## Overview

This repository provides experimental evidence for the **GeoNLA theory** — the structural isomorphism between geographic layered systems and neural network layered architectures.

Key design choices mapped from the paper:

| Geographic Concept | Neural Network Analog |
|---|---|
| Geographic elements (elevation, climate, soil…) | Input layer (12D) |
| Natural environment emergence | Hidden layer (32D) |
| Human activity response | Output layer (6D) |
| Time-varying coupling W_k(t) | Multiplicative time modulation |
| Layered causal chain | Deep layered architecture |

We compare **GeoNLA** (layered MLP with multiplicative temporal gating) against **FlatMLP** (single hidden layer, no modulation) under identical parameter budgets, on both synthetic and real raster data.

## Key Results

| Metric | Dataset | GeoNLA | FlatMLP | Improvement |
|---|---|---|---|---|
| RMSE | Synthetic | 0.0412 | 0.0441 | **6.6%** |
| MAE | Synthetic | 0.0312 | 0.0360 | **13.3%** |
| RMSE | Raster (Linxia) | 0.0535 | 0.0550 | **2.7%** |
| MAE | Raster (Linxia) | 0.0405 | 0.0418 | **3.1%** |

Parameters are nearly identical (652 vs 678), so improvements come from **architecture design**, not capacity.

## Project Structure

```
geonla-demo/
├── README.md                 # This file
├── LICENSE                   # MIT license
├── .gitignore                # Git ignore rules
├── requirements.txt          # Python dependencies
├── config.py                 # Hyperparameters and data paths
├── make_demo_rasters.py      # Generate synthetic raster TIFFs for demo
├── data_utils.py             # Data loading (synthetic & raster)
├── models.py                 # GeoNLA and FlatMLP model definitions
├── train.py                  # Training loop with ablation logging
├── visualize.py              # Publication-quality figures
├── run_demo.py               # Main entry point: data → train → plot
├── data/
│   └── raw/                  # Place real .tif rasters here
└── outputs/                  # Saved models, figures, CSVs
```

## Installation

```bash
pip install -r requirements.txt
```

Dependencies: `torch>=2.0`, `numpy>=1.24`, `matplotlib>=3.7`, `seaborn>=0.13`, `scikit-learn>=1.3`, `pandas>=2.0`, `rasterio>=1.3`

## Quick Start

### Synthetic data experiment

```bash
python run_demo.py --source synthetic
```

### Real raster data experiment

```bash
# Generate demo rasters if you don't have real data
python make_demo_rasters.py

# Or use real Linxia Basin data (place .tif files in data/raw/)
python run_demo.py --source raster
```

### Run training only

```bash
python train.py
```

### Generate visualizations only

```bash
python visualize.py
```

## Data Sources

| Source | Dataset | Description |
|---|---|---|
| **Synthetic** | Simulated Linxia Basin | 12D geographic causal chain with multiplicative time coupling |
| **Real Raster** | Linxia Basin (35.3°N–35.8°N, 102.9°E–103.5°E) | |
| | DEM | SRTM 30m |
| | Climate | WorldClim v2.1 |
| | Soil | SoilGrids v2.0 |
| | Land Cover | ESA WorldCover |
| | Population | WorldPop |
| | Nighttime Light | VIIRS |

## Model Architecture

**GeoNLA** enforces the geographic layered causal structure:

```
x (12D) → [Linear → SiLU → Linear] → h (32D)
                                ↓
              Multiplicative modulation by t (6D)
                                ↓
                         y_hat (6D)
```

The multiplicative time gate corresponds to the paper's time-varying coupling coefficient W_k(t), which modulates inter-layer information flow — analogous to how temporal environmental changes (seasonal cycles, climate variability) modulate the strength of geographic causal relationships.

**FlatMLP** removes the layered constraint and time modulation, collapsing everything into a single hidden layer.

See `models.py` for implementation details.

## Citation

```bibtex
@misc{ma2026geonla,
  author = {Ma, Rui},
  title  = {Geo-NLA: Neural Network Layered Architecture as an Analogy for Geographic Systems},
  year   = {2026},
  publisher = {Zenodo},
  doi    = {10.5281/zenodo.21350728},
  url    = {https://doi.org/10.5281/zenodo.21350728}
}
```

## License

MIT. See [LICENSE](LICENSE).

## Full Paper

Detailed experimental report: [docs/experiment_report.md](docs/experiment_report.md)
