# Ocean LSTM Profile Reconstruction

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](LICENSE)

PyTorch implementation of a stacked Long Short-Term Memory (LSTM) neural network with Monte Carlo Dropout (MCDO) for reconstructing complete Arctic ocean hydrographic profiles (temperature, salinity, and steric height) from combined satellite surface observations and sparse in-situ measurements.

Developed within the **FRESH-CARE** project (WP2) at the Institut de Ciències del Mar (ICM-CSIC), Barcelona.

## Features

- **Stacked LSTM with Monte Carlo Dropout**: Provides probabilistic uncertainty estimates on reconstructed profiles
- **Variable-Length Sequence Support**: Handles profiles with NaN tails (varying ocean floor depths) via dynamic padding and masking
- **EASE-grid Arctic Reconstruction**: Full pipeline to reconstruct 3D Arctic T/S/SH fields on an EASE2 grid
- **Early Stopping on MC-Dev Loss**: Patience-based stopping evaluated on MC dropout dev loss for robust convergence
- **Comprehensive Output**: Climatology, anomalies, full profiles, RMSE statistics, and geostrophic currents
- **GPU Acceleration**: Automatic CUDA detection and utilization when available

## Repository Structure

```
.
├── 1_JN_data_prep/          # Notebooks: prepare input NetCDF datasets (surface obs, GLORYS profiles, bathymetry, ice)
├── 2_lstm_train_test/       # Main training/testing script and MC convergence diagnostics
│   └── lstm_pytorch_pd_mcdo.py   # Main script
├── 3_JN_plot_test_results/  # Notebooks: visualize test set performance
├── 4_arctic_reconst_ease/   # Full Arctic reconstruction pipeline on EASE2 grid
├── 5_compute_reconst_stats/ # Scripts: compute reconstruction statistics and regional tables
├── 6_plot_reconst_results/  # Notebooks and scripts: plot reconstruction maps, transects, velocities
├── AA_winner_model_LSTM_52_46_bs16_lr2e-4_pat6x5_do0.2/  # Pre-trained winner model and test results
├── data_for_lstm/           # Input NetCDF datasets (train / dev / test splits)
├── lstm_pytorch_utils.py    # Shared utilities (model loader, MC dropout prediction)
└── compression_test.py      # NetCDF compression helper
```

## Requirements

```
torch
numpy
xarray
matplotlib
scipy
netCDF4
pyproj
rasterio
shapely
geopandas
gsw
cmocean
cmcrameri
affine
tqdm
pyyaml
```

Install with:

```bash
pip install torch numpy xarray matplotlib scipy netCDF4 pyproj rasterio shapely geopandas gsw cmocean cmcrameri affine tqdm pyyaml
```

## Installation

```bash
git clone https://github.com/Barcelona-Polar-Lab/fresh_care_wp2_ocean_lstm.git
cd fresh_care_wp2_ocean_lstm
pip install torch numpy xarray matplotlib scipy netCDF4 pyproj rasterio shapely geopandas gsw cmocean cmcrameri affine tqdm pyyaml
```

## Usage

### Training and Testing

```bash
cd 2_lstm_train_test/

# Train and test with default parameters
python lstm_pytorch_pd_mcdo.py --mode both

# Train only
python lstm_pytorch_pd_mcdo.py --mode train

# Test only (requires trained model)
python lstm_pytorch_pd_mcdo.py --mode test
```

### Custom Architecture

```bash
# Custom LSTM architecture (e.g., 3 layers with 50, 40, 30 units)
python lstm_pytorch_pd_mcdo.py --lstm_units 50 40 30 --batch_size 32 --max_epochs 200

# Adjust learning rate and dropout
python lstm_pytorch_pd_mcdo.py --learning_rate 0.0005 --dropout_rate 0.3

# Custom early stopping patience
python lstm_pytorch_pd_mcdo.py --patience 10
```

### Arctic Reconstruction

Edit the config YAML in `4_arctic_reconst_ease/configs/` then run:

```bash
cd 4_arctic_reconst_ease/
bash run_reconstruction.sh
```

## Configuration

Edit the `Config` class in `2_lstm_train_test/lstm_pytorch_pd_mcdo.py` to customize:

- **Input variables**: SST, SSS, ADT anomalies, spatial coordinates, seasonal cycle
- **Model architecture**: LSTM units, dropout rate
- **Training parameters**: Batch size, learning rate, max epochs
- **Early stopping**: Patience evaluations, MC-dev evaluation frequency, minimum delta
- **File paths**: Training, validation, and test data locations

## Input Data Format

Expected NetCDF files with dimensions `(profile, depth)` containing:

- `TEMP`, `PSAL`, `SH`: In-situ measurements
- `SST`, `SSS`, `ADT`: Satellite surface observations
- `T_glorys`, `S_glorys`, `SH_glorys`: GLORYS12 climatology
- `LATITUDE`, `LONGITUDE`, `X_EASE`, `Y_EASE`: Spatial coordinates
- `day_of_year`, `TIME`: Temporal information

Pre-processed datasets are provided in `data_for_lstm/`.

## Output

Training produces:
- `model_LSTM_X_Y/model.pth`: Trained model with configuration and normalization parameters
- `model_LSTM_X_Y/training_history.png`: Loss curves with early stopping marker

Testing produces:
- `model_LSTM_X_Y/mc_test_results.nc`: NetCDF with climatology, anomalies, full reconstructed profiles, MC uncertainty, and RMSE statistics

The pre-trained winner model (`LSTM [52, 46]`, bs=16, lr=2e-4) and its test results are included in `AA_winner_model_LSTM_52_46_bs16_lr2e-4_pat6x5_do0.2/`.

## Model Architecture

```
Input → Dropout → LSTM[0] → ... → LSTM[N] → Dropout → Linear → Output
```

- Variable-length sequences handled via dynamic padding and loss masking
- Monte Carlo Dropout active at inference time for uncertainty quantification
- Batch-first processing; MSE loss masked over valid (non-NaN) depth levels

## Citation

If you use this code or data, please cite:

> Pelletier, N. W., & Buongiorno Nardelli, B. (2026). *Ocean LSTM Profile Reconstruction* (WP2, FRESH-CARE). Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX

*(DOI will be updated upon Zenodo release.)*

## Authors

**Original implementation:**  
Bruno Buongiorno Nardelli  
Consiglio Nazionale delle Ricerche — Istituto di Scienze Marine, Napoli, Italia

**PyTorch translation, MCDO, refactoring, and Arctic reconstruction pipeline:**  
Nicolas Werner Pelletier  
Institut de Ciències del Mar (ICM-CSIC), Barcelona, España

## License

This project is licensed under the GNU General Public License v3.0 — see the [LICENSE](LICENSE) file for details.
