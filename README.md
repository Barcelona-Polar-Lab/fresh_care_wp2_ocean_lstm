# Ocean LSTM Profile Reconstruction

PyTorch implementation of a Long Short-Term Memory (LSTM) neural network for reconstructing complete ocean hydrographic profiles (temperature, salinity, and steric height) from combined satellite surface observations and sparse in-situ measurements.

## Features

- **Flexible LSTM Architecture**: Configurable stacked LSTM layers with customizable units per layer
- **Variable-Length Sequence Support**: Automatically handles profiles with NaN tails (e.g., different ocean floor depths)
- **Early Stopping**: Prevents overfitting with configurable patience and validation monitoring
- **Comprehensive Output**: Saves climatology, anomalies, full profiles, and detailed error statistics
- **GPU Acceleration**: Automatic CUDA detection and utilization when available

## Requirements

```bash
numpy
xarray
torch
matplotlib
```

## Installation

```bash
git clone https://github.com/YOUR-USERNAME/ocean-lstm-pytorch.git
cd ocean-lstm-pytorch
pip install -r requirements.txt
```

## Usage

### Training and Testing

```bash
# Train and test with default parameters
python lstm_pytorch_pd.py --mode both

# Train only
python lstm_pytorch_pd.py --mode train

# Test only (requires trained model)
python lstm_pytorch_pd.py --mode test
```

### Custom Architecture

```bash
# Custom LSTM architecture (e.g., 3 layers with 50, 40, 30 units)
python lstm_pytorch_pd.py --lstm_units 50 40 30 --batch_size 32 --max_epochs 200

# Adjust learning rate and dropout
python lstm_pytorch_pd.py --learning_rate 0.0005 --dropout_rate 0.3

# Custom early stopping patience
python lstm_pytorch_pd.py --patience 10
```

## Configuration

Edit the `Config` class in `lstm_pytorch_pd.py` to customize:

- **Input variables**: Enable/disable SST, SSS, ADT anomalies, spatial coordinates, seasonal cycle
- **Model architecture**: LSTM units, dropout rate, activation function
- **Training parameters**: Batch size, learning rate, max epochs
- **Early stopping**: Patience, minimum delta threshold
- **File paths**: Training, validation, and test data locations

## Input Data Format

Expected NetCDF files with dimensions `(profile, depth)` containing:

- `TEMP`, `PSAL`, `SH`: In-situ measurements
- `SST`, `SSS`, `ADT`: Satellite surface observations  
- `T_glorys`, `S_glorys`, `SH_glorys`: Climatology
- `LATITUDE`, `LONGITUDE`, `X_EASE`, `Y_EASE`: Spatial coordinates
- `day_of_year`, `TIME`: Temporal information

## Output

Training produces:
- `model_LSTM_X_Y/model.pth`: Trained model with configuration and normalization parameters
- `model_LSTM_X_Y/training_history.png`: Loss curves with early stopping marker

Testing produces:
- `model_LSTM_X_Y/test_results.nc`: Comprehensive NetCDF dataset with:
  - Climatology fields
  - Observed and predicted anomalies
  - Full reconstructed profiles
  - Error statistics (RMSE by profile and depth)

## Model Architecture

```
Input → Dropout → LSTM[0] → ... → LSTM[N] → Dropout → Linear → Output
```

- Supports variable-length sequences via dynamic padding and masking
- Batch-first processing for efficiency
- MSE loss with masked computation for variable lengths

## Author

Bruno Buongiorno Nardelli (adapted to PyTorch)  
Consiglio Nazionale delle Ricerche  
Istituto di Scienze Marine  
Napoli, Italia

## License

[Add your license here]
