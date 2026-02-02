#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyTorch LSTM Network for Ocean Hydrographic Profile Reconstruction

DESCRIPTION:
This script implements a stacked Long-Short Term Memory (LSTM) neural network 
to reconstruct complete ocean hydrographic profiles (temperature, salinity, 
and steric height) from combined satellite surface observations and sparse 
in-situ measurements using PyTorch.

Combines training and testing in a single script with clean organization.

AUTHOR: 
Bruno Buongiorno Nardelli (adapted to PyTorch)
Consiglio Nazionale delle Ricerche
Istituto di Scienze Marine
Napoli, Italia
"""

import os
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import argparse
import time
warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration class for easy parameter adjustment"""

    # Max depth - hardcoded for simplicity (common values: 250, 500, 1000)
    MAX_DEPTH = 50  # Change this as needed for your dataset

    # File paths
    TRAIN_FILE = 'fresh_data/50m_data_for_LSTM_filled_with_SH_train63.nc'
    DEV_FILE = 'fresh_data/50m_data_for_LSTM_filled_with_SH_dev21.nc'
    TEST_FILE = 'fresh_data/50m_data_for_LSTM_filled_with_SH_test16.nc'
    MODEL_DIR = None  # Will be set dynamically based on max depth and LSTM units
        
    # Model architecture
    LSTM_UNITS = [35, 35]  # Can be changed to [25, 25] or any list like:
                           # [50], [30, 30], [20, 25, 35], etc.
    DROPOUT_RATE = 0.2
    ACTIVATION = 'tanh'
    
    # Training parameters
    BATCH_SIZE = 16
    MAX_EPOCHS = 500
    LEARNING_RATE = 0.001
    
    # Early stopping parameters
    PATIENCE = 5
    MIN_DELTA = 1e-6
    
    # Input variables configuration (easy to modify)
    INPUT_VARS = {
        'sst_anomaly': True,   # Sea surface temperature anomaly
        'sss_anomaly': True,   # Sea surface salinity anomaly  
        'adt_anomaly': True,   # Absolute dynamic topography anomaly
        'latitude': False,      # Profile latitude
        'longitude': False,     # Profile longitude
        'x_ease': True,      # EASE grid x-coordinate
        'y_ease': True,      # EASE grid y-coordinate
        'seasonal_cos': True,  # Cosine of seasonal cycle
        'seasonal_sin': True   # Sine of seasonal cycle
    }
    
    # Output variables
    OUTPUT_VARS = ['steric_height', 'temperature', 'salinity']
    
    @staticmethod
    def get_model_dir(lstm_units, batch_size, dropout_rate, learning_rate):
        """Generate model directory name based on max depth, LSTM units, and hyperparameters"""
        if isinstance(lstm_units, list):
            units_str = '_'.join(map(str, lstm_units))
        else:
            units_str = str(lstm_units)
        
        # Format hyperparameters for directory name
        batch_str = f"batch{batch_size}"
        drop_str = f"drop{dropout_rate}"
        lr_str = f"lr{learning_rate}"
        
        model_name = f"model_{Config.MAX_DEPTH}m_LSTM_{units_str}_{batch_str}_{drop_str}_{lr_str}"
        return f"trained_models/{model_name}"

# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class OceanLSTM(nn.Module):
    """
    LSTM model for ocean profile reconstruction
    Flexible architecture that can be easily modified
    """
    
    def __init__(self, input_size, output_size, lstm_units, dropout_rate=0.2):
        super(OceanLSTM, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.lstm_units = lstm_units if isinstance(lstm_units, list) else [lstm_units]
        
        # Input dropout
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()
        layer_input_size = input_size
        
        for i, units in enumerate(self.lstm_units):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_size,
                    hidden_size=units,
                    batch_first=True,
                    dropout=dropout_rate if i < len(self.lstm_units) - 1 else 0
                )
            )
            layer_input_size = units
        
        # Output layer (applied to each time step)
        self.output_layer = nn.Linear(self.lstm_units[-1], output_size)
        self.output_dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        # Input dropout
        x = self.input_dropout(x)
        
        # LSTM layers
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
            
        # Output dropout and projection
        x = self.output_dropout(x)
        x = self.output_layer(x)
        
        return x

# ============================================================================
# MAIN EXECUTION FUNCTIONS (GENERAL)
# ============================================================================

def main():
    """Main function with argument parsing"""
    
    parser = argparse.ArgumentParser(description='PyTorch LSTM for Ocean Profile Reconstruction')
    parser.add_argument('--mode', choices=['train', 'test', 'both'], default='both',
                       help='Run mode: train, test, or both (default: both)')
    parser.add_argument('--lstm_units', nargs='+', type=int, default=None,
                       help='LSTM units per layer (e.g., --lstm_units 35 35)')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--max_epochs', type=int, default=None,
                       help='Maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate for optimizer')
    parser.add_argument('--dropout_rate', type=float, default=None,
                       help='Dropout rate for regularization')
    parser.add_argument('--patience', type=int, default=None,
                       help='Early stopping patience (number of epochs)')
     
    args = parser.parse_args()    # Override config if command line arguments provided
    if args.lstm_units:
        Config.LSTM_UNITS = args.lstm_units
    if args.batch_size:
        Config.BATCH_SIZE = args.batch_size
    if args.max_epochs:
        Config.MAX_EPOCHS = args.max_epochs
    if args.learning_rate:
        Config.LEARNING_RATE = args.learning_rate
    if args.dropout_rate:
        Config.DROPOUT_RATE = args.dropout_rate
    if args.patience:
        Config.PATIENCE = args.patience
    
    print(f"Configuration: LSTM={Config.LSTM_UNITS}, Batch={Config.BATCH_SIZE}, Max Epochs={Config.MAX_EPOCHS}")
    print(f"LR={Config.LEARNING_RATE}, Dropout={Config.DROPOUT_RATE}, Patience={Config.PATIENCE}")
    
    # Set model directory with hyperparameters
    Config.MODEL_DIR = Config.get_model_dir(Config.LSTM_UNITS, Config.BATCH_SIZE, Config.DROPOUT_RATE, Config.LEARNING_RATE)
    print(f"Model directory: {Config.MODEL_DIR}")
    
    # Run based on mode
    if args.mode in ['train', 'both']:
        run_training()
        
    if args.mode in ['test', 'both']:
        run_testing()

def run_training():
    """Run model training"""
    
    print("=== TRAINING MODE ===")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Create model directory (including parent directories)
    Path(Config.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    ds_train, ds_dev, ds_test = load_datasets()
    
    # Note: Model directory already set in main() before calling this function
    # Prepare data
    train_data = prepare_dataset(ds_train, 'train')
    dev_data = prepare_dataset(ds_dev, 'dev')
    test_data = prepare_dataset(ds_test, 'test')  # Keep for final evaluation only
    
    # Normalize data (exclude test to prevent data leakage)
    train_data, dev_data, norm_params = normalize_data(train_data, dev_data)
    
    # Apply same normalization to test data
    test_data['X_norm'] = (test_data['X'] - norm_params['X_min']) / norm_params['X_range']
    test_data['y_norm'] = (test_data['y'] - norm_params['y_min']) / norm_params['y_range']
    
    # Get data dimensions
    n_input_vars = train_data['X'].shape[2]
    n_output_vars = train_data['y'].shape[2]
    
    print(f"Training profiles: {train_data['X'].shape[0]}")
    print(f"Input variables: {n_input_vars} - {train_data['input_names']}")
    print(f"Output variables: {n_output_vars}")
    
    # Create model
    model = OceanLSTM(
        input_size=n_input_vars,
        output_size=n_output_vars,
        lstm_units=Config.LSTM_UNITS,
        dropout_rate=Config.DROPOUT_RATE
    ).to(device)
    
    print(f"Model architecture: {Config.LSTM_UNITS} LSTM units")
    print(f"Model device: {next(model.parameters()).device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(train_data, dev_data, batch_size=Config.BATCH_SIZE)
    
    # Train model
    model, train_losses, val_losses, stopped_epoch = train_model(model, train_loader, val_loader, device)
    
    # Plot training history
    plot_training_history(train_losses, val_losses, stopped_epoch)
    
    # Save final model and metadata
    config_dict = {
        'TRAIN_FILE': Config.TRAIN_FILE,
        'DEV_FILE': Config.DEV_FILE,
        'TEST_FILE': Config.TEST_FILE,
        'MODEL_DIR': Config.MODEL_DIR,
        'LSTM_UNITS': Config.LSTM_UNITS,
        'DROPOUT_RATE': Config.DROPOUT_RATE,
        'BATCH_SIZE': Config.BATCH_SIZE,
        'MAX_EPOCHS': Config.MAX_EPOCHS,
        'LEARNING_RATE': Config.LEARNING_RATE,
        'PATIENCE': Config.PATIENCE,
        'MIN_DELTA': Config.MIN_DELTA,
        'INPUT_VARS': Config.INPUT_VARS,
        'OUTPUT_VARS': Config.OUTPUT_VARS,
        'MAX_DEPTH': Config.MAX_DEPTH
    }
    
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config_dict,
        'norm_params': norm_params,
        'input_names': train_data['input_names'],
        'model_architecture': {
            'input_size': n_input_vars,
            'output_size': n_output_vars,
            'lstm_units': Config.LSTM_UNITS
        }
    }, Path(Config.MODEL_DIR) / 'model.pth')
    
    print(f"\nTraining completed!")
    if len(val_losses) > 0:
        print(f"Best validation loss: {min(val_losses):.6f}")
        print(f"Final training loss: {train_losses[-1]:.6f}")
    else:
        print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Model saved to: {Config.MODEL_DIR}")

def run_testing():
    """Run model testing"""
    
    print("\n=== TESTING MODE ===")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if trained model exists
    model_path = Path(Config.MODEL_DIR) / 'model.pth'
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}. Please run training first.")
    
    # Load trained model
    print("Loading trained model...")
    checkpoint = torch.load(model_path, map_location=device)
    
    model_config = checkpoint['model_architecture']
    norm_params = checkpoint['norm_params']
    input_names = checkpoint['input_names']
    
    # Recreate model
    model = OceanLSTM(
        input_size=model_config['input_size'],
        output_size=model_config['output_size'],  
        lstm_units=model_config['lstm_units'],
        dropout_rate=Config.DROPOUT_RATE
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Model loaded: {model_config['lstm_units']} LSTM units")
    
    # Load test data
    ds_train, ds_dev, ds_test = load_datasets()
    test_data = prepare_dataset(ds_test, 'test')
    
    # Apply same normalization as training
    test_data['X_norm'] = (test_data['X'] - norm_params['X_min']) / norm_params['X_range']
    test_data['y_norm'] = (test_data['y'] - norm_params['y_min']) / norm_params['y_range']
    
    print(f"Test data: {test_data['X'].shape[0]} profiles, {test_data['X'].shape[1]} depths")
    
    # Make predictions
    y_pred = make_predictions(model, test_data, norm_params, device)
    
    # Compute error statistics
    error_stats = compute_error_statistics(y_pred, test_data['y'])
    
    # Print overall RMSE
    print(f"\nOverall RMSE:")
    print(f"Steric Height: {error_stats['rmse_total'][0]:.3f} cm")
    print(f"Temperature: {error_stats['rmse_total'][1]:.3f} °C")
    print(f"Salinity: {error_stats['rmse_total'][2]:.3f} PSU")
    
    # Create comprehensive results dataset
    ds_results = create_results_dataset(test_data, y_pred, error_stats)
    
    # Save results
    results_file = Path(Config.MODEL_DIR) / 'test_results.nc'
    print(f"\nSaving results to {results_file}...")
    ds_results.to_netcdf(results_file)
    
    print(f"\nTesting completed successfully!")
    print(f"Results saved with {ds_results.dims['profile']} profiles and {ds_results.dims['depth']} depth levels")
    print(f"Dataset contains: climatology, anomalies, full profiles, and error statistics")



# ============================================================================
# TRAINING FUNCTIONS  
# ============================================================================

def train_model(model, train_loader, val_loader, device):
    """Train the LSTM model with early stopping using validation set"""
    
    print(f"Training model on {device}...")
    if torch.cuda.is_available():
        print(f"Initial GPU memory: {torch.cuda.memory_allocated()/1e6:.1f} MB")
    
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    
    # History tracking
    train_losses = []
    val_losses = []
    epoch_times = []
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    stopped_epoch = Config.MAX_EPOCHS
    
    print(f"Early stopping: patience={Config.PATIENCE}, min_delta={Config.MIN_DELTA}")
    
    # Main training loop
    for epoch in range(Config.MAX_EPOCHS):
        epoch_start_time = time.time()
        
        print(f"\nEpoch {epoch+1}/{Config.MAX_EPOCHS}")
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Early stopping check
        if val_loss < best_val_loss - Config.MIN_DELTA:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  → New best validation loss: {val_loss:.6f}")
        else:
            patience_counter += 1
            
        # Calculate epoch timing
        epoch_time = time.time() - epoch_start_time
        epoch_times.append(epoch_time)
        
        # Epoch summary
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = Config.MAX_EPOCHS - (epoch + 1)
        estimated_remaining = avg_epoch_time * remaining_epochs
        
        gpu_mem = f" | GPU: {torch.cuda.memory_allocated()/1e6:.0f}MB" if torch.cuda.is_available() else ""
        print(f'Epoch {epoch+1} completed - Train Loss: {train_loss:.6f} | '
              f'Val Loss: {val_loss:.6f} | Patience: {patience_counter}/{Config.PATIENCE} | '
              f'Time: {epoch_time:.1f}s | Max ETA: {estimated_remaining/60:.1f}min{gpu_mem}')
        
        # Check for early stopping
        if patience_counter >= Config.PATIENCE:
            print(f"\nEarly stopping triggered! No improvement for {Config.PATIENCE} epochs.")
            stopped_epoch = epoch + 1
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Loaded best model with validation loss: {best_val_loss:.6f}")
    
    return model, train_losses, val_losses, stopped_epoch

def create_data_loaders(train_data, dev_data, batch_size=16):
    """Create PyTorch data loaders for training and validation"""
    
    # Training data
    X_train = torch.FloatTensor(train_data['X_norm'])
    y_train = torch.FloatTensor(train_data['y_norm'])
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Validation data  
    X_val = torch.FloatTensor(dev_data['X_norm'])
    y_val = torch.FloatTensor(dev_data['y_norm'])
    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

def plot_training_history(train_losses, val_losses, stopped_epoch):
    """Plot training and validation losses with early stopping marker"""
    
    plt.figure(figsize=(12, 6))
    epochs = range(1, len(train_losses) + 1)
    
    plt.plot(epochs, train_losses, label='Training Loss', color='blue')
    plt.plot(epochs, val_losses, label='Validation Loss', color='red')
    
    # Mark early stopping point
    if stopped_epoch < len(train_losses) and len(train_losses) > 0:
        plt.axvline(x=stopped_epoch, color='green', linestyle='--', 
                   label=f'Early Stopping (Epoch {stopped_epoch})')
    
    plt.title(f'LSTM {"--".join(map(str, Config.LSTM_UNITS))} Training History')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    plt.grid(True)
    
    # Save plot
    plt.savefig(Path(Config.MODEL_DIR) / 'training_history.png', dpi=300, bbox_inches='tight')
    plt.close()  # Close figure to free memory

# ============================================================================
# TESTING FUNCTIONS
# ============================================================================

def make_predictions(model, test_data, norm_params, device):
    """Make predictions on test data and denormalize results"""
    
    print("Making predictions on test data...")
    
    # Normalize test inputs using training normalization parameters
    X_test_norm = test_data['X_norm']
    
    # Convert to tensor
    X_test_tensor = torch.FloatTensor(X_test_norm).to(device)
    
    # Make predictions in batches to handle large datasets
    model.eval()
    predictions = []
    batch_size = Config.BATCH_SIZE * 4  # Larger batch for inference
    
    print("Making predictions...")
    total_batches = (X_test_tensor.shape[0] + batch_size - 1) // batch_size
    batch_count = 0
    
    with torch.no_grad():
        for i in range(0, X_test_tensor.shape[0], batch_size):
            batch_X = X_test_tensor[i:i+batch_size]
            batch_pred = model(batch_X).cpu().numpy()
            predictions.append(batch_pred)
            
            batch_count += 1
            if batch_count % 50 == 0 or batch_count == total_batches:
                progress = (batch_count / total_batches) * 100
                print(f"  Prediction batch {batch_count}/{total_batches} ({progress:.1f}%)")
    
    y_pred_norm = np.concatenate(predictions, axis=0)
    
    # Denormalize predictions
    y_pred = y_pred_norm * norm_params['y_range'] + norm_params['y_min']
    
    return y_pred

def compute_error_statistics(y_pred, y_true):
    """Compute comprehensive error statistics"""
    
    print("Computing error statistics...")
    
    # Error fields (predicted - observed)
    # This has shape (profiles, depths, num of outp vars)
    errors = y_pred - y_true
    
    # RMSE by profile (average over depth for each profile)
    rmse_profiles = np.sqrt(np.mean(errors**2, axis=1))
    
    # RMSE by depth (average over profiles for each depth) 
    rmse_depths = np.sqrt(np.mean(errors**2, axis=0))
    
    # Overall RMSE per variable
    rmse_total = np.sqrt(np.mean(errors**2, axis=(0,1)))

    # Sum of all total RMSEs
    rmse_sum = np.sum(rmse_total)
    
    return {
        'errors': errors,
        'rmse_profiles': rmse_profiles,
        'rmse_depths': rmse_depths,
        'rmse_total': rmse_total,  #this has shape (num of outp vars,)
        'rmse_sum': rmse_sum
    }

def create_results_dataset(test_data, y_pred, error_stats):
    """Create comprehensive results dataset with all profiles and statistics"""
    
    print("Creating results dataset...")
    
    # Extract predictions and observations (anomalies)
    SH_pred_anom, T_pred_anom, S_pred_anom = y_pred[:,:,0], y_pred[:,:,1], y_pred[:,:,2]
    SH_obs_anom, T_obs_anom, S_obs_anom = test_data['y'][:,:,0], test_data['y'][:,:,1], test_data['y'][:,:,2]
    
    # Get climatology
    T_glorys = test_data['climatology']['T_glorys']
    S_glorys = test_data['climatology']['S_glorys'] 
    SH_glorys = test_data['climatology']['SH_glorys']
    
    # Compute full profiles (anomalies + climatology)
    SH_pred = SH_pred_anom + SH_glorys
    T_pred = T_pred_anom + T_glorys
    S_pred = S_pred_anom + S_glorys
    
    # Observed full profiles
    T_obs = test_data['full_profiles']['T']
    S_obs = test_data['full_profiles']['S']  
    SH_obs = test_data['full_profiles']['SH']
    
    # Extract error statistics
    SH_errors, T_errors, S_errors = error_stats['errors'][:,:,0], error_stats['errors'][:,:,1], error_stats['errors'][:,:,2]
    SH_rmse_prof, T_rmse_prof, S_rmse_prof = error_stats['rmse_profiles'][:,0], error_stats['rmse_profiles'][:,1], error_stats['rmse_profiles'][:,2]
    SH_rmse_depth, T_rmse_depth, S_rmse_depth = error_stats['rmse_depths'][:,0], error_stats['rmse_depths'][:,1], error_stats['rmse_depths'][:,2]
    
    # Compute climatology errors (climatology - observed full profiles)
    T_glorys_errors = T_glorys - T_obs
    S_glorys_errors = S_glorys - S_obs
    SH_glorys_errors = SH_glorys - SH_obs
    
    # Compute climatology RMSE by profile (average over depth for each profile)
    T_glorys_rmse_prof = np.sqrt(np.mean(T_glorys_errors**2, axis=1))
    S_glorys_rmse_prof = np.sqrt(np.mean(S_glorys_errors**2, axis=1))
    SH_glorys_rmse_prof = np.sqrt(np.mean(SH_glorys_errors**2, axis=1))
    
    # Compute climatology RMSE by depth (average over profiles for each depth)
    T_glorys_rmse_depth = np.sqrt(np.mean(T_glorys_errors**2, axis=0))
    S_glorys_rmse_depth = np.sqrt(np.mean(S_glorys_errors**2, axis=0))
    SH_glorys_rmse_depth = np.sqrt(np.mean(SH_glorys_errors**2, axis=0))
    
    # Get metadata
    metadata = test_data['metadata']
    n_profiles = len(metadata['latitude'])
    
    # Create dataset data dictionary
    dataset_data = {
        # ================== CLIMATOLOGY ==================
        'T_glorys': (['profile', 'depth'], T_glorys),
        'S_glorys': (['profile', 'depth'], S_glorys),
        'SH_glorys': (['profile', 'depth'], SH_glorys),
        
        # ================== ANOMALIES ==================
        # Observed anomalies
        'T_obs_anomaly': (['profile', 'depth'], T_obs_anom),
        'S_obs_anomaly': (['profile', 'depth'], S_obs_anom),
        'SH_obs_anomaly': (['profile', 'depth'], SH_obs_anom),
        
        # Predicted anomalies
        'T_pred_anomaly': (['profile', 'depth'], T_pred_anom),
        'S_pred_anomaly': (['profile', 'depth'], S_pred_anom),
        'SH_pred_anomaly': (['profile', 'depth'], SH_pred_anom),
        
        # ================== FULL PROFILES ==================
        # Observed full profiles (in-situ)
        'T_obs_insitu': (['profile', 'depth'], T_obs),
        'S_obs_insitu': (['profile', 'depth'], S_obs),
        'SH_obs_insitu': (['profile', 'depth'], SH_obs),
        
        # Predicted full profiles  
        'T_pred': (['profile', 'depth'], T_pred),
        'S_pred': (['profile', 'depth'], S_pred),
        'SH_pred': (['profile', 'depth'], SH_pred),
        
        # ================== ERROR STATISTICS ==================
        # Error fields (predicted - observed anomalies)
        'T_error': (['profile', 'depth'], T_errors),
        'S_error': (['profile', 'depth'], S_errors),
        'SH_error': (['profile', 'depth'], SH_errors),
        
        # RMSE by profile
        'T_rmse_profile': (['profile'], T_rmse_prof),
        'S_rmse_profile': (['profile'], S_rmse_prof),
        'SH_rmse_profile': (['profile'], SH_rmse_prof),
        
        # RMSE by depth
        'T_rmse_depth': (['depth'], T_rmse_depth),
        'S_rmse_depth': (['depth'], S_rmse_depth),
        'SH_rmse_depth': (['depth'], SH_rmse_depth),
        
        # ================== CLIMATOLOGY ERROR STATISTICS ==================
        # Climatology error fields (climatology - observed full profiles)
        'T_glorys_error': (['profile', 'depth'], T_glorys_errors),
        'S_glorys_error': (['profile', 'depth'], S_glorys_errors),
        'SH_glorys_error': (['profile', 'depth'], SH_glorys_errors),
        
        # Climatology RMSE by profile
        'T_glorys_rmse_profile': (['profile'], T_glorys_rmse_prof),
        'S_glorys_rmse_profile': (['profile'], S_glorys_rmse_prof),
        'SH_glorys_rmse_profile': (['profile'], SH_glorys_rmse_prof),
        
        # Climatology RMSE by depth
        'T_glorys_rmse_depth': (['depth'], T_glorys_rmse_depth),
        'S_glorys_rmse_depth': (['depth'], S_glorys_rmse_depth),
        'SH_glorys_rmse_depth': (['depth'], SH_glorys_rmse_depth),
        
        # ================== METADATA ==================
        'LATITUDE': (['profile'], metadata['latitude']),
        'LONGITUDE': (['profile'], metadata['longitude']),
        'X_EASE': (['profile'], metadata['x_ease']),
        'Y_EASE': (['profile'], metadata['y_ease']),
        'TIME': (['profile'], metadata['time']),
        'day_of_year': (['profile'], metadata['day_of_year']),

    }
    
    # Add augmentation variables if they exist
    if test_data['augmentation']['TEMP_aug_fraction'] is not None:
        dataset_data['TEMP_aug_fraction'] = (['profile'], test_data['augmentation']['TEMP_aug_fraction'])
    if test_data['augmentation']['PSAL_aug_fraction'] is not None:
        dataset_data['PSAL_aug_fraction'] = (['profile'], test_data['augmentation']['PSAL_aug_fraction'])
    if test_data['augmentation']['TEMP_augs'] is not None:
        dataset_data['TEMP_augs'] = (['profile', 'depth'], test_data['augmentation']['TEMP_augs'])
    if test_data['augmentation']['PSAL_augs'] is not None:
        dataset_data['PSAL_augs'] = (['profile', 'depth'], test_data['augmentation']['PSAL_augs'])
    
    # Create dataset
    ds_results = xr.Dataset(
        dataset_data,
        coords={
            'profile': range(n_profiles),
            'depth': metadata['depth'],
        },
        attrs={
            'title': 'LSTM Model Test Results - Complete Dataset',
            'description': 'Comprehensive results including climatology, anomalies, full profiles, and error statistics',
            'model_architecture': f"LSTM {'-'.join(map(str, Config.LSTM_UNITS))}",
            'test_data_file': Config.TEST_FILE,
            'T_rmse_total': float(error_stats['rmse_total'][1]),
            'S_rmse_total': float(error_stats['rmse_total'][2]),
            'SH_rmse_total': float(error_stats['rmse_total'][0]),
            'RMSEs_sum': float(error_stats['rmse_sum']),
            'n_test_profiles': n_profiles,
            'n_depth_levels': len(metadata['depth']),
        }
    )
    
    # Add TIME coordinate if available
    if metadata['time'] is not None:
        ds_results['TIME'] = (['profile'], metadata['time'])
    
    # ================== VARIABLE ATTRIBUTES ==================
    
    # Climatology attributes
    ds_results['T_glorys'].attrs = {'long_name': 'GLORYS Temperature Climatology', 'units': 'degree_C'}
    ds_results['S_glorys'].attrs = {'long_name': 'GLORYS Salinity Climatology', 'units': '1'}
    ds_results['SH_glorys'].attrs = {'long_name': 'GLORYS Steric Height Climatology', 'units': 'm'}
    
    # Anomaly attributes
    ds_results['T_obs_anomaly'].attrs = {'long_name': 'Observed Temperature Anomaly (in-situ - climatology)', 'units': 'degree_C'}
    ds_results['S_obs_anomaly'].attrs = {'long_name': 'Observed Salinity Anomaly (in-situ - climatology)', 'units': '1'}
    ds_results['SH_obs_anomaly'].attrs = {'long_name': 'Observed Steric Height Anomaly (in-situ - climatology)', 'units': 'm'}
    
    ds_results['T_pred_anomaly'].attrs = {'long_name': 'Predicted Temperature Anomaly', 'units': 'degree_C'}
    ds_results['S_pred_anomaly'].attrs = {'long_name': 'Predicted Salinity Anomaly', 'units': '1'}  
    ds_results['SH_pred_anomaly'].attrs = {'long_name': 'Predicted Steric Height Anomaly', 'units': 'm'}
    
    # Full profile attributes
    ds_results['T_obs_insitu'].attrs = {'long_name': 'Observed Temperature (in-situ)', 'units': 'degree_C'}
    ds_results['S_obs_insitu'].attrs = {'long_name': 'Observed Salinity (in-situ)', 'units': '1'}
    ds_results['SH_obs_insitu'].attrs = {'long_name': 'Observed Steric Height (in-situ)', 'units': 'm'}
    
    ds_results['T_pred'].attrs = {'long_name': 'Predicted Temperature', 'units': 'degree_C'}
    ds_results['S_pred'].attrs = {'long_name': 'Predicted Salinity', 'units': '1'}
    ds_results['SH_pred'].attrs = {'long_name': 'Predicted Steric Height', 'units': 'm'}
    
    # Error attributes
    ds_results['T_error'].attrs = {'long_name': 'Temperature Error (predicted - observed anomaly)', 'units': 'degree_C'}
    ds_results['S_error'].attrs = {'long_name': 'Salinity Error (predicted - observed anomaly)', 'units': '1'}
    ds_results['SH_error'].attrs = {'long_name': 'Steric Height Error (predicted - observed anomaly)', 'units': 'm'}
    
    ds_results['T_rmse_profile'].attrs = {'long_name': 'Temperature RMSE by profile', 'units': 'degree_C'}
    ds_results['S_rmse_profile'].attrs = {'long_name': 'Salinity RMSE by profile', 'units': '1'}
    ds_results['SH_rmse_profile'].attrs = {'long_name': 'Steric Height RMSE by profile', 'units': 'm'}
    
    ds_results['T_rmse_depth'].attrs = {'long_name': 'Temperature RMSE by depth', 'units': 'degree_C'}
    ds_results['S_rmse_depth'].attrs = {'long_name': 'Salinity RMSE by depth', 'units': '1'}
    ds_results['SH_rmse_depth'].attrs = {'long_name': 'Steric Height RMSE by depth', 'units': 'm'}
    
    # Climatology error attributes
    ds_results['T_glorys_error'].attrs = {'long_name': 'Temperature Climatology Error (climatology - observed)', 'units': 'degree_C'}
    ds_results['S_glorys_error'].attrs = {'long_name': 'Salinity Climatology Error (climatology - observed)', 'units': '1'}
    ds_results['SH_glorys_error'].attrs = {'long_name': 'Steric Height Climatology Error (climatology - observed)', 'units': 'm'}
    
    ds_results['T_glorys_rmse_profile'].attrs = {'long_name': 'Temperature Climatology RMSE by profile', 'units': 'degree_C'}
    ds_results['S_glorys_rmse_profile'].attrs = {'long_name': 'Salinity Climatology RMSE by profile', 'units': '1'}
    ds_results['SH_glorys_rmse_profile'].attrs = {'long_name': 'Steric Height Climatology RMSE by profile', 'units': 'm'}
    
    ds_results['T_glorys_rmse_depth'].attrs = {'long_name': 'Temperature Climatology RMSE by depth', 'units': 'degree_C'}
    ds_results['S_glorys_rmse_depth'].attrs = {'long_name': 'Salinity Climatology RMSE by depth', 'units': '1'}
    ds_results['SH_glorys_rmse_depth'].attrs = {'long_name': 'Steric Height Climatology RMSE by depth', 'units': 'm'}
    
    # Augmentation variable attributes
    if 'TEMP_aug_fraction' in ds_results:
        ds_results['TEMP_aug_fraction'].attrs = {'long_name': 'Temperature Augmentation Fraction', 'units': '1', 'description': 'Fraction of temperature profile that was augmented'}
    if 'PSAL_aug_fraction' in ds_results:
        ds_results['PSAL_aug_fraction'].attrs = {'long_name': 'Salinity Augmentation Fraction', 'units': '1', 'description': 'Fraction of salinity profile that was augmented'}
    if 'TEMP_augs' in ds_results:
        ds_results['TEMP_augs'].attrs = {'long_name': 'Temperature Augmented Values', 'units': 'degree_C', 'description': 'Temperature values after augmentation'}
    if 'PSAL_augs' in ds_results:
        ds_results['PSAL_augs'].attrs = {'long_name': 'Salinity Augmented Values', 'units': '1', 'description': 'Salinity values after augmentation'}
    
    # Metadata attributes (add these after the existing attribute assignments)
    if 'LATITUDE' in ds_results:
        ds_results['LATITUDE'].attrs = {'long_name': 'Latitude', 'units': 'degrees_north'}
    if 'LONGITUDE' in ds_results:
        ds_results['LONGITUDE'].attrs = {'long_name': 'Longitude', 'units': 'degrees_east'}
    if 'X_EASE' in ds_results:    
        ds_results['X_EASE'].attrs = {'long_name': 'EASE Grid X Coordinate', 'units': 'km'}
    if 'Y_EASE' in ds_results:    
        ds_results['Y_EASE'].attrs = {'long_name': 'EASE Grid Y Coordinate', 'units': 'km'}
    if 'TIME' in ds_results:
        ds_results['TIME'].attrs = {'long_name': 'Time', 'units': 'days since 1950-01-01'}
    if 'day_of_year' in ds_results:
        ds_results['day_of_year'].attrs = {'long_name': 'Day of Year', 'units': 'day'}    

    return ds_results

# ============================================================================
# DATA HANDLING FUNCTIONS (HELPER FUNCTIONS)
# ============================================================================

def load_datasets():
    """Load training, development, and test datasets"""
    
    print("Loading datasets...")
    
    # Check if files exist before loading
    for file_path, name in [(Config.TRAIN_FILE, 'Training'), (Config.DEV_FILE, 'Development'), (Config.TEST_FILE, 'Test')]:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"{name} dataset file not found: {file_path}")
    
    try:
        # Load datasets
        ds_train = xr.open_dataset(Config.TRAIN_FILE, decode_times=False)
        ds_dev = xr.open_dataset(Config.DEV_FILE, decode_times=False)
        ds_test = xr.open_dataset(Config.TEST_FILE, decode_times=False)
    except Exception as e:
        raise RuntimeError(f"Error loading dataset files: {e}")
    
    print(f"Training data: {ds_train.dims}")
    print(f"Development data: {ds_dev.dims}")
    print(f"Test data: {ds_test.dims}")
    
    return ds_train, ds_dev, ds_test

def prepare_dataset(ds, dataset_type):
    """Prepare input and output arrays for a dataset"""
    
    # Climatology data
    T_glorys = ds['T_glorys'].values
    S_glorys = ds['S_glorys'].values  
    SH_glorys = ds['SH_glorys'].values
    
    # Surface data (anomalies from surface climatology)
    sst_anomaly = np.repeat(
        ds['SST'].values[:, np.newaxis], T_glorys.shape[1], axis=1
    ) - np.repeat(T_glorys[:,0][:, np.newaxis], T_glorys.shape[1], axis=1)
    
    sss_anomaly = np.repeat(
        ds['SSS'].values[:, np.newaxis], S_glorys.shape[1], axis=1  
    ) - np.repeat(S_glorys[:,0][:, np.newaxis], S_glorys.shape[1], axis=1)
    
    adt_anomaly = np.repeat(
        ds['ADT'].values[:, np.newaxis], SH_glorys.shape[1], axis=1
    ) - np.repeat(SH_glorys[:,0][:, np.newaxis], SH_glorys.shape[1], axis=1)
    
    # In-situ data (anomalies from climatology)
    T_anom = ds['TEMP'].values - T_glorys
    S_anom = ds['PSAL'].values - S_glorys
    SH_anom = ds['SH'].values - ds['SH_glorys'].values
    
    # Metadata
    n_profiles = sst_anomaly.shape[0]
    n_depth = sst_anomaly.shape[1]
    
    # Seasonal cycle
    day_of_year = ds['day_of_year'].values
    seasonal_cos = np.cos(2 * np.pi * (day_of_year / 365) + 1)
    seasonal_sin = np.sin(2 * np.pi * (day_of_year / 365) + 1)
    
    # Prepare input arrays based on configuration
    input_arrays = []
    input_names = []
    
    if Config.INPUT_VARS['sst_anomaly']:
        input_arrays.append(sst_anomaly)
        input_names.append('sst_anomaly')
        
    if Config.INPUT_VARS['sss_anomaly']:
        input_arrays.append(sss_anomaly)
        input_names.append('sss_anomaly')
        
    if Config.INPUT_VARS['adt_anomaly']:
        input_arrays.append(adt_anomaly)
        input_names.append('adt_anomaly')
        
    if Config.INPUT_VARS['latitude']:
        lat_array = np.repeat(ds['LATITUDE'].values[:, np.newaxis], n_depth, axis=1)
        input_arrays.append(lat_array)
        input_names.append('latitude')
        
    if Config.INPUT_VARS['longitude']:
        lon_array = np.repeat(ds['LONGITUDE'].values[:, np.newaxis], n_depth, axis=1)
        input_arrays.append(lon_array)
        input_names.append('longitude')

    if Config.INPUT_VARS['x_ease']:
        # Apply false easting: add offset to avoid negative values and zero-crossing
        x_ease_array = np.repeat(ds['X_EASE'].values[:, np.newaxis], n_depth, axis=1)
        input_arrays.append(x_ease_array)
        input_names.append('x_ease')
        
    if Config.INPUT_VARS['y_ease']:
        # Apply false easting: add offset to avoid negative values and zero-crossing
        y_ease_array = np.repeat(ds['Y_EASE'].values[:, np.newaxis], n_depth, axis=1)
        input_arrays.append(y_ease_array)
        input_names.append('y_ease')
        
    if Config.INPUT_VARS['seasonal_cos']:
        cos_array = np.repeat(seasonal_cos[:, np.newaxis], n_depth, axis=1)
        input_arrays.append(cos_array)
        input_names.append('seasonal_cos')
        
    if Config.INPUT_VARS['seasonal_sin']:
        sin_array = np.repeat(seasonal_sin[:, np.newaxis], n_depth, axis=1)
        input_arrays.append(sin_array)
        input_names.append('seasonal_sin')
    
    # Stack input arrays
    X = np.stack(input_arrays, axis=2)  # [profiles, depth, variables]
    
    # Output arrays
    y = np.stack([SH_anom, T_anom, S_anom], axis=2)  # [profiles, depth, variables]
    
    return {
        'X': X,
        'y': y,
        'input_names': input_names,
        'climatology': {
            'T_glorys': T_glorys,
            'S_glorys': S_glorys,
            'SH_glorys': SH_glorys
        },
        'full_profiles': {
            'T': ds['TEMP'].values,
            'S': ds['PSAL'].values,
            'SH': ds['SH'].values
        },
        'augmentation': {
            'TEMP_aug_fraction': ds['TEMP_aug_fraction'].values if 'TEMP_aug_fraction' in ds else None,
            'PSAL_aug_fraction': ds['PSAL_aug_fraction'].values if 'PSAL_aug_fraction' in ds else None,
            'TEMP_augs': ds['TEMP_augs'].values if 'TEMP_augs' in ds else None,
            'PSAL_augs': ds['PSAL_augs'].values if 'PSAL_augs' in ds else None
        },
        'metadata': {
            'latitude': ds['LATITUDE'].values,
            'longitude': ds['LONGITUDE'].values,
            'x_ease': ds['X_EASE'].values,
            'y_ease': ds['Y_EASE'].values,
            'day_of_year': day_of_year,
            'time': ds['TIME'].values,
            'depth': ds['depth'].values
        }
    }

def normalize_data(train_data, dev_data):
    """Normalize input and output data using combined train+dev statistics (excluding test to prevent data leakage)"""
    
    print("Normalizing data...")
    
    # Combined statistics for inputs (train + dev only)
    X_combined = np.concatenate([train_data['X'], dev_data['X']], axis=0)
    X_min = X_combined.min(axis=(0,1))
    X_max = X_combined.max(axis=(0,1))
    X_range = X_max - X_min
    X_range[X_range == 0] = 1  # Avoid division by zero
    
    # Combined statistics for outputs (train + dev only)
    y_combined = np.concatenate([train_data['y'], dev_data['y']], axis=0)
    y_min = y_combined.min(axis=(0,1))
    y_max = y_combined.max(axis=(0,1))
    y_range = y_max - y_min
    y_range[y_range == 0] = 1  # Avoid division by zero
    
    # Normalize
    train_data['X_norm'] = (train_data['X'] - X_min) / X_range
    train_data['y_norm'] = (train_data['y'] - y_min) / y_range
    
    dev_data['X_norm'] = (dev_data['X'] - X_min) / X_range  
    dev_data['y_norm'] = (dev_data['y'] - y_min) / y_range
    
    # Store normalization parameters
    norm_params = {
        'X_min': X_min, 'X_max': X_max, 'X_range': X_range,
        'y_min': y_min, 'y_max': y_max, 'y_range': y_range
    }
    
    return train_data, dev_data, norm_params

if __name__ == "__main__":
    main()
