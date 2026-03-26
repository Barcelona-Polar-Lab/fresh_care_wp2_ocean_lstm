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
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration class for easy parameter adjustment"""
    
    # File paths
    TRAIN_FILE = 'fresh_data/var_depths_data_for_LSTM_B_wg_train63.nc'
    DEV_FILE = 'fresh_data/var_depths_data_for_LSTM_B_wg_dev21.nc'
    TEST_FILE = 'fresh_data/var_depths_data_for_LSTM_B_wg_test16.nc'
    MODEL_PARENT_DIR = 'models_surface_comparison'  # Parent directory for models
    MODEL_DIR = None  # Will be set dynamically based on LSTM units
    
    # Model architecture
    LSTM_UNITS = [40, 40]  # Can be changed to any list of integers
    DROPOUT_RATE = 0.2
    ACTIVATION = 'tanh'
    
    # Training parameters
    BATCH_SIZE = 16
    MAX_EPOCHS = 500
    LEARNING_RATE = 0.0001
    
    # Early stopping parameters
    PATIENCE = 5
    MIN_DELTA = 1e-6
    
    # Testing parameters
    TEST_REAL_DATA_ONLY = True  # Compute errors only on real (non-augmented) data points
    
    # Surface data source
    SURFACE_TS = 'satellite'  # 'satellite' for SST/SSS or 'glorys' for SST_glorys/SSS_glorys
    
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
    
    # Output variables (this is just informative!)
    OUTPUT_VARS = ['steric_height', 'temperature', 'salinity']
    
    @staticmethod
    def get_model_dir(lstm_units):
        """Get MODEL_DIR based on LSTM units and surface T/S source"""
        units_str = '_'.join(map(str, lstm_units))
        surface_suffix = '_sat' if Config.SURFACE_TS == 'satellite' else '_glor'
        return f'{Config.MODEL_PARENT_DIR}/model_LSTM_{units_str}{surface_suffix}'

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

        """
        input_size: type(int)
        
            number of input features. eg:
            A batch with 7 of them might look like:
            batch_X.shape = [16, 50, 7]
                             |   |    ─ 7 features at each depth
                             |    ────  50 depth levels (sequence)
                              ───────── 16 profiles (batch)

            So each element in a sequence has 7 features

        output_size: type(int)
        
            same principle, a label/prediction might
            have the shape [50, 3]
                            |   output features at each depth
                            depth levels

        lstm_units: type(list of int or just an int)
            number of LSTM units in each layer. 
            
             eg: lstm_units=35 means 35 units in a single layer
                 lstm_units=[50, 30] means 50 units in the first layer and 30 in the second layer

         dropout_rate: type(float)
            dropout rate for regularization (default: 0.2)
        """
        
        self.input_size = input_size
        self.output_size = output_size
        self.lstm_units = lstm_units if isinstance(lstm_units, list) else [lstm_units]

        # Input dropout
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList() # modules list allows tracking params
                                           # for multiple layers in a clean way
        layer_input_size = input_size
        
        for i, units in enumerate(self.lstm_units):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_size, # input size of the layer
                    hidden_size=units, # num of LSTM units: output size of the layer
                    batch_first=True,
                    dropout=dropout_rate if i < len(self.lstm_units) - 1 else 0
                    # dropout only between layers, not on the last layer's output
                    # we set that below on the output layer
                )
            )
            layer_input_size = units
        
        # Output layer (applied to each time/depth step)
        self.output_layer = nn.Linear(
            self.lstm_units[-1], # last LSTM layer's output size
            output_size # number of output features (e.g., 3 for SH, T, S)
        )
        self.output_dropout = nn.Dropout(dropout_rate)
    

    def forward(self, x, lengths=None):
        """
        Forward pass with two supported input modes:
        
        MODE 1: Fixed-length sequences (no padding)
            - x: Regular tensor [batch, max_seq_len, features]
            - lengths: None
            - All sequences must have the same length with NO padding values
            - LSTM processes all positions directly
            
        MODE 2: Variable-length sequences (with packing)
            - x: Regular tensor [batch, max_seq_len, features] OR PackedSequence
            - lengths: Tensor of actual sequence lengths
            - Data will be packed to skip padding during LSTM processing
            - This prevents padding contamination in LSTM states
            
        IMPORTANT: Do NOT pass padded tensors without lengths parameter.
        This would cause padding values to contaminate LSTM hidden states.
        """
        # Handle both packed and regular sequences
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            # Input is already packed, unpack for dropout, then repack
            packed_input = x
            x, lengths = torch.nn.utils.rnn.pad_packed_sequence(packed_input, batch_first=True)
            # Apply input dropout
            x = self.input_dropout(x)
            # Repack for LSTM processing
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        else:
            # Regular tensor input, apply dropout directly
            x = self.input_dropout(x)

            if lengths is not None: # In case of fixed-length sequences
                                    # lengths can be None and we skip packing
                
                # Pack the sequence for variable lengths
                x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
            
        # All below can handle packed or regular sequences, 
        # LSTM layers natively accept both packed and regular tensors.

        # In the case we have a regular tensor and no lengths and some padding, this
        # will just process it as is without packing sending padding values through
        # the LSTM, wich is BAD.

        # LSTM layers
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        
        # Unpack if needed for final layer processing
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)

        # QUESTION FOR MARIO: applying the output linear layer (Wx+b) to a zero
        # padded sequence, this is not affecting the the computation thanks
        # to the mask we apply during loss calculation, right? I don't really
        # understand why...
        
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
    parser.add_argument('--test_real_data_only', type=bool, default=None,
                       help='Compute RMSE only on real (non-augmented) data points')
    parser.add_argument('--surface_ts', choices=['satellite', 'glorys'], default=None,
                       help='Surface T/S data source: satellite (SST/SSS) or glorys (SST_glorys/SSS_glorys)')
     
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
    if args.test_real_data_only is not None:
        Config.TEST_REAL_DATA_ONLY = args.test_real_data_only
    if args.surface_ts:
        Config.SURFACE_TS = args.surface_ts
    
    print(f"Configuration: LSTM={Config.LSTM_UNITS}, Batch={Config.BATCH_SIZE}, Max Epochs={Config.MAX_EPOCHS}")
    print(f"LR={Config.LEARNING_RATE}, Dropout={Config.DROPOUT_RATE}, Patience={Config.PATIENCE}")
    print(f"Surface T/S source: {Config.SURFACE_TS}")
    
    # Set model directory
    Config.MODEL_DIR = Config.get_model_dir(Config.LSTM_UNITS)
    print(f"Model directory: {Config.MODEL_DIR}")
    
    # Run based on mode
    if args.mode in ['train', 'both']:
        if not check_model_directory():
            return
        run_training()
        
    if args.mode in ['test', 'both']:
        run_testing()

def check_model_directory():
    """Check if model directory exists and prompt user for action"""
    model_path = Path(Config.MODEL_DIR)
    if model_path.exists():
        print(f"\nWarning: Model directory '{Config.MODEL_DIR}' already exists.")
        while True:
            response = input("Do you want to overwrite it? [y/N]: ").strip().lower()
            if response in ['y', 'yes']:
                print("Proceeding with overwrite...")
                return True
            elif response in ['n', 'no', '']:
                print("Aborting to avoid overwriting existing model.")
                return False
            else:
                print("Please answer 'y' for yes or 'n' for no.")
    return True


def run_training():
    """Run model training"""
    
    print("=== TRAINING MODE ===")
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("No GPU found, using CPU.")
    
    # Create model directory
    Path(Config.MODEL_DIR).mkdir(parents=True, exist_ok=True)
    
    # Load datasets
    ds_train, ds_dev, ds_test = load_datasets()
    
    # Note: Model directory already set in main() before calling this function
    # Prepare data
    train_data = prepare_dataset(ds_train, 'train')
    dev_data = prepare_dataset(ds_dev, 'dev')
    test_data = prepare_dataset(ds_test, 'test')  # Keep for final evaluation only
    
    # Normalize data (exclude test from statistics to prevent data leakage)
    train_data, dev_data, test_data, norm_params = normalize_data(train_data, dev_data, test_data)
    
    # Get data dimensions (handle both list and array formats)
    if isinstance(train_data['X'], list):
        n_input_vars = train_data['X'][0].shape[1]  # For variable-length sequences
        n_output_vars = train_data['y'][0].shape[1]
        n_profiles = len(train_data['X'])
    else:
        n_input_vars = train_data['X'].shape[2]  # For fixed-length sequences
        n_output_vars = train_data['y'].shape[2]
        n_profiles = train_data['X'].shape[0]
    
    print(f"Training profiles: {n_profiles}")
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
        'SURFACE_TS': Config.SURFACE_TS
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
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
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
    if test_data.get('has_bathymetry', False):
        # Variable-length sequences
        test_data['X_norm'] = [(X - norm_params['X_min']) / norm_params['X_range'] for X in test_data['X']]
        test_data['y_norm'] = [(y - norm_params['y_min']) / norm_params['y_range'] for y in test_data['y']]
    else:
        # Fixed-length sequences
        test_data['X_norm'] = (test_data['X'] - norm_params['X_min']) / norm_params['X_range']
        test_data['y_norm'] = (test_data['y'] - norm_params['y_min']) / norm_params['y_range']
    
    # Print test data info (handle both list and array formats)
    if test_data.get('has_bathymetry', False):
        print(f"Test data: {len(test_data['X'])} profiles with variable depths")
    else:
        print(f"Test data: {test_data['X'].shape[0]} profiles, {test_data['X'].shape[1]} depths")
    
    # Make predictions
    y_pred = make_predictions(model, test_data, norm_params, device)
    
    # Compute error statistics
    error_stats = compute_error_statistics(y_pred, test_data['y'], test_data)
    
    # Print overall RMSE
    print(f"\nOverall RMSE:")
    print(f"Steric Height: {error_stats['rmse_total'][0]:.3f} cm")
    print(f"Temperature: {error_stats['rmse_total'][1]:.3f} °C")
    print(f"Salinity: {error_stats['rmse_total'][2]:.3f} PSU")
    print(f"Sum of RMSEs: {error_stats['rmse_sum']:.3f}")
    
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
        
        for batch_data in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
            if len(batch_data) == 3:  # Variable-length sequences
                batch_X, batch_y, lengths = batch_data
                batch_X, batch_y, lengths = batch_X.to(device), batch_y.to(device), lengths.to(device)
                
                optimizer.zero_grad()
                outputs = model(batch_X, lengths)
                
                # Create mask for variable lengths to ignore padded positions in loss
                mask = torch.zeros_like(batch_y, dtype=torch.bool)
                for i, length in enumerate(lengths):
                    mask[i, :length] = True
                
                # Compute masked loss
                loss = criterion(outputs[mask], batch_y[mask])
            else:  # Fixed-length sequences
                batch_X, batch_y = batch_data
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
            for batch_data in val_loader:
                if len(batch_data) == 3:  # Variable-length sequences
                    batch_X, batch_y, lengths = batch_data
                    batch_X, batch_y, lengths = batch_X.to(device), batch_y.to(device), lengths.to(device)
                    
                    outputs = model(batch_X, lengths)
                    
                    # Create mask for variable lengths
                    mask = torch.zeros_like(batch_y, dtype=torch.bool)
                    for i, length in enumerate(lengths):
                        mask[i, :length] = True
                    
                    # Compute masked loss
                    loss = criterion(outputs[mask], batch_y[mask])
                else:  # Fixed-length sequences
                    batch_X, batch_y = batch_data
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
    """Create PyTorch data loaders for training and validation with support for variable-length sequences"""
    
    # Validate data based on sequence type
    if not train_data.get('has_bathymetry', False):
        # Fixed-length sequences: validate no padding values present
        print("Validating fixed-length data (checking for padding values)...")
        
        # Check for common padding values: NaN, -999, -99
        for data_dict, name in [(train_data, 'training'), (dev_data, 'validation')]:
            X_norm = data_dict['X_norm']
            y_norm = data_dict['y_norm']
            
            # Check for NaN
            if np.any(np.isnan(X_norm)) or np.any(np.isnan(y_norm)):
                raise ValueError(
                    f"ERROR: Found NaN values in {name} data with fixed-length sequences.\n"
                    f"Fixed-length mode requires all sequences to be fully filled with NO padding.\n"
                    f"Use variable-length mode (has_bathymetry=True) if data contains NaN padding."
                )
            
            # Check for -999 padding value
            if np.any(X_norm == -999.0) or np.any(y_norm == -999.0):
                raise ValueError(
                    f"ERROR: Found -999 padding values in {name} data with fixed-length sequences.\n"
                    f"Fixed-length mode requires all sequences to be fully filled with NO padding.\n"
                    f"Use variable-length mode (has_bathymetry=True) if data contains padding."
                )
            
            # Check for -99 padding value
            if np.any(X_norm == -99.0) or np.any(y_norm == -99.0):
                raise ValueError(
                    f"ERROR: Found -99 padding values in {name} data with fixed-length sequences.\n"
                    f"Fixed-length mode requires all sequences to be fully filled with NO padding.\n"
                    f"Use variable-length mode (has_bathymetry=True) if data contains padding."
                )
        
        print("  ✓ No padding values detected - data is valid for fixed-length mode")
    
    if train_data.get('has_bathymetry', False):
        # Variable-length sequences: need custom dataset and collate function
        from torch.nn.utils.rnn import pad_sequence
        
        def collate_variable_length(batch):
            """Custom collate function for variable-length sequences"""
            X_batch, y_batch, lengths_batch = zip(*batch)
            
            # Convert to tensors
            X_tensors = [torch.FloatTensor(x) for x in X_batch]
            y_tensors = [torch.FloatTensor(y) for y in y_batch]
            lengths = torch.LongTensor(lengths_batch)
            
            # Pad sequences to same length within batch
            X_padded = pad_sequence(X_tensors, batch_first=True, padding_value=-999.0)
            y_padded = pad_sequence(y_tensors, batch_first=True, padding_value=-999.0)
            
            return X_padded, y_padded, lengths
        
        # Custom dataset class for variable-length sequences
        class VariableLengthDataset:
            def __init__(self, X_list, y_list, lengths):
                self.X_list = X_list
                self.y_list = y_list
                self.lengths = lengths
                
            def __len__(self):
                return len(self.X_list)
                
            def __getitem__(self, idx):
                return self.X_list[idx], self.y_list[idx], self.lengths[idx]
        
        # Create datasets
        train_dataset = VariableLengthDataset(train_data['X_norm'], train_data['y_norm'], train_data['lengths'])
        val_dataset = VariableLengthDataset(dev_data['X_norm'], dev_data['y_norm'], dev_data['lengths'])
        
        # Create data loaders with custom collate function
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_variable_length)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_variable_length)
        
    else:
        # Fixed-length sequences (original behavior)
        X_train = torch.FloatTensor(train_data['X_norm'])
        y_train = torch.FloatTensor(train_data['y_norm'])
        train_dataset = TensorDataset(X_train, y_train)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        
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
    
    model.eval()
    predictions = []
    
    if test_data.get('has_bathymetry', False):
        # Variable-length sequences: process individually or in small batches
        print("Processing variable-length sequences...")
        
        from torch.nn.utils.rnn import pad_sequence
        
        # Process in batches for efficiency
        batch_size = Config.BATCH_SIZE * 2  # Smaller batch for variable lengths
        n_profiles = len(test_data['X_norm'])
        total_batches = (n_profiles + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in range(total_batches):
                start_idx = batch_idx * batch_size
                end_idx = min(start_idx + batch_size, n_profiles)
                
                # Prepare batch
                batch_X = test_data['X_norm'][start_idx:end_idx]
                batch_lengths = test_data['lengths'][start_idx:end_idx]
                
                # Convert to tensors and pad
                X_tensors = [torch.FloatTensor(x) for x in batch_X]
                lengths_tensor = torch.LongTensor(batch_lengths)
                X_padded = pad_sequence(X_tensors, batch_first=True, padding_value=-999.0).to(device)
                lengths_tensor = lengths_tensor.to(device)
                
                # Make predictions
                batch_pred = model(X_padded, lengths_tensor).cpu().numpy()
                
                # Extract only valid (non-padded) predictions for each profile
                for i, length in enumerate(batch_lengths):
                    profile_pred = batch_pred[i, :length, :]  # Only up to actual length
                    predictions.append(profile_pred)
                
                if batch_idx % 20 == 0 or batch_idx == total_batches - 1:
                    progress = ((batch_idx + 1) / total_batches) * 100
                    print(f"  Prediction batch {batch_idx + 1}/{total_batches} ({progress:.1f}%)")
        
        # Denormalize predictions (each profile separately)
        y_pred = [(pred * norm_params['y_range'] + norm_params['y_min']) for pred in predictions]
        
    else:
        # Fixed-length sequences (original behavior)
        X_test_norm = test_data['X_norm']
        X_test_tensor = torch.FloatTensor(X_test_norm).to(device)
        
        batch_size = Config.BATCH_SIZE * 4  # Larger batch for inference
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
        y_pred = y_pred_norm * norm_params['y_range'] + norm_params['y_min']
    
    return y_pred

def compute_error_statistics(y_pred, y_true, test_data=None):
    """Compute comprehensive error statistics for both fixed and variable-length sequences"""
    
    print("Computing error statistics...")
    
    # Create mask for real (non-augmented) data if requested
    real_data_mask = None
    if (Config.TEST_REAL_DATA_ONLY and 
        test_data and 
        test_data['augmentation']['TEMP_augs'] is not None and 
        test_data['augmentation']['PSAL_augs'] is not None):
        # Mask: True where data is real (augs == 0 for both T and S)
        temp_real = (test_data['augmentation']['TEMP_augs'] == 0)
        psal_real = (test_data['augmentation']['PSAL_augs'] == 0)
        real_data_mask = temp_real & psal_real
        
        n_real = np.sum(real_data_mask)
        n_total = real_data_mask.size
        print(f"Testing on real data only: {n_real}/{n_total} depth points ({100*n_real/n_total:.1f}%)")
    
    if test_data and test_data.get('has_bathymetry', False):
        # Variable-length sequences: need to handle different lengths
        lengths = test_data['lengths']
        max_length = max(lengths)
        n_profiles = len(y_pred)
        n_outputs = y_pred[0].shape[1]
        
        # Truncate real_data_mask to max_length if it exists
        if real_data_mask is not None:
            real_data_mask = real_data_mask[:, :max_length]
        
        # Pad predictions and observations to consistent shape for analysis
        y_pred_padded = np.full((n_profiles, max_length, n_outputs), np.nan)
        y_true_padded = np.full((n_profiles, max_length, n_outputs), np.nan)
        
        for i, length in enumerate(lengths):
            y_pred_padded[i, :length, :] = y_pred[i]
            if isinstance(y_true, list):
                y_true_padded[i, :length, :] = y_true[i]
            else:
                y_true_padded[i, :length, :] = y_true[i, :length, :]
        
        # Compute errors only for valid (non-NaN) positions
        errors = y_pred_padded - y_true_padded
        
        # RMSE by profile (average over valid depths for each profile)
        rmse_profiles = np.full((n_profiles, n_outputs), np.nan)
        for i, length in enumerate(lengths):
            valid_errors = errors[i, :length, :]
            
            # Apply real data mask if provided
            if real_data_mask is not None:
                # Get mask for this profile's valid depths
                profile_real_mask = real_data_mask[i, :length]
                if np.any(profile_real_mask):
                    # Only compute RMSE on real data points
                    valid_errors = valid_errors[profile_real_mask]
                    rmse_profiles[i, :] = np.sqrt(np.mean(valid_errors**2, axis=0))
                # else: leave as NaN if no real data in this profile
            else:
                rmse_profiles[i, :] = np.sqrt(np.mean(valid_errors**2, axis=0))
        
        # RMSE by depth (average over profiles at each depth, only where data exists)
        rmse_depths = np.full((max_length, n_outputs), np.nan)
        for d in range(max_length):
            # Find profiles that have data at this depth
            valid_mask = ~np.isnan(errors[:, d, :])
            
            # Apply real data mask if provided
            if real_data_mask is not None:
                valid_mask = valid_mask & real_data_mask[:, d:d+1]  # Broadcast depth dimension
            
            for out_idx in range(n_outputs):
                valid_errors_at_depth = errors[:, d, out_idx][valid_mask[:, out_idx]]
                if len(valid_errors_at_depth) > 0:
                    rmse_depths[d, out_idx] = np.sqrt(np.mean(valid_errors_at_depth**2))
        
        # Overall RMSE (using all valid points)
        valid_mask = ~np.isnan(errors)
        
        # Apply real data mask if provided
        if real_data_mask is not None:
            # Broadcast real_data_mask to match errors shape [profiles, depth, outputs]
            real_mask_3d = np.repeat(real_data_mask[:, :, np.newaxis], n_outputs, axis=2)
            valid_mask = valid_mask & real_mask_3d
        
        rmse_total = np.full(n_outputs, np.nan)
        for out_idx in range(n_outputs):
            valid_errors = errors[:, :, out_idx][valid_mask[:, :, out_idx]]
            if len(valid_errors) > 0:
                rmse_total[out_idx] = np.sqrt(np.mean(valid_errors**2))
        
        # Sum of all total RMSEs
        rmse_sum = np.sum(rmse_total)
        
        return {
            'errors': errors,  # Padded with NaN for missing depths
            'rmse_profiles': rmse_profiles,
            'rmse_depths': rmse_depths,  # NaN for depths not reached by any profile
            'rmse_total': rmse_total,
            'rmse_sum': rmse_sum,
            'lengths': lengths,  # Store lengths for later use
            'has_variable_lengths': True
        }
        
    else:
        # Fixed-length sequences (original behavior)
        if isinstance(y_pred, list):
            # Convert list to array if needed
            y_pred = np.array(y_pred)
        if isinstance(y_true, list):
            y_true = np.array(y_true)
            
        errors = y_pred - y_true
        
        # Apply real data mask if provided
        if real_data_mask is not None:
            # Mask out augmented data points
            errors_masked = np.ma.masked_where(~real_data_mask[:, :, np.newaxis], errors)
            
            # RMSE by profile (average over depth for each profile, only real data)
            rmse_profiles = np.sqrt(np.mean(errors_masked**2, axis=1))
            
            # RMSE by depth (average over profiles for each depth, only real data)
            rmse_depths = np.sqrt(np.mean(errors_masked**2, axis=0))
            
            # Overall RMSE (only real data)
            rmse_total = np.sqrt(np.mean(errors_masked**2, axis=(0,1)))
        else:
            # RMSE by profile (average over depth for each profile)
            rmse_profiles = np.sqrt(np.mean(errors**2, axis=1))
            
            # RMSE by depth (average over profiles for each depth) 
            rmse_depths = np.sqrt(np.mean(errors**2, axis=0))
            
            # Overall RMSE
            rmse_total = np.sqrt(np.mean(errors**2, axis=(0,1)))
        
        # Sum of all total RMSEs
        rmse_sum = np.sum(rmse_total)
        
        return {
            'errors': errors,
            'rmse_profiles': rmse_profiles,
            'rmse_depths': rmse_depths,
            'rmse_total': rmse_total,
            'rmse_sum': rmse_sum,
            'has_variable_lengths': False
        }

def create_results_dataset(test_data, y_pred, error_stats):
    """Create comprehensive results dataset with all profiles and statistics"""
    
    print("Creating results dataset...")
    
    # Handle variable vs fixed length data
    if test_data.get('has_bathymetry', False):
        # Variable-length sequences: need to pad to consistent shape
        lengths = test_data['lengths']
        max_length = max(lengths)
        n_profiles = len(y_pred)
        
        # Pad all arrays to max_length for xarray compatibility
        def pad_to_max_length(data_list, fill_value=np.nan):
            padded = np.full((n_profiles, max_length), fill_value)
            for i, length in enumerate(lengths):
                if isinstance(data_list[i], np.ndarray) and data_list[i].ndim == 1:
                    padded[i, :length] = data_list[i]
                elif isinstance(data_list[i], np.ndarray) and data_list[i].ndim == 2:
                    # Handle 2D profiles (multiple variables)
                    return pad_to_max_length_2d(data_list, fill_value)
            return padded
        
        def pad_to_max_length_2d(data_list, fill_value=np.nan):
            n_vars = data_list[0].shape[1] if len(data_list) > 0 else 3
            padded = np.full((n_profiles, max_length, n_vars), fill_value)
            for i, length in enumerate(lengths):
                padded[i, :length, :] = data_list[i]
            return padded
        
        # Get climatology and pad to max_length
        T_glorys = test_data['climatology']['T_glorys']
        S_glorys = test_data['climatology']['S_glorys']
        SH_glorys = test_data['climatology']['SH_glorys']
        
        # Pad climatology arrays
        T_glorys_padded = np.full((n_profiles, max_length), np.nan)
        S_glorys_padded = np.full((n_profiles, max_length), np.nan)
        SH_glorys_padded = np.full((n_profiles, max_length), np.nan)
        
        for i, length in enumerate(lengths):
            T_glorys_padded[i, :length] = T_glorys[i, :length]
            S_glorys_padded[i, :length] = S_glorys[i, :length]
            SH_glorys_padded[i, :length] = SH_glorys[i, :length]
        
        # Extract predictions and observations (use padded arrays from error_stats)
        y_pred_padded = pad_to_max_length_2d(y_pred)
        y_obs_padded = pad_to_max_length_2d(test_data['y'])
        
        SH_pred_anom, T_pred_anom, S_pred_anom = y_pred_padded[:,:,0], y_pred_padded[:,:,1], y_pred_padded[:,:,2]
        SH_obs_anom, T_obs_anom, S_obs_anom = y_obs_padded[:,:,0], y_obs_padded[:,:,1], y_obs_padded[:,:,2]
        
        # Use provided depth array (full depth range)
        depth_array = test_data['metadata']['depth'][:max_length]  # Truncate to max_length
        
    else:
        # Fixed-length sequences (original behavior)
        SH_pred_anom, T_pred_anom, S_pred_anom = y_pred[:,:,0], y_pred[:,:,1], y_pred[:,:,2]
        SH_obs_anom, T_obs_anom, S_obs_anom = test_data['y'][:,:,0], test_data['y'][:,:,1], test_data['y'][:,:,2]
        
        T_glorys_padded = test_data['climatology']['T_glorys']
        S_glorys_padded = test_data['climatology']['S_glorys'] 
        SH_glorys_padded = test_data['climatology']['SH_glorys']
        
        depth_array = test_data['metadata']['depth']
    
    # Compute full profiles (anomalies + climatology)
    SH_pred = SH_pred_anom + SH_glorys_padded
    T_pred = T_pred_anom + T_glorys_padded
    S_pred = S_pred_anom + S_glorys_padded
    
    # Observed full profiles (pad if needed)
    if test_data.get('has_bathymetry', False):
        T_obs_full = test_data['full_profiles']['T']
        S_obs_full = test_data['full_profiles']['S']
        SH_obs_full = test_data['full_profiles']['SH']
        
        # Pad full profiles
        T_obs = np.full((n_profiles, max_length), np.nan)
        S_obs = np.full((n_profiles, max_length), np.nan)
        SH_obs = np.full((n_profiles, max_length), np.nan)
        
        for i, length in enumerate(lengths):
            T_obs[i, :length] = T_obs_full[i, :length]
            S_obs[i, :length] = S_obs_full[i, :length]
            SH_obs[i, :length] = SH_obs_full[i, :length]
    else:
        T_obs = test_data['full_profiles']['T']
        S_obs = test_data['full_profiles']['S']  
        SH_obs = test_data['full_profiles']['SH']
    
    # Extract error statistics (handle variable lengths)
    errors = error_stats['errors']
    SH_errors, T_errors, S_errors = errors[:,:,0], errors[:,:,1], errors[:,:,2]
    
    # RMSE statistics
    SH_rmse_prof, T_rmse_prof, S_rmse_prof = error_stats['rmse_profiles'][:,0], error_stats['rmse_profiles'][:,1], error_stats['rmse_profiles'][:,2]
    SH_rmse_depth, T_rmse_depth, S_rmse_depth = error_stats['rmse_depths'][:,0], error_stats['rmse_depths'][:,1], error_stats['rmse_depths'][:,2]
    
    # Compute climatology errors (climatology - observed full profiles)
    T_glorys_errors = T_glorys_padded - T_obs
    S_glorys_errors = S_glorys_padded - S_obs
    SH_glorys_errors = SH_glorys_padded - SH_obs
    
    # Compute climatology RMSE (handle NaN values for variable lengths)
    if test_data.get('has_bathymetry', False):
        # For variable lengths, compute RMSE only over valid depths
        T_glorys_rmse_prof = np.full(n_profiles, np.nan)
        S_glorys_rmse_prof = np.full(n_profiles, np.nan)
        SH_glorys_rmse_prof = np.full(n_profiles, np.nan)
        
        for i, length in enumerate(lengths):
            T_glorys_rmse_prof[i] = np.sqrt(np.nanmean(T_glorys_errors[i, :length]**2))
            S_glorys_rmse_prof[i] = np.sqrt(np.nanmean(S_glorys_errors[i, :length]**2))
            SH_glorys_rmse_prof[i] = np.sqrt(np.nanmean(SH_glorys_errors[i, :length]**2))
        
        # RMSE by depth (average over profiles, ignoring NaN)
        T_glorys_rmse_depth = np.sqrt(np.nanmean(T_glorys_errors**2, axis=0))
        S_glorys_rmse_depth = np.sqrt(np.nanmean(S_glorys_errors**2, axis=0))
        SH_glorys_rmse_depth = np.sqrt(np.nanmean(SH_glorys_errors**2, axis=0))
        
    else:
        # Fixed lengths (original behavior)
        T_glorys_rmse_prof = np.sqrt(np.mean(T_glorys_errors**2, axis=1))
        S_glorys_rmse_prof = np.sqrt(np.mean(S_glorys_errors**2, axis=1))
        SH_glorys_rmse_prof = np.sqrt(np.mean(SH_glorys_errors**2, axis=1))
        
        T_glorys_rmse_depth = np.sqrt(np.mean(T_glorys_errors**2, axis=0))
        S_glorys_rmse_depth = np.sqrt(np.mean(S_glorys_errors**2, axis=0))
        SH_glorys_rmse_depth = np.sqrt(np.mean(SH_glorys_errors**2, axis=0))
    
    # Get metadata
    metadata = test_data['metadata']
    n_profiles = len(metadata['latitude'])
    
    # Create dataset
    ds_results = xr.Dataset(
        {
            # ================== CLIMATOLOGY ==================
            'T_glorys': (['profile', 'depth'], T_glorys_padded),
            'S_glorys': (['profile', 'depth'], S_glorys_padded),
            'SH_glorys': (['profile', 'depth'], SH_glorys_padded),
            
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
        },
        coords={
            'profile': range(n_profiles),
            'depth': depth_array,
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
            'n_depth_levels': len(depth_array),
        }
    )
    
    # Add augmentation variables if they exist
    if test_data['augmentation']['TEMP_aug_fraction'] is not None:
        ds_results['TEMP_aug_fraction'] = (['profile'], test_data['augmentation']['TEMP_aug_fraction'])
    if test_data['augmentation']['PSAL_aug_fraction'] is not None:
        ds_results['PSAL_aug_fraction'] = (['profile'], test_data['augmentation']['PSAL_aug_fraction'])
    if test_data['augmentation']['TEMP_augs'] is not None:
        temp_augs = test_data['augmentation']['TEMP_augs'][:, :len(depth_array)]
        ds_results['TEMP_augs'] = (['profile', 'depth'], temp_augs)
    if test_data['augmentation']['PSAL_augs'] is not None:
        psal_augs = test_data['augmentation']['PSAL_augs'][:, :len(depth_array)]
        ds_results['PSAL_augs'] = (['profile', 'depth'], psal_augs)
    
    # Add TIME coordinate if available
    if metadata['time'] is not None:
        ds_results['TIME'] = (['profile'], metadata['time'])
    
    # Add bathymetry information if available
    if test_data.get('has_bathymetry', False):
        ds_results['max_depth_idx'] = (['profile'], test_data['max_depth_idx'])
        ds_results['profile_lengths'] = (['profile'], lengths)
        ds_results['max_depth_idx'].attrs = {'long_name': 'Maximum depth index for each profile', 'units': '1'}
        ds_results['profile_lengths'].attrs = {'long_name': 'Number of valid depth levels per profile', 'units': '1'}
    
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
        ds_results['TEMP_aug_fraction'].attrs = {'long_name': 'Temperature Augmentation Fraction', 'units': '1'}
    if 'PSAL_aug_fraction' in ds_results:
        ds_results['PSAL_aug_fraction'].attrs = {'long_name': 'Salinity Augmentation Fraction', 'units': '1'}
    if 'TEMP_augs' in ds_results:
        ds_results['TEMP_augs'].attrs = {'long_name': 'Temperature Augmentations', 'units': '1'}
    if 'PSAL_augs' in ds_results:
        ds_results['PSAL_augs'].attrs = {'long_name': 'Salinity Augmentations', 'units': '1'}
    
    # Metadata attributes (add these after the existing attribute assignments)
    if 'LATITUDE' in ds_results:
        ds_results['LATITUDE'].attrs = {'long_name': 'Profile Latitude', 'units': 'degrees_north'}
    if 'LONGITUDE' in ds_results:
        ds_results['LONGITUDE'].attrs = {'long_name': 'Profile Longitude', 'units': 'degrees_east'}
    
    # Augmentation variable attributes
    if 'TEMP_aug_fraction' in ds_results:
        ds_results['TEMP_aug_fraction'].attrs = {'long_name': 'Temperature Augmentation Fraction', 'units': '1'}
    if 'PSAL_aug_fraction' in ds_results:
        ds_results['PSAL_aug_fraction'].attrs = {'long_name': 'Salinity Augmentation Fraction', 'units': '1'}
    if 'TEMP_augs' in ds_results:
        ds_results['TEMP_augs'].attrs = {'long_name': 'Temperature Augmentations', 'units': '1'}
    if 'PSAL_augs' in ds_results:
        ds_results['PSAL_augs'].attrs = {'long_name': 'Salinity Augmentations', 'units': '1'}
    
    # Metadata attributes (add these after the existing attribute assignments)
    if 'LATITUDE' in ds_results:
        ds_results['LATITUDE'].attrs = {'long_name': 'Profile Latitude', 'units': 'degrees_north'}
    if 'LONGITUDE' in ds_results:
        ds_results['LONGITUDE'].attrs = {'long_name': 'Profile Longitude', 'units': 'degrees_east'}
    if 'X_EASE' in ds_results:    
        ds_results['X_EASE'].attrs = {'long_name': 'EASE Grid X Coordinate', 'units': 'm'}
    if 'Y_EASE' in ds_results:    
        ds_results['Y_EASE'].attrs = {'long_name': 'EASE Grid Y Coordinate', 'units': 'm'}
    if 'TIME' in ds_results:
        # Copy original TIME attributes from source dataset
        if metadata.get('time_attrs'):
            ds_results['TIME'].attrs = metadata['time_attrs']
        else:
            ds_results['TIME'].attrs = {'long_name': 'Profile Time'}
    if 'day_of_year' in ds_results:
        ds_results['day_of_year'].attrs = {'long_name': 'Day of Year'}
    
    return ds_results

# ============================================================================
# DATA HANDLING FUNCTIONS (HELPER FUNCTIONS)
# ============================================================================

def detect_nan_tails(T_data, S_data, SH_data):
    """
    Detect if data has variable-length sequences with NaN tails.
    Returns (has_nan_tails, lengths_array) where lengths_array contains
    the last valid index for each profile.
    """
    n_profiles = T_data.shape[0]
    n_depths = T_data.shape[1]
    
    # Check if there are any NaNs at all
    has_any_nans = np.any(np.isnan(T_data)) or np.any(np.isnan(S_data)) or np.any(np.isnan(SH_data))
    
    if not has_any_nans:
        return False, None
    
    # Compute last valid index for each profile (minimum across all three variables)
    lengths = np.zeros(n_profiles, dtype=int)
    
    for i in range(n_profiles):
        # Find last valid index in each variable
        T_valid = np.where(~np.isnan(T_data[i, :]))[0]
        S_valid = np.where(~np.isnan(S_data[i, :]))[0]
        SH_valid = np.where(~np.isnan(SH_data[i, :]))[0]
        
        # Get last valid index (0-based)
        T_last = T_valid[-1] if len(T_valid) > 0 else -1
        S_last = S_valid[-1] if len(S_valid) > 0 else -1
        SH_last = SH_valid[-1] if len(SH_valid) > 0 else -1
        
        # Take minimum to be conservative
        last_valid = max(0, min(T_last, S_last, SH_last))
        lengths[i] = last_valid
    
    # Check if we have actual variable lengths (not all the same)
    unique_lengths = len(set(lengths))
    
    # Consider it variable-length if:
    # 1. We have more than one unique length, OR
    # 2. At least one profile doesn't reach the full depth
    has_variable = unique_lengths > 1 or np.any(lengths < n_depths - 1)
    
    if has_variable:
        return True, lengths
    else:
        return False, None

def load_datasets():
    """Load training, development, and test datasets"""
    
    print("Loading datasets...")
    
    # Check if files exist before loading
    for file_path, name in [(Config.TRAIN_FILE, 'Training'), (Config.DEV_FILE, 'Development'), (Config.TEST_FILE, 'Test')]:
        if not Path(file_path).exists():
            raise FileNotFoundError(f"{name} dataset file not found: {file_path}")
    
    try:
        # Load datasets (decode_times=False to preserve time encoding)
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
    """Prepare input and output arrays for a dataset with optional NaN-detected variable depths"""
    
    # Check for NaN patterns in the data to detect variable-length sequences
    T_sample = ds['TEMP'].values
    S_sample = ds['PSAL'].values
    SH_sample = ds['SH'].values
    
    has_nan_tails, detected_lengths = detect_nan_tails(T_sample, S_sample, SH_sample)
    
    if has_nan_tails:
        print(f"{dataset_type} dataset: Using NaN-tail detected variable depths")
        last_valid_idx = detected_lengths
        max_depth_idx = detected_lengths  # Use same as last_valid for NaN detection
        print(f"Detected sequence lengths range: {last_valid_idx.min()} to {last_valid_idx.max()}")
        print(f"Number of unique profile lengths: {len(set(last_valid_idx))}")
        use_variable_lengths = True
    else:
        print(f"{dataset_type} dataset: Using fixed depths (no NaN tails detected)")
        max_depth_idx = None
        use_variable_lengths = False
    
    # Climatology data
    T_glorys = ds['T_glorys'].values
    S_glorys = ds['S_glorys'].values  
    SH_glorys = ds['SH_glorys'].values
    
    # Surface data selection (satellite or GLORYS)
    if Config.SURFACE_TS == 'satellite':
        sst_surface = ds['SST'].values
        sss_surface = ds['SSS'].values
    elif Config.SURFACE_TS == 'glorys':
        sst_surface = ds['SST_glorys'].values
        sss_surface = ds['SSS_glorys'].values
    else:
        raise ValueError(f"Invalid SURFACE_TS value: {Config.SURFACE_TS}. Must be 'satellite' or 'glorys'")
    
    # Surface data (anomalies from surface climatology)
    sst_anomaly = np.repeat(
        sst_surface[:, np.newaxis], T_glorys.shape[1], axis=1
    ) - np.repeat(T_glorys[:,0][:, np.newaxis], T_glorys.shape[1], axis=1)
    
    sss_anomaly = np.repeat(
        sss_surface[:, np.newaxis], S_glorys.shape[1], axis=1  
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
    day_of_year = ds['day_of_year'].values.astype('int32')
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
        x_ease_array = np.repeat(ds['X_EASE'].values[:, np.newaxis], n_depth, axis=1)
        input_arrays.append(x_ease_array)
        input_names.append('x_ease')
        
    if Config.INPUT_VARS['y_ease']:
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
    
    # Handle variable vs fixed length sequences
    if use_variable_lengths:
        # For variable lengths, we'll store as lists initially
        X_list = []
        y_list = []
        lengths = []
        
        n_profiles = len(last_valid_idx)
        
        for i in range(n_profiles):
            # Get the effective depth for this profile (limited by both bathymetry and data quality)
            profile_length = last_valid_idx[i] + 1  # +1 because index is 0-based
            lengths.append(profile_length)
            
            # Extract profile data up to bathymetry depth
            profile_X = np.stack([arr[i, :profile_length] for arr in input_arrays], axis=1)
            profile_y = np.stack([SH_anom[i, :profile_length], T_anom[i, :profile_length], S_anom[i, :profile_length]], axis=1)
            
            X_list.append(profile_X)
            y_list.append(profile_y)
        
        return {
            'X': X_list,  # List of variable-length sequences
            'y': y_list,  # List of variable-length sequences
            'lengths': lengths,  # Sequence lengths for pack_padded_sequence
            'input_names': input_names,
            'has_bathymetry': True,
            'max_depth_idx': max_depth_idx,
            'last_valid_idx': last_valid_idx,
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
                'time_attrs': dict(ds['TIME'].attrs) if 'TIME' in ds else {},
                'depth': ds['depth'].values
            }
        }
    else:
        # Fixed length sequences (original behavior)
        X = np.stack(input_arrays, axis=2)  # [profiles, depth, variables]
        y = np.stack([SH_anom, T_anom, S_anom], axis=2)  # [profiles, depth, variables]
        
        return {
            'X': X,
            'y': y,
            'input_names': input_names,
            'has_bathymetry': False,
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
                'time_attrs': dict(ds['TIME'].attrs) if 'TIME' in ds else {},
                'depth': ds['depth'].values
            }
        }

def normalize_data(train_data, dev_data, test_data):
    """Normalize input and output data using combined train+dev statistics (excluding test to prevent data leakage)"""
    
    print("Normalizing data...")
    
    # Handle both variable-length lists and fixed arrays
    if train_data.get('has_bathymetry', False):
        # Variable-length sequences: concatenate all profiles from train+dev only
        X_all_profiles = []
        y_all_profiles = []
        
        for X_profile in train_data['X'] + dev_data['X']:  # Only train+dev for stats
            X_all_profiles.append(X_profile)
        for y_profile in train_data['y'] + dev_data['y']:  # Only train+dev for stats
            y_all_profiles.append(y_profile)
            
        # Stack all depth points from all profiles
        X_combined = np.concatenate(X_all_profiles, axis=0)  # [total_depth_points, n_features]
        y_combined = np.concatenate(y_all_profiles, axis=0)  # [total_depth_points, n_outputs]
        
        # Compute statistics
        X_min = X_combined.min(axis=0)  # [n_features]
        X_max = X_combined.max(axis=0)  # [n_features]
        y_min = y_combined.min(axis=0)  # [n_outputs]
        y_max = y_combined.max(axis=0)  # [n_outputs]
        
    else:
        # Fixed-length sequences (original behavior) - only train+dev for stats
        X_combined = np.concatenate([train_data['X'], dev_data['X']], axis=0)
        X_min = X_combined.min(axis=(0,1))
        X_max = X_combined.max(axis=(0,1))
        
        y_combined = np.concatenate([train_data['y'], dev_data['y']], axis=0)
        y_min = y_combined.min(axis=(0,1))
        y_max = y_combined.max(axis=(0,1))
    
    X_range = X_max - X_min
    X_range[X_range == 0] = 1  # Avoid division by zero
    
    y_range = y_max - y_min
    y_range[y_range == 0] = 1  # Avoid division by zero
    
    # Normalize all three datasets
    if train_data.get('has_bathymetry', False):
        # Normalize variable-length sequences
        train_data['X_norm'] = [(X - X_min) / X_range for X in train_data['X']]
        train_data['y_norm'] = [(y - y_min) / y_range for y in train_data['y']]
        
        dev_data['X_norm'] = [(X - X_min) / X_range for X in dev_data['X']]
        dev_data['y_norm'] = [(y - y_min) / y_range for y in dev_data['y']]
        
        test_data['X_norm'] = [(X - X_min) / X_range for X in test_data['X']]
        test_data['y_norm'] = [(y - y_min) / y_range for y in test_data['y']]
    else:
        # Normalize fixed-length sequences
        train_data['X_norm'] = (train_data['X'] - X_min) / X_range
        train_data['y_norm'] = (train_data['y'] - y_min) / y_range
        
        dev_data['X_norm'] = (dev_data['X'] - X_min) / X_range  
        dev_data['y_norm'] = (dev_data['y'] - y_min) / y_range
        
        test_data['X_norm'] = (test_data['X'] - X_min) / X_range
        test_data['y_norm'] = (test_data['y'] - y_min) / y_range
    
    # Store normalization parameters
    norm_params = {
        'X_min': X_min, 'X_max': X_max, 'X_range': X_range,
        'y_min': y_min, 'y_max': y_max, 'y_range': y_range
    }
    
    return train_data, dev_data, test_data, norm_params

if __name__ == "__main__":
    main()
