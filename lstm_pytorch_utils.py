#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared utilities for LSTM Ocean Profile Reconstruction

This module contains shared utilities used by both:
- lstm_pytorch_pd_mcdo.py (training and testing)
- arctic_reconstruction/arctic_reconstruction.py (Arctic-wide reconstruction)

Includes:
- OceanLSTM model class
- Normalization/denormalization functions
- Model loading utilities
- MC Dropout prediction functions
"""

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from tqdm import tqdm


# ============================================================================
# NEURAL NETWORK MODEL
# ============================================================================

class OceanLSTM(nn.Module):
    """
    LSTM model for ocean profile reconstruction
    Flexible architecture that can be easily modified
    """
    
    def __init__(self, input_size, output_size, lstm_units, dropout_rate=0.2):
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
        super(OceanLSTM, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.lstm_units = lstm_units if isinstance(lstm_units, list) else [lstm_units]

        # Input dropout
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # LSTM layers
        self.lstm_layers = nn.ModuleList()  # modules list allows tracking params
                                            # for multiple layers in a clean way
        layer_input_size = input_size
        
        for i, units in enumerate(self.lstm_units):
            self.lstm_layers.append(
                nn.LSTM(
                    input_size=layer_input_size,  # input size of the layer
                    hidden_size=units,  # num of LSTM units: output size of the layer
                    batch_first=True,
                    dropout=dropout_rate if i < len(self.lstm_units) - 1 else 0
                    # dropout only between layers, not on the last layer's output
                    # we set that below on the output layer
                )
            )
            layer_input_size = units
        
        # Output dropout
        self.output_dropout = nn.Dropout(dropout_rate)

        # Output layer (applied to each time/depth step)
        self.output_layer = nn.Linear(
            self.lstm_units[-1],  # last LSTM layer's output size
            output_size  # number of output features (e.g., 3 for SH, T, S)
        )

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
            # Repack for LSTM processing (lengths must be on CPU)
            x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        else:
            # Regular tensor input, apply dropout directly
            x = self.input_dropout(x)

            if lengths is not None:  # In case of fixed-length sequences
                                     # lengths can be None and we skip packing
                
                # Pack the sequence for variable lengths (lengths must be on CPU)
                x = torch.nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
            
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

        # Output dropout and projection
        x = self.output_dropout(x)
        x = self.output_layer(x)
        
        return x


# ============================================================================
# NORMALIZATION FUNCTIONS
# ============================================================================

def normalize_data(data, X_mean, X_std, y_mean, y_std):
    """
    Apply z-score normalization to a dataset.
    
    Args:
        data: Dictionary containing 'X' and 'y' arrays or lists
        X_mean, X_std: Mean and std for input features
        y_mean, y_std: Mean and std for output features
    
    Returns:
        Modified data dictionary with 'X_norm' and 'y_norm' added
    """
    if data.get('variable_lengths', False):
        # Variable-length sequences
        data['X_norm'] = [(X - X_mean) / X_std for X in data['X']]
        data['y_norm'] = [(y - y_mean) / y_std for y in data['y']]
    else:
        # Fixed-length sequences
        data['X_norm'] = (data['X'] - X_mean) / X_std
        data['y_norm'] = (data['y'] - y_mean) / y_std
    
    return data


def normalize_array(X, X_mean, X_std):
    """
    Apply z-score normalization to a single array.
    
    Args:
        X: Input array of shape (n_samples, n_depths, n_features) or (n_depths, n_features)
        X_mean, X_std: Mean and std for each feature
    
    Returns:
        Normalized array
    """
    return (X - X_mean) / X_std


def denormalize_data(data_norm, mean, std, variable_lengths=False):
    """
    Reverse z-score normalization.
    
    Args:
        data_norm: Normalized data (array or list of arrays)
        mean, std: Normalization parameters
        variable_lengths: Whether data is list of variable-length sequences
    
    Returns:
        Denormalized data in original scale
    """
    if variable_lengths:
        # Variable-length sequences (list of arrays)
        return [pred * std + mean for pred in data_norm]
    else:
        # Fixed-length sequences (single array)
        return data_norm * std + mean


def denormalize_array(y_norm, y_mean, y_std):
    """
    Reverse z-score normalization for a single array.
    
    Args:
        y_norm: Normalized output array
        y_mean, y_std: Mean and std for each output feature
    
    Returns:
        Denormalized array in original scale
    """
    return y_norm * y_std + y_mean


# ============================================================================
# MODEL LOADING
# ============================================================================

def load_model_checkpoint(model_path, device=None):
    """
    Load a trained LSTM model from checkpoint.
    
    Args:
        model_path: Path to the model.pth file
        device: torch device (if None, auto-detect GPU/CPU)
    
    Returns:
        tuple: (model, norm_params, model_config, input_names)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model_path = Path(model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    print(f"Using device: {device}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    model_config = checkpoint['model_architecture']
    norm_params = checkpoint['norm_params']
    input_names = checkpoint.get('input_names', None)
    config = checkpoint.get('config', {})
    
    # Recreate model
    model = OceanLSTM(
        input_size=model_config['input_size'],
        output_size=model_config['output_size'],
        lstm_units=model_config['lstm_units'],
        dropout_rate=config.get('DROPOUT_RATE', 0.2)
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"Model architecture: {model_config['lstm_units']} LSTM units")
    print(f"Input features: {model_config['input_size']}")
    print(f"Output features: {model_config['output_size']}")
    
    return model, norm_params, model_config, input_names


# ============================================================================
# MC DROPOUT PREDICTION
# ============================================================================

def mc_dropout_predict_batch(model, X_batch, n_mc_samples=50, device=None, 
                              show_mc_progress=False, mc_progress_desc="MC samples"):
    """
    Make predictions with Monte Carlo Dropout for uncertainty estimation.
    
    This function runs multiple forward passes with dropout enabled to estimate
    prediction uncertainty. Designed for batch processing.
    
    Args:
        model: OceanLSTM model (will be set to train mode for dropout)
        X_batch: Input tensor of shape (batch_size, n_depths, n_features)
                 Already normalized!
        n_mc_samples: Number of MC forward passes (default: 50)
        device: torch device
        show_mc_progress: Whether to show progress bar for MC samples
        mc_progress_desc: Description for MC progress bar
    
    Returns:
        tuple: (y_mean, y_std, y_ci_lower, y_ci_upper)
            All of shape (batch_size, n_depths, n_outputs)
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Ensure model is in training mode for dropout
    model.train()
    
    # Convert to tensor if needed
    if not isinstance(X_batch, torch.Tensor):
        X_batch = torch.FloatTensor(X_batch)
    X_batch = X_batch.to(device)
    
    # Collect MC samples
    mc_predictions = []
    
    mc_iter = range(n_mc_samples)
    if show_mc_progress:
        mc_iter = tqdm(mc_iter, desc=mc_progress_desc, leave=False)
    
    with torch.no_grad():
        for _ in mc_iter:
            y_pred = model(X_batch).cpu().numpy()
            mc_predictions.append(y_pred)
    
    # Stack: (n_mc_samples, batch_size, n_depths, n_outputs)
    mc_array = np.stack(mc_predictions, axis=0)
    
    # Compute statistics across MC samples (axis=0)
    y_mean = np.mean(mc_array, axis=0)
    y_std = np.std(mc_array, axis=0)
    
    # Confidence intervals (95% by default)
    alpha = 0.05 / 2  # 2.5% and 97.5%
    y_ci_lower = np.percentile(mc_array, alpha * 100, axis=0)
    y_ci_upper = np.percentile(mc_array, (1 - alpha) * 100, axis=0)
    
    return y_mean, y_std, y_ci_lower, y_ci_upper


def mc_dropout_predict_chunked(model, X, norm_params, n_mc_samples=50, 
                                chunk_size=5000, device=None, 
                                show_progress=True, show_mc_progress=True):
    """
    Make MC Dropout predictions on large datasets using chunked processing.
    
    Handles normalization and denormalization internally.
    
    Args:
        model: OceanLSTM model
        X: Input array of shape (n_profiles, n_depths, n_features) - NOT normalized
        norm_params: Dictionary with 'X_mean', 'X_std', 'y_mean', 'y_std'
        n_mc_samples: Number of MC forward passes
        chunk_size: Number of profiles per chunk
        device: torch device
        show_progress: Whether to show chunk progress bar
        show_mc_progress: Whether to show MC sample progress bar per chunk
    
    Returns:
        tuple: (y_mean, y_std, y_ci_lower, y_ci_upper)
            All of shape (n_profiles, n_depths, n_outputs), denormalized
    """
    if device is None:
        device = next(model.parameters()).device
    
    n_profiles = X.shape[0]
    n_depths = X.shape[1]
    n_outputs = model.output_size
    
    # Pre-allocate output arrays
    y_mean_all = np.zeros((n_profiles, n_depths, n_outputs), dtype=np.float32)
    y_std_all = np.zeros((n_profiles, n_depths, n_outputs), dtype=np.float32)
    y_ci_lower_all = np.zeros((n_profiles, n_depths, n_outputs), dtype=np.float32)
    y_ci_upper_all = np.zeros((n_profiles, n_depths, n_outputs), dtype=np.float32)
    
    # Normalize input
    X_norm = normalize_array(X, norm_params['X_mean'], norm_params['X_std'])
    
    # Process in chunks
    n_chunks = (n_profiles + chunk_size - 1) // chunk_size
    
    chunk_iter = range(0, n_profiles, chunk_size)
    if show_progress:
        chunk_iter = tqdm(chunk_iter, desc="Processing chunks", total=n_chunks)
    
    for start_idx in chunk_iter:
        end_idx = min(start_idx + chunk_size, n_profiles)
        X_chunk = X_norm[start_idx:end_idx]
        chunk_num = start_idx // chunk_size + 1
        
        # Run MC Dropout prediction
        y_mean, y_std, y_ci_lower, y_ci_upper = mc_dropout_predict_batch(
            model, X_chunk, n_mc_samples=n_mc_samples, device=device,
            show_mc_progress=show_mc_progress,
            mc_progress_desc=f"MC samples (chunk {chunk_num}/{n_chunks})"
        )
        
        # Denormalize outputs
        y_mean = denormalize_array(y_mean, norm_params['y_mean'], norm_params['y_std'])
        y_std = y_std * norm_params['y_std']  # Std scales with y_std only
        y_ci_lower = denormalize_array(y_ci_lower, norm_params['y_mean'], norm_params['y_std'])
        y_ci_upper = denormalize_array(y_ci_upper, norm_params['y_mean'], norm_params['y_std'])
        
        # Store results
        y_mean_all[start_idx:end_idx] = y_mean
        y_std_all[start_idx:end_idx] = y_std
        y_ci_lower_all[start_idx:end_idx] = y_ci_lower
        y_ci_upper_all[start_idx:end_idx] = y_ci_upper
    
    return y_mean_all, y_std_all, y_ci_lower_all, y_ci_upper_all
