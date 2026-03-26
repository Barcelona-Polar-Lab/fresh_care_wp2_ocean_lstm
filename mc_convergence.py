#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MC Dropout Convergence Analysis (Standalone)

Loads a small test dataset (e.g. 100 random profiles), loads a trained model,
and runs MC Dropout inference for increasing N values — all in one process.
For each N, it computes the per-profile mean std (uncertainty averaged over depths).

Produces a 2-row x 3-col plot:
  Top row:    Per-profile mean std vs N_MC for T, S, SH (one thin line per profile)
  Bottom row: |Δ std| between consecutive N values (same thin lines)

USAGE:
    python mc_convergence.py
    python mc_convergence.py --test_file path/to/file.nc --model_path path/to/model.pth
"""

import argparse
import os
import sys
import warnings
import numpy as np
import xarray as xr
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore', message='Degrees of freedom')
from pathlib import Path
from tqdm import tqdm

# ============================================================================
# MODEL DEFINITION (must match training code)
# ============================================================================

class OceanLSTM(nn.Module):
    """LSTM model for ocean profile reconstruction (mirrors lstm_pytorch_pd_mcdo.py)"""
    
    def __init__(self, input_size, output_size, lstm_units, dropout_rate=0.2):
        super(OceanLSTM, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.lstm_units = lstm_units if isinstance(lstm_units, list) else [lstm_units]
        
        self.input_dropout = nn.Dropout(dropout_rate)
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
        
        self.output_dropout = nn.Dropout(dropout_rate)
        self.output_layer = nn.Linear(self.lstm_units[-1], output_size)
    
    def forward(self, x, lengths=None):
        x = self.input_dropout(x)
        if lengths is not None:
            x = torch.nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        for lstm in self.lstm_layers:
            x, _ = lstm(x)
        if isinstance(x, torch.nn.utils.rnn.PackedSequence):
            x, _ = torch.nn.utils.rnn.pad_packed_sequence(x, batch_first=True)
        x = self.output_dropout(x)
        x = self.output_layer(x)
        return x


# ============================================================================
# DATA PREPARATION (mirrors prepare_dataset from lstm_pytorch_pd_mcdo.py)
# ============================================================================

def detect_nan_tails(T_data, S_data, SH_data):
    """Detect variable-length profiles by looking for NaN tails."""
    n_profiles = T_data.shape[0]
    n_depth = T_data.shape[1]
    detected_lengths = np.full(n_profiles, n_depth, dtype=int)
    has_nan_tails = False
    
    for i in range(n_profiles):
        for d in range(n_depth - 1, -1, -1):
            if not (np.isnan(T_data[i, d]) and np.isnan(S_data[i, d]) and np.isnan(SH_data[i, d])):
                detected_lengths[i] = d + 1
                if d + 1 < n_depth:
                    has_nan_tails = True
                break
    
    return has_nan_tails, detected_lengths


def prepare_inputs(ds, input_names, surface_ts='satellite'):
    """
    Build X input array from dataset, matching the order recorded in the checkpoint.
    Also detect variable-length profiles and build y target arrays.
    """
    # Detect variable lengths
    T_sample = ds['TEMP'].values
    S_sample = ds['PSAL'].values
    SH_sample = ds['SH'].values
    has_nan_tails, detected_lengths = detect_nan_tails(T_sample, S_sample, SH_sample)
    
    # Climatology
    T_glorys = ds['T_glorys'].values
    S_glorys = ds['S_glorys'].values
    SH_glorys = ds['SH_glorys'].values
    
    # Surface data
    if surface_ts == 'satellite':
        sst_surface = ds['SST'].values
        sss_surface = ds['SSS'].values
    else:
        sst_surface = ds['SST_glorys'].values
        sss_surface = ds['SSS_glorys'].values
    
    n_profiles = T_glorys.shape[0]
    n_depth = T_glorys.shape[1]
    
    # Precompute all possible input arrays
    sst_anomaly = np.repeat(sst_surface[:, np.newaxis], n_depth, axis=1) \
                  - np.repeat(T_glorys[:, 0][:, np.newaxis], n_depth, axis=1)
    sss_anomaly = np.repeat(sss_surface[:, np.newaxis], n_depth, axis=1) \
                  - np.repeat(S_glorys[:, 0][:, np.newaxis], n_depth, axis=1)
    
    sst_glorys_anomaly = np.repeat(ds['SST_glorys'].values[:, np.newaxis], n_depth, axis=1) \
                         - np.repeat(T_glorys[:, 0][:, np.newaxis], n_depth, axis=1)
    sss_glorys_anomaly = np.repeat(ds['SSS_glorys'].values[:, np.newaxis], n_depth, axis=1) \
                         - np.repeat(S_glorys[:, 0][:, np.newaxis], n_depth, axis=1)
    
    day_of_year = ds['day_of_year'].values.astype('int32')
    seasonal_cos = np.repeat(np.cos(2 * np.pi * day_of_year / 365)[:, np.newaxis], n_depth, axis=1)
    seasonal_sin = np.repeat(np.sin(2 * np.pi * day_of_year / 365)[:, np.newaxis], n_depth, axis=1)
    
    adt_array = np.repeat(ds['ADT'].values[:, np.newaxis], n_depth, axis=1)
    
    # Map names to arrays — support both old and new naming
    available = {
        'sst_anomaly': sst_anomaly,
        'sss_anomaly': sss_anomaly,
        'sst_glorys_anomaly': sst_glorys_anomaly,
        'sss_glorys_anomaly': sss_glorys_anomaly,
        'seasonal_cos': seasonal_cos,
        'seasonal_sin': seasonal_sin,
        'adt': adt_array,
        'adt_anomaly': adt_array,  # old name, same data
        'x_ease': np.repeat(ds['X_EASE'].values[:, np.newaxis], n_depth, axis=1),
        'y_ease': np.repeat(ds['Y_EASE'].values[:, np.newaxis], n_depth, axis=1),
        'latitude': np.repeat(ds['LATITUDE'].values[:, np.newaxis], n_depth, axis=1),
        'longitude': np.repeat(ds['LONGITUDE'].values[:, np.newaxis], n_depth, axis=1),
        'bathymetry': np.repeat(ds['bathymetry'].values[:, np.newaxis], n_depth, axis=1) if 'bathymetry' in ds else None,
    }
    
    # Build input arrays in the exact order the model expects
    input_arrays = []
    for name in input_names:
        if name not in available or available[name] is None:
            raise ValueError(f"Input variable '{name}' not found in dataset or available computations")
        input_arrays.append(available[name])
    
    # In-situ anomalies (targets)
    T_anom = ds['TEMP'].values - T_glorys
    S_anom = ds['PSAL'].values - S_glorys
    SH_anom = ds['SH'].values - SH_glorys
    
    if has_nan_tails:
        # Variable-length: return lists
        X_list, y_list, lengths = [], [], []
        for i in range(n_profiles):
            L = detected_lengths[i]
            lengths.append(L)
            X_list.append(np.stack([arr[i, :L] for arr in input_arrays], axis=1))
            y_list.append(np.stack([SH_anom[i, :L], T_anom[i, :L], S_anom[i, :L]], axis=1))
        return X_list, y_list, lengths, True
    else:
        X = np.stack(input_arrays, axis=2)
        y = np.stack([SH_anom, T_anom, S_anom], axis=2)
        return X, y, None, False


# ============================================================================
# MC DROPOUT INFERENCE
# ============================================================================

def run_mc_dropout(model, X_data, lengths, variable_lengths, norm_params, 
                   n_mc_samples, device, batch_size=64):
    """
    Run MC Dropout inference and return per-profile mean std for each output variable.
    
    Returns:
        profile_mean_std: array of shape (n_profiles, 3) — mean std over depths for [SH, T, S]
    """
    model.train()  # keep dropout active
    
    y_mean_norm = norm_params['y_mean']
    y_std_norm = norm_params['y_std']
    X_mean = norm_params['X_mean']
    X_std = norm_params['X_std']
    
    if variable_lengths:
        from torch.nn.utils.rnn import pad_sequence
        
        # Normalize inputs
        X_norm = [(X - X_mean) / X_std for X in X_data]
        n_profiles = len(X_norm)
        max_length = max(lengths)
        n_outputs = 3
        
        # Run MC samples
        mc_array = np.full((n_mc_samples, n_profiles, max_length, n_outputs), np.nan)
        
        for mc_idx in tqdm(range(n_mc_samples), desc=f"MC samples (N={n_mc_samples})", leave=False):
            preds = []
            for batch_start in range(0, n_profiles, batch_size):
                batch_end = min(batch_start + batch_size, n_profiles)
                X_batch = X_norm[batch_start:batch_end]
                lengths_batch = lengths[batch_start:batch_end]
                
                X_tensors = [torch.FloatTensor(x) for x in X_batch]
                X_padded = pad_sequence(X_tensors, batch_first=True, padding_value=0.0).to(device)
                lengths_tensor = torch.LongTensor(lengths_batch)
                
                with torch.no_grad():
                    y_pred = model(X_padded, lengths_tensor).cpu().numpy()
                
                for i, L in enumerate(lengths_batch):
                    pred_denorm = y_pred[i, :L, :] * y_std_norm + y_mean_norm
                    preds.append(pred_denorm)
            
            for prof_idx, L in enumerate(lengths):
                mc_array[mc_idx, prof_idx, :L, :] = preds[prof_idx]
        
        # Compute std across MC samples, then mean over depths per profile
        std_per_point = np.nanstd(mc_array, axis=0)  # (n_profiles, max_length, 3)
        profile_mean_std = np.nanmean(std_per_point, axis=1)  # (n_profiles, 3)
        
    else:
        # Fixed-length: normalize and run
        X_norm = (X_data - X_mean) / X_std
        X_tensor = torch.FloatTensor(X_norm).to(device)
        n_profiles = X_tensor.shape[0]
        
        all_preds = []
        for mc_idx in tqdm(range(n_mc_samples), desc=f"MC samples (N={n_mc_samples})", leave=False):
            preds = []
            with torch.no_grad():
                for batch_start in range(0, n_profiles, batch_size):
                    batch = X_tensor[batch_start:batch_start + batch_size]
                    y_pred = model(batch).cpu().numpy()
                    preds.append(y_pred)
            preds = np.concatenate(preds, axis=0)
            # Denormalize
            preds = preds * y_std_norm + y_mean_norm
            all_preds.append(preds)
        
        mc_array = np.stack(all_preds, axis=0)  # (n_mc, n_profiles, n_depth, 3)
        std_per_point = np.std(mc_array, axis=0)  # (n_profiles, n_depth, 3)
        profile_mean_std = np.mean(std_per_point, axis=1)  # (n_profiles, 3)
    
    return profile_mean_std  # (n_profiles, 3) for [SH, T, S]


# ============================================================================
# PLOTTING
# ============================================================================

def plot_convergence(mc_values, all_stds, output_path):
    """
    Plot per-profile std evolution and incremental deltas.
    
    Args:
        mc_values: list of N_MC values tested
        all_stds: dict {n_mc: array(n_profiles, 3)} with columns [SH, T, S]
        output_path: path to save figure
    """
    var_names = ['Temperature', 'Salinity', 'Steric Height']
    var_idx   = [1, 2, 0]  # T, S, SH columns in the (n_profiles, 3) array
    var_units = ['°C', 'PSU', 'm']
    var_colors = ['#d62728', '#2ca02c', '#1f77b4']
    
    n_mc_arr = np.array(mc_values)
    n_profiles = all_stds[mc_values[0]].shape[0]
    
    # Build matrix: (n_profiles, len(mc_values)) for each variable
    std_matrices = {}
    for vi, vidx in zip(var_names, var_idx):
        mat = np.column_stack([all_stds[n][: , vidx] for n in mc_values])
        std_matrices[vi] = mat  # (n_profiles, len(mc_values))
    
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    
    for col, (vname, vidx, vunit, vcol) in enumerate(zip(var_names, var_idx, var_units, var_colors)):
        mat = std_matrices[vname]  # (n_profiles, len(mc_values))
        
        # --- Top row: std evolution ---
        ax = axes[0, col]
        for p in range(n_profiles):
            ax.plot(n_mc_arr, mat[p, :], color=vcol, alpha=0.25, linewidth=0.7)
        # Bold median line
        median = np.median(mat, axis=0)
        ax.plot(n_mc_arr, median, color='black', linewidth=2, label='Median')
        
        ax.set_title(f'{vname} ({vunit})', fontsize=12, fontweight='bold')
        ax.set_xlabel('N MC Samples')
        ax.set_ylabel(f'Per-profile mean std ({vunit})')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(n_mc_arr[0], n_mc_arr[-1])
        
        # --- Bottom row: |Δ std| between consecutive N ---
        ax2 = axes[1, col]
        delta_mat = np.abs(np.diff(mat, axis=1))  # (n_profiles, len(mc_values)-1)
        delta_n = n_mc_arr[1:]
        
        for p in range(n_profiles):
            ax2.plot(delta_n, delta_mat[p, :], color=vcol, alpha=0.25, linewidth=0.7)
        median_delta = np.median(delta_mat, axis=0)
        ax2.plot(delta_n, median_delta, color='black', linewidth=2, label='Median')
        
        ax2.set_title(f'|Δ std| consecutive N', fontsize=11)
        ax2.set_xlabel('N MC Samples')
        ax2.set_ylabel(f'|Δ per-profile mean std| ({vunit})')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 0.03)
        if len(delta_n) > 1:
            ax2.set_xlim(delta_n[0], delta_n[-1])
    
    fig.suptitle('MC Dropout Convergence: Per-Profile Uncertainty (100 random test profiles)',
                 fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to: {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='MC Dropout Convergence Analysis (Standalone)')
    parser.add_argument('--test_file', type=str, 
                       default='data_for_lstm/100_random_test_profiles.nc',
                       help='Path to small test dataset')
    parser.add_argument('--model_path', type=str,
                       default='trained_models/model_LSTM_40_40_sat_znorm/model.pth',
                       help='Path to trained model checkpoint')
    parser.add_argument('--output', type=str, default='plots/mc_convergence.png',
                       help='Output plot path')
    parser.add_argument('--mc_values', nargs='+', type=int,
                       default=[5, 10, 15, 20, 25, 30, 35, 40, 50, 60, 75, 90, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 450, 500],
                       help='List of N_MC_SAMPLES values to test')
    args = parser.parse_args()
    
    # --- Device ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- Load model ---
    print(f"Loading model from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    arch = checkpoint['model_architecture']
    norm_params = checkpoint['norm_params']
    input_names = checkpoint['input_names']
    surface_ts = checkpoint.get('config', {}).get('SURFACE_TS', 'satellite')
    dropout_rate = checkpoint.get('config', {}).get('DROPOUT_RATE', 0.2)
    
    model = OceanLSTM(
        input_size=arch['input_size'],
        output_size=arch['output_size'],
        lstm_units=arch['lstm_units'],
        dropout_rate=dropout_rate
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Model: LSTM {arch['lstm_units']}, inputs={input_names}")
    
    # --- Load test data ---
    print(f"Loading test data from {args.test_file}...")
    ds = xr.open_dataset(args.test_file, decode_times=False)
    X_data, y_data, lengths, variable_lengths = prepare_inputs(ds, input_names, surface_ts)
    
    n_profiles = len(X_data) if variable_lengths else X_data.shape[0]
    print(f"Test profiles: {n_profiles}, Variable lengths: {variable_lengths}")
    
    # --- Run MC Dropout for each N ---
    mc_values = sorted(args.mc_values)
    all_stds = {}
    
    print(f"\nRunning MC Dropout for N = {mc_values}")
    print("=" * 60)
    
    for n_mc in mc_values:
        profile_mean_std = run_mc_dropout(
            model, X_data, lengths, variable_lengths, norm_params,
            n_mc_samples=n_mc, device=device
        )
        all_stds[n_mc] = profile_mean_std
        
        # Print summary for this N
        med_T = np.median(profile_mean_std[:, 1])
        med_S = np.median(profile_mean_std[:, 2])
        med_SH = np.median(profile_mean_std[:, 0])
        print(f"  N={n_mc:>4d}  |  median std  T={med_T:.5f}°C  S={med_S:.5f}PSU  SH={med_SH:.6f}m")
    
    # --- Plot ---
    print(f"\nGenerating plot...")
    plot_convergence(mc_values, all_stds, args.output)
    
    ds.close()
    print("Done!")


if __name__ == "__main__":
    main()
