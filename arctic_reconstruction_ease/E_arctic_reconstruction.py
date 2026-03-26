#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arctic LSTM Reconstruction Pipeline

This script performs Arctic-wide reconstruction of T/S/SH profiles using a trained
LSTM model with Monte Carlo Dropout for uncertainty estimation.

Pipeline steps:
1. Grid to profiles: Extract ocean pixels from gridded input, compute model inputs
2. MC Dropout predictions: Run predictions in memory-efficient chunks
3. Save anomaly profiles: Store predicted anomalies in profile format
4. Regrid to grid: Map profiles back to spatial grid
5. Final reconstruction: Add GLORYS reanalysis reference to get full profiles

Usage:
    python arctic_reconstruction.py [options]
    
    # Process single file
    python arctic_reconstruction.py --input_file /path/to/model_input_2012_01.nc
    
    # Process all files in directory
    python arctic_reconstruction.py --input_dir /path/to/model_input/

Author: Based on lstm_pytorch_pd_mcdo.py by Bruno Buongiorno Nardelli
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lstm_pytorch_utils import (
    load_model_checkpoint,
    mc_dropout_predict_chunked,
    normalize_array
)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Default paths
DEFAULT_INPUT_DIR = '/home/nico/SACO/FRESH-CARE/Data_lstm_reconstruction/model_input/'
DEFAULT_OUTPUT_DIR = '/home/nico/SACO/FRESH-CARE/Data_lstm_reconstruction/reconstruction_outputs/'
DEFAULT_MODEL_PATH = '/home/nico/SACO/FRESH-CARE/Codes/WP2/buongiorno_to_pytorch_padding/model_LSTM_40_40_sat_znorm/model.pth'

'''
The script will produce intermediate files in the following structure:
'''

# Processing parameters
DEFAULT_CHUNK_SIZE = 5000  # Profiles per chunk (tuned for <8GB RAM)
DEFAULT_N_MC_SAMPLES = 50  # Monte Carlo Dropout samples


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def estimate_memory_requirements(n_ocean_pixels, n_depths, n_mc_samples, chunk_size):
    """
    Estimate memory requirements for processing.
    
    Returns:
        dict: Memory estimates in MB for different components
    """
    bytes_per_float = 4  # float32
    
    # Input array per chunk: (chunk_size, n_depths, 7 features)
    input_chunk_mb = chunk_size * n_depths * 7 * bytes_per_float / (1024**2)
    
    # MC predictions per chunk: (n_mc_samples, chunk_size, n_depths, 3 outputs)
    mc_chunk_mb = n_mc_samples * chunk_size * n_depths * 3 * bytes_per_float / (1024**2)
    
    # Output arrays: (n_ocean_pixels, n_depths, 3) x 4 (mean, std, ci_lower, ci_upper)
    output_mb = n_ocean_pixels * n_depths * 3 * 4 * bytes_per_float / (1024**2)
    
    # Total peak memory estimate
    peak_mb = input_chunk_mb + mc_chunk_mb + output_mb
    
    return {
        'input_chunk_mb': input_chunk_mb,
        'mc_chunk_mb': mc_chunk_mb,
        'output_total_mb': output_mb,
        'peak_estimate_mb': peak_mb,
        'n_chunks': (n_ocean_pixels + chunk_size - 1) // chunk_size
    }


def get_timestamp_from_file(ds, time_idx=0):
    """Extract timestamp from dataset for output filenames."""
    time_val = ds['time'].values[time_idx]
    if isinstance(time_val, np.datetime64):
        dt = pd.Timestamp(time_val).to_pydatetime()
    else:
        # Handle numeric time (days since reference)
        import cftime
        time_units = ds['time'].attrs.get('units', 'days since 1950-01-01')
        time_calendar = ds['time'].attrs.get('calendar', 'standard')
        dt = cftime.num2date(time_val, time_units, time_calendar)
        if hasattr(dt, 'year'):
            dt = datetime(dt.year, dt.month, dt.day)
    return dt


def get_n_timesteps(ds):
    """Get the number of timesteps in a dataset."""
    return ds.sizes.get('time', ds.dims.get('time', 1))


# ============================================================================
# STEP A: GRID TO PROFILES
# ============================================================================

def grid_to_profiles(ds, time_idx=0):
    """
    Extract ocean pixels from gridded data and prepare model inputs.
    
    The model expects 7 input features at each depth level:
    - sst_anomaly: SST - T_glorys[surface]
    - sss_anomaly: SSS - S_glorys[surface]  
    - adt: Absolute dynamic topography
    - x_ease: EASE grid X coordinate
    - y_ease: EASE grid Y coordinate
    - seasonal_cos: cos(2π × DOY/365 + 1)
    - seasonal_sin: sin(2π × DOY/365 + 1)
    
    Args:
        ds: xarray Dataset with input data
        time_idx: Index of the timestep to process (default: 0)
        
    Returns:
        dict: {
            'X': array (n_profiles, n_depths, 7) - model input features
            'y_idx': array (n_profiles,) - y indices for regridding
            'x_idx': array (n_profiles,) - x indices for regridding
            'x_ease_vals': array (n_profiles,) - actual X_EASE values
            'y_ease_vals': array (n_profiles,) - actual Y_EASE values
            'n_depths': int - number of depth levels
            'depth': array - depth coordinate values
        }
    """
    print(f"Step A: Converting grid to profiles (time_idx={time_idx})...")
    
    # Get ocean mask and find valid indices
    ocean_mask = ds['ocean_mask'].values  # (y_ease, x_ease)
    y_idx, x_idx = np.where(ocean_mask == 1)
    n_profiles = len(y_idx)
    print(f"  Ocean pixels: {n_profiles}")
    
    # Get depth info
    depth = ds['depth'].values
    n_depths = len(depth)
    print(f"  Depth levels: {n_depths}")
    
    # Extract surface data at specified timestep
    SST = ds['SST'].values[time_idx]  # (y_ease, x_ease)
    SSS = ds['SSS'].values[time_idx]
    ADT = ds['ADT'].values[time_idx]
    
    # Extract GLORYS surface values (depth=0) at specified timestep
    T_glorys_surf = ds['T_glorys'].values[time_idx, 0]  # (y_ease, x_ease)
    S_glorys_surf = ds['S_glorys'].values[time_idx, 0]
    SH_glorys_surf = ds['SH_glorys'].values[time_idx, 0]
    
    # Extract coordinates
    X_EASE = ds['X_EASE'].values  # (y_ease, x_ease)
    Y_EASE = ds['Y_EASE'].values
    
    # Get day of year at specified timestep
    DOY = int(ds['DOY'].values[time_idx])
    
    # Compute seasonal features
    seasonal_cos = np.cos(2 * np.pi * (DOY / 365) + 1)
    seasonal_sin = np.sin(2 * np.pi * (DOY / 365) + 1)
    
    print(f"  DOY: {DOY}, seasonal_cos: {seasonal_cos:.4f}, seasonal_sin: {seasonal_sin:.4f}")
    
    # Extract values at ocean pixels
    sst_vals = SST[y_idx, x_idx]  # (n_profiles,)
    sss_vals = SSS[y_idx, x_idx]
    adt_vals = ADT[y_idx, x_idx]
    t_glorys_surf_vals = T_glorys_surf[y_idx, x_idx]
    s_glorys_surf_vals = S_glorys_surf[y_idx, x_idx]
    sh_glorys_surf_vals = SH_glorys_surf[y_idx, x_idx]
    x_ease_vals = X_EASE[y_idx, x_idx]
    y_ease_vals = Y_EASE[y_idx, x_idx]
    
    # Compute surface anomalies (input features)
    sst_anomaly = sst_vals - t_glorys_surf_vals  # (n_profiles,)
    sss_anomaly = sss_vals - s_glorys_surf_vals
    adt = adt_vals - sh_glorys_surf_vals
    
    # Build input array: (n_profiles, n_depths, 7)
    # Each feature is constant across depth (surface anomaly repeated)
    X = np.zeros((n_profiles, n_depths, 7), dtype=np.float32)
    
    # Broadcast surface values to all depths
    X[:, :, 0] = sst_anomaly[:, np.newaxis]  # sst_anomaly
    X[:, :, 1] = sss_anomaly[:, np.newaxis]  # sss_anomaly
    X[:, :, 2] = adt[:, np.newaxis]  # adt
    X[:, :, 3] = x_ease_vals[:, np.newaxis]  # x_ease
    X[:, :, 4] = y_ease_vals[:, np.newaxis]  # y_ease
    X[:, :, 5] = seasonal_cos  # seasonal_cos (scalar broadcast)
    X[:, :, 6] = seasonal_sin  # seasonal_sin (scalar broadcast)
    
    print(f"  Input array shape: {X.shape}")
    print(f"  Input features: [sst_anomaly, sss_anomaly, adt, x_ease, y_ease, seasonal_cos, seasonal_sin]")
    
    # Check for NaN values in inputs
    nan_profiles = np.any(np.isnan(X), axis=(1, 2))
    n_nan = np.sum(nan_profiles)
    if n_nan > 0:
        print(f"  Warning: {n_nan} profiles have NaN inputs (will produce NaN outputs)")
    
    return {
        'X': X,
        'y_idx': y_idx,
        'x_idx': x_idx,
        'x_ease_vals': x_ease_vals,
        'y_ease_vals': y_ease_vals,
        'n_depths': n_depths,
        'depth': depth
    }


# ============================================================================
# STEP B: MC DROPOUT PREDICTIONS (uses lstm_pytorch_utils)
# ============================================================================

def run_predictions(profile_data, model, norm_params, n_mc_samples, chunk_size, device):
    """
    Run MC Dropout predictions on profiles.
    
    Args:
        profile_data: dict from grid_to_profiles()
        model: loaded OceanLSTM model
        norm_params: normalization parameters from checkpoint
        n_mc_samples: number of MC samples
        chunk_size: profiles per chunk
        device: torch device
        
    Returns:
        dict: {
            'y_mean': (n_profiles, n_depths, 3) - mean predictions [SH, T, S]
            'y_std': (n_profiles, n_depths, 3) - uncertainty estimates
            'y_ci_lower': (n_profiles, n_depths, 3) - lower CI bounds
            'y_ci_upper': (n_profiles, n_depths, 3) - upper CI bounds
        }
    """
    print("\nStep B: Running MC Dropout predictions...")
    
    X = profile_data['X']
    n_profiles = X.shape[0]
    
    # Estimate memory
    mem_est = estimate_memory_requirements(
        n_profiles, profile_data['n_depths'], n_mc_samples, chunk_size
    )
    print(f"  Memory estimates:")
    print(f"    Input chunk: {mem_est['input_chunk_mb']:.1f} MB")
    print(f"    MC predictions chunk: {mem_est['mc_chunk_mb']:.1f} MB")
    print(f"    Output arrays total: {mem_est['output_total_mb']:.1f} MB")
    print(f"    Peak estimate: {mem_est['peak_estimate_mb']:.1f} MB")
    print(f"    Number of chunks: {mem_est['n_chunks']}")
    
    # Run chunked predictions
    y_mean, y_std, y_ci_lower, y_ci_upper = mc_dropout_predict_chunked(
        model=model,
        X=X,
        norm_params=norm_params,
        n_mc_samples=n_mc_samples,
        chunk_size=chunk_size,
        device=device,
        show_progress=True,
        show_mc_progress=True
    )
    
    print(f"  Output shape: {y_mean.shape}")
    print(f"  Output order: [SH_anom, T_anom, S_anom]")
    
    return {
        'y_mean': y_mean,
        'y_std': y_std,
        'y_ci_lower': y_ci_lower,
        'y_ci_upper': y_ci_upper
    }


# ============================================================================
# STEP C: SAVE ANOMALY PROFILES
# ============================================================================

def save_anomaly_profiles(profile_data, predictions, ds_input, output_dir, timestamp):
    """
    Save predicted anomaly profiles to NetCDF.
    
    Args:
        profile_data: dict from grid_to_profiles()
        predictions: dict from run_predictions()
        ds_input: original input dataset (for metadata)
        output_dir: output directory path
        timestamp: datetime for filename
    """
    print("\nStep C: Saving anomaly profiles...")
    
    # Create output subdirectory
    prof_dir = Path(output_dir) / 'predicted_anom_prof'
    prof_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    y_mean = predictions['y_mean']  # (n_profiles, n_depths, 3)
    y_std = predictions['y_std']
    y_ci_lower = predictions['y_ci_lower']
    y_ci_upper = predictions['y_ci_upper']
    
    n_profiles = y_mean.shape[0]
    depth = profile_data['depth']
    
    # Create time value (days since 1950-01-01)
    ref_date = datetime(1950, 1, 1)
    days_since_ref = (timestamp - ref_date).days
    
    # Create dataset
    ds = xr.Dataset(
        coords={
            'profile': ('profile', np.arange(n_profiles)),
            'depth': ('depth', depth, {
                'standard_name': 'depth',
                'long_name': 'Depth',
                'units': 'm',
                'positive': 'down'
            }),
            'time': ('time', [days_since_ref], {
                'standard_name': 'time',
                'long_name': 'Time',
                'units': 'days since 1950-01-01T00:00:00',
                'calendar': 'standard'
            })
        }
    )
    
    # Add predicted anomalies (mean)
    ds['SH_anom_pred'] = xr.DataArray(
        y_mean[:, :, 0],
        dims=['profile', 'depth'],
        attrs={
            'long_name': 'Predicted Steric Height Anomaly (mean)',
            'units': 'cm',
            'description': 'MC Dropout mean prediction'
        }
    )
    ds['T_anom_pred'] = xr.DataArray(
        y_mean[:, :, 1],
        dims=['profile', 'depth'],
        attrs={
            'long_name': 'Predicted Temperature Anomaly (mean)',
            'units': 'degrees_C',
            'description': 'MC Dropout mean prediction'
        }
    )
    ds['S_anom_pred'] = xr.DataArray(
        y_mean[:, :, 2],
        dims=['profile', 'depth'],
        attrs={
            'long_name': 'Predicted Salinity Anomaly (mean)',
            'units': 'PSU',
            'description': 'MC Dropout mean prediction'
        }
    )
    
    # Add uncertainty (std)
    ds['SH_anom_std'] = xr.DataArray(
        y_std[:, :, 0],
        dims=['profile', 'depth'],
        attrs={
            'long_name': 'Steric Height Anomaly Uncertainty',
            'units': 'cm',
            'description': 'MC Dropout standard deviation'
        }
    )
    ds['T_anom_std'] = xr.DataArray(
        y_std[:, :, 1],
        dims=['profile', 'depth'],
        attrs={
            'long_name': 'Temperature Anomaly Uncertainty',
            'units': 'degrees_C',
            'description': 'MC Dropout standard deviation'
        }
    )
    ds['S_anom_std'] = xr.DataArray(
        y_std[:, :, 2],
        dims=['profile', 'depth'],
        attrs={
            'long_name': 'Salinity Anomaly Uncertainty',
            'units': 'PSU',
            'description': 'MC Dropout standard deviation'
        }
    )
    
    # Add confidence intervals
    ds['SH_anom_ci_lower'] = xr.DataArray(
        y_ci_lower[:, :, 0],
        dims=['profile', 'depth'],
        attrs={'long_name': 'Steric Height Anomaly 2.5% CI', 'units': 'cm'}
    )
    ds['SH_anom_ci_upper'] = xr.DataArray(
        y_ci_upper[:, :, 0],
        dims=['profile', 'depth'],
        attrs={'long_name': 'Steric Height Anomaly 97.5% CI', 'units': 'cm'}
    )
    ds['T_anom_ci_lower'] = xr.DataArray(
        y_ci_lower[:, :, 1],
        dims=['profile', 'depth'],
        attrs={'long_name': 'Temperature Anomaly 2.5% CI', 'units': 'degrees_C'}
    )
    ds['T_anom_ci_upper'] = xr.DataArray(
        y_ci_upper[:, :, 1],
        dims=['profile', 'depth'],
        attrs={'long_name': 'Temperature Anomaly 97.5% CI', 'units': 'degrees_C'}
    )
    ds['S_anom_ci_lower'] = xr.DataArray(
        y_ci_lower[:, :, 2],
        dims=['profile', 'depth'],
        attrs={'long_name': 'Salinity Anomaly 2.5% CI', 'units': 'PSU'}
    )
    ds['S_anom_ci_upper'] = xr.DataArray(
        y_ci_upper[:, :, 2],
        dims=['profile', 'depth'],
        attrs={'long_name': 'Salinity Anomaly 97.5% CI', 'units': 'PSU'}
    )
    
    # Add profile metadata (for regridding)
    ds['y_idx'] = xr.DataArray(
        profile_data['y_idx'],
        dims=['profile'],
        attrs={'long_name': 'Y grid index for regridding'}
    )
    ds['x_idx'] = xr.DataArray(
        profile_data['x_idx'],
        dims=['profile'],
        attrs={'long_name': 'X grid index for regridding'}
    )
    ds['x_ease'] = xr.DataArray(
        profile_data['x_ease_vals'],
        dims=['profile'],
        attrs={'long_name': 'EASE-Grid X coordinate', 'units': 'm'}
    )
    ds['y_ease'] = xr.DataArray(
        profile_data['y_ease_vals'],
        dims=['profile'],
        attrs={'long_name': 'EASE-Grid Y coordinate', 'units': 'm'}
    )
    
    # Global attributes
    ds.attrs = {
        'title': 'Predicted Arctic Profile Anomalies',
        'institution': 'CNR-ISMAR',
        'source': 'LSTM MC Dropout prediction',
        'history': f'Created {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        'time_coverage_start': timestamp.strftime('%Y-%m-%dT%H:%M:%S'),
        'n_profiles': n_profiles,
        'n_mc_samples': 50,
        'confidence_level': 0.95
    }
    
    # Encoding with compression (chunksizes for netCDF4)
    n_prof = len(ds.profile)
    encoding = {
        'SH_anom_pred': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
        'T_anom_pred': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
        'S_anom_pred': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
        'SH_anom_std': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
        'T_anom_std': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
        'S_anom_std': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
    }
    
    # Save
    filename = f"anom_profiles_{timestamp.strftime('%Y%m%d')}.nc"
    output_path = prof_dir / filename
    ds.to_netcdf(output_path, encoding=encoding)
    print(f"  Saved: {output_path}")
    
    return output_path


# ============================================================================
# STEP D: REGRID TO GRID FORMAT
# ============================================================================

def regrid_profiles_to_grid(profile_data, predictions, ds_input, output_dir, timestamp):
    """
    Regrid profile predictions back to spatial grid format.
    
    Args:
        profile_data: dict from grid_to_profiles()
        predictions: dict from run_predictions()
        ds_input: original input dataset (for grid structure)
        output_dir: output directory path
        timestamp: datetime for filename
    """
    print("\nStep D: Regridding predictions to grid format...")
    
    # Create output subdirectory
    grid_dir = Path(output_dir) / 'predicted_anom_grid'
    grid_dir.mkdir(parents=True, exist_ok=True)
    
    # Get grid dimensions from input
    n_y = ds_input.sizes['y_ease']
    n_x = ds_input.sizes['x_ease']
    n_depths = profile_data['n_depths']
    depth = profile_data['depth']
    
    # Get indices
    y_idx = profile_data['y_idx']
    x_idx = profile_data['x_idx']
    
    # Extract predictions
    y_mean = predictions['y_mean']  # (n_profiles, n_depths, 3)
    y_std = predictions['y_std']
    y_ci_lower = predictions['y_ci_lower']
    y_ci_upper = predictions['y_ci_upper']
    
    # Initialize grid arrays with NaN
    SH_anom_grid = np.full((1, n_depths, n_y, n_x), np.nan, dtype=np.float32)
    T_anom_grid = np.full((1, n_depths, n_y, n_x), np.nan, dtype=np.float32)
    S_anom_grid = np.full((1, n_depths, n_y, n_x), np.nan, dtype=np.float32)
    SH_std_grid = np.full((1, n_depths, n_y, n_x), np.nan, dtype=np.float32)
    T_std_grid = np.full((1, n_depths, n_y, n_x), np.nan, dtype=np.float32)
    S_std_grid = np.full((1, n_depths, n_y, n_x), np.nan, dtype=np.float32)
    SH_ci_lower_grid = np.full((1, n_depths, n_y, n_x), np.nan, dtype=np.float32)
    SH_ci_upper_grid = np.full((1, n_depths, n_y, n_x), np.nan, dtype=np.float32)
    T_ci_lower_grid = np.full((1, n_depths, n_y, n_x), np.nan, dtype=np.float32)
    T_ci_upper_grid = np.full((1, n_depths, n_y, n_x), np.nan, dtype=np.float32)
    S_ci_lower_grid = np.full((1, n_depths, n_y, n_x), np.nan, dtype=np.float32)
    S_ci_upper_grid = np.full((1, n_depths, n_y, n_x), np.nan, dtype=np.float32)
    
    # Fill grid with profile values
    print(f"  Grid shape: (1, {n_depths}, {n_y}, {n_x})")
    print(f"  Filling {len(y_idx)} ocean pixels (vectorized)...")
    
    # Use numpy advanced indexing for fast regridding (vectorized)
    # y_mean has shape (n_profiles, n_depths, 3) -> need to transpose for grid assignment
    # Target: grid[0, :, y_idx, x_idx] = values  (broadcast across all depths)
    
    # For each variable, reshape predictions from (n_profiles, n_depths) to match grid indexing
    # Grid indexing: grid[0, depth_idx, y_idx, x_idx] where we loop implicitly over depths
    
    for d in range(n_depths):
        SH_anom_grid[0, d, y_idx, x_idx] = y_mean[:, d, 0]
        T_anom_grid[0, d, y_idx, x_idx] = y_mean[:, d, 1]
        S_anom_grid[0, d, y_idx, x_idx] = y_mean[:, d, 2]
        SH_std_grid[0, d, y_idx, x_idx] = y_std[:, d, 0]
        T_std_grid[0, d, y_idx, x_idx] = y_std[:, d, 1]
        S_std_grid[0, d, y_idx, x_idx] = y_std[:, d, 2]
        SH_ci_lower_grid[0, d, y_idx, x_idx] = y_ci_lower[:, d, 0]
        SH_ci_upper_grid[0, d, y_idx, x_idx] = y_ci_upper[:, d, 0]
        T_ci_lower_grid[0, d, y_idx, x_idx] = y_ci_lower[:, d, 1]
        T_ci_upper_grid[0, d, y_idx, x_idx] = y_ci_upper[:, d, 1]
        S_ci_lower_grid[0, d, y_idx, x_idx] = y_ci_lower[:, d, 2]
        S_ci_upper_grid[0, d, y_idx, x_idx] = y_ci_upper[:, d, 2]
    
    print(f"  Regridding complete.")
    
    # Create time coordinate
    ref_date = datetime(1950, 1, 1)
    days_since_ref = (timestamp - ref_date).days
    
    # Create dataset
    ds = xr.Dataset(
        coords={
            'time': ('time', [days_since_ref], {
                'standard_name': 'time',
                'long_name': 'Time',
                'units': 'days since 1950-01-01T00:00:00',
                'calendar': 'standard'
            }),
            'depth': ('depth', depth, {
                'standard_name': 'depth',
                'long_name': 'Depth',
                'units': 'm',
                'positive': 'down'
            }),
            'y_ease': ('y_ease', ds_input['y_ease'].values),
            'x_ease': ('x_ease', ds_input['x_ease'].values)
        }
    )
    
    # Add gridded anomalies
    ds['SH_anom_pred'] = xr.DataArray(
        SH_anom_grid, dims=['time', 'depth', 'y_ease', 'x_ease'],
        attrs={'long_name': 'Predicted Steric Height Anomaly', 'units': 'cm'}
    )
    ds['T_anom_pred'] = xr.DataArray(
        T_anom_grid, dims=['time', 'depth', 'y_ease', 'x_ease'],
        attrs={'long_name': 'Predicted Temperature Anomaly', 'units': 'degrees_C'}
    )
    ds['S_anom_pred'] = xr.DataArray(
        S_anom_grid, dims=['time', 'depth', 'y_ease', 'x_ease'],
        attrs={'long_name': 'Predicted Salinity Anomaly', 'units': 'PSU'}
    )
    
    # Add uncertainty grids
    ds['SH_anom_std'] = xr.DataArray(
        SH_std_grid, dims=['time', 'depth', 'y_ease', 'x_ease'],
        attrs={'long_name': 'Steric Height Anomaly Uncertainty', 'units': 'cm'}
    )
    ds['T_anom_std'] = xr.DataArray(
        T_std_grid, dims=['time', 'depth', 'y_ease', 'x_ease'],
        attrs={'long_name': 'Temperature Anomaly Uncertainty', 'units': 'degrees_C'}
    )
    ds['S_anom_std'] = xr.DataArray(
        S_std_grid, dims=['time', 'depth', 'y_ease', 'x_ease'],
        attrs={'long_name': 'Salinity Anomaly Uncertainty', 'units': 'PSU'}
    )
    
    # Add CI grids
    ds['SH_anom_ci_lower'] = xr.DataArray(
        SH_ci_lower_grid, dims=['time', 'depth', 'y_ease', 'x_ease'],
        attrs={'long_name': 'Steric Height Anomaly 2.5% CI', 'units': 'cm'}
    )
    ds['SH_anom_ci_upper'] = xr.DataArray(
        SH_ci_upper_grid, dims=['time', 'depth', 'y_ease', 'x_ease'],
        attrs={'long_name': 'Steric Height Anomaly 97.5% CI', 'units': 'cm'}
    )
    ds['T_anom_ci_lower'] = xr.DataArray(
        T_ci_lower_grid, dims=['time', 'depth', 'y_ease', 'x_ease'],
        attrs={'long_name': 'Temperature Anomaly 2.5% CI', 'units': 'degrees_C'}
    )
    ds['T_anom_ci_upper'] = xr.DataArray(
        T_ci_upper_grid, dims=['time', 'depth', 'y_ease', 'x_ease'],
        attrs={'long_name': 'Temperature Anomaly 97.5% CI', 'units': 'degrees_C'}
    )
    ds['S_anom_ci_lower'] = xr.DataArray(
        S_ci_lower_grid, dims=['time', 'depth', 'y_ease', 'x_ease'],
        attrs={'long_name': 'Salinity Anomaly 2.5% CI', 'units': 'PSU'}
    )
    ds['S_anom_ci_upper'] = xr.DataArray(
        S_ci_upper_grid, dims=['time', 'depth', 'y_ease', 'x_ease'],
        attrs={'long_name': 'Salinity Anomaly 97.5% CI', 'units': 'PSU'}
    )
    
    # Global attributes
    ds.attrs = {
        'title': 'Predicted Arctic Profile Anomalies (gridded)',
        'institution': 'ICM-CSIC',
        'source': 'LSTM MC Dropout prediction',
        'history': f'Created {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        'time_coverage_start': timestamp.strftime('%Y-%m-%dT%H:%M:%S'),
        'n_mc_samples': 50,
        'confidence_level': 0.95
    }
    
    # Encoding with chunking for sparse data
    encoding = {}
    for var in ds.data_vars:
        encoding[var] = {
            'dtype': 'float32',
            'zlib': True,
            'complevel': 4,
            'chunksizes': (1, 17, 50, 50)  # Good for sparse 3D data
        }
    
    # Save
    filename = f"anom_grid_{timestamp.strftime('%Y%m%d')}.nc"
    output_path = grid_dir / filename
    ds.to_netcdf(output_path, encoding=encoding)
    print(f"  Saved: {output_path}")
    
    return output_path


# ============================================================================
# STEP E: FINAL RECONSTRUCTION
# ============================================================================

def create_reconstruction(ds_input, anom_grid_path, output_dir, timestamp):
    """
    Create final reconstruction by adding GLORYS reanalysis reference to predicted anomalies.
    
    Args:
        ds_input: original input dataset (for GLORYS fields)
        anom_grid_path: path to anomaly grid file from Step D
        output_dir: output directory path
        timestamp: datetime for filename
    """
    print("\nStep E: Creating final reconstruction...")
    
    # Create output subdirectory
    recon_dir = Path(output_dir) / 'reconstruction_data'
    recon_dir.mkdir(parents=True, exist_ok=True)
    
    # Load anomaly grid
    ds_anom = xr.open_dataset(anom_grid_path)
    
    # Get GLORYS reanalysis reference from input
    T_glorys = ds_input['T_glorys'].values  # (1, depth, y_ease, x_ease)
    S_glorys = ds_input['S_glorys'].values
    SH_glorys = ds_input['SH_glorys'].values
    
    # Get predicted anomalies
    T_anom = ds_anom['T_anom_pred'].values  # (1, depth, y_ease, x_ease)
    S_anom = ds_anom['S_anom_pred'].values
    SH_anom = ds_anom['SH_anom_pred'].values
    
    # Apply NaN mask: where GLORYS is NaN (i.e. below the local seabed), all
    # predicted quantities are physically meaningless and should be masked out.
    glorys_nan_mask = np.isnan(T_glorys) | np.isnan(S_glorys) | np.isnan(SH_glorys)
    T_anom = np.where(glorys_nan_mask, np.nan, T_anom)
    S_anom = np.where(glorys_nan_mask, np.nan, S_anom)
    SH_anom = np.where(glorys_nan_mask, np.nan, SH_anom)
    
    # Apply the same mask to standard deviations and confidence intervals
    T_std = np.where(glorys_nan_mask, np.nan, ds_anom['T_anom_std'].values)
    S_std = np.where(glorys_nan_mask, np.nan, ds_anom['S_anom_std'].values)
    SH_std = np.where(glorys_nan_mask, np.nan, ds_anom['SH_anom_std'].values)
    T_ci_lower = np.where(glorys_nan_mask, np.nan, ds_anom['T_anom_ci_lower'].values)
    T_ci_upper = np.where(glorys_nan_mask, np.nan, ds_anom['T_anom_ci_upper'].values)
    S_ci_lower = np.where(glorys_nan_mask, np.nan, ds_anom['S_anom_ci_lower'].values)
    S_ci_upper = np.where(glorys_nan_mask, np.nan, ds_anom['S_anom_ci_upper'].values)
    SH_ci_lower = np.where(glorys_nan_mask, np.nan, ds_anom['SH_anom_ci_lower'].values)
    SH_ci_upper = np.where(glorys_nan_mask, np.nan, ds_anom['SH_anom_ci_upper'].values)
    
    # Reconstruct full profiles: anomaly + reanalysis reference
    T_recon = T_anom + T_glorys
    S_recon = S_anom + S_glorys
    SH_recon = SH_anom + SH_glorys
    
    print(f"  Reconstructed T range: [{np.nanmin(T_recon):.2f}, {np.nanmax(T_recon):.2f}] °C")
    print(f"  Reconstructed S range: [{np.nanmin(S_recon):.2f}, {np.nanmax(S_recon):.2f}] PSU")
    print(f"  Reconstructed SH range: [{np.nanmin(SH_recon):.2f}, {np.nanmax(SH_recon):.2f}] cm")
    
    # Start with a copy of input data structure
    ds = ds_input.copy(deep=True)
    
    # Write masked arrays back into ds_anom so attrs are preserved when added to ds
    ds_anom['T_anom_pred'].values = T_anom
    ds_anom['S_anom_pred'].values = S_anom
    ds_anom['SH_anom_pred'].values = SH_anom
    ds_anom['T_anom_std'].values = T_std
    ds_anom['S_anom_std'].values = S_std
    ds_anom['SH_anom_std'].values = SH_std
    ds_anom['T_anom_ci_lower'].values = T_ci_lower
    ds_anom['T_anom_ci_upper'].values = T_ci_upper
    ds_anom['S_anom_ci_lower'].values = S_ci_lower
    ds_anom['S_anom_ci_upper'].values = S_ci_upper
    ds_anom['SH_anom_ci_lower'].values = SH_ci_lower
    ds_anom['SH_anom_ci_upper'].values = SH_ci_upper
    
    # Add predicted anomalies (masked)
    ds['T_anom_pred'] = ds_anom['T_anom_pred']
    ds['S_anom_pred'] = ds_anom['S_anom_pred']
    ds['SH_anom_pred'] = ds_anom['SH_anom_pred']
    
    # Add uncertainty fields (masked)
    ds['T_anom_std'] = ds_anom['T_anom_std']
    ds['S_anom_std'] = ds_anom['S_anom_std']
    ds['SH_anom_std'] = ds_anom['SH_anom_std']
    
    # Add CI fields (masked)
    ds['T_anom_ci_lower'] = ds_anom['T_anom_ci_lower']
    ds['T_anom_ci_upper'] = ds_anom['T_anom_ci_upper']
    ds['S_anom_ci_lower'] = ds_anom['S_anom_ci_lower']
    ds['S_anom_ci_upper'] = ds_anom['S_anom_ci_upper']
    ds['SH_anom_ci_lower'] = ds_anom['SH_anom_ci_lower']
    ds['SH_anom_ci_upper'] = ds_anom['SH_anom_ci_upper']
    
    # Add reconstructed profiles
    ds['T_recon'] = xr.DataArray(
        T_recon, dims=['time', 'depth', 'y_ease', 'x_ease'],
        attrs={
            'long_name': 'Reconstructed Temperature',
            'units': 'degrees_C',
            'description': 'T_anom_pred + T_glorys'
        }
    )
    ds['S_recon'] = xr.DataArray(
        S_recon, dims=['time', 'depth', 'y_ease', 'x_ease'],
        attrs={
            'long_name': 'Reconstructed Salinity',
            'units': 'PSU',
            'description': 'S_anom_pred + S_glorys'
        }
    )
    ds['SH_recon'] = xr.DataArray(
        SH_recon, dims=['time', 'depth', 'y_ease', 'x_ease'],
        attrs={
            'long_name': 'Reconstructed Steric Height',
            'units': 'cm',
            'description': 'SH_anom_pred + SH_glorys'
        }
    )
    
    # Update global attributes
    ds.attrs['title'] = 'Arctic Profile Reconstruction'
    ds.attrs['history'] = f'Reconstructed {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    ds.attrs['source'] = 'LSTM MC Dropout prediction + GLORYS reanalysis reference'
    
    # Encoding
    encoding = {}
    for var in ds.data_vars:
        if var == 'ease_grid_mapping':
            continue
        encoding[var] = {
            'dtype': 'float32',
            'zlib': True,
            'complevel': 4
        }
        # Add chunking for 3D/4D variables
        if 'depth' in ds[var].dims:
            encoding[var]['chunksizes'] = (1, 17, 50, 50)
    
    # Save
    filename = f"reconstruction_{timestamp.strftime('%Y%m%d')}.nc"
    output_path = recon_dir / filename
    ds.to_netcdf(output_path, encoding=encoding)
    print(f"  Saved: {output_path}")
    
    # Clean up
    ds_anom.close()
    
    return output_path


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_single_timestep(ds_input, time_idx, model, norm_params, output_dir, 
                            n_mc_samples, chunk_size, device):
    """
    Process a single timestep through the full pipeline.
    
    Args:
        ds_input: xarray Dataset with input data
        time_idx: Index of the timestep to process
        model: loaded OceanLSTM model
        norm_params: normalization parameters
        output_dir: output directory
        n_mc_samples: number of MC samples
        chunk_size: profiles per chunk
        device: torch device
    
    Returns:
        path to reconstruction file
    """
    # Get timestamp for this timestep
    try:
        timestamp = get_timestamp_from_file(ds_input, time_idx)
    except Exception as e:
        # Fallback: use current date
        timestamp = datetime.now()
    
    print(f"\n  Timestamp: {timestamp}")
    
    # Step A: Grid to profiles
    profile_data = grid_to_profiles(ds_input, time_idx)
    
    # Step B: MC Dropout predictions
    predictions = run_predictions(
        profile_data, model, norm_params, 
        n_mc_samples, chunk_size, device
    )
    
    # Step C: Save anomaly profiles
    prof_path = save_anomaly_profiles(
        profile_data, predictions, ds_input, output_dir, timestamp
    )
    
    # Step D: Regrid to grid format
    grid_path = regrid_profiles_to_grid(
        profile_data, predictions, ds_input, output_dir, timestamp
    )
    
    # Step E: Final reconstruction
    # Create a sliced dataset with only this timestep for reconstruction
    ds_slice = ds_input.isel(time=slice(time_idx, time_idx+1))
    recon_path = create_reconstruction(
        ds_slice, grid_path, output_dir, timestamp
    )
    
    return recon_path


def process_single_file(input_file, model, norm_params, output_dir, 
                        n_mc_samples, chunk_size, device):
    """
    Process a single input file through the full pipeline.
    Handles files with multiple timesteps.
    
    Args:
        input_file: path to input NetCDF file
        model: loaded OceanLSTM model
        norm_params: normalization parameters
        output_dir: output directory
        n_mc_samples: number of MC samples
        chunk_size: profiles per chunk
        device: torch device
    """
    print(f"\n{'='*70}")
    print(f"Processing: {Path(input_file).name}")
    print(f"{'='*70}")
    
    # Open input file
    ds_input = xr.open_dataset(input_file)
    
    # Get number of timesteps
    n_timesteps = get_n_timesteps(ds_input)
    print(f"Found {n_timesteps} timestep(s) in file")
    
    recon_paths = []
    
    # Process each timestep
    for time_idx in range(n_timesteps):
        if n_timesteps > 1:
            print(f"\n{'─'*50}")
            print(f"Processing timestep {time_idx + 1}/{n_timesteps}")
            print(f"{'─'*50}")
        
        recon_path = process_single_timestep(
            ds_input=ds_input,
            time_idx=time_idx,
            model=model,
            norm_params=norm_params,
            output_dir=output_dir,
            n_mc_samples=n_mc_samples,
            chunk_size=chunk_size,
            device=device
        )
        recon_paths.append(recon_path)
    
    # Clean up
    ds_input.close()
    
    print(f"\nCompleted processing: {Path(input_file).name}")
    return recon_paths


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Arctic LSTM Reconstruction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in default input directory
  python arctic_reconstruction.py
  
  # Process single file
  python arctic_reconstruction.py --input_file /path/to/model_input_2012_01.nc
  
  # Process all files in specific directory
  python arctic_reconstruction.py --input_dir /path/to/model_input/
  
  # Custom chunk size for limited RAM
  python arctic_reconstruction.py --chunk_size 3000
        """
    )
    
    # Input options (mutually exclusive, defaults to DEFAULT_INPUT_DIR if neither specified)
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('--input_file', type=str, help='Single input file to process')
    input_group.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR,
                            help=f'Directory with input files to process (default: {DEFAULT_INPUT_DIR})')
    
    # Output and model options
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help=f'Path to trained model (default: {DEFAULT_MODEL_PATH})')
    
    # Processing options
    parser.add_argument('--chunk_size', type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f'Profiles per chunk (default: {DEFAULT_CHUNK_SIZE})')
    parser.add_argument('--n_mc_samples', type=int, default=DEFAULT_N_MC_SAMPLES,
                        help=f'MC Dropout samples (default: {DEFAULT_N_MC_SAMPLES})')
    
    args = parser.parse_args()
    
    # Print configuration
    print("="*70)
    print("Arctic LSTM Reconstruction Pipeline")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"MC samples: {args.n_mc_samples}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    model, norm_params, model_config, input_names = load_model_checkpoint(
        args.model_path, device
    )
    
    # Get input files
    if args.input_file:
        input_files = [args.input_file]
    else:
        input_dir = Path(args.input_dir)
        input_files = sorted(input_dir.glob('model_input_*.nc'))
        print(f"\nFound {len(input_files)} input files in {input_dir}")
    
    # Process each file
    for input_file in input_files:
        process_single_file(
            input_file=input_file,
            model=model,
            norm_params=norm_params,
            output_dir=args.output_dir,
            n_mc_samples=args.n_mc_samples,
            chunk_size=args.chunk_size,
            device=device
        )
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
