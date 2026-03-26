#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Arctic LSTM Reconstruction Pipeline for ROMS Grid

This script performs Arctic-wide reconstruction of T/S/SH profiles at ROMS grid locations
using a trained LSTM model with Monte Carlo Dropout for uncertainty estimation.

Key features:
- Predictions made directly at ROMS grid locations
- EASE coordinates used as model input features (not the output grid)
- Disk streaming: writes results chunk-by-chunk to minimize RAM usage
- Output dimensions: (eta_rho, xi_rho, depth)

Pipeline steps:
1. Grid to profiles: Extract ocean pixels from ROMS grid, compute model inputs
2. MC Dropout predictions: Run predictions in memory-efficient chunks
3. Disk streaming: Write each chunk directly to output file (no RAM accumulation)
4. Final reconstruction: Add GLORYS climatology to anomalies

Memory strategy:
- Peak RAM ~200 MB regardless of grid size
- Uses memory-mapped NetCDF for incremental writes

Usage:
    python E_arctic_reconstruction.py [options]
    
    # Process single file
    python E_arctic_reconstruction.py --input_file /path/to/model_input_2012_01.nc
    
    # Process all files in directory
    python E_arctic_reconstruction.py --input_dir /path/to/model_input/
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import netCDF4 as nc4
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
DEFAULT_INPUT_DIR = '/home/nico/SACO/FRESH-CARE/Data_lstm_reconstruction_ROMSgrid/model_input/'
DEFAULT_OUTPUT_DIR = '/home/nico/SACO/FRESH-CARE/Data_lstm_reconstruction_ROMSgrid/reconstruction_outputs/'
DEFAULT_MODEL_PATH = '/home/nico/SACO/FRESH-CARE/Codes/WP2/buongiorno_to_pytorch_padding/model_LSTM_40_40_sat_znorm/model.pth'

# Processing parameters
# IMPORTANT: Lower chunk size for disk streaming (we don't accumulate, so can be smaller)
DEFAULT_CHUNK_SIZE = 2000  # Profiles per chunk
DEFAULT_N_MC_SAMPLES = 50  # Monte Carlo Dropout samples


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_timestamp_from_file(ds, time_idx=0):
    """Extract timestamp from dataset for output filenames."""
    time_val = ds['time'].values[time_idx]
    if isinstance(time_val, np.datetime64):
        dt = pd.Timestamp(time_val).to_pydatetime()
    else:
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
    Extract ocean pixels from ROMS grid and prepare model inputs.
    
    The model expects 7 input features at each depth level:
    - sst_anomaly: SST - T_glorys[surface]
    - sss_anomaly: SSS - S_glorys[surface]  
    - adt: Absolute dynamic topography
    - ease_x: EASE grid X coordinate (model input feature)
    - ease_y: EASE grid Y coordinate (model input feature)
    - seasonal_cos: cos(2π × DOY/365 + 1)
    - seasonal_sin: sin(2π × DOY/365 + 1)
    
    Note: We use EASE coordinates as INPUT FEATURES, predictions happen at ROMS locations.
    
    Args:
        ds: xarray Dataset with input data
        time_idx: Index of the timestep to process
        
    Returns:
        dict: profile data including indices for regridding
    """
    print(f"Step A: Converting ROMS grid to profiles (time_idx={time_idx})...")
    
    # Get ocean mask and find valid indices
    mask_rho = ds['mask_rho'].values  # (eta_rho, xi_rho)
    eta_idx, xi_idx = np.where(mask_rho == 1)
    n_profiles = len(eta_idx)
    print(f"  Ocean pixels: {n_profiles:,}")
    
    # Get depth info
    depth = ds['depth'].values
    n_depths = len(depth)
    print(f"  Depth levels: {n_depths}")
    
    # Extract surface data at specified timestep
    SST = ds['SST'].values[time_idx]  # (eta_rho, xi_rho)
    SSS = ds['SSS'].values[time_idx]
    ADT = ds['ADT'].values[time_idx]
    
    # Extract GLORYS surface values (depth=0)
    T_glorys_surf = ds['T_glorys'].values[time_idx, 0]  # (eta_rho, xi_rho)
    S_glorys_surf = ds['S_glorys'].values[time_idx, 0]
    SH_glorys_surf = ds['SH_glorys'].values[time_idx, 0]
    
    # Extract EASE coordinates (these are model input features)
    EASE_X = ds['EASE_X'].values  # (eta_rho, xi_rho)
    EASE_Y = ds['EASE_Y'].values
    
    # Get day of year
    DOY = int(ds['DOY'].values[time_idx])
    
    # Compute seasonal features
    seasonal_cos = np.cos(2 * np.pi * (DOY / 365) + 1)
    seasonal_sin = np.sin(2 * np.pi * (DOY / 365) + 1)
    
    print(f"  DOY: {DOY}, seasonal_cos: {seasonal_cos:.4f}, seasonal_sin: {seasonal_sin:.4f}")
    
    # Extract values at ocean pixels
    sst_vals = SST[eta_idx, xi_idx]
    sss_vals = SSS[eta_idx, xi_idx]
    adt_vals = ADT[eta_idx, xi_idx]
    t_glorys_surf_vals = T_glorys_surf[eta_idx, xi_idx]
    s_glorys_surf_vals = S_glorys_surf[eta_idx, xi_idx]
    sh_glorys_surf_vals = SH_glorys_surf[eta_idx, xi_idx]
    ease_x_vals = EASE_X[eta_idx, xi_idx]
    ease_y_vals = EASE_Y[eta_idx, xi_idx]
    
    # Compute surface anomalies (input features)
    sst_anomaly = sst_vals - t_glorys_surf_vals
    sss_anomaly = sss_vals - s_glorys_surf_vals
    adt = adt_vals - sh_glorys_surf_vals
    
    # Build input array: (n_profiles, n_depths, 7)
    X = np.zeros((n_profiles, n_depths, 7), dtype=np.float32)
    
    # Broadcast surface values to all depths
    X[:, :, 0] = sst_anomaly[:, np.newaxis]
    X[:, :, 1] = sss_anomaly[:, np.newaxis]
    X[:, :, 2] = adt[:, np.newaxis]
    X[:, :, 3] = ease_x_vals[:, np.newaxis]  # EASE X as input feature
    X[:, :, 4] = ease_y_vals[:, np.newaxis]  # EASE Y as input feature
    X[:, :, 5] = seasonal_cos
    X[:, :, 6] = seasonal_sin
    
    print(f"  Input array shape: {X.shape}")
    print(f"  Input features: [sst_anomaly, sss_anomaly, adt, ease_x, ease_y, seasonal_cos, seasonal_sin]")
    
    # Check for NaN values
    nan_profiles = np.any(np.isnan(X), axis=(1, 2))
    n_nan = np.sum(nan_profiles)
    if n_nan > 0:
        print(f"  Warning: {n_nan:,} profiles have NaN inputs")
    
    return {
        'X': X,
        'eta_idx': eta_idx,
        'xi_idx': xi_idx,
        'ease_x_vals': ease_x_vals,
        'ease_y_vals': ease_y_vals,
        'n_depths': n_depths,
        'depth': depth
    }


# ============================================================================
# DISK STREAMING OUTPUT
# ============================================================================

def create_output_file(output_path, ds_input, depth, timestamp):
    """
    Create output NetCDF file with pre-allocated arrays filled with NaN.
    
    Uses netCDF4 directly for incremental writing.
    
    Args:
        output_path: path for output file
        ds_input: input dataset (for grid dimensions)
        depth: depth coordinate array
        timestamp: datetime for time coordinate
        
    Returns:
        netCDF4.Dataset: open dataset for writing
    """
    n_eta = ds_input.sizes['eta_rho']
    n_xi = ds_input.sizes['xi_rho']
    n_depth = len(depth)
    
    # Create time value (days since 1950-01-01)
    ref_date = datetime(1950, 1, 1)
    days_since_ref = (timestamp - ref_date).days
    
    # Create file
    ncfile = nc4.Dataset(output_path, 'w', format='NETCDF4')
    
    # Dimensions
    ncfile.createDimension('time', 1)
    ncfile.createDimension('depth', n_depth)
    ncfile.createDimension('eta_rho', n_eta)
    ncfile.createDimension('xi_rho', n_xi)
    
    # Coordinates
    time_var = ncfile.createVariable('time', 'f8', ('time',))
    time_var[:] = [days_since_ref]
    time_var.standard_name = 'time'
    time_var.units = 'days since 1950-01-01T00:00:00'
    time_var.calendar = 'standard'
    
    depth_var = ncfile.createVariable('depth', 'f4', ('depth',))
    depth_var[:] = depth
    depth_var.standard_name = 'depth'
    depth_var.units = 'm'
    depth_var.positive = 'down'
    
    eta_var = ncfile.createVariable('eta_rho', 'i4', ('eta_rho',))
    eta_var[:] = ds_input['eta_rho'].values
    eta_var.long_name = 'eta index'
    
    xi_var = ncfile.createVariable('xi_rho', 'i4', ('xi_rho',))
    xi_var[:] = ds_input['xi_rho'].values
    xi_var.long_name = 'xi index'
    
    # Output variables (pre-filled with NaN)
    var_specs = [
        ('T_anom_pred', 'Predicted Temperature Anomaly', 'degrees_C'),
        ('S_anom_pred', 'Predicted Salinity Anomaly', 'PSU'),
        ('SH_anom_pred', 'Predicted Steric Height Anomaly', 'm'),
        ('T_anom_std', 'Temperature Anomaly Uncertainty', 'degrees_C'),
        ('S_anom_std', 'Salinity Anomaly Uncertainty', 'PSU'),
        ('SH_anom_std', 'Steric Height Anomaly Uncertainty', 'm'),
        ('T_recon', 'Reconstructed Temperature', 'degrees_C'),
        ('S_recon', 'Reconstructed Salinity', 'PSU'),
        ('SH_recon', 'Reconstructed Steric Height', 'm'),
    ]
    
    for var_name, long_name, units in var_specs:
        var = ncfile.createVariable(
            var_name, 'f4', ('time', 'depth', 'eta_rho', 'xi_rho'),
            zlib=True, complevel=4, fill_value=np.nan
        )
        var.long_name = long_name
        var.units = units
        # Initialize with NaN
        var[:] = np.nan
    
    # Copy reference data from input
    for ref_var in ['lat_rho', 'lon_rho', 'mask_rho', 'h', 'EASE_X', 'EASE_Y']:
        if ref_var in ds_input:
            da = ds_input[ref_var]
            if ref_var == 'mask_rho':
                v = ncfile.createVariable(ref_var, 'i1', ('eta_rho', 'xi_rho'))
            else:
                v = ncfile.createVariable(ref_var, 'f4', ('eta_rho', 'xi_rho'))
            v[:] = da.values
            for attr_name, attr_val in da.attrs.items():
                try:
                    setattr(v, attr_name, attr_val)
                except:
                    pass
    
    # Copy GLORYS for final reconstruction
    for glorys_var in ['T_glorys', 'S_glorys', 'SH_glorys']:
        if glorys_var in ds_input:
            da = ds_input[glorys_var].isel(time=0)  # Single timestep
            v = ncfile.createVariable(
                glorys_var, 'f4', ('time', 'depth', 'eta_rho', 'xi_rho'),
                zlib=True, complevel=4
            )
            v[0, :, :, :] = da.values
            for attr_name, attr_val in da.attrs.items():
                try:
                    setattr(v, attr_name, attr_val)
                except:
                    pass
    
    # Global attributes
    ncfile.title = 'Arctic Profile Reconstruction (ROMS Grid)'
    ncfile.institution = 'CNR-ISMAR'
    ncfile.source = 'LSTM MC Dropout prediction + GLORYS climatology'
    ncfile.history = f'Created {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}'
    ncfile.grid_type = 'ROMS curvilinear grid'
    ncfile.time_coverage_start = timestamp.strftime('%Y-%m-%dT%H:%M:%S')
    ncfile.n_mc_samples = DEFAULT_N_MC_SAMPLES
    ncfile.note = 'Predictions made at ROMS grid locations using EASE coords as model features'
    
    return ncfile


def write_chunk_to_file(ncfile, predictions, eta_idx, xi_idx, chunk_start, chunk_end,
                        T_glorys_chunk, S_glorys_chunk, SH_glorys_chunk):
    """
    Write a chunk of predictions directly to the output file.
    
    Args:
        ncfile: open netCDF4 Dataset
        predictions: dict with y_mean, y_std for this chunk
        eta_idx, xi_idx: full index arrays
        chunk_start, chunk_end: indices into eta_idx/xi_idx for this chunk
        T_glorys_chunk, S_glorys_chunk, SH_glorys_chunk: GLORYS values for reconstruction
    """
    # Get indices for this chunk
    eta_chunk = eta_idx[chunk_start:chunk_end]
    xi_chunk = xi_idx[chunk_start:chunk_end]
    
    # Extract predictions (chunk_size, n_depths, 3)
    y_mean = predictions['y_mean']  # [SH, T, S]
    y_std = predictions['y_std']
    
    n_profiles = y_mean.shape[0]
    n_depths = y_mean.shape[1]
    
    # Write each profile's predictions to the correct grid location
    # This is the key memory-efficient step: we write immediately, not accumulate
    for i in range(n_profiles):
        eta_i = eta_chunk[i]
        xi_i = xi_chunk[i]
        
        # Anomaly predictions
        ncfile['SH_anom_pred'][0, :, eta_i, xi_i] = y_mean[i, :, 0]
        ncfile['T_anom_pred'][0, :, eta_i, xi_i] = y_mean[i, :, 1]
        ncfile['S_anom_pred'][0, :, eta_i, xi_i] = y_mean[i, :, 2]
        
        # Uncertainties
        ncfile['SH_anom_std'][0, :, eta_i, xi_i] = y_std[i, :, 0]
        ncfile['T_anom_std'][0, :, eta_i, xi_i] = y_std[i, :, 1]
        ncfile['S_anom_std'][0, :, eta_i, xi_i] = y_std[i, :, 2]
        
        # Reconstructed = anomaly + GLORYS
        ncfile['SH_recon'][0, :, eta_i, xi_i] = y_mean[i, :, 0] + SH_glorys_chunk[i, :]
        ncfile['T_recon'][0, :, eta_i, xi_i] = y_mean[i, :, 1] + T_glorys_chunk[i, :]
        ncfile['S_recon'][0, :, eta_i, xi_i] = y_mean[i, :, 2] + S_glorys_chunk[i, :]
    
    # Sync to disk
    ncfile.sync()


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_single_timestep_streaming(ds_input, time_idx, model, norm_params, output_dir,
                                      n_mc_samples, chunk_size, device):
    """
    Process a single timestep with disk streaming.
    
    Writes results chunk-by-chunk to minimize RAM usage.
    """
    try:
        timestamp = get_timestamp_from_file(ds_input, time_idx)
    except:
        timestamp = datetime.now()
    
    print(f"\n  Timestamp: {timestamp}")
    
    # Step A: Grid to profiles
    profile_data = grid_to_profiles(ds_input, time_idx)
    
    X = profile_data['X']
    eta_idx = profile_data['eta_idx']
    xi_idx = profile_data['xi_idx']
    n_profiles = X.shape[0]
    n_depths = profile_data['n_depths']
    depth = profile_data['depth']
    
    # Extract GLORYS profiles for reconstruction (needed per chunk)
    T_glorys = ds_input['T_glorys'].values[time_idx]  # (depth, eta_rho, xi_rho)
    S_glorys = ds_input['S_glorys'].values[time_idx]
    SH_glorys = ds_input['SH_glorys'].values[time_idx]
    
    # Create output file
    output_path = Path(output_dir) / f'reconstruction_{timestamp.strftime("%Y%m%d")}.nc'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nCreating output file: {output_path}")
    ncfile = create_output_file(output_path, ds_input.isel(time=slice(time_idx, time_idx+1)), 
                                depth, timestamp)
    
    # Process in chunks with disk streaming
    n_chunks = (n_profiles + chunk_size - 1) // chunk_size
    print(f"\nStep B: MC Dropout predictions with disk streaming...")
    print(f"  Processing {n_profiles:,} profiles in {n_chunks} chunks of {chunk_size}")
    
    import torch
    
    for chunk_idx in tqdm(range(n_chunks), desc="Processing chunks"):
        chunk_start = chunk_idx * chunk_size
        chunk_end = min(chunk_start + chunk_size, n_profiles)
        
        # Get input for this chunk
        X_chunk = X[chunk_start:chunk_end]
        
        # Get GLORYS values for this chunk (for reconstruction)
        eta_chunk = eta_idx[chunk_start:chunk_end]
        xi_chunk = xi_idx[chunk_start:chunk_end]
        
        # Extract GLORYS profiles: (chunk_size, n_depths)
        T_glorys_chunk = np.array([T_glorys[:, e, x] for e, x in zip(eta_chunk, xi_chunk)])
        S_glorys_chunk = np.array([S_glorys[:, e, x] for e, x in zip(eta_chunk, xi_chunk)])
        SH_glorys_chunk = np.array([SH_glorys[:, e, x] for e, x in zip(eta_chunk, xi_chunk)])
        
        # Run MC Dropout predictions for this chunk
        y_mean, y_std, _, _ = mc_dropout_predict_chunked(
            model=model,
            X=X_chunk,
            norm_params=norm_params,
            n_mc_samples=n_mc_samples,
            chunk_size=len(X_chunk),  # Process whole chunk at once
            device=device,
            show_progress=False,
            show_mc_progress=False
        )
        
        predictions = {
            'y_mean': y_mean,
            'y_std': y_std
        }
        
        # Write chunk to disk immediately
        write_chunk_to_file(
            ncfile, predictions, eta_idx, xi_idx, chunk_start, chunk_end,
            T_glorys_chunk, S_glorys_chunk, SH_glorys_chunk
        )
        
        # Free memory
        del X_chunk, y_mean, y_std, predictions
        del T_glorys_chunk, S_glorys_chunk, SH_glorys_chunk
    
    # Close file
    ncfile.close()
    
    print(f"\n  Saved: {output_path}")
    
    # Print reconstruction statistics
    ds_out = xr.open_dataset(output_path)
    T_recon = ds_out['T_recon'].values
    S_recon = ds_out['S_recon'].values
    print(f"  Reconstructed T range: [{np.nanmin(T_recon):.2f}, {np.nanmax(T_recon):.2f}] °C")
    print(f"  Reconstructed S range: [{np.nanmin(S_recon):.2f}, {np.nanmax(S_recon):.2f}] PSU")
    ds_out.close()
    
    return output_path


def process_single_file(input_file, model, norm_params, output_dir,
                        n_mc_samples, chunk_size, device):
    """
    Process a single input file through the pipeline.
    """
    print(f"\n{'='*70}")
    print(f"Processing: {Path(input_file).name}")
    print(f"{'='*70}")
    
    ds_input = xr.open_dataset(input_file)
    
    n_timesteps = get_n_timesteps(ds_input)
    print(f"Found {n_timesteps} timestep(s) in file")
    
    recon_paths = []
    
    for time_idx in range(n_timesteps):
        if n_timesteps > 1:
            print(f"\n{'─'*50}")
            print(f"Processing timestep {time_idx + 1}/{n_timesteps}")
            print(f"{'─'*50}")
        
        recon_path = process_single_timestep_streaming(
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
    
    ds_input.close()
    
    print(f"\nCompleted processing: {Path(input_file).name}")
    return recon_paths


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Arctic LSTM Reconstruction Pipeline (ROMS Grid)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all files in default input directory
  python E_arctic_reconstruction.py
  
  # Process single file
  python E_arctic_reconstruction.py --input_file /path/to/model_input_2012_01.nc
  
  # Custom chunk size for very limited RAM
  python E_arctic_reconstruction.py --chunk_size 1000
        """
    )
    
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument('--input_file', type=str, help='Single input file to process')
    input_group.add_argument('--input_dir', type=str, default=DEFAULT_INPUT_DIR,
                            help=f'Directory with input files (default: {DEFAULT_INPUT_DIR})')
    
    parser.add_argument('--output_dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help=f'Output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--model_path', type=str, default=DEFAULT_MODEL_PATH,
                        help=f'Path to trained model (default: {DEFAULT_MODEL_PATH})')
    
    parser.add_argument('--chunk_size', type=int, default=DEFAULT_CHUNK_SIZE,
                        help=f'Profiles per chunk (default: {DEFAULT_CHUNK_SIZE})')
    parser.add_argument('--n_mc_samples', type=int, default=DEFAULT_N_MC_SAMPLES,
                        help=f'MC Dropout samples (default: {DEFAULT_N_MC_SAMPLES})')
    
    args = parser.parse_args()
    
    print("="*70)
    print("Arctic LSTM Reconstruction Pipeline (ROMS Grid)")
    print("="*70)
    print(f"Model: {args.model_path}")
    print(f"Output: {args.output_dir}")
    print(f"Chunk size: {args.chunk_size}")
    print(f"MC samples: {args.n_mc_samples}")
    print("Memory strategy: Disk streaming (low RAM usage)")
    
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
