#!/usr/bin/env python3
"""
Sample GLORYS reanalysis data at ROMS grid locations with depth interpolation.

This script:
1. Reads GLORYS monthly data from lat/lon grid
2. Samples at ROMS grid horizontal locations using bilinear interpolation
3. Interpolates depth to WOA standard levels
4. Computes steric height from T/S profiles
5. Saves individual time slices preserving all attributes

Uses scipy RegularGridInterpolator for efficient 3D interpolation.
Memory efficient: processes one time slice at a time.
"""

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import gsw

# ============================================================================
# CONFIGURATION
# ============================================================================

# ROMS grid reference file (created by A_create_grid_reference.py)
ROMS_GRID_REF = '/home/nico/SACO/FRESH-CARE/Data_lstm_reconstruction_ROMSgrid/data_for_reconstruction/roms_grid_reference.nc'

# Input GLORYS directory
INPUT_DIR = Path('/media/nico/DATOS/Data_reanalysis/GLORYS_monthly/data')

# Output directory
OUTPUT_DIR = Path('/home/nico/SACO/FRESH-CARE/Data_lstm_reconstruction_ROMSgrid/data_for_reconstruction/glorys_roms_woaDepths')

# Year filter (set to None for all years)
YEAR_FILTER = '2012'

# Variables to process
VARS_3D = ['thetao', 'so']

# ============================================================================
# TARGET DEPTH LEVELS (WOA STANDARD)
# ============================================================================

TARGET_DEPTHS = np.array([
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95,
    100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400, 425, 450,
    475, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100,
    1150, 1200, 1250, 1300, 1350, 1400, 1450, 1500, 1550, 1600, 1650, 1700,
    1750, 1800, 1850, 1900, 1950, 2000, 2100, 2200, 2300, 2400, 2500, 2600,
    2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800,
    3900, 4000, 4100, 4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000,
    5100, 5200, 5300, 5400, 5500
], dtype=np.float64)

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# ROMS GRID FUNCTIONS
# ============================================================================


def load_roms_grid():
    """
    Load ROMS grid reference.
    
    Returns:
    --------
    tuple
        (lat_rho, lon_rho, mask_rho, eta_rho, xi_rho)
    """
    logger.info(f"Loading ROMS grid reference from {ROMS_GRID_REF}")
    ds = xr.open_dataset(ROMS_GRID_REF)
    
    lat_rho = ds['lat_rho'].values
    lon_rho = ds['lon_rho'].values
    mask_rho = ds['mask_rho'].values
    eta_rho = ds['eta_rho'].values
    xi_rho = ds['xi_rho'].values
    
    logger.info(f"  Grid shape: {lat_rho.shape}")
    
    ds.close()
    return lat_rho, lon_rho, mask_rho, eta_rho, xi_rho


# ============================================================================
# COORDINATE HANDLING
# ============================================================================


def prepare_coordinates(ds):
    """
    Prepare and validate coordinates from the dataset.
    
    Returns:
    --------
    tuple
        (lat, lon, depth, flip_lat, flip_lon, flip_depth)
    """
    lat = ds['latitude'].values.astype(np.float64)
    lon = ds['longitude'].values.astype(np.float64)
    depth = ds['depth'].values.astype(np.float64)
    
    # Check if coordinates need flipping
    flip_lat = lat[0] > lat[-1]
    flip_lon = lon[0] > lon[-1]
    flip_depth = depth[0] > depth[-1]
    
    # Sort coordinates to be increasing
    if flip_lat:
        lat = lat[::-1]
    if flip_lon:
        lon = lon[::-1]
    if flip_depth:
        depth = depth[::-1]
    
    # Handle longitude wrapping
    lon = np.where(lon > 180, lon - 360, lon)
    
    return lat, lon, depth, flip_lat, flip_lon, flip_depth


def flip_data_if_needed(data, flip_lat, flip_lon, flip_depth=False, is_3d=False):
    """
    Flip data arrays to match coordinate ordering.
    """
    data = data.copy()
    
    if is_3d:
        if flip_depth:
            data = data[::-1, :, :]
        if flip_lat:
            data = data[:, ::-1, :]
        if flip_lon:
            data = data[:, :, ::-1]
    else:
        if flip_lat:
            data = data[::-1, :]
        if flip_lon:
            data = data[:, ::-1]
    
    return data


# ============================================================================
# INTERPOLATION
# ============================================================================


def interpolate_3d_to_roms(data_3d, depth_orig, lat_orig, lon_orig,
                           depth_target, lat_target, lon_target, fill_value=np.nan):
    """
    Interpolate 3D data from (depth, lat, lon) to ROMS grid points at target depths.
    
    Parameters:
    -----------
    data_3d : np.ndarray
        3D data array (depth, lat, lon)
    depth_orig : np.ndarray
        Original depth coordinates (1D, must be increasing)
    lat_orig : np.ndarray
        Original latitude coordinates (1D, must be increasing)
    lon_orig : np.ndarray
        Original longitude coordinates (1D, must be increasing)
    depth_target : np.ndarray
        Target depth levels (1D)
    lat_target : np.ndarray
        Target latitude coordinates (2D, ROMS grid)
    lon_target : np.ndarray
        Target longitude coordinates (2D, ROMS grid)
    fill_value : float
        Value for points outside domain
    
    Returns:
    --------
    np.ndarray
        Interpolated data (depth_target, eta_rho, xi_rho)
    """
    # Handle longitude wrapping
    lon_target_work = np.where(lon_target > 180, lon_target - 360, lon_target)
    
    # Handle NaN values in data
    data_work = np.where(np.isnan(data_3d), fill_value, data_3d)
    
    # Check if we need to extrapolate to shallower depths
    min_depth_orig = depth_orig.min()
    min_depth_target = depth_target.min()
    
    if min_depth_target < min_depth_orig:
        # Extrapolate to depth 0
        d0, d1 = depth_orig[0], depth_orig[1]
        layer_0 = data_work[0, :, :]
        layer_1 = data_work[1, :, :]
        
        slope = (layer_1 - layer_0) / (d1 - d0)
        layer_at_0 = layer_0 + slope * (0 - d0)
        
        data_work = np.concatenate([layer_at_0[np.newaxis, :, :], data_work], axis=0)
        depth_orig = np.concatenate([[0], depth_orig])
    
    max_depth_orig = depth_orig.max()
    
    try:
        # Create 3D interpolator
        interpolator = RegularGridInterpolator(
            (depth_orig, lat_orig, lon_orig),
            data_work,
            method='linear',
            bounds_error=False,
            fill_value=fill_value
        )
        
        # Output shape
        n_depth = len(depth_target)
        eta_size, xi_size = lat_target.shape
        
        # Initialize output
        data_roms = np.full((n_depth, eta_size, xi_size), fill_value, dtype=np.float32)
        
        # Interpolate each depth level
        for d_idx, depth_val in enumerate(depth_target):
            if depth_val > max_depth_orig:
                continue
            
            depth_2d = np.full_like(lat_target, depth_val)
            points = np.stack([depth_2d, lat_target, lon_target_work], axis=-1)
            data_roms[d_idx, :, :] = interpolator(points)
        
    except Exception as e:
        logger.warning(f"3D interpolation failed: {e}. Returning fill values.")
        eta_size, xi_size = lat_target.shape
        data_roms = np.full((len(depth_target), eta_size, xi_size), fill_value, dtype=np.float32)
    
    return data_roms


# ============================================================================
# STERIC HEIGHT
# ============================================================================


def compute_steric_height(thetao, so, depth, lat_2d, lon_2d):
    """
    Compute steric height from potential temperature and practical salinity.
    
    Uses GSW with surface reference (0 dbar).
    
    Parameters:
    -----------
    thetao : np.ndarray
        Potential temperature (depth, eta_rho, xi_rho) in degrees C
    so : np.ndarray
        Practical salinity (depth, eta_rho, xi_rho) in PSU
    depth : np.ndarray
        Depth levels (1D) in meters
    lat_2d : np.ndarray
        Latitude grid (eta_rho, xi_rho)
    lon_2d : np.ndarray
        Longitude grid (eta_rho, xi_rho)
    
    Returns:
    --------
    np.ndarray
        Steric height (depth, eta_rho, xi_rho) in meters
    """
    n_depth, n_eta, n_xi = thetao.shape
    
    # Broadcast arrays
    depth_3d = depth[:, np.newaxis, np.newaxis] * np.ones((n_depth, n_eta, n_xi))
    lat_3d = lat_2d[np.newaxis, :, :] * np.ones((n_depth, n_eta, n_xi))
    lon_3d = lon_2d[np.newaxis, :, :] * np.ones((n_depth, n_eta, n_xi))
    
    with np.errstate(invalid='ignore'):
        p = gsw.p_from_z(-depth_3d, lat_3d)
        SA = gsw.SA_from_SP(so, p, lon_3d, lat_3d)
        CT = gsw.CT_from_pt(SA, thetao)
        
        steric_height = np.full_like(thetao, np.nan, dtype=np.float32)
        p_ref = 0
        
        for j in range(n_eta):
            for i in range(n_xi):
                sa_prof = SA[:, j, i]
                ct_prof = CT[:, j, i]
                p_prof = p[:, j, i]
                
                valid = ~np.isnan(sa_prof) & ~np.isnan(ct_prof)
                if valid.sum() < 3:
                    continue
                
                try:
                    dyn_h = gsw.geo_strf_dyn_height(sa_prof, ct_prof, p_prof, p_ref)
                    steric_height[:, j, i] = -dyn_h / 9.7963
                except Exception:
                    pass
    
    return steric_height


# ============================================================================
# PROCESSING
# ============================================================================


def process_time_slice(ds_slice, time_val, lat_orig, lon_orig, depth_orig,
                       flip_lat, flip_lon, flip_depth,
                       lat_rho, lon_rho, eta_rho, xi_rho):
    """
    Process a single time slice and return output dataset.
    """
    time_str = pd.Timestamp(time_val).strftime('%Y%m%d')
    output_file = OUTPUT_DIR / f'glorys_roms_{time_str}.nc'
    
    if output_file.exists():
        logger.info(f"  Skipping {output_file.name} (already exists)")
        return True
    
    logger.info(f"  Processing time slice: {time_str}")
    
    # Coordinates
    coords = {
        'time': ('time', [time_val], {'standard_name': 'time', 'axis': 'T'}),
        'depth': ('depth', TARGET_DEPTHS, {
            'standard_name': 'depth',
            'long_name': 'Depth',
            'units': 'm',
            'positive': 'down',
            'axis': 'Z'
        }),
        'eta_rho': ('eta_rho', eta_rho, {'long_name': 'eta index', 'units': '1'}),
        'xi_rho': ('xi_rho', xi_rho, {'long_name': 'xi index', 'units': '1'}),
    }
    
    data_vars = {}
    interpolated_data = {}
    
    # Process 3D variables
    for var_name in VARS_3D:
        if var_name not in ds_slice.data_vars:
            logger.warning(f"  Variable {var_name} not found, skipping")
            continue
        
        logger.info(f"    Interpolating 3D variable: {var_name}")
        
        da = ds_slice[var_name]
        data_3d = da.values.squeeze()
        
        # Flip if needed
        data_3d = flip_data_if_needed(data_3d, flip_lat, flip_lon, flip_depth, is_3d=True)
        
        # Interpolate
        data_roms = interpolate_3d_to_roms(
            data_3d, depth_orig, lat_orig, lon_orig,
            TARGET_DEPTHS, lat_rho, lon_rho
        )
        
        interpolated_data[var_name] = data_roms
        
        attrs = dict(da.attrs)
        attrs.pop('scale_factor', None)
        attrs.pop('add_offset', None)
        attrs.pop('_FillValue', None)
        attrs.pop('missing_value', None)
        
        data_vars[var_name] = (('time', 'depth', 'eta_rho', 'xi_rho'),
                               data_roms[np.newaxis, :, :, :].astype(np.float32),
                               attrs)
    
    # Compute steric height
    if 'thetao' in interpolated_data and 'so' in interpolated_data:
        logger.info(f"    Computing steric height...")
        
        sh = compute_steric_height(
            interpolated_data['thetao'],
            interpolated_data['so'],
            TARGET_DEPTHS,
            lat_rho,
            lon_rho
        )
        
        data_vars['SH'] = (('time', 'depth', 'eta_rho', 'xi_rho'),
                          sh[np.newaxis, :, :, :].astype(np.float32),
                          {
                              'long_name': 'steric height',
                              'units': 'm',
                              'standard_name': 'steric_change_in_sea_surface_height',
                              'description': 'Computed from so and thetao using GSW',
                              'reference_pressure': '0 dbar (sea surface)'
                          })
    
    # Create dataset
    ds_out = xr.Dataset(data_vars, coords=coords)
    
    # Add lat/lon reference
    ds_out['lat_rho'] = xr.DataArray(
        lat_rho, dims=['eta_rho', 'xi_rho'],
        attrs={'long_name': 'latitude', 'units': 'degrees_north'}
    )
    ds_out['lon_rho'] = xr.DataArray(
        lon_rho, dims=['eta_rho', 'xi_rho'],
        attrs={'long_name': 'longitude', 'units': 'degrees_east'}
    )
    
    # Attributes
    ds_out.attrs = dict(ds_slice.attrs)
    ds_out.attrs['title'] = 'GLORYS reanalysis sampled at ROMS grid points'
    ds_out.attrs['source'] = 'MERCATOR GLORYS12V1 sampled at ROMS grid'
    ds_out.attrs['grid_type'] = 'ROMS curvilinear grid'
    ds_out.attrs['depth_levels'] = 'WOA standard depth levels'
    ds_out.attrs['processing_date'] = datetime.now().isoformat()
    ds_out.attrs['conventions'] = 'CF-1.8'
    
    # Save with compression
    encoding = {}
    for var in data_vars:
        encoding[var] = {
            'dtype': 'float32',
            'zlib': True,
            'complevel': 4,
            '_FillValue': np.nan
        }
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ds_out.to_netcdf(output_file, encoding=encoding)
    logger.info(f"    Saved: {output_file.name}")
    
    return True


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("GLORYS to ROMS Grid Sampling")
    logger.info(f"Target depth levels: {len(TARGET_DEPTHS)} (WOA standard)")
    if YEAR_FILTER:
        logger.info(f"Year filter: {YEAR_FILTER}")
    logger.info(f"Input:  {INPUT_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info("=" * 60)
    
    # Load ROMS grid
    lat_rho, lon_rho, mask_rho, eta_rho, xi_rho = load_roms_grid()
    
    # Find input files
    input_files = sorted(INPUT_DIR.glob('*.nc'))
    
    if YEAR_FILTER:
        year_pattern = f'_mean_{YEAR_FILTER}'
        input_files = [f for f in input_files if year_pattern in f.name]
        logger.info(f"Filtered to year {YEAR_FILTER}: {len(input_files)} files")
    
    if len(input_files) == 0:
        logger.error("No input files found!")
        return
    
    # Get coordinate info from first file
    ds_sample = xr.open_dataset(input_files[0])
    lat_orig, lon_orig, depth_orig, flip_lat, flip_lon, flip_depth = prepare_coordinates(ds_sample)
    logger.info(f"Original grid: lat={len(lat_orig)}, lon={len(lon_orig)}, depth={len(depth_orig)}")
    ds_sample.close()
    
    # Process files
    processed = 0
    errors = 0
    
    for nc_file in input_files:
        logger.info(f"\nProcessing file: {nc_file.name}")
        
        try:
            ds = xr.open_dataset(nc_file)
            time_vals = ds['time'].values
            
            for t_idx, time_val in enumerate(time_vals):
                ds_slice = ds.isel(time=t_idx).compute()
                
                success = process_time_slice(
                    ds_slice, time_val,
                    lat_orig, lon_orig, depth_orig,
                    flip_lat, flip_lon, flip_depth,
                    lat_rho, lon_rho, eta_rho, xi_rho
                )
                
                if success:
                    processed += 1
                else:
                    errors += 1
            
            ds.close()
            
        except Exception as e:
            logger.error(f"Error processing {nc_file.name}: {e}")
            import traceback
            traceback.print_exc()
            errors += 1
    
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info(f"Processed: {processed} time slices")
    logger.info(f"Errors:    {errors}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
