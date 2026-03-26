#!/usr/bin/env python3
"""
Sample satellite surface data (SST, SSS, ADT) at ROMS grid locations.

Uses bilinear interpolation via scipy RegularGridInterpolator to sample
original lat/lon satellite data at each ROMS grid point.

Input: Satellite data on regular lat/lon grids
Output: Satellite data sampled at ROMS lat/lon points, preserving ROMS grid structure
"""

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
import logging
from datetime import datetime
from glob import glob

# ============================================================================
# CONFIGURATION
# ============================================================================

# ROMS grid reference file (created by A_create_grid_reference.py)
ROMS_GRID_REF = '/home/nico/SACO/FRESH-CARE/Data_lstm_reconstruction_ROMSgrid/data_for_reconstruction/roms_grid_reference.nc'

# Input directories: original satellite data on lat/lon grids
INPUT_DIRS = {
    'SST': '/home/nico/SACO/FRESH-CARE/Data_satellite/SST/data',
    'SSS': '/home/nico/SACO/FRESH-CARE/Data_satellite/SSS/sss_cci_v55/regridded_filled_wg',
    'ADT': '/home/nico/SACO/FRESH-CARE/Data_satellite/AVISO/regridded/regridded_0.25_north_pole_interp',
}

# Output base directory
OUTPUT_BASE = '/home/nico/SACO/FRESH-CARE/Data_lstm_reconstruction_ROMSgrid/data_for_reconstruction'

# Year filter for processing (set to None for all years)
YEAR_FILTER = '2012'

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
    Load ROMS grid reference with lat/lon coordinates.
    
    Returns:
    --------
    tuple
        (lat_rho, lon_rho, mask_rho, ds) where ds is the open dataset
    """
    logger.info(f"Loading ROMS grid reference from {ROMS_GRID_REF}")
    ds = xr.open_dataset(ROMS_GRID_REF)
    
    lat_rho = ds['lat_rho'].values
    lon_rho = ds['lon_rho'].values
    mask_rho = ds['mask_rho'].values
    
    logger.info(f"  Grid shape: {lat_rho.shape}")
    logger.info(f"  Ocean points: {np.sum(mask_rho == 1):,}")
    
    return lat_rho, lon_rho, mask_rho, ds


# ============================================================================
# COORDINATE HANDLING
# ============================================================================


def has_latlon_coords(ds):
    """
    Check if dataset has lat/lon coordinate structure.
    
    Returns:
    --------
    tuple
        (has_coords, lat_name, lon_name) or (False, None, None)
    """
    lat_names = ['lat', 'latitude', 'LAT', 'LATITUDE']
    lon_names = ['lon', 'longitude', 'LON', 'LONGITUDE']
    
    lat_name = None
    lon_name = None
    
    for name in lat_names:
        if name in ds.dims or name in ds.coords:
            lat_name = name
            break
    
    for name in lon_names:
        if name in ds.dims or name in ds.coords:
            lon_name = name
            break
    
    if lat_name is not None and lon_name is not None:
        return True, lat_name, lon_name
    
    return False, None, None


def prepare_coordinates(lat_orig, lon_orig):
    """
    Prepare coordinates for interpolation (ensure increasing order).
    
    Returns:
    --------
    tuple
        (lat_sorted, lon_sorted, flip_lat, flip_lon)
    """
    # Check coordinate ordering
    flip_lat = lat_orig[0] > lat_orig[-1]
    flip_lon = lon_orig[0] > lon_orig[-1]
    
    # Handle longitude wrapping
    lon_work = np.where(lon_orig > 180, lon_orig - 360, lon_orig)
    flip_lon = flip_lon or (lon_work[0] > lon_work[-1])
    
    # Sort to increasing
    lat_sorted = lat_orig[::-1] if flip_lat else lat_orig.copy()
    lon_sorted = lon_work[::-1] if flip_lon else lon_work.copy()
    
    return lat_sorted, lon_sorted, flip_lat, flip_lon


# ============================================================================
# INTERPOLATION
# ============================================================================


def interpolate_to_roms_points(data_2d, lat_orig, lon_orig, lat_target, lon_target,
                                flip_lat=False, flip_lon=False, fill_value=np.nan):
    """
    Interpolate 2D data from lat/lon grid to ROMS grid points using bilinear interpolation.
    
    Parameters:
    -----------
    data_2d : np.ndarray
        2D data array on lat/lon grid (lat, lon)
    lat_orig : np.ndarray
        Original latitude coordinates (1D, already sorted to increasing)
    lon_orig : np.ndarray
        Original longitude coordinates (1D, already sorted to increasing)
    lat_target : np.ndarray
        Target latitude coordinates (2D, ROMS grid)
    lon_target : np.ndarray
        Target longitude coordinates (2D, ROMS grid)
    flip_lat, flip_lon : bool
        Whether to flip data to match sorted coordinates
    fill_value : float
        Value to use for points outside the original domain
    
    Returns:
    --------
    np.ndarray
        Interpolated data on ROMS grid (eta_rho, xi_rho)
    """
    # Load data and apply flipping
    data_work = data_2d.copy()
    
    if flip_lat:
        data_work = data_work[::-1, :]
    if flip_lon:
        data_work = data_work[:, ::-1]
    
    # Handle NaN values
    data_work = np.where(np.isnan(data_work), fill_value, data_work)
    
    # Handle target longitude wrapping
    lon_target_work = np.where(lon_target > 180, lon_target - 360, lon_target)
    
    try:
        # Create interpolator on original lat/lon grid
        interpolator = RegularGridInterpolator(
            (lat_orig, lon_orig),
            data_work,
            method='linear',
            bounds_error=False,
            fill_value=fill_value
        )
        
        # Interpolate at target points
        points = np.stack([lat_target, lon_target_work], axis=-1)
        data_roms = interpolator(points)
        
    except Exception as e:
        logger.warning(f"Interpolation failed: {e}. Returning fill values.")
        data_roms = np.full(lat_target.shape, fill_value)
    
    return data_roms


def process_variable_with_time(var_data, lat_orig, lon_orig, lat_target, lon_target,
                                flip_lat, flip_lon, time_dim=None):
    """
    Process a variable, handling time dimension if present.
    
    Returns:
    --------
    np.ndarray
        Interpolated data array (with time dim if present)
    """
    data_values = var_data.values
    
    if time_dim is not None and time_dim in var_data.dims:
        # Process each time step
        n_times = data_values.shape[0]
        result = np.zeros((n_times,) + lat_target.shape, dtype=np.float32)
        
        for t in range(n_times):
            result[t] = interpolate_to_roms_points(
                data_values[t], lat_orig, lon_orig,
                lat_target, lon_target, flip_lat, flip_lon
            )
        return result
    else:
        # No time dimension
        return interpolate_to_roms_points(
            data_values, lat_orig, lon_orig,
            lat_target, lon_target, flip_lat, flip_lon
        )


# ============================================================================
# FILE PROCESSING
# ============================================================================


def process_single_file(input_path, output_path, lat_rho, lon_rho, eta_rho, xi_rho):
    """
    Process a single NetCDF file: sample at ROMS grid points.
    
    Returns:
    --------
    bool
        True if successful
    """
    logger.info(f"Processing: {input_path.name}")
    
    try:
        ds = xr.open_dataset(input_path)
        
        # Check for lat/lon coordinates
        has_coords, lat_name, lon_name = has_latlon_coords(ds)
        if not has_coords:
            logger.warning(f"  Skipping: No lat/lon coordinates found")
            ds.close()
            return False
        
        # Get original coordinates
        lat_orig = ds[lat_name].values
        lon_orig = ds[lon_name].values
        
        # Prepare coordinates (sort to increasing)
        lat_sorted, lon_sorted, flip_lat, flip_lon = prepare_coordinates(lat_orig, lon_orig)
        
        # Identify time dimension
        time_dim = None
        for name in ['time', 'TIME', 'Time']:
            if name in ds.dims:
                time_dim = name
                break
        
        # Create output coordinates
        coords = {
            'eta_rho': ('eta_rho', eta_rho, {'long_name': 'eta index', 'units': '1'}),
            'xi_rho': ('xi_rho', xi_rho, {'long_name': 'xi index', 'units': '1'})
        }
        
        # Add time coordinate if present
        if time_dim is not None:
            coords[time_dim] = ds[time_dim]
        
        data_vars = {}
        
        # Process each data variable
        for var_name in ds.data_vars:
            var = ds[var_name]
            
            # Check if variable has spatial dimensions
            has_lat = lat_name in var.dims
            has_lon = lon_name in var.dims
            
            if not (has_lat and has_lon):
                # Keep non-spatial variables as-is
                continue
            
            logger.info(f"  Sampling variable: {var_name}")
            
            # Interpolate
            data_interp = process_variable_with_time(
                var, lat_sorted, lon_sorted,
                lat_rho, lon_rho, flip_lat, flip_lon, time_dim
            )
            
            # Determine output dimensions
            if time_dim is not None and time_dim in var.dims:
                dims = (time_dim, 'eta_rho', 'xi_rho')
            else:
                dims = ('eta_rho', 'xi_rho')
            
            # Preserve attributes
            attrs = dict(var.attrs)
            attrs['sampling_method'] = 'bilinear interpolation'
            attrs['sampled_from'] = str(input_path.name)
            
            data_vars[var_name] = (dims, data_interp.astype(np.float32), attrs)
        
        # Create output dataset
        ds_out = xr.Dataset(data_vars, coords=coords)
        
        # Add lat/lon for reference
        ds_out['lat_rho'] = xr.DataArray(
            lat_rho, dims=['eta_rho', 'xi_rho'],
            attrs={'long_name': 'latitude', 'units': 'degrees_north'}
        )
        ds_out['lon_rho'] = xr.DataArray(
            lon_rho, dims=['eta_rho', 'xi_rho'],
            attrs={'long_name': 'longitude', 'units': 'degrees_east'}
        )
        
        # Copy and update global attributes
        ds_out.attrs = dict(ds.attrs)
        ds_out.attrs['title'] = ds.attrs.get('title', 'Satellite data') + ' sampled at ROMS grid points'
        ds_out.attrs['grid_type'] = 'ROMS curvilinear grid'
        ds_out.attrs['sampling_method'] = 'bilinear interpolation'
        ds_out.attrs['original_file'] = str(input_path.name)
        ds_out.attrs['processing_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Encoding with compression
        encoding = {
            var: {'zlib': True, 'complevel': 4, 'dtype': 'float32'}
            for var in ds_out.data_vars
        }
        
        ds_out.to_netcdf(output_path, encoding=encoding)
        logger.info(f"  Saved: {output_path}")
        
        ds.close()
        ds_out.close()
        
        return True
        
    except Exception as e:
        logger.error(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def process_directory(input_dir, output_dir, dataset_name, lat_rho, lon_rho, eta_rho, xi_rho):
    """
    Process all NetCDF files in a directory tree.
    
    Returns:
    --------
    tuple
        (processed_count, skipped_count, error_count)
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {dataset_name} dataset")
    logger.info(f"Input:  {input_dir}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'='*60}")
    
    processed = 0
    skipped = 0
    errors = 0
    
    # Find all NetCDF files
    nc_files = list(input_dir.rglob('*.nc'))
    
    # Filter by year if specified
    if YEAR_FILTER:
        nc_files = [f for f in nc_files if YEAR_FILTER in str(f)]
        logger.info(f"Filtered to year {YEAR_FILTER}: {len(nc_files)} files")
    else:
        logger.info(f"Found {len(nc_files)} NetCDF files")
    
    for nc_file in nc_files:
        # Compute relative path from input root
        rel_path = nc_file.relative_to(input_dir)
        output_file = output_dir / rel_path
        
        # Process the file
        success = process_single_file(nc_file, output_file, lat_rho, lon_rho, eta_rho, xi_rho)
        
        if success:
            processed += 1
        elif success is False:
            skipped += 1
        else:
            errors += 1
    
    logger.info(f"\n{dataset_name} summary: {processed} processed, {skipped} skipped, {errors} errors")
    
    return processed, skipped, errors


def main():
    """Main entry point."""
    logger.info("="*60)
    logger.info("Satellite Data Sampling at ROMS Grid Points")
    if YEAR_FILTER:
        logger.info(f"Year filter: {YEAR_FILTER}")
    logger.info("="*60)
    
    # Load ROMS grid
    lat_rho, lon_rho, mask_rho, ds_grid = load_roms_grid()
    eta_rho = ds_grid['eta_rho'].values
    xi_rho = ds_grid['xi_rho'].values
    ds_grid.close()
    
    total_processed = 0
    total_skipped = 0
    total_errors = 0
    
    for dataset_name, input_dir in INPUT_DIRS.items():
        input_path = Path(input_dir)
        
        if not input_path.exists():
            logger.warning(f"Input directory does not exist: {input_dir}")
            continue
        
        # Output directory
        output_dir = Path(OUTPUT_BASE) / f"{dataset_name}_roms"
        
        processed, skipped, errors = process_directory(
            input_path, output_dir, dataset_name,
            lat_rho, lon_rho, eta_rho, xi_rho
        )
        
        total_processed += processed
        total_skipped += skipped
        total_errors += errors
    
    logger.info("\n" + "="*60)
    logger.info("FINAL SUMMARY")
    logger.info(f"Total processed: {total_processed}")
    logger.info(f"Total skipped:   {total_skipped}")
    logger.info(f"Total errors:    {total_errors}")
    logger.info("="*60)


if __name__ == '__main__':
    main()
