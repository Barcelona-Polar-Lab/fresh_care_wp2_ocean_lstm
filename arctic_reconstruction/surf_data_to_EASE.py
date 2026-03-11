#!/usr/bin/env python3
"""
Regrid satellite surface data (SST, SSS, ADT) from lat/lon to EASE grid.
Uses bilinear interpolation via scipy RegularGridInterpolator.

Input data must have lat/lon (or latitude/longitude) coordinates.
Output preserves all attributes, time coordinates, and directory structure.
"""

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
import pyproj
import logging
from datetime import datetime

# ============================================================================
# EASE GRID CONFIGURATION PARAMETERS
# ============================================================================

# Grid Resolution and Size
GRID_RESOLUTION_M = 25000  # meters (25 km)
GRID_SIZE_X = 350          # number of grid cells in X direction
GRID_SIZE_Y = 350          # number of grid cells in Y direction

# EASE Grid Projection Parameters (Arctic Lambert Azimuthal Equal Area)
EASE_LAT_0 = 90            # latitude of projection origin (North Pole)
EASE_LON_0 = 0             # longitude of projection origin
EASE_FALSE_EASTING = 0     # false easting
EASE_FALSE_NORTHING = 0    # false northing

# Derived PROJ4 string for EASE grid
EASE_PROJ4 = f"+proj=laea +lat_0={EASE_LAT_0} +lon_0={EASE_LON_0} +x_0={EASE_FALSE_EASTING} +y_0={EASE_FALSE_NORTHING} +datum=WGS84 +units=m"

# ============================================================================
# INPUT/OUTPUT CONFIGURATION
# ============================================================================

# Set USE_TEST_DIRS to True for testing, False for production
USE_TEST_DIRS = False

# --- TEST DIRECTORIES ---
TEST_INPUT_DIRS = {
    'SST': '/home/nico/SACO/FRESH-CARE/Codes/WP2/buongiorno_to_pytorch_padding/arctic_reconstruction/test_in/SST',
    'SSS': '/home/nico/SACO/FRESH-CARE/Codes/WP2/buongiorno_to_pytorch_padding/arctic_reconstruction/test_in/SSS',
    'ADT': '/home/nico/SACO/FRESH-CARE/Codes/WP2/buongiorno_to_pytorch_padding/arctic_reconstruction/test_in/ADT',
}
TEST_OUTPUT_BASE = '/home/nico/SACO/FRESH-CARE/Codes/WP2/buongiorno_to_pytorch_padding/arctic_reconstruction/test_out'

# --- PRODUCTION DIRECTORIES ---
# Input directories: where the tree organizing nc files starts
# Output: will be INPUT_DIR_ease (e.g., SST -> SST_ease in the same parent directory)
PROD_INPUT_DIRS = {
    'SST': '/home/nico/Desktop/AUX_DIR_FRESH_CARE/satellite/SST/data',
    'SSS': '/home/nico/Desktop/AUX_DIR_FRESH_CARE/satellite/SSS_cci_v55/regridded_filled_wg',
    'ADT': '/home/nico/Desktop/AUX_DIR_FRESH_CARE/satellite/ADT/aviso_regridded_0.25_north_pole_interp',
}
# None means output goes to sibling directory with _ease suffix
PROD_OUTPUT_BASE = None

# Select active configuration
if USE_TEST_DIRS:
    INPUT_DIRS = TEST_INPUT_DIRS
    OUTPUT_BASE = TEST_OUTPUT_BASE
else:
    INPUT_DIRS = PROD_INPUT_DIRS
    OUTPUT_BASE = PROD_OUTPUT_BASE

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# ============================================================================
# EASE GRID FUNCTIONS
# ============================================================================


def create_ease_grid_structure():
    """
    Create EASE grid coordinates and metadata from configuration parameters.
    
    Returns:
    --------
    tuple
        (x_ease, y_ease, grid_mapping_attrs) where:
        - x_ease: 1D array of X coordinates (meters)
        - y_ease: 1D array of Y coordinates (meters)
        - grid_mapping_attrs: dict of CF-compliant grid mapping attributes
    """
    # Calculate grid extent (centered on projection origin)
    x_min = -(GRID_SIZE_X * GRID_RESOLUTION_M) / 2
    y_min = -(GRID_SIZE_Y * GRID_RESOLUTION_M) / 2
    
    # Create coordinate arrays (cell centers)
    x_ease = np.arange(GRID_SIZE_X) * GRID_RESOLUTION_M + x_min + GRID_RESOLUTION_M / 2
    y_ease = np.arange(GRID_SIZE_Y) * GRID_RESOLUTION_M + y_min + GRID_RESOLUTION_M / 2
    
    # Create CF-compliant grid mapping attributes
    grid_mapping_attrs = {
        'grid_mapping_name': 'lambert_azimuthal_equal_area',
        'longitude_of_projection_origin': EASE_LON_0,
        'latitude_of_projection_origin': EASE_LAT_0,
        'false_easting': EASE_FALSE_EASTING,
        'false_northing': EASE_FALSE_NORTHING,
        'grid_resolution_meters': GRID_RESOLUTION_M,
        'spatial_ref': EASE_PROJ4,
        'proj4_string': EASE_PROJ4
    }
    
    return x_ease, y_ease, grid_mapping_attrs


def latlon_to_ease(lat, lon, transformer):
    """
    Convert lat/lon coordinates to EASE grid coordinates.
    
    Parameters:
    -----------
    lat : array
        Latitude values
    lon : array
        Longitude values
    transformer : pyproj.Transformer
        Coordinate transformer from WGS84 to EASE
    
    Returns:
    --------
    tuple
        (x_ease, y_ease) coordinate arrays
    """
    # Create meshgrid if 1D arrays provided
    if lat.ndim == 1 and lon.ndim == 1:
        lon_2d, lat_2d = np.meshgrid(lon, lat)
    else:
        lon_2d, lat_2d = lon, lat
    
    # Transform coordinates
    x_ease, y_ease = transformer.transform(lat_2d, lon_2d)
    
    return x_ease, y_ease


def has_latlon_coords(ds):
    """
    Check if dataset has lat/lon coordinate structure.
    
    Parameters:
    -----------
    ds : xr.Dataset
        Dataset to check
    
    Returns:
    --------
    tuple
        (has_coords, lat_name, lon_name) or (False, None, None)
    """
    # Check for common lat/lon naming conventions
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


def interpolate_to_ease(data_2d, lat_orig, lon_orig, x_ease_target, y_ease_target, 
                        transformer, fill_value=np.nan):
    """
    Interpolate 2D data from lat/lon grid to EASE grid using bilinear interpolation.
    
    Parameters:
    -----------
    data_2d : np.ndarray
        2D data array on lat/lon grid (lat, lon)
    lat_orig : np.ndarray
        Original latitude coordinates (1D)
    lon_orig : np.ndarray
        Original longitude coordinates (1D)
    x_ease_target : np.ndarray
        Target EASE X coordinates (1D)
    y_ease_target : np.ndarray
        Target EASE Y coordinates (1D)
    transformer : pyproj.Transformer
        Coordinate transformer from WGS84 to EASE
    fill_value : float
        Value to use for points outside the original domain
    
    Returns:
    --------
    np.ndarray
        Interpolated data on EASE grid (y_ease, x_ease)
    """
    # Get EASE coordinates for original lat/lon grid
    x_ease_orig, y_ease_orig = latlon_to_ease(lat_orig, lon_orig, transformer)
    
    # For RegularGridInterpolator, we need 1D coordinates
    # Since the original grid is regular in lat/lon, we extract the EASE coords
    # along the center row and column
    mid_lat_idx = len(lat_orig) // 2
    mid_lon_idx = len(lon_orig) // 2
    
    # The original data is on a regular lat/lon grid
    # We'll interpolate in the original lat/lon space, then map to EASE
    # This is more accurate for polar regions
    
    # Create target lat/lon from EASE coordinates
    transformer_inverse = pyproj.Transformer.from_crs(
        pyproj.CRS.from_proj4(EASE_PROJ4),
        pyproj.CRS.from_epsg(4326),
        always_xy=True
    )
    
    # Create meshgrid of target EASE coordinates
    x_ease_2d, y_ease_2d = np.meshgrid(x_ease_target, y_ease_target)
    
    # Convert target EASE to lat/lon
    lon_target, lat_target = transformer_inverse.transform(x_ease_2d, y_ease_2d)
    
    # Handle NaN values in the original data
    data_filled = np.where(np.isnan(data_2d), fill_value, data_2d)
    
    # Create interpolator on original lat/lon grid
    # Note: RegularGridInterpolator expects (y, x) order for coordinates
    # and data shape should be (len(y), len(x))
    try:
        interpolator = RegularGridInterpolator(
            (lat_orig, lon_orig),
            data_filled,
            method='linear',
            bounds_error=False,
            fill_value=fill_value
        )
        
        # Interpolate at target points
        # Stack lat/lon for interpolator input
        points = np.stack([lat_target, lon_target], axis=-1)
        data_ease = interpolator(points)
        
    except Exception as e:
        logger.warning(f"Interpolation failed: {e}. Returning fill values.")
        data_ease = np.full((len(y_ease_target), len(x_ease_target)), fill_value)
    
    return data_ease


def process_variable(var_data, lat_orig, lon_orig, x_ease_target, y_ease_target,
                    transformer, time_dim=None, flip_lat=False, flip_lon=False):
    """
    Process a single variable, handling time dimension if present.
    
    Parameters:
    -----------
    var_data : xr.DataArray
        Variable data to process
    lat_orig : np.ndarray
        Original latitude coordinates (already sorted to increasing)
    lon_orig : np.ndarray
        Original longitude coordinates (already sorted to increasing)
    x_ease_target : np.ndarray
        Target EASE X coordinates
    y_ease_target : np.ndarray
        Target EASE Y coordinates
    transformer : pyproj.Transformer
        Coordinate transformer
    time_dim : str or None
        Name of time dimension if present
    flip_lat : bool
        Whether to flip data along latitude axis
    flip_lon : bool
        Whether to flip data along longitude axis
    
    Returns:
    --------
    np.ndarray
        Interpolated data array
    """
    # Load all data into memory first (avoids lazy loading issues with flipping)
    data_values = var_data.values
    
    # Apply flipping on numpy array
    if flip_lat:
        if data_values.ndim == 2:
            data_values = data_values[::-1, :]
        elif data_values.ndim == 3:
            data_values = data_values[:, ::-1, :]
    
    if flip_lon:
        if data_values.ndim == 2:
            data_values = data_values[:, ::-1]
        elif data_values.ndim == 3:
            data_values = data_values[:, :, ::-1]
    
    if time_dim is not None and time_dim in var_data.dims:
        # Process each time step
        n_times = data_values.shape[0]
        result = np.zeros((n_times, len(y_ease_target), len(x_ease_target)))
        
        for t in range(n_times):
            data_2d = data_values[t]
            result[t] = interpolate_to_ease(
                data_2d, lat_orig, lon_orig, 
                x_ease_target, y_ease_target, transformer
            )
        return result
    else:
        # No time dimension - 2D data
        return interpolate_to_ease(
            data_values, lat_orig, lon_orig,
            x_ease_target, y_ease_target, transformer
        )


def regrid_file_to_ease(input_path, output_path):
    """
    Regrid a single NetCDF file from lat/lon to EASE grid.
    
    Parameters:
    -----------
    input_path : Path
        Path to input NetCDF file
    output_path : Path
        Path to output NetCDF file
    
    Returns:
    --------
    bool
        True if successful, False otherwise
    """
    logger.info(f"Processing: {input_path.name}")
    
    try:
        # Load input dataset
        ds = xr.open_dataset(input_path)
        
        # Check for lat/lon coordinates
        has_coords, lat_name, lon_name = has_latlon_coords(ds)
        if not has_coords:
            logger.warning(f"  Skipping: No lat/lon coordinates found in {input_path.name}")
            ds.close()
            return False
        
        # Get original coordinates
        lat_orig = ds[lat_name].values
        lon_orig = ds[lon_name].values
        
        # Check coordinate ordering (RegularGridInterpolator requires increasing order)
        lat_increasing = lat_orig[0] < lat_orig[-1]
        
        # Handle longitude wrapping (ensure continuous)
        # Convert to -180 to 180 if needed
        lon_orig = np.where(lon_orig > 180, lon_orig - 360, lon_orig)
        lon_increasing = lon_orig[0] < lon_orig[-1]
        
        # Set flip flags based on original coordinate ordering
        flip_lat = not lat_increasing
        flip_lon = not lon_increasing
        
        # Sort coordinates to be increasing (required for RegularGridInterpolator)
        if not lat_increasing:
            lat_orig = lat_orig[::-1]
        if not lon_increasing:
            lon_orig = lon_orig[::-1]
        
        # Create EASE grid
        x_ease, y_ease, grid_mapping_attrs = create_ease_grid_structure()
        
        # Create coordinate transformer
        transformer = pyproj.Transformer.from_crs(
            pyproj.CRS.from_epsg(4326),
            pyproj.CRS.from_proj4(EASE_PROJ4),
            always_xy=True
        )
        
        # Identify time dimension
        time_dim = None
        for name in ['time', 'TIME', 'Time']:
            if name in ds.dims:
                time_dim = name
                break
        
        # Create output dataset
        coords = {
            'x_ease': ('x_ease', x_ease, {
                'standard_name': 'projection_x_coordinate',
                'units': f'{GRID_RESOLUTION_M} m',
                'long_name': 'EASE-Grid X coordinate',
                'axis': 'X'
            }),
            'y_ease': ('y_ease', y_ease, {
                'standard_name': 'projection_y_coordinate',
                'units': f'{GRID_RESOLUTION_M} m',
                'long_name': 'EASE-Grid Y coordinate',
                'axis': 'Y'
            })
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
                # Keep non-spatial variables as-is (e.g., scalar grid mapping)
                data_vars[var_name] = var
                continue
            
            logger.info(f"  Interpolating variable: {var_name}")
            
            # Process the variable (flip flags handle coordinate reordering)
            data_interp = process_variable(
                var, lat_orig, lon_orig,
                x_ease, y_ease, transformer, time_dim,
                flip_lat=flip_lat, flip_lon=flip_lon
            )
            
            # Determine output dimensions
            if time_dim is not None and time_dim in var.dims:
                dims = (time_dim, 'y_ease', 'x_ease')
            else:
                dims = ('y_ease', 'x_ease')
            
            # Preserve attributes
            attrs = dict(var.attrs)
            attrs['grid_mapping'] = 'ease_grid_mapping'
            
            data_vars[var_name] = (dims, data_interp, attrs)
        
        # Create output dataset
        ds_out = xr.Dataset(data_vars, coords=coords)
        
        # Add grid mapping variable
        ds_out['ease_grid_mapping'] = xr.DataArray(
            data=0,
            attrs=grid_mapping_attrs
        )
        
        # Copy and update global attributes
        ds_out.attrs = dict(ds.attrs)
        ds_out.attrs['title'] = ds.attrs.get('title', 'Surface data') + ' on EASE Grid'
        ds_out.attrs['grid_resolution'] = f'{GRID_RESOLUTION_M/1000:.1f} km'
        ds_out.attrs['grid_resolution_meters'] = GRID_RESOLUTION_M
        ds_out.attrs['grid_size'] = f'{GRID_SIZE_X} x {GRID_SIZE_Y}'
        ds_out.attrs['projection'] = 'Lambert Azimuthal Equal Area (Arctic)'
        ds_out.attrs['projection_latitude_of_origin'] = EASE_LAT_0
        ds_out.attrs['projection_longitude_of_origin'] = EASE_LON_0
        ds_out.attrs['projection_false_easting'] = EASE_FALSE_EASTING
        ds_out.attrs['projection_false_northing'] = EASE_FALSE_NORTHING
        ds_out.attrs['proj4_string'] = EASE_PROJ4
        ds_out.attrs['regridding_method'] = 'bilinear interpolation'
        ds_out.attrs['regridding_date'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ds_out.attrs['original_file'] = str(input_path.name)
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create encoding with zlib compression for all data variables
        encoding = {
            var: {'zlib': True, 'complevel': 4}
            for var in ds_out.data_vars
        }
        
        # Save output with compression
        ds_out.to_netcdf(output_path, encoding=encoding)
        logger.info(f"  Saved: {output_path}")
        
        ds.close()
        ds_out.close()
        
        return True
        
    except Exception as e:
        logger.error(f"  Error processing {input_path.name}: {e}")
        return False


def process_directory(input_dir, output_dir, dataset_name):
    """
    Process all NetCDF files in a directory tree.
    
    Parameters:
    -----------
    input_dir : Path
        Root input directory
    output_dir : Path
        Root output directory
    dataset_name : str
        Name of the dataset (for logging)
    
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
    logger.info(f"Found {len(nc_files)} NetCDF files")
    
    for nc_file in nc_files:
        # Compute relative path from input root
        rel_path = nc_file.relative_to(input_dir)
        output_file = output_dir / rel_path
        
        # Process the file
        success = regrid_file_to_ease(nc_file, output_file)
        
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
    logger.info("Surface Data to EASE Grid Regridding")
    logger.info(f"EASE Grid: {GRID_SIZE_X}x{GRID_SIZE_Y} at {GRID_RESOLUTION_M/1000:.1f} km")
    logger.info(f"Projection: {EASE_PROJ4}")
    logger.info(f"Mode: {'TEST' if USE_TEST_DIRS else 'PRODUCTION'}")
    logger.info("="*60)
    
    total_processed = 0
    total_skipped = 0
    total_errors = 0
    
    for dataset_name, input_dir in INPUT_DIRS.items():
        input_path = Path(input_dir)
        
        if not input_path.exists():
            logger.warning(f"Input directory does not exist: {input_dir}")
            continue
        
        # Determine output directory
        if OUTPUT_BASE is None:
            # Create sibling directory with _ease suffix
            # e.g., /path/to/SST -> /path/to/SST_ease
            output_dir = input_path.parent / f"{input_path.name}_ease"
        else:
            # Use OUTPUT_BASE with dataset name
            output_dir = Path(OUTPUT_BASE) / f"{dataset_name}_ease"
        
        processed, skipped, errors = process_directory(
            input_path, output_dir, dataset_name
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
