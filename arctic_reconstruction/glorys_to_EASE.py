#!/usr/bin/env python3
"""
Regrid GLORYS reanalysis data from lat/lon to EASE grid with depth interpolation.

This script:
1. Reads GLORYS monthly data from lat/lon grid
2. Transforms horizontal coordinates to EASE grid (25km)
3. Interpolates depth to WOA standard levels
4. Saves individual time slices preserving all attributes

Uses scipy RegularGridInterpolator for efficient 3D interpolation.
Memory efficient: processes one time slice at a time.
"""

import numpy as np
import xarray as xr
import pyproj
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
import logging
from datetime import datetime
import pandas as pd
import gsw

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
EASE_PROJ4 = (f"+proj=laea +lat_0={EASE_LAT_0} +lon_0={EASE_LON_0} "
              f"+x_0={EASE_FALSE_EASTING} +y_0={EASE_FALSE_NORTHING} "
              f"+datum=WGS84 +units=m")

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
# INPUT/OUTPUT CONFIGURATION
# ============================================================================

# Set to True for testing with limited files, False for production
TEST_MODE = False
# For production, filter by year (e.g., '2012') or None for all years
PRODUCTION_YEAR_FILTER = '2012'

INPUT_DIR = Path('/media/nico/DATOS/Data_reanalysis/GLORYS_monthly/data')

# Test vs production output directories
TEST_OUTPUT_DIR = Path('/home/nico/SACO/FRESH-CARE/Codes/WP2/buongiorno_to_pytorch_padding/arctic_reconstruction/test_out/glorys_ease')
PROD_OUTPUT_DIR = Path('/home/nico/Desktop/AUX_DIR_FRESH_CARE/glorys_aux')

OUTPUT_DIR = TEST_OUTPUT_DIR if TEST_MODE else PROD_OUTPUT_DIR

# Number of files to process in test mode
TEST_MAX_FILES = 1

# Variables to process (3D only: time, depth, latitude, longitude)
# thetao: potential temperature, so: practical salinity
VARS_3D = ['thetao', 'so']

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


def create_transformers():
    """
    Create coordinate transformers between WGS84 and EASE grid.
    
    Returns:
    --------
    tuple
        (transformer_to_ease, transformer_from_ease)
    """
    crs_wgs84 = pyproj.CRS.from_epsg(4326)
    crs_ease = pyproj.CRS.from_proj4(EASE_PROJ4)
    
    transformer_to_ease = pyproj.Transformer.from_crs(
        crs_wgs84, crs_ease, always_xy=True
    )
    transformer_from_ease = pyproj.Transformer.from_crs(
        crs_ease, crs_wgs84, always_xy=True
    )
    
    return transformer_to_ease, transformer_from_ease


def compute_ease_coords_for_original_grid(lat, lon, transformer_to_ease):
    """
    Compute EASE grid coordinates for each point in the original lat/lon grid.
    
    Parameters:
    -----------
    lat : np.ndarray
        1D array of latitude values
    lon : np.ndarray
        1D array of longitude values
    transformer_to_ease : pyproj.Transformer
        Transformer from WGS84 to EASE
    
    Returns:
    --------
    tuple
        (x_ease_orig, y_ease_orig) 2D arrays of EASE coordinates
    """
    # Create 2D meshgrid
    lon_2d, lat_2d = np.meshgrid(lon, lat)
    
    # Transform to EASE coordinates
    x_ease_orig, y_ease_orig = transformer_to_ease.transform(lon_2d, lat_2d)
    
    return x_ease_orig, y_ease_orig


# ============================================================================
# INTERPOLATION FUNCTIONS
# ============================================================================


def interpolate_2d_to_ease(data_2d, lat_orig, lon_orig, x_ease_target, y_ease_target,
                           transformer_from_ease, fill_value=np.nan):
    """
    Interpolate 2D data from lat/lon grid to EASE grid.
    
    Uses the approach of converting target EASE coordinates back to lat/lon
    and interpolating on the original regular lat/lon grid.
    
    Parameters:
    -----------
    data_2d : np.ndarray
        2D data array (lat, lon)
    lat_orig : np.ndarray
        Original latitude coordinates (1D, must be increasing)
    lon_orig : np.ndarray
        Original longitude coordinates (1D, must be increasing)
    x_ease_target : np.ndarray
        Target EASE X coordinates (1D)
    y_ease_target : np.ndarray
        Target EASE Y coordinates (1D)
    transformer_from_ease : pyproj.Transformer
        Transformer from EASE to WGS84
    fill_value : float
        Value for points outside domain
    
    Returns:
    --------
    np.ndarray
        Interpolated data on EASE grid (y_ease, x_ease)
    """
    # Create meshgrid of target EASE coordinates
    x_ease_2d, y_ease_2d = np.meshgrid(x_ease_target, y_ease_target)
    
    # Convert target EASE to lat/lon
    lon_target, lat_target = transformer_from_ease.transform(x_ease_2d, y_ease_2d)
    
    # Handle data with NaN values
    data_work = np.where(np.isnan(data_2d), fill_value, data_2d)
    
    # Create interpolator on original lat/lon grid
    try:
        interpolator = RegularGridInterpolator(
            (lat_orig, lon_orig),
            data_work,
            method='linear',
            bounds_error=False,
            fill_value=fill_value
        )
        
        # Interpolate at target points
        points = np.stack([lat_target, lon_target], axis=-1)
        data_ease = interpolator(points)
        
    except Exception as e:
        logger.warning(f"2D interpolation failed: {e}. Returning fill values.")
        data_ease = np.full((len(y_ease_target), len(x_ease_target)), fill_value)
    
    return data_ease


def interpolate_3d_to_ease(data_3d, depth_orig, lat_orig, lon_orig,
                           depth_target, x_ease_target, y_ease_target,
                           transformer_from_ease, fill_value=np.nan):
    """
    Interpolate 3D data from (depth, lat, lon) to (depth_target, y_ease, x_ease).
    
    Performs full 3D interpolation using RegularGridInterpolator.
    First extrapolates input data to depth 0 on the original grid (if needed),
    then performs 3D interpolation including the extrapolated layer.
    
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
    x_ease_target : np.ndarray
        Target EASE X coordinates (1D)
    y_ease_target : np.ndarray
        Target EASE Y coordinates (1D)
    transformer_from_ease : pyproj.Transformer
        Transformer from EASE to WGS84
    fill_value : float
        Value for points outside domain
    
    Returns:
    --------
    np.ndarray
        Interpolated data (depth_target, y_ease, x_ease)
    """
    # Create meshgrid of target coordinates
    x_ease_2d, y_ease_2d = np.meshgrid(x_ease_target, y_ease_target)
    
    # Convert target EASE to lat/lon
    lon_target, lat_target = transformer_from_ease.transform(x_ease_2d, y_ease_2d)
    
    # Handle data with NaN values
    data_work = np.where(np.isnan(data_3d), fill_value, data_3d)
    
    # Check if we need to extrapolate to shallower depths
    min_depth_orig = depth_orig.min()
    min_depth_target = depth_target.min()
    
    if min_depth_target < min_depth_orig:
        # Extrapolate input data to depth 0 on the original lat/lon grid
        # using linear extrapolation from first two depth levels
        d0, d1 = depth_orig[0], depth_orig[1]
        layer_0 = data_work[0, :, :]  # values at d0
        layer_1 = data_work[1, :, :]  # values at d1
        
        # Compute slope on original grid (dval/ddepth)
        slope = (layer_1 - layer_0) / (d1 - d0)
        
        # Extrapolate to depth 0
        layer_at_0 = layer_0 + slope * (0 - d0)
        
        # Prepend extrapolated layer to data and depth arrays
        data_work = np.concatenate([layer_at_0[np.newaxis, :, :], data_work], axis=0)
        depth_orig = np.concatenate([[0], depth_orig])
    
    # Now depth_orig starts at 0, covering all target depths from above
    max_depth_orig = depth_orig.max()
    
    try:
        # Create 3D interpolator on (possibly augmented) original grid
        interpolator = RegularGridInterpolator(
            (depth_orig, lat_orig, lon_orig),
            data_work,
            method='linear',
            bounds_error=False,
            fill_value=fill_value
        )
        
        # Output shape
        n_depth = len(depth_target)
        n_y = len(y_ease_target)
        n_x = len(x_ease_target)
        
        # Initialize output
        data_ease = np.full((n_depth, n_y, n_x), fill_value, dtype=np.float32)
        
        # Interpolate each depth level
        for d_idx, depth_val in enumerate(depth_target):
            # Skip depths beyond original data range
            if depth_val > max_depth_orig:
                continue
            
            # Create points array for this depth level
            depth_2d = np.full_like(lat_target, depth_val)
            points = np.stack([depth_2d, lat_target, lon_target], axis=-1)
            
            # Interpolate
            data_ease[d_idx, :, :] = interpolator(points)
        
    except Exception as e:
        logger.warning(f"3D interpolation failed: {e}. Returning fill values.")
        data_ease = np.full((len(depth_target), len(y_ease_target), len(x_ease_target)), 
                          fill_value, dtype=np.float32)
    
    return data_ease


# ============================================================================
# STERIC HEIGHT COMPUTATION
# ============================================================================


def compute_latlon_grid(x_ease, y_ease, transformer_from_ease):
    """
    Compute latitude/longitude grids for EASE coordinates.
    
    Parameters:
    -----------
    x_ease : np.ndarray
        EASE X coordinates (1D)
    y_ease : np.ndarray
        EASE Y coordinates (1D)
    transformer_from_ease : pyproj.Transformer
        Transformer from EASE to WGS84
    
    Returns:
    --------
    tuple
        (lon_2d, lat_2d) 2D arrays of lat/lon for each EASE grid point
    """
    x_ease_2d, y_ease_2d = np.meshgrid(x_ease, y_ease)
    lon_2d, lat_2d = transformer_from_ease.transform(x_ease_2d, y_ease_2d)
    return lon_2d, lat_2d


def compute_steric_height(thetao, so, depth, lat_2d, lon_2d):
    """
    Compute steric height from potential temperature and practical salinity.
    
    Uses GSW (Gibbs SeaWater) oceanographic toolbox with surface reference (0 dbar).
    
    Parameters:
    -----------
    thetao : np.ndarray
        Potential temperature (depth, y_ease, x_ease) in degrees C
    so : np.ndarray
        Practical salinity (depth, y_ease, x_ease) in PSU
    depth : np.ndarray
        Depth levels (1D) in meters
    lat_2d : np.ndarray
        Latitude grid (y_ease, x_ease)
    lon_2d : np.ndarray
        Longitude grid (y_ease, x_ease)
    
    Returns:
    --------
    np.ndarray
        Steric height (depth, y_ease, x_ease) in meters
    """
    n_depth, n_y, n_x = thetao.shape
    
    # Broadcast depth to 3D (depth, y, x)
    depth_3d = depth[:, np.newaxis, np.newaxis] * np.ones((n_depth, n_y, n_x))
    
    # Broadcast lat/lon to 3D
    lat_3d = lat_2d[np.newaxis, :, :] * np.ones((n_depth, n_y, n_x))
    lon_3d = lon_2d[np.newaxis, :, :] * np.ones((n_depth, n_y, n_x))
    
    # Calculate pressure from depth (negative depth for gsw)
    # p_from_z expects depth below sea surface as negative values
    with np.errstate(invalid='ignore'):
        p = gsw.p_from_z(-depth_3d, lat_3d)
        
        # Calculate absolute salinity from practical salinity
        SA = gsw.SA_from_SP(so, p, lon_3d, lat_3d)
        
        # Convert potential temperature to conservative temperature
        # GLORYS thetao is potential temperature referenced to surface
        CT = gsw.CT_from_pt(SA, thetao)
        
        # Initialize steric height array
        steric_height = np.full_like(thetao, np.nan, dtype=np.float32)
        
        # Compute dynamic height for each horizontal point
        # geo_strf_dyn_height requires depth to increase (p to increase) along axis
        # We need to process column by column since geo_strf_dyn_height works on profiles
        p_ref = 0  # Surface reference pressure
        
        for j in range(n_y):
            for i in range(n_x):
                # Get profile
                sa_prof = SA[:, j, i]
                ct_prof = CT[:, j, i]
                p_prof = p[:, j, i]
                
                # Skip if too many NaNs
                valid = ~np.isnan(sa_prof) & ~np.isnan(ct_prof)
                if valid.sum() < 3:
                    continue
                
                try:
                    # Compute dynamic height for this profile
                    dyn_h = gsw.geo_strf_dyn_height(sa_prof, ct_prof, p_prof, p_ref)
                    # Convert to steric height in meters
                    steric_height[:, j, i] = -dyn_h / 9.7963
                except Exception:
                    # Keep as NaN if computation fails
                    pass
    
    return steric_height


# ============================================================================
# DATA PROCESSING FUNCTIONS
# ============================================================================


def prepare_coordinates(ds):
    """
    Prepare and validate coordinates from the dataset.
    
    Ensures coordinates are in increasing order as required by RegularGridInterpolator.
    
    Parameters:
    -----------
    ds : xr.Dataset
        Input dataset
    
    Returns:
    --------
    tuple
        (lat, lon, depth, flip_lat, flip_lon, flip_depth)
    """
    # Get coordinates
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
    
    # Handle longitude wrapping if needed
    lon = np.where(lon > 180, lon - 360, lon)
    
    return lat, lon, depth, flip_lat, flip_lon, flip_depth


def flip_data_if_needed(data, flip_lat, flip_lon, flip_depth=False, is_3d=False):
    """
    Flip data arrays to match coordinate ordering.
    
    Parameters:
    -----------
    data : np.ndarray
        Data array (2D or 3D)
    flip_lat : bool
        Flip along latitude axis
    flip_lon : bool
        Flip along longitude axis
    flip_depth : bool
        Flip along depth axis (only for 3D)
    is_3d : bool
        Whether data is 3D (depth, lat, lon)
    
    Returns:
    --------
    np.ndarray
        Flipped data
    """
    data = data.copy()
    
    if is_3d:
        # Shape is (depth, lat, lon)
        if flip_depth:
            data = data[::-1, :, :]
        if flip_lat:
            data = data[:, ::-1, :]
        if flip_lon:
            data = data[:, :, ::-1]
    else:
        # Shape is (lat, lon)
        if flip_lat:
            data = data[::-1, :]
        if flip_lon:
            data = data[:, ::-1]
    
    return data


def process_time_slice(ds_slice, time_val, lat_orig, lon_orig, depth_orig,
                       flip_lat, flip_lon, flip_depth,
                       x_ease, y_ease, grid_mapping_attrs,
                       transformer_from_ease, output_dir):
    """
    Process a single time slice and save to file.
    
    Parameters:
    -----------
    ds_slice : xr.Dataset
        Dataset slice for one time step (loaded into memory)
    time_val : datetime
        Time value for this slice
    lat_orig, lon_orig, depth_orig : np.ndarray
        Original coordinates (sorted to increasing)
    flip_lat, flip_lon, flip_depth : bool
        Flags indicating if data needs flipping
    x_ease, y_ease : np.ndarray
        Target EASE coordinates
    grid_mapping_attrs : dict
        Grid mapping attributes
    transformer_from_ease : pyproj.Transformer
        Coordinate transformer
    output_dir : Path
        Output directory
    
    Returns:
    --------
    bool
        True if successful
    """
    # Create output filename from time
    time_str = pd.Timestamp(time_val).strftime('%Y%m%d')
    output_file = output_dir / f'glorys_ease_{time_str}.nc'
    
    if output_file.exists():
        logger.info(f"  Skipping {output_file.name} (already exists)")
        return True
    
    logger.info(f"  Processing time slice: {time_str}")
    
    # Initialize output coordinates
    coords = {
        'time': ('time', [time_val], {'standard_name': 'time', 'axis': 'T'}),
        'depth': ('depth', TARGET_DEPTHS, {
            'standard_name': 'depth',
            'long_name': 'Depth',
            'units': 'm',
            'positive': 'down',
            'axis': 'Z'
        }),
        'y_ease': ('y_ease', y_ease, {
            'standard_name': 'projection_y_coordinate',
            'long_name': 'EASE-Grid Y coordinate',
            'units': 'm',
            'axis': 'Y'
        }),
        'x_ease': ('x_ease', x_ease, {
            'standard_name': 'projection_x_coordinate',
            'long_name': 'EASE-Grid X coordinate',
            'units': 'm',
            'axis': 'X'
        }),
    }
    
    data_vars = {}
    interpolated_data = {}  # Store for steric height computation
    
    # Process 3D variables (thetao, so)
    for var_name in VARS_3D:
        if var_name not in ds_slice.data_vars:
            logger.warning(f"  Variable {var_name} not found, skipping")
            continue
        
        logger.info(f"    Interpolating 3D variable: {var_name}")
        
        # Get data
        da = ds_slice[var_name]
        data_3d = da.values.squeeze()  # Remove time dimension, now (depth, lat, lon)
        
        # Flip if needed
        data_3d = flip_data_if_needed(data_3d, flip_lat, flip_lon, flip_depth, is_3d=True)
        
        # Interpolate to EASE grid with target depths
        data_ease = interpolate_3d_to_ease(
            data_3d, depth_orig, lat_orig, lon_orig,
            TARGET_DEPTHS, x_ease, y_ease, transformer_from_ease
        )
        
        # Store for steric height computation
        interpolated_data[var_name] = data_ease
        
        # Copy attributes, update grid mapping
        attrs = dict(da.attrs)
        attrs['grid_mapping'] = 'ease_grid_mapping'
        # Remove scale/offset as we're storing float values
        attrs.pop('scale_factor', None)
        attrs.pop('add_offset', None)
        attrs.pop('_FillValue', None)
        attrs.pop('missing_value', None)
        
        data_vars[var_name] = (('time', 'depth', 'y_ease', 'x_ease'),
                               data_ease[np.newaxis, :, :, :].astype(np.float32),
                               attrs)
    
    # Compute steric height if both thetao and so are available
    if 'thetao' in interpolated_data and 'so' in interpolated_data:
        logger.info(f"    Computing steric height...")
        
        # Get lat/lon grid for EASE coordinates
        lon_2d, lat_2d = compute_latlon_grid(x_ease, y_ease, transformer_from_ease)
        
        # Compute steric height
        sh = compute_steric_height(
            interpolated_data['thetao'],
            interpolated_data['so'],
            TARGET_DEPTHS,
            lat_2d,
            lon_2d
        )
        
        data_vars['SH'] = (('time', 'depth', 'y_ease', 'x_ease'),
                          sh[np.newaxis, :, :, :].astype(np.float32),
                          {
                              'long_name': 'steric height',
                              'units': 'm',
                              'standard_name': 'steric_change_in_sea_surface_height',
                              'description': 'Steric height computed from so and thetao using GSW',
                              'reference_pressure': '0 dbar (sea surface)',
                              'grid_mapping': 'ease_grid_mapping'
                          })
    
    # Create output dataset
    ds_out = xr.Dataset(data_vars, coords=coords)
    
    # Add grid mapping variable
    ds_out['ease_grid_mapping'] = xr.DataArray(data=0, attrs=grid_mapping_attrs)
    
    # Copy global attributes from original dataset
    ds_out.attrs = dict(ds_slice.attrs)
    ds_out.attrs['title'] = 'GLORYS reanalysis regridded to EASE grid'
    ds_out.attrs['source'] = 'MERCATOR GLORYS12V1 regridded to EASE 25km grid'
    ds_out.attrs['projection'] = 'Lambert Azimuthal Equal Area (Arctic)'
    ds_out.attrs['grid_resolution'] = f'{GRID_RESOLUTION_M/1000:.1f} km'
    ds_out.attrs['depth_levels'] = 'WOA standard depth levels'
    ds_out.attrs['regrid_history'] = f'Regridded from lat/lon to EASE grid on {datetime.now().isoformat()}'
    ds_out.attrs['conventions'] = 'CF-1.8'
    
    # Save to file with compression
    encoding = {}
    for var in data_vars:
        encoding[var] = {
            'dtype': 'float32',
            'zlib': True,
            'complevel': 4,
            '_FillValue': np.nan
        }
    
    ds_out.to_netcdf(output_file, encoding=encoding)
    logger.info(f"    Saved: {output_file.name}")
    
    return True


def main():
    """Main entry point."""
    logger.info("=" * 60)
    logger.info("GLORYS to EASE Grid Regridding")
    logger.info(f"Mode: {'TEST' if TEST_MODE else 'PRODUCTION'}")
    logger.info(f"EASE Grid: {GRID_SIZE_X}x{GRID_SIZE_Y} at {GRID_RESOLUTION_M/1000:.1f} km")
    logger.info(f"Target depth levels: {len(TARGET_DEPTHS)} (WOA standard)")
    logger.info(f"Input:  {INPUT_DIR}")
    logger.info(f"Output: {OUTPUT_DIR}")
    logger.info("=" * 60)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create EASE grid structure
    logger.info("Creating EASE grid structure...")
    x_ease, y_ease, grid_mapping_attrs = create_ease_grid_structure()
    
    # Create coordinate transformers
    logger.info("Setting up coordinate transformers...")
    transformer_to_ease, transformer_from_ease = create_transformers()
    
    # Find all input files
    input_files = sorted(INPUT_DIR.glob('*.nc'))
    
    # Filter by year if in production mode
    # Match pattern _mean_YYYY to avoid substring matches (e.g., 2012 in 202012)
    if not TEST_MODE and PRODUCTION_YEAR_FILTER:
        year_pattern = f'_mean_{PRODUCTION_YEAR_FILTER}'
        input_files = [f for f in input_files if year_pattern in f.name]
        logger.info(f"Filtered to year {PRODUCTION_YEAR_FILTER}: {len(input_files)} files")
    elif TEST_MODE:
        input_files = input_files[:TEST_MAX_FILES]
        logger.info(f"Test mode: limiting to {len(input_files)} files")
    else:
        logger.info(f"Found {len(input_files)} input files")
    
    if len(input_files) == 0:
        logger.error("No input files found!")
        return
    
    # Open dataset lazily to get coordinate information
    logger.info("Opening dataset lazily to extract coordinates...")
    ds_sample = xr.open_dataset(input_files[0])
    
    # Prepare coordinates (sorted to increasing for interpolation)
    lat_orig, lon_orig, depth_orig, flip_lat, flip_lon, flip_depth = prepare_coordinates(ds_sample)
    
    logger.info(f"Original grid: lat={len(lat_orig)}, lon={len(lon_orig)}, depth={len(depth_orig)}")
    logger.info(f"Flip flags: lat={flip_lat}, lon={flip_lon}, depth={flip_depth}")
    
    # Compute EASE coordinates for original grid (for reference/debugging)
    x_ease_orig, y_ease_orig = compute_ease_coords_for_original_grid(
        lat_orig, lon_orig, transformer_to_ease
    )
    logger.info(f"Original grid EASE extents: X=[{x_ease_orig.min():.0f}, {x_ease_orig.max():.0f}], "
                f"Y=[{y_ease_orig.min():.0f}, {y_ease_orig.max():.0f}]")
    
    ds_sample.close()
    
    # Process each file
    processed = 0
    errors = 0
    
    for nc_file in input_files:
        logger.info(f"\nProcessing file: {nc_file.name}")
        
        try:
            # Open file and load time slice
            ds = xr.open_dataset(nc_file)
            
            # Get time values
            time_vals = ds['time'].values
            
            for t_idx, time_val in enumerate(time_vals):
                # Select and load time slice into memory
                ds_slice = ds.isel(time=t_idx).compute()
                
                success = process_time_slice(
                    ds_slice, time_val,
                    lat_orig, lon_orig, depth_orig,
                    flip_lat, flip_lon, flip_depth,
                    x_ease, y_ease, grid_mapping_attrs,
                    transformer_from_ease, OUTPUT_DIR
                )
                
                if success:
                    processed += 1
                else:
                    errors += 1
            
            ds.close()
            
        except Exception as e:
            logger.error(f"Error processing {nc_file.name}: {e}")
            errors += 1
    
    logger.info("\n" + "=" * 60)
    logger.info("PROCESSING COMPLETE")
    logger.info(f"Processed: {processed} time slices")
    logger.info(f"Errors:    {errors}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
