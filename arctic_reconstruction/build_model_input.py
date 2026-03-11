#!/usr/bin/env python3
"""
Build Model Input Data for Arctic Profile Reconstruction

This script assembles all required data to run the LSTM model trained by
lstm_pytorch_pd_mcdo.py to predict hydrographic profiles over the Arctic.

It combines:
- Bathymetry from GEBCO (on EASE grid)
- Ocean mask (on EASE grid)
- SST from satellite data (interpolated to monthly timesteps)
- SSS from satellite data (interpolated to monthly timesteps)
- ADT from satellite data (interpolated to monthly timesteps)
- GLORYS 3D fields: temperature, salinity, steric height (monthly)
- Day of year (DOY)
- Lat/lon grids computed from EASE coordinates

Output: Monthly NetCDF files with all variables on the EASE grid.
"""

import numpy as np
import xarray as xr
import pyproj
from pathlib import Path
from glob import glob
from datetime import datetime, timedelta
import logging
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# CONFIGURATION
# ============================================================================

# Year range to process
START_YEAR = 2012
END_YEAR = 2012

# EASE Grid Parameters (must match other scripts)
GRID_RESOLUTION_M = 25000  # meters (25 km)
GRID_SIZE_X = 350
GRID_SIZE_Y = 350
EASE_LAT_0 = 90
EASE_LON_0 = 0
EASE_FALSE_EASTING = 0
EASE_FALSE_NORTHING = 0
EASE_PROJ4 = (f"+proj=laea +lat_0={EASE_LAT_0} +lon_0={EASE_LON_0} "
              f"+x_0={EASE_FALSE_EASTING} +y_0={EASE_FALSE_NORTHING} "
              f"+datum=WGS84 +units=m")

# Input data directories
DATA_DIR = Path('/home/nico/SACO/FRESH-CARE/Codes/WP2/buongiorno_to_pytorch_padding/arctic_reconstruction/data')
SST_BASE_DIR = Path('/home/nico/Desktop/AUX_DIR_FRESH_CARE/satellite/SST/data_ease')
SSS_BASE_DIR = Path('/home/nico/Desktop/AUX_DIR_FRESH_CARE/satellite/SSS_cci_v55/regridded_filled_wg_ease')
ADT_BASE_DIR = Path('/home/nico/Desktop/AUX_DIR_FRESH_CARE/satellite/ADT/aviso_regridded_0.25_north_pole_interp_ease')
GLORYS_BASE_DIR = Path('/home/nico/Desktop/AUX_DIR_FRESH_CARE/glorys_2012_ease_woaDepths')

# Output directory
OUTPUT_DIR = Path('/home/nico/Desktop/AUX_DIR_FRESH_CARE/model_input')

# Static data files
GEBCO_FILE = DATA_DIR / 'gebco_ease_grid_25km.nc'
OCEAN_MASK_FILE = DATA_DIR / 'ocean_mask_ease_grid_25km.nc'

# Time interpolation window (days before and after target)
TIME_WINDOW_DAYS = 16

# Output time units (CF convention)
TIME_UNITS = 'days since 1950-01-01T00:00:00+00:00'
TIME_CALENDAR = 'standard'

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
# GRID AND COORDINATE FUNCTIONS
# ============================================================================


def create_ease_grid_structure():
    """Create EASE grid coordinates and metadata."""
    x_min = -(GRID_SIZE_X * GRID_RESOLUTION_M) / 2
    y_min = -(GRID_SIZE_Y * GRID_RESOLUTION_M) / 2
    
    x_ease = np.arange(GRID_SIZE_X) * GRID_RESOLUTION_M + x_min + GRID_RESOLUTION_M / 2
    y_ease = np.arange(GRID_SIZE_Y) * GRID_RESOLUTION_M + y_min + GRID_RESOLUTION_M / 2
    
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


def compute_latlon_grids(x_ease, y_ease):
    """
    Compute lat/lon 2D grids from EASE coordinates.
    
    Returns:
        tuple: (lon_2d, lat_2d) arrays of shape (y_ease, x_ease)
    """
    crs_wgs84 = pyproj.CRS.from_epsg(4326)
    crs_ease = pyproj.CRS.from_proj4(EASE_PROJ4)
    
    transformer_from_ease = pyproj.Transformer.from_crs(
        crs_ease, crs_wgs84, always_xy=True
    )
    
    x_ease_2d, y_ease_2d = np.meshgrid(x_ease, y_ease)
    lon_2d, lat_2d = transformer_from_ease.transform(x_ease_2d, y_ease_2d)
    
    return lon_2d, lat_2d


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================


def load_static_data():
    """Load bathymetry and ocean mask (time-independent)."""
    logger.info("Loading static data (bathymetry and ocean mask)...")
    
    ds_gebco = xr.open_dataset(GEBCO_FILE)
    ds_mask = xr.open_dataset(OCEAN_MASK_FILE)
    
    elevation = ds_gebco['elevation'].values
    ocean_mask = ds_mask['ocean_mask'].values
    
    logger.info(f"  Bathymetry shape: {elevation.shape}")
    logger.info(f"  Ocean mask shape: {ocean_mask.shape}")
    logger.info(f"  Ocean pixels: {np.sum(ocean_mask == 1)}")
    
    ds_gebco.close()
    ds_mask.close()
    
    return elevation, ocean_mask


def find_sst_files_for_time_window(target_time, window_days=TIME_WINDOW_DAYS):
    """
    Find SST files covering the time window around target_time.
    May need to access files from adjacent months/years.
    """
    start_time = target_time - timedelta(days=window_days)
    end_time = target_time + timedelta(days=window_days)
    
    files = []
    current = start_time
    
    while current <= end_time:
        year = current.year
        month = current.month
        day = current.day
        
        # SST files are organized as YYYY/MM/file_YYYY-MM-DD.nc
        pattern = SST_BASE_DIR / str(year) / f'{month:02d}' / f'*_{year}-{month:02d}-{day:02d}_*.nc'
        found = glob(str(pattern))
        files.extend(found)
        
        current += timedelta(days=1)
    
    return sorted(set(files))


def find_sss_files_for_time_window(target_time, window_days=TIME_WINDOW_DAYS):
    """
    Find SSS files covering the time window around target_time.
    SSS data is in yearly files, so may need to load adjacent years.
    """
    start_time = target_time - timedelta(days=window_days)
    end_time = target_time + timedelta(days=window_days)
    
    years_needed = set()
    current = start_time
    while current <= end_time:
        years_needed.add(current.year)
        current += timedelta(days=1)
    
    files = []
    for year in sorted(years_needed):
        pattern = SSS_BASE_DIR / f'sss_merge_cci_{year}_regridded_025_filled_wg.nc'
        if pattern.exists():
            files.append(str(pattern))
    
    return files


def find_adt_files_for_time_window(target_time, window_days=TIME_WINDOW_DAYS):
    """
    Find ADT files covering the time window around target_time.
    ADT files are organized as YYYY/file_YYYYMMDD.nc
    """
    start_time = target_time - timedelta(days=window_days)
    end_time = target_time + timedelta(days=window_days)
    
    files = []
    years_to_check = set()
    current = start_time
    while current <= end_time:
        years_to_check.add(current.year)
        current += timedelta(days=1)
    
    for year in sorted(years_to_check):
        year_dir = ADT_BASE_DIR / str(year)
        if year_dir.exists():
            year_files = sorted(glob(str(year_dir / '*.nc')))
            for f in year_files:
                # Extract date from filename
                fname = Path(f).name
                # Format: dt_arctic_multimission_sea_level_YYYYMMDD_regridded_025.nc
                date_str = fname.split('_')[5]  # YYYYMMDD (index 5, not 4)
                try:
                    file_date = datetime.strptime(date_str, '%Y%m%d')
                    if start_time <= file_date <= end_time:
                        files.append(f)
                except (ValueError, IndexError):
                    continue
    
    return sorted(files)


def load_sst_for_target_time(target_time, window_days=TIME_WINDOW_DAYS):
    """Load and interpolate SST data to target time."""
    files = find_sst_files_for_time_window(target_time, window_days)
    
    if not files:
        logger.warning(f"No SST files found for {target_time}")
        return None
    
    # Load files and concatenate
    ds = xr.open_mfdataset(files, combine='by_coords')
    
    # Select time window and interpolate
    start_time = target_time - timedelta(days=window_days)
    end_time = target_time + timedelta(days=window_days)
    
    ds_window = ds.sel(time=slice(start_time, end_time))
    
    # Interpolate to target time
    target_np = np.datetime64(target_time)
    sst_interp = ds_window['analysed_sst'].interp(time=target_np, method='linear')
    
    result = sst_interp.values
    ds.close()
    
    return result


def load_sss_for_target_time(target_time, window_days=TIME_WINDOW_DAYS):
    """Load and interpolate SSS data to target time."""
    files = find_sss_files_for_time_window(target_time, window_days)
    
    if not files:
        logger.warning(f"No SSS files found for {target_time}")
        return None
    
    # Load files and concatenate
    ds = xr.open_mfdataset(files, combine='by_coords')
    
    # Select time window and interpolate
    start_time = target_time - timedelta(days=window_days)
    end_time = target_time + timedelta(days=window_days)
    
    ds_window = ds.sel(time=slice(start_time, end_time))
    
    # Interpolate to target time
    target_np = np.datetime64(target_time)
    sss_interp = ds_window['sss'].interp(time=target_np, method='linear')
    
    result = sss_interp.values
    ds.close()
    
    return result


def load_adt_for_target_time(target_time, window_days=TIME_WINDOW_DAYS):
    """Load and interpolate ADT data to target time."""
    files = find_adt_files_for_time_window(target_time, window_days)
    
    if not files:
        logger.warning(f"No ADT files found for {target_time}")
        return None
    
    # Load files and concatenate
    ds = xr.open_mfdataset(files, combine='by_coords')
    
    # Select time window and interpolate
    start_time = target_time - timedelta(days=window_days)
    end_time = target_time + timedelta(days=window_days)
    
    ds_window = ds.sel(time=slice(start_time, end_time))
    
    # Interpolate to target time
    target_np = np.datetime64(target_time)
    adt_interp = ds_window['adt'].interp(time=target_np, method='linear')
    
    result = adt_interp.values
    ds.close()
    
    return result


def load_glorys_for_month(year, month):
    """
    Load GLORYS data for a specific month.
    GLORYS files are named: glorys_ease_YYYYMMDD.nc (usually 15th or 16th of month)
    """
    # Find the GLORYS file for this month
    pattern = GLORYS_BASE_DIR / f'glorys_ease_{year}{month:02d}*.nc'
    files = glob(str(pattern))
    
    if not files:
        logger.warning(f"No GLORYS file found for {year}-{month:02d}")
        return None, None, None, None
    
    # Take the first matching file
    glorys_file = files[0]
    logger.info(f"  Loading GLORYS from: {Path(glorys_file).name}")
    
    ds = xr.open_dataset(glorys_file)
    
    thetao = ds['thetao'].values.squeeze()  # Remove time dim: (depth, y, x)
    so = ds['so'].values.squeeze()
    SH = ds['SH'].values.squeeze()
    depth = ds['depth'].values
    
    ds.close()
    
    return thetao, so, SH, depth


# ============================================================================
# OUTPUT DATASET CREATION
# ============================================================================


def create_output_dataset(target_time, x_ease, y_ease, depth, grid_mapping_attrs,
                          sst, sss, adt, thetao, so, SH, elevation, ocean_mask,
                          lon_2d, lat_2d):
    """
    Create the output xarray Dataset for a single monthly timestep.
    
    Variable names match training data format:
    - SST, SSS, ADT for surface data
    - T_glorys, S_glorys, SH_glorys for 3D fields
    """
    # Compute day of year
    doy = target_time.timetuple().tm_yday
    
    # Create time coordinate with proper units
    # Reference: 1950-01-01
    ref_date = datetime(1950, 1, 1)
    days_since_ref = (target_time - ref_date).days
    
    # Create Dataset
    ds = xr.Dataset(
        coords={
            'time': ('time', [days_since_ref], {
                'standard_name': 'time',
                'long_name': 'Time',
                'units': TIME_UNITS,
                'calendar': TIME_CALENDAR
            }),
            'depth': ('depth', depth, {
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
            })
        }
    )
    
    # Add surface satellite data (time, y_ease, x_ease)
    ds['SST'] = xr.DataArray(
        sst[np.newaxis, :, :],
        dims=['time', 'y_ease', 'x_ease'],
        attrs={
            'standard_name': 'sea_surface_temperature',
            'long_name': 'Sea Surface Temperature (from satellite)',
            'units': 'K',
            'source': 'C3S-GLO-SST-L4-REP',
            'grid_mapping': 'ease_grid_mapping'
        }
    )
    
    ds['SSS'] = xr.DataArray(
        sss[np.newaxis, :, :],
        dims=['time', 'y_ease', 'x_ease'],
        attrs={
            'standard_name': 'sea_surface_salinity',
            'long_name': 'Sea Surface Salinity (from satellite)',
            'units': 'PSU',
            'source': 'ESA CCI SSS v5.5',
            'grid_mapping': 'ease_grid_mapping'
        }
    )
    
    ds['ADT'] = xr.DataArray(
        adt[np.newaxis, :, :],
        dims=['time', 'y_ease', 'x_ease'],
        attrs={
            'standard_name': 'absolute_dynamic_topography',
            'long_name': 'Absolute Dynamic Topography',
            'units': 'm',
            'source': 'AVISO Arctic Altimetry',
            'grid_mapping': 'ease_grid_mapping'
        }
    )
    
    # Add GLORYS 3D fields (time, depth, y_ease, x_ease)
    ds['T_glorys'] = xr.DataArray(
        thetao[np.newaxis, :, :, :],
        dims=['time', 'depth', 'y_ease', 'x_ease'],
        attrs={
            'standard_name': 'sea_water_potential_temperature',
            'long_name': 'GLORYS Temperature (climatology reference)',
            'units': 'degrees_C',
            'source': 'GLORYS12V1',
            'grid_mapping': 'ease_grid_mapping'
        }
    )
    
    ds['S_glorys'] = xr.DataArray(
        so[np.newaxis, :, :, :],
        dims=['time', 'depth', 'y_ease', 'x_ease'],
        attrs={
            'standard_name': 'sea_water_practical_salinity',
            'long_name': 'GLORYS Salinity (climatology reference)',
            'units': 'PSU',
            'source': 'GLORYS12V1',
            'grid_mapping': 'ease_grid_mapping'
        }
    )
    
    ds['SH_glorys'] = xr.DataArray(
        SH[np.newaxis, :, :, :],
        dims=['time', 'depth', 'y_ease', 'x_ease'],
        attrs={
            'standard_name': 'steric_height',
            'long_name': 'GLORYS Steric Height (climatology reference)',
            'units': 'm',
            'source': 'Computed from GLORYS12V1 T/S',
            'grid_mapping': 'ease_grid_mapping'
        }
    )
    
    # Add static data (no time dimension)
    ds['elevation'] = xr.DataArray(
        elevation,
        dims=['y_ease', 'x_ease'],
        attrs={
            'standard_name': 'height_above_reference_ellipsoid',
            'long_name': 'Bathymetry/Elevation',
            'units': 'm',
            'positive': 'up',
            'source': 'GEBCO 2025',
            'grid_mapping': 'ease_grid_mapping'
        }
    )
    
    ds['ocean_mask'] = xr.DataArray(
        ocean_mask,
        dims=['y_ease', 'x_ease'],
        attrs={
            'long_name': 'Ocean mask (1=ocean, 0=land)',
            'flag_values': [0, 1],
            'flag_meanings': 'land ocean',
            'source': 'Natural Earth',
            'grid_mapping': 'ease_grid_mapping'
        }
    )
    
    # Add lat/lon grids
    ds['latitude'] = xr.DataArray(
        lat_2d,
        dims=['y_ease', 'x_ease'],
        attrs={
            'standard_name': 'latitude',
            'long_name': 'Latitude',
            'units': 'degrees_north'
        }
    )
    
    ds['longitude'] = xr.DataArray(
        lon_2d,
        dims=['y_ease', 'x_ease'],
        attrs={
            'standard_name': 'longitude',
            'long_name': 'Longitude',
            'units': 'degrees_east'
        }
    )
    
    # Add 2D coordinate grids for x_ease and y_ease (for model input)
    x_ease_2d, y_ease_2d = np.meshgrid(x_ease, y_ease)
    
    ds['X_EASE'] = xr.DataArray(
        x_ease_2d,
        dims=['y_ease', 'x_ease'],
        attrs={
            'long_name': 'EASE-Grid X coordinate (2D grid)',
            'units': 'm',
            'description': '2D grid of X coordinates for model input'
        }
    )
    
    ds['Y_EASE'] = xr.DataArray(
        y_ease_2d,
        dims=['y_ease', 'x_ease'],
        attrs={
            'long_name': 'EASE-Grid Y coordinate (2D grid)',
            'units': 'm',
            'description': '2D grid of Y coordinates for model input'
        }
    )
    
    # Add day of year (time-dependent scalar)
    ds['DOY'] = xr.DataArray(
        [doy],
        dims=['time'],
        attrs={
            'long_name': 'Day of Year',
            'units': '1',
            'valid_range': [1, 366]
        }
    )
    
    # Add grid mapping variable
    ds['ease_grid_mapping'] = xr.DataArray(
        0,
        attrs=grid_mapping_attrs
    )
    
    # Global attributes
    ds.attrs = {
        'title': 'Arctic Model Input Data for LSTM Profile Reconstruction',
        'institution': 'CNR-ISMAR',
        'source': 'Combined satellite and GLORYS reanalysis data',
        'history': f'Created {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        'references': 'lstm_pytorch_pd_mcdo.py',
        'Conventions': 'CF-1.8',
        'projection': 'Lambert Azimuthal Equal Area (Arctic)',
        'projection_latitude_of_origin': EASE_LAT_0,
        'projection_longitude_of_origin': EASE_LON_0,
        'projection_false_easting': EASE_FALSE_EASTING,
        'projection_false_northing': EASE_FALSE_NORTHING,
        'proj4_string': EASE_PROJ4,
        'grid_resolution': f'{GRID_RESOLUTION_M/1000:.1f} km',
        'grid_size': f'{GRID_SIZE_X} x {GRID_SIZE_Y}',
        'time_interpolation_method': 'linear',
        'time_reference': target_time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    return ds


# ============================================================================
# MAIN PROCESSING FUNCTIONS
# ============================================================================


def generate_monthly_timesteps(year):
    """
    Generate target timesteps: 15th of each month at 12:00.
    
    Returns:
        list of datetime objects
    """
    timesteps = []
    for month in range(1, 13):
        target = datetime(year, month, 15, 12, 0, 0)
        timesteps.append(target)
    return timesteps


def process_single_month(target_time, x_ease, y_ease, grid_mapping_attrs,
                         elevation, ocean_mask, lon_2d, lat_2d):
    """
    Process data for a single monthly timestep.
    
    Returns:
        xr.Dataset or None if data is missing
    """
    year = target_time.year
    month = target_time.month
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {target_time.strftime('%Y-%m-%d')}")
    logger.info(f"{'='*60}")
    
    # Load satellite surface data (interpolated to target time)
    logger.info("Loading SST data...")
    sst = load_sst_for_target_time(target_time)
    if sst is None:
        logger.error(f"Failed to load SST for {target_time}")
        return None
    logger.info(f"  SST shape: {sst.shape}, range: [{np.nanmin(sst):.2f}, {np.nanmax(sst):.2f}]")
    
    logger.info("Loading SSS data...")
    sss = load_sss_for_target_time(target_time)
    if sss is None:
        logger.error(f"Failed to load SSS for {target_time}")
        return None
    logger.info(f"  SSS shape: {sss.shape}, range: [{np.nanmin(sss):.2f}, {np.nanmax(sss):.2f}]")
    
    logger.info("Loading ADT data...")
    adt = load_adt_for_target_time(target_time)
    if adt is None:
        logger.error(f"Failed to load ADT for {target_time}")
        return None
    logger.info(f"  ADT shape: {adt.shape}, range: [{np.nanmin(adt):.2f}, {np.nanmax(adt):.2f}]")
    
    # Load GLORYS 3D fields (monthly data as-is)
    logger.info("Loading GLORYS 3D data...")
    thetao, so, SH, depth = load_glorys_for_month(year, month)
    if thetao is None:
        logger.error(f"Failed to load GLORYS for {year}-{month:02d}")
        return None
    logger.info(f"  GLORYS T shape: {thetao.shape}")
    logger.info(f"  GLORYS S shape: {so.shape}")
    logger.info(f"  GLORYS SH shape: {SH.shape}")
    logger.info(f"  Depth levels: {len(depth)}")
    
    # Create output dataset
    logger.info("Creating output dataset...")
    ds = create_output_dataset(
        target_time, x_ease, y_ease, depth, grid_mapping_attrs,
        sst, sss, adt, thetao, so, SH, elevation, ocean_mask,
        lon_2d, lat_2d
    )
    
    return ds


def main():
    """Main processing pipeline."""
    logger.info("="*70)
    logger.info("Arctic Model Input Builder")
    logger.info(f"Processing years: {START_YEAR} to {END_YEAR}")
    logger.info("="*70)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Initialize EASE grid
    logger.info("\nInitializing EASE grid...")
    x_ease, y_ease, grid_mapping_attrs = create_ease_grid_structure()
    logger.info(f"  Grid size: {GRID_SIZE_X} x {GRID_SIZE_Y}")
    logger.info(f"  Resolution: {GRID_RESOLUTION_M/1000:.1f} km")
    
    # Compute lat/lon grids
    logger.info("Computing lat/lon grids from EASE coordinates...")
    lon_2d, lat_2d = compute_latlon_grids(x_ease, y_ease)
    logger.info(f"  Latitude range: [{lat_2d.min():.2f}, {lat_2d.max():.2f}]")
    logger.info(f"  Longitude range: [{lon_2d.min():.2f}, {lon_2d.max():.2f}]")
    
    # Load static data (only once)
    elevation, ocean_mask = load_static_data()
    
    # Process each year
    total_processed = 0
    total_failed = 0
    
    for year in range(START_YEAR, END_YEAR + 1):
        logger.info(f"\n{'#'*70}")
        logger.info(f"# YEAR {year}")
        logger.info(f"{'#'*70}")
        
        # Generate monthly timesteps
        timesteps = generate_monthly_timesteps(year)
        
        for target_time in timesteps:
            try:
                ds = process_single_month(
                    target_time, x_ease, y_ease, grid_mapping_attrs,
                    elevation, ocean_mask, lon_2d, lat_2d
                )
                
                if ds is not None:
                    # Save to file
                    output_file = OUTPUT_DIR / f'model_input_{year}_{target_time.month:02d}.nc'
                    
                    # Encoding for compression and proper time handling
                    encoding = {
                        'time': {'dtype': 'float64'},
                        'SST': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
                        'SSS': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
                        'ADT': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
                        'T_glorys': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
                        'S_glorys': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
                        'SH_glorys': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
                        'elevation': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
                        'ocean_mask': {'dtype': 'uint8'},
                        'latitude': {'dtype': 'float32'},
                        'longitude': {'dtype': 'float32'},
                        'X_EASE': {'dtype': 'float32'},
                        'Y_EASE': {'dtype': 'float32'},
                        'DOY': {'dtype': 'int16'}
                    }
                    
                    ds.to_netcdf(output_file, encoding=encoding)
                    logger.info(f"  Saved: {output_file.name}")
                    total_processed += 1
                else:
                    total_failed += 1
                    
            except Exception as e:
                logger.error(f"Error processing {target_time}: {e}")
                import traceback
                traceback.print_exc()
                total_failed += 1
    
    # Summary
    logger.info("\n" + "="*70)
    logger.info("PROCESSING COMPLETE")
    logger.info(f"  Files created: {total_processed}")
    logger.info(f"  Files failed: {total_failed}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
