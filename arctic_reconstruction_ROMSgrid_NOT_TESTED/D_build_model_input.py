#!/usr/bin/env python3
"""
Build Model Input Data for Arctic Profile Reconstruction on ROMS Grid

This script assembles all required data to run the LSTM model at ROMS grid locations.

It combines:
- ROMS grid reference (lat_rho, lon_rho, mask_rho, h, ease_x, ease_y)
- SST from satellite data (sampled at ROMS points, interpolated to monthly timesteps)
- SSS from satellite data (sampled at ROMS points, interpolated to monthly timesteps)
- ADT from satellite data (sampled at ROMS points, interpolated to monthly timesteps)
- GLORYS 3D fields: temperature, salinity, steric height (sampled at ROMS points, monthly)
- Day of year (DOY)

Output: Monthly NetCDF files with all variables on the ROMS grid.

Key difference from EASE version: EASE coordinates (ease_x, ease_y) are model INPUT FEATURES,
not the output grid. Predictions happen directly at ROMS grid locations.
"""

import numpy as np
import xarray as xr
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

# Input data directories
ROMS_GRID_REF = Path('/home/nico/SACO/FRESH-CARE/Data_lstm_reconstruction_ROMSgrid/data_for_reconstruction/roms_grid_reference.nc')
SST_BASE_DIR = Path('/home/nico/SACO/FRESH-CARE/Data_lstm_reconstruction_ROMSgrid/data_for_reconstruction/SST_roms')
SSS_BASE_DIR = Path('/home/nico/SACO/FRESH-CARE/Data_lstm_reconstruction_ROMSgrid/data_for_reconstruction/SSS_roms')
ADT_BASE_DIR = Path('/home/nico/SACO/FRESH-CARE/Data_lstm_reconstruction_ROMSgrid/data_for_reconstruction/ADT_roms')
GLORYS_BASE_DIR = Path('/home/nico/SACO/FRESH-CARE/Data_lstm_reconstruction_ROMSgrid/data_for_reconstruction/glorys_roms_woaDepths')

# Output directory
OUTPUT_DIR = Path('/home/nico/SACO/FRESH-CARE/Data_lstm_reconstruction_ROMSgrid/model_input')

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
# GRID FUNCTIONS
# ============================================================================


def load_roms_grid_reference():
    """Load ROMS grid reference with all necessary fields."""
    logger.info(f"Loading ROMS grid reference from {ROMS_GRID_REF}")
    ds = xr.open_dataset(ROMS_GRID_REF)
    
    grid = {
        'lat_rho': ds['lat_rho'].values,
        'lon_rho': ds['lon_rho'].values,
        'mask_rho': ds['mask_rho'].values,
        'h': ds['h'].values,
        'ease_x': ds['ease_x'].values,
        'ease_y': ds['ease_y'].values,
        'eta_rho': ds['eta_rho'].values,
        'xi_rho': ds['xi_rho'].values,
    }
    
    logger.info(f"  Grid shape: {grid['lat_rho'].shape}")
    logger.info(f"  Ocean pixels: {np.sum(grid['mask_rho'] == 1):,}")
    
    ds.close()
    return grid


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================


def find_sst_files_for_time_window(target_time, window_days=TIME_WINDOW_DAYS):
    """Find SST files covering the time window around target_time."""
    start_time = target_time - timedelta(days=window_days)
    end_time = target_time + timedelta(days=window_days)
    
    files = []
    current = start_time
    
    while current <= end_time:
        year = current.year
        month = current.month
        day = current.day
        
        # SST files organized as YYYY/MM/file_YYYY-MM-DD.nc
        pattern = SST_BASE_DIR / str(year) / f'{month:02d}' / f'*_{year}-{month:02d}-{day:02d}_*.nc'
        found = glob(str(pattern))
        files.extend(found)
        
        current += timedelta(days=1)
    
    return sorted(set(files))


def find_sss_files_for_time_window(target_time, window_days=TIME_WINDOW_DAYS):
    """Find SSS files covering the time window around target_time."""
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
    """Find ADT files covering the time window around target_time."""
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
                fname = Path(f).name
                # Format: dt_arctic_multimission_sea_level_YYYYMMDD_regridded_025.nc
                date_str = fname.split('_')[5]
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
    
    ds = xr.open_mfdataset(files, combine='by_coords')
    
    start_time = target_time - timedelta(days=window_days)
    end_time = target_time + timedelta(days=window_days)
    
    ds_window = ds.sel(time=slice(start_time, end_time))
    
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
    
    ds = xr.open_mfdataset(files, combine='by_coords')
    
    start_time = target_time - timedelta(days=window_days)
    end_time = target_time + timedelta(days=window_days)
    
    ds_window = ds.sel(time=slice(start_time, end_time))
    
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
    
    ds = xr.open_mfdataset(files, combine='by_coords')
    
    start_time = target_time - timedelta(days=window_days)
    end_time = target_time + timedelta(days=window_days)
    
    ds_window = ds.sel(time=slice(start_time, end_time))
    
    target_np = np.datetime64(target_time)
    adt_interp = ds_window['adt'].interp(time=target_np, method='linear')
    
    result = adt_interp.values
    ds.close()
    
    return result


def load_glorys_for_month(year, month):
    """Load GLORYS data for a specific month."""
    pattern = GLORYS_BASE_DIR / f'glorys_roms_{year}{month:02d}*.nc'
    files = glob(str(pattern))
    
    if not files:
        logger.warning(f"No GLORYS file found for {year}-{month:02d}")
        return None, None, None, None
    
    glorys_file = files[0]
    logger.info(f"  Loading GLORYS from: {Path(glorys_file).name}")
    
    ds = xr.open_dataset(glorys_file)
    
    thetao = ds['thetao'].values.squeeze()  # (depth, eta_rho, xi_rho)
    so = ds['so'].values.squeeze()
    SH = ds['SH'].values.squeeze()
    depth = ds['depth'].values
    
    ds.close()
    
    return thetao, so, SH, depth


# ============================================================================
# OUTPUT DATASET CREATION
# ============================================================================


def create_output_dataset(target_time, grid, depth, sst, sss, adt, thetao, so, SH):
    """
    Create the output xarray Dataset for a single monthly timestep.
    
    Variable names match training data format.
    Key: EASE coordinates (ease_x, ease_y) are INPUT FEATURES, not the grid.
    """
    doy = target_time.timetuple().tm_yday
    
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
            'eta_rho': ('eta_rho', grid['eta_rho'], {
                'long_name': 'eta index of rho-points',
                'units': '1'
            }),
            'xi_rho': ('xi_rho', grid['xi_rho'], {
                'long_name': 'xi index of rho-points',
                'units': '1'
            })
        }
    )
    
    # Surface satellite data (time, eta_rho, xi_rho)
    # Convert SST from Kelvin to Celsius
    sst_celsius = sst - 273.15
    ds['SST'] = xr.DataArray(
        sst_celsius[np.newaxis, :, :],
        dims=['time', 'eta_rho', 'xi_rho'],
        attrs={
            'standard_name': 'sea_surface_temperature',
            'long_name': 'Sea Surface Temperature (from satellite)',
            'units': 'degrees_C',
            'source': 'C3S-GLO-SST-L4-REP sampled at ROMS grid'
        }
    )
    
    ds['SSS'] = xr.DataArray(
        sss[np.newaxis, :, :],
        dims=['time', 'eta_rho', 'xi_rho'],
        attrs={
            'standard_name': 'sea_surface_salinity',
            'long_name': 'Sea Surface Salinity (from satellite)',
            'units': 'PSU',
            'source': 'ESA CCI SSS v5.5 sampled at ROMS grid'
        }
    )
    
    ds['ADT'] = xr.DataArray(
        adt[np.newaxis, :, :],
        dims=['time', 'eta_rho', 'xi_rho'],
        attrs={
            'standard_name': 'absolute_dynamic_topography',
            'long_name': 'Absolute Dynamic Topography',
            'units': 'm',
            'source': 'AVISO Arctic Altimetry sampled at ROMS grid'
        }
    )
    
    # GLORYS 3D fields (time, depth, eta_rho, xi_rho)
    ds['T_glorys'] = xr.DataArray(
        thetao[np.newaxis, :, :, :],
        dims=['time', 'depth', 'eta_rho', 'xi_rho'],
        attrs={
            'standard_name': 'sea_water_potential_temperature',
            'long_name': 'GLORYS Temperature (climatology reference)',
            'units': 'degrees_C',
            'source': 'GLORYS12V1 sampled at ROMS grid'
        }
    )
    
    ds['S_glorys'] = xr.DataArray(
        so[np.newaxis, :, :, :],
        dims=['time', 'depth', 'eta_rho', 'xi_rho'],
        attrs={
            'standard_name': 'sea_water_practical_salinity',
            'long_name': 'GLORYS Salinity (climatology reference)',
            'units': 'PSU',
            'source': 'GLORYS12V1 sampled at ROMS grid'
        }
    )
    
    ds['SH_glorys'] = xr.DataArray(
        SH[np.newaxis, :, :, :],
        dims=['time', 'depth', 'eta_rho', 'xi_rho'],
        attrs={
            'standard_name': 'steric_height',
            'long_name': 'GLORYS Steric Height (climatology reference)',
            'units': 'm',
            'source': 'Computed from GLORYS12V1 T/S at ROMS grid'
        }
    )
    
    # Static/reference data (no time dimension)
    ds['lat_rho'] = xr.DataArray(
        grid['lat_rho'],
        dims=['eta_rho', 'xi_rho'],
        attrs={
            'standard_name': 'latitude',
            'long_name': 'Latitude',
            'units': 'degrees_north'
        }
    )
    
    ds['lon_rho'] = xr.DataArray(
        grid['lon_rho'],
        dims=['eta_rho', 'xi_rho'],
        attrs={
            'standard_name': 'longitude',
            'long_name': 'Longitude',
            'units': 'degrees_east'
        }
    )
    
    ds['mask_rho'] = xr.DataArray(
        grid['mask_rho'],
        dims=['eta_rho', 'xi_rho'],
        attrs={
            'long_name': 'Ocean mask (1=ocean, 0=land)',
            'flag_values': [0, 1],
            'flag_meanings': 'land ocean'
        }
    )
    
    ds['h'] = xr.DataArray(
        grid['h'],
        dims=['eta_rho', 'xi_rho'],
        attrs={
            'long_name': 'Bathymetry',
            'units': 'm',
            'positive': 'down'
        }
    )
    
    # EASE coordinates - these are MODEL INPUT FEATURES
    ds['EASE_X'] = xr.DataArray(
        grid['ease_x'],
        dims=['eta_rho', 'xi_rho'],
        attrs={
            'long_name': 'EASE-Grid X coordinate (model input feature)',
            'units': 'm',
            'description': 'X coordinate in EASE projection for each ROMS point - used as model input'
        }
    )
    
    ds['EASE_Y'] = xr.DataArray(
        grid['ease_y'],
        dims=['eta_rho', 'xi_rho'],
        attrs={
            'long_name': 'EASE-Grid Y coordinate (model input feature)',
            'units': 'm',
            'description': 'Y coordinate in EASE projection for each ROMS point - used as model input'
        }
    )
    
    # Day of year
    ds['DOY'] = xr.DataArray(
        [doy],
        dims=['time'],
        attrs={
            'long_name': 'Day of Year',
            'units': '1',
            'valid_range': [1, 366]
        }
    )
    
    # Global attributes
    ds.attrs = {
        'title': 'Arctic Model Input Data for LSTM Profile Reconstruction (ROMS Grid)',
        'institution': 'CNR-ISMAR',
        'source': 'Combined satellite and GLORYS data sampled at ROMS grid points',
        'history': f'Created {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        'references': 'lstm_pytorch_pd_mcdo.py',
        'Conventions': 'CF-1.8',
        'grid_type': 'ROMS curvilinear grid',
        'time_interpolation_method': 'linear',
        'time_reference': target_time.strftime('%Y-%m-%d %H:%M:%S'),
        'note': 'EASE_X and EASE_Y are model input features, not the output grid'
    }
    
    return ds


# ============================================================================
# MAIN PROCESSING
# ============================================================================


def generate_monthly_timesteps(year):
    """Generate target timesteps: 15th of each month at 12:00."""
    timesteps = []
    for month in range(1, 13):
        target = datetime(year, month, 15, 12, 0, 0)
        timesteps.append(target)
    return timesteps


def process_single_month(target_time, grid):
    """Process data for a single monthly timestep."""
    year = target_time.year
    month = target_time.month
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Processing {target_time.strftime('%Y-%m-%d')}")
    logger.info(f"{'='*60}")
    
    # Load satellite data
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
    
    # Load GLORYS
    logger.info("Loading GLORYS 3D data...")
    thetao, so, SH, depth = load_glorys_for_month(year, month)
    if thetao is None:
        logger.error(f"Failed to load GLORYS for {year}-{month:02d}")
        return None
    logger.info(f"  GLORYS T shape: {thetao.shape}")
    logger.info(f"  Depth levels: {len(depth)}")
    
    # Create output
    logger.info("Creating output dataset...")
    ds = create_output_dataset(
        target_time, grid, depth, sst, sss, adt, thetao, so, SH
    )
    
    return ds


def main():
    """Main processing pipeline."""
    logger.info("="*70)
    logger.info("Arctic Model Input Builder (ROMS Grid)")
    logger.info(f"Processing years: {START_YEAR} to {END_YEAR}")
    logger.info("="*70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {OUTPUT_DIR}")
    
    # Load ROMS grid
    grid = load_roms_grid_reference()
    
    # Process each year
    total_processed = 0
    total_failed = 0
    
    for year in range(START_YEAR, END_YEAR + 1):
        logger.info(f"\n{'#'*70}")
        logger.info(f"# YEAR {year}")
        logger.info(f"{'#'*70}")
        
        timesteps = generate_monthly_timesteps(year)
        
        for target_time in timesteps:
            try:
                ds = process_single_month(target_time, grid)
                
                if ds is not None:
                    output_file = OUTPUT_DIR / f'model_input_{year}_{target_time.month:02d}.nc'
                    
                    encoding = {
                        'time': {'dtype': 'float64'},
                        'SST': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
                        'SSS': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
                        'ADT': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
                        'T_glorys': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
                        'S_glorys': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
                        'SH_glorys': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
                        'lat_rho': {'dtype': 'float32'},
                        'lon_rho': {'dtype': 'float32'},
                        'mask_rho': {'dtype': 'int8'},
                        'h': {'dtype': 'float32'},
                        'EASE_X': {'dtype': 'float32'},
                        'EASE_Y': {'dtype': 'float32'},
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
    
    logger.info("\n" + "="*70)
    logger.info("PROCESSING COMPLETE")
    logger.info(f"  Files created: {total_processed}")
    logger.info(f"  Files failed: {total_failed}")
    logger.info("="*70)


if __name__ == '__main__':
    main()
