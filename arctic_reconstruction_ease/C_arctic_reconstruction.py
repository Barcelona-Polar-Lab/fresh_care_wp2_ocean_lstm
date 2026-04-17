#!/usr/bin/env python3
"""
Step C — Arctic LSTM Reconstruction Pipeline

For every target date the script:
    1. Loads static data (mask, bathymetry, lat/lon, X_EASE/Y_EASE)
    2. Loads & time-interpolates satellite surface fields (SST, SSS, ADT)
    3. Regrids GLORYS reanalysis for that date (via C_glorys_to_EASE)
    4. Builds per-profile model input  (7 features × n_depths)
    5. Runs MC Dropout predictions     (outputs: T_anom, S_anom)
    6. Regrids profiles back to the EASE grid
    7. Reconstructs full T/S = anomaly + GLORYS reference
    8. Saves one NetCDF file per date

Usage:
    python C_arctic_reconstruction.py --config my_config.yaml
"""

import os
import sys
import argparse
import numpy as np
import xarray as xr
from pathlib import Path
from datetime import datetime, timedelta
from glob import glob
import logging
import warnings

warnings.filterwarnings("ignore")

# Parent dir for lstm_pytorch_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lstm_pytorch_utils import load_model_checkpoint, mc_dropout_predict_chunked

from scipy.interpolate import RegularGridInterpolator

from config_utils import (
    load_config,
    create_ease_grid,
    get_proj4_string,
    get_resolution_label,
    get_woa_target_depths,
    get_target_dates,
    get_glorys_file_for_date,
    get_satellite_ease_dirs,
    get_static_data_path,
    get_reconstruction_dir,
    resolve_glorys_mode,
    check_time_range,
    create_transformers,
    compute_latlon_grids,
    get_ease_latlon_bbox,
    load_pipeline_plan,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def _build_encoding(ds):
    """
    Build per-variable netCDF encoding with int16 quantization for float32
    variables and zlib=4 compression for all variables.

    float32 → int16 mapping:
        scale_factor = (vmax - vmin) / 65534
        add_offset   = vmin + scale_factor * 32767
        _FillValue   = -32768  (int16 min, reserved for NaN/masked)
    Quantization error ≤ scale_factor / 2, which is well below
    observational uncertainty for all oceanographic fields here.
    """
    encoding = {}
    CHUNK_T, CHUNK_D, CHUNK_XY = 1, 17, 50

    for v in ds.data_vars:
        da = ds[v]
        if v == 'ease_grid_mapping':
            encoding[v] = {'zlib': False}
            continue

        ndim = da.ndim
        if ndim == 4:
            chunks = (CHUNK_T, CHUNK_D, CHUNK_XY, CHUNK_XY)
        elif ndim == 3:
            chunks = (CHUNK_T, CHUNK_XY, CHUNK_XY)
        elif ndim == 2:
            chunks = (CHUNK_XY, CHUNK_XY)
        else:
            encoding[v] = {'zlib': False}
            continue

        base = {'zlib': True, 'complevel': 4, 'shuffle': True,
                'chunksizes': chunks}

        if da.dtype == np.float32:
            arr = da.values
            vmin = float(np.nanmin(arr))
            vmax = float(np.nanmax(arr))
            if np.isfinite(vmin) and np.isfinite(vmax) and vmax > vmin:
                sf = (vmax - vmin) / 65534.0
                offset = vmin + sf * 32767.0
                base.update({
                    'dtype': 'int16',
                    'scale_factor': sf,
                    'add_offset': offset,
                    '_FillValue': np.iinfo(np.int16).min,
                })
            else:
                base['dtype'] = 'float32'
        else:
            base['dtype'] = str(da.dtype)

        encoding[v] = base

    return encoding

# GLORYS variables to regrid (3-D: depth × lat × lon)
_GLORYS_VARS_3D = ['thetao', 'so']

# Satellite variable names used by the reconstruction
_SAT_VARS = {'SST': 'analysed_sst', 'SSS': 'sss', 'ADT': 'adt'}


# ============================================================================
# GLORYS REGRIDDING  (absorbed from C_glorys_to_EASE.py)
# ============================================================================

def _glorys_interpolate_3d(data_3d, depth_orig, lat_orig, lon_orig,
                           depth_target, x_ease, y_ease,
                           transformer_from_ease, fill_value=np.nan):
    """Interpolate (depth, lat, lon) → (depth_target, y_ease, x_ease).

    *data_3d*, *depth_orig*, *lat_orig*, *lon_orig* must all be sorted
    ascending.
    """
    x2d, y2d = np.meshgrid(x_ease, y_ease)
    lon_tgt, lat_tgt = transformer_from_ease.transform(x2d, y2d)

    data_work = np.where(np.isnan(data_3d), fill_value, data_3d)

    # Extrapolate to depth 0 if the shallowest GLORYS level is > 0
    if depth_target.min() < depth_orig.min():
        d0, d1 = depth_orig[0], depth_orig[1]
        slope = (data_work[1] - data_work[0]) / (d1 - d0)
        layer0 = data_work[0] + slope * (0.0 - d0)
        data_work = np.concatenate([layer0[np.newaxis], data_work], axis=0)
        depth_orig = np.concatenate([[0.0], depth_orig])

    max_d = depth_orig.max()

    try:
        interp = RegularGridInterpolator(
            (depth_orig, lat_orig, lon_orig), data_work,
            method='linear', bounds_error=False, fill_value=fill_value,
        )
        n_d, n_y, n_x = len(depth_target), len(y_ease), len(x_ease)
        out = np.full((n_d, n_y, n_x), fill_value, dtype=np.float32)

        for di, dv in enumerate(depth_target):
            if dv > max_d:
                continue
            d2d = np.full_like(lat_tgt, dv)
            pts = np.stack([d2d, lat_tgt, lon_tgt], axis=-1)
            out[di] = interp(pts)

    except Exception as e:
        logger.warning(f"3-D interpolation failed: {e}")
        out = np.full((len(depth_target), len(y_ease), len(x_ease)),
                      fill_value, dtype=np.float32)
    return out


def regrid_glorys_single_timestep(glorys_path, cfg,
                                  x_ease=None, y_ease=None,
                                  gm_attrs=None,
                                  transformer_from_ease=None,
                                  bbox=None):
    """
    Regrid one GLORYS file to the target EASE grid + WOA depths.

    The file is opened lazily and spatially subsetted BEFORE loading
    data arrays into memory, so only the regional subset is read.

    Returns
    -------
    xr.Dataset
        Dataset with dims (depth, y_ease, x_ease) and variables
        ``thetao``, ``so``, plus coordinates and grid mapping.
    """
    target_depths = get_woa_target_depths()

    if x_ease is None or y_ease is None or gm_attrs is None:
        x_ease, y_ease, gm_attrs = create_ease_grid(cfg)
    if transformer_from_ease is None:
        _, transformer_from_ease = create_transformers(cfg)

    # Open lazily — coordinate arrays are read, data arrays stay on disk
    ds = xr.open_dataset(glorys_path)

    # Squeeze time dimension (one timestep per call)
    if 'time' in ds.dims:
        ds = ds.isel(time=0)

    # Normalise longitude to [-180, 180]
    lon_raw = ds['longitude'].values
    if lon_raw.max() > 180:
        ds = ds.assign_coords(
            longitude=('longitude', np.where(lon_raw > 180, lon_raw - 360, lon_raw))
        )

    # Ensure coordinates are sorted ascending (cheap index re-order)
    for dim in ('depth', 'latitude', 'longitude'):
        if dim in ds.dims:
            vals = ds[dim].values
            if len(vals) > 1 and vals[0] > vals[-1]:
                ds = ds.isel({dim: slice(None, None, -1)})

    # Spatial subsetting BEFORE loading data into memory
    if bbox is not None:
        n_lat_orig = ds.dims['latitude']
        n_lon_orig = ds.dims['longitude']
        ds = ds.sel(
            latitude=slice(bbox['lat_min'], bbox['lat_max']),
            longitude=slice(bbox['lon_min'], bbox['lon_max']),
        )
        logger.info(f"  GLORYS subset: lat {n_lat_orig}→{ds.dims['latitude']}, "
                    f"lon {n_lon_orig}→{ds.dims['longitude']}")

    # Extract sorted coordinate arrays from the (now small) subset
    lat = ds['latitude'].values.astype(np.float64)
    lon = ds['longitude'].values.astype(np.float64)
    depth = ds['depth'].values.astype(np.float64)

    coords = {
        'depth': ('depth', target_depths, {
            'standard_name': 'depth', 'units': 'm',
            'positive': 'down', 'axis': 'Z',
        }),
        'y_ease': ('y_ease', y_ease, {
            'standard_name': 'projection_y_coordinate',
            'units': 'm', 'axis': 'Y',
        }),
        'x_ease': ('x_ease', x_ease, {
            'standard_name': 'projection_x_coordinate',
            'units': 'm', 'axis': 'X',
        }),
    }

    data_vars = {}
    for var_name in _GLORYS_VARS_3D:
        if var_name not in ds.data_vars:
            logger.warning(f"Variable {var_name} not in {glorys_path}")
            continue

        # NOW loads data — but only the spatial subset
        raw = ds[var_name].values  # (depth_sub, lat_sub, lon_sub)

        ease = _glorys_interpolate_3d(
            raw, depth, lat, lon, target_depths,
            x_ease, y_ease, transformer_from_ease,
        )
        attrs = {k: v for k, v in ds[var_name].attrs.items()
                 if k not in ('scale_factor', 'add_offset', '_FillValue', 'missing_value')}
        attrs['grid_mapping'] = 'ease_grid_mapping'
        data_vars[var_name] = (['depth', 'y_ease', 'x_ease'], ease, attrs)
        del raw

    ds.close()

    ds_out = xr.Dataset(data_vars, coords=coords)
    ds_out['ease_grid_mapping'] = xr.DataArray(data=0, attrs=gm_attrs)
    return ds_out


# ============================================================================
# SATELLITE DATA LOADING  (absorbed from D_build_model_input.py)
# ============================================================================

def _find_satellite_files(base_dir, target_time, window_days, var_kind):
    """
    Gather satellite NetCDF files within ±window_days of *target_time*.

    *var_kind* is one of 'SST', 'SSS', 'ADT'.
    The directory layout is auto-detected:
        SST:  base_dir/YYYY/MM/*.nc  (daily files with date in name)
        SSS:  base_dir/sss_*_YYYY_*.nc  (yearly files)
        ADT:  base_dir/YYYY/*.nc  (daily files with YYYYMMDD in name)
    """
    start = (target_time - timedelta(days=window_days)).replace(hour=0, minute=0, second=0)
    end = (target_time + timedelta(days=window_days)).replace(hour=23, minute=59, second=59)

    files = []

    if var_kind == 'SST':
        current = start
        while current <= end:
            y, m, d = current.year, current.month, current.day
            pat = base_dir / str(y) / f'{m:02d}' / f'*_{y}-{m:02d}-{d:02d}_*.nc'
            files.extend(glob(str(pat)))
            current += timedelta(days=1)

    elif var_kind == 'SSS':
        years = set()
        current = start
        while current <= end:
            years.add(current.year)
            current += timedelta(days=1)
        for y in sorted(years):
            p = base_dir / f'sss_merge_cci_{y}_EASE_filled_wg.nc'
            if p.exists():
                files.append(str(p))

    elif var_kind == 'ADT':
        years = set()
        current = start
        while current <= end:
            years.add(current.year)
            current += timedelta(days=1)
        for y in sorted(years):
            ydir = base_dir / str(y)
            if not ydir.is_dir():
                continue
            for f in sorted(ydir.glob('*.nc')):
                parts = f.stem.split('_')
                for part in parts:
                    if len(part) == 8 and part.isdigit():
                        try:
                            fd = datetime.strptime(part, '%Y%m%d')
                            if start <= fd <= end:
                                files.append(str(f))
                        except ValueError:
                            pass
                        break

    return sorted(set(files))


def load_satellite_for_time(base_dir, target_time, window_days, var_kind,
                            var_name):
    """
    Load satellite data in a ±window around *target_time* and linearly
    interpolate to the exact target date.

    Returns a 2-D numpy array (y_ease, x_ease) or None.
    """
    files = _find_satellite_files(base_dir, target_time, window_days, var_kind)
    if not files:
        logger.warning(f"No {var_kind} files for {target_time:%Y-%m-%d}")
        return None

    start = (target_time - timedelta(days=window_days)).replace(hour=0, minute=0, second=0)
    end = (target_time + timedelta(days=window_days)).replace(hour=23, minute=59, second=59)
    tgt = np.datetime64(target_time)

    # Open lazily, select only the needed variable and time window,
    # then load into memory — avoids reading entire yearly files.
    chunks = []
    for f in files:
        ds = xr.open_dataset(f)
        if 'time' in ds.dims:
            ds = ds.sel(time=slice(start, end))
        chunk = ds[var_name].load()      # load the subset into memory
        ds.close()
        if chunk.sizes.get('time', 1) > 0:
            chunks.append(chunk)

    if not chunks:
        logger.warning(f"No {var_kind} data in window for {target_time:%Y-%m-%d}")
        return None

    da = xr.concat(chunks, dim='time').sortby('time')
    result = da.interp(time=tgt, method='linear').values
    return result


# ============================================================================
# GRID ↔ PROFILE CONVERSION
# ============================================================================

def grid_to_profiles(ocean_mask, SST, SSS, ADT,
                     T_glorys_surf, S_glorys_surf,
                     X_EASE_2d, Y_EASE_2d, DOY, n_depths):
    """
    Extract ocean pixels and build the model-input array.

    Input feature order (must match model training):
        [sst_anomaly, sss_anomaly, seasonal_cos, seasonal_sin,
         adt, x_ease, y_ease]

    Returns
    -------
    dict with keys: X, y_idx, x_idx, x_ease_vals, y_ease_vals
    """
    y_idx, x_idx = np.where(ocean_mask == 1)
    n_prof = len(y_idx)

    sst_anom = SST[y_idx, x_idx] - T_glorys_surf[y_idx, x_idx]
    sss_anom = SSS[y_idx, x_idx] - S_glorys_surf[y_idx, x_idx]
    adt_vals = ADT[y_idx, x_idx]          # raw ADT — no SH correction
    x_vals = X_EASE_2d[y_idx, x_idx]
    y_vals = Y_EASE_2d[y_idx, x_idx]

    phase = 2.0 * np.pi * (DOY / 365.0) + 1.0
    s_cos = np.cos(phase)
    s_sin = np.sin(phase)

    # (n_profiles, n_depths, 7)
    X = np.zeros((n_prof, n_depths, 7), dtype=np.float32)
    X[:, :, 0] = sst_anom[:, None]
    X[:, :, 1] = sss_anom[:, None]
    X[:, :, 2] = s_cos               # scalar broadcast
    X[:, :, 3] = s_sin
    X[:, :, 4] = adt_vals[:, None]
    X[:, :, 5] = x_vals[:, None]
    X[:, :, 6] = y_vals[:, None]

    nan_prof = np.any(np.isnan(X), axis=(1, 2)).sum()
    if nan_prof:
        logger.info(f"  {nan_prof}/{n_prof} profiles contain NaN inputs")

    return {
        'X': X,
        'y_idx': y_idx,
        'x_idx': x_idx,
        'x_ease_vals': x_vals,
        'y_ease_vals': y_vals,
    }


def regrid_profiles_to_grid(y_mean, y_std,
                            y_idx, x_idx, n_y, n_x, n_depths):
    """
    Map flat profile arrays back onto (depth, y_ease, x_ease) grids.

    Returns dict of 3-D arrays keyed by variable name.
    """
    def _empty():
        return np.full((n_depths, n_y, n_x), np.nan, dtype=np.float32)

    grids = {
        'T_anom_pred': _empty(), 'S_anom_pred': _empty(),
        'T_anom_std':  _empty(), 'S_anom_std':  _empty(),
    }

    for d in range(n_depths):
        grids['T_anom_pred'][d, y_idx, x_idx] = y_mean[:, d, 0]
        grids['S_anom_pred'][d, y_idx, x_idx] = y_mean[:, d, 1]
        grids['T_anom_std'][d, y_idx, x_idx]  = y_std[:, d, 0]
        grids['S_anom_std'][d, y_idx, x_idx]  = y_std[:, d, 1]

    return grids


# ============================================================================
# SINGLE-TIMESTEP RECONSTRUCTION → NetCDF
# ============================================================================

def reconstruct_single_date(target_date, cfg, model, norm_params,
                            static_ds, x_ease, y_ease, gm_attrs,
                            sat_dirs, tf_from, device, bbox=None,
                            overwrite=False):
    """
    Full reconstruction for one date.  Returns the output path or None.
    """
    import torch
    date_str = target_date.strftime('%Y%m%d')
    out_dir = get_reconstruction_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"reconstruction_{date_str}.nc"

    if out_file.exists() and not overwrite:
        logger.info(f"  {out_file.name} exists — skipping")
        return out_file

    n_depths = len(get_woa_target_depths())
    n_y, n_x = len(y_ease), len(x_ease)

    # --- Static fields ---
    ocean_mask = static_ds['ocean_mask'].values          # (ny, nx)
    X_EASE_2d = np.broadcast_to(x_ease[None, :], (n_y, n_x)).copy()
    Y_EASE_2d = np.broadcast_to(y_ease[:, None], (n_y, n_x)).copy()

    # --- GLORYS regridding ---
    glorys_path = get_glorys_file_for_date(cfg, target_date)
    if glorys_path is None:
        logger.error(f"  No GLORYS file for {date_str}")
        return None

    logger.info(f"  [{date_str}] Regridding GLORYS: {glorys_path.name}")
    ds_glorys = regrid_glorys_single_timestep(
        glorys_path, cfg, x_ease, y_ease, gm_attrs, tf_from, bbox=bbox,
    )
    T_glorys = ds_glorys['thetao'].values   # (depth, ny, nx)
    S_glorys = ds_glorys['so'].values
    ds_glorys.close()
    logger.info(f"  [{date_str}] GLORYS regridding done")

    T_glorys_surf = T_glorys[0]   # (ny, nx)
    S_glorys_surf = S_glorys[0]

    # --- Satellite surface data ---
    logger.info(f"  [{date_str}] Loading satellite surface data")
    window = cfg['processing']['time_window_days']
    SST = load_satellite_for_time(
        sat_dirs['SST'], target_date, window, 'SST', _SAT_VARS['SST'])
    SSS = load_satellite_for_time(
        sat_dirs['SSS'], target_date, window, 'SSS', _SAT_VARS['SSS'])
    ADT = load_satellite_for_time(
        sat_dirs['ADT'], target_date, window, 'ADT', _SAT_VARS['ADT'])

    if SST is None or SSS is None or ADT is None:
        logger.error(f"  Missing satellite data for {date_str}")
        return None

    # SST: Kelvin → Celsius
    SST = SST - 273.15

    DOY = target_date.timetuple().tm_yday

    # --- Build profiles ---
    logger.info(f"  [{date_str}] Building profiles")
    prof = grid_to_profiles(
        ocean_mask, SST, SSS, ADT,
        T_glorys_surf, S_glorys_surf,
        X_EASE_2d, Y_EASE_2d, DOY, n_depths,
    )
    n_prof = prof['X'].shape[0]
    logger.info(f"  [{date_str}] {n_prof} ocean profiles, "
                f"MC dropout ({cfg['processing']['n_mc_samples']} samples)")

    # --- MC Dropout predictions ---
    y_mean, y_std = mc_dropout_predict_chunked(
        model=model,
        X=prof['X'],
        norm_params=norm_params,
        n_mc_samples=cfg['processing']['n_mc_samples'],
        chunk_size=cfg['processing']['chunk_size'],
        device=device,
        show_progress=False,
        show_mc_progress=False,
    )
    del prof['X']  # free profile input array (keep y_idx, x_idx)
    # y_mean shape: (n_prof, n_depths, 2)  → index 0 = T_anom, 1 = S_anom

    # --- Regrid to grid ---
    logger.info(f"  [{date_str}] Regridding predictions to grid & saving")
    grids = regrid_profiles_to_grid(
        y_mean, y_std,
        prof['y_idx'], prof['x_idx'],
        n_y, n_x, n_depths,
    )
    del prof, y_mean, y_std

    # --- NaN mask: where GLORYS T or S is NaN (below seabed) ---
    glorys_nan = np.isnan(T_glorys) | np.isnan(S_glorys)
    for k in grids:
        grids[k] = np.where(glorys_nan, np.nan, grids[k])

    # --- Reconstruct full profiles: anomaly + GLORYS ---
    T_recon = grids['T_anom_pred'] + T_glorys
    S_recon = grids['S_anom_pred'] + S_glorys

    # --- Build output dataset ---
    depth = get_woa_target_depths()
    time_val = np.datetime64(target_date.strftime('%Y-%m-%dT%H:%M:%S'))

    coords = {
        'time':   ('time',   [time_val]),
        'depth':  ('depth',  depth, {'units': 'm', 'positive': 'down'}),
        'y_ease': ('y_ease', y_ease, {'units': 'm', 'axis': 'Y'}),
        'x_ease': ('x_ease', x_ease, {'units': 'm', 'axis': 'X'}),
    }

    def _da(arr, long_name, units):
        return xr.DataArray(
            arr[np.newaxis],   # add time dim → (1, depth, ny, nx)
            dims=['time', 'depth', 'y_ease', 'x_ease'],
            attrs={'long_name': long_name, 'units': units,
                   'grid_mapping': 'ease_grid_mapping'},
        )

    def _da2d(arr, long_name, units):
        return xr.DataArray(
            arr[np.newaxis],
            dims=['time', 'y_ease', 'x_ease'],
            attrs={'long_name': long_name, 'units': units,
                   'grid_mapping': 'ease_grid_mapping'},
        )

    ds_out = xr.Dataset(coords=coords)

    # Predicted anomalies
    ds_out['T_anom_pred'] = _da(grids['T_anom_pred'], 'Predicted temperature anomaly', 'degrees_C')
    ds_out['S_anom_pred'] = _da(grids['S_anom_pred'], 'Predicted salinity anomaly', 'PSU')

    # Uncertainty
    ds_out['T_anom_std'] = _da(grids['T_anom_std'], 'Temperature anomaly std', 'degrees_C')
    ds_out['S_anom_std'] = _da(grids['S_anom_std'], 'Salinity anomaly std', 'PSU')

    # Reconstructed full profiles
    ds_out['T_recon'] = _da(T_recon, 'Reconstructed temperature (anom + GLORYS)', 'degrees_C')
    ds_out['S_recon'] = _da(S_recon, 'Reconstructed salinity (anom + GLORYS)', 'PSU')

    # GLORYS reference
    ds_out['T_glorys'] = _da(T_glorys, 'GLORYS temperature', 'degrees_C')
    ds_out['S_glorys'] = _da(S_glorys, 'GLORYS salinity', 'PSU')

    # Surface satellite inputs
    ds_out['SST'] = _da2d(SST, 'Sea surface temperature (satellite)', 'degrees_C')
    ds_out['SSS'] = _da2d(SSS, 'Sea surface salinity (satellite)', 'PSU')
    ds_out['ADT'] = _da2d(ADT, 'Absolute dynamic topography (satellite)', 'm')

    # DOY
    ds_out['DOY'] = xr.DataArray([DOY], dims=['time'],
                                 attrs={'long_name': 'Day of year'})

    # Static (copy from static_ds)
    for vname in ('ocean_mask', 'elevation', 'latitude', 'longitude'):
        if vname in static_ds:
            ds_out[vname] = static_ds[vname]

    ds_out['ease_grid_mapping'] = xr.DataArray(data=0, attrs=gm_attrs)

    ds_out.attrs = {
        'title': f'Arctic reconstruction ({get_resolution_label(cfg)})',
        'source': 'LSTM MC Dropout + GLORYS reanalysis',
        'history': f'Created {datetime.now():%Y-%m-%d %H:%M:%S}',
        'conventions': 'CF-1.8',
        'grid_resolution': f"{cfg['grid']['resolution_km']} km",
        'proj4_string': get_proj4_string(cfg),
        'n_mc_samples': cfg['processing']['n_mc_samples'],
    }

    # Encoding — int16 quantization halves file size with negligible precision loss
    encoding = _build_encoding(ds_out)

    ds_out.to_netcdf(out_file, encoding=encoding)
    ds_out.close()
    del grids, T_recon, S_recon, T_glorys, S_glorys
    logger.info(f"  [{date_str}] Saved → {out_file.name}")
    return out_file


# ============================================================================
# CLI / MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Step C — Arctic LSTM Reconstruction",
    )
    parser.add_argument('--config', required=True,
                        help='Path to the pipeline YAML config file')
    args = parser.parse_args()

    cfg = load_config(args.config)
    resolve_glorys_mode(cfg)
    check_time_range(cfg)
    plan = load_pipeline_plan(cfg)
    overwrite = plan.get('overwrite_reconstruction', False)

    label = get_resolution_label(cfg)
    proj4 = get_proj4_string(cfg)
    x_ease, y_ease, gm_attrs = create_ease_grid(cfg)
    _, tf_from = create_transformers(cfg)
    bbox = get_ease_latlon_bbox(cfg, pad_deg=2.0)

    logger.info("=" * 60)
    logger.info("Arctic LSTM Reconstruction Pipeline")
    logger.info(f"  Grid: {cfg['grid']['n_cells_x']}×{cfg['grid']['n_cells_y']} "
                f"@ {cfg['grid']['resolution_km']} km ({label})")
    logger.info(f"  Spatial bbox: lat=[{bbox['lat_min']:.1f}, {bbox['lat_max']:.1f}], "
                f"lon=[{bbox['lon_min']:.1f}, {bbox['lon_max']:.1f}]")
    logger.info(f"  GLORYS mode: {cfg['processing']['glorys_mode']}")
    logger.info(f"  Time range: {cfg['time']['start_month']} → {cfg['time']['end_month']}")
    logger.info("=" * 60)

    # --- Load static data ---
    static_path = get_static_data_path(cfg)
    if not static_path.exists():
        logger.error(f"Static data not found: {static_path}")
        logger.error("Run A_create_ocean_mask.py first.")
        sys.exit(1)
    static_ds = xr.open_dataset(static_path)
    logger.info(f"Loaded static data from {static_path}")

    # --- Load model ---
    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, norm_params, model_config, input_names = load_model_checkpoint(
        cfg['paths']['model_path'], device,
    )
    logger.info(f"Model outputs: {model_config['output_size']}  "
                f"({', '.join(input_names or [])})")

    # --- Satellite directories ---
    sat_dirs = get_satellite_ease_dirs(cfg)

    # --- Process each target date ---
    dates = get_target_dates(cfg)
    logger.info(f"\nProcessing {len(dates)} target dates\n")

    ok = errors = skipped = 0
    n_dates = len(dates)
    for i, dt in enumerate(dates, 1):
        logger.info(f"\n--- Date {i}/{n_dates}: {dt:%Y-%m-%d} ---")
        result = reconstruct_single_date(
            dt, cfg, model, norm_params,
            static_ds, x_ease, y_ease, gm_attrs,
            sat_dirs, tf_from, device, bbox=bbox,
            overwrite=overwrite,
        )
        if result is None:
            errors += 1
        else:
            ok += 1
        logger.info(f"  Progress: {i}/{n_dates} ({ok} ok, {errors} errors)")

    static_ds.close()

    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info(f"  Successful: {ok}")
    logger.info(f"  Errors:     {errors}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
