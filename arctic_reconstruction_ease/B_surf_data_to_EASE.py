#!/usr/bin/env python3
"""
Step B — Regrid satellite surface data (SST, SSS, ADT) from lat/lon to EASE grid.

Uses bilinear interpolation via scipy RegularGridInterpolator.
Output directories are labelled with the target resolution
(e.g. ``data_ease_25km/``).

Usage:
    python B_surf_data_to_EASE.py --config my_config.yaml [--force]
"""

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from pathlib import Path
import pyproj
import logging
import argparse
from datetime import datetime

from config_utils import (
    load_config,
    create_ease_grid,
    get_proj4_string,
    get_resolution_label,
    get_satellite_ease_dirs,
    get_ease_latlon_bbox,
    get_satellite_date_range,
    load_pipeline_plan,
    filter_files_by_date_range,
    build_global_attrs,
    format_eta,
)

# ============================================================================
# LOGGING
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def has_latlon_coords(ds):
    """Return (True, lat_name, lon_name) if *ds* has recognisable 1-D lat/lon."""
    lat_name = lon_name = None
    for name in ('lat', 'latitude', 'LAT', 'LATITUDE'):
        if name in ds.dims:
            lat_name = name
            break
        # Accept 1-D coordinate variables (not 2-D auxiliary coords)
        if name in ds.coords and ds.coords[name].ndim == 1:
            lat_name = name
            break
    for name in ('lon', 'longitude', 'LON', 'LONGITUDE'):
        if name in ds.dims:
            lon_name = name
            break
        if name in ds.coords and ds.coords[name].ndim == 1:
            lon_name = name
            break
    if lat_name and lon_name:
        return True, lat_name, lon_name
    return False, None, None


def has_ease_coords(ds):
    """Return True if *ds* uses 1-D x_ease / y_ease dimensions."""
    return ('x_ease' in ds.dims and 'y_ease' in ds.dims)


def interpolate_to_ease(data_2d, lat_orig, lon_orig,
                        x_ease_target, y_ease_target, proj4,
                        fill_value=np.nan):
    """
    Bilinear interpolation of a 2-D lat/lon field onto an EASE grid.

    Parameters
    ----------
    data_2d : np.ndarray (lat, lon)
    lat_orig, lon_orig : 1-D sorted arrays
    x_ease_target, y_ease_target : 1-D EASE cell-center arrays
    proj4 : str  — PROJ4 string for the target EASE CRS
    fill_value : float

    Returns
    -------
    np.ndarray (ny, nx)
    """
    crs_ease = pyproj.CRS.from_proj4(proj4)
    crs_wgs84 = pyproj.CRS.from_epsg(4326)
    inv_tf = pyproj.Transformer.from_crs(crs_ease, crs_wgs84, always_xy=True)

    x2d, y2d = np.meshgrid(x_ease_target, y_ease_target)
    lon_tgt, lat_tgt = inv_tf.transform(x2d, y2d)

    try:
        interp = RegularGridInterpolator(
            (lat_orig, lon_orig), data_2d,
            method='linear', bounds_error=False, fill_value=fill_value,
        )
        pts = np.stack([lat_tgt, lon_tgt], axis=-1)
        return interp(pts)
    except Exception as e:
        logger.warning(f"Interpolation failed: {e}. Returning fill values.")
        return np.full((len(y_ease_target), len(x_ease_target)), fill_value)


def interpolate_ease_to_ease(data_2d, x_src, y_src,
                            x_tgt, y_tgt, fill_value=np.nan):
    """
    Bilinear interpolation from one EASE grid to another.

    Parameters
    ----------
    data_2d : np.ndarray (y_src, x_src)
    x_src, y_src : 1-D sorted arrays (source EASE coordinates in metres)
    x_tgt, y_tgt : 1-D sorted arrays (target EASE coordinates)
    fill_value : float

    Returns
    -------
    np.ndarray (len(y_tgt), len(x_tgt))
    """
    try:
        interp = RegularGridInterpolator(
            (y_src, x_src), data_2d,
            method='linear', bounds_error=False, fill_value=fill_value,
        )
        y2d, x2d = np.meshgrid(y_tgt, x_tgt, indexing='ij')
        pts = np.stack([y2d, x2d], axis=-1)
        return interp(pts)
    except Exception as e:
        logger.warning(f"EASE→EASE interpolation failed: {e}. Returning fill values.")
        return np.full((len(y_tgt), len(x_tgt)), fill_value)


def process_variable(var_data, lat_orig, lon_orig,
                     x_ease_target, y_ease_target, proj4,
                     time_dim=None, flip_lat=False, flip_lon=False,
                     lat_slice=None, lon_slice=None):
    """Interpolate one variable, loading one time step at a time."""

    def _load_2d(arr):
        """Apply flip + spatial subset to a 2-D (lat, lon) numpy array."""
        if flip_lat:
            arr = np.flip(arr, axis=0)
        if flip_lon:
            arr = np.flip(arr, axis=1)
        if lat_slice is not None and lon_slice is not None:
            arr = arr[lat_slice, lon_slice]
        return arr

    if time_dim is not None and time_dim in var_data.dims:
        n_t = var_data.sizes[time_dim]
        out = np.zeros((n_t, len(y_ease_target), len(x_ease_target)))
        for t in range(n_t):
            data_t = var_data.isel({time_dim: t}).values   # one 2-D slice
            data_t = _load_2d(data_t)
            out[t] = interpolate_to_ease(
                data_t, lat_orig, lon_orig,
                x_ease_target, y_ease_target, proj4,
            )
        return out

    data = _load_2d(var_data.values)
    return interpolate_to_ease(
        data, lat_orig, lon_orig,
        x_ease_target, y_ease_target, proj4,
    )


def process_variable_ease(var_data, x_src, y_src,
                         x_tgt, y_tgt,
                         time_dim=None,
                         y_slice=None, x_slice=None):
    """Interpolate one variable from source EASE grid to target EASE grid."""

    def _load_2d(arr):
        if y_slice is not None and x_slice is not None:
            arr = arr[y_slice, x_slice]
        return arr

    if time_dim is not None and time_dim in var_data.dims:
        n_t = var_data.sizes[time_dim]
        out = np.zeros((n_t, len(y_tgt), len(x_tgt)))
        for t in range(n_t):
            data_t = var_data.isel({time_dim: t}).values
            data_t = _load_2d(data_t)
            out[t] = interpolate_ease_to_ease(
                data_t, x_src, y_src, x_tgt, y_tgt,
            )
        return out

    data = _load_2d(var_data.values)
    return interpolate_ease_to_ease(
        data, x_src, y_src, x_tgt, y_tgt,
    )


# ============================================================================
# PER-FILE REGRIDDING
# ============================================================================

def regrid_file_to_ease(input_path, output_path, x_ease, y_ease,
                        grid_mapping_attrs, proj4, cfg, bbox=None,
                        target_vars=None):
    """
    Regrid one NetCDF file to the target EASE grid.

    Supports two input formats:
    - lat/lon grids  → bilinear interpolation via lat/lon → EASE
    - EASE grids (x_ease/y_ease dims) → bilinear EASE → EASE interpolation

    Returns True on success, False if skipped/errored.
    """
    try:
        ds = xr.open_dataset(input_path)
        ease_input = has_ease_coords(ds)
        has, lat_name, lon_name = has_latlon_coords(ds)

        if not ease_input and not has:
            logger.warning(f"  Skipping (no lat/lon or EASE coords): {input_path.name}")
            ds.close()
            return False

        time_dim = None
        for name in ('time', 'TIME', 'Time'):
            if name in ds.dims:
                time_dim = name
                break

        res_m = cfg['grid']['resolution_km'] * 1000.0

        # Build output coordinates
        coords = {
            'x_ease': ('x_ease', x_ease, {
                'standard_name': 'projection_x_coordinate',
                'units': 'm',
                'long_name': 'EASE-Grid X coordinate',
                'axis': 'X',
            }),
            'y_ease': ('y_ease', y_ease, {
                'standard_name': 'projection_y_coordinate',
                'units': 'm',
                'long_name': 'EASE-Grid Y coordinate',
                'axis': 'Y',
            }),
        }
        if time_dim is not None:
            coords[time_dim] = ds[time_dim]

        data_vars = {}

        if ease_input:
            # ---- EASE → EASE path ----
            x_src = ds['x_ease'].values.astype(np.float64)
            y_src = ds['y_ease'].values.astype(np.float64)

            # Ensure ascending
            if x_src[0] > x_src[-1]:
                x_src = x_src[::-1]
                ds = ds.isel(x_ease=slice(None, None, -1))
            if y_src[0] > y_src[-1]:
                y_src = y_src[::-1]
                ds = ds.isel(y_ease=slice(None, None, -1))

            # Spatial subsetting on EASE coordinates (with buffer)
            buf = 50_000  # 50 km buffer
            _y_sl = _x_sl = None
            y_sel = (y_src >= y_ease.min() - buf) & (y_src <= y_ease.max() + buf)
            x_sel = (x_src >= x_ease.min() - buf) & (x_src <= x_ease.max() + buf)
            if y_sel.sum() > 0 and x_sel.sum() > 0:
                yi = np.where(y_sel)[0]
                xi = np.where(x_sel)[0]
                _y_sl = slice(int(yi[0]), int(yi[-1]) + 1)
                _x_sl = slice(int(xi[0]), int(xi[-1]) + 1)
                y_src = y_src[_y_sl]
                x_src = x_src[_x_sl]

            vars_to_process = target_vars if target_vars is not None else list(ds.data_vars)
            for var_name in vars_to_process:
                if var_name not in ds.data_vars:
                    logger.warning(f"  Variable '{var_name}' not found in {input_path.name}")
                    continue
                var = ds[var_name]
                if 'y_ease' not in var.dims or 'x_ease' not in var.dims:
                    continue

                data_interp = process_variable_ease(
                    var, x_src, y_src, x_ease, y_ease,
                    time_dim=time_dim,
                    y_slice=_y_sl, x_slice=_x_sl,
                )
                dims = (time_dim, 'y_ease', 'x_ease') if (
                    time_dim and time_dim in var.dims) else ('y_ease', 'x_ease')
                attrs = dict(var.attrs)
                attrs['grid_mapping'] = 'ease_grid_mapping'
                data_vars[var_name] = (dims, data_interp, attrs)

        else:
            # ---- lat/lon → EASE path ----
            lat_orig = ds[lat_name].values
            lon_orig = ds[lon_name].values
            lon_orig = np.where(lon_orig > 180, lon_orig - 360, lon_orig)

            flip_lat = lat_orig[0] > lat_orig[-1]
            flip_lon = lon_orig[0] > lon_orig[-1]
            if flip_lat:
                lat_orig = lat_orig[::-1]
            if flip_lon:
                lon_orig = lon_orig[::-1]

            # Spatial subsetting — slice to bounding box before interpolation
            _lat_slice = _lon_slice = None
            if bbox is not None:
                lat_sel = (lat_orig >= bbox['lat_min']) & (lat_orig <= bbox['lat_max'])
                lon_sel = (lon_orig >= bbox['lon_min']) & (lon_orig <= bbox['lon_max'])
                if lat_sel.sum() > 0 and lon_sel.sum() > 0:
                    lat_idx = np.where(lat_sel)[0]
                    lon_idx = np.where(lon_sel)[0]
                    lat_slice = slice(int(lat_idx[0]), int(lat_idx[-1]) + 1)
                    lon_slice = slice(int(lon_idx[0]), int(lon_idx[-1]) + 1)
                    lat_orig = lat_orig[lat_slice]
                    lon_orig = lon_orig[lon_slice]
                    _lat_slice = lat_slice
                    _lon_slice = lon_slice

            vars_to_process = target_vars if target_vars is not None else list(ds.data_vars)
            for var_name in vars_to_process:
                if var_name not in ds.data_vars:
                    logger.warning(f"  Variable '{var_name}' not found in {input_path.name}")
                    continue
                var = ds[var_name]
                if lat_name not in var.dims or lon_name not in var.dims:
                    continue

                data_interp = process_variable(
                    var, lat_orig, lon_orig,
                    x_ease, y_ease, proj4,
                    time_dim=time_dim, flip_lat=flip_lat, flip_lon=flip_lon,
                    lat_slice=_lat_slice, lon_slice=_lon_slice,
                )
                dims = (time_dim, 'y_ease', 'x_ease') if (
                    time_dim and time_dim in var.dims) else ('y_ease', 'x_ease')
                attrs = dict(var.attrs)
                attrs['grid_mapping'] = 'ease_grid_mapping'
                data_vars[var_name] = (dims, data_interp, attrs)

        ds_out = xr.Dataset(data_vars, coords=coords)
        ds_out['ease_grid_mapping'] = xr.DataArray(data=0, attrs=grid_mapping_attrs)

        ds_out.attrs = dict(ds.attrs)
        ds_out.attrs.update(build_global_attrs(
            cfg,
            title=(ds.attrs.get('title', 'Surface data') + ' on EASE grid'),
            source=f'Bilinear interpolation from native grid to EASE {get_resolution_label(cfg)}',
            extra={
                'regridding_method': 'bilinear interpolation',
                'original_file': str(input_path.name),
            },
        ))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        encoding = {v: {'zlib': True, 'complevel': 4} for v in ds_out.data_vars}
        ds_out.to_netcdf(output_path, encoding=encoding)

        ds.close()
        ds_out.close()
        return True

    except Exception as e:
        logger.error(f"  Error processing {input_path.name}: {e}")
        return False


# ============================================================================
# DIRECTORY PROCESSING
# ============================================================================

def process_directory(input_dir, output_dir, dataset_name,
                      x_ease, y_ease, gm_attrs, proj4, cfg, bbox=None,
                      target_vars=None, date_range=None):
    """Iterate over *.nc files under *input_dir*, preserving subfolders."""
    nc_files = sorted(input_dir.rglob('*.nc'))
    total_raw = len(nc_files)

    if date_range is not None:
        nc_files = filter_files_by_date_range(nc_files, date_range[0], date_range[1])

    total = len(nc_files)
    if total != total_raw:
        logger.info(f"  {dataset_name}: {total} files to process "
                    f"(filtered from {total_raw} by date range)")
    else:
        logger.info(f"  {dataset_name}: {total} files to process")

    processed = reused = errors = 0
    import time as _time
    t0 = _time.monotonic()
    for i, nc_file in enumerate(nc_files, 1):
        rel = nc_file.relative_to(input_dir)
        out_file = output_dir / rel

        if out_file.exists():
            reused += 1
            continue

        ok = regrid_file_to_ease(nc_file, out_file, x_ease, y_ease,
                                 gm_attrs, proj4, cfg, bbox=bbox,
                                 target_vars=target_vars)
        if ok:
            processed += 1
        elif ok is False:
            errors += 1
        else:
            errors += 1

        if i % 100 == 0 or i == total:
            logger.info(f"  {dataset_name}: {i}/{total} "
                        f"({processed} new, {reused} reused) | {format_eta(t0, i, total)}")

    logger.info(f"  {dataset_name} done: {processed} processed, "
                f"{reused} existing, {errors} errors")
    return processed, reused, errors


# Hardcoded variable names to extract per dataset — only these are kept.
_DATASET_TARGET_VARS = {
    'SST': ['analysed_sst'],
    'SSS': ['sss'],
    'ADT': ['adt'],
}


# ============================================================================
# CLI / MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Step B — Regrid satellite data to EASE grid",
    )
    parser.add_argument('--config', required=True,
                        help='Path to the pipeline YAML config file')
    parser.add_argument('--force', action='store_true',
                        help='Force re-processing (ignore pipeline plan)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    plan = load_pipeline_plan(cfg)
    proj4 = get_proj4_string(cfg)
    label = get_resolution_label(cfg)
    x_ease, y_ease, gm_attrs = create_ease_grid(cfg)
    bbox = get_ease_latlon_bbox(cfg, pad_deg=2.0)
    date_range = get_satellite_date_range(cfg)

    logger.info("=" * 60)
    logger.info("Step B: Surface Data → EASE Grid")
    logger.info(f"  Grid: {cfg['grid']['n_cells_x']}×{cfg['grid']['n_cells_y']} "
                f"@ {cfg['grid']['resolution_km']} km  ({label})")
    logger.info(f"  Date range: {date_range[0]:%Y-%m-%d} → {date_range[1]:%Y-%m-%d}")
    logger.info("=" * 60)

    sat_dirs = get_satellite_ease_dirs(cfg)

    datasets = {
        'SST': (Path(cfg['paths']['sst_raw_dir']), sat_dirs['SST']),
        'SSS': (Path(cfg['paths']['sss_raw_dir']), sat_dirs['SSS']),
        'ADT': (Path(cfg['paths']['adt_raw_dir']), sat_dirs['ADT']),
    }

    total_p = total_r = total_e = 0
    for name, (in_dir, out_dir) in datasets.items():
        # Check pipeline plan
        if not args.force and not plan.get('run_step_B', {}).get(name, True):
            logger.info(f"{name}: reusing existing EASE data (per pipeline plan)")
            continue

        if not in_dir.exists():
            logger.warning(f"{name}: raw data directory missing — {in_dir}")
            continue

        p, r, e = process_directory(in_dir, out_dir, name,
                                    x_ease, y_ease, gm_attrs, proj4, cfg,
                                    bbox=bbox,
                                    target_vars=_DATASET_TARGET_VARS.get(name),
                                    date_range=date_range)
        total_p += p
        total_r += r
        total_e += e

    logger.info(f"Step B done — {total_p} processed, {total_r} existing, {total_e} errors")


if __name__ == '__main__':
    main()
