#!/usr/bin/env python3
"""
Step C — Per-date GLORYS surface (depth=0) on EASE grid.

Extracts only the surface layer of the daily GLORYS reanalysis,
regrids it to the EASE grid, and writes a small per-date NetCDF
under ``intermediate_files/glorys_surface/``. These files are the
input GLORYS reference required by Step D to build the satellite
anomalies fed to the LSTM (≈ 1 MB / file at 25 km).

By extracting just the surface here, Step D can run on a remote
GPU server without ever opening the raw 700 GB GLORYS archive.

Usage:
    python C_glorys_surface_to_EASE.py --config my_config.yaml [--force]
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

from config_utils import (
    atomic_to_netcdf,
    TIME_ENCODING,
    build_var_encoding,
    check_time_range,
    create_ease_grid,
    create_transformers,
    format_eta,
    get_ease_latlon_bbox,
    get_glorys_file_for_date,
    get_glorys_surface_dir,
    get_resolution_label,
    get_target_dates,
    load_config,
    load_pipeline_plan,
    resolve_glorys_mode,
)
from glorys_regrid import regrid_glorys_single_timestep

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def process_single_date(target_date, cfg, x_ease, y_ease, gm_attrs,
                        tf_from, bbox, out_dir, overwrite=False):
    out_file = out_dir / f"glorys_surf_{target_date:%Y%m%d}.nc"
    if out_file.exists() and not overwrite:
        return out_file, 'skip'

    glorys_path = get_glorys_file_for_date(cfg, target_date)
    if glorys_path is None:
        logger.error(f"  No GLORYS file for {target_date:%Y-%m-%d}")
        return None, 'error'

    ds = regrid_glorys_single_timestep(
        glorys_path, cfg,
        x_ease=x_ease, y_ease=y_ease, gm_attrs=gm_attrs,
        transformer_from_ease=tf_from, bbox=bbox,
        target_depths=[0.0],
    )

    # Drop the singleton depth axis and rename to the names Step C expects.
    T_surf = ds['thetao'].isel(depth=0).values  # (ny, nx)
    S_surf = ds['so'].isel(depth=0).values
    ds.close()

    time_val = np.datetime64(target_date.strftime('%Y-%m-%dT%H:%M:%S'))
    coords = {
        'time': ('time', [time_val], {
            'standard_name': 'time', 'long_name': 'Time', 'axis': 'T',
        }),
        'y_ease': ('y_ease', y_ease, {
            'standard_name': 'projection_y_coordinate',
            'long_name': 'EASE-Grid Y coordinate',
            'units': 'm', 'axis': 'Y',
        }),
        'x_ease': ('x_ease', x_ease, {
            'standard_name': 'projection_x_coordinate',
            'long_name': 'EASE-Grid X coordinate',
            'units': 'm', 'axis': 'X',
        }),
    }

    def _da2d(arr, long_name, units, standard_name=None):
        attrs = {'long_name': long_name, 'units': units,
                 'grid_mapping': 'ease_grid_mapping'}
        if standard_name:
            attrs['standard_name'] = standard_name
        return xr.DataArray(
            arr[np.newaxis], dims=['time', 'y_ease', 'x_ease'], attrs=attrs)

    ds_out = xr.Dataset(coords=coords)
    ds_out['T_glorys_surf'] = _da2d(
        T_surf, 'GLORYS reanalysis potential temperature at the surface',
        'degC', standard_name='sea_water_potential_temperature')
    ds_out['S_glorys_surf'] = _da2d(
        S_surf, 'GLORYS reanalysis practical salinity at the surface',
        '1e-3', standard_name='sea_water_practical_salinity')
    ds_out['ease_grid_mapping'] = xr.DataArray(data=0, attrs=gm_attrs)

    ds_out.attrs = {
        'title': f'GLORYS surface intermediate ({get_resolution_label(cfg)}, {target_date:%Y-%m-%d})',
        'Conventions': 'CF-1.8',
        'comment': ('Surface layer (depth=0 m) of GLORYS12 reanalysis, '
                    'regridded to the EASE grid by Step C. Used as model '
                    'input baseline by Step D.'),
        'source_file': str(glorys_path.name),
    }

    encoding = build_var_encoding(ds_out)
    encoding['time'] = dict(TIME_ENCODING)

    atomic_to_netcdf(ds_out, out_file, encoding=encoding)
    ds_out.close()
    return out_file, 'ok'


def main():
    parser = argparse.ArgumentParser(
        description="Step C — GLORYS surface → EASE grid",
    )
    parser.add_argument('--config', required=True,
                        help='Path to the pipeline YAML config file.')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing surface intermediates.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    resolve_glorys_mode(cfg)
    check_time_range(cfg)
    plan = load_pipeline_plan(cfg)
    overwrite = args.force or plan.get('overwrite_glorys_surface', False)

    x_ease, y_ease, gm_attrs = create_ease_grid(cfg)
    _, tf_from = create_transformers(cfg)
    bbox = get_ease_latlon_bbox(cfg, pad_deg=2.0)

    out_dir = get_glorys_surface_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = get_target_dates(cfg)
    logger.info("=" * 60)
    logger.info("Step C: GLORYS surface → EASE grid")
    logger.info(f"  Grid: {cfg['grid']['n_cells_x']}×{cfg['grid']['n_cells_y']} "
                f"@ {cfg['grid']['resolution_km']} km ({get_resolution_label(cfg)})")
    logger.info(f"  Output dir: {out_dir}")
    logger.info(f"  Dates: {len(dates)}  ({'overwrite' if overwrite else 'skip-existing'})")
    logger.info("=" * 60)

    n_ok = n_skip = n_err = 0
    import time as _time
    t0 = _time.monotonic()
    for i, dt in enumerate(dates, 1):
        result, status = process_single_date(
            dt, cfg, x_ease, y_ease, gm_attrs, tf_from, bbox,
            out_dir, overwrite=overwrite,
        )
        if status == 'ok':
            n_ok += 1
            if i % 50 == 0 or i == len(dates):
                logger.info(f"  [{i}/{len(dates)}] processed (ok={n_ok}, skip={n_skip}, err={n_err}) | {format_eta(t0, i, len(dates))}")
        elif status == 'skip':
            n_skip += 1
        else:
            n_err += 1

    logger.info("=" * 60)
    logger.info(f"Step C done — {n_ok} processed, {n_skip} existing, {n_err} errors")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
