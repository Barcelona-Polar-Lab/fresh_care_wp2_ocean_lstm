#!/usr/bin/env python3
"""
Step E — Finalize the reconstruction with full 3-D GLORYS reference.

For each target date this script:
    1. Loads the per-date anomaly file produced by Step D
       (T_anom_pred, S_anom_pred, T_anom_std, S_anom_std).
    2. Regrids the corresponding GLORYS reanalysis file to the EASE grid
       at the full WOA depth axis (3-D field).
    3. Loads the EASE-regridded satellite SST/SSS/ADT for that date.
    4. Builds the final per-date NetCDF in the *exact same schema* as the
       previous monolithic Step D output (variable names, units,
       attributes, global attributes).

This keeps the published product identical while moving the heavy
LSTM/MC-Dropout step into a separate, server-portable Step D.

Usage:
    python E_finalize_with_glorys.py --config CONFIG.yaml [--force]
"""

import argparse
import calendar
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

# Parent dir for any utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config_utils import (
    atomic_to_netcdf,
    SAT_VARS,
    TIME_ENCODING,
    build_global_attrs,
    build_var_encoding,
    check_time_range,
    create_ease_grid,
    create_transformers,
    format_eta,
    get_anomalies_file,
    get_ease_latlon_bbox,
    get_glorys_file_for_date,
    get_reconstruction_dir,
    get_resolution_label,
    get_satellite_ease_dirs,
    get_static_data_path,
    get_target_dates,
    get_woa_target_depths,
    load_config,
    load_pipeline_plan,
    load_satellite_for_time,
    print_packing_spec,
    resolve_glorys_mode,
)
from glorys_regrid import regrid_glorys_single_timestep
from geos_currents import compute_geostrophic_currents

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


def finalize_single_date(target_date, cfg, static_ds, x_ease, y_ease,
                         gm_attrs, sat_dirs, tf_from, bbox,
                         out_dir, overwrite=False):
    date_str = target_date.strftime('%Y%m%d')
    out_file = out_dir / f"TS_currents_lstm_{date_str}.nc"
    if out_file.exists() and not overwrite:
        logger.info(f"  {out_file.name} exists — skipping")
        return out_file

    # --- Load anomalies (Step D output) ---
    anom_file = get_anomalies_file(cfg, target_date)
    if not anom_file.exists():
        logger.error(f"  No anomalies file for {date_str} (expected {anom_file}). "
                     f"Run Step D first.")
        return None

    with xr.open_dataset(anom_file) as ds_anom:
        T_anom_pred = np.asarray(ds_anom['T_anom_pred'].values).squeeze(0)  # (depth, ny, nx)
        S_anom_pred = np.asarray(ds_anom['S_anom_pred'].values).squeeze(0)
        T_anom_std = np.asarray(ds_anom['T_anom_std'].values).squeeze(0)
        S_anom_std = np.asarray(ds_anom['S_anom_std'].values).squeeze(0)

    # --- Regrid full 3-D GLORYS ---
    glorys_path = get_glorys_file_for_date(cfg, target_date)
    if glorys_path is None:
        logger.error(f"  No GLORYS file for {date_str}")
        return None
    logger.info(f"  [{date_str}] Regridding GLORYS: {glorys_path.name}")
    ds_glorys = regrid_glorys_single_timestep(
        glorys_path, cfg,
        x_ease=x_ease, y_ease=y_ease, gm_attrs=gm_attrs,
        transformer_from_ease=tf_from, bbox=bbox,
    )
    T_glorys = ds_glorys['thetao'].values   # (depth, ny, nx)
    S_glorys = ds_glorys['so'].values
    ds_glorys.close()

    # --- Mask anomalies where GLORYS is NaN (below seabed) ---
    glorys_nan = np.isnan(T_glorys) | np.isnan(S_glorys)
    T_anom_pred = np.where(glorys_nan, np.nan, T_anom_pred)
    S_anom_pred = np.where(glorys_nan, np.nan, S_anom_pred)
    T_anom_std = np.where(glorys_nan, np.nan, T_anom_std)
    S_anom_std = np.where(glorys_nan, np.nan, S_anom_std)

    # --- Reconstruct full profiles: anomaly + GLORYS ---
    T_recon = T_anom_pred + T_glorys
    S_recon = S_anom_pred + S_glorys

    # --- Satellite surface inputs ---
    logger.info(f"  [{date_str}] Loading satellite surface data")
    window = cfg['processing']['time_window_days']
    SST = load_satellite_for_time(
        sat_dirs['SST'], target_date, window, 'SST', SAT_VARS['SST'])
    SSS = load_satellite_for_time(
        sat_dirs['SSS'], target_date, window, 'SSS', SAT_VARS['SSS'])
    ADT = load_satellite_for_time(
        sat_dirs['ADT'], target_date, window, 'ADT', SAT_VARS['ADT'])
    if SST is None or SSS is None or ADT is None:
        logger.error(f"  Missing satellite data for {date_str}")
        return None
    SST = SST - 273.15
    DOY = target_date.timetuple().tm_yday

    # --- Geostrophic currents from reconstructed T/S + satellite ADT ---
    logger.info(f"  [{date_str}] Computing geostrophic currents")
    depth = get_woa_target_depths()
    lat2d = static_ds['latitude'].values
    lon2d = static_ds['longitude'].values
    ADH, vel_gos_x, vel_gos_y, u_gos, v_gos = compute_geostrophic_currents(
        T_recon, S_recon, ADT, depth, lat2d, lon2d, x_ease, y_ease,
        coast_buffer_cells=cfg['processing'].get('geos_coast_buffer_cells', 0),
    )

    # --- Build output dataset (schema MUST match legacy Step D output) ---
    # Daily product timestamp = NOON UTC of target_date (centroid of the 24-h
    # window). This matches the convention in the published archive; do not
    # change to midnight without re-aligning all downstream consumers.
    time_val = np.datetime64(target_date.strftime('%Y-%m-%dT12:00:00'))

    coords = {
        'time':   ('time',   [time_val], {'standard_name': 'time',
                                          'long_name': 'Time',
                                          'axis': 'T'}),
        'depth':  ('depth',  depth, {'standard_name': 'depth',
                                     'long_name': 'depth below sea surface',
                                     'units': 'm', 'positive': 'down',
                                     'axis': 'Z'}),
        'y_ease': ('y_ease', y_ease, {'standard_name': 'projection_y_coordinate',
                                      'long_name': 'EASE-Grid Y coordinate',
                                      'units': 'm', 'axis': 'Y'}),
        'x_ease': ('x_ease', x_ease, {'standard_name': 'projection_x_coordinate',
                                      'long_name': 'EASE-Grid X coordinate',
                                      'units': 'm', 'axis': 'X'}),
    }

    def _da(arr, long_name, units, standard_name=None):
        attrs = {'long_name': long_name, 'units': units,
                 'grid_mapping': 'ease_grid_mapping'}
        if standard_name:
            attrs['standard_name'] = standard_name
        return xr.DataArray(
            arr[np.newaxis],   # add time dim → (1, depth, ny, nx)
            dims=['time', 'depth', 'y_ease', 'x_ease'],
            attrs=attrs,
        )

    def _da2d(arr, long_name, units, standard_name=None):
        attrs = {'long_name': long_name, 'units': units,
                 'grid_mapping': 'ease_grid_mapping'}
        if standard_name:
            attrs['standard_name'] = standard_name
        return xr.DataArray(
            arr[np.newaxis],
            dims=['time', 'y_ease', 'x_ease'],
            attrs=attrs,
        )

    ds_out = xr.Dataset(coords=coords)

    # Predicted anomalies
    ds_out['T_anom_pred'] = _da(T_anom_pred,
        'Predicted temperature anomaly', 'degC')
    ds_out['S_anom_pred'] = _da(S_anom_pred,
        'Predicted practical salinity anomaly', '1e-3')

    # Uncertainty (MC Dropout standard deviation)
    ds_out['T_anom_std'] = _da(T_anom_std,
        'MC-Dropout standard deviation of temperature anomaly', 'degC')
    ds_out['S_anom_std'] = _da(S_anom_std,
        'MC-Dropout standard deviation of salinity anomaly', '1e-3')

    # Reconstructed full profiles
    ds_out['T_recon'] = _da(T_recon,
        'Reconstructed temperature (anomaly + GLORYS reference)',
        'degC', standard_name='sea_water_temperature')
    ds_out['S_recon'] = _da(S_recon,
        'Reconstructed practical salinity (anomaly + GLORYS reference)',
        '1e-3', standard_name='sea_water_practical_salinity')

    # GLORYS reference
    ds_out['T_glorys'] = _da(T_glorys,
        'GLORYS reanalysis potential temperature', 'degC',
        standard_name='sea_water_potential_temperature')
    ds_out['S_glorys'] = _da(S_glorys,
        'GLORYS reanalysis practical salinity', '1e-3',
        standard_name='sea_water_practical_salinity')

    # Surface satellite inputs
    ds_out['SST'] = _da2d(SST,
        'Sea surface temperature (satellite L4)', 'degC',
        standard_name='sea_surface_temperature')
    ds_out['SSS'] = _da2d(SSS,
        'Sea surface salinity (satellite L4)', '1e-3',
        standard_name='sea_surface_salinity')
    ds_out['ADT'] = _da2d(ADT,
        'Absolute dynamic topography (satellite L4)', 'm',
        standard_name='sea_surface_height_above_geoid')

    # Geostrophic currents (derived from T_recon, S_recon, ADT)
    ds_out['ADH'] = _da(ADH,
        'Absolute dynamic height (ADT - steric height)', 'm',
        standard_name='geopotential_height')
    ds_out['vel_gos_x'] = _da(vel_gos_x,
        'Geostrophic velocity, EASE-grid x component', 'm s-1')
    ds_out['vel_gos_y'] = _da(vel_gos_y,
        'Geostrophic velocity, EASE-grid y component', 'm s-1')
    ds_out['u_gos'] = _da(u_gos,
        'Eastward geostrophic velocity', 'm s-1',
        standard_name='eastward_sea_water_velocity')
    ds_out['v_gos'] = _da(v_gos,
        'Northward geostrophic velocity', 'm s-1',
        standard_name='northward_sea_water_velocity')

    # DOY
    ds_out['DOY'] = xr.DataArray([DOY], dims=['time'],
                                 attrs={'long_name': 'Day of year'})

    # Static (copy from static_ds). latitude/longitude are promoted to
    # non-dim coords (not data_vars) so xarray automatically emits
    # `coordinates = "latitude longitude"` on every (y_ease, x_ease)-
    # dimensioned data variable, per CF §5.6.
    for vname in ('ocean_mask', 'elevation'):
        if vname in static_ds:
            ds_out[vname] = static_ds[vname]
    # CF §3.5: flag_values dtype must match the variable's dtype.
    if 'ocean_mask' in ds_out and 'flag_values' in ds_out['ocean_mask'].attrs:
        ds_out['ocean_mask'].attrs['flag_values'] = np.asarray(
            ds_out['ocean_mask'].attrs['flag_values'], dtype=np.int8)
    if 'latitude' in static_ds and 'longitude' in static_ds:
        ds_out = ds_out.assign_coords(
            latitude=static_ds['latitude'].astype(np.float64),
            longitude=static_ds['longitude'].astype(np.float64),
        )

    ds_out['ease_grid_mapping'] = xr.DataArray(data=0, attrs=gm_attrs)

    t_start = cfg['time']['_start']
    t_end = cfg['time']['_end']
    last_day = calendar.monthrange(t_end.year, t_end.month)[1]
    t_end_full = t_end.replace(day=last_day)
    period_label = f'{t_start:%Y-%m-%d} to {t_end_full:%Y-%m-%d}'

    ds_out.attrs = build_global_attrs(
        cfg,
        title=f'Arctic 4-D ocean reconstruction ({get_resolution_label(cfg)}, {period_label})',
        source=('LSTM with Monte-Carlo Dropout, trained on Arctic in-situ profiles, '
                'driven by satellite SST/SSS/ADT and added to GLORYS12 reanalysis reference'),
        extra={
            'model_path': str(cfg['paths']['model_path']),
            'glorys_mode': cfg['processing']['glorys_mode'],
            'n_mc_samples': cfg['processing']['n_mc_samples'],
            'satellite_time_window_days': cfg['processing']['time_window_days'],
            'reconstruction_date': target_date.strftime('%Y-%m-%d'),
        },
    )

    encoding = build_var_encoding(ds_out)
    encoding['time'] = dict(TIME_ENCODING)

    atomic_to_netcdf(ds_out, out_file, encoding=encoding)
    ds_out.close()
    logger.info(f"  [{date_str}] Saved → {out_file.name}")
    return out_file


def main():
    parser = argparse.ArgumentParser(
        description="Step E — Finalize reconstruction with full GLORYS",
    )
    parser.add_argument('--config', required=True,
                        help='Path to the pipeline YAML config file')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing reconstruction files.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    resolve_glorys_mode(cfg)
    check_time_range(cfg)
    plan = load_pipeline_plan(cfg)
    overwrite = args.force or plan.get('overwrite_reconstruction', False)

    label = get_resolution_label(cfg)
    x_ease, y_ease, gm_attrs = create_ease_grid(cfg)
    _, tf_from = create_transformers(cfg)
    bbox = get_ease_latlon_bbox(cfg, pad_deg=2.0)

    # --- Static data ---
    static_path = get_static_data_path(cfg)
    if not static_path.exists():
        logger.error(f"Static data not found: {static_path}")
        sys.exit(1)
    static_ds = xr.open_dataset(static_path)

    sat_dirs = get_satellite_ease_dirs(cfg)

    out_dir = get_reconstruction_dir(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)

    dates = get_target_dates(cfg)
    logger.info("=" * 60)
    logger.info("Step E — Finalize reconstruction")
    logger.info(f"  Grid: {cfg['grid']['n_cells_x']}×{cfg['grid']['n_cells_y']} "
                f"@ {cfg['grid']['resolution_km']} km ({label})")
    logger.info(f"  GLORYS mode: {cfg['processing']['glorys_mode']}")
    logger.info(f"  Output dir: {out_dir}")
    logger.info(f"  Dates: {len(dates)}  ({'overwrite' if overwrite else 'skip-existing'})")
    logger.info("=" * 60)
    print_packing_spec(logger)

    ok = errors = skipped = 0
    failed_dates = []
    import time as _time
    t0 = _time.monotonic()
    for i, dt in enumerate(dates, 1):
        logger.info(f"\n--- Date {i}/{len(dates)}: {dt:%Y-%m-%d} ---")
        # Pre-check skip so the ETA can exclude instant no-ops from the rate
        will_skip = (out_dir / f"TS_currents_lstm_{dt:%Y%m%d}.nc").exists() and not overwrite
        result = finalize_single_date(
            dt, cfg, static_ds, x_ease, y_ease, gm_attrs,
            sat_dirs, tf_from, bbox, out_dir, overwrite=overwrite,
        )
        if result is None:
            errors += 1
            failed_dates.append(dt)
        else:
            ok += 1
            if will_skip:
                skipped += 1
        logger.info(f"  Progress: {i}/{len(dates)} ({ok} ok, {errors} errors) | {format_eta(t0, i, len(dates), skipped=skipped)}")

    static_ds.close()

    logger.info("\n" + "=" * 60)
    logger.info("Step E complete")
    logger.info(f"  Successful: {ok}")
    logger.info(f"  Errors:     {errors}")
    if failed_dates:
        logger.warning("  Failed dates: " + ", ".join(d.strftime('%Y-%m-%d') for d in failed_dates))
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
