#!/usr/bin/env python3
"""
Step D — LSTM + MC Dropout anomaly inference

For each target date this script:
    1. Loads static EASE data (mask, X/Y).
    2. Loads the GLORYS-surface intermediate written by Step C.
    3. Loads + time-interpolates EASE-regridded satellite SST/SSS/ADT.
    4. Builds the per-profile model input (7 features × n_depths).
    5. Runs MC Dropout predictions → T_anom_pred, S_anom_pred,
       T_anom_std, S_anom_std.
    6. Writes one small NetCDF per date under
       ``intermediate_files/anomalies/anomalies_YYYYMMDD.nc``.

Step D does **not** open the raw GLORYS archive. Together with the
intermediates produced by Steps A, B and C it is fully self-contained
and can run on a remote GPU server. Step E consumes its outputs
locally to assemble the final products.

Usage:
    python D_arctic_reconstruction.py --config CONFIG.yaml [--device {auto,cpu,cuda}]
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path

import numpy as np
import xarray as xr

warnings.filterwarnings("ignore")

# Parent dir for lstm_pytorch_utils
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lstm_pytorch_utils import load_model_checkpoint, mc_dropout_predict_chunked

from config_utils import (
    atomic_to_netcdf,
    SAT_VARS,
    TIME_ENCODING,
    build_var_encoding,
    check_time_range,
    create_ease_grid,
    create_transformers,
    format_eta,
    get_anomalies_file,
    get_anomalies_dir,
    get_ease_latlon_bbox,
    get_glorys_surface_file,
    get_resolution_label,
    get_satellite_ease_dirs,
    get_static_data_path,
    get_target_dates,
    get_woa_target_depths,
    load_config,
    load_pipeline_plan,
    load_satellite_for_time,
    resolve_glorys_mode,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)


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
# SINGLE-DATE ANOMALY INFERENCE → NetCDF
# ============================================================================

def reconstruct_single_date(target_date, cfg, model, norm_params,
                            static_ds, x_ease, y_ease, gm_attrs,
                            sat_dirs, device, overwrite=False):
    """
    Anomaly inference for one date.  Returns the output path or None.
    """
    date_str = target_date.strftime('%Y%m%d')
    out_file = get_anomalies_file(cfg, target_date)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if out_file.exists() and not overwrite:
        logger.info(f"  {out_file.name} exists — skipping")
        return out_file

    n_depths = len(get_woa_target_depths())
    n_y, n_x = len(y_ease), len(x_ease)

    # --- Static fields ---
    ocean_mask = static_ds['ocean_mask'].values          # (ny, nx)
    X_EASE_2d = np.broadcast_to(x_ease[None, :], (n_y, n_x)).copy()
    Y_EASE_2d = np.broadcast_to(y_ease[:, None], (n_y, n_x)).copy()

    # --- GLORYS surface (from B1 intermediate) ---
    surf_file = get_glorys_surface_file(cfg, target_date)
    if not surf_file.exists():
        logger.error(f"  No GLORYS surface intermediate for {date_str} "
                     f"(expected {surf_file}). Run Step C first.")
        return None

    with xr.open_dataset(surf_file) as ds_surf:
        T_glorys_surf = np.asarray(ds_surf['T_glorys_surf'].values).squeeze()  # (ny, nx)
        S_glorys_surf = np.asarray(ds_surf['S_glorys_surf'].values).squeeze()

    # --- Satellite surface data ---
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
    del prof['X']

    # --- Regrid to grid ---
    logger.info(f"  [{date_str}] Regridding predictions to grid & saving")
    grids = regrid_profiles_to_grid(
        y_mean, y_std,
        prof['y_idx'], prof['x_idx'],
        n_y, n_x, n_depths,
    )
    del prof, y_mean, y_std

    # --- Build output dataset (anomalies only) ---
    depth = get_woa_target_depths()
    time_val = np.datetime64(target_date.strftime('%Y-%m-%dT%H:%M:%S'))

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

    def _da(arr, long_name, units):
        return xr.DataArray(
            arr[np.newaxis],   # add time dim → (1, depth, ny, nx)
            dims=['time', 'depth', 'y_ease', 'x_ease'],
            attrs={'long_name': long_name, 'units': units,
                   'grid_mapping': 'ease_grid_mapping'},
        )

    ds_out = xr.Dataset(coords=coords)
    ds_out['T_anom_pred'] = _da(grids['T_anom_pred'],
        'Predicted temperature anomaly (LSTM mean)', 'degC')
    ds_out['S_anom_pred'] = _da(grids['S_anom_pred'],
        'Predicted practical salinity anomaly (LSTM mean)', '1e-3')
    ds_out['T_anom_std'] = _da(grids['T_anom_std'],
        'MC-Dropout standard deviation of temperature anomaly', 'degC')
    ds_out['S_anom_std'] = _da(grids['S_anom_std'],
        'MC-Dropout standard deviation of salinity anomaly', '1e-3')
    ds_out['ease_grid_mapping'] = xr.DataArray(data=0, attrs=gm_attrs)

    ds_out.attrs = {
        'title': f'Arctic LSTM anomaly inference ({get_resolution_label(cfg)}, {target_date:%Y-%m-%d})',
        'Conventions': 'CF-1.8',
        'comment': ('Per-date LSTM + MC-Dropout anomaly fields produced by '
                    'Step D. Step E combines these with the full 3-D GLORYS '
                    'reference and the satellite surface inputs to assemble '
                    'the final reconstruction product.'),
        'n_mc_samples': cfg['processing']['n_mc_samples'],
        'reconstruction_date': target_date.strftime('%Y-%m-%d'),
    }

    encoding = build_var_encoding(ds_out)
    encoding['time'] = dict(TIME_ENCODING)

    atomic_to_netcdf(ds_out, out_file, encoding=encoding)
    ds_out.close()
    del grids
    logger.info(f"  [{date_str}] Saved → {out_file.name}")
    return out_file


# ============================================================================
# CLI / MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Step D — LSTM + MC-Dropout anomaly inference",
    )
    parser.add_argument('--config', required=True,
                        help='Path to the pipeline YAML config file')
    parser.add_argument('--device', choices=('auto', 'cpu', 'cuda'), default='auto',
                        help='Device for inference (default: auto — use CUDA if available)')
    args = parser.parse_args()

    cfg = load_config(args.config)
    # GLORYS surface intermediates are read; we still resolve the mode so
    # cfg fields are populated for logging consistency.
    resolve_glorys_mode(cfg)
    check_time_range(cfg)
    plan = load_pipeline_plan(cfg)
    overwrite = plan.get('overwrite_anomalies', False)

    label = get_resolution_label(cfg)
    x_ease, y_ease, gm_attrs = create_ease_grid(cfg)
    bbox = get_ease_latlon_bbox(cfg, pad_deg=2.0)

    logger.info("=" * 60)
    logger.info("Step D — LSTM + MC-Dropout anomaly inference")
    logger.info(f"  Grid: {cfg['grid']['n_cells_x']}×{cfg['grid']['n_cells_y']} "
                f"@ {cfg['grid']['resolution_km']} km ({label})")
    logger.info(f"  Spatial bbox: lat=[{bbox['lat_min']:.1f}, {bbox['lat_max']:.1f}], "
                f"lon=[{bbox['lon_min']:.1f}, {bbox['lon_max']:.1f}]")
    logger.info(f"  Time range: {cfg['time']['start_month']} → {cfg['time']['end_month']}")
    logger.info(f"  Output dir: {get_anomalies_dir(cfg)}")
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
    if args.device == 'cuda':
        if not torch.cuda.is_available():
            logger.error("--device cuda requested but no CUDA device is available")
            sys.exit(1)
        device = torch.device('cuda')
    elif args.device == 'cpu':
        device = torch.device('cpu')
    else:  # auto
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"  Inference device: {device}")

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

    ok = errors = 0
    failed_dates = []
    n_dates = len(dates)
    import time as _time
    t0 = _time.monotonic()
    for i, dt in enumerate(dates, 1):
        logger.info(f"\n--- Date {i}/{n_dates}: {dt:%Y-%m-%d} ---")
        result = reconstruct_single_date(
            dt, cfg, model, norm_params,
            static_ds, x_ease, y_ease, gm_attrs,
            sat_dirs, device, overwrite=overwrite,
        )
        if result is None:
            errors += 1
            failed_dates.append(dt)
        else:
            ok += 1
        logger.info(f"  Progress: {i}/{n_dates} ({ok} ok, {errors} errors) | {format_eta(t0, i, n_dates)}")

    static_ds.close()

    logger.info("\n" + "=" * 60)
    logger.info("Step D complete")
    logger.info(f"  Successful: {ok}")
    logger.info(f"  Errors:     {errors}")
    if failed_dates:
        logger.warning("  Failed dates: " + ", ".join(d.strftime('%Y-%m-%d') for d in failed_dates))
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
