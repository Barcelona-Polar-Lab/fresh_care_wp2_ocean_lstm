#!/usr/bin/env python3
"""
Shared configuration utilities for the Arctic Reconstruction Pipeline.

Provides:
- YAML config loading and validation
- EASE grid construction from config parameters
- Resolution labelling
- Target date generation (monthly / daily)
- Training-period checks
- GLORYS file resolution
- WOA standard depth levels
- Satellite EASE directory resolution
"""

import json
import logging
import os
import re
import numpy as np
import pyproj
import yaml
import sys
import xarray as xr
from glob import glob
from pathlib import Path
from datetime import datetime, timedelta
import calendar

logger = logging.getLogger(__name__)


# ============================================================================
# SHARED ENCODING / METADATA CONSTANTS
# ============================================================================

# Time epoch shared by every output NetCDF in the pipeline. Pinning a
# fixed reference date makes daily files concatenate cleanly without
# per-file decoding tricks.
TIME_ENCODING = {
    'units': 'days since 1950-01-01T00:00:00+00:00',
    'calendar': 'standard',
    'dtype': 'float64',
}

# Names of satellite variables inside the EASE-regridded intermediate files.
SAT_VARS = {'SST': 'analysed_sst', 'SSS': 'sss', 'ADT': 'adt'}


# ============================================================================
# WOA STANDARD DEPTH LEVELS (102 levels, 0 – 5500 m)
# ============================================================================

WOA_DEPTHS = np.array([
    0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80,
    85, 90, 95, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350,
    375, 400, 425, 450, 475, 500, 550, 600, 650, 700, 750, 800, 850,
    900, 950, 1000, 1050, 1100, 1150, 1200, 1250, 1300, 1350, 1400,
    1450, 1500, 1550, 1600, 1650, 1700, 1750, 1800, 1850, 1900, 1950,
    2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000,
    3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100,
    4200, 4300, 4400, 4500, 4600, 4700, 4800, 4900, 5000, 5100, 5200,
    5300, 5400, 5500
], dtype=np.float64)


# ============================================================================
# CONFIGURATION LOADING
# ============================================================================

def load_config(yaml_path):
    """
    Load and validate a reconstruction configuration YAML file.

    Parameters
    ----------
    yaml_path : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed and validated configuration dictionary.
    """
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(yaml_path, 'r') as f:
        cfg = yaml.safe_load(f)

    # --- apply host overrides (PIPELINE_HOST env var, default 'local') ---
    # When PIPELINE_HOST != 'local', any keys under cfg['host_overrides'][HOST]
    # are merged on top of cfg['paths']. This lets the same YAML drive both
    # local and remote runs (currently used for Phase D on bec112).
    host = os.environ.get('PIPELINE_HOST', 'local')
    overrides = (cfg.get('host_overrides') or {}).get(host, {}) or {}
    if overrides:
        cfg.setdefault('paths', {}).update(overrides)

    # --- basic presence checks ---
    for section in ('grid', 'projection', 'time', 'paths', 'processing'):
        if section not in cfg:
            raise ValueError(f"Missing required config section: '{section}'")

    grid = cfg['grid']
    for key in ('resolution_km', 'center_x_m', 'center_y_m', 'width_km', 'height_km'):
        if key not in grid:
            raise ValueError(f"Missing grid parameter: '{key}'")

    # --- grid divisibility ---
    res = grid['resolution_km']
    if res <= 0:
        raise ValueError(f"resolution_km must be > 0, got {res}")
    n_x = grid['width_km'] / res
    n_y = grid['height_km'] / res
    if not (n_x == int(n_x) and n_y == int(n_y)):
        raise ValueError(
            f"Grid extent must be evenly divisible by resolution. "
            f"width_km/resolution_km = {n_x}, height_km/resolution_km = {n_y}"
        )
    grid['n_cells_x'] = int(n_x)
    grid['n_cells_y'] = int(n_y)

    # --- resolve model_path relative to workspace root ---
    model_path = Path(cfg['paths']['model_path'])
    if not model_path.is_absolute():
        # Resolve relative to the directory containing this source file's
        # parent (i.e. the workspace root buongiorno_to_pytorch_padding/)
        workspace_root = Path(__file__).resolve().parent.parent
        model_path = workspace_root / model_path
    cfg['paths']['model_path'] = str(model_path)

    # --- time parsing ---
    cfg['time']['_start'] = datetime.strptime(cfg['time']['start_month'], '%Y-%m')
    cfg['time']['_end'] = datetime.strptime(cfg['time']['end_month'], '%Y-%m')
    if cfg['time']['_start'] > cfg['time']['_end']:
        raise ValueError("start_month must be <= end_month")

    # --- training period parsing ---
    tp = cfg.get('training_period', {})
    cfg['training_period'] = {
        'start_month': tp.get('start_month', '2011-01'),
        'end_month': tp.get('end_month', '2021-12'),
    }
    cfg['training_period']['_start'] = datetime.strptime(
        cfg['training_period']['start_month'], '%Y-%m')
    cfg['training_period']['_end'] = datetime.strptime(
        cfg['training_period']['end_month'], '%Y-%m')

    # --- processing defaults ---
    proc = cfg['processing']
    proc.setdefault('glorys_mode', 'auto')
    proc.setdefault('n_mc_samples', 50)
    proc.setdefault('chunk_size', 5000)
    proc.setdefault('time_window_days', 16)
    proc.setdefault('reconstruction_interval_days', 1)
    proc.setdefault('geos_coast_buffer_cells', 0)

    # --- metadata defaults (used for tidy CF-compliant global attrs) ---
    cfg.setdefault('metadata', {})
    md = cfg['metadata']
    md.setdefault('institution', '')
    md.setdefault('authors', '')
    md.setdefault('contact', '')
    md.setdefault('project', '')
    md.setdefault('references', '')
    md.setdefault('license', '')
    md.setdefault('version', '')
    md.setdefault('region', '')
    md.setdefault('comment', '')

    return cfg


# ============================================================================
# GLOBAL ATTRIBUTES (CF-compliant, tidy)
# ============================================================================

def build_global_attrs(cfg, title, source, extra=None):
    """
    Build a tidy, CF-1.8-compliant global-attribute dictionary.

    Parameters
    ----------
    cfg : dict
        Loaded configuration.
    title : str
        Short, human-readable title for the dataset.
    source : str
        Description of the data source / production method.
    extra : dict or None
        Additional attributes to merge in (overrides defaults).

    Returns
    -------
    dict
    """
    from datetime import datetime as _dt

    md = cfg.get('metadata', {}) or {}
    grid = cfg['grid']
    proj4 = get_proj4_string(cfg)
    label = get_resolution_label(cfg)

    now_iso = _dt.utcnow().strftime('%Y-%m-%dT%H:%M:%SZ')

    attrs = {
        'title': title,
        'summary': title,
        'Conventions': 'CF-1.8, ACDD-1.3',
        'source': source,
        'history': f'{now_iso}: file created',
        'creation_date': now_iso,
        'institution': md.get('institution', ''),
        'creator_name': md.get('authors', ''),
        'creator_email': md.get('contact', ''),
        'project': md.get('project', ''),
        'references': md.get('references', ''),
        'license': md.get('license', ''),
        'product_version': md.get('version', ''),
        'region': md.get('region', ''),
        'comment': md.get('comment', ''),
        'geospatial_projection': 'Lambert Azimuthal Equal Area (Arctic, EASE-2)',
        'proj4_string': proj4,
        'grid_resolution': f"{grid['resolution_km']} km",
        'grid_resolution_meters': grid['resolution_km'] * 1000.0,
        'grid_size': f"{grid['n_cells_x']} x {grid['n_cells_y']}",
        'grid_label': label,
    }
    if extra:
        attrs.update(extra)

    # Drop empty values for tidiness
    return {k: v for k, v in attrs.items() if v not in (None, '', [])}


# ============================================================================
# EASE GRID CONSTRUCTION
# ============================================================================

def get_proj4_string(cfg):
    """Build PROJ4 string from the projection section of the config."""
    proj = cfg['projection']
    return (
        f"+proj=laea "
        f"+lat_0={proj['lat_0']} +lon_0={proj['lon_0']} "
        f"+x_0={proj['false_easting']} +y_0={proj['false_northing']} "
        f"+datum=WGS84 +units=m"
    )


def create_ease_grid(cfg):
    """
    Build EASE grid coordinate arrays and CF grid-mapping attributes.

    Parameters
    ----------
    cfg : dict
        Loaded configuration (from ``load_config``).

    Returns
    -------
    x_ease : np.ndarray (1-D)
        Cell-center X coordinates in meters.
    y_ease : np.ndarray (1-D)
        Cell-center Y coordinates in meters.
    grid_mapping_attrs : dict
        CF-compliant attributes for the ``ease_grid_mapping`` variable.
    """
    grid = cfg['grid']
    proj = cfg['projection']
    res_m = grid['resolution_km'] * 1000.0
    n_x = grid['n_cells_x']
    n_y = grid['n_cells_y']
    cx = grid['center_x_m']
    cy = grid['center_y_m']

    x_min = cx - (n_x * res_m) / 2.0
    y_min = cy - (n_y * res_m) / 2.0

    x_ease = np.arange(n_x) * res_m + x_min + res_m / 2.0
    y_ease = np.arange(n_y) * res_m + y_min + res_m / 2.0

    proj4 = get_proj4_string(cfg)

    grid_mapping_attrs = {
        'grid_mapping_name': 'lambert_azimuthal_equal_area',
        'longitude_of_projection_origin': proj['lon_0'],
        'latitude_of_projection_origin': proj['lat_0'],
        'false_easting': proj['false_easting'],
        'false_northing': proj['false_northing'],
        'grid_resolution_meters': res_m,
        'spatial_ref': proj4,
        'proj4_string': proj4,
    }

    return x_ease, y_ease, grid_mapping_attrs


# ============================================================================
# COORDINATE TRANSFORMERS
# ============================================================================

def create_transformers(cfg):
    """
    Create pyproj transformers between WGS84 and EASE projection.

    Returns
    -------
    transformer_to_ease : pyproj.Transformer
    transformer_from_ease : pyproj.Transformer
    """
    proj4 = get_proj4_string(cfg)
    crs_wgs84 = pyproj.CRS.from_epsg(4326)
    crs_ease = pyproj.CRS.from_proj4(proj4)
    to_ease = pyproj.Transformer.from_crs(crs_wgs84, crs_ease, always_xy=True)
    from_ease = pyproj.Transformer.from_crs(crs_ease, crs_wgs84, always_xy=True)
    return to_ease, from_ease


def compute_latlon_grids(x_ease, y_ease, cfg):
    """
    Compute 2-D latitude / longitude grids from EASE coordinates.

    Returns
    -------
    lon_2d, lat_2d : np.ndarray
        Shape ``(len(y_ease), len(x_ease))``.
    """
    _, from_ease = create_transformers(cfg)
    x2d, y2d = np.meshgrid(x_ease, y_ease)
    lon_2d, lat_2d = from_ease.transform(x2d, y2d)
    return lon_2d, lat_2d


def get_ease_latlon_bbox(cfg, pad_deg=2.0):
    """
    Compute the lat/lon bounding box that covers the EASE grid, with padding.

    Samples the grid corners, edges, and center to handle polar-projection
    distortion, then adds *pad_deg* degrees on each side.

    Parameters
    ----------
    cfg : dict
        Loaded config.
    pad_deg : float
        Padding in degrees added to each side of the bounding box.

    Returns
    -------
    dict  with keys 'lat_min', 'lat_max', 'lon_min', 'lon_max'
    """
    x_ease, y_ease, _ = create_ease_grid(cfg)
    _, from_ease = create_transformers(cfg)

    # Sample corners + edge midpoints + center for robust bbox
    xs = [x_ease[0], x_ease[-1], x_ease[0], x_ease[-1],
          x_ease[len(x_ease) // 2], x_ease[0], x_ease[-1],
          x_ease[len(x_ease) // 2], x_ease[len(x_ease) // 2]]
    ys = [y_ease[0], y_ease[0], y_ease[-1], y_ease[-1],
          y_ease[len(y_ease) // 2], y_ease[len(y_ease) // 2],
          y_ease[len(y_ease) // 2], y_ease[0], y_ease[-1]]

    # Also sample along all 4 edges densely (handles curvature)
    n = 50
    edge_x = np.concatenate([
        np.linspace(x_ease[0], x_ease[-1], n),  # bottom
        np.linspace(x_ease[0], x_ease[-1], n),  # top
        np.full(n, x_ease[0]),                   # left
        np.full(n, x_ease[-1]),                  # right
    ])
    edge_y = np.concatenate([
        np.full(n, y_ease[0]),                   # bottom
        np.full(n, y_ease[-1]),                  # top
        np.linspace(y_ease[0], y_ease[-1], n),   # left
        np.linspace(y_ease[0], y_ease[-1], n),   # right
    ])

    all_x = np.concatenate([xs, edge_x])
    all_y = np.concatenate([ys, edge_y])

    lons, lats = from_ease.transform(all_x, all_y)

    lat_min = max(float(np.nanmin(lats)) - pad_deg, -90.0)
    lat_max = min(float(np.nanmax(lats)) + pad_deg, 90.0)
    lon_min = float(np.nanmin(lons)) - pad_deg
    lon_max = float(np.nanmax(lons)) + pad_deg

    # If the grid spans more than 350° of longitude, treat it as global
    if (lon_max - lon_min) >= 350:
        lon_min, lon_max = -180.0, 180.0

    return {
        'lat_min': lat_min, 'lat_max': lat_max,
        'lon_min': lon_min, 'lon_max': lon_max,
    }


def subset_latlon_data(lat, lon, data, bbox):
    """
    Subset 1-D lat/lon arrays and their corresponding data to a bounding box.

    Parameters
    ----------
    lat, lon : 1-D arrays (sorted ascending)
    data : np.ndarray — 2-D (lat, lon) or 3-D (depth/time, lat, lon)
    bbox : dict with 'lat_min', 'lat_max', 'lon_min', 'lon_max'

    Returns
    -------
    lat_sub, lon_sub, data_sub
    """
    lat_mask = (lat >= bbox['lat_min']) & (lat <= bbox['lat_max'])
    lon_mask = (lon >= bbox['lon_min']) & (lon <= bbox['lon_max'])

    lat_sub = lat[lat_mask]
    lon_sub = lon[lon_mask]

    if data.ndim == 2:
        data_sub = data[np.ix_(lat_mask, lon_mask)]
    elif data.ndim == 3:
        data_sub = data[:, np.ix_(lat_mask, lon_mask)[0][:, 0], np.ix_(lat_mask, lon_mask)[1][0, :]]
    else:
        data_sub = data  # fallback — no subsetting

    return lat_sub, lon_sub, data_sub


# ============================================================================
# RESOLUTION LABEL
# ============================================================================

def get_resolution_label(cfg):
    """
    Return a human-readable resolution tag, e.g. ``'ease_25km'``.
    Fractional resolutions keep their decimal form (``'ease_3.125km'``).
    """
    res = cfg['grid']['resolution_km']
    if res == int(res):
        return f"ease_{int(res)}km"
    return f"ease_{res}km"


# ============================================================================
# TARGET DEPTH LEVELS
# ============================================================================

def get_woa_target_depths():
    """Return WOA standard depth levels (102 values, 0–5500 m)."""
    return WOA_DEPTHS.copy()


# ============================================================================
# TARGET DATE GENERATION
# ============================================================================

def get_target_dates(cfg):
    """
    Generate the list of target reconstruction dates.

    For ``glorys_mode == 'monthly'``, one date per month (the 15th at 12:00).
    For ``glorys_mode == 'daily'``, dates are spaced by
    ``reconstruction_interval_days`` (default 1 = every day) **within each
    month**: the day-of-month always lies in {1, 1+interval, 1+2*interval, …}
    capped at the month length. For interval=3 this gives days
    {1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31} every month.
    For ``glorys_mode == 'auto'``, the mode is resolved by
    ``resolve_glorys_mode()`` before this function is called.

    Returns
    -------
    list of datetime
    """
    mode = cfg['processing']['glorys_mode']
    start = cfg['time']['_start']
    end = cfg['time']['_end']
    interval = cfg['processing'].get('reconstruction_interval_days', 1)

    dates = []

    if mode == 'monthly':
        current = datetime(start.year, start.month, 1)
        last_month = datetime(end.year, end.month, 1)
        while current <= last_month:
            dates.append(datetime(current.year, current.month, 15, 12, 0, 0))
            if current.month == 12:
                current = datetime(current.year + 1, 1, 1)
            else:
                current = datetime(current.year, current.month + 1, 1)
    elif mode == 'daily':
        # Walk month-by-month so the day-of-month pattern restarts each month.
        cur_y, cur_m = start.year, start.month
        end_y, end_m = end.year, end.month
        while (cur_y, cur_m) <= (end_y, end_m):
            n_days = calendar.monthrange(cur_y, cur_m)[1]
            day = 1
            while day <= n_days:
                dates.append(datetime(cur_y, cur_m, day, 12, 0, 0))
                day += interval
            if cur_m == 12:
                cur_y, cur_m = cur_y + 1, 1
            else:
                cur_m += 1
    else:
        raise ValueError(f"glorys_mode must be 'monthly' or 'daily', got '{mode}'")

    return dates


# ============================================================================
# TRAINING-PERIOD CHECK
# ============================================================================

def check_time_range(cfg, interactive=True):
    """
    Warn if the requested time range extends beyond the model training period.

    Parameters
    ----------
    cfg : dict
    interactive : bool
        If True, prompt the user for confirmation.  If False, just print
        a warning and continue.

    Returns
    -------
    bool
        True if the user confirms or the range is within bounds.
    """
    req_start = cfg['time']['_start']
    req_end = cfg['time']['_end']
    tp_start = cfg['training_period']['_start']
    tp_end = cfg['training_period']['_end']

    out_before = req_start < tp_start
    out_after = req_end > tp_end

    if not (out_before or out_after):
        return True

    print("\n" + "=" * 60)
    print("WARNING: Requested time range extends beyond the model training period.")
    print(f"  Training period : {tp_start:%Y-%m} to {tp_end:%Y-%m}")
    print(f"  Requested range : {req_start:%Y-%m} to {req_end:%Y-%m}")
    if out_before:
        print(f"  → {req_start:%Y-%m} is BEFORE training start ({tp_start:%Y-%m})")
    if out_after:
        print(f"  → {req_end:%Y-%m} is AFTER training end ({tp_end:%Y-%m})")
    print("Predictions outside the training period may be unreliable.")
    print("=" * 60)

    if interactive:
        answer = input("Proceed anyway? [y/N]: ").strip().lower()
        if answer != 'y':
            print("Aborted by user.")
            sys.exit(0)

    return True


# ============================================================================
# GLORYS MODE RESOLUTION
# ============================================================================

def resolve_glorys_mode(cfg):
    """
    Resolve ``glorys_mode='auto'`` to either ``'daily'`` or ``'monthly'``.

    Checks which GLORYS directory has data covering the requested period.
    Prefers daily if both are available.

    Mutates ``cfg['processing']['glorys_mode']`` in place and returns the
    resolved mode string.
    """
    mode = cfg['processing']['glorys_mode']
    if mode in ('daily', 'monthly'):
        return mode

    daily_dir = Path(cfg['paths'].get('glorys_daily_dir', ''))
    monthly_dir = Path(cfg['paths'].get('glorys_monthly_dir', ''))

    start = cfg['time']['_start']
    end = cfg['time']['_end']

    # Check daily availability — look for year directories
    daily_ok = False
    if daily_dir.is_dir():
        # Check that at least the first requested month has files
        test_dir = daily_dir / f"{start.year}" / f"{start.month:02d}"
        if test_dir.is_dir() and any(test_dir.glob('*.nc')):
            daily_ok = True

    # Check monthly availability
    monthly_ok = False
    if monthly_dir.is_dir():
        # Monthly files match *_mean_YYYYMM_*.nc
        pattern_str = f"*_mean_{start.year}{start.month:02d}*"
        if any(monthly_dir.glob(pattern_str)):
            monthly_ok = True

    if daily_ok:
        resolved = 'daily'
    elif monthly_ok:
        resolved = 'monthly'
    else:
        raise FileNotFoundError(
            f"No GLORYS data found for {start:%Y-%m}. "
            f"Checked daily dir: {daily_dir} and monthly dir: {monthly_dir}"
        )

    cfg['processing']['glorys_mode'] = resolved
    print(f"GLORYS mode resolved to: {resolved}")
    return resolved


# ============================================================================
# GLORYS FILE RESOLUTION
# ============================================================================

def get_glorys_file_for_date(cfg, target_date):
    """
    Return the path to the GLORYS file for a given date.

    Parameters
    ----------
    cfg : dict
    target_date : datetime

    Returns
    -------
    Path or None
    """
    mode = cfg['processing']['glorys_mode']
    y, m, d = target_date.year, target_date.month, target_date.day

    if mode == 'daily':
        base = Path(cfg['paths']['glorys_daily_dir'])
        day_dir = base / f"{y}" / f"{m:02d}"
        pattern = f"*_{y}-{m:02d}-{d:02d}_*"
        matches = sorted(day_dir.glob(pattern)) if day_dir.is_dir() else []
    elif mode == 'monthly':
        base = Path(cfg['paths']['glorys_monthly_dir'])
        pattern = f"*_mean_{y}{m:02d}*"
        matches = sorted(base.glob(pattern))
    else:
        raise ValueError(f"Unknown glorys_mode: {mode}")

    if matches:
        return matches[0]
    return None


# ============================================================================
# SATELLITE EASE DIRECTORY HELPERS
# ============================================================================

def get_satellite_ease_dirs(cfg):
    """
    Return ``{'SST': Path, 'SSS': Path, 'ADT': Path}`` pointing to the
    EASE-regridded satellite data inside the output directory.
    """
    base = get_intermediate_dir(cfg) / 'satellite_ease'
    return {'SST': base / 'SST', 'SSS': base / 'SSS', 'ADT': base / 'ADT'}


# ============================================================================
# STATIC OUTPUT PATH HELPERS
# ============================================================================

def get_static_data_path(cfg):
    """
    Return the path where script A writes the ocean_mask + bathymetry file.
    """
    res_label = get_resolution_label(cfg)
    return get_intermediate_dir(cfg) / 'static' / f'ocean_mask_bathy_{res_label}.nc'


# ============================================================================
# OUTPUT DIRECTORY HELPERS
# ============================================================================

def get_intermediate_dir(cfg):
    """Return the base path for intermediate pipeline files."""
    return Path(cfg['paths']['output_dir']) / 'intermediate_files'


def get_reconstruction_dir(cfg):
    """Return the directory for final reconstruction NetCDF files."""
    return Path(cfg['paths']['output_dir']) / 'final_TS_reconstruction'


def get_glorys_surface_dir(cfg):
    """Return the directory holding per-date GLORYS-surface intermediates
    written by Step C and consumed by Step D."""
    return get_intermediate_dir(cfg) / 'glorys_surface'


def get_anomalies_dir(cfg):
    """Return the directory holding per-date anomaly intermediates written
    by Step D and consumed by Step E."""
    return get_intermediate_dir(cfg) / 'anomalies'


def get_glorys_surface_file(cfg, target_date):
    """Path to the GLORYS-surface intermediate for a given date."""
    return get_glorys_surface_dir(cfg) / f"glorys_surf_{target_date:%Y%m%d}.nc"


def get_anomalies_file(cfg, target_date):
    """Path to the anomalies intermediate for a given date."""
    return get_anomalies_dir(cfg) / f"anomalies_{target_date:%Y%m%d}.nc"


# ============================================================================
# NetCDF ENCODING (int16 quantization + zlib compression)
# ============================================================================

def build_var_encoding(ds, chunk_t=1, chunk_d=17, chunk_xy=50):
    """
    Build per-variable NetCDF encoding with int16 quantization for float32
    variables and zlib=4 compression for everything else.

    float32 → int16 mapping:
        scale_factor = (vmax - vmin) / 65534
        add_offset   = vmin + scale_factor * 32767
        _FillValue   = -32768  (int16 min, reserved for NaN/masked)

    Quantization error ≤ scale_factor / 2, well below observational
    uncertainty for all oceanographic fields here.
    """
    encoding = {}
    for v in ds.data_vars:
        da = ds[v]
        if v == 'ease_grid_mapping':
            encoding[v] = {'zlib': False}
            continue

        ndim = da.ndim
        if ndim == 4:
            chunks = (chunk_t, chunk_d, chunk_xy, chunk_xy)
        elif ndim == 3:
            chunks = (chunk_t, chunk_xy, chunk_xy)
        elif ndim == 2:
            chunks = (chunk_xy, chunk_xy)
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


def atomic_to_netcdf(ds, out_path, encoding=None, **kwargs):
    """Write *ds* to *out_path* atomically.

    Writes to a sibling ``<out_path>.tmp`` then ``os.replace`` to the final
    path. POSIX guarantees ``os.replace`` is atomic on the same filesystem,
    so a Ctrl+C / crash during the write leaves either the previous file
    intact (if any) or no file at all — never a corrupt partial NetCDF.

    Stale ``.tmp`` files from a previous interrupted run are removed before
    writing.
    """
    out_path = Path(out_path)
    tmp_path = out_path.with_suffix(out_path.suffix + '.tmp')
    if tmp_path.exists():
        tmp_path.unlink()
    ds.to_netcdf(tmp_path, encoding=encoding, **kwargs)
    os.replace(tmp_path, out_path)
    return out_path


def format_eta(start_time, done, total, skipped=0):
    """Return a compact progress string with elapsed / rate / ETA.

    Example output::
        elapsed=12m, rate=8.3/min, ETA=18:42 (~6h 04m left)

    *start_time* is a ``time.monotonic()`` value captured at loop start.
    *done* is the count of items completed so far (including skipped),
    *total* is the total number of items planned.
    *skipped* is the number of items that were skipped instantly (e.g.
    because their output already existed). Skips are excluded from the
    rate so the ETA reflects the actual processing speed; this matters a
    lot when resuming a run that has many pre-existing outputs.
    """
    import time as _time
    elapsed = max(_time.monotonic() - start_time, 1e-6)
    real_done = done - skipped
    if real_done <= 0:
        return f"elapsed={_fmt_dur(elapsed)} (skipping existing)"
    rate = real_done / elapsed  # real items per second (skips ~free)
    if rate <= 0 or done >= total:
        return f"elapsed={_fmt_dur(elapsed)}"
    remaining = (total - done) / rate
    eta_abs = datetime.now() + timedelta(seconds=remaining)
    return (f"elapsed={_fmt_dur(elapsed)}, rate={rate*60:.1f}/min, "
            f"ETA={eta_abs:%H:%M} (~{_fmt_dur(remaining)} left)")


def _fmt_dur(seconds):
    """Compact duration: '45s', '12m', '3h 04m', '2d 05h'."""
    s = int(seconds)
    if s < 60:
        return f"{s}s"
    m, s = divmod(s, 60)
    if m < 60:
        return f"{m}m"
    h, m = divmod(m, 60)
    if h < 24:
        return f"{h}h {m:02d}m"
    d, h = divmod(h, 24)
    return f"{d}d {h:02d}h"


# ============================================================================
# SATELLITE DATA LOADING (EASE-regridded intermediates)
# ============================================================================

def _find_satellite_files(base_dir, target_time, window_days, var_kind):
    """
    Gather satellite NetCDF files within ±window_days of *target_time*.

    *var_kind* is one of 'SST', 'SSS', 'ADT'. The expected directory
    layout matches what Step B writes.
    """
    base_dir = Path(base_dir)
    start = (target_time - timedelta(days=window_days)).replace(
        hour=0, minute=0, second=0)
    end = (target_time + timedelta(days=window_days)).replace(
        hour=23, minute=59, second=59)

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
                            var_name=None):
    """
    Load satellite data in a ±window around *target_time* and linearly
    interpolate to the exact target date.

    Returns a 2-D numpy array (y_ease, x_ease) or None on failure.
    """
    if var_name is None:
        var_name = SAT_VARS[var_kind]

    files = _find_satellite_files(base_dir, target_time, window_days, var_kind)
    if not files:
        logger.warning(f"No {var_kind} files for {target_time:%Y-%m-%d}")
        return None

    start = (target_time - timedelta(days=window_days)).replace(
        hour=0, minute=0, second=0)
    end = (target_time + timedelta(days=window_days)).replace(
        hour=23, minute=59, second=59)
    tgt = np.datetime64(target_time)

    chunks = []
    for f in files:
        ds = xr.open_dataset(f)
        if 'time' in ds.dims:
            ds = ds.sel(time=slice(start, end))
        chunk = ds[var_name].load()
        ds.close()
        if chunk.sizes.get('time', 1) > 0:
            chunks.append(chunk)

    if not chunks:
        logger.warning(f"No {var_kind} data in window for {target_time:%Y-%m-%d}")
        return None

    da = xr.concat(chunks, dim='time').sortby('time')
    result = da.interp(time=tgt, method='linear').values
    if np.all(np.isnan(result)):
        logger.warning(
            f"  {var_kind} interpolation returned all-NaN for {target_time:%Y-%m-%d} "
            f"(target outside data range {da.time.values[0]} – {da.time.values[-1]})")
        return None
    return result


# ============================================================================
# SATELLITE DATE-RANGE FILTERING
# ============================================================================

def get_satellite_date_range(cfg):
    """
    Return ``(start_date, end_date)`` for satellite data needed by the
    configured time range, including the time-window buffer.
    """
    start = cfg['time']['_start']
    end = cfg['time']['_end']
    window = cfg['processing']['time_window_days']
    last_day = calendar.monthrange(end.year, end.month)[1]
    return (
        start - timedelta(days=window),
        datetime(end.year, end.month, last_day) + timedelta(days=window),
    )


def _extract_file_date_range(filepath):
    """
    Try to extract a date (range) from a file path.

    Returns ``(start, end)`` as datetime or None.
    """
    name = filepath.stem

    # YYYY-MM-DD in filename
    m = re.search(r'(\d{4})-(\d{2})-(\d{2})', name)
    if m:
        d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return d, d

    # YYYYMMDD as an underscore-separated token
    for part in name.split('_'):
        if len(part) == 8 and part.isdigit():
            try:
                d = datetime.strptime(part, '%Y%m%d')
                return d, d
            except ValueError:
                pass

    # Year only from filename (e.g.  sss_..._2012_...)
    m = re.search(r'(?:^|[_-])((?:19|20)\d{2})(?:[_-]|$)', name)
    if m:
        y = int(m.group(1))
        return datetime(y, 1, 1), datetime(y, 12, 31)

    return None


def filter_files_by_date_range(nc_files, start_date, end_date):
    """Keep only files whose date overlaps ``[start_date, end_date]``."""
    out = []
    for f in nc_files:
        dr = _extract_file_date_range(f)
        if dr is None:
            out.append(f)          # unknown date → include to be safe
        elif dr[1] >= start_date and dr[0] <= end_date:
            out.append(f)
    return out


# ============================================================================
# PIPELINE PLAN (preflight decisions)
# ============================================================================

def get_pipeline_plan_path(cfg):
    """Path to the JSON plan file written by ``run_preflight``."""
    return Path(cfg['paths']['output_dir']) / '.pipeline_plan.json'


def run_preflight(cfg):
    """
    Check existing pipeline outputs and ask user what to reuse / regenerate.
    Saves decisions to a plan file read by each pipeline step.
    """
    print("\n  Checking existing pipeline outputs...\n")
    plan = {}

    # --- Static data (Step A) ---
    static_path = get_static_data_path(cfg)
    if static_path.exists():
        ans = input(f"  Static data exists ({static_path.name}). Reuse? [Y/n]: ").strip().lower()
        plan['run_step_A'] = (ans == 'n')
    else:
        plan['run_step_A'] = True

    # --- Satellite EASE data (Step B) ---
    sat_dirs = get_satellite_ease_dirs(cfg)
    plan['run_step_B'] = {}
    for name in ('SST', 'SSS', 'ADT'):
        sdir = sat_dirs[name]
        if sdir.exists():
            nc = list(sdir.rglob('*.nc'))
            if nc:
                ans = input(f"  {name} EASE data: {len(nc)} files found. Reuse? [Y/n]: ").strip().lower()
                plan['run_step_B'][name] = (ans == 'n')
                continue
        plan['run_step_B'][name] = True

    # --- GLORYS-surface intermediates (Step C) ---
    gs_dir = get_glorys_surface_dir(cfg)
    if gs_dir.exists() and any(gs_dir.glob('glorys_surf_*.nc')):
        n = len(list(gs_dir.glob('glorys_surf_*.nc')))
        ans = input(f"  GLORYS surface: {n} files found. Overwrite? [y/N]: ").strip().lower()
        plan['overwrite_glorys_surface'] = (ans == 'y')
    else:
        plan['overwrite_glorys_surface'] = False

    # --- Anomaly intermediates (Step D) ---
    an_dir = get_anomalies_dir(cfg)
    if an_dir.exists() and any(an_dir.glob('anomalies_*.nc')):
        n = len(list(an_dir.glob('anomalies_*.nc')))
        ans = input(f"  Anomalies: {n} files found. Overwrite? [y/N]: ").strip().lower()
        plan['overwrite_anomalies'] = (ans == 'y')
    else:
        plan['overwrite_anomalies'] = False

    # --- Final reconstruction files (Step E) ---
    recon_dir = get_reconstruction_dir(cfg)
    if recon_dir.exists():
        nc = list(recon_dir.glob('reconstruction_*.nc'))
        if nc:
            ans = input(f"  Reconstruction: {len(nc)} files found. Overwrite? [y/N]: ").strip().lower()
            plan['overwrite_reconstruction'] = (ans == 'y')
        else:
            plan['overwrite_reconstruction'] = False
    else:
        plan['overwrite_reconstruction'] = False

    # --- Save plan ---
    plan_path = get_pipeline_plan_path(cfg)
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    with open(plan_path, 'w') as f:
        json.dump(plan, f, indent=2)

    print("\n  Pipeline plan:")
    print(f"    Step A  (static):           {'SKIP (reuse)' if not plan['run_step_A'] else 'RUN'}")
    for n in ('SST', 'SSS', 'ADT'):
        run = plan['run_step_B'].get(n, True)
        print(f"    Step B  ({n}):              {'SKIP (reuse)' if not run else 'RUN'}")
    print(f"    Step C  (GLORYS surface):   {'OVERWRITE existing' if plan['overwrite_glorys_surface'] else 'KEEP existing, fill gaps'}")
    print(f"    Step D  (anomalies):        {'OVERWRITE existing' if plan['overwrite_anomalies'] else 'KEEP existing, fill gaps'}")
    print(f"    Step E  (final recon):      {'OVERWRITE existing' if plan['overwrite_reconstruction'] else 'KEEP existing, fill gaps'}")
    print()
    return plan


def load_pipeline_plan(cfg):
    """Load the pipeline plan. Returns defaults (run everything) if not found."""
    pp = get_pipeline_plan_path(cfg)
    if pp.exists():
        with open(pp) as f:
            plan = json.load(f)
    else:
        plan = {}
    # Backfill defaults so older plan files keep working
    plan.setdefault('run_step_A', True)
    plan.setdefault('run_step_B', {'SST': True, 'SSS': True, 'ADT': True})
    plan.setdefault('overwrite_glorys_surface', False)
    plan.setdefault('overwrite_anomalies', False)
    plan.setdefault('overwrite_reconstruction', False)
    return plan
