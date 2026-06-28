"""
GLORYS reanalysis regridding to EASE grid.

Used by:
- Step C  — to produce per-date 2-D surface intermediates
            (target_depths=[0.0]).
- Step E  — to regrid the full 3-D field at WOA standard depths during
            the final-output assembly stage.

Keeping this isolated from D_arctic_reconstruction.py lets the D step
run on a remote GPU server without ever touching the raw GLORYS archive.
"""

import logging
import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from config_utils import (
    create_ease_grid,
    create_transformers,
    get_woa_target_depths,
)

logger = logging.getLogger(__name__)

# 3-D GLORYS variables to regrid (depth × lat × lon)
GLORYS_VARS_3D = ['thetao', 'so']


def _interpolate_3d(data_3d, depth_orig, lat_orig, lon_orig,
                    depth_target, x_ease, y_ease,
                    transformer_from_ease, fill_value=np.nan):
    """Interpolate (depth, lat, lon) → (depth_target, y_ease, x_ease).

    Inputs must be sorted ascending in all three axes.
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
        out = np.full((n_d, n_y, n_x), fill_value, dtype=np.float64)

        for di, dv in enumerate(depth_target):
            if dv > max_d:
                continue
            d2d = np.full_like(lat_tgt, dv)
            pts = np.stack([d2d, lat_tgt, lon_tgt], axis=-1)
            out[di] = interp(pts)

    except Exception as e:
        logger.warning(f"3-D interpolation failed: {e}")
        out = np.full((len(depth_target), len(y_ease), len(x_ease)),
                      fill_value, dtype=np.float64)
    return out


def regrid_glorys_single_timestep(glorys_path, cfg,
                                  x_ease=None, y_ease=None,
                                  gm_attrs=None,
                                  transformer_from_ease=None,
                                  bbox=None,
                                  target_depths=None):
    """
    Regrid one GLORYS file to the target EASE grid at the given depths.

    Parameters
    ----------
    glorys_path : Path
        Path to the GLORYS NetCDF file (one timestep).
    cfg : dict
        Loaded pipeline configuration.
    x_ease, y_ease, gm_attrs : optional
        Pre-computed EASE grid arrays + grid_mapping attrs.
    transformer_from_ease : optional
        pyproj transformer from EASE to WGS84.
    bbox : dict, optional
        Lat/lon bounding box for source-side spatial subsetting.
    target_depths : array-like, optional
        Vertical levels to interpolate to. Defaults to the full WOA-102
        standard set. Pass ``[0.0]`` to extract just the surface layer.

    Returns
    -------
    xr.Dataset with dims (depth, y_ease, x_ease) and variables
    ``thetao``, ``so``, plus coordinates and grid mapping.
    """
    if target_depths is None:
        target_depths = get_woa_target_depths()
    target_depths = np.asarray(target_depths, dtype=np.float64)

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
        logger.debug(f"  GLORYS subset: lat {n_lat_orig}→{ds.dims['latitude']}, "
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
    for var_name in GLORYS_VARS_3D:
        if var_name not in ds.data_vars:
            logger.warning(f"Variable {var_name} not in {glorys_path}")
            continue

        # NOW loads data — but only the spatial subset
        raw = ds[var_name].values  # (depth_sub, lat_sub, lon_sub)

        ease = _interpolate_3d(
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
