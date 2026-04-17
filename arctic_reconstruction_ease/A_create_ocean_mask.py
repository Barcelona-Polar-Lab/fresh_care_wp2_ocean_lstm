#!/usr/bin/env python3
"""
Step A — Create ocean mask and interpolated bathymetry on the target EASE grid.

Reads configuration from a YAML file (--config) and produces a NetCDF file
containing:
    - ocean_mask   (1 = ocean, 0 = land)
    - elevation    (GEBCO bathymetry interpolated from 1 km)
    - latitude / longitude grids
    - X_EASE / Y_EASE coordinate arrays
    - CF-compliant ease_grid_mapping

Usage:
    python A_create_ocean_mask.py --config my_config.yaml
"""

import numpy as np
import xarray as xr
import geopandas as gpd
from rasterio import features
from affine import Affine
from scipy.interpolate import RegularGridInterpolator
import argparse
from pathlib import Path

from config_utils import (
    load_config,
    create_ease_grid,
    get_proj4_string,
    get_resolution_label,
    get_static_data_path,
    compute_latlon_grids,
    get_ease_latlon_bbox,
    load_pipeline_plan,
)


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def create_ease_transform(x_ease, y_ease):
    """
    Build an affine transform for rasterio.features.rasterize.

    Parameters
    ----------
    x_ease, y_ease : 1-D arrays
        Cell-center coordinates (meters).

    Returns
    -------
    Affine
    """
    dx = x_ease[1] - x_ease[0] if len(x_ease) > 1 else 25000.0
    dy = y_ease[1] - y_ease[0] if len(y_ease) > 1 else 25000.0
    x_min = x_ease[0] - dx / 2.0
    y_min = y_ease[0] - dy / 2.0
    return Affine(dx, 0, x_min, 0, dy, y_min)


def rasterize_ocean_shapefile(shapefile_path, transform, shape, x_ease, y_ease,
                              proj4):
    """
    Rasterize an ocean shapefile onto the target EASE grid.

    Parameters
    ----------
    shapefile_path : str or Path
    transform : Affine
    shape : tuple (ny, nx)
    x_ease, y_ease : 1-D arrays
    proj4 : str
        PROJ4 string of the target CRS.

    Returns
    -------
    np.ndarray  (uint8, shape = (ny, nx))
        1 = ocean, 0 = land.
    """
    from shapely.geometry import box

    print(f"Reading ocean shapefile: {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)

    buffer = 100_000  # 100 km
    x_lo, x_hi = x_ease.min() - buffer, x_ease.max() + buffer
    y_lo, y_hi = y_ease.min() - buffer, y_ease.max() + buffer

    print(f"Reprojecting from {gdf.crs} → EASE projection")
    gdf_ease = gdf.to_crs(proj4)

    invalid = (~gdf_ease.geometry.is_valid).sum()
    if invalid > 0:
        print(f"Fixing {invalid}/{len(gdf_ease)} invalid geometries "
              "(common after polar reprojection)")
        gdf_ease["geometry"] = gdf_ease.geometry.buffer(0)

    clip_box = box(x_lo, y_lo, x_hi, y_hi)
    print(f"Clipping to grid extent  x=[{x_lo:.0f}, {x_hi:.0f}], "
          f"y=[{y_lo:.0f}, {y_hi:.0f}]")
    gdf_ease = gdf_ease.clip(clip_box)
    print(f"After clipping: {len(gdf_ease)} features")

    print(f"Rasterizing ocean mask → shape {shape}")
    ocean_mask = features.rasterize(
        ((geom, 1) for geom in gdf_ease.geometry),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True,
    )
    return ocean_mask


def interpolate_gebco(gebco_path, x_ease, y_ease):
    """
    Interpolate GEBCO 1 km bathymetry onto the target EASE grid.

    Parameters
    ----------
    gebco_path : str or Path
        Path to the GEBCO NetCDF file on a 1 km EASE grid.
    x_ease, y_ease : 1-D arrays
        Target grid coordinates (meters).

    Returns
    -------
    np.ndarray  (float32, shape = (ny, nx))
        Elevation in metres (negative = below sea level).
    """
    print(f"Loading GEBCO 1 km bathymetry: {gebco_path}")
    ds = xr.open_dataset(gebco_path)

    # Identify coordinate / variable names (the 1 km EASE file may use
    # 'x_ease'/'y_ease', 'x'/'y', or similar).
    x_name = None
    for candidate in ('x_ease', 'x', 'X'):
        if candidate in ds.coords or candidate in ds.dims:
            x_name = candidate
            break
    y_name = None
    for candidate in ('y_ease', 'y', 'Y'):
        if candidate in ds.coords or candidate in ds.dims:
            y_name = candidate
            break
    elev_name = None
    for candidate in ('elevation', 'z', 'Band1'):
        if candidate in ds.data_vars:
            elev_name = candidate
            break

    if x_name is None or y_name is None or elev_name is None:
        raise RuntimeError(
            f"Cannot identify GEBCO coordinate/variable names. "
            f"Coords: {list(ds.coords)}, Vars: {list(ds.data_vars)}"
        )

    src_x = ds[x_name].values.astype(np.float64)
    src_y = ds[y_name].values.astype(np.float64)
    src_elev = ds[elev_name].values.astype(np.float64)
    ds.close()

    # Subset to the target region + buffer for speed on small grids
    buf = 50_000  # 50 km buffer in metres
    x_mask = (src_x >= x_ease.min() - buf) & (src_x <= x_ease.max() + buf)
    y_mask = (src_y >= y_ease.min() - buf) & (src_y <= y_ease.max() + buf)
    if x_mask.sum() > 0 and y_mask.sum() > 0 and (x_mask.sum() < len(src_x) or y_mask.sum() < len(src_y)):
        src_x = src_x[x_mask]
        src_y = src_y[y_mask]
        src_elev = src_elev[np.ix_(y_mask, x_mask)]
        print(f"Subsetted GEBCO to region: ({len(src_y)}, {len(src_x)})")

    print(f"Building RegularGridInterpolator  "
          f"src shape=({len(src_y)}, {len(src_x)}) → "
          f"target shape=({len(y_ease)}, {len(x_ease)})")
    interp = RegularGridInterpolator(
        (src_y, src_x), src_elev,
        method='linear', bounds_error=False, fill_value=np.nan,
    )

    tgt_x2d, tgt_y2d = np.meshgrid(x_ease, y_ease)
    pts = np.column_stack([tgt_y2d.ravel(), tgt_x2d.ravel()])
    elev = interp(pts).reshape(len(y_ease), len(x_ease)).astype(np.float32)
    print(f"Bathymetry interpolation done — "
          f"range [{np.nanmin(elev):.0f}, {np.nanmax(elev):.0f}] m")
    return elev


# ============================================================================
# MAIN DATASET BUILDER
# ============================================================================

def create_static_dataset(cfg):
    """
    Build the static-data xarray.Dataset for a given configuration.

    Parameters
    ----------
    cfg : dict
        Loaded config (from ``load_config``).

    Returns
    -------
    xr.Dataset
    """
    label = get_resolution_label(cfg)
    proj4 = get_proj4_string(cfg)
    res_m = cfg['grid']['resolution_km'] * 1000.0

    # 1. EASE grid coordinates
    x_ease, y_ease, gm_attrs = create_ease_grid(cfg)
    n_y, n_x = len(y_ease), len(x_ease)
    print(f"Target EASE grid: {n_x}×{n_y} @ {cfg['grid']['resolution_km']} km "
          f"  ({label})")

    # 2. Lat / Lon grids
    lon_2d, lat_2d = compute_latlon_grids(x_ease, y_ease, cfg)

    # 3. Ocean mask
    shp_path = cfg['paths']['natural_earth_shapefile']
    transform = create_ease_transform(x_ease, y_ease)
    ocean_mask = rasterize_ocean_shapefile(
        shp_path, transform, (n_y, n_x), x_ease, y_ease, proj4
    )

    # 4. Bathymetry (GEBCO 1 km → target resolution)
    gebco_path = cfg['paths']['gebco_1km_file']
    elevation = interpolate_gebco(gebco_path, x_ease, y_ease)

    # 5. Assemble dataset
    ds = xr.Dataset(
        {
            'ocean_mask': (['y_ease', 'x_ease'], ocean_mask, {
                'long_name': 'Ocean mask',
                'description': '1 = ocean, 0 = land',
                'grid_mapping': 'ease_grid_mapping',
            }),
            'elevation': (['y_ease', 'x_ease'], elevation, {
                'long_name': 'GEBCO elevation (positive up)',
                'units': 'm',
                'source': f'Interpolated from 1 km GEBCO to {label}',
                'grid_mapping': 'ease_grid_mapping',
            }),
            'latitude': (['y_ease', 'x_ease'], lat_2d.astype(np.float32), {
                'long_name': 'Latitude',
                'units': 'degrees_north',
                'grid_mapping': 'ease_grid_mapping',
            }),
            'longitude': (['y_ease', 'x_ease'], lon_2d.astype(np.float32), {
                'long_name': 'Longitude',
                'units': 'degrees_east',
                'grid_mapping': 'ease_grid_mapping',
            }),
        },
        coords={
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
        },
    )

    # Grid mapping variable
    ds['ease_grid_mapping'] = xr.DataArray(data=0, attrs=gm_attrs)

    # Global attributes
    ds.attrs.update({
        'title': f'Static EASE grid data ({label})',
        'grid_resolution': f"{cfg['grid']['resolution_km']} km",
        'grid_resolution_meters': res_m,
        'grid_size': f'{n_x} x {n_y}',
        'projection': 'Lambert Azimuthal Equal Area (Arctic)',
        'proj4_string': proj4,
        'conventions': 'CF-1.8',
        'ocean_shapefile': str(shp_path),
        'gebco_source': str(gebco_path),
    })

    # Statistics
    n_ocean = int(np.sum(ocean_mask == 1))
    total = ocean_mask.size
    print(f"\nOcean mask statistics:")
    print(f"  Ocean pixels: {n_ocean} ({100 * n_ocean / total:.1f}%)")
    print(f"  Land pixels:  {total - n_ocean} ({100 * (total - n_ocean) / total:.1f}%)")
    print(f"  Total pixels: {total}")

    return ds


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Step A — Create ocean mask & bathymetry on EASE grid",
    )
    parser.add_argument(
        '--config', required=True,
        help='Path to the pipeline YAML config file',
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    plan = load_pipeline_plan(cfg)
    if not plan.get('run_step_A', True):
        out_path = get_static_data_path(cfg)
        print(f"Step A: reusing existing static data → {out_path}")
        return

    ds = create_static_dataset(cfg)

    out_path = get_static_data_path(cfg)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    encoding = {v: {'zlib': True, 'complevel': 4}
                for v in ds.data_vars if v != 'ease_grid_mapping'}
    print(f"\nSaving → {out_path}")
    ds.to_netcdf(out_path, encoding=encoding)
    print("Done.")


if __name__ == '__main__':
    main()
