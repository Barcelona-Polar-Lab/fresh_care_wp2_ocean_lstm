#!/usr/bin/env python3
"""
Create an ocean mask on EASE grid using Natural Earth ocean data.
Generates EASE grid structure from configuration parameters.
"""

import numpy as np
import xarray as xr
import geopandas as gpd
from rasterio import features
from affine import Affine
import argparse
from pathlib import Path


# ============================================================================
# EASE GRID CONFIGURATION PARAMETERS
# ============================================================================

# Grid Resolution and Size
GRID_RESOLUTION_M = 25000  # meters (25 km)
GRID_SIZE_X = 350          # number of grid cells in X direction
GRID_SIZE_Y = 350          # number of grid cells in Y direction

# EASE Grid Projection Parameters (Arctic Lambert Azimuthal Equal Area)
EASE_LAT_0 = 90            # latitude of projection origin (North Pole)
EASE_LON_0 = 0             # longitude of projection origin
EASE_FALSE_EASTING = 0     # false easting
EASE_FALSE_NORTHING = 0    # false northing

# Derived PROJ4 string for EASE grid
EASE_PROJ4 = f"+proj=laea +lat_0={EASE_LAT_0} +lon_0={EASE_LON_0} +x_0={EASE_FALSE_EASTING} +y_0={EASE_FALSE_NORTHING} +datum=WGS84 +units=m"

# ============================================================================

DEFAULT_OUTPUT = '/home/nico/SACO/FRESH-CARE/Arctic_masks/natural_earth_data/ne_10m_ocean_EASE_masks/ocean_mask_ease_grid_25km.nc'
DEFAULT_SHAPE_FILE = '/home/nico/SACO/FRESH-CARE/Arctic_masks/natural_earth_data/ne_10m_ocean/ne_10m_ocean.shp'


def create_ease_grid_structure():
    """
    Create EASE grid coordinates and metadata from configuration parameters.
    
    Returns:
    --------
    tuple
        (x_ease, y_ease, grid_mapping_attrs) where:
        - x_ease: 1D array of X coordinates (meters)
        - y_ease: 1D array of Y coordinates (meters)
        - grid_mapping_attrs: dict of CF-compliant grid mapping attributes
    """
    # Calculate grid extent (centered on projection origin)
    x_min = -(GRID_SIZE_X * GRID_RESOLUTION_M) / 2
    y_min = -(GRID_SIZE_Y * GRID_RESOLUTION_M) / 2
    
    # Create coordinate arrays
    x_ease = np.arange(GRID_SIZE_X) * GRID_RESOLUTION_M + x_min + GRID_RESOLUTION_M / 2
    y_ease = np.arange(GRID_SIZE_Y) * GRID_RESOLUTION_M + y_min + GRID_RESOLUTION_M / 2
    
    # Create CF-compliant grid mapping attributes
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


def create_ease_transform(x_ease, y_ease):
    """
    Create an affine transform for the EASE grid.
    
    Parameters:
    -----------
    x_ease : array
        X coordinates in EASE grid (in meters)
    y_ease : array
        Y coordinates in EASE grid (in meters)
    
    Returns:
    --------
    Affine
        Affine transform for rasterization
    """
    # Calculate pixel size from coordinate spacing
    dx = x_ease[1] - x_ease[0] if len(x_ease) > 1 else 25000
    dy = y_ease[1] - y_ease[0] if len(y_ease) > 1 else 25000
    
    # Since y_ease is in increasing order and we want row 0 to correspond to y_ease[0],
    # we use the minimum y value and positive dy
    # This matches the xarray/NetCDF storage convention
    x_min = x_ease[0] - dx / 2
    y_min = y_ease[0] - dy / 2
    
    # Create affine transform
    # Positive dy means rows increase upward (matching y_ease ordering)
    transform = Affine(dx, 0, x_min,
                      0, dy, y_min)
    
    return transform


def rasterize_ocean_shapefile(shapefile_path, transform, shape, x_ease, y_ease):
    """
    Rasterize ocean shapefile to create ocean mask.
    
    Parameters:
    -----------
    shapefile_path : str or Path
        Path to the ocean shapefile
    transform : Affine
        Affine transform for the target grid
    shape : tuple
        Shape of the output array (ny, nx)
    x_ease : array
        X coordinates of the EASE grid
    y_ease : array
        Y coordinates of the EASE grid
    
    Returns:
    --------
    np.ndarray
        Ocean mask (1 = ocean, 0 = land/other)
    """
    from shapely.geometry import box
    
    # Read the shapefile
    print(f"Reading ocean shapefile from {shapefile_path}")
    gdf = gpd.read_file(shapefile_path)
    
    # Create bounding box for the EASE grid (with some buffer)
    buffer = 100000  # 100 km buffer
    x_min, x_max = x_ease.min() - buffer, x_ease.max() + buffer
    y_min, y_max = y_ease.min() - buffer, y_ease.max() + buffer
    
    # Reproject to EASE grid projection (Lambert Azimuthal Equal Area, Arctic)
    print(f"Reprojecting from {gdf.crs} to EASE grid projection")
    gdf_ease = gdf.to_crs(EASE_PROJ4)
    
    # Fix invalid geometries (common after polar projection transformations)
    invalid_count = (~gdf_ease.geometry.is_valid).sum()
    if invalid_count > 0:
        print(f"Fixing {invalid_count}/{len(gdf_ease)} geometries invalidated by reprojection (normal for polar projections)")
        gdf_ease['geometry'] = gdf_ease.geometry.buffer(0)
    
    # Clip to EASE grid extent
    clip_box = box(x_min, y_min, x_max, y_max)
    print(f"Clipping to EASE grid extent: x=[{x_min:.0f}, {x_max:.0f}], y=[{y_min:.0f}, {y_max:.0f}]")
    gdf_ease = gdf_ease.clip(clip_box)
    
    print(f"After clipping: {len(gdf_ease)} features")
    
    # Rasterize: where ocean polygons exist, set to 1
    print(f"Rasterizing ocean mask to shape {shape}")
    ocean_mask = features.rasterize(
        ((geom, 1) for geom in gdf_ease.geometry),
        out_shape=shape,
        transform=transform,
        fill=0,
        dtype=np.uint8,
        all_touched=True  # Include pixels touched by ocean polygons
    )
    
    return ocean_mask


def create_ocean_mask_dataset(shapefile_path, output_path=None):
    """
    Create ocean mask dataset on EASE grid from ocean shapefile.
    
    Parameters:
    -----------
    shapefile_path : str or Path
        Path to ocean shapefile
    output_path : str or Path, optional
        Path to save output NetCDF file. If None, returns dataset without saving.
    
    Returns:
    --------
    xr.Dataset
        Dataset with EASE grid coordinates and ocean_mask variable
    """
    # Create EASE grid structure from configuration parameters
    print(f"Creating EASE grid structure ({GRID_SIZE_X}x{GRID_SIZE_Y} at {GRID_RESOLUTION_M/1000:.1f} km resolution)")
    x_ease, y_ease, grid_mapping_attrs = create_ease_grid_structure()
    
    print(f"EASE grid shape: y_ease={len(y_ease)}, x_ease={len(x_ease)}")
    
    # Create affine transform
    transform = create_ease_transform(x_ease, y_ease)
    print(f"Affine transform:\n {transform}")
    
    # Rasterize ocean shapefile
    shape = (len(y_ease), len(x_ease))
    ocean_mask = rasterize_ocean_shapefile(shapefile_path, transform, shape, x_ease, y_ease)
    
    # Create dataset
    ds = xr.Dataset(
        {
            'ocean_mask': (('y_ease', 'x_ease'), ocean_mask,
                          {
                              'long_name': 'Ocean mask',
                              'description': 'Ocean mask: 1 = ocean, 0 = land/no data',
                              'source': f'Natural Earth ocean data rasterized to EASE grid',
                              'grid_mapping': 'ease_grid_mapping'
                          })
        },
        coords={
            'x_ease': ('x_ease', x_ease, {
                'standard_name': 'projection_x_coordinate',
                'units': f'{GRID_RESOLUTION_M} m',
                'long_name': 'EASE-Grid X coordinate',
                'axis': 'X'
            }),
            'y_ease': ('y_ease', y_ease, {
                'standard_name': 'projection_y_coordinate',
                'units': f'{GRID_RESOLUTION_M} m',
                'long_name': 'EASE-Grid Y coordinate',
                'axis': 'Y'
            })
        }
    )
    
    # Add grid mapping variable
    ds['ease_grid_mapping'] = xr.DataArray(
        data=0,
        attrs=grid_mapping_attrs
    )
    
    # Add global attributes
    ds.attrs['title'] = 'Ocean mask on EASE Grid'
    ds.attrs['source'] = f'Natural Earth ocean shapefile rasterized to EASE grid'
    ds.attrs['ocean_shapefile_source'] = str(shapefile_path)
    ds.attrs['grid_resolution'] = f'{GRID_RESOLUTION_M/1000:.1f} km'
    ds.attrs['grid_resolution_meters'] = GRID_RESOLUTION_M
    ds.attrs['grid_size'] = f'{GRID_SIZE_X} x {GRID_SIZE_Y}'
    ds.attrs['projection'] = 'Lambert Azimuthal Equal Area (Arctic)'
    ds.attrs['projection_latitude_of_origin'] = EASE_LAT_0
    ds.attrs['projection_longitude_of_origin'] = EASE_LON_0
    ds.attrs['projection_false_easting'] = EASE_FALSE_EASTING
    ds.attrs['projection_false_northing'] = EASE_FALSE_NORTHING
    ds.attrs['proj4_string'] = EASE_PROJ4
    ds.attrs['conventions'] = 'CF-1.8'
    
    # Print statistics
    ocean_count = np.sum(ocean_mask == 1)
    land_count = np.sum(ocean_mask == 0)
    total = ocean_mask.size
    print(f"\nOcean mask statistics:")
    print(f"  Ocean pixels: {ocean_count} ({100*ocean_count/total:.1f}%)")
    print(f"  Land pixels:  {land_count} ({100*land_count/total:.1f}%)")
    print(f"  Total pixels: {total}")
    
    # Save if output path provided
    if output_path:
        print(f"\nSaving ocean mask to {output_path}")
        ds.to_netcdf(output_path)
        print("Done!")
    
    return ds


def main():
    parser = argparse.ArgumentParser(
        description='Create ocean mask on EASE grid from Natural Earth data'
    )
    parser.add_argument(
        '--shapefile',
        type=str,
        default=DEFAULT_SHAPE_FILE,
        help='Path to Natural Earth ocean shapefile'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=DEFAULT_OUTPUT,
        help='Output NetCDF file path'
    )
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create ocean mask
    ds = create_ocean_mask_dataset(args.shapefile, args.output)
    
    return ds


if __name__ == '__main__':
    main()
