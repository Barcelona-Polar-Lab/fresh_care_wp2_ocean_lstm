#!/usr/bin/env python3
"""
Create grid reference file for ROMS grid reconstruction pipeline.

Extracts from ROMS grid file:
- lat_rho, lon_rho: Geographic coordinates at tracer points
- mask_rho: Land/water mask (0=land, 1=water)
- h: Bathymetry (meters)
- ease_x, ease_y: EASE grid coordinates computed for each ROMS point (for model input)

The EASE coordinates are computed by projecting each ROMS lat/lon point to the 
EASE Lambert Azimuthal Equal Area projection. These are used as model input features,
not as the output grid.
"""

import numpy as np
import xarray as xr
import pyproj
import argparse
from pathlib import Path


# ============================================================================
# EASE GRID PROJECTION PARAMETERS (for computing EASE coordinates)
# ============================================================================

# EASE Grid Projection Parameters (Arctic Lambert Azimuthal Equal Area)
# These must match the training data projection
EASE_LAT_0 = 90            # latitude of projection origin (North Pole)
EASE_LON_0 = 0             # longitude of projection origin
EASE_FALSE_EASTING = 0     # false easting
EASE_FALSE_NORTHING = 0    # false northing

# Derived PROJ4 string for EASE grid
EASE_PROJ4 = f"+proj=laea +lat_0={EASE_LAT_0} +lon_0={EASE_LON_0} +x_0={EASE_FALSE_EASTING} +y_0={EASE_FALSE_NORTHING} +datum=WGS84 +units=m"

# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_ROMS_GRID = '/home/nico/SACO/FRESH-CARE/FC-model/FC_grid_2km_roms_z.nc'
DEFAULT_OUTPUT = '/home/nico/SACO/FRESH-CARE/Data_lstm_reconstruction_ROMSgrid/data_for_reconstruction/roms_grid_reference.nc'


def compute_ease_coordinates(lat_2d, lon_2d):
    """
    Compute EASE grid X/Y coordinates for each lat/lon point.
    
    Parameters:
    -----------
    lat_2d : np.ndarray
        2D array of latitudes (eta_rho, xi_rho)
    lon_2d : np.ndarray
        2D array of longitudes (eta_rho, xi_rho)
    
    Returns:
    --------
    tuple
        (ease_x, ease_y) 2D arrays of EASE coordinates in meters
    """
    # Create transformer from WGS84 to EASE
    crs_wgs84 = pyproj.CRS.from_epsg(4326)
    crs_ease = pyproj.CRS.from_proj4(EASE_PROJ4)
    
    transformer = pyproj.Transformer.from_crs(
        crs_wgs84, crs_ease, always_xy=True
    )
    
    # Transform all points
    ease_x, ease_y = transformer.transform(lon_2d, lat_2d)
    
    return ease_x, ease_y


def create_grid_reference(roms_grid_path, output_path=None):
    """
    Create grid reference dataset from ROMS grid file.
    
    Parameters:
    -----------
    roms_grid_path : str or Path
        Path to ROMS grid file
    output_path : str or Path, optional
        Path to save output NetCDF file. If None, returns dataset without saving.
    
    Returns:
    --------
    xr.Dataset
        Dataset with ROMS grid coordinates, mask, bathymetry, and EASE coordinates
    """
    print(f"Reading ROMS grid from {roms_grid_path}")
    ds_roms = xr.open_dataset(roms_grid_path)
    
    # Extract relevant variables
    lat_rho = ds_roms['lat_rho'].values  # (eta_rho, xi_rho)
    lon_rho = ds_roms['lon_rho'].values
    mask_rho = ds_roms['mask_rho'].values
    h = ds_roms['h'].values  # bathymetry
    
    # Get dimensions
    eta_rho, xi_rho = lat_rho.shape
    print(f"ROMS grid shape: eta_rho={eta_rho}, xi_rho={xi_rho}")
    print(f"Total points: {eta_rho * xi_rho:,}")
    
    # Compute EASE coordinates for each point
    print("Computing EASE coordinates for all ROMS points...")
    ease_x, ease_y = compute_ease_coordinates(lat_rho, lon_rho)
    
    print(f"EASE X range: [{ease_x.min():.0f}, {ease_x.max():.0f}] m")
    print(f"EASE Y range: [{ease_y.min():.0f}, {ease_y.max():.0f}] m")
    
    # Create coordinate arrays for dimensions
    eta_rho_coord = np.arange(eta_rho)
    xi_rho_coord = np.arange(xi_rho)
    
    # Create dataset
    ds = xr.Dataset(
        coords={
            'eta_rho': ('eta_rho', eta_rho_coord, {
                'long_name': 'eta index of rho-points',
                'units': '1'
            }),
            'xi_rho': ('xi_rho', xi_rho_coord, {
                'long_name': 'xi index of rho-points',
                'units': '1'
            })
        }
    )
    
    # Add latitude
    ds['lat_rho'] = xr.DataArray(
        lat_rho,
        dims=['eta_rho', 'xi_rho'],
        attrs={
            'long_name': 'latitude of rho-points',
            'units': 'degrees_north',
            'standard_name': 'latitude'
        }
    )
    
    # Add longitude
    ds['lon_rho'] = xr.DataArray(
        lon_rho,
        dims=['eta_rho', 'xi_rho'],
        attrs={
            'long_name': 'longitude of rho-points',
            'units': 'degrees_east',
            'standard_name': 'longitude'
        }
    )
    
    # Add mask
    ds['mask_rho'] = xr.DataArray(
        mask_rho.astype(np.int8),
        dims=['eta_rho', 'xi_rho'],
        attrs={
            'long_name': 'mask at rho-points',
            'units': 'land/water (0/1)',
            'flag_values': [0, 1],
            'flag_meanings': 'land water'
        }
    )
    
    # Add bathymetry
    ds['h'] = xr.DataArray(
        h,
        dims=['eta_rho', 'xi_rho'],
        attrs={
            'long_name': 'bathymetry at rho-points',
            'units': 'm',
            'positive': 'down'
        }
    )
    
    # Add EASE coordinates
    ds['ease_x'] = xr.DataArray(
        ease_x,
        dims=['eta_rho', 'xi_rho'],
        attrs={
            'long_name': 'EASE-Grid X coordinate',
            'units': 'm',
            'description': 'X coordinate in EASE Lambert Azimuthal Equal Area projection',
            'projection': EASE_PROJ4
        }
    )
    
    ds['ease_y'] = xr.DataArray(
        ease_y,
        dims=['eta_rho', 'xi_rho'],
        attrs={
            'long_name': 'EASE-Grid Y coordinate',
            'units': 'm',
            'description': 'Y coordinate in EASE Lambert Azimuthal Equal Area projection',
            'projection': EASE_PROJ4
        }
    )
    
    # Global attributes
    ds.attrs = {
        'title': 'ROMS Grid Reference for LSTM Reconstruction',
        'source': str(roms_grid_path),
        'grid_type': 'curvilinear',
        'projection_for_ease_coords': 'Lambert Azimuthal Equal Area (Arctic)',
        'ease_projection_latitude_of_origin': EASE_LAT_0,
        'ease_projection_longitude_of_origin': EASE_LON_0,
        'ease_proj4_string': EASE_PROJ4,
        'conventions': 'CF-1.8',
        'history': f'Created from ROMS grid file'
    }
    
    # Print statistics
    ocean_count = np.sum(mask_rho == 1)
    land_count = np.sum(mask_rho == 0)
    total = mask_rho.size
    print(f"\nGrid statistics:")
    print(f"  Ocean pixels: {ocean_count:,} ({100*ocean_count/total:.1f}%)")
    print(f"  Land pixels:  {land_count:,} ({100*land_count/total:.1f}%)")
    print(f"  Total pixels: {total:,}")
    print(f"  Latitude range: [{lat_rho.min():.2f}, {lat_rho.max():.2f}]")
    print(f"  Longitude range: [{lon_rho.min():.2f}, {lon_rho.max():.2f}]")
    print(f"  Bathymetry range: [{np.nanmin(h):.1f}, {np.nanmax(h):.1f}] m")
    
    # Close input
    ds_roms.close()
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Encoding with compression
        encoding = {
            'lat_rho': {'dtype': 'float64', 'zlib': True, 'complevel': 4},
            'lon_rho': {'dtype': 'float64', 'zlib': True, 'complevel': 4},
            'mask_rho': {'dtype': 'int8'},
            'h': {'dtype': 'float32', 'zlib': True, 'complevel': 4},
            'ease_x': {'dtype': 'float64', 'zlib': True, 'complevel': 4},
            'ease_y': {'dtype': 'float64', 'zlib': True, 'complevel': 4},
        }
        
        print(f"\nSaving to {output_path}")
        ds.to_netcdf(output_path, encoding=encoding)
        print("Done!")
    
    return ds


def main():
    parser = argparse.ArgumentParser(
        description='Create grid reference file for ROMS grid reconstruction'
    )
    parser.add_argument(
        '--roms_grid',
        type=str,
        default=DEFAULT_ROMS_GRID,
        help='Path to ROMS grid file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=DEFAULT_OUTPUT,
        help='Output NetCDF file path'
    )
    
    args = parser.parse_args()
    
    # Create grid reference
    ds = create_grid_reference(args.roms_grid, args.output)
    
    return ds


if __name__ == '__main__':
    main()
