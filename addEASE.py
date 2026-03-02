#!/usr/bin/env python3
"""
Add EASE Grid 2.0 coordinates (X_EASE, Y_EASE) to monthly profile files.

This script processes NetCDF files and adds EASE Grid 2.0 North coordinates based on the LATITUDE and LONGITUDE variables using Lambert Azimuthal Equal Area projection.
"""

import os
import numpy as np
import xarray as xr
from pathlib import Path
from pyproj import Transformer
from tqdm import tqdm

# Global configuration variables
INPUT_DIR = "fresh_data"
OUTPUT_DIR = "fresh_data_ease"

#INPUT_DIR = "/home/nico/SACO/FRESH-CARE/Data_in_situ/MERGED_PROFILES/data_with_SH"
#INPUT_DIR = TEST_INPUT_DIR  # For testing purposes
#OUTPUT_DIR = "/home/nico/SACO/FRESH-CARE/Data_in_situ/MERGED_PROFILES/data_with_SH_EASE"
#OUTPUT_DIR = TEST_OUTPUT_DIR  # For testing purposes

# EASE Grid 2.0 North projection parameters
proj_type = 'laea'
central_lat = 90      # North Pole
central_lon = 0
false_easting = 0
false_northing = 0


def main():
    """Main function to add EASE coordinates to all NetCDF files in the input directory."""
    print("=== Adding EASE Grid 2.0 coordinates to monthly files ===")
    
    # Create output directory if it doesn't exist
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    # Find all NetCDF files in input directory
    input_path = Path(INPUT_DIR)
    if not input_path.exists():
        print(f"ERROR: Input directory not found: {INPUT_DIR}")
        return
    
    nc_files = list(input_path.glob("*.nc"))
    if not nc_files:
        print(f"No .nc files found in {INPUT_DIR}")
        return
    
    print(f"Found {len(nc_files)} NetCDF files to process")
    
    # Build projection string
    proj_string = (
        f'+proj={proj_type} '
        f'+lat_0={central_lat} '
        f'+lon_0={central_lon} '
        f'+x_0={false_easting} '
        f'+y_0={false_northing} '
        f'+datum=WGS84 +units=m'
    )
    
    print(f"Using projection: {proj_string}")
    
    # Create transformer
    transformer = Transformer.from_crs('EPSG:4326', proj_string, always_xy=True)
    
    # Process each file
    for nc_file in tqdm(nc_files, desc="Processing files"):
        try:
            output_file = output_path / nc_file.name
            process_file(nc_file, output_file, transformer)
        except Exception as e:
            print(f"  ERROR processing {nc_file.name}: {e}")
    
    print("=== EASE coordinate processing complete ===")


def process_file(input_file_path, output_file_path, transformer):
    """Process a single NetCDF file to add EASE coordinates."""
    # Open dataset
    ds = xr.open_dataset(input_file_path, decode_times=False)
    
    try:
        # Check that LATITUDE and LONGITUDE exist
        if 'LATITUDE' not in ds:
            raise ValueError("LATITUDE variable not found in dataset")
        if 'LONGITUDE' not in ds:
            raise ValueError("LONGITUDE variable not found in dataset")
        
        # Check that LATITUDE and LONGITUDE depend on a single dimension
        lat_sizes = ds['LATITUDE'].sizes
        lon_sizes = ds['LONGITUDE'].sizes
        
        if len(lat_sizes) != 1:
            raise ValueError(f"LATITUDE must depend on a single dimension, found: {list(lat_sizes.keys())}")
        if len(lon_sizes) != 1:
            raise ValueError(f"LONGITUDE must depend on a single dimension, found: {list(lon_sizes.keys())}")
        if lat_sizes != lon_sizes:
            raise ValueError(f"LATITUDE and LONGITUDE must have the same dimensions. "
                           f"LATITUDE: {list(lat_sizes.keys())}, LONGITUDE: {list(lon_sizes.keys())}")
        
        # Get dimension name
        coord_dim = list(lat_sizes.keys())[0]
        
        # Get coordinate values
        lats = ds['LATITUDE'].values
        lons = ds['LONGITUDE'].values
        
        # Transform coordinates to EASE grid
        x_coords, y_coords = transformer.transform(lons, lats)
        
        # Convert to integers (1 meter resolution is sufficient)
        x_coords = x_coords.astype(np.int32)
        y_coords = y_coords.astype(np.int32)
        
        # Add EASE coordinates to dataset
        ds['X_EASE'] = (coord_dim, x_coords)
        ds['X_EASE'].attrs = {
            'long_name': 'EASE Grid 2.0 North X coordinate',
            'units': 'm',
            'standard_name': 'projection_x_coordinate',
            'description': 'X coordinate in EASE Grid 2.0 North (25km) projection computed from LONGITUDE'
        }
        
        ds['Y_EASE'] = (coord_dim, y_coords)
        ds['Y_EASE'].attrs = {
            'long_name': 'EASE Grid 2.0 North Y coordinate', 
            'units': 'm',
            'standard_name': 'projection_y_coordinate',
            'description': 'Y coordinate in EASE Grid 2.0 North (25km) projection computed from LATITUDE'
        }
        
        # Create encoding for new variables
        encoding = {}
        for var in ds.data_vars:
            if var in ['X_EASE', 'Y_EASE']:
                encoding[var] = {
                    'zlib': True,
                    'complevel': 1,
                    'shuffle': True
                }
        
        # Save to temporary file in output directory
        temp_file = output_file_path.with_suffix('.tmp')
        ds.to_netcdf(temp_file, encoding=encoding)
        
        # Close dataset
        ds.close()
        
        # Move temp file to final output location
        temp_file.replace(output_file_path)
        
    except Exception as e:
        ds.close()
        # Clean up temp file if it exists
        temp_file = output_file_path.with_suffix('.tmp')
        if temp_file.exists():
            temp_file.unlink()
        raise e


if __name__ == "__main__":
    main()
