"""
Toy Example 1: Convert EASE Grid Coordinates to Lat/Lon
Shows how to transform target EASE coordinates to latitude/longitude
"""

import numpy as np
import pyproj

# ============================================================================
# EASE GRID CONFIGURATION
# ============================================================================
EASE_LAT_0 = 90
EASE_LON_0 = 0
EASE_FALSE_EASTING = 0
EASE_FALSE_NORTHING = 0
EASE_PROJ4 = f"+proj=laea +lat_0={EASE_LAT_0} +lon_0={EASE_LON_0} +x_0={EASE_FALSE_EASTING} +y_0={EASE_FALSE_NORTHING} +datum=WGS84 +units=m"

# Grid parameters
GRID_RESOLUTION_M = 25000  # 25 km
GRID_SIZE_X = 350
GRID_SIZE_Y = 350

# ============================================================================
# CREATE EASE GRID IN METERS
# ============================================================================
x_min = -(GRID_SIZE_X * GRID_RESOLUTION_M) / 2
y_min = -(GRID_SIZE_Y * GRID_RESOLUTION_M) / 2

x_ease = np.arange(GRID_SIZE_X) * GRID_RESOLUTION_M + x_min + GRID_RESOLUTION_M / 2
y_ease = np.arange(GRID_SIZE_Y) * GRID_RESOLUTION_M + y_min + GRID_RESOLUTION_M / 2

print(f"EASE Grid X range: {x_ease.min()/1e6:.2f} to {x_ease.max()/1e6:.2f} (million meters)")
print(f"EASE Grid Y range: {y_ease.min()/1e6:.2f} to {y_ease.max()/1e6:.2f} (million meters)")

# ============================================================================
# TRANSFORM EASE COORDINATES TO LAT/LON
# ============================================================================

# Create inverse transformer: EASE -> WGS84 (lat/lon)
transformer_inv = pyproj.Transformer.from_crs(
    pyproj.CRS.from_proj4(EASE_PROJ4),
    pyproj.CRS.from_epsg(4326),
    always_xy=True  # Important: means we work with (x, y) not (lat, lon)
)

# Create 2D meshgrid of EASE coordinates
x_ease_2d, y_ease_2d = np.meshgrid(x_ease, y_ease)
print(f"\n2D EASE grid shape: {x_ease_2d.shape}")

# Transform to lat/lon
lon_target, lat_target = transformer_inv.transform(x_ease_2d, y_ease_2d)
print(f"2D Lat/Lon grid shape: {lat_target.shape}")

print(f"\nLatitude range:  {lat_target.min():.2f} to {lat_target.max():.2f} degrees")
print(f"Longitude range: {lon_target.min():.2f} to {lon_target.max():.2f} degrees")

# ============================================================================
# EXAMPLE: GET LAT/LON FOR SPECIFIC EASE INDICES
# ============================================================================

# Get lat/lon at specific grid indices
idx_x, idx_y = 175, 175  # Center of grid
print(f"\nGrid indices: x={idx_x}, y={idx_y}")
print(f"EASE coordinates: x={x_ease_2d[idx_y, idx_x]:.0f} m, y={y_ease_2d[idx_y, idx_x]:.0f} m")
print(f"Lat/Lon: lat={lat_target[idx_y, idx_x]:.4f}°, lon={lon_target[idx_y, idx_x]:.4f}°")

# Get lat/lon at several points
indices = [(0, 0), (174, 174), (349, 349)]
print("\nLat/Lon at corner points:")
for idx_y, idx_x in indices:
    print(f"  Grid[{idx_y}, {idx_x}]: {lat_target[idx_y, idx_x]:.4f}°N, {lon_target[idx_y, idx_x]:.4f}°E")
