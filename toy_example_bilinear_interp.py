"""
Toy Example 2: Bilinear Interpolation with SciPy
Shows how to use RegularGridInterpolator for bilinear interpolation
"""

import numpy as np
from scipy.interpolate import RegularGridInterpolator

# ============================================================================
# CREATE SYNTHETIC DATA ON A REGULAR LAT/LON GRID
# ============================================================================

# Original coordinates (1D arrays, must be increasing)
lat_original = np.linspace(60, 85, 10)  # 10 latitude points
lon_original = np.linspace(-30, 30, 12)  # 12 longitude points

print("Original grid:")
print(f"  Latitude: {lat_original[0]:.1f}° to {lat_original[-1]:.1f}° ({len(lat_original)} points)")
print(f"  Longitude: {lon_original[0]:.1f}° to {lon_original[-1]:.1f}° ({len(lon_original)} points)")

# Create synthetic 2D data (e.g., sea surface temperature)
# Shape is (n_lat, n_lon)
lat_2d, lon_2d = np.meshgrid(lat_original, lon_original, indexing='ij')
data_original = 10 + 0.5 * lat_original[:, np.newaxis] + 0.3 * np.sin(lon_2d)

print(f"\nOriginal data shape: {data_original.shape} (lat={len(lat_original)}, lon={len(lon_original)})")
print(f"Original data range: {data_original.min():.2f} to {data_original.max():.2f}")

# ============================================================================
# CREATE INTERPOLATOR OBJECT
# ============================================================================

# Create the interpolator
# Note: coordinates must be in INCREASING order (already are in this example)
interpolator = RegularGridInterpolator(
    (lat_original, lon_original),  # (lat, lon) coordinates
    data_original,                  # data values
    method='linear',                # bilinear interpolation
    bounds_error=False,             # don't raise error outside bounds
    fill_value=np.nan               # use NaN for out-of-bounds points
)

print("\nInterpolator created successfully")

# ============================================================================
# INTERPOLATE AT NEW POINTS
# ============================================================================

# Method 1: Interpolate at a few individual points
print("\n" + "="*60)
print("METHOD 1: Interpolate at individual points")
print("="*60)

# Points as 2D array: shape (n_points, 2) where each row is (lat, lon)
points = np.array([
    [65.0,  0.0],    # Point 1
    [70.5, 10.5],    # Point 2
    [75.3, -15.2],   # Point 3
])

interpolated_values = interpolator(points)
print(f"\nPoints to interpolate:\n{points}")
print(f"\nInterpolated values:\n{interpolated_values}")

# ============================================================================
# INTERPOLATE ON A REGULAR NEW GRID (MORE REFINED)
# ============================================================================

print("\n" + "="*60)
print("METHOD 2: Interpolate on a finer regular grid")
print("="*60)

# Create a finer target grid
lat_target = np.linspace(62, 84, 40)  # Finer latitude grid
lon_target = np.linspace(-25, 25, 50)  # Finer longitude grid

print(f"Target grid:")
print(f"  Latitude: {lat_target[0]:.1f}° to {lat_target[-1]:.1f}° ({len(lat_target)} points)")
print(f"  Longitude: {lon_target[0]:.1f}° to {lon_target[-1]:.1f}° ({len(lon_target)} points)")

# Create 2D meshgrid of target coordinates
lat_target_2d, lon_target_2d = np.meshgrid(lat_target, lon_target, indexing='ij')

# Stack coordinates for interpolator input: (n_points, 2) array
# Flatten the 2D grid, stack, then reshape back
points_flat = np.stack([lat_target_2d.ravel(), lon_target_2d.ravel()], axis=-1)

# Interpolate
data_interpolated_flat = interpolator(points_flat)

# Reshape back to 2D grid
data_interpolated = data_interpolated_flat.reshape(lat_target_2d.shape)

print(f"\nInterpolated data shape: {data_interpolated.shape}")
print(f"Interpolated data range: {np.nanmin(data_interpolated):.2f} to {np.nanmax(data_interpolated):.2f}")

# ============================================================================
# COMPARE ORIGINAL AND INTERPOLATED
# ============================================================================

print("\n" + "="*60)
print("COMPARISON: Original vs Interpolated")
print("="*60)

# Sample a point that exists in original grid
idx_lat_orig, idx_lon_orig = 5, 6
lat_sample = lat_original[idx_lat_orig]
lon_sample = lon_original[idx_lon_orig]
value_original = data_original[idx_lat_orig, idx_lon_orig]

# Find nearest in interpolated grid
idx_target = np.argmin(np.abs(lat_target - lat_sample))
jdx_target = np.argmin(np.abs(lon_target - lon_sample))
value_interpolated = data_interpolated[idx_target, jdx_target]

print(f"\nSample point: lat={lat_sample:.2f}°, lon={lon_sample:.2f}°")
print(f"Original value:      {value_original:.6f}")
print(f"Interpolated value:  {value_interpolated:.6f}")
print(f"Difference:          {abs(value_original - value_interpolated):.6f}")

# ============================================================================
# TEST: POINT OUTSIDE BOUNDS
# ============================================================================

print("\n" + "="*60)
print("TEST: Interpolation outside bounds")
print("="*60)

out_of_bounds_point = np.array([[90.0, 0.0]])  # Outside domain
value_oob = interpolator(out_of_bounds_point)
print(f"\nPoint outside bounds: lat=90.0°, lon=0.0°")
print(f"Interpolated value: {value_oob[0]} (should be NaN)")
