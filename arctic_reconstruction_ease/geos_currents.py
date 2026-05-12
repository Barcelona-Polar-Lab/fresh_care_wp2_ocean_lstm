"""
Geostrophic-current helpers used by Step E.

All functions operate on a single time step (no time dimension) so they
can be called from inside ``finalize_single_date``:

    ADH, vel_gos_x, vel_gos_y, u_gos, v_gos = compute_geostrophic_currents(
        T_recon, S_recon, ADT, depth, lat2d, lon2d, x_ease, y_ease)

Inputs
------
T_recon, S_recon : (nz, ny, nx) float
    In-situ temperature [degC] and practical salinity [PSU].
ADT              : (ny, nx) float
    Absolute dynamic topography [m].
depth            : (nz,) float, positive-down [m].
lat2d, lon2d     : (ny, nx) float, degrees.
x_ease, y_ease   : (nx,), (ny,) float, EASE-grid coordinates in meters.

Returns
-------
ADH                  : (nz, ny, nx) float32, meters
vel_gos_x, vel_gos_y : (nz, ny, nx) float32, m s-1, on EASE axes
u_gos, v_gos         : (nz, ny, nx) float32, m s-1, geographic east/north
"""

from __future__ import annotations

import gsw
import numpy as np
from scipy.ndimage import binary_erosion


# Physical constants
G_GSW = 9.7963            # gravity used inside gsw (m s-2)
OMEGA = 7.2921e-5         # Earth rotation rate (rad s-1)
F_FLOOR = 1e-10           # mask |f|<floor (safety; never triggers near poles)


def compute_SH(T, S, depth, lat2d, lon2d, g=G_GSW):
    """Steric height SH(z) = -dyn_height(z)/g, shape (nz, ny, nx) [m]."""
    p3d = gsw.p_from_z(-depth[:, None, None], lat2d[None, :, :])
    with np.errstate(invalid="ignore"):
        SA = gsw.SA_from_SP(S, p3d, lon2d[None, :, :], lat2d[None, :, :])
        pt = gsw.pt_from_t(SA, T, p3d, 0)
        CT = gsw.CT_from_pt(SA, pt)
        del pt
        dyn_h = gsw.geo_strf_dyn_height(SA, CT, p3d, p_ref=0, axis=0)
    return (-dyn_h / g).astype(np.float32)


def compute_geos_ease(ADH, dx, dy, f2d, g=G_GSW, f_floor=F_FLOOR):
    """Geostrophic velocities on EASE axes from ADH(nz, ny, nx)."""
    # np.gradient axis order: (y, x) = (-2, -1)
    dAdy, dAdx = np.gradient(ADH, dy, dx, axis=(-2, -1))
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_f = np.where(np.abs(f2d) > f_floor, 1.0 / f2d, np.nan)
        vel_x = -g * inv_f[None, :, :] * dAdy
        vel_y = g * inv_f[None, :, :] * dAdx
    return vel_x.astype(np.float32), vel_y.astype(np.float32)


def rotate_to_lonlat(vel_x, vel_y, lon2d_deg):
    """Rotate EASE-grid (x, y) components to geographic (east, north).

    For polar LAEA with lon_0=0, at a point of longitude L:
        east_hat  = ( cos L,  sin L)
        north_hat = (-sin L,  cos L)
    """
    L = np.deg2rad(lon2d_deg)
    cL, sL = np.cos(L)[None, :, :], np.sin(L)[None, :, :]
    u = cL * vel_x + sL * vel_y
    v = -sL * vel_x + cL * vel_y
    return u.astype(np.float32), v.astype(np.float32)


def compute_geostrophic_currents(T_recon, S_recon, ADT, depth,
                                 lat2d, lon2d, x_ease, y_ease,
                                 coast_buffer_cells=0):
    """One-shot helper: returns (ADH, vel_gos_x, vel_gos_y, u_gos, v_gos).

    Parameters
    ----------
    coast_buffer_cells : int, default 0
        If > 0, NaN-out velocity cells within this many cells of any land
        / seabed boundary (per depth level), to suppress edge artifacts
        from np.gradient at sharp ADH discontinuities. ADH itself is left
        untouched.
    """
    T = np.asarray(T_recon, dtype=np.float64)
    S = np.asarray(S_recon, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    lat2d = np.asarray(lat2d, dtype=np.float64)
    lon2d = np.asarray(lon2d, dtype=np.float64)

    SH = compute_SH(T, S, depth, lat2d, lon2d)
    ADH = (ADT[None, :, :].astype(np.float32) - SH)

    dx = float(np.mean(np.diff(np.asarray(x_ease, dtype=np.float64))))
    dy = float(np.mean(np.diff(np.asarray(y_ease, dtype=np.float64))))
    f2d = (2.0 * OMEGA * np.sin(np.deg2rad(lat2d))).astype(np.float64)

    vel_x, vel_y = compute_geos_ease(ADH, dx, dy, f2d)
    u, v = rotate_to_lonlat(vel_x, vel_y, lon2d)

    if coast_buffer_cells and coast_buffer_cells > 0:
        # Per-depth lateral erosion of the velocity wet mask: NaN cells
        # within `coast_buffer_cells` of any land or seabed boundary.
        # We erode the velocity mask (already ~1 cell smaller than ADH's
        # because np.gradient propagates NaNs by 1 cell), so the buffer
        # acts relative to where velocities are actually defined.
        # ADH itself is left untouched.
        wet = np.isfinite(vel_x)
        for d in range(vel_x.shape[0]):
            if not wet[d].any():
                continue
            keep = binary_erosion(wet[d], iterations=coast_buffer_cells)
            mask = ~keep
            vel_x[d][mask] = np.nan
            vel_y[d][mask] = np.nan
            u[d][mask] = np.nan
            v[d][mask] = np.nan

    return ADH, vel_x, vel_y, u, v
