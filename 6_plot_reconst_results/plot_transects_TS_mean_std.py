#!/usr/bin/env python3
"""
2x2 figure of gateway T/S transects (full-record mean and std).

DEPENDENCY NOTE
---------------
Requires plot_transects_batch.py (same directory) as a config/utils module.
Imports: BASE_DIR, GATEWAYS, PLOT_LIMITS.

The local _transect_indices function is nearly identical to the one in
plot_transects_velocity.py (both are independent reimplementations of the
equivalent logic in plot_transects_batch).
The local _draw_section helper parallels plot_transects_batch.
plot_single_gateway_section, with differences in the colorbar extend= and
the absence of an asymmetric norm.

Panels:
    (a) T_recon_mean  (cmocean.thermal)
    (b) T_recon_std   (crameri.lajolla_r)
    (c) S_recon_mean  (cmocean.haline)
    (d) S_recon_std   (crameri.lajolla_r)

Colour ranges:
    * Mean panels: 2nd-98th percentile of transect values, extend="both".
    * Std panels:  0 to 98th percentile,                    extend="max".

Output:
    <reconst-dir>/plots_transects/<Gateway>_TS_recon_mean_std.png

Intended for fram_6p25 and barents_6p25; pass via --configs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cmcrameri import cm as cmc
from matplotlib.ticker import FormatStrFormatter, FuncFormatter, MaxNLocator
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist

from plot_transects_batch import BASE_DIR, GATEWAYS, PLOT_LIMITS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

mpl.rcParams["font.size"] = mpl.rcParams["font.size"] * 1.2

DEFAULT_CONFIGS = ["fram_6p25", "barents_6p25"]

CMAP_T_MEAN = cmocean.cm.thermal
CMAP_S_MEAN = cmocean.cm.haline
CMAP_STD = cmc.lajolla_r

LABEL_FS = 11
TITLE_FS = 11

XAXIS_TICKS = {
    "barents_6p25": list(range(71, 78)),
}

CBAR_LABELS = {
    "T_recon_mean": r"$\mathrm{T}$ (°C)",
    "T_recon_std":  r"$\mathrm{\sigma_{T}}$ (°C)",
    "S_recon_mean": r"$\mathrm{S}$",
    "S_recon_std":  r"$\mathrm{\sigma_{S}}$",
}

# Target number of colorbar tick labels per panel.
CBAR_NBINS = 7

# Per-(cfg, key) overrides: any of {"vmin", "vmax", "ticks"}.
OVERRIDES = {
    "fram_6p25": {
        "T_recon_mean": {"vmin": -2.0,
                          "ticks": [-2, -1, 0, 1, 2, 3, 4]},
    },
    "barents_6p25": {
        "T_recon_mean": {"ticks": [1, 2, 3, 4, 5, 6, 7]},
    },
}

# Extra depth cutoffs (m) to render in addition to the default PLOT_LIMITS
# max_depth. ``None`` is implicitly always included.
EXTRA_DEPTHS = {
    "fram_6p25": [2000, 1500, 1000],
}


# ---------------------------------------------------------------------------
# Transect helpers (same conventions as plot_transects_velocity.py)
# ---------------------------------------------------------------------------

def _transect_indices(template, endpoints, n_points=50):
    (lon1, lat1), (lon2, lat2) = endpoints
    lat_grid = template["latitude"].values
    lon_grid = template["longitude"].values
    line_lons = np.linspace(lon1, lon2, n_points)
    line_lats = np.linspace(lat1, lat2, n_points)
    grid_points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    line_points = np.column_stack([line_lons, line_lats])
    distances = cdist(line_points, grid_points)
    nearest = np.argmin(distances, axis=1)
    y_idx, x_idx = np.unravel_index(nearest, lat_grid.shape)
    sec_lon = lon_grid[y_idx, x_idx]
    sec_lat = lat_grid[y_idx, x_idx]
    if (np.max(sec_lon) - np.min(sec_lon)) > (np.max(sec_lat) - np.min(sec_lat)):
        x_coords = sec_lon
        xlabel = "Longitude"
    else:
        x_coords = sec_lat
        xlabel = "Latitude"
    return y_idx, x_idx, x_coords, xlabel


def _gather_section(template, field, endpoints):
    y_idx, x_idx, x_coords, xlabel = _transect_indices(template, endpoints)
    elevation = template["elevation"].values
    depth_coords = template["depth"].values
    sec_bathy = elevation[y_idx, x_idx]
    h_bathy = sec_bathy.copy()
    if np.nanmax(h_bathy) > 0:
        h_bathy = -np.abs(h_bathy)

    x_section, z_section, var_section = [], [], []
    for i, (j, k) in enumerate(zip(y_idx, x_idx)):
        profile = field[:, j, k]
        valid = ~np.isnan(profile)
        if np.any(valid):
            x_section.extend([x_coords[i]] * int(valid.sum()))
            z_section.extend(-depth_coords[valid])
            var_section.extend(profile[valid])
    return (np.asarray(x_section), np.asarray(z_section),
            np.asarray(var_section), x_coords, h_bathy, xlabel)


def _draw_section(ax, x_section, z_section, var_section,
                  x_coords, h_bathy, *,
                  cmap, vmin, vmax, max_depth, extend):
    if max_depth is None:
        max_depth_local = (np.nanmin(z_section) - 300) if z_section.size else -300
    else:
        max_depth_local = -abs(max_depth)

    ax.fill_between(x_coords, np.maximum(h_bathy, max_depth_local),
                    max_depth_local, color="darkgray", alpha=0.7, zorder=1)

    pc = None
    if x_section.size:
        x_min, x_max = np.nanmin(x_coords), np.nanmax(x_coords)
        nx, nz = 100, 60
        xi = np.linspace(x_min, x_max, nx)
        zi = np.linspace(max_depth_local, 0, nz)
        Xi, Zi = np.meshgrid(xi, zi)
        try:
            Vi = griddata(np.column_stack([x_section, z_section]),
                          var_section, (Xi, Zi), method="linear")
            if not np.any(np.isfinite(Vi)):
                raise RuntimeError("all-NaN")
            bathy_interp = np.interp(xi, x_coords, h_bathy)
            for i in range(nx):
                Vi[:, i] = np.where(Zi[:, i] < bathy_interp[i],
                                    np.nan, Vi[:, i])
            lvls = np.linspace(vmin, vmax, 20)
            pc = ax.contourf(Xi, Zi, Vi, levels=lvls, cmap=cmap,
                             vmin=vmin, vmax=vmax, extend=extend, zorder=-1)
        except Exception as e:
            logger.info(f"  contour failed ({str(e).splitlines()[0]}); scatter")
            pc = ax.scatter(x_section, z_section, c=var_section, s=8,
                            cmap=cmap, vmin=vmin, vmax=vmax, zorder=10)

    ax.set_xlim(np.nanmin(x_coords), np.nanmax(x_coords))
    ax.set_ylim(max_depth_local, 0)
    return pc


def _percentile_range(values, lo, hi):
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    return float(np.percentile(finite, lo)), float(np.percentile(finite, hi))


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

def build_figure(cfg: str, overwrite: bool, max_depth_override=None):
    gateway_name, endpoints = GATEWAYS[cfg]
    limits = PLOT_LIMITS[cfg]
    if max_depth_override is None:
        max_depth = limits["max_depth"]
        suffix = ""
    else:
        max_depth = max_depth_override
        suffix = f"_{int(max_depth_override)}m"
    reconst_dir = BASE_DIR / cfg
    fpath = reconst_dir / "product_stats" / "full" / "mean_2011_2021.nc"
    out_dir = reconst_dir / "plots_transects"
    out_path = out_dir / f"{gateway_name}_TS_recon_mean_std{suffix}.png"

    if out_path.exists() and not overwrite:
        logger.info(f"[{cfg}] exists, skip ({out_path.name})")
        return
    if not fpath.exists():
        logger.error(f"[{cfg}] missing {fpath}"); return

    with xr.open_dataset(fpath) as ds:
        t_mean = ds["T_recon_mean"].isel(time=0).values
        t_std = ds["T_recon_std"].isel(time=0).values
        s_mean = ds["S_recon_mean"].isel(time=0).values
        s_std = ds["S_recon_std"].isel(time=0).values
        template = ds[["latitude", "longitude",
                       "elevation", "depth"]].load()

    fields = {
        "T_recon_mean": (t_mean, CMAP_T_MEAN, "mean"),
        "T_recon_std":  (t_std,  CMAP_STD,    "std"),
        "S_recon_mean": (s_mean, CMAP_S_MEAN, "mean"),
        "S_recon_std":  (s_std,  CMAP_STD,    "std"),
    }

    # Gather transect samples once per field, derive per-panel ranges.
    gathered = {}
    ranges = {}
    tick_overrides = {}
    cfg_over = OVERRIDES.get(cfg, {})
    for key, (field, _cmap, kind) in fields.items():
        gathered[key] = _gather_section(template, field, endpoints)
        vals = gathered[key][2]
        if kind == "mean":
            vmin, vmax = _percentile_range(vals, 2, 98)
            extend = "both"
        else:
            _, vmax = _percentile_range(vals, 0, 98)
            vmin, extend = 0.0, "max"
        ov = cfg_over.get(key, {})
        vmin = ov.get("vmin", vmin)
        vmax = ov.get("vmax", vmax)
        ranges[key] = ((vmin, vmax), extend)
        if "ticks" in ov:
            tick_overrides[key] = list(ov["ticks"])

    fig, axes = plt.subplots(2, 2, figsize=(13, 9.5), sharey="row")
    fig.subplots_adjust(left=0.08, right=0.93, top=0.88, bottom=0.10,
                        hspace=0.35, wspace=0.2)

    (p1, p2) = endpoints
    endpoint_txt = (
        f"[Lon, Lat] endpoints: [{p1[0]:.2f}°, {p1[1]:.2f}°], "
        f"[{p2[0]:.2f}°, {p2[1]:.2f}°]")

    panel_letters = ["a", "b", "c", "d"]
    panel_titles = {
        "T_recon_mean": f"{gateway_name} mean Temperature",
        "T_recon_std":  f"{gateway_name} Temperature Standard Deviation σ",
        "S_recon_mean": f"{gateway_name} mean Salinity",
        "S_recon_std":  f"{gateway_name} Salinity Standard Deviation σ",
    }
    order = ["T_recon_mean", "T_recon_std", "S_recon_mean", "S_recon_std"]

    xlabel_seen = None
    for idx, key in enumerate(order):
        ri, ci = divmod(idx, 2)
        ax = axes[ri, ci]
        _field, cmap, _kind = fields[key]
        (vmin, vmax), extend = ranges[key]
        x_section, z_section, var_section, x_coords, h_bathy, xlabel = \
            gathered[key]
        xlabel_seen = xlabel

        pc = _draw_section(ax, x_section, z_section, var_section,
                           x_coords, h_bathy,
                           cmap=cmap, vmin=vmin, vmax=vmax,
                           max_depth=max_depth, extend=extend)

        ax.set_title(
            f"({panel_letters[idx]}) {panel_titles[key]}\n{endpoint_txt}",
            fontsize=TITLE_FS, linespacing=1.5)

        if pc is not None:
            cb = fig.colorbar(pc, ax=ax, extend=extend,
                              shrink=0.85, pad=0.02)
            cb.set_label(CBAR_LABELS[key], fontsize=LABEL_FS)
            cb.ax.tick_params(labelsize=LABEL_FS)
            if key in tick_overrides:
                ticks = tick_overrides[key]
            else:
                ticks = MaxNLocator(
                    nbins=CBAR_NBINS, steps=[1, 2, 2.5, 5, 10]
                ).tick_values(vmin, vmax)
                ticks = [t for t in ticks
                         if vmin - 1e-9 <= t <= vmax + 1e-9]
            cb.set_ticks(ticks)
            cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    deg_fmt = FuncFormatter(lambda x, _: f"{x:g}°")
    abs_fmt = FuncFormatter(lambda y, _: f"{abs(y):g}")
    xticks = XAXIS_TICKS.get(cfg)
    for ax in axes.flat:
        ax.set_xlabel(f"{xlabel_seen} (°)", fontsize=LABEL_FS)
        ax.xaxis.set_major_formatter(deg_fmt)
        ax.yaxis.set_major_formatter(abs_fmt)
        ax.set_ylabel("Depth (m)", fontsize=LABEL_FS)
        ax.tick_params(axis="both", labelsize=LABEL_FS, labelleft=True)
        if xticks is not None:
            ax.set_xticks(xticks)

    fig.suptitle(f"{gateway_name} Transect — 2011–2021 mean & std of "
                 "reconstructed Temperature and Salinity",
                 y=0.96)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[{cfg}] saved → {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="*", default=DEFAULT_CONFIGS,
                    help=f"Gateways to process (default: {DEFAULT_CONFIGS})")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-render and overwrite existing PNG(s).")
    args = ap.parse_args()

    for cfg in args.configs:
        if cfg not in GATEWAYS:
            logger.error(f"Unknown config: {cfg}"); continue
        build_figure(cfg, args.overwrite)
        for d in EXTRA_DEPTHS.get(cfg, []):
            build_figure(cfg, args.overwrite, max_depth_override=d)
    logger.info("Done.")


if __name__ == "__main__":
    main()
