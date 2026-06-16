#!/usr/bin/env python3
"""
4x2 seasonal climatology figure of T & S predicted anomalies per gateway.

DEPENDENCY NOTE
---------------
Requires plot_transects_batch.py (same directory) as a config/utils module.
Imports: BASE_DIR, CONFIGS, GATEWAYS, VAR_INFO, PLOT_LIMITS, SEASONS,
         SEASON_LABEL, _make_asym_cmap_norm, discover_seasonal.

The local _draw_section helper duplicates the bathy-fill + griddata +
contourf core in plot_transects_batch.plot_single_gateway_section, with
minor differences (shared norm/cmap, no per-figure save inside the helper).

For each config in CONFIGS, builds one figure:
    rows = seasons (DJF, MAM, JJA, SON, top-to-bottom)
    cols = (T_anom_pred, S_anom_pred)
Each panel = climatological mean of all seasonal_*_<SEAS>(_partial)?.nc files
(2011-2021), drawn as a vertical transect across the gateway using the same
style as ``plot_transects_batch.py``.

Saved as:
    <reconst-dir>/plots_transects/<Gateway>_seasonal_clim_anomalies.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist

from plot_transects_batch import (
    BASE_DIR, CONFIGS, GATEWAYS, VAR_INFO, PLOT_LIMITS,
    SEASONS, SEASON_LABEL, _make_asym_cmap_norm, discover_seasonal,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

mpl.rcParams["font.size"] = mpl.rcParams["font.size"] * 1.2

ANOM_VARS = ["T_anom_pred", "S_anom_pred"]

# Override colorbar labels for this summary figure.
CBAR_LABEL = {
    "T_anom_pred": "Temperature Predicted Anomaly [°C]",
    "S_anom_pred": "Salinity Predicted Anomaly",
}

# Hardcoded colorbar ticks for this summary figure (overrides PLOT_LIMITS).
# Per-config; ``None`` means fall back to PLOT_LIMITS[cfg]["ticks"].
CBAR_TICKS = {
    "fram_6p25": {
        "T_anom_pred": [-0.6, -0.3, 0.0, 0.3, 0.6, 0.9, 1.2],
        "S_anom_pred": [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4],
    },
    "barents_6p25": None,
    "bering_6p25": None,
    "davis_6p25": None,
}

# For fram_6p25 only: render the figure at multiple depth cutoffs.
# ``None`` = use PLOT_LIMITS[cfg]["max_depth"]; else cutoff in metres.
FRAM_DEPTHS = [None, 2000, 1500, 1000]

# Optional explicit x-axis ticks per config (applied to all axes via sharex).
XAXIS_TICKS = {
    "barents_6p25": list(range(71, 75)),
}


# ---------------------------------------------------------------------------
# Climatology + per-axes section draw
# ---------------------------------------------------------------------------

def _seasonal_climatology(files, var_name):
    """Average <var_name> across all given files; return (template, mean)."""
    arrs, template = [], None
    for p in files:
        with xr.open_dataset(p) as ds:
            if template is None:
                template = ds[["latitude", "longitude",
                               "elevation", "depth"]].load()
            da = ds[var_name]
            if "time" in da.dims:
                da = da.isel(time=0)
            arrs.append(da.values)
    return template, np.nanmean(np.stack(arrs, axis=0), axis=0)


def _draw_section(ax, template, var_field, endpoints, *,
                  cmap, norm, vmin, vmax, max_depth):
    """Draw a single vertical transect onto ``ax``. Returns (mappable, xlabel,
    plot_type_used)."""
    (lon1, lat1), (lon2, lat2) = endpoints
    lat_grid = template["latitude"].values
    lon_grid = template["longitude"].values
    elevation = template["elevation"].values
    depth_coords = template["depth"].values

    n_points = 50
    line_lons = np.linspace(lon1, lon2, n_points)
    line_lats = np.linspace(lat1, lat2, n_points)
    grid_points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])
    line_points = np.column_stack([line_lons, line_lats])
    distances = cdist(line_points, grid_points)
    nearest = np.argmin(distances, axis=1)
    y_idx, x_idx = np.unravel_index(nearest, lat_grid.shape)

    sec_lon = lon_grid[y_idx, x_idx]
    sec_lat = lat_grid[y_idx, x_idx]
    sec_bathy = elevation[y_idx, x_idx]

    if (np.max(sec_lon) - np.min(sec_lon)) > (np.max(sec_lat) - np.min(sec_lat)):
        x_coords = sec_lon
        xlabel = "Longitude [°]"
    else:
        x_coords = sec_lat
        xlabel = "Latitude [°]"

    h_bathy = sec_bathy.copy()
    if np.nanmax(h_bathy) > 0:
        h_bathy = -np.abs(h_bathy)

    x_section, z_section, var_section = [], [], []
    for i, (j, k) in enumerate(zip(y_idx, x_idx)):
        profile = var_field[:, j, k]
        valid = ~np.isnan(profile)
        if np.any(valid):
            x_section.extend([x_coords[i]] * int(valid.sum()))
            z_section.extend(-depth_coords[valid])
            var_section.extend(profile[valid])
    x_section = np.asarray(x_section)
    z_section = np.asarray(z_section)
    var_section = np.asarray(var_section)

    if max_depth is None:
        max_depth_local = (np.nanmin(z_section) - 300) if z_section.size else -300
    else:
        max_depth_local = -abs(max_depth)

    ax.fill_between(x_coords, np.maximum(h_bathy, max_depth_local),
                    max_depth_local, color="darkgray", alpha=0.7, zorder=1)

    pc = None
    used = "contour"
    if x_section.size:
        x_min, x_max = np.nanmin(x_coords), np.nanmax(x_coords)
        nx, nz = 100, 50
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
            kw = dict(levels=lvls, cmap=cmap, extend="both", zorder=-1)
            if norm is not None:
                kw["norm"] = norm
            else:
                kw["vmin"], kw["vmax"] = vmin, vmax
            pc = ax.contourf(Xi, Zi, Vi, **kw)
        except Exception as e:
            logger.info(f"  contour failed ({str(e).splitlines()[0]}); scatter")

        if pc is None:
            kw = dict(c=var_section, s=8, cmap=cmap, alpha=1.0, zorder=10)
            if norm is not None:
                kw["norm"] = norm
            else:
                kw["vmin"], kw["vmax"] = vmin, vmax
            pc = ax.scatter(x_section, z_section, **kw)
            used = "scatter"

    ax.set_xlim(np.nanmin(x_coords), np.nanmax(x_coords))
    ax.set_ylim(max_depth_local, 0)
    return pc, xlabel, used


# ---------------------------------------------------------------------------
# Per-gateway driver
# ---------------------------------------------------------------------------

def make_gateway_figure(cfg: str, overwrite: bool,
                        max_depth_override=None):
    gateway_name, endpoints = GATEWAYS[cfg]
    limits = PLOT_LIMITS[cfg]
    reconst_dir = BASE_DIR / cfg
    stats_dir = reconst_dir / "product_stats"
    out_dir = reconst_dir / "plots_transects"
    if max_depth_override is None:
        max_depth = limits["max_depth"]
        suffix = ""
    else:
        max_depth = max_depth_override
        suffix = f"_{int(max_depth_override)}m"
    out_path = out_dir / f"{gateway_name}_seasonal_clim_anomalies{suffix}.png"

    if out_path.exists() and not overwrite:
        logger.info(f"[{cfg}] exists, skip ({out_path.name})")
        return
    if not stats_dir.is_dir():
        logger.error(f"[{cfg}] no product_stats/ — skip"); return

    files_all = discover_seasonal(stats_dir, skip_partial=False)
    by_season = {s: [p for (_y, ss, p) in files_all if ss == s]
                 for s in SEASONS}
    for s in SEASONS:
        if not by_season[s]:
            logger.warning(f"[{cfg}] no files for season {s}; skip figure")
            return
        logger.info(f"[{cfg}] {s}: {len(by_season[s])} files")

    # Shared cmap+norm per column (anomaly: vcenter=0 with clamp)
    col_info = {}
    for var_key in ANOM_VARS:
        vmin, vmax = limits["vrange"][var_key]
        base_cmap = VAR_INFO[var_key]["cmap"]
        cf = float(np.clip((0.0 - vmin) / (vmax - vmin), 0.25, 0.75))
        cmap_a, norm_a = _make_asym_cmap_norm(base_cmap, vmin, 0.0, vmax, cf=cf)
        col_info[var_key] = (cmap_a, norm_a, vmin, vmax)

    fig, axes = plt.subplots(len(SEASONS), len(ANOM_VARS),
                             figsize=(12, 13),
                             sharex=True, sharey=True)
    fig.subplots_adjust(left=0.10, right=0.93, top=0.88, bottom=0.12,
                        hspace=0.12, wspace=0.06)
            

    mappables = {}
    xlabel_seen = None
    for ci, var_key in enumerate(ANOM_VARS):
        cmap, norm, vmin, vmax = col_info[var_key]
        vname = f"{var_key}_mean"
        for ri, season in enumerate(SEASONS):
            ax = axes[ri, ci]
            template, mean_field = _seasonal_climatology(
                by_season[season], vname)
            pc, xlabel, used = _draw_section(
                ax, template, mean_field, endpoints,
                cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
                max_depth=max_depth)
            mappables[var_key] = pc
            xlabel_seen = xlabel
            panel_letter = chr(ord("a") + ri * len(ANOM_VARS) + ci)
            ax.text(0.02, 0.04, f"({panel_letter})",
                    transform=ax.transAxes,
                    ha="left", va="bottom", zorder=20,
                    bbox=dict(facecolor="white", edgecolor="none",
                              alpha=0.7, pad=1.5))
            if ci == len(ANOM_VARS) - 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(SEASON_LABEL[season],
                              rotation=270, labelpad=20)
            logger.info(f"  [{used:7s}] {var_key} {season}")

    # X-axis on TOP margin only, with degree-suffixed tick labels.
    deg_fmt = FuncFormatter(lambda x, _: f"{x:g}°")
    xticks = XAXIS_TICKS.get(cfg)
    for ax in axes.flat:
        ax.tick_params(axis="x", bottom=True, top=True,
                       labelbottom=False, labeltop=False)
        if xticks is not None:
            ax.set_xticks(xticks)
    for ax in axes[0, :]:
        ax.tick_params(axis="x", labeltop=True)
        ax.xaxis.set_major_formatter(deg_fmt)
        ax.xaxis.set_label_position("top")
        if xlabel_seen is not None:
            ax.set_xlabel(xlabel_seen.split(" [")[0])

    # Show depth tick labels as positive numbers (axis values are negative).
    abs_fmt = FuncFormatter(lambda y, _: f"{abs(y):g}")
    for ax in axes.flat:
        ax.yaxis.set_major_formatter(abs_fmt)

    # Shared Depth label on the left margin.
    fig.supylabel("Depth (m)", x=0.02)

    # Suptitle: gateway + endpoints
    (p1, p2) = endpoints
    fig.suptitle(
        f"{gateway_name} Seasonal Mean Anomalies\n"
        f"[Lon, Lat] endpoints: [{p1[0]:.2f}°, {p1[1]:.2f}°], "
        f"[{p2[0]:.2f}°, {p2[1]:.2f}°]",
        y=0.98, linespacing=1.5)

    # Horizontal shared colorbars below each column
    for ci, var_key in enumerate(ANOM_VARS):
        bbox = axes[-1, ci].get_position()
        cax = fig.add_axes([bbox.x0, 0.08, bbox.width, 0.018])
        cbar = fig.colorbar(mappables[var_key], cax=cax,
                            orientation="horizontal", extend="both")
        cbar.set_label(CBAR_LABEL[var_key])
        cfg_ticks = CBAR_TICKS.get(cfg)
        if cfg_ticks is None:
            ticks = limits.get("ticks", {}).get(var_key)
        else:
            ticks = cfg_ticks.get(var_key)
        if ticks is not None:
            cbar.set_ticks(list(ticks))
        cbar.ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"[{cfg}] saved → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="*", default=CONFIGS,
                    help=f"Subset of configs (default: {CONFIGS})")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-render and overwrite existing PNG.")
    args = ap.parse_args()

    for cfg in args.configs:
        if cfg not in GATEWAYS:
            logger.error(f"Unknown config: {cfg}"); continue
        if cfg == "fram_6p25":
            for d in FRAM_DEPTHS:
                make_gateway_figure(cfg, args.overwrite,
                                    max_depth_override=d)
        else:
            make_gateway_figure(cfg, args.overwrite)
    logger.info("Done.")


if __name__ == "__main__":
    main()
