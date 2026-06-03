#!/usr/bin/env python3
"""
Batch-generate gateway transect plots for the regional LSTM reconstructions.

DEPENDENCY NOTE
---------------
This file doubles as a **shared config/utils module** imported by the other
plotting scripts in this directory.  Do NOT rename or move it without updating
the imports in the scripts below.

    plot_transects_seasonal_clim.py
        imports: BASE_DIR, CONFIGS, GATEWAYS, VAR_INFO, PLOT_LIMITS,
                 SEASONS, SEASON_LABEL, _make_asym_cmap_norm,
                 discover_seasonal
        own draw helpers: _seasonal_climatology, _draw_section
        (duplicates the bathy-fill + griddata + contourf core from here)

    plot_transects_velocity.py
        imports: BASE_DIR, GATEWAYS, PLOT_LIMITS
        own draw helpers: _transect_indices, _draw_field,
                          _overlay_direction_markers, _normal_unit_vector
        (_transect_indices is an independent reimplementation)

    plot_transects_TS_mean_std.py
        imports: BASE_DIR, GATEWAYS, PLOT_LIMITS
        own draw helpers: _transect_indices, _gather_section, _draw_section
        (_transect_indices is nearly identical to the one in
        plot_transects_velocity.py; _draw_section parallels the logic in
        plot_single_gateway_section below)

The draw helpers are intentionally left duplicated (no shared utils file)
because each script has minor but coupled differences in extend=, norm=,
bathy masking, etc.  If new scripts are added, consider extracting a
``transect_utils.py``.

Iterates over configs (bering_6p25, fram_6p25, davis_6p25, barents_6p25),
each tied to its own gateway transect. For each (config, variable) pair,
plots:

* Full-record mean        (product_stats/full/mean_YYYY_YYYY.nc)
* Per-season means        (product_stats/seasonal/seasonal_YYYY_<SEAS>.nc)
* Per-month means         (product_stats/monthly/monthly_YYYYMM.nc)
* Four specific 2012 dates (TS_currents_lstm/TS_currents_lstm_YYYYMMDD.nc):
    2012-01-16, 2012-04-16, 2012-07-16, 2012-10-16.

Output tree (per config), sibling to product_stats/:

    <reconst-dir>/plots_transects/
        full/{var}/<Gateway>_<var>_2011-2021_mean.png
        seasonal/{var}/<Gateway>_<var>_<SEAS>_<YYYY>.png
        monthly/{var}/<Gateway>_<var>_<YYYY-MM>.png
        dates/{var}/<Gateway>_<var>_<YYYY-MM-DD>.png

Stat files store variables as ``<var>_mean``; per-date files store ``<var>``.
"""

from __future__ import annotations

import argparse
import calendar
import logging
import os
import re
from pathlib import Path

import cmocean
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cmcrameri import cm as cmc
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

mpl.rcParams["font.size"] = mpl.rcParams["font.size"] * 1.2


# ---------------------------------------------------------------------------
# Static config
# ---------------------------------------------------------------------------

BASE_DIR = Path("/home/nicolas/SACO/FRESH-CARE/Data_lstm_reconstruction")

CONFIGS = ["bering_6p25", "fram_6p25", "davis_6p25", "barents_6p25"]

# (lon1, lat1) -> (lon2, lat2)
GATEWAYS = {
    "bering_6p25":  ("Bering",  [[-169.7,   66.0333], [-168.0996, 65.685]]),
    "fram_6p25":    ("Fram",    [[-17.5,    79.0],    [11.0,      79.0]]),
    "davis_6p25":   ("Davis",   [[-61.0,    66.0],    [-54.0,     66.0]]),
    "barents_6p25": ("Barents", [[20.0,     70.0],    [20.0,      78.0]]),
}

# Per-variable colormap + colorbar label (used for both _mean and base names)
VAR_INFO = {
    "T_anom_pred": {"cmap": cmc.vik_r,
                    "pretty": "T Pred. Anom.",
                    "cbar_label": "Temperature Anomaly [°C]"},
    "S_anom_pred": {"cmap": cmc.vik_r,
                    "pretty": "S Pred. Anom.",
                    "cbar_label": "Salinity Anomaly"},
    "T_recon":     {"cmap": cmocean.cm.thermal,
                    "pretty": "T Recon.",
                    "cbar_label": "Temperature [°C]"},
    "S_recon":     {"cmap": cmocean.cm.haline,
                    "pretty": "S Recon.",
                    "cbar_label": "Salinity"},
}
VARIABLES = ["T_anom_pred", "S_anom_pred", "T_recon", "S_recon"]

# Per-config (max_depth, {var: (vmin, vmax)}, {var: [tick values]}).
# Ranges = 2nd/98th percentile of all seasonal-mean transect values
# (2011-2021, 43 files per gateway), snapped to nice steps.
# Tick lists are hardcoded so colorbar labels appear at IDENTICAL positions
# across all slides for a given (gateway, variable). Edit freely.
# For anomalies (T_anom_pred, S_anom_pred) the zero position on the colorbar
# is the linear position of 0 in [vmin, vmax], clamped to [0.25, 0.75] so 0
# never sits within 1/4 of either end.
PLOT_LIMITS: dict[str, dict] = {
    "bering_6p25": {
        "max_depth": 60,
        "vrange": {
            "T_anom_pred": (-0.60,  0.40),
            "S_anom_pred": (-0.65,  0.15),
            "T_recon":     (-2.00,  7.50),
            "S_recon":     (29.60, 32.90),
        },
        "ticks": {
            "T_anom_pred": [-0.60, -0.40, -0.20, 0.00, 0.10, 0.20, 0.30, 0.40],
            "S_anom_pred": [-0.60, -0.40, -0.20, 0.00, 0.05, 0.10, 0.15],
            "T_recon":     [-2.0, 0.0, 2.0, 4.0, 6.0],
            "S_recon":     [30.0, 30.5, 31.0, 31.5, 32.0, 32.5],
        },
    },
    "davis_6p25": {
        "max_depth": 700,
        "vrange": {
            "T_anom_pred": (-0.60,  1.50),
            "S_anom_pred": (-0.35,  0.15),
            "T_recon":     (-2.00,  5.00),
            "S_recon":     (31.60, 34.80),
        },
        "ticks": {
            "T_anom_pred": [-0.60, -0.40, -0.20, 0.00, 0.50, 1.00, 1.50],
            "S_anom_pred": [-0.30, -0.20, -0.10, 0.00, 0.05, 0.10, 0.15],
            "T_recon":     [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0],
            "S_recon":     [32.0, 32.5, 33.0, 33.5, 34.0, 34.5],
        },
    },
    "fram_6p25": {
        "max_depth": None,
        "vrange": {
            "T_anom_pred": (-0.60,  1.40),
            "S_anom_pred": (-0.85,  0.35),
            "T_recon":     (-2.00,  5.50),
            "S_recon":     (31.30, 35.10),
        },
        "ticks": {
            "T_anom_pred": [-0.60, -0.40, -0.20, 0.00, 0.50, 1.00, 1.40],
            "S_anom_pred": [-0.80, -0.60, -0.40, -0.20, 0.00, 0.10, 0.20, 0.30],
            "T_recon":     [-2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "S_recon":     [31.5, 32.0, 32.5, 33.0, 33.5, 34.0, 34.5, 35.0],
        },
    },
    "barents_6p25": {
        "max_depth": 500,
        "vrange": {
            "T_anom_pred": (-0.60,  0.60),
            "S_anom_pred": (-0.30,  0.10),
            "T_recon":     (-1.50,  9.00),
            "S_recon":     (33.90, 35.20),
        },
        "ticks": {
            "T_anom_pred": [-0.60, -0.40, -0.20, 0.00, 0.20, 0.40, 0.60],
            "S_anom_pred": [-0.30, -0.20, -0.10, 0.00, 0.05, 0.10],
            "T_recon":     [-2.0, 0.0, 2.0, 4.0, 6.0, 8.0],
            "S_recon":     [34.0, 34.2, 34.4, 34.6, 34.8, 35.0, 35.2],
        },
    },
}

SEASONS = ["DJF", "MAM", "JJA", "SON"]
SEASON_LABEL = {"DJF": "Winter (DJF)", "MAM": "Spring (MAM)",
                "JJA": "Summer (JJA)", "SON": "Autumn (SON)"}

DATE_PLOTS = [
    ("2012-01-16", "20120116"),
    ("2012-04-16", "20120416"),
    ("2012-07-16", "20120716"),
    ("2012-10-16", "20121016"),
]

FULL_RE = re.compile(r"^mean_(\d{4})_(\d{4})\.nc$")
SEAS_RE = re.compile(r"^seasonal_(\d{4})_(DJF|MAM|JJA|SON)(_partial)?\.nc$")
MON_RE  = re.compile(r"^monthly_(\d{4})(\d{2})(_partial)?\.nc$")


# ---------------------------------------------------------------------------
# Asymmetric colormap norm (lifted from the notebook)
# ---------------------------------------------------------------------------

def _make_asym_cmap_norm(cmap, vmin, vcenter, vmax, cf=None):
    if cf is None:
        cf = (vcenter - vmin) / (vmax - vmin)
    n = 256
    n_neg = max(1, min(n - 1, round(n * cf)))
    base = cmap if callable(cmap) else plt.get_cmap(cmap)
    colors = np.vstack([
        base(np.linspace(0.0, 0.5, n_neg)),
        base(np.linspace(0.5, 1.0, n - n_neg)),
    ])
    new_cmap = mpl.colors.LinearSegmentedColormap.from_list(
        f"{base.name}_asym", colors)

    class _AsymNorm(mpl.colors.Normalize):
        def __init__(self):
            self._vc, self._cf = vcenter, cf
            super().__init__(vmin, vmax)

        def __call__(self, value, clip=None):
            v = np.ma.asarray(value)
            res = np.interp(v.filled(np.nan),
                            [self.vmin, self._vc, self.vmax],
                            [0, self._cf, 1])
            return np.ma.array(res, mask=np.ma.getmaskarray(v))

        def inverse(self, value):
            return np.interp(value, [0, self._cf, 1],
                             [self.vmin, self._vc, self.vmax])

    return new_cmap, _AsymNorm()


# ---------------------------------------------------------------------------
# Core plot (adapted from the notebook)
# ---------------------------------------------------------------------------

def plot_single_gateway_section(
        ds, gateway_name, endpoints, var_in_ds, *,
        title, cmap, cbar_label,
        vmin=None, vmax=None, vcenter=0,
        ticks=None,
        max_depth=None, plot_type="contour",
        save_path, dpi=200):
    """Plot a vertical transect across a gateway, save PNG. Returns the
    actual plot type used (may fall back from 'contour' to 'scatter')."""

    # Asym cmap/norm for anomalies: place vcenter at its linear position in
    # [vmin, vmax], clamped to [0.25, 0.75] so 0 never sits within 1/4 of an end.
    norm = None
    if (vcenter is not None and vmin is not None and vmax is not None
            and vmin < vcenter < vmax and cmap is not None):
        cf = (vcenter - vmin) / (vmax - vmin)
        cf = float(np.clip(cf, 0.25, 0.75))
        cmap, norm = _make_asym_cmap_norm(cmap, vmin, vcenter, vmax, cf=cf)

    (lon1, lat1), (lon2, lat2) = endpoints
    lat_grid = ds["latitude"].values
    lon_grid = ds["longitude"].values
    elevation = ds["elevation"].values

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
    x_bathy = x_coords.copy()

    var_data = ds[var_in_ds]
    if "time" in var_data.dims:
        var_data = var_data.isel(time=0)
    depth_coords = ds["depth"].values

    x_section, z_section, var_section = [], [], []
    for i, (j, k) in enumerate(zip(y_idx, x_idx)):
        profile = var_data.values[:, j, k]
        valid = ~np.isnan(profile)
        if np.any(valid):
            x_section.extend([x_coords[i]] * int(valid.sum()))
            z_section.extend(-depth_coords[valid])
            var_section.extend(profile[valid])
    x_section = np.array(x_section)
    z_section = np.array(z_section)
    var_section = np.array(var_section)
    if x_section.size == 0:
        logger.warning(f"  no valid data — skip ({save_path.name})")
        return None

    if max_depth is None:
        max_depth = np.nanmin(z_section) - 300
    else:
        max_depth = -abs(max_depth)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.fill_between(x_bathy, np.maximum(h_bathy, max_depth), max_depth,
                    color="darkgray", alpha=0.7, zorder=1)

    used_type = plot_type
    pc = None
    if plot_type == "contour":
        x_min, x_max = np.nanmin(x_coords), np.nanmax(x_coords)
        nx, nz = 100, 50
        xi = np.linspace(x_min, x_max, nx)
        zi = np.linspace(max_depth, 0, nz)
        Xi, Zi = np.meshgrid(xi, zi)
        try:
            Vi = griddata(np.column_stack([x_section, z_section]),
                          var_section, (Xi, Zi), method="linear")
            if not np.any(np.isfinite(Vi)):
                raise RuntimeError("interpolation produced all-NaN field")
            bathy_interp = np.interp(xi, x_bathy, h_bathy)
            for i in range(nx):
                Vi[:, i] = np.where(Zi[:, i] < bathy_interp[i], np.nan, Vi[:, i])
            levels = 20
            lvls = np.linspace(vmin, vmax, levels)
            if norm is not None:
                pc = ax.contourf(Xi, Zi, Vi, levels=lvls, cmap=cmap, norm=norm,
                                 extend="both", zorder=-1)
            else:
                pc = ax.contourf(Xi, Zi, Vi, levels=lvls, cmap=cmap,
                                 vmin=vmin, vmax=vmax, extend="both",
                                 zorder=-1)
        except Exception as e:
            msg = str(e).splitlines()[0]
            logger.info(f"  contour failed ({msg}); using scatter")
            used_type = "scatter"

    if pc is None:  # scatter (explicit or fallback)
        kw = dict(c=var_section, s=8, cmap=cmap, alpha=1.0, zorder=10)
        if norm is not None:
            kw["norm"] = norm
        else:
            kw["vmin"], kw["vmax"] = vmin, vmax
        pc = ax.scatter(x_section, z_section, **kw)
        used_type = "scatter"

    ax.set_xlim(np.nanmin(x_coords), np.nanmax(x_coords))
    ax.set_ylim(max_depth, 0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Depth [m]")
    ax.set_title(title)

    cbar = fig.colorbar(pc, ax=ax, label=cbar_label, shrink=0.8, extend="both")
    if ticks is not None:
        cbar.set_ticks(list(ticks))
    cbar.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    # Use bbox_inches="tight" to trim whitespace. Hardcoded ticks keep the
    # colorbar label set identical across slides, so any residual shift
    # comes only from per-slide text width differences (typically negligible).
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return used_type


# ---------------------------------------------------------------------------
# Title + filename helpers
# ---------------------------------------------------------------------------

def title_with_endpoints(main: str, endpoints) -> str:
    (p1, p2) = endpoints
    return (f"{main}\n"
            f"[Lon, Lat] endpoints: [{p1[0]:.2f}, {p1[1]:.2f}], "
            f"[{p2[0]:.2f}, {p2[1]:.2f}]")


def discover_seasonal(stats_dir: Path, skip_partial=True):
    out = []
    for p in sorted((stats_dir / "seasonal").glob("seasonal_*.nc")):
        m = SEAS_RE.match(p.name)
        if not m:
            continue
        if m.group(3) and skip_partial:
            continue
        out.append((int(m.group(1)), m.group(2), p))
    return out


def discover_monthly(stats_dir: Path, skip_partial=True):
    out = []
    for p in sorted((stats_dir / "monthly").glob("monthly_*.nc")):
        m = MON_RE.match(p.name)
        if not m:
            continue
        if m.group(3) and skip_partial:
            continue
        out.append((int(m.group(1)), int(m.group(2)), p))
    return out


def discover_full(stats_dir: Path):
    for p in sorted((stats_dir / "full").glob("mean_*.nc")):
        m = FULL_RE.match(p.name)
        if m:
            return (int(m.group(1)), int(m.group(2)), p)
    return None


# ---------------------------------------------------------------------------
# Per-config driver
# ---------------------------------------------------------------------------

def process_config(cfg: str, overwrite: bool,
                   stages: set[str], include_partial_monthly: bool):
    gateway_name, endpoints = GATEWAYS[cfg]
    limits = PLOT_LIMITS[cfg]
    reconst_dir = BASE_DIR / cfg
    stats_dir = reconst_dir / "product_stats"
    dates_dir = reconst_dir / "TS_currents_lstm"
    out_root = reconst_dir / "plots_transects"

    if not stats_dir.is_dir():
        logger.error(f"[{cfg}] no product_stats/ — skip"); return

    logger.info(f"=== {cfg} ({gateway_name}) — stages: {sorted(stages)} ===")

    def _vminmax(var):
        return limits["vrange"][var]

    def _plot(ds, var_in_ds, var_key, title, out_path):
        if out_path.exists() and not overwrite:
            return
        vmin, vmax = _vminmax(var_key)
        info = VAR_INFO[var_key]
        # vcenter=0 only matters for diverging anomaly cmaps; for absolute
        # T/S use plain Normalize so no spurious 0 tick is drawn.
        vcenter = 0 if var_key.endswith("_anom_pred") else None
        ticks = limits.get("ticks", {}).get(var_key)
        used = plot_single_gateway_section(
            ds, gateway_name, endpoints, var_in_ds,
            title=title_with_endpoints(title, endpoints),
            cmap=info["cmap"], cbar_label=info["cbar_label"],
            vmin=vmin, vmax=vmax, vcenter=vcenter, ticks=ticks,
            max_depth=limits["max_depth"], plot_type="contour",
            save_path=out_path)
        if used is not None:
            logger.info(f"  [{used:7s}] {out_path.relative_to(out_root)}")

    # ---- Full ----
    if "full" in stages:
        full = discover_full(stats_dir)
        if full:
            y0, y1, fpath = full
            with xr.open_dataset(fpath) as ds:
                for var_key in VARIABLES:
                    vname = f"{var_key}_mean"
                    if vname not in ds:
                        logger.warning(f"  missing {vname} in {fpath.name}"); continue
                    title = f"{gateway_name} {VAR_INFO[var_key]['pretty']} -- {y0}-{y1} Mean"
                    fn = f"{gateway_name}_{var_key}_{y0}-{y1}_mean.png"
                    _plot(ds, vname, var_key, title,
                          out_root / "full" / var_key / fn)
        else:
            logger.warning(f"  [{cfg}] no full-record mean file")

    # ---- Seasonal ----
    if "seasonal" in stages:
        for year, season, spath in discover_seasonal(stats_dir):
            with xr.open_dataset(spath) as ds:
                for var_key in VARIABLES:
                    vname = f"{var_key}_mean"
                    if vname not in ds: continue
                    title = (f"{gateway_name} {VAR_INFO[var_key]['pretty']} -- "
                             f"Seasonal Mean -- {SEASON_LABEL[season]} {year}")
                    fn = f"{gateway_name}_{var_key}_{season}_{year}.png"
                    _plot(ds, vname, var_key, title,
                          out_root / "seasonal" / var_key / fn)

    # ---- Monthly ----
    if "monthly" in stages:
        for year, month, mpath in discover_monthly(
                stats_dir, skip_partial=not include_partial_monthly):
            with xr.open_dataset(mpath) as ds:
                for var_key in VARIABLES:
                    vname = f"{var_key}_mean"
                    if vname not in ds: continue
                    title = (f"{gateway_name} {VAR_INFO[var_key]['pretty']} -- "
                             f"Monthly Mean -- {calendar.month_name[month]} {year}")
                    fn = f"{gateway_name}_{var_key}_{year}-{month:02d}.png"
                    _plot(ds, vname, var_key, title,
                          out_root / "monthly" / var_key / fn)

    # ---- Specific dates ----
    if "dates" in stages:
        for iso, compact in DATE_PLOTS:
            dpath = dates_dir / f"TS_currents_lstm_{compact}.nc"
            if not dpath.exists():
                logger.warning(f"  date file missing: {dpath.name}"); continue
            with xr.open_dataset(dpath) as ds:
                for var_key in VARIABLES:
                    if var_key not in ds: continue
                    title = f"{gateway_name} {VAR_INFO[var_key]['pretty']} -- {iso}"
                    fn = f"{gateway_name}_{var_key}_{iso}.png"
                    _plot(ds, var_key, var_key, title,
                          out_root / "dates" / var_key / fn)


def main():
    all_stages = ["full", "seasonal", "monthly", "dates"]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--configs", nargs="*", default=CONFIGS,
                    help=f"Subset of configs (default: {CONFIGS})")
    ap.add_argument("--stages", nargs="*", default=all_stages,
                    choices=all_stages,
                    help=f"Which plot stages to run (default: {all_stages})")
    ap.add_argument("--include-partial-monthly", action="store_true",
                    help="Also plot monthly_*_partial.nc files.")
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-render and overwrite existing PNGs.")
    args = ap.parse_args()

    stages = set(args.stages)
    for cfg in args.configs:
        if cfg not in GATEWAYS:
            logger.error(f"Unknown config: {cfg}"); continue
        process_config(cfg, args.overwrite, stages, args.include_partial_monthly)
    logger.info("Done.")


if __name__ == "__main__":
    main()
