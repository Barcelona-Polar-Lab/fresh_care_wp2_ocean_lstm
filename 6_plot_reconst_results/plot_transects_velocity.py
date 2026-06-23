#!/usr/bin/env python3
"""
2x2 figure of gateway velocity transects (Fram + Barents).

DEPENDENCY NOTE
---------------
Requires plot_transects_batch.py (same directory) as a config/utils module.
Imports: BASE_DIR, GATEWAYS, PLOT_LIMITS.

The local _transect_indices function is an independent reimplementation of
the equivalent logic in plot_transects_batch and plot_transects_TS_mean_std
(all three are nearly identical).

Panels:
    (a) Fram    -- mean current intensity |v|  with in/out markers
    (b) Fram    -- total speed variability sqrt(cov_uu + cov_vv)
    (c) Barents -- mean current intensity |v|  with in/out markers
    (d) Barents -- total speed variability

* "Intensity" = sqrt(u_gos_mean**2 + v_gos_mean**2).
* "Std" combines the 2D variability ellipse into a single magnitude as
  sqrt(vel_cov_uu + vel_cov_vv) = sqrt(major**2 + minor**2).
* In/out markers: a dot (out of page, toward viewer) or cross (into page)
  is drawn at a coarse subsample of (x, depth) cells, based on the sign of
  the cross-transect component of the mean velocity. Convention: the
  in-plane "page" is the vertical plane spanned by the (lon, lat) tangent
  from endpoint1 -> endpoint2; "out of page" is the right-hand-rule
  normal in the horizontal plane (east, north): n = (-t_y, t_x).

Output:
    <BASE_DIR>/plots_common/Fram_Barents_velocity_transects.png
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from cmcrameri import cm as cmc
from matplotlib.ticker import FormatStrFormatter, FuncFormatter
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist

from plot_transects_batch import BASE_DIR, GATEWAYS, PLOT_LIMITS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

mpl.rcParams["font.size"] = mpl.rcParams["font.size"] * 1.2

# Gateways included in the figure, top-to-bottom.
ROWS = ["fram_6p25", "barents_6p25"]

# Colormaps (crameri).
CMAP_MEAN = cmc.oslo_r
CMAP_STD = cmc.lajolla_r

# Per-gateway colour ranges and hardcoded colorbar ticks (m/s).
PANEL_LIMITS = {
    "fram_6p25":    {"mean": (0.0, 0.15), "std": (0.0, 0.45)},
    "barents_6p25": {"mean": (0.0, 0.10), "std": (0.0, 0.15)},
}
CBAR_TICKS_MEAN = {
    "fram_6p25":    [0.00, 0.03, 0.06, 0.09, 0.12, 0.15],
    "barents_6p25": [0.00, 0.02, 0.04, 0.06, 0.08, 0.10],
}
CBAR_TICKS_STD = {
    "fram_6p25":    [0.00, 0.1, 0.2, 0.3, 0.4],
    "barents_6p25": [0.00, 0.03, 0.06, 0.09, 0.12, 0.15],
}

# Mean panel field per config: "intensity" (default) | "u" | "v".
MEAN_FIELD = {
    "fram_6p25":    "v",
    "barents_6p25": "u",
}

# Colour ranges for u / v components (diverging, symmetric around 0).
PANEL_LIMITS_UV = {
    "fram_6p25":    {"u": (-0.10, 0.10), "v": (-0.10, 0.10)},
    "barents_6p25": {"u": (-0.07, 0.07), "v": (-0.07, 0.07)},
}
CBAR_TICKS_UV = {
    "fram_6p25": {
        "u": [-0.10, -0.05, 0.00, 0.05, 0.10],
        "v": [-0.10, -0.05, 0.00, 0.05, 0.10],
    },
    "barents_6p25": {
        "u": [-0.07, -0.035, 0.00, 0.035, 0.07],
        "v": [-0.07, -0.035, 0.00, 0.035, 0.07],
    },
}

#CMAP_UV = cmc.bam  # diverging colormap for signed velocity components
#CMAP_UV = cmc.roma  # diverging colormap for signed velocity components
#CMAP_UV = cmc.vik  # diverging colormap for signed velocity components
CMAP_UV = cmc.cork  # diverging colormap for signed velocity components


# Optional explicit x-axis ticks per config.
XAXIS_TICKS = {
    "barents_6p25": list(range(71, 75)),
}

# Extra subsampling factor for in/out markers in the upper 100 m (per cfg).
# 2 = keep every other shallow row.
SHALLOW_STRIDE = {
    "fram_6p25": 2,
    "barents_6p25": 1,
}

# Font sizes (~15% smaller than the global default).
LABEL_FS = 12
TITLE_FS = 13

# Subsampling for the in/out direction markers (every Nth in x and depth).
MARKER_STRIDE_X = 3
MARKER_STRIDE_Z = 6
MARKER_SIZE = 36


# ---------------------------------------------------------------------------
# Transect helpers
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
        xlabel = "Longitude [°]"
    else:
        x_coords = sec_lat
        xlabel = "Latitude [°]"
    return y_idx, x_idx, x_coords, xlabel


def _normal_unit_vector(endpoints):
    """Horizontal 'out of page' direction for a vertical section drawn
    with endpoint1 on the left and endpoint2 on the right. Defined as
    ``tangent x up`` (right-hand rule): if t = (t_e, t_n, 0) and
    up = (0, 0, 1), then n = (t_n, -t_e, 0). Returns (n_east, n_north).
    Positive projection of (u_east, v_north) on this n means the velocity
    points 'out of the page' (toward the viewer)."""
    (lon1, lat1), (lon2, lat2) = endpoints
    mean_lat = np.deg2rad(0.5 * (lat1 + lat2))
    t_east = (lon2 - lon1) * np.cos(mean_lat)
    t_north = (lat2 - lat1)
    norm = np.hypot(t_east, t_north)
    t_east /= norm
    t_north /= norm
    return t_north, -t_east  # tangent x up (90 deg CW in horizontal plane)


def _draw_field(ax, template, field, endpoints, *,
                cmap, vmin, vmax, max_depth, extend="max"):
    """Filled-contour a single (depth, transect) field. Returns mappable
    and (x_coords, depths, x_bathy, h_bathy, max_depth_local)."""
    y_idx, x_idx, x_coords, _ = _transect_indices(template, endpoints)
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
    return pc, (x_coords, depth_coords, h_bathy, max_depth_local,
                y_idx, x_idx)


def _overlay_direction_markers(ax, template, u_field, v_field, endpoints,
                               geom, *, shallow_stride=1):
    """Subsample the transect and draw dot/cross markers for the sign of
    the cross-transect component. ``shallow_stride`` further decimates
    rows shallower than 100 m by that factor."""
    x_coords, depth_coords, h_bathy, max_depth_local, y_idx, x_idx = geom
    n_east, n_north = _normal_unit_vector(endpoints)

    # Subsample x indices.
    x_sel = np.arange(0, len(x_coords), MARKER_STRIDE_X)
    # Subsample depth indices, only within plotted depth window.
    z_sel = []
    for di, d in enumerate(depth_coords):
        if -d < max_depth_local:
            break
        z_sel.append(di)
    z_sel = z_sel[::MARKER_STRIDE_Z]

    # Further decimate the shallow (<100 m) rows by ``shallow_stride``.
    shallow = [di for di in z_sel if depth_coords[di] < 100]
    deep = [di for di in z_sel if depth_coords[di] >= 100]
    z_sel = sorted(set(shallow[::shallow_stride]) | set(deep))

    xs_out, zs_out = [], []
    xs_in, zs_in = [], []
    for xi in x_sel:
        j, k = y_idx[xi], x_idx[xi]
        bathy_xi = h_bathy[xi]
        for di in z_sel:
            z_val = -depth_coords[di]
            if z_val < bathy_xi:
                continue
            u = u_field[di, j, k]
            v = v_field[di, j, k]
            if not (np.isfinite(u) and np.isfinite(v)):
                continue
            cross = u * n_east + v * n_north
            if cross >= 0:
                xs_out.append(x_coords[xi]); zs_out.append(z_val)
            else:
                xs_in.append(x_coords[xi]); zs_in.append(z_val)

    # Circle with a dot inside (out of page); circle with a cross (into page).
    if xs_out:
        ax.scatter(xs_out, zs_out, marker="o", s=MARKER_SIZE,
                   facecolors="none", edgecolors="black", linewidths=0.8,
                   zorder=15)
        ax.scatter(xs_out, zs_out, marker=".", s=MARKER_SIZE * 0.18,
                   c="black", zorder=16)
    if xs_in:
        ax.scatter(xs_in, zs_in, marker="o", s=MARKER_SIZE,
                   facecolors="none", edgecolors="black", linewidths=0.8,
                   zorder=15)
        ax.scatter(xs_in, zs_in, marker="x", s=MARKER_SIZE * 0.55,
                   c="black", linewidths=0.8, zorder=16)


# ---------------------------------------------------------------------------
# Direction-convention label
# ---------------------------------------------------------------------------

def _direction_words(endpoints):
    """Return a short human-readable description of the 'out of page'
    direction (where dots point)."""
    n_east, n_north = _normal_unit_vector(endpoints)
    parts = []
    if abs(n_north) > 0.3:
        parts.append("north" if n_north > 0 else "south")
    if abs(n_east) > 0.3:
        parts.append("east" if n_east > 0 else "west")
    return "+".join(parts) if parts else "horizontal"


# ---------------------------------------------------------------------------
# Build figure
# ---------------------------------------------------------------------------

def build_figure(out_path: Path, overwrite: bool,
                 depth_overrides: dict[str, int] | None = None):
    if out_path.exists() and not overwrite:
        logger.info(f"exists, skip ({out_path})")
        return
    if depth_overrides is None:
        depth_overrides = {}

    fig, axes = plt.subplots(len(ROWS), 2, figsize=(13, 10),
                             sharey="row")
    fig.subplots_adjust(left=0.08, right=0.93, top=0.85, bottom=0.10,
                        hspace=0.27, wspace=0.36)

    panel_letters = ["a", "b", "c", "d"]

    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerTuple

    _ep_lines = []
    for _c in ROWS:
        _gn, _ep = GATEWAYS[_c]
        _p1, _p2 = _ep
        _ep_lines.append(f"{_gn}: [{_p1[0]:.2f}°, {_p1[1]:.2f}°] → "
                         f"[{_p2[0]:.2f}°, {_p2[1]:.2f}°]")
    _endpoint_txt_global = " | ".join(_ep_lines)

    for ri, cfg in enumerate(ROWS):
        gateway_name, endpoints = GATEWAYS[cfg]
        limits = PLOT_LIMITS[cfg]
        max_depth = depth_overrides.get(cfg, limits["max_depth"])
        fpath = (BASE_DIR / cfg / "product_stats" / "full"
                 / "mean_2011_2021.nc")
        if not fpath.exists():
            logger.error(f"[{cfg}] missing {fpath}"); continue

        with xr.open_dataset(fpath) as ds:
            u = ds["u_gos_mean"].isel(time=0).values
            v = ds["v_gos_mean"].isel(time=0).values
            cuu = ds["vel_cov_uu"].isel(time=0).values
            cvv = ds["vel_cov_vv"].isel(time=0).values
            template = ds[["latitude", "longitude",
                           "elevation", "depth"]].load()

        intensity = np.hypot(u, v)
        std_tot = np.sqrt(np.maximum(cuu + cvv, 0.0))

        (p1, p2) = endpoints
        endpoint_txt = (
            f"[Lon, Lat] endpoints: [{p1[0]:.2f}°, {p1[1]:.2f}°], "
            f"[{p2[0]:.2f}°, {p2[1]:.2f}°]")

        # ---- mean panel (field may be intensity, u, or v) ----
        ax_mean = axes[ri, 0]
        mean_field_key = MEAN_FIELD.get(cfg, "intensity")
        if mean_field_key == "u":
            mean_data = u
            vmin_m, vmax_m = PANEL_LIMITS_UV[cfg]["u"]
            cmap_m = CMAP_UV
            mean_ticks = CBAR_TICKS_UV[cfg]["u"]
            mean_panel_title = "mean eastward velocity u"
            mean_cbar_label = "u (m s$^{-1}$)"
            mean_extend = "both"
        elif mean_field_key == "v":
            mean_data = v
            vmin_m, vmax_m = PANEL_LIMITS_UV[cfg]["v"]
            cmap_m = CMAP_UV
            mean_ticks = CBAR_TICKS_UV[cfg]["v"]
            mean_panel_title = "mean northward velocity v"
            mean_cbar_label = "v (m s$^{-1}$)"
            mean_extend = "both"
        else:
            mean_data = intensity
            vmin_m, vmax_m = PANEL_LIMITS[cfg]["mean"]
            cmap_m = CMAP_MEAN
            mean_ticks = CBAR_TICKS_MEAN[cfg]
            mean_panel_title = "mean current intensity |v|"
            mean_cbar_label = "|v| (m s$^{-1}$)"
            mean_extend = "max"

        pc_m, geom = _draw_field(
            ax_mean, template, mean_data, endpoints,
            cmap=cmap_m, vmin=vmin_m, vmax=vmax_m, max_depth=max_depth,
            extend=mean_extend)
        ax_mean.set_title(
            f"({panel_letters[ri * 2]}) {gateway_name} {mean_panel_title}",
            fontsize=TITLE_FS, fontweight='bold', pad=8)

        if mean_field_key == "intensity":
            _overlay_direction_markers(
                ax_mean, template, u, v, endpoints, geom,
                shallow_stride=SHALLOW_STRIDE.get(cfg, 1))
            out_dir = _direction_words(endpoints)
            opp_dir = _direction_words(((p2[0], p2[1]), (p1[0], p1[1])))
            circle_handle = Line2D(
                [0], [0], marker="o", color="none",
                markerfacecolor="none", markeredgecolor="black",
                markersize=8, linestyle="none")
            dot_handle = Line2D(
                [0], [0], marker=".", color="black",
                markersize=3, linestyle="none")
            x_handle = Line2D(
                [0], [0], marker="x", color="black",
                markersize=6, linestyle="none")
            ax_mean.legend(
                [(circle_handle, dot_handle), (circle_handle, x_handle)],
                [f"out of page ({out_dir})", f"into page ({opp_dir})"],
                handler_map={tuple: HandlerTuple(ndivide=1, pad=0)},
                loc="lower left",
                fontsize=LABEL_FS - 2, framealpha=0.75,
                handletextpad=0.5, borderpad=0.4)

        if pc_m is not None:
            cb_extend = "both" if mean_extend == "both" else "neither"
            _div_m = make_axes_locatable(ax_mean)
            _cax_m = _div_m.append_axes("right", size="5%", pad=0.08)
            cb = fig.colorbar(pc_m, cax=_cax_m, extend=cb_extend,
                              ticks=mean_ticks)
            cb.set_label(mean_cbar_label, fontsize=LABEL_FS)
            cb.ax.tick_params(labelsize=LABEL_FS)
            cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        # ---- std panel ----
        ax_std = axes[ri, 1]
        vmin_s, vmax_s = PANEL_LIMITS[cfg]["std"]
        pc_s, _ = _draw_field(
            ax_std, template, std_tot, endpoints,
            cmap=CMAP_STD, vmin=vmin_s, vmax=vmax_s, max_depth=max_depth)
        ax_std.set_title(
            f"({panel_letters[ri * 2 + 1]}) {gateway_name} "
            f"Velocity Standard Deviation σ",
            fontsize=TITLE_FS, fontweight='bold', pad=8)
        if pc_s is not None:
            _div_s = make_axes_locatable(ax_std)
            _cax_s = _div_s.append_axes("right", size="5%", pad=0.08)
            cb = fig.colorbar(pc_s, cax=_cax_s, extend="neither",
                              ticks=CBAR_TICKS_STD[cfg])
            cb.set_label("σ (m s$^{-1}$)",
                         fontsize=LABEL_FS)
            cb.ax.tick_params(labelsize=LABEL_FS)
            cb.ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

        # ---- axis labels (per row) ----
        _, _, _, xlabel = _transect_indices(template, endpoints)
        deg_fmt = FuncFormatter(lambda x, _: f"{x:g}°")
        abs_fmt = FuncFormatter(lambda y, _: f"{abs(y):g}")
        xticks = XAXIS_TICKS.get(cfg)
        for ci, ax in enumerate((ax_mean, ax_std)):
            ax.set_xlabel(xlabel.split(" [")[0] + " (°)",
                          fontsize=LABEL_FS)
            ax.xaxis.set_major_formatter(deg_fmt)
            ax.yaxis.set_major_formatter(abs_fmt)
            ax.set_ylabel("Depth (m)", fontsize=LABEL_FS)
            ax.tick_params(axis="both", labelsize=LABEL_FS, labelleft=True)
            if xticks is not None:
                ax.set_xticks(xticks)

        logger.info(f"[{cfg}] done")

    fig.suptitle("Gateway transects — 2011–2021 mean "
                 f"Geostrophic Velocity & Variability\n{_endpoint_txt_global}",
                 y=0.96)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"saved → {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overwrite", action="store_true",
                    help="Re-render and overwrite the existing PNG.")
    ap.add_argument(
        "--depth-clip", nargs="*", default=[], metavar="CFG:DEPTH",
        help=("Per-gateway depth clip in metres, e.g. fram_6p25:1000. "
              "Can be repeated or space-separated. "
              "Produces an extra output file with a depth suffix."))
    args = ap.parse_args()

    # Parse depth-clip pairs.
    depth_overrides: dict[str, int] = {}
    for item in (args.depth_clip or []):
        cfg_key, _, depth_str = item.partition(":")
        if not depth_str:
            ap.error(f"--depth-clip: expected CFG:DEPTH, got '{item}'")
        depth_overrides[cfg_key.strip()] = int(depth_str)

    # Build suffix from any clipped gateways, e.g. "_fram_6p25_1000m".
    suffix = "".join(
        f"_{cfg}_{d}m" for cfg, d in sorted(depth_overrides.items()))

    out_path = (BASE_DIR / "plots_common"
                / f"Fram_Barents_velocity_transects{suffix}.png")
    build_figure(out_path, args.overwrite, depth_overrides)
    logger.info("Done.")


if __name__ == "__main__":
    main()
