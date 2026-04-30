#!/usr/bin/env python3
"""
Patch the ``time`` variable of existing reconstruction NetCDF files so that
they all share the same epoch::

    units    = "days since 1950-01-01T00:00:00+00:00"
    calendar = "standard"
    dtype    = float64

Why this exists
---------------
Earlier versions of ``D_arctic_reconstruction.py`` did not pin a time epoch
when writing the output NetCDF files. xarray therefore used each file's own
timestamp as the reference epoch, producing files with::

    time = [0]   (int64)
    units = "days since <that file's date>"

This is technically valid CF, but concatenating files across dates becomes
incorrect unless every file is decoded individually first. The pipeline now
writes a fixed epoch (``1950-01-01T00:00:00+00:00``) as float64, and this
script back-fills the same convention into already-produced files.

How it works
------------
NetCDF4/HDF5 does not allow changing a variable's dtype (int64 -> float64)
in place, and our reconstruction targets are at 12:00 UTC (X.5 days), so a
true in-place patch would lose the noon offset. We therefore rewrite each
file via a sibling temp file, then atomically replace the original:

    * every variable other than ``time`` is copied **raw** (auto-scaling
      disabled) -- int16-quantized data is preserved byte-for-byte. No
      requantization, no precision loss, no compression rebuild.
    * ``time`` is recreated as float64 with the target units and calendar.

The on-disk size cost of switching ``time`` from int64 to float64 is zero
(single-element variable, 8 bytes either way).
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import netCDF4 as nc4
import cftime


# ---------------------------------------------------------------------------
# Target convention — MUST stay in sync with D_arctic_reconstruction.py
# ---------------------------------------------------------------------------
TARGET_UNITS = "days since 1950-01-01T00:00:00+00:00"
TARGET_CALENDAR = "standard"
TARGET_DTYPE = "f8"  # float64
TARGET_VAR_ATTRS = {
    "standard_name": "time",
    "units": TARGET_UNITS,
    "long_name": "Time",
    "axis": "T",
    "calendar": TARGET_CALENDAR,
}


# ---------------------------------------------------------------------------
# Variable attribute overrides — also kept in sync with the pipeline.
# Maps variable name -> {attr: new_value, ...}.  Use None to delete an attr.
#
# Rationale: T_anom_pred and T_recon are reconstructions of in-situ
# temperature (the LSTM target was in-situ TEMP from Argo/EN.4), not strictly
# potential temperature, even though GLORYS thetao is used as the reference
# background.  The 'potential' qualifier was inaccurate in the long_names
# and in T_recon's standard_name.
# ---------------------------------------------------------------------------
VAR_ATTR_OVERRIDES: dict[str, dict[str, str | None]] = {
    "T_anom_pred": {
        "long_name": "Predicted temperature anomaly",
        # No standard_name was previously set; make sure none lingers.
        "standard_name": None,
    },
    "T_recon": {
        "long_name": "Reconstructed temperature (anomaly + GLORYS reference)",
        "standard_name": "sea_water_temperature",
    },
}


def _var_attrs_already_ok(src: nc4.Dataset) -> bool:
    """True if every override-tracked variable already has the target attrs."""
    for vname, overrides in VAR_ATTR_OVERRIDES.items():
        if vname not in src.variables:
            continue
        var = src.variables[vname]
        existing = set(var.ncattrs())
        for k, v in overrides.items():
            current = var.getncattr(k) if k in existing else None
            if v is None:
                if current is not None:
                    return False
            else:
                if current != v:
                    return False
    return True


def _is_already_ok(src: nc4.Dataset) -> bool:
    """True if the file already uses the target time convention AND var attrs."""
    if "time" not in src.variables:
        return False
    tvar = src.variables["time"]
    units = getattr(tvar, "units", None)
    calendar = getattr(tvar, "calendar", "standard")
    dtype_ok = np.dtype(tvar.dtype) == np.dtype(np.float64)
    time_ok = (
        units == TARGET_UNITS
        and calendar in (TARGET_CALENDAR, "gregorian")
        and dtype_ok
    )
    return time_ok and _var_attrs_already_ok(src)


def _copy_attrs(src_var, dst_var, skip=()):
    """Copy attributes from src_var to dst_var, except those in *skip*."""
    for k in src_var.ncattrs():
        if k in skip or k == "_FillValue":
            # _FillValue must be set at variable creation, not after.
            continue
        dst_var.setncattr(k, src_var.getncattr(k))


def _rewrite_with_patched_time(src_path: Path, tmp_path: Path) -> None:
    """Rewrite *src_path* into *tmp_path* with patched time variable."""
    with nc4.Dataset(src_path, mode="r") as src, \
         nc4.Dataset(tmp_path, mode="w", format=src.data_model) as dst:

        # --- global attributes ---
        for k in src.ncattrs():
            dst.setncattr(k, src.getncattr(k))

        # --- dimensions ---
        for name, dim in src.dimensions.items():
            dst.createDimension(name, (len(dim) if not dim.isunlimited() else None))

        # --- variables ---
        for name, src_var in src.variables.items():
            if name == "time":
                old_units = getattr(src_var, "units", None)
                old_calendar = getattr(src_var, "calendar", "standard")
                old_values = np.asarray(src_var[:])

                if old_units is None:
                    raise RuntimeError(
                        f"{src_path.name}: 'time' is missing its 'units' attribute"
                    )

                decoded = cftime.num2date(
                    old_values,
                    units=old_units,
                    calendar=old_calendar or "standard",
                    only_use_cftime_datetimes=False,
                    only_use_python_datetimes=True,
                )
                new_values = np.asarray(
                    cftime.date2num(decoded,
                                    units=TARGET_UNITS,
                                    calendar=TARGET_CALENDAR),
                    dtype=np.float64,
                )

                dst_var = dst.createVariable(
                    name,
                    TARGET_DTYPE,
                    dimensions=src_var.dimensions,
                    zlib=False,
                )
                for k, v in TARGET_VAR_ATTRS.items():
                    dst_var.setncattr(k, v)
                dst_var[:] = new_values
                continue

            # --- all other variables: byte-identical raw copy ---
            # Disable auto-scaling/masking so reads return the raw on-disk
            # int16 (or whatever) without applying scale_factor/add_offset.
            src_var.set_auto_maskandscale(False)

            kwargs = dict(
                varname=name,
                datatype=src_var.dtype,
                dimensions=src_var.dimensions,
            )
            filters = src_var.filters() or {}
            kwargs["zlib"] = bool(filters.get("zlib", False))
            kwargs["complevel"] = int(filters.get("complevel", 0)) or 4
            kwargs["shuffle"] = bool(filters.get("shuffle", False))
            kwargs["fletcher32"] = bool(filters.get("fletcher32", False))
            chunking = src_var.chunking()
            if chunking and chunking != "contiguous":
                kwargs["chunksizes"] = tuple(chunking)
            if "_FillValue" in src_var.ncattrs():
                kwargs["fill_value"] = src_var.getncattr("_FillValue")

            dst_var = dst.createVariable(**kwargs)
            dst_var.set_auto_maskandscale(False)

            overrides = VAR_ATTR_OVERRIDES.get(name, {})
            attrs_to_drop = {k for k, v in overrides.items() if v is None}
            attrs_to_set = {k: v for k, v in overrides.items() if v is not None}
            _copy_attrs(src_var, dst_var,
                        skip=tuple(attrs_to_drop | set(attrs_to_set)))
            for k, v in attrs_to_set.items():
                dst_var.setncattr(k, v)

            if src_var.shape == ():
                dst_var[...] = src_var[...]
            else:
                dst_var[:] = src_var[:]


def patch_file(path: Path, dry_run: bool = False) -> str:
    """
    Patch one NetCDF file via a sibling temp file + atomic replace.

    Returns one of: 'patched', 'skipped (already ok)', 'skipped (no time variable)',
    'error: ...'.
    """
    try:
        with nc4.Dataset(path, mode="r") as src:
            if "time" not in src.variables:
                return "skipped (no time variable)"
            if _is_already_ok(src):
                return "skipped (already ok)"
            tvar = src.variables["time"]
            old_units = getattr(tvar, "units", None)
            old_calendar = getattr(tvar, "calendar", "standard")
            old_values = np.asarray(tvar[:])

        if old_units is None:
            return "error: missing 'units' attribute on time"

        if dry_run:
            decoded = cftime.num2date(
                old_values,
                units=old_units,
                calendar=old_calendar or "standard",
                only_use_cftime_datetimes=False,
                only_use_python_datetimes=True,
            )
            new_values = np.asarray(
                cftime.date2num(decoded,
                                units=TARGET_UNITS,
                                calendar=TARGET_CALENDAR),
                dtype=np.float64,
            )
            return (
                f"would patch: old_units={old_units!r} "
                f"old_values={list(old_values)} -> new_values={list(new_values)}"
            )

        tmp_fd, tmp_name = tempfile.mkstemp(
            prefix=path.stem + ".",
            suffix=".tmp.nc",
            dir=str(path.parent),
        )
        os.close(tmp_fd)
        tmp_path = Path(tmp_name)
        try:
            _rewrite_with_patched_time(path, tmp_path)
            shutil.move(str(tmp_path), str(path))
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise
        return "patched"

    except Exception as exc:  # pragma: no cover — defensive
        return f"error: {exc!r}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Patch time units in reconstruction NetCDF files.",
    )
    parser.add_argument(
        "root",
        type=Path,
        help="Output directory to walk recursively (e.g. .../arctic_25km).",
    )
    parser.add_argument(
        "--pattern",
        default="reconstruction_*.nc",
        help="Glob pattern to match (default: reconstruction_*.nc).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would change without writing.",
    )
    args = parser.parse_args()

    if not args.root.exists():
        print(f"ERROR: root does not exist: {args.root}", file=sys.stderr)
        return 2

    files = sorted(args.root.rglob(args.pattern))
    if not files:
        print(f"No files matching {args.pattern!r} under {args.root}")
        return 0

    print(f"Found {len(files)} files. dry_run={args.dry_run}")
    print(f"Target units    : {TARGET_UNITS}")
    print(f"Target calendar : {TARGET_CALENDAR}")
    print(f"Target dtype    : float64\n")

    counts: dict[str, int] = {}
    first_examples: dict[str, str] = {}
    for i, f in enumerate(files, 1):
        status = patch_file(f, dry_run=args.dry_run)
        key = status.split(":", 1)[0].strip()
        counts[key] = counts.get(key, 0) + 1
        if key not in first_examples:
            first_examples[key] = f"{f.name} -> {status}"
        if i % 100 == 0 or i == len(files):
            print(f"  [{i}/{len(files)}]  " +
                  "  ".join(f"{k}={v}" for k, v in counts.items()))

    print("\nSummary:")
    for k, v in counts.items():
        print(f"  {k}: {v}")
    print("\nFirst example per outcome:")
    for k, ex in first_examples.items():
        print(f"  {k}: {ex}")

    # Non-zero exit code if any file errored.
    return 1 if any(k.startswith("error") for k in counts) else 0


if __name__ == "__main__":
    sys.exit(main())
