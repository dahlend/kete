"""
Repack Horizons SPK kernels in the kete cache.

Converts Type 1 (Hermite interpolating polynomial) segments downloaded from
JPL Horizons into compact Type 2 (Chebyshev) segments.  Repacked files are
written alongside the originals with a ``_repack`` suffix.

Usage
-----
    python repack_horizons_kernels.py [--threshold KM] [--degree N] [--force] [--dir PATH]

Options
-------
--threshold KM   Position accuracy threshold in km (default: 1.0)
--degree N       Chebyshev polynomial degree (default: 15)
--force          Overwrite existing repacked files
--dir PATH       Kernel directory to process (default: ~/.kete/kernels)
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import kete
import kete.spice

CENTER_ID = 10  # Horizons kernels use the Sun (NAIF 10) as center


def repack_file(
    src: str, threshold_km: float, degree: int, force: bool
) -> tuple[str, bool]:
    """
    Repack a single BSP file.

    Returns (message, success).
    """
    dst = src.replace(".bsp", "_repack.bsp")
    if dst == src:
        dst = src + "_repack.bsp"

    if os.path.exists(dst) and not force:
        sz = os.path.getsize(dst)
        return f"  SKIP  {os.path.basename(src)} (repack exists, {sz // 1024} KB)", True

    # Load the source file and identify the objects inside.
    try:
        kete.spice.kernel_reload(filenames=[src], include_planets=False)
    except Exception as exc:
        return f"  ERROR {os.path.basename(src)}: failed to load — {exc}", False

    objects = kete.spice.loaded_objects()
    if not objects:
        return f"  SKIP  {os.path.basename(src)}: no objects found", True

    sz_in = os.path.getsize(src)
    t0 = time.monotonic()

    try:
        results = kete.spice.repack_spk(
            src,
            dst,
            center_id=CENTER_ID,
            threshold_km=threshold_km,
            degree=degree,
            output_type=2,
        )
    except Exception as exc:
        return f"  ERROR {os.path.basename(src)}: repack failed — {exc}", False

    output_type = 2

    elapsed = time.monotonic() - t0
    sz_out = os.path.getsize(dst)
    ratio = sz_in / sz_out if sz_out else float("inf")
    n_arrays = sum(r[1] for r in results)
    n_records = sum(r[2] for r in results)
    obj_ids = [r[0] for r in results]

    msg = (
        f"  OK    {os.path.basename(src)}"
        f"  {sz_in // 1024:>6} KB → {sz_out // 1024:>6} KB  ({ratio:.1f}x)"
        f"  type={output_type}  {n_arrays} segs / {n_records} recs"
        f"  obj={obj_ids}"
        f"  [{elapsed:.1f}s]"
    )
    return msg, True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        metavar="KM",
        help="Position accuracy threshold in km (default: 1.0)",
    )
    parser.add_argument(
        "--degree",
        type=int,
        default=15,
        metavar="N",
        help="Chebyshev polynomial degree (default: 15)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing repacked files",
    )
    parser.add_argument(
        "--dir",
        default=None,
        metavar="PATH",
        help="Kernel directory to process (default: ~/.kete/kernels)",
    )
    args = parser.parse_args()

    kernel_dir = args.dir or os.path.join(kete.cache.cache_path(), "kernels", "core")
    if not os.path.isdir(kernel_dir):
        print(f"Directory not found: {kernel_dir}", file=sys.stderr)
        sys.exit(1)

    # Find all .bsp files that are not already repacked files.
    bsp_files = sorted(
        p
        for p in (os.path.join(kernel_dir, f) for f in os.listdir(kernel_dir))
        if p.endswith(".bsp") and not p.endswith("_repack.bsp")
    )

    if not bsp_files:
        print(f"No .bsp files found in {kernel_dir}")
        return

    print(
        f"Repacking {len(bsp_files)} kernel(s) in {kernel_dir}"
        f"  threshold={args.threshold} km  degree={args.degree}"
    )
    print()

    n_ok = n_skip = n_err = 0
    for src in bsp_files:
        msg, ok = repack_file(src, args.threshold, args.degree, args.force)
        print(msg)
        if "SKIP" in msg:
            n_skip += 1
        elif ok:
            n_ok += 1
        else:
            n_err += 1

    print()
    print(f"Done: {n_ok} repacked, {n_skip} skipped, {n_err} errors")


if __name__ == "__main__":
    main()
