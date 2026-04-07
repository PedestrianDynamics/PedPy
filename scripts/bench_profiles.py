#!/usr/bin/env python3
"""Benchmark profile computation: timing and peak memory.

Usage:
    python scripts/bench_profiles.py [--frames N] [--peds N] [--grid-size F] [--repeats N]

Run on both main and the feature branch to compare.
"""

import argparse
import time
import tracemalloc

import numpy as np
import pandas as pd
import shapely

from pedpy.data.geometry import WalkableArea
from pedpy.methods.profile_calculator import (
    DensityMethod,
    SpeedMethod,
    compute_density_profile,
    compute_grid_cell_polygon_intersection_area,
    compute_profiles,
    compute_speed_profile,
    get_grid_cells,
)


def generate_data(n_frames: int, n_peds: int, area_size: float = 10.0) -> pd.DataFrame:
    """Generate synthetic pedestrian data with Voronoi-like polygons."""
    rng = np.random.default_rng(42)
    rows = []
    for frame in range(n_frames):
        xs = rng.uniform(0.5, area_size - 0.5, n_peds)
        ys = rng.uniform(0.5, area_size - 0.5, n_peds)
        speeds = rng.uniform(0.5, 2.0, n_peds)
        # Simple square polygons centered on each pedestrian
        cell_size = 0.4
        polys = [shapely.box(x - cell_size, y - cell_size, x + cell_size, y + cell_size) for x, y in zip(xs, ys)]
        for i in range(n_peds):
            rows.append(
                {
                    "id": i,
                    "frame": frame,
                    "x": xs[i],
                    "y": ys[i],
                    "speed": speeds[i],
                    "polygon": polys[i],
                }
            )
    return pd.DataFrame(rows)


def bench(func, label, repeats=3):
    """Run func, measure wall time and peak memory."""
    times = []
    peak_mem = 0
    for _ in range(repeats):
        tracemalloc.start()
        t0 = time.perf_counter()
        try:
            func()
        except Exception as e:
            tracemalloc.stop()
            print(f"  {label:45s}  SKIPPED ({type(e).__name__})")
            return
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        times.append(elapsed)
        peak_mem = max(peak_mem, peak)

    median_t = np.median(times)
    print(f"  {label:45s}  {median_t:7.3f}s  peak_mem={peak_mem / 1024 / 1024:7.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Benchmark profile computation")
    parser.add_argument("--frames", type=int, default=200, help="Number of frames")
    parser.add_argument("--peds", type=int, default=80, help="Pedestrians per frame")
    parser.add_argument("--grid-size", type=float, default=0.5, help="Grid cell size")
    parser.add_argument("--repeats", type=int, default=3, help="Timing repeats")
    args = parser.parse_args()

    area_size = 10.0
    walkable_area = WalkableArea(shapely.box(0, 0, area_size, area_size))

    print(f"Generating data: {args.frames} frames x {args.peds} peds = {args.frames * args.peds} rows")
    data = generate_data(args.frames, args.peds, area_size)
    print(f"Grid size: {args.grid_size}  =>  {int(area_size / args.grid_size)}x{int(area_size / args.grid_size)} cells")
    print()

    grid_cells, _, _ = get_grid_cells(walkable_area=walkable_area, grid_size=args.grid_size)

    # --- With precomputed intersections ---
    print("Precomputing grid intersections...")
    t0 = time.perf_counter()
    precomputed, sorted_data = compute_grid_cell_polygon_intersection_area(data=data, grid_cells=grid_cells)
    print(f"  Precomputation took {time.perf_counter() - t0:.3f}s")
    print()

    print(f"Benchmarks (median of {args.repeats} runs):")

    bench(
        lambda: compute_density_profile(
            data=sorted_data,
            walkable_area=walkable_area,
            grid_size=args.grid_size,
            density_method=DensityMethod.VORONOI,
            grid_intersections_area=precomputed,
        ),
        "density_profile (voronoi, precomputed)",
        args.repeats,
    )

    bench(
        lambda: compute_density_profile(
            data=data,
            walkable_area=walkable_area,
            grid_size=args.grid_size,
            density_method=DensityMethod.VORONOI,
        ),
        "density_profile (voronoi, on-the-fly)",
        args.repeats,
    )

    bench(
        lambda: compute_speed_profile(
            data=sorted_data,
            walkable_area=walkable_area,
            grid_size=args.grid_size,
            speed_method=SpeedMethod.VORONOI,
            grid_intersections_area=precomputed,
        ),
        "speed_profile (voronoi, precomputed)",
        args.repeats,
    )

    bench(
        lambda: compute_speed_profile(
            data=data,
            walkable_area=walkable_area,
            grid_size=args.grid_size,
            speed_method=SpeedMethod.VORONOI,
        ),
        "speed_profile (voronoi, on-the-fly)",
        args.repeats,
    )

    bench(
        lambda: compute_profiles(
            data=data,
            walkable_area=walkable_area,
            grid_size=args.grid_size,
            speed_method=SpeedMethod.VORONOI,
        ),
        "compute_profiles (voronoi, on-the-fly)",
        args.repeats,
    )


if __name__ == "__main__":
    main()
