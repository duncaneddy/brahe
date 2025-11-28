#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "brahe",
#     "skyfield>=1.49",
#     "numpy",
#     "plotly",
# ]
# ///
"""
Benchmark and validate access computation between brahe and skyfield libraries.

Compares:
1. AOS/LOS times between implementations (validation)
2. Execution time per location (benchmarking)
3. Generates plotly comparison chart

Usage:
    uv run scripts/benchmark_access_skyfield.py --n-locations 10 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import random
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np
import plotly.graph_objects as go
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.timelib import Time
from skyfield.toposlib import GeographicPosition

import brahe as bh

# Skyfield time scale (loaded once)
time_scale = load.timescale()

# Default ISS TLE
DEFAULT_TLE_LINE1 = (
    "1 25544U 98067A   25321.32008588  .00017447  00000-0  31567-3 0  9995"
)
DEFAULT_TLE_LINE2 = (
    "2 25544  51.6331 268.2969 0004176  80.2379 279.9081 15.49753355538919"
)

# Constants
MIN_ELEVATION_DEG = 5.0
WINDOW_DAYS = 2


@dataclass
class Target:
    """Ground station target location."""

    latitude: float  # degrees
    longitude: float  # degrees
    altitude: float  # meters
    name: str | None = None


@dataclass
class TLE:
    """Two-line element set."""

    sat_name: str
    line1: str
    line2: str


@dataclass
class Access:
    """Access window with AOS/LOS times."""

    aos: datetime
    los: datetime
    target: Target | None = None


class EventType(IntEnum):
    """Skyfield event types."""

    AOS = 0
    CULMINATION = 1
    LOS = 2


@dataclass
class ValidationResult:
    """Result of validating two sets of access windows."""

    passed: bool
    warnings: list[str]
    max_diff_aos: float
    max_diff_los: float
    num_matched: int
    num_unmatched_brahe: int
    num_unmatched_skyfield: int


@dataclass
class BenchmarkResult:
    """Benchmark timing result."""

    mean_time: float
    std_time: float
    num_windows: int


def generate_random_locations(n: int, seed: int | None = None) -> list[Target]:
    """Generate N random ground station locations worldwide."""
    if seed is not None:
        random.seed(seed)

    locations = []
    for i in range(n):
        lat = random.uniform(-70, 70)  # Avoid extreme poles
        lon = random.uniform(-180, 180)
        alt = random.uniform(0, 500)  # 0-500m altitude
        name = f"Location_{i + 1}"
        locations.append(Target(latitude=lat, longitude=lon, altitude=alt, name=name))

    return locations


def compute_skyfield_accesses(
    tle: TLE, target: Target, window_open: datetime, window_close: datetime
) -> list[Access]:
    """Compute access windows using skyfield."""
    t0: Time = time_scale.from_datetime(window_open.replace(tzinfo=UTC))
    t1: Time = time_scale.from_datetime(window_close.replace(tzinfo=UTC))
    topo: GeographicPosition = wgs84.latlon(
        target.latitude, target.longitude, target.altitude
    )
    satellite = EarthSatellite(tle.line1, tle.line2, tle.sat_name, time_scale)
    times, events = satellite.find_events(
        topo, t0, t1, altitude_degrees=MIN_ELEVATION_DEG
    )

    accesses = []
    aos, los = None, None
    for event_time, event in zip(times, events):
        if event == EventType.AOS:
            aos = event_time.utc_datetime()
        if event == EventType.LOS:
            los = event_time.utc_datetime()
        if aos and los:
            accesses.append(Access(aos=aos, los=los, target=target))
            aos, los = None, None

    return accesses


def compute_brahe_accesses(
    tle: TLE, target: Target, window_open: datetime, window_close: datetime
) -> list[Access]:
    """Compute access windows using brahe."""
    # Create propagator from TLE
    propagator = bh.SGPPropagator.from_tle(tle.line1, tle.line2, 60.0).with_name(
        tle.sat_name
    )

    # Create location (brahe uses lon, lat, alt order)
    location = bh.PointLocation(
        target.longitude, target.latitude, target.altitude
    ).with_name(target.name or "Unknown")

    # Create epoch objects
    epoch_start = bh.Epoch.from_datetime(
        window_open.year,
        window_open.month,
        window_open.day,
        window_open.hour,
        window_open.minute,
        float(window_open.second),
        0.0,
        bh.TimeSystem.UTC,
    )
    epoch_end = bh.Epoch.from_datetime(
        window_close.year,
        window_close.month,
        window_close.day,
        window_close.hour,
        window_close.minute,
        float(window_close.second),
        0.0,
        bh.TimeSystem.UTC,
    )

    # Define constraint
    constraint = bh.ElevationConstraint(min_elevation_deg=MIN_ELEVATION_DEG)

    # Compute access windows
    windows = bh.location_accesses(
        location, propagator, epoch_start, epoch_end, constraint
    )

    # Convert to Access objects
    accesses = []
    for window in windows:
        # to_datetime() returns (year, month, day, hour, minute, second, nanosecond)
        aos_tuple = window.window_open.to_datetime()
        los_tuple = window.window_close.to_datetime()

        # Convert tuples to datetime objects
        aos_dt = datetime(
            int(aos_tuple[0]),
            int(aos_tuple[1]),
            int(aos_tuple[2]),
            int(aos_tuple[3]),
            int(aos_tuple[4]),
            int(aos_tuple[5]),
            int(aos_tuple[6] / 1000),  # nanoseconds to microseconds
            tzinfo=UTC,
        )
        los_dt = datetime(
            int(los_tuple[0]),
            int(los_tuple[1]),
            int(los_tuple[2]),
            int(los_tuple[3]),
            int(los_tuple[4]),
            int(los_tuple[5]),
            int(los_tuple[6] / 1000),  # nanoseconds to microseconds
            tzinfo=UTC,
        )
        accesses.append(Access(aos=aos_dt, los=los_dt, target=target))

    return accesses


def validate_results(
    brahe_accesses: list[Access],
    skyfield_accesses: list[Access],
    tolerance_seconds: float = 0.1,
) -> ValidationResult:
    """
    Compare AOS/LOS times between implementations.

    Matches windows by closest AOS time and validates LOS within tolerance.
    """
    warnings = []
    max_diff_aos = 0.0
    max_diff_los = 0.0
    matched_brahe = set()
    matched_skyfield = set()

    # Match windows by AOS time
    for i, b_access in enumerate(brahe_accesses):
        best_match_idx = None
        best_diff = float("inf")

        for j, s_access in enumerate(skyfield_accesses):
            if j in matched_skyfield:
                continue
            aos_diff = abs((b_access.aos - s_access.aos).total_seconds())
            if aos_diff < best_diff:
                best_diff = aos_diff
                best_match_idx = j

        if (
            best_match_idx is not None and best_diff < 60.0
        ):  # Allow up to 60s for matching
            s_access = skyfield_accesses[best_match_idx]
            aos_diff = abs((b_access.aos - s_access.aos).total_seconds())
            los_diff = abs((b_access.los - s_access.los).total_seconds())

            max_diff_aos = max(max_diff_aos, aos_diff)
            max_diff_los = max(max_diff_los, los_diff)

            if aos_diff > tolerance_seconds:
                warnings.append(
                    f"AOS mismatch at window {i + 1}: diff={aos_diff:.3f}s > tolerance={tolerance_seconds}s"
                )
            if los_diff > tolerance_seconds:
                warnings.append(
                    f"LOS mismatch at window {i + 1}: diff={los_diff:.3f}s > tolerance={tolerance_seconds}s"
                )

            matched_brahe.add(i)
            matched_skyfield.add(best_match_idx)

    num_unmatched_brahe = len(brahe_accesses) - len(matched_brahe)
    num_unmatched_skyfield = len(skyfield_accesses) - len(matched_skyfield)

    if num_unmatched_brahe > 0:
        warnings.append(f"{num_unmatched_brahe} brahe windows not matched to skyfield")
    if num_unmatched_skyfield > 0:
        warnings.append(
            f"{num_unmatched_skyfield} skyfield windows not matched to brahe"
        )

    passed = len(warnings) == 0

    return ValidationResult(
        passed=passed,
        warnings=warnings,
        max_diff_aos=max_diff_aos,
        max_diff_los=max_diff_los,
        num_matched=len(matched_brahe),
        num_unmatched_brahe=num_unmatched_brahe,
        num_unmatched_skyfield=num_unmatched_skyfield,
    )


def benchmark_function(
    func: Callable[..., Any], args: tuple, iterations: int = 5
) -> BenchmarkResult:
    """Benchmark a function and return mean/std execution time."""
    times = []
    result = None

    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times_arr = np.array(times)
    num_windows = len(result) if result is not None else 0

    return BenchmarkResult(
        mean_time=float(np.mean(times_arr)),
        std_time=float(np.std(times_arr)),
        num_windows=num_windows,
    )


def export_accesses_to_csv(
    all_brahe_accesses: list[tuple[Target, list[Access]]],
    all_skyfield_accesses: list[tuple[Target, list[Access]]],
    output_file: str,
) -> None:
    """Export all access windows to a CSV file."""
    with Path(output_file).open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "library",
                "location_name",
                "latitude",
                "longitude",
                "altitude",
                "aos",
                "los",
                "duration_s",
            ]
        )

        for target, accesses in all_brahe_accesses:
            for access in accesses:
                duration = (access.los - access.aos).total_seconds()
                writer.writerow(
                    [
                        "brahe",
                        target.name or "Unknown",
                        f"{target.latitude:.6f}",
                        f"{target.longitude:.6f}",
                        f"{target.altitude:.1f}",
                        access.aos.isoformat(),
                        access.los.isoformat(),
                        f"{duration:.3f}",
                    ]
                )

        for target, accesses in all_skyfield_accesses:
            for access in accesses:
                duration = (access.los - access.aos).total_seconds()
                writer.writerow(
                    [
                        "skyfield",
                        target.name or "Unknown",
                        f"{target.latitude:.6f}",
                        f"{target.longitude:.6f}",
                        f"{target.altitude:.1f}",
                        access.aos.isoformat(),
                        access.los.isoformat(),
                        f"{duration:.3f}",
                    ]
                )

    print(f"Access windows saved to: {output_file}")


def create_comparison_chart(
    locations: list[Target],
    brahe_results: list[BenchmarkResult],
    skyfield_results: list[BenchmarkResult],
    output_file: str,
) -> None:
    """Create a plotly grouped bar chart comparing execution times."""
    location_names = [loc.name or f"Loc {i + 1}" for i, loc in enumerate(locations)]

    fig = go.Figure()

    # Brahe bars
    fig.add_trace(
        go.Bar(
            name="Brahe",
            x=location_names,
            y=[r.mean_time * 1000 for r in brahe_results],  # Convert to ms
            error_y=dict(
                type="data",
                array=[r.std_time * 1000 for r in brahe_results],
                visible=True,
            ),
            marker_color="rgb(55, 83, 109)",
        )
    )

    # Skyfield bars
    fig.add_trace(
        go.Bar(
            name="Skyfield",
            x=location_names,
            y=[r.mean_time * 1000 for r in skyfield_results],  # Convert to ms
            error_y=dict(
                type="data",
                array=[r.std_time * 1000 for r in skyfield_results],
                visible=True,
            ),
            marker_color="rgb(26, 118, 255)",
        )
    )

    fig.update_layout(
        title="Brahe vs Skyfield Access Computation Performance",
        xaxis_title="Location",
        yaxis_title="Execution Time (ms)",
        barmode="group",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255, 255, 255, 0.8)"),
        template="plotly_white",
    )

    fig.write_html(output_file)
    print(f"\nChart saved to: {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark brahe vs skyfield access computation"
    )
    parser.add_argument(
        "--n-locations",
        type=int,
        default=10,
        help="Number of random locations to test (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=1.0,
        help="Validation tolerance in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Benchmark iterations per location (default: 5)",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip generating plotly chart",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_access_comparison.html",
        help="Output HTML file for plot (default: benchmark_access_comparison.html)",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Output CSV file for access windows (optional)",
    )
    args = parser.parse_args()

    # Initialize brahe EOP data
    bh.initialize_eop()

    # Generate locations
    locations = generate_random_locations(args.n_locations, args.seed)

    # TLE and time window
    tle = TLE(
        sat_name="ISS (ZARYA)",
        line1=DEFAULT_TLE_LINE1,
        line2=DEFAULT_TLE_LINE2,
    )
    window_open = datetime(2025, 11, 18, tzinfo=UTC)
    window_close = datetime(2025, 11, 18 + WINDOW_DAYS, tzinfo=UTC)

    # Print header
    seed_str = str(args.seed) if args.seed is not None else "None"
    print("=" * 60)
    print("Brahe vs Skyfield Access Computation Benchmark")
    print("=" * 60)
    print(
        f"Locations: {args.n_locations} | Seed: {seed_str} | "
        f"Window: {WINDOW_DAYS} days | Min Elevation: {MIN_ELEVATION_DEG}° | "
        f"Tolerance: {args.tolerance}s"
    )
    print()

    # Results storage
    brahe_results: list[BenchmarkResult] = []
    skyfield_results: list[BenchmarkResult] = []
    all_brahe_accesses: list[tuple[Target, list[Access]]] = []
    all_skyfield_accesses: list[tuple[Target, list[Access]]] = []
    all_passed = 0
    all_warned = 0

    # Run benchmarks for each location
    for i, location in enumerate(locations):
        print(
            f"Location {i + 1}: {location.name} ({location.latitude:.2f}, {location.longitude:.2f})"
        )

        # Benchmark brahe
        brahe_bench = benchmark_function(
            compute_brahe_accesses,
            (tle, location, window_open, window_close),
            iterations=args.iterations,
        )
        brahe_results.append(brahe_bench)

        # Benchmark skyfield
        skyfield_bench = benchmark_function(
            compute_skyfield_accesses,
            (tle, location, window_open, window_close),
            iterations=args.iterations,
        )
        skyfield_results.append(skyfield_bench)

        # Get actual results for validation
        brahe_accesses = compute_brahe_accesses(
            tle, location, window_open, window_close
        )
        skyfield_accesses = compute_skyfield_accesses(
            tle, location, window_open, window_close
        )

        # Store for CSV export
        all_brahe_accesses.append((location, brahe_accesses))
        all_skyfield_accesses.append((location, skyfield_accesses))

        # Validate
        validation = validate_results(
            brahe_accesses, skyfield_accesses, tolerance_seconds=args.tolerance
        )

        # Print results
        print(
            f"  Brahe:    {brahe_bench.num_windows:3d} windows, "
            f"{brahe_bench.mean_time:.4f}s ± {brahe_bench.std_time:.4f}s"
        )
        print(
            f"  Skyfield: {skyfield_bench.num_windows:3d} windows, "
            f"{skyfield_bench.mean_time:.4f}s ± {skyfield_bench.std_time:.4f}s"
        )

        # Print warnings
        for warning in validation.warnings:
            print(f"  WARNING: {warning}")

        if validation.passed:
            print(
                f"  Validation: PASS (max diff: {max(validation.max_diff_aos, validation.max_diff_los):.4f}s)"
            )
            all_passed += 1
        else:
            print(
                f"  Validation: WARN ({len(validation.warnings)} issues, "
                f"max diff: {max(validation.max_diff_aos, validation.max_diff_los):.4f}s)"
            )
            all_warned += 1

        print()

    # Summary
    print("=" * 60)
    print("Summary")
    print("=" * 60)

    total_brahe_windows = sum(r.num_windows for r in brahe_results)
    total_skyfield_windows = sum(r.num_windows for r in skyfield_results)
    avg_brahe_time = np.mean([r.mean_time for r in brahe_results])
    avg_skyfield_time = np.mean([r.mean_time for r in skyfield_results])
    speedup = avg_skyfield_time / avg_brahe_time if avg_brahe_time > 0 else 0

    print(
        f"Total windows: Brahe={total_brahe_windows}, Skyfield={total_skyfield_windows}"
    )
    print(f"Avg time: Brahe={avg_brahe_time:.4f}s, Skyfield={avg_skyfield_time:.4f}s")
    print(f"Speedup: {speedup:.2f}x")
    print(f"Validation: {all_passed}/{args.n_locations} passed, {all_warned} warnings")

    # Generate chart
    if not args.no_plot:
        create_comparison_chart(locations, brahe_results, skyfield_results, args.output)

    # Export CSV if requested
    if args.csv:
        export_accesses_to_csv(all_brahe_accesses, all_skyfield_accesses, args.csv)


if __name__ == "__main__":
    main()
