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
Five-way benchmark comparing access computation between:
1. Skyfield (Python) - baseline
2. Brahe Python bindings (serial) - one location per call
3. Brahe Python bindings (parallel) - all locations in single call
4. Brahe Rust native (serial) - one location per call (via rust-script)
5. Brahe Rust native (parallel) - all locations in single call (via rust-script)

Usage:
    uv run scripts/benchmark_access_three_way.py --n-locations 10 --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
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

# Path to Rust script
RUST_SCRIPT_PATH = Path(__file__).parent / "benchmark_access_rust.rs"


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
    max_diff: float


@dataclass
class BenchmarkResult:
    """Benchmark timing result."""

    mean_time: float
    std_time: float
    num_windows: int
    accesses: list[Access]


@dataclass
class ParallelBenchmarkResult:
    """Parallel benchmark timing result (all locations computed at once)."""

    total_mean_time: float  # Time for entire batch
    total_std_time: float
    per_location_mean_time: float  # total_mean_time / num_locations
    per_location_std_time: float
    num_locations: int
    total_windows: int
    accesses: list[Access]


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


# =============================================================================
# Skyfield Implementation
# =============================================================================


def compute_skyfield_accesses(
    tle: TLE, target: Target, window_open: datetime, window_close: datetime
) -> list[Access]:
    """Compute access windows using Skyfield."""
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


# =============================================================================
# Brahe Python Implementation
# =============================================================================


def compute_brahe_python_accesses(
    tle: TLE, target: Target, window_open: datetime, window_close: datetime
) -> list[Access]:
    """Compute access windows using Brahe Python bindings."""
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


def compute_brahe_python_parallel(
    tle: TLE,
    targets: list[Target],
    window_open: datetime,
    window_close: datetime,
) -> list[Access]:
    """Compute access windows for ALL locations in a single call (parallel)."""
    # Create propagator from TLE
    propagator = bh.SGPPropagator.from_tle(tle.line1, tle.line2, 60.0).with_name(
        tle.sat_name
    )

    # Create all locations
    locations = [
        bh.PointLocation(t.longitude, t.latitude, t.altitude).with_name(
            t.name or f"Loc_{i}"
        )
        for i, t in enumerate(targets)
    ]

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

    # Compute access windows for all locations at once
    windows = bh.location_accesses(
        locations, propagator, epoch_start, epoch_end, constraint
    )

    # Convert to Access objects
    accesses = []
    for window in windows:
        aos_tuple = window.window_open.to_datetime()
        los_tuple = window.window_close.to_datetime()

        aos_dt = datetime(
            int(aos_tuple[0]),
            int(aos_tuple[1]),
            int(aos_tuple[2]),
            int(aos_tuple[3]),
            int(aos_tuple[4]),
            int(aos_tuple[5]),
            int(aos_tuple[6] / 1000),
            tzinfo=UTC,
        )
        los_dt = datetime(
            int(los_tuple[0]),
            int(los_tuple[1]),
            int(los_tuple[2]),
            int(los_tuple[3]),
            int(los_tuple[4]),
            int(los_tuple[5]),
            int(los_tuple[6] / 1000),
            tzinfo=UTC,
        )
        # Find matching target by location name
        loc_name = window.location_name
        target = next((t for t in targets if t.name == loc_name), None)
        accesses.append(Access(aos=aos_dt, los=los_dt, target=target))

    return accesses


# =============================================================================
# Brahe Rust Native Implementation
# =============================================================================


def compute_brahe_rust_accesses(
    tle: TLE,
    targets: list[Target],
    window_open: datetime,
    window_close: datetime,
    iterations: int = 5,
    parallel: bool = False,
) -> dict[str, BenchmarkResult] | tuple[dict[str, BenchmarkResult], dict]:
    """
    Compute access windows using Brahe Rust native via rust-script.

    Args:
        parallel: If True, compute all locations in a single call. Otherwise, compute serially.

    Returns:
        If parallel=False: dict mapping location name to BenchmarkResult
        If parallel=True: tuple of (dict mapping location name to BenchmarkResult, parallel_timing dict)
    """
    # Build JSON input
    input_data = {
        "locations": [
            {
                "lat": t.latitude,
                "lon": t.longitude,
                "alt": t.altitude,
                "name": t.name or f"Location_{i}",
            }
            for i, t in enumerate(targets)
        ],
        "tle_line1": tle.line1,
        "tle_line2": tle.line2,
        "window_open": window_open.isoformat(),
        "window_close": window_close.isoformat(),
        "min_elevation_deg": MIN_ELEVATION_DEG,
        "iterations": iterations,
        "parallel": parallel,
    }

    # Run rust-script
    try:
        result = subprocess.run(
            ["rust-script", str(RUST_SCRIPT_PATH)],
            input=json.dumps(input_data),
            capture_output=True,
            text=True,
            cwd=RUST_SCRIPT_PATH.parent,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        print("ERROR: Rust script timed out")
        return {}
    except FileNotFoundError:
        print("ERROR: rust-script not found. Install with: cargo install rust-script")
        return {}

    if result.returncode != 0:
        print(f"ERROR: Rust script failed:\n{result.stderr}")
        return {}

    # Parse JSON output
    try:
        output = json.loads(result.stdout)
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse Rust output: {e}")
        print(f"stdout: {result.stdout}")
        return {}

    # Convert to BenchmarkResult objects
    results: dict[str, BenchmarkResult] = {}
    for loc_result in output.get("results", []):
        location_name = loc_result["location_name"]

        # Find the matching target
        target = next(
            (t for t in targets if t.name == location_name),
            Target(
                latitude=loc_result["lat"],
                longitude=loc_result["lon"],
                altitude=0.0,
                name=location_name,
            ),
        )

        # Convert accesses
        accesses = []
        for acc in loc_result.get("accesses", []):
            aos = datetime.fromisoformat(acc["aos"].replace("Z", "+00:00"))
            los = datetime.fromisoformat(acc["los"].replace("Z", "+00:00"))
            accesses.append(Access(aos=aos, los=los, target=target))

        timing = loc_result["timing"]
        results[location_name] = BenchmarkResult(
            mean_time=timing["mean_seconds"],
            std_time=timing["std_seconds"],
            num_windows=timing["num_windows"],
            accesses=accesses,
        )

    # Return parallel timing if in parallel mode
    if parallel:
        parallel_timing = output.get("parallel_timing", {})
        return results, parallel_timing

    return results


# =============================================================================
# Validation and Benchmarking
# =============================================================================


def validate_accesses(
    accesses1: list[Access],
    accesses2: list[Access],
    name1: str,
    name2: str,
    tolerance_seconds: float = 0.5,
) -> ValidationResult:
    """Compare two sets of access windows."""
    warnings = []
    max_diff = 0.0

    if len(accesses1) != len(accesses2):
        warnings.append(
            f"Window count mismatch: {name1}={len(accesses1)}, {name2}={len(accesses2)}"
        )

    # Match windows by closest AOS time
    matched = set()
    for i, a1 in enumerate(accesses1):
        best_match_idx = None
        best_diff = float("inf")

        for j, a2 in enumerate(accesses2):
            if j in matched:
                continue
            aos_diff = abs((a1.aos - a2.aos).total_seconds())
            if aos_diff < best_diff:
                best_diff = aos_diff
                best_match_idx = j

        if best_match_idx is not None and best_diff < 60.0:
            a2 = accesses2[best_match_idx]
            aos_diff = abs((a1.aos - a2.aos).total_seconds())
            los_diff = abs((a1.los - a2.los).total_seconds())
            max_diff = max(max_diff, aos_diff, los_diff)

            if aos_diff > tolerance_seconds:
                warnings.append(f"AOS mismatch window {i + 1}: {aos_diff:.3f}s")
            if los_diff > tolerance_seconds:
                warnings.append(f"LOS mismatch window {i + 1}: {los_diff:.3f}s")

            matched.add(best_match_idx)

    passed = len(warnings) == 0
    return ValidationResult(passed=passed, warnings=warnings, max_diff=max_diff)


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
    accesses = result if result is not None else []

    return BenchmarkResult(
        mean_time=float(np.mean(times_arr)),
        std_time=float(np.std(times_arr)),
        num_windows=len(accesses),
        accesses=accesses,
    )


# =============================================================================
# Visualization
# =============================================================================


def create_comparison_chart(
    locations: list[Target],
    skyfield_results: list[BenchmarkResult],
    brahe_py_results: list[BenchmarkResult],
    brahe_rust_results: list[BenchmarkResult],
    output_file: str,
    plot_style: str = "bar",
    brahe_py_parallel: ParallelBenchmarkResult | None = None,
    brahe_rust_parallel: ParallelBenchmarkResult | None = None,
) -> None:
    """Create a plotly chart comparing execution times.

    Args:
        plot_style: Either "bar" for grouped bar chart or "scatter" for scatter plot with points
        brahe_py_parallel: Optional parallel Python benchmark result
        brahe_rust_parallel: Optional parallel Rust benchmark result
    """
    location_names = [loc.name or f"Loc {i + 1}" for i, loc in enumerate(locations)]

    # Define colors for each implementation
    colors = {
        "skyfield": "rgb(26, 118, 255)",  # Blue
        "brahe_py": "rgb(255, 127, 14)",  # Orange
        "brahe_rust": "rgb(44, 160, 44)",  # Green
        "brahe_py_parallel": "rgb(255, 187, 120)",  # Light orange
        "brahe_rust_parallel": "rgb(152, 223, 138)",  # Light green
    }

    fig = go.Figure()

    if plot_style == "scatter":
        # Scatter plot with points - better for many locations
        fig.add_trace(
            go.Scatter(
                name="Skyfield",
                x=location_names,
                y=[r.mean_time * 1000 for r in skyfield_results],
                mode="markers",
                marker=dict(color=colors["skyfield"], size=8),
                error_y=dict(
                    type="data",
                    array=[r.std_time * 1000 for r in skyfield_results],
                    visible=True,
                    color=colors["skyfield"],
                    thickness=1.5,
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                name="Brahe-Python (serial)",
                x=location_names,
                y=[r.mean_time * 1000 for r in brahe_py_results],
                mode="markers",
                marker=dict(color=colors["brahe_py"], size=8),
                error_y=dict(
                    type="data",
                    array=[r.std_time * 1000 for r in brahe_py_results],
                    visible=True,
                    color=colors["brahe_py"],
                    thickness=1.5,
                ),
            )
        )

        fig.add_trace(
            go.Scatter(
                name="Brahe-Rust (serial)",
                x=location_names,
                y=[r.mean_time * 1000 for r in brahe_rust_results],
                mode="markers",
                marker=dict(color=colors["brahe_rust"], size=8),
                error_y=dict(
                    type="data",
                    array=[r.std_time * 1000 for r in brahe_rust_results],
                    visible=True,
                    color=colors["brahe_rust"],
                    thickness=1.5,
                ),
            )
        )

        # Add parallel results as horizontal lines (same value for all locations)
        if brahe_py_parallel is not None:
            fig.add_trace(
                go.Scatter(
                    name="Brahe-Python (parallel)",
                    x=location_names,
                    y=[brahe_py_parallel.per_location_mean_time * 1000]
                    * len(locations),
                    mode="lines",
                    line=dict(color=colors["brahe_py_parallel"], width=2, dash="dash"),
                )
            )

        if brahe_rust_parallel is not None:
            fig.add_trace(
                go.Scatter(
                    name="Brahe-Rust (parallel)",
                    x=location_names,
                    y=[brahe_rust_parallel.per_location_mean_time * 1000]
                    * len(locations),
                    mode="lines",
                    line=dict(
                        color=colors["brahe_rust_parallel"], width=2, dash="dash"
                    ),
                )
            )
    else:
        # Bar chart - default, better for fewer locations
        fig.add_trace(
            go.Bar(
                name="Skyfield",
                x=location_names,
                y=[r.mean_time * 1000 for r in skyfield_results],
                error_y=dict(
                    type="data",
                    array=[r.std_time * 1000 for r in skyfield_results],
                    visible=True,
                ),
                marker_color=colors["skyfield"],
            )
        )

        fig.add_trace(
            go.Bar(
                name="Brahe-Python (serial)",
                x=location_names,
                y=[r.mean_time * 1000 for r in brahe_py_results],
                error_y=dict(
                    type="data",
                    array=[r.std_time * 1000 for r in brahe_py_results],
                    visible=True,
                ),
                marker_color=colors["brahe_py"],
            )
        )

        fig.add_trace(
            go.Bar(
                name="Brahe-Rust (serial)",
                x=location_names,
                y=[r.mean_time * 1000 for r in brahe_rust_results],
                error_y=dict(
                    type="data",
                    array=[r.std_time * 1000 for r in brahe_rust_results],
                    visible=True,
                ),
                marker_color=colors["brahe_rust"],
            )
        )

        # Add parallel results as bars (aggregate for all locations)
        if brahe_py_parallel is not None:
            fig.add_trace(
                go.Bar(
                    name="Brahe-Python (parallel)",
                    x=["Parallel (per-loc avg)"],
                    y=[brahe_py_parallel.per_location_mean_time * 1000],
                    error_y=dict(
                        type="data",
                        array=[brahe_py_parallel.per_location_std_time * 1000],
                        visible=True,
                    ),
                    marker_color=colors["brahe_py_parallel"],
                )
            )

        if brahe_rust_parallel is not None:
            fig.add_trace(
                go.Bar(
                    name="Brahe-Rust (parallel)",
                    x=["Parallel (per-loc avg)"],
                    y=[brahe_rust_parallel.per_location_mean_time * 1000],
                    error_y=dict(
                        type="data",
                        array=[brahe_rust_parallel.per_location_std_time * 1000],
                        visible=True,
                    ),
                    marker_color=colors["brahe_rust_parallel"],
                )
            )

        fig.update_layout(barmode="group")

    title = "Access Computation Performance Comparison"
    if brahe_py_parallel or brahe_rust_parallel:
        title = "Access Computation Performance: Serial vs Parallel"

    fig.update_layout(
        title=title,
        xaxis_title="Location",
        yaxis_title="Execution Time (ms)",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(255, 255, 255, 0.8)"),
        template="plotly_white",
    )

    fig.write_html(output_file)
    print(f"\nChart saved to: {output_file}")


def export_accesses_to_csv(
    all_accesses: list[tuple[str, Target, list[Access]]],
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

        for library, target, accesses in all_accesses:
            for access in accesses:
                duration = (access.los - access.aos).total_seconds()
                writer.writerow(
                    [
                        library,
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


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Five-way benchmark: Skyfield vs Brahe (serial/parallel, Python/Rust)"
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
        default=0.5,
        help="Validation tolerance in seconds (default: 0.5)",
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
        default="benchmark_five_way.html",
        help="Output HTML file for plot (default: benchmark_five_way.html)",
    )
    parser.add_argument(
        "--plot-style",
        type=str,
        choices=["bar", "scatter"],
        default="bar",
        help="Plot style: 'bar' for grouped bars, 'scatter' for points (default: bar)",
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
    print("=" * 70)
    print("Five-Way Access Computation Benchmark")
    print("Skyfield vs Brahe (Python/Rust, Serial/Parallel)")
    print("=" * 70)
    print(
        f"Locations: {args.n_locations} | Seed: {seed_str} | "
        f"Window: {WINDOW_DAYS} days | Min Elevation: {MIN_ELEVATION_DEG}°"
    )
    print()

    # First, run the Rust benchmark (serial) for all locations
    print("Running Brahe-Rust (serial) benchmark...")
    rust_results = compute_brahe_rust_accesses(
        tle,
        locations,
        window_open,
        window_close,
        iterations=args.iterations,
        parallel=False,
    )
    print(f"  Completed: {len(rust_results)} locations benchmarked")
    print()

    # Run Rust benchmark (parallel) - all locations at once
    print("Running Brahe-Rust (parallel) benchmark...")
    rust_parallel_results, rust_parallel_timing = compute_brahe_rust_accesses(
        tle,
        locations,
        window_open,
        window_close,
        iterations=args.iterations,
        parallel=True,
    )
    brahe_rust_parallel = ParallelBenchmarkResult(
        total_mean_time=rust_parallel_timing.get("total_mean_seconds", 0),
        total_std_time=rust_parallel_timing.get("total_std_seconds", 0),
        per_location_mean_time=rust_parallel_timing.get("per_location_mean_seconds", 0),
        per_location_std_time=rust_parallel_timing.get("per_location_std_seconds", 0),
        num_locations=rust_parallel_timing.get("num_locations", len(locations)),
        total_windows=rust_parallel_timing.get("total_windows", 0),
        accesses=[
            acc for res in rust_parallel_results.values() for acc in res.accesses
        ],
    )
    print(
        f"  Completed: {brahe_rust_parallel.total_windows} windows in "
        f"{brahe_rust_parallel.total_mean_time * 1000:.2f}ms total "
        f"({brahe_rust_parallel.per_location_mean_time * 1000:.2f}ms per location)"
    )
    print()

    # Run Python benchmark (parallel) - all locations at once
    print("Running Brahe-Python (parallel) benchmark...")
    parallel_py_times = []
    parallel_py_accesses: list[Access] = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        parallel_py_accesses = compute_brahe_python_parallel(
            tle, locations, window_open, window_close
        )
        elapsed = time.perf_counter() - start
        parallel_py_times.append(elapsed)

    parallel_py_mean = np.mean(parallel_py_times)
    parallel_py_std = np.std(parallel_py_times)
    brahe_py_parallel = ParallelBenchmarkResult(
        total_mean_time=parallel_py_mean,
        total_std_time=parallel_py_std,
        per_location_mean_time=parallel_py_mean / len(locations),
        per_location_std_time=parallel_py_std / len(locations),
        num_locations=len(locations),
        total_windows=len(parallel_py_accesses),
        accesses=parallel_py_accesses,
    )
    print(
        f"  Completed: {brahe_py_parallel.total_windows} windows in "
        f"{brahe_py_parallel.total_mean_time * 1000:.2f}ms total "
        f"({brahe_py_parallel.per_location_mean_time * 1000:.2f}ms per location)"
    )
    print()

    # Results storage
    skyfield_results: list[BenchmarkResult] = []
    brahe_py_results: list[BenchmarkResult] = []
    brahe_rust_results: list[BenchmarkResult] = []
    all_accesses: list[tuple[str, Target, list[Access]]] = []
    all_passed = 0
    all_warned = 0

    # Run serial benchmarks for each location
    print("Running serial benchmarks per location...")
    print()
    for i, location in enumerate(locations):
        print(
            f"Location {i + 1}: {location.name} ({location.latitude:.2f}, {location.longitude:.2f})"
        )

        # Benchmark Skyfield
        skyfield_bench = benchmark_function(
            compute_skyfield_accesses,
            (tle, location, window_open, window_close),
            iterations=args.iterations,
        )
        skyfield_results.append(skyfield_bench)

        # Benchmark Brahe Python (serial)
        brahe_py_bench = benchmark_function(
            compute_brahe_python_accesses,
            (tle, location, window_open, window_close),
            iterations=args.iterations,
        )
        brahe_py_results.append(brahe_py_bench)

        # Get Rust results from pre-computed batch
        rust_bench = rust_results.get(
            location.name,
            BenchmarkResult(mean_time=0, std_time=0, num_windows=0, accesses=[]),
        )
        brahe_rust_results.append(rust_bench)

        # Store accesses for CSV export
        all_accesses.append(("skyfield", location, skyfield_bench.accesses))
        all_accesses.append(("brahe-python-serial", location, brahe_py_bench.accesses))
        all_accesses.append(("brahe-rust-serial", location, rust_bench.accesses))

        # Print results
        print(
            f"  Skyfield:           {skyfield_bench.num_windows:3d} windows, "
            f"{skyfield_bench.mean_time * 1000:7.2f}ms ± {skyfield_bench.std_time * 1000:.2f}ms"
        )
        print(
            f"  Brahe-Py (serial):  {brahe_py_bench.num_windows:3d} windows, "
            f"{brahe_py_bench.mean_time * 1000:7.2f}ms ± {brahe_py_bench.std_time * 1000:.2f}ms"
        )
        print(
            f"  Brahe-Rs (serial):  {rust_bench.num_windows:3d} windows, "
            f"{rust_bench.mean_time * 1000:7.2f}ms ± {rust_bench.std_time * 1000:.2f}ms"
        )

        # Validate all three against each other
        val_sf_py = validate_accesses(
            skyfield_bench.accesses,
            brahe_py_bench.accesses,
            "Skyfield",
            "Brahe-Python",
            args.tolerance,
        )
        val_sf_rs = validate_accesses(
            skyfield_bench.accesses,
            rust_bench.accesses,
            "Skyfield",
            "Brahe-Rust",
            args.tolerance,
        )

        all_warnings = val_sf_py.warnings + val_sf_rs.warnings
        max_diff = max(val_sf_py.max_diff, val_sf_rs.max_diff)

        if len(all_warnings) == 0:
            print(f"  Validation: PASS (max diff: {max_diff:.3f}s)")
            all_passed += 1
        else:
            for w in all_warnings[:3]:  # Limit warnings shown
                print(f"  WARNING: {w}")
            if len(all_warnings) > 3:
                print(f"  ... and {len(all_warnings) - 3} more warnings")
            print(
                f"  Validation: WARN ({len(all_warnings)} issues, max diff: {max_diff:.3f}s)"
            )
            all_warned += 1

        print()

    # Summary
    print("=" * 70)
    print("Summary")
    print("=" * 70)

    total_sf = sum(r.num_windows for r in skyfield_results)
    total_py = sum(r.num_windows for r in brahe_py_results)
    total_rs = sum(r.num_windows for r in brahe_rust_results)
    avg_sf = np.mean([r.mean_time for r in skyfield_results])
    avg_py = np.mean([r.mean_time for r in brahe_py_results])
    avg_rs = (
        np.mean([r.mean_time for r in brahe_rust_results]) if brahe_rust_results else 0
    )
    avg_py_par = brahe_py_parallel.per_location_mean_time
    avg_rs_par = brahe_rust_parallel.per_location_mean_time

    print(
        f"Total windows: Skyfield={total_sf}, Brahe-Python={total_py}, Brahe-Rust={total_rs}"
    )
    print(f"Validation: {all_passed}/{args.n_locations} passed, {all_warned} warnings")
    print()

    # Performance comparison table
    def format_comparison(time1: float, time2: float, is_baseline: bool = False) -> str:
        """Format a comparison as 'Nx faster/slower' or 'baseline'."""
        if is_baseline:
            return "baseline"
        if time1 <= 0:
            return "N/A"
        ratio = time2 / time1
        if ratio >= 1:
            return f"{ratio:.1f}x faster"
        else:
            return f"{1 / ratio:.1f}x slower"

    print("Performance Comparison (per-location average):")
    print()
    print(
        "| Implementation         | Avg Time  | vs Skyfield    | vs Brahe-Py-Serial |"
    )
    print(
        "|------------------------|-----------|----------------|---------------------|"
    )

    # Skyfield row
    sf_vs_sf = "baseline"
    sf_vs_py = format_comparison(avg_sf, avg_py)
    print(
        f"| Skyfield               | {avg_sf * 1000:6.2f}ms  | {sf_vs_sf:<14} | {sf_vs_py:<19} |"
    )

    # Brahe-Rust (serial) row
    rs_vs_sf = format_comparison(avg_rs, avg_sf)
    rs_vs_py = format_comparison(avg_rs, avg_py)
    print(
        f"| Brahe-Rust (serial)    | {avg_rs * 1000:6.2f}ms  | {rs_vs_sf:<14} | {rs_vs_py:<19} |"
    )

    # Brahe-Rust (parallel) row
    rs_par_vs_sf = format_comparison(avg_rs_par, avg_sf)
    rs_par_vs_py = format_comparison(avg_rs_par, avg_py)
    print(
        f"| Brahe-Rust (parallel)  | {avg_rs_par * 1000:6.2f}ms  | {rs_par_vs_sf:<14} | {rs_par_vs_py:<19} |"
    )

    # Brahe-Python (serial) row
    py_vs_sf = format_comparison(avg_py, avg_sf)
    py_vs_py = "baseline"
    print(
        f"| Brahe-Python (serial)  | {avg_py * 1000:6.2f}ms  | {py_vs_sf:<14} | {py_vs_py:<19} |"
    )

    # Brahe-Python (parallel) row
    py_par_vs_sf = format_comparison(avg_py_par, avg_sf)
    py_par_vs_py = format_comparison(avg_py_par, avg_py)
    print(
        f"| Brahe-Python (parallel)| {avg_py_par * 1000:6.2f}ms  | {py_par_vs_sf:<14} | {py_par_vs_py:<19} |"
    )

    # Generate chart
    if not args.no_plot:
        create_comparison_chart(
            locations,
            skyfield_results,
            brahe_py_results,
            brahe_rust_results,
            args.output,
            plot_style=args.plot_style,
            brahe_py_parallel=brahe_py_parallel,
            brahe_rust_parallel=brahe_rust_parallel,
        )

    # Export CSV if requested
    if args.csv:
        export_accesses_to_csv(all_accesses, args.csv)


if __name__ == "__main__":
    main()
