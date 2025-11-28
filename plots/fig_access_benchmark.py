# /// script
# dependencies = ["brahe", "plotly", "numpy", "skyfield>=1.49"]
# FLAGS = ["IGNORE"]
# ///
"""
Access computation benchmark comparing Brahe vs Skyfield performance.

Generates light and dark themed benchmark comparison charts showing:
1. Skyfield (Python) - baseline
2. Brahe Python bindings (serial)
3. Brahe Python bindings (parallel)
4. Brahe Rust native (serial)
5. Brahe Rust native (parallel)

This script is marked IGNORE because it takes several minutes to run the full
benchmark with 100 locations. Run manually with:
    uv run python make.py make-plots --ignore
"""

import json
import os
import pathlib
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import numpy as np
import plotly.graph_objects as go
from skyfield.api import EarthSatellite, load, wgs84
from skyfield.timelib import Time
from skyfield.toposlib import GeographicPosition

import brahe as bh

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from brahe_theme import get_theme_colors, save_themed_html

# Configuration
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
RUST_SCRIPT_PATH = (
    pathlib.Path(__file__).parent.parent / "scripts" / "benchmark_access_rust.rs"
)

# Ensure output directory exists
os.makedirs(OUTDIR, exist_ok=True)

# Benchmark configuration
N_LOCATIONS = 100
SEED = 42
ITERATIONS = 5
MIN_ELEVATION_DEG = 5.0
WINDOW_DAYS = 2

# Default ISS TLE
DEFAULT_TLE_LINE1 = (
    "1 25544U 98067A   25321.32008588  .00017447  00000-0  31567-3 0  9995"
)
DEFAULT_TLE_LINE2 = (
    "2 25544  51.6331 268.2969 0004176  80.2379 279.9081 15.49753355538919"
)

# Skyfield time scale
time_scale = load.timescale()


@dataclass
class Target:
    """Ground station target location."""

    latitude: float
    longitude: float
    altitude: float
    name: str | None = None


@dataclass
class BenchmarkResult:
    """Benchmark timing result."""

    mean_time: float
    std_time: float
    num_windows: int


@dataclass
class ParallelBenchmarkResult:
    """Parallel benchmark timing result."""

    per_location_mean_time: float
    per_location_std_time: float
    num_locations: int
    total_windows: int


def generate_random_locations(n: int, seed: int | None = None) -> list[Target]:
    """Generate N random ground station locations worldwide."""
    if seed is not None:
        random.seed(seed)

    locations = []
    for i in range(n):
        lat = random.uniform(-70, 70)
        lon = random.uniform(-180, 180)
        alt = random.uniform(0, 500)
        name = f"Location_{i + 1}"
        locations.append(Target(latitude=lat, longitude=lon, altitude=alt, name=name))

    return locations


def compute_skyfield_accesses(
    target: Target, window_open: datetime, window_close: datetime
) -> int:
    """Compute access windows using Skyfield and return count."""
    t0: Time = time_scale.from_datetime(window_open.replace(tzinfo=UTC))
    t1: Time = time_scale.from_datetime(window_close.replace(tzinfo=UTC))
    topo: GeographicPosition = wgs84.latlon(
        target.latitude, target.longitude, target.altitude
    )
    satellite = EarthSatellite(DEFAULT_TLE_LINE1, DEFAULT_TLE_LINE2, "ISS", time_scale)
    times, events = satellite.find_events(
        topo, t0, t1, altitude_degrees=MIN_ELEVATION_DEG
    )

    # Count windows (AOS=0, LOS=2)
    aos_count = sum(1 for e in events if e == 0)
    return aos_count


def compute_brahe_python_accesses(
    target: Target, window_open: datetime, window_close: datetime
) -> int:
    """Compute access windows using Brahe Python bindings and return count."""
    propagator = bh.SGPPropagator.from_tle(DEFAULT_TLE_LINE1, DEFAULT_TLE_LINE2, 60.0)
    location = bh.PointLocation(target.longitude, target.latitude, target.altitude)

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

    constraint = bh.ElevationConstraint(min_elevation_deg=MIN_ELEVATION_DEG)
    windows = bh.location_accesses(
        location, propagator, epoch_start, epoch_end, constraint
    )
    return len(windows)


def compute_brahe_python_parallel(
    targets: list[Target], window_open: datetime, window_close: datetime
) -> int:
    """Compute access windows for all locations in a single call."""
    propagator = bh.SGPPropagator.from_tle(DEFAULT_TLE_LINE1, DEFAULT_TLE_LINE2, 60.0)
    locations = [
        bh.PointLocation(t.longitude, t.latitude, t.altitude).with_name(
            t.name or f"Loc_{i}"
        )
        for i, t in enumerate(targets)
    ]

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

    constraint = bh.ElevationConstraint(min_elevation_deg=MIN_ELEVATION_DEG)
    windows = bh.location_accesses(
        locations, propagator, epoch_start, epoch_end, constraint
    )
    return len(windows)


def compute_brahe_rust_accesses(
    targets: list[Target],
    window_open: datetime,
    window_close: datetime,
    parallel: bool = False,
) -> dict[str, BenchmarkResult] | tuple[dict[str, BenchmarkResult], dict]:
    """Compute access windows using Brahe Rust native via rust-script."""
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
        "tle_line1": DEFAULT_TLE_LINE1,
        "tle_line2": DEFAULT_TLE_LINE2,
        "window_open": window_open.isoformat(),
        "window_close": window_close.isoformat(),
        "min_elevation_deg": MIN_ELEVATION_DEG,
        "iterations": ITERATIONS,
        "parallel": parallel,
    }

    result = subprocess.run(
        ["rust-script", str(RUST_SCRIPT_PATH)],
        input=json.dumps(input_data),
        capture_output=True,
        text=True,
        cwd=RUST_SCRIPT_PATH.parent,
        timeout=600,
    )

    if result.returncode != 0:
        print(f"ERROR: Rust script failed:\n{result.stderr}")
        return {} if not parallel else ({}, {})

    output = json.loads(result.stdout)

    results: dict[str, BenchmarkResult] = {}
    for loc_result in output.get("results", []):
        location_name = loc_result["location_name"]
        timing = loc_result["timing"]
        results[location_name] = BenchmarkResult(
            mean_time=timing["mean_seconds"],
            std_time=timing["std_seconds"],
            num_windows=timing["num_windows"],
        )

    if parallel:
        parallel_timing = output.get("parallel_timing", {})
        return results, parallel_timing

    return results


def benchmark_function(func, args, iterations: int = 5) -> BenchmarkResult:
    """Benchmark a function and return mean/std execution time."""
    times = []
    result = None

    for _ in range(iterations):
        start = time.perf_counter()
        result = func(*args)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

    times_arr = np.array(times)
    return BenchmarkResult(
        mean_time=float(np.mean(times_arr)),
        std_time=float(np.std(times_arr)),
        num_windows=result if isinstance(result, int) else 0,
    )


def run_benchmarks():
    """Run all benchmarks and return results."""
    print(f"Running access benchmark with {N_LOCATIONS} locations...")

    # Initialize brahe EOP data
    bh.initialize_eop()

    # Generate locations
    locations = generate_random_locations(N_LOCATIONS, SEED)

    # Time window
    window_open = datetime(2025, 11, 18, tzinfo=UTC)
    window_close = datetime(2025, 11, 18 + WINDOW_DAYS, tzinfo=UTC)

    # Run Rust benchmarks (serial)
    print("  Running Brahe-Rust (serial)...")
    rust_results = compute_brahe_rust_accesses(
        locations, window_open, window_close, parallel=False
    )

    # Run Rust benchmarks (parallel)
    print("  Running Brahe-Rust (parallel)...")
    rust_parallel_results, rust_parallel_timing = compute_brahe_rust_accesses(
        locations, window_open, window_close, parallel=True
    )
    brahe_rust_parallel = ParallelBenchmarkResult(
        per_location_mean_time=rust_parallel_timing.get("per_location_mean_seconds", 0),
        per_location_std_time=rust_parallel_timing.get("per_location_std_seconds", 0),
        num_locations=rust_parallel_timing.get("num_locations", len(locations)),
        total_windows=rust_parallel_timing.get("total_windows", 0),
    )

    # Run Python parallel benchmark
    print("  Running Brahe-Python (parallel)...")
    parallel_py_times = []
    parallel_py_windows = 0
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        parallel_py_windows = compute_brahe_python_parallel(
            locations, window_open, window_close
        )
        elapsed = time.perf_counter() - start
        parallel_py_times.append(elapsed)

    parallel_py_mean = np.mean(parallel_py_times)
    parallel_py_std = np.std(parallel_py_times)
    brahe_py_parallel = ParallelBenchmarkResult(
        per_location_mean_time=parallel_py_mean / len(locations),
        per_location_std_time=parallel_py_std / len(locations),
        num_locations=len(locations),
        total_windows=parallel_py_windows,
    )

    # Run serial benchmarks per location
    print("  Running serial benchmarks per location...")
    skyfield_results: list[BenchmarkResult] = []
    brahe_py_results: list[BenchmarkResult] = []
    brahe_rust_results: list[BenchmarkResult] = []

    for i, location in enumerate(locations):
        if (i + 1) % 20 == 0:
            print(f"    Location {i + 1}/{N_LOCATIONS}...")

        # Skyfield
        skyfield_bench = benchmark_function(
            compute_skyfield_accesses,
            (location, window_open, window_close),
            iterations=ITERATIONS,
        )
        skyfield_results.append(skyfield_bench)

        # Brahe Python (serial)
        brahe_py_bench = benchmark_function(
            compute_brahe_python_accesses,
            (location, window_open, window_close),
            iterations=ITERATIONS,
        )
        brahe_py_results.append(brahe_py_bench)

        # Brahe Rust (serial) - from pre-computed batch
        rust_bench = rust_results.get(
            location.name,
            BenchmarkResult(mean_time=0, std_time=0, num_windows=0),
        )
        brahe_rust_results.append(rust_bench)

    return {
        "locations": locations,
        "skyfield": skyfield_results,
        "brahe_py_serial": brahe_py_results,
        "brahe_rust_serial": brahe_rust_results,
        "brahe_py_parallel": brahe_py_parallel,
        "brahe_rust_parallel": brahe_rust_parallel,
    }


def create_figure(theme: str, benchmark_data: dict) -> go.Figure:
    """Create figure with theme-specific colors."""
    colors = get_theme_colors(theme)

    locations = benchmark_data["locations"]
    skyfield_results = benchmark_data["skyfield"]
    brahe_py_results = benchmark_data["brahe_py_serial"]
    brahe_rust_results = benchmark_data["brahe_rust_serial"]
    brahe_py_parallel = benchmark_data["brahe_py_parallel"]
    brahe_rust_parallel = benchmark_data["brahe_rust_parallel"]

    location_names = [loc.name or f"Loc {i + 1}" for i, loc in enumerate(locations)]

    # Define colors for each implementation
    impl_colors = {
        "skyfield": colors["primary"],
        "brahe_py": colors["secondary"],
        "brahe_rust": colors["accent"],
        "brahe_py_parallel": "#ffcc80" if theme == "dark" else "#ffaa44",
        "brahe_rust_parallel": "#a5d6a7" if theme == "dark" else "#66bb6a",
    }

    fig = go.Figure()

    # Scatter plot with points for serial implementations
    fig.add_trace(
        go.Scatter(
            name="Skyfield",
            x=location_names,
            y=[r.mean_time * 1000 for r in skyfield_results],
            mode="markers",
            marker=dict(color=impl_colors["skyfield"], size=6, opacity=0.7),
            error_y=dict(
                type="data",
                array=[r.std_time * 1000 for r in skyfield_results],
                visible=True,
                color=impl_colors["skyfield"],
                thickness=1,
            ),
            hovertemplate="Skyfield<br>%{x}<br>%{y:.2f}ms<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            name="Brahe-Python (serial)",
            x=location_names,
            y=[r.mean_time * 1000 for r in brahe_py_results],
            mode="markers",
            marker=dict(color=impl_colors["brahe_py"], size=6, opacity=0.7),
            error_y=dict(
                type="data",
                array=[r.std_time * 1000 for r in brahe_py_results],
                visible=True,
                color=impl_colors["brahe_py"],
                thickness=1,
            ),
            hovertemplate="Brahe-Python (serial)<br>%{x}<br>%{y:.2f}ms<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            name="Brahe-Rust (serial)",
            x=location_names,
            y=[r.mean_time * 1000 for r in brahe_rust_results],
            mode="markers",
            marker=dict(color=impl_colors["brahe_rust"], size=6, opacity=0.7),
            error_y=dict(
                type="data",
                array=[r.std_time * 1000 for r in brahe_rust_results],
                visible=True,
                color=impl_colors["brahe_rust"],
                thickness=1,
            ),
            hovertemplate="Brahe-Rust (serial)<br>%{x}<br>%{y:.2f}ms<extra></extra>",
        )
    )

    # Add parallel results as horizontal lines
    fig.add_trace(
        go.Scatter(
            name="Brahe-Python (parallel)",
            x=location_names,
            y=[brahe_py_parallel.per_location_mean_time * 1000] * len(locations),
            mode="lines",
            line=dict(color=impl_colors["brahe_py_parallel"], width=2, dash="dash"),
            hovertemplate=f"Brahe-Python (parallel)<br>{brahe_py_parallel.per_location_mean_time * 1000:.2f}ms per location<extra></extra>",
        )
    )

    fig.add_trace(
        go.Scatter(
            name="Brahe-Rust (parallel)",
            x=location_names,
            y=[brahe_rust_parallel.per_location_mean_time * 1000] * len(locations),
            mode="lines",
            line=dict(color=impl_colors["brahe_rust_parallel"], width=2, dash="dash"),
            hovertemplate=f"Brahe-Rust (parallel)<br>{brahe_rust_parallel.per_location_mean_time * 1000:.2f}ms per location<extra></extra>",
        )
    )

    # Configure layout
    fig.update_layout(
        title="Access Computation Performance: Brahe vs Skyfield",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
        ),
        margin=dict(t=100),
    )

    fig.update_xaxes(
        title_text="Location",
        showticklabels=False,
        title_standoff=5,
    )

    fig.update_yaxes(
        title_text="Execution Time (ms)",
        type="log",
    )

    return fig


# Run benchmarks
print("=" * 60)
print("Access Computation Benchmark")
print("=" * 60)

benchmark_data = run_benchmarks()

# Calculate and print summary statistics
avg_sf = np.mean([r.mean_time for r in benchmark_data["skyfield"]])
avg_py = np.mean([r.mean_time for r in benchmark_data["brahe_py_serial"]])
avg_rs = np.mean([r.mean_time for r in benchmark_data["brahe_rust_serial"]])
avg_py_par = benchmark_data["brahe_py_parallel"].per_location_mean_time
avg_rs_par = benchmark_data["brahe_rust_parallel"].per_location_mean_time

print()
print("Performance Summary (per-location average):")
print(f"  Skyfield:               {avg_sf * 1000:6.2f}ms")
print(
    f"  Brahe-Rust (serial):    {avg_rs * 1000:6.2f}ms ({avg_sf / avg_rs:.1f}x faster than Skyfield)"
)
print(
    f"  Brahe-Rust (parallel):  {avg_rs_par * 1000:6.2f}ms ({avg_sf / avg_rs_par:.1f}x faster than Skyfield)"
)
print(
    f"  Brahe-Python (serial):  {avg_py * 1000:6.2f}ms ({avg_py / avg_sf:.1f}x slower than Skyfield)"
)
print(
    f"  Brahe-Python (parallel):{avg_py_par * 1000:6.2f}ms ({avg_sf / avg_py_par:.1f}x vs Skyfield)"
)
print()


# Generate and save both themed versions
def create_fig_with_data(theme):
    return create_figure(theme, benchmark_data)


light_path, dark_path = save_themed_html(create_fig_with_data, OUTDIR / SCRIPT_NAME)
print(f"Generated {light_path}")
print(f"Generated {dark_path}")
