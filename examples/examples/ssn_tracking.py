#!/usr/bin/env python
# /// script
# dependencies = ["brahe", "plotly", "numpy"]
# ///

"""
SSN Radar Tracking: Extended and Unscented Kalman Filters

This example simulates Space Surveillance Network (SSN) radar tracking of a
LEO object and estimates its orbit sequentially with an Extended Kalman
Filter and an Unscented Kalman Filter. It loads the Vallado SSN sensor
dataset, builds az/el/range sensors, visualizes the sensor network's ground
coverage against a simulated LEO ground track, simulates radar measurements
during passes over a 6-hour tracking arc, processes them sequentially with
both filters while propagating through the gaps between passes, and
compares each filter's true position error against its formal 3-sigma
uncertainty.

A batch-estimation companion, ssn_tracking_bls.py, reproduces the same
scenario and solves for the orbit with Batch Least Squares instead; it is
excluded from the automated example tests because BLS re-propagates the
full arc at every iteration and takes on the order of three minutes to run.
"""

# --8<-- [start:all]
# --8<-- [start:preamble]
import os
import pathlib
import sys
import time

import brahe as bh
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

bh.initialize_eop()

# Configuration
MEAS_INTERVAL = 15.0  # seconds between measurements during a pass
DURATION = 6 * 3600.0  # tracking duration (seconds)
# The run numbers quoted in the docs depend on this seed AND on the rand
# crate's StdRng stream; a rand version bump can shift the exact figures while
# the example's tolerance-based asserts still pass.
SEED = 42
GAP_SPLIT = 600.0  # start a new arc when consecutive obs are > 10 min apart
PROPAGATE_STEP = 60.0  # gap-propagation step (seconds)
# --8<-- [end:preamble]

# Configuration for output files
SCRIPT_NAME = pathlib.Path(__file__).stem
OUTDIR = pathlib.Path(os.getenv("BRAHE_FIGURE_OUTPUT_DIR", "./docs/figures/"))
os.makedirs(OUTDIR, exist_ok=True)

# --8<-- [start:load_sensors]
# Load the Vallado SSN sensor sites and build az/el/range sensors. Sites
# without azel_range calibration (optical trackers, sites missing noise
# values) are skipped.
sites = bh.datasets.ssn_sensors.load()
sensors = bh.SimpleSSNSensor.from_locations(sites, seed=SEED)
print(f"Loaded {len(sites)} SSN sites, {len(sensors)} az/el/range sensors")

print("\n" + "=" * 100)
print("SSN Sensor Network")
print("=" * 100)
print(
    f"{'Name':<28}{'System':<12}{'El Min':>8}{'Range Max':>12}"
    f"{'Az Noise':>10}{'El Noise':>10}{'Range Noise':>13}"
)
print("-" * 100)
for s in sensors:
    props = s.location.properties
    system = props.get("system", "?")
    az_noise = props.get("az_noise_deg", float("nan"))
    el_noise = props.get("el_noise_deg", float("nan"))
    range_noise = props.get("range_noise_m", float("nan"))
    range_max_str = (
        f"{s.range_max / 1e3:>9.0f} km" if s.range_max is not None else "  unlimited"
    )
    print(
        f"{s.name:<28}{system:<12}{s.el_min:>7.1f}°{range_max_str:>12}"
        f"{az_noise:>9.4f}°{el_noise:>9.4f}°{range_noise:>11.1f} m"
    )
print("=" * 100)
# --8<-- [end:load_sensors]

# Truth orbit: LEO at 700 km, 72 degree inclination
epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
oe = np.array([bh.R_EARTH + 700e3, 0.001, 72.0, 30.0, 0.0, 0.0])
true_state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Hermite-cubic interpolation is required so the trajectory (stored at the
# propagator's ~60 s adaptive-step cadence) can be sampled accurately at the
# much finer measurement cadence below; linear interpolation of a curved
# orbit between 60 s-spaced points is off by kilometers at intermediate
# epochs.
truth_config = bh.NumericalPropagationConfig.default().with_interpolation_method(
    bh.InterpolationMethod.HERMITE_CUBIC
)
truth_prop = bh.NumericalOrbitPropagator(
    epoch,
    true_state,
    truth_config,
    bh.ForceModelConfig.two_body(),
)
epoch_end = epoch + DURATION
truth_prop.propagate_to(epoch_end)
truth_traj = truth_prop.trajectory

# --8<-- [start:visualize_network]
fig_network = bh.plot_groundtrack(
    trajectories=[
        {"trajectory": truth_prop.trajectory, "color": "red", "line_width": 2}
    ],
    ground_stations=[
        {"stations": [s.location for s in sensors], "color": "blue", "alpha": 0.15}
    ],
    gs_cone_altitude=700e3,
    gs_min_elevation=5.0,
    basemap="natural_earth",
    backend="plotly",
)
# --8<-- [end:visualize_network]

# --8<-- [start:simulate_measurements]
# Find passes and simulate measurements only inside them. Sensor measurement
# buckets for the figure below are filled as each pass is simulated, since
# per-sensor plotting order doesn't depend on the global time order that the
# sequential filters need; only `observations` has to be globally sorted.
PALETTE = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
    "#aec7e8",
    "#ffbb78",
    "#98df8a",
    "#ff9896",
    "#c5b0d5",
]
observations = []
passes = []  # (sensor_name, window, n_obs)
sensor_meas = {
    i: {"t": [], "az": [], "el": [], "range_km": []} for i in range(len(sensors))
}
for i, sensor in enumerate(sensors):
    constraint = bh.ElevationConstraint(min_elevation_deg=max(sensor.el_min, 1.0))
    windows = bh.location_accesses(
        sensor.location, truth_prop, epoch, epoch_end, constraint
    )
    for w in windows:
        obs = sensor.simulate_observations(
            truth_traj, w.window_open, w.window_close, MEAS_INTERVAL, i
        )
        observations.extend(obs)
        if obs:
            passes.append((sensor.name, w, len(obs)))
        bucket = sensor_meas[i]
        for o in obs:
            bucket["t"].append((o.epoch - epoch) / 60.0)
            bucket["az"].append(o.measurement[0])
            bucket["el"].append(o.measurement[1])
            bucket["range_km"].append(o.measurement[2] / 1e3)

observations.sort(key=lambda o: o.epoch)
passes.sort(key=lambda p: p[1].window_open)
print(f"\nSimulated {len(observations)} measurements over {len(passes)} passes")

print("\n" + "=" * 100)
print("Pass Table")
print("=" * 100)
print(f"{'Site':<28}{'Start':<22}{'End':<22}{'Duration (min)':>16}{'Obs':>8}")
print("-" * 100)
for sensor_name, w, n_obs in passes:
    start_str = str(w.window_open).split(".")[0]
    end_str = str(w.window_close).split(".")[0]
    duration_min = w.duration / 60.0
    print(
        f"{sensor_name:<28}{start_str:<22}{end_str:<22}{duration_min:>14.1f} m{n_obs:>8}"
    )
print("=" * 100)

fig_measurements = make_subplots(
    rows=3,
    cols=1,
    shared_xaxes=True,
    subplot_titles=("Azimuth (deg)", "Elevation (deg)", "Range (km)"),
    vertical_spacing=0.08,
)
for i, sensor in enumerate(sensors):
    bucket = sensor_meas[i]
    if not bucket["t"]:
        continue
    color = PALETTE[i % len(PALETTE)]
    fig_measurements.add_trace(
        go.Scatter(
            x=bucket["t"],
            y=bucket["az"],
            mode="markers",
            name=sensor.name,
            legendgroup=sensor.name,
            marker=dict(color=color, size=5),
        ),
        row=1,
        col=1,
    )
    fig_measurements.add_trace(
        go.Scatter(
            x=bucket["t"],
            y=bucket["el"],
            mode="markers",
            name=sensor.name,
            legendgroup=sensor.name,
            showlegend=False,
            marker=dict(color=color, size=5),
        ),
        row=2,
        col=1,
    )
    fig_measurements.add_trace(
        go.Scatter(
            x=bucket["t"],
            y=bucket["range_km"],
            mode="markers",
            name=sensor.name,
            legendgroup=sensor.name,
            showlegend=False,
            marker=dict(color=color, size=5),
        ),
        row=3,
        col=1,
    )

fig_measurements.update_xaxes(title_text="Time (minutes)", row=3, col=1)
fig_measurements.update_yaxes(title_text="Azimuth (deg)", row=1, col=1)
fig_measurements.update_yaxes(title_text="Elevation (deg)", row=2, col=1)
fig_measurements.update_yaxes(title_text="Range (km)", row=3, col=1)
fig_measurements.update_layout(
    title="Simulated SSN Radar Measurements (6-hour Tracking Arc)",
    height=800,
    margin=dict(l=60, r=40, t=80, b=60),
)
# --8<-- [end:simulate_measurements]

# --8<-- [start:run_filters]


def run_sequential_filter(filt, obs_list, t0, gap_split, propagate_step):
    """Process observations in epoch order, propagating through gaps.

    Advances the filter through gaps between passes in fixed steps so that
    covariance growth during the gap is recorded, then processes each
    observation in turn.
    """
    prev_epoch = t0
    for observation in obs_list:
        if observation.epoch - prev_epoch > gap_split:
            t = prev_epoch + propagate_step
            while t < observation.epoch:
                filt.propagate_to(t)
                t = t + propagate_step
        filt.process_observation(observation)
        prev_epoch = observation.epoch
    return filt


def extract_error_series(filt, truth_trajectory, t0):
    """Extract (time_min, position_error_m, position_1sigma_m) from filter records."""
    t_min, errors, sigmas = [], [], []
    for rec in filt.records():
        truth_state = truth_trajectory.interpolate(rec.epoch)
        err = np.linalg.norm(rec.state_updated[:3] - truth_state[:3])
        sigma = np.sqrt(np.sum(np.diag(rec.covariance_updated)[:3]))
        t_min.append((rec.epoch - t0) / 60.0)
        errors.append(err)
        sigmas.append(sigma)
    return np.array(t_min), np.array(errors), np.array(sigmas)


# Shared initial guess: perturbed by 1 km in x-position, 1 m/s in y-velocity
initial_state = np.array(true_state)
initial_state[0] += 1000.0
initial_state[4] += 1.0
p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
models = [s.measurement_model() for s in sensors]

# Extended Kalman Filter
start_time = time.time()
ekf = bh.ExtendedKalmanFilter(
    epoch,
    initial_state,
    p0,
    measurement_models=models,
    propagation_config=bh.NumericalPropagationConfig.default(),
    force_config=bh.ForceModelConfig.two_body(),
)
run_sequential_filter(ekf, observations, epoch, GAP_SPLIT, PROPAGATE_STEP)
print(
    f"\nEKF processed {len(observations)} observations in {time.time() - start_time:.1f} s"
)

# Unscented Kalman Filter
start_time = time.time()
ukf = bh.UnscentedKalmanFilter(
    epoch,
    initial_state,
    p0,
    propagation_config=bh.NumericalPropagationConfig.default(),
    force_config=bh.ForceModelConfig.two_body(),
    measurement_models=models,
)
run_sequential_filter(ukf, observations, epoch, GAP_SPLIT, PROPAGATE_STEP)
print(
    f"UKF processed {len(observations)} observations in {time.time() - start_time:.1f} s"
)

ekf_t, ekf_err, ekf_sigma = extract_error_series(ekf, truth_traj, epoch)
ukf_t, ukf_err, ukf_sigma = extract_error_series(ukf, truth_traj, epoch)
# --8<-- [end:run_filters]

# --8<-- [start:analyze_results]
fig_filters = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=(
        "True Position Error vs. 3-sigma Envelope",
        "Formal 3-sigma Position Uncertainty (log scale)",
    ),
    vertical_spacing=0.1,
)

for t_min, err, sigma, name, color in (
    (ekf_t, ekf_err, ekf_sigma, "EKF", "steelblue"),
    (ukf_t, ukf_err, ukf_sigma, "UKF", "coral"),
):
    fig_filters.add_trace(
        go.Scatter(
            x=t_min,
            y=err,
            mode="lines",
            name=f"{name} true error",
            line=dict(color=color, width=2),
        ),
        row=1,
        col=1,
    )
    fig_filters.add_trace(
        go.Scatter(
            x=t_min,
            y=3 * sigma,
            mode="lines",
            name=f"{name} 3-sigma",
            line=dict(color=color, width=1.5, dash="dash"),
        ),
        row=1,
        col=1,
    )
    fig_filters.add_trace(
        go.Scatter(
            x=t_min,
            y=3 * sigma,
            mode="lines",
            name=f"{name} 3-sigma",
            legendgroup=f"{name} 3-sigma",
            showlegend=False,
            line=dict(color=color, width=2),
        ),
        row=2,
        col=1,
    )

# Shade pass windows on both rows
for _, w, _ in passes:
    t0_min = (w.window_open - epoch) / 60.0
    t1_min = (w.window_close - epoch) / 60.0
    for row in (1, 2):
        fig_filters.add_vrect(
            x0=t0_min,
            x1=t1_min,
            fillcolor="gray",
            opacity=0.15,
            line_width=0,
            row=row,
            col=1,
        )

fig_filters.update_xaxes(title_text="Time (minutes)", row=2, col=1)
fig_filters.update_yaxes(title_text="Position error (m)", row=1, col=1)
fig_filters.update_yaxes(title_text="3-sigma (m)", type="log", row=2, col=1)
fig_filters.update_layout(
    title="EKF / UKF Filter Consistency",
    height=800,
    margin=dict(l=60, r=40, t=80, b=60),
)

print("\n" + "=" * 80)
print("Filter Summary")
print("=" * 80)
print(
    f"EKF final position error: {ekf_err[-1]:.1f} m  (3-sigma = {3 * ekf_sigma[-1]:.1f} m)"
)
print(
    f"UKF final position error: {ukf_err[-1]:.1f} m  (3-sigma = {3 * ukf_sigma[-1]:.1f} m)"
)
print("=" * 80)

assert ekf_err[-1] < 500.0, "EKF should converge to a small position error"
assert ukf_err[-1] < 500.0, "UKF should converge to a small position error"
print("\nExample validated successfully!")
# --8<-- [end:analyze_results]
# --8<-- [end:all]

# ============================================================================
# Plot Output Section (for documentation generation)
# ============================================================================

# Add plots directory to path for importing brahe_theme
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "plots"))
from brahe_theme import save_themed_html  # noqa: E402

# Save the sensor network ground track as themed HTML
light_path, dark_path = save_themed_html(fig_network, OUTDIR / f"{SCRIPT_NAME}_network")
print(f"\n✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")

# Save the measurement time series as themed HTML
light_path, dark_path = save_themed_html(
    fig_measurements, OUTDIR / f"{SCRIPT_NAME}_measurements"
)
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")

# Save the filter comparison figure as themed HTML
light_path, dark_path = save_themed_html(fig_filters, OUTDIR / f"{SCRIPT_NAME}_filters")
print(f"✓ Generated {light_path}")
print(f"✓ Generated {dark_path}")
