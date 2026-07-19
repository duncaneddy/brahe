#!/usr/bin/env python
# /// script
# dependencies = ["brahe", "numpy"]
# FLAGS = ["MANUAL"]
# ///

"""
SSN Radar Tracking: Batch Least Squares (manual companion)

This is the batch-estimation companion to ssn_tracking.py. It reproduces
the same 6-hour SSN tracking scenario -- the Vallado SSN sensor dataset,
the 700 km/72 degree LEO truth orbit, and the simulated az/el/range
measurements -- and estimates the orbit with Batch Least Squares (BLS)
instead of a sequential filter.

It carries `FLAGS = ["MANUAL"]` and is never run by the automated example
tests: BLS re-propagates the full 6-hour arc at every Gauss-Newton
iteration, which takes minutes locally and exceeds the example-runner
timeout on CI machines. Run this script directly to regenerate its
committed output (docs/outputs/examples/ssn_tracking_bls.py.txt), which
keeps the results available in the documentation without requiring a run.
"""

# --8<-- [start:all]
import time

import brahe as bh
import numpy as np

bh.initialize_eop()

# Configuration -- identical scenario to ssn_tracking.py.
MEAS_INTERVAL = 15.0  # seconds between measurements during a pass
DURATION = 6 * 3600.0  # tracking duration (seconds)
SEED = 42

# Load the Vallado SSN sensor sites and build az/el/range sensors. Sites
# without azel_range calibration (optical trackers, sites missing noise
# values) are skipped.
sites = bh.datasets.ssn_sensors.load()
sensors = bh.SimpleSSNSensor.from_locations_calibrated(sites, seed=SEED)
print(f"Loaded {len(sites)} SSN sites, {len(sensors)} az/el/range sensors")

# Truth orbit: LEO at 700 km, 72 degree inclination
epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
oe = np.array([bh.R_EARTH + 700e3, 0.001, 72.0, 30.0, 0.0, 0.0])
true_state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Hermite-cubic interpolation (the propagator default) lets the trajectory,
# stored at the ~60 s adaptive-step cadence, be sampled accurately at the much
# finer measurement cadence used for measurement simulation.
truth_config = bh.NumericalPropagationConfig.default()
truth_prop = bh.NumericalOrbitPropagator(
    epoch,
    true_state,
    truth_config,
    bh.ForceModelConfig.two_body(),
)
epoch_end = epoch + DURATION
truth_prop.propagate_to(epoch_end)
truth_traj = truth_prop.trajectory

# Find passes and simulate measurements only inside them
observations = []
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

observations.sort(key=lambda o: o.epoch)
print(f"Simulated {len(observations)} measurements")

# Shared initial guess: perturbed by 1 km in x-position, 1 m/s in y-velocity
initial_state = np.array(true_state)
initial_state[0] += 1000.0
initial_state[4] += 1.0
p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])
models = [s.measurement_model() for s in sensors]

# --8<-- [start:run_bls]
# Batch Least Squares over the full observation set. The default
# state-correction threshold (1e-8, mixed position/velocity units) is far
# tighter than the ~cm-level precision this measurement set supports, so a
# looser threshold is used to stop once corrections are physically
# insignificant rather than exhausting max_iterations.
start_time = time.time()
bls_config = bh.BLSConfig(
    state_correction_threshold=1e-3, cost_convergence_threshold=1e-6
)
bls = bh.BatchLeastSquares(
    epoch,
    initial_state,
    p0,
    propagation_config=bh.NumericalPropagationConfig.default(),
    force_config=bh.ForceModelConfig.two_body(),
    measurement_models=models,
    config=bls_config,
)
bls.solve(observations)
print(
    f"BLS solved {len(observations)} observations in {time.time() - start_time:.1f} s"
)

bls_err = np.linalg.norm(bls.current_state()[:3] - true_state[:3])
bls_sigma = np.sqrt(np.sum(np.diag(bls.formal_covariance())[:3]))

print(f"BLS position error: {bls_err:.1f} m  (3-sigma formal = {3 * bls_sigma:.1f} m)")
print(
    f"BLS converged: {bls.converged()}, iterations: {bls.iterations_completed()}, "
    f"final cost: {bls.final_cost():.6e}"
)
# --8<-- [end:run_bls]

assert bls_err < 500.0, "BLS should converge to a small position error"
print("\nExample validated successfully!")
# --8<-- [end:all]
