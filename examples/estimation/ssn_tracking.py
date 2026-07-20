# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Simulate SSN radar tracking of a LEO object and estimate its state with an EKF.

Loads the Vallado SSN sensor dataset, finds passes with the access module,
simulates az/el/range measurements during passes, and processes them with an
Extended Kalman Filter, propagating through gaps between passes.
"""

import numpy as np
import brahe as bh

bh.initialize_eop()

# Configuration
MEAS_INTERVAL = 15.0  # seconds between measurements during a pass
DURATION = 2 * 3600.0  # tracking duration (seconds)
SEED = 42

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

# Build sensors from the Vallado SSN dataset (calibrated radar and optical
# sites; radar measures az/el/range, optical measures angles-only az/el)
sites = bh.datasets.ssn_sensors.load()
sensors = bh.SimpleSSNSensor.from_locations_calibrated(sites, seed=SEED)
print(f"Loaded {len(sites)} SSN sites, {len(sensors)} calibrated sensors")

# Find passes and simulate measurements only inside them
observations = []
passes = []
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
            passes.append((sensor.name, w))

observations.sort(key=lambda o: o.epoch)
print(f"Simulated {len(observations)} measurements over {len(passes)} passes")

# EKF from a perturbed initial state, using each sensor's matching model
initial_state = np.array(true_state)
initial_state[0] += 1000.0
initial_state[4] += 1.0
p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

ekf = bh.ExtendedKalmanFilter(
    epoch,
    initial_state,
    p0,
    measurement_models=[s.measurement_model() for s in sensors],
    propagation_config=bh.NumericalPropagationConfig.default(),
    force_config=bh.ForceModelConfig.two_body(),
)

# Process observations in order; propagate through gaps between passes
GAP_SPLIT = 600.0  # start a new arc when consecutive obs are > 10 min apart
prev_epoch = epoch
for obs in observations:
    if obs.epoch - prev_epoch > GAP_SPLIT:
        # advance through the gap in 60 s steps to record covariance growth
        t = prev_epoch + 60.0
        while t < obs.epoch:
            ekf.propagate_to(t)
            t = t + 60.0
    ekf.process_observation(obs)
    prev_epoch = obs.epoch

# Compare final estimate to truth
truth_final = truth_traj.interpolate(ekf.current_epoch())
err = np.linalg.norm(ekf.current_state()[:3] - truth_final[:3])
print(f"Final position error: {err:.1f} m")
sigma = np.sqrt(np.diag(ekf.current_covariance()))
print(f"Final position 1-sigma: [{sigma[0]:.1f}, {sigma[1]:.1f}, {sigma[2]:.1f}] m")

assert err < 500.0, "EKF should converge to a small position error"
print("Example validated successfully!")
