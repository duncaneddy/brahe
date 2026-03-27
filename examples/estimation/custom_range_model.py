# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Define a custom measurement model in Python and use it with the EKF.

Shows how to subclass MeasurementModel to implement a range (distance)
measurement from a ground station, then mix it with a built-in model.
"""

import numpy as np
import brahe as bh

bh.initialize_eop()


# Define a custom range measurement model
class RangeModel(bh.MeasurementModel):
    """Measures distance from a ground station to the satellite."""

    def __init__(self, station_eci, sigma):
        super().__init__()
        self.station_eci = np.array(station_eci)
        self.sigma = sigma

    def predict(self, epoch, state):
        pos = state[:3]
        return np.array([np.linalg.norm(pos - self.station_eci)])

    def noise_covariance(self):
        return np.array([[self.sigma**2]])

    def measurement_dim(self):
        return 1

    def name(self):
        return "Range"


# Set up orbit and truth propagator
epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
r = bh.R_EARTH + 500e3
v = (bh.GM_EARTH / r) ** 0.5
true_state = np.array([r, 0.0, 0.0, 0.0, v, 0.0])

truth_prop = bh.NumericalOrbitPropagator(
    epoch,
    true_state,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.two_body(),
)

# Ground station (approximately on the equator)
station = np.array([bh.R_EARTH, 0.0, 0.0])

# Create EKF with both a built-in model and our custom range model
position_model = bh.InertialPositionMeasurementModel(10.0)
range_model = RangeModel(station, 100.0)

initial_state = true_state.copy()
initial_state[0] += 500.0  # 500m position offset
p0 = np.diag([1e6, 1e6, 1e6, 1e2, 1e2, 1e2])

ekf = bh.ExtendedKalmanFilter(
    epoch,
    initial_state,
    p0,
    measurement_models=[position_model, range_model],
    propagation_config=bh.NumericalPropagationConfig.default(),
    force_config=bh.ForceModelConfig.two_body(),
)

# Alternate between position and range observations
dt = 60.0
for i in range(1, 21):
    obs_epoch = epoch + dt * i
    truth_prop.propagate_to(obs_epoch)
    truth_st = truth_prop.current_state()

    if i % 2 == 0:
        # Position observation (model_index=0)
        obs = bh.Observation(obs_epoch, truth_st[:3], model_index=0)
    else:
        # Range observation (model_index=1)
        true_range = np.linalg.norm(truth_st[:3] - station)
        obs = bh.Observation(obs_epoch, np.array([true_range]), model_index=1)

    record = ekf.process_observation(obs)
    print(
        f"  {record.measurement_name:20s} prefit residual norm: "
        f"{np.linalg.norm(record.prefit_residual):.3f}"
    )

# Summary
truth_prop.propagate_to(ekf.current_epoch())
pos_error = np.linalg.norm(ekf.current_state()[:3] - truth_prop.current_state()[:3])
print(f"\nFinal position error: {pos_error:.2f} m")
print(
    f"Records: {len(ekf.records())} "
    f"(InertialPosition: {sum(1 for r in ekf.records() if r.measurement_name == 'InertialPosition')}, "
    f"Range: {sum(1 for r in ekf.records() if r.measurement_name == 'Range')})"
)
