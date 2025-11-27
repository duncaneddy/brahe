# /// script
# dependencies = ["brahe"]
# ///
import brahe as bh
import numpy as np

# Initialize Earth Orientation Parameter data
bh.initialize_eop()

# Define orbital elements
a = bh.constants.R_EARTH + 700e3  # Semi-major axis in meters (700 km altitude)
e = 0.001  # Eccentricity
i = 98.7  # Inclination in radians
raan = 15.0  # Right Ascension of Ascending Node in radians
arg_periapsis = 30.0  # Argument of Periapsis in radians
mean_anomaly = 45.0  # Mean Anomaly

# Create a state vector from orbital elements
state_kep = np.array([a, e, i, raan, arg_periapsis, mean_anomaly])
print(
    f"Keplerian State: {a:.3f} m, {e:.6f}, {i:.3f} deg, {raan:.3f} deg, {arg_periapsis:.3f} deg, {mean_anomaly:.3f} deg"
)

# Convert Keplerian state to ECI coordinates
state_eci = bh.state_koe_to_eci(state_kep, bh.AngleFormat.DEGREES)
print(
    f"ECI State: {state_eci[0] / 1e3:.3f} km, {state_eci[1] / 1e3:.3f} km, {state_eci[2] / 1e3:.3f} km, {state_eci[3]:.3f} m/s, {state_eci[4]:.3f} m/s, {state_eci[5]:.3f} m/s"
)

# Define a time epoch
epoch = bh.Epoch(2024, 6, 1, 12, 0, 0.0, time_system=bh.TimeSystem.UTC)
print("Epoch:", epoch)

# Convert ECI coordinates to ECEF coordinates at the given epoch
state_ecef = bh.state_eci_to_ecef(epoch, state_eci)
print(
    f"ECEF State: {state_ecef[0] / 1e3:.3f} km, {state_ecef[1] / 1e3:.3f} km, {state_ecef[2] / 1e3:.3f} km, {state_ecef[3]:.3f} m/s, {state_ecef[4]:.3f} m/s, {state_ecef[5]:.3f} m/s"
)

# Convert back from ECEF to ECI coordinates
state_eci_2 = bh.state_ecef_to_eci(epoch, state_ecef)
print(
    f"Converted back ECI State: {state_eci_2[0] / 1e3:.3f} km, {state_eci_2[1] / 1e3:.3f} km, {state_eci_2[2] / 1e3:.3f} km, {state_eci_2[3]:.3f} m/s, {state_eci_2[4]:.3f} m/s, {state_eci_2[5]:.3f} m/s"
)

# Convert back from ECI to Keplerian elements
state_kep_2 = bh.state_eci_to_koe(state_eci_2, bh.AngleFormat.DEGREES)
print(
    f"Converted back Keplerian State: {state_kep_2[0]:.3f} m, {state_kep_2[1]:.6f}, {state_kep_2[2]:.3f} deg, {state_kep_2[3]:.3f} deg, {state_kep_2[4]:.3f} deg, {state_kep_2[5]:.3f} deg"
)
