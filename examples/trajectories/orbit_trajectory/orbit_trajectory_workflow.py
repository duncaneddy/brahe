# /// script
# dependencies = ["brahe"]
# ///
"""
Complete workflow: propagation, frame conversion, and analysis
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# 1. Define orbit and create propagator
a = bh.R_EARTH + 500e3  # 500 km altitude
e = 0.001  # Nearly circular
i = 97.8  # Sun-synchronous
raan = 15.0
argp = 30.0
M = 0.0
oe = np.array([a, e, i, raan, argp, M])

epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
propagator = bh.KeplerianPropagator.from_keplerian(
    epoch, oe, bh.AngleFormat.DEGREES, 60.0
)

# 2. Propagate for one orbit period
period = bh.orbital_period(a)
end_epoch = epoch + period
propagator.propagate_to(end_epoch)

# 3. Get trajectory in ECI Cartesian
traj_eci = propagator.trajectory
print(f"Propagated {len(traj_eci)} states over {traj_eci.timespan() / 60:.1f} minutes")

# 4. Convert to ECEF to analyze ground track
traj_ecef = traj_eci.to_ecef()
print("\nGround track in ECEF frame:")
for i, (epoch, state_ecef) in enumerate(traj_ecef):
    if i % 10 == 0:  # Sample every 10 states
        # Convert ECEF to geodetic for latitude/longitude
        lat, lon, alt = bh.position_ecef_to_geodetic(
            state_ecef[0:3], bh.AngleFormat.DEGREES
        )
        print(f"  {epoch}: Lat={lat:6.2f}°, Lon={lon:7.2f}°, Alt={alt / 1e3:6.2f} km")

# 5. Convert to Keplerian to analyze orbital evolution
traj_kep = traj_eci.to_keplerian(bh.AngleFormat.DEGREES)
first_oe = traj_kep.state_at_idx(0)
last_oe = traj_kep.state_at_idx(len(traj_kep) - 1)

print("\nOrbital element evolution:")
print(f"  Semi-major axis: {first_oe[0] / 1e3:.2f} km → {last_oe[0] / 1e3:.2f} km")
print(f"  Eccentricity: {first_oe[1]:.6f} → {last_oe[1]:.6f}")
print(f"  Inclination: {first_oe[2]:.2f}° → {last_oe[2]:.2f}°")
print(f"  True anomaly: {first_oe[5]:.2f}° → {last_oe[5]:.2f}°")
