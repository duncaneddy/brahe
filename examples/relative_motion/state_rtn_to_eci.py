# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Transform relative RTN state to absolute deputy ECI state
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

# Define chief satellite orbital elements
# LEO orbit: 700 km altitude, nearly circular, sun-synchronous inclination
oe_chief = np.array(
    [
        bh.R_EARTH + 700e3,  # Semi-major axis (m)
        0.001,  # Eccentricity
        97.8,  # Inclination (deg)
        15.0,  # Right ascension of ascending node (deg)
        30.0,  # Argument of perigee (deg)
        45.0,  # Mean anomaly (deg)
    ]
)

# Convert to Cartesian ECI state
x_chief = bh.state_koe_to_eci(oe_chief, bh.AngleFormat.DEGREES)

print("Chief ECI state:")
print(
    f"Position:  [{x_chief[0] / 1e3:.3f}, {x_chief[1] / 1e3:.3f}, {x_chief[2] / 1e3:.3f}] km"
)
print(
    f"Velocity:  [{x_chief[3] / 1e3:.6f}, {x_chief[4] / 1e3:.6f}, {x_chief[5] / 1e3:.6f}] km/s\n"
)
# Position:  [1999.015, -424.663, 6771.472] km
# Velocity:  [-6.939780, -2.131872, 1.920555] km/s

# Define relative state in RTN frame
# Deputy is 1 km radial, 500 m along-track, -300 m cross-track
# with small relative velocity
x_rel_rtn = np.array(
    [
        1000.0,  # Radial offset (m)
        500.0,  # Along-track offset (m)
        -300.0,  # Cross-track offset (m)
        0.1,  # Radial velocity (m/s)
        -0.05,  # Along-track velocity (m/s)
        0.02,  # Cross-track velocity (m/s)
    ]
)

print("Relative state in RTN frame:")
print(f"Radial (R):      {x_rel_rtn[0]:.3f} m")
print(f"Along-track (T): {x_rel_rtn[1]:.3f} m")
print(f"Cross-track (N): {x_rel_rtn[2]:.3f} m")
print(f"Velocity R:      {x_rel_rtn[3]:.3f} m/s")
print(f"Velocity T:      {x_rel_rtn[4]:.3f} m/s")
print(f"Velocity N:      {x_rel_rtn[5]:.3f} m/s\n")
# Radial (R):      1000.000 m
# Along-track (T): 500.000 m
# Cross-track (N): -300.000 m
# Velocity R:      0.100 m/s
# Velocity T:      -0.050 m/s
# Velocity N:      0.020 m/s

# Transform to absolute deputy ECI state
x_deputy = bh.state_rtn_to_eci(x_chief, x_rel_rtn)

print("Deputy ECI state:")
print(
    f"Position:  [{x_deputy[0] / 1e3:.3f}, {x_deputy[1] / 1e3:.3f}, {x_deputy[2] / 1e3:.3f}] km"
)
print(
    f"Velocity:  [{x_deputy[3] / 1e3:.6f}, {x_deputy[4] / 1e3:.6f}, {x_deputy[5] / 1e3:.6f}] km/s\n"
)
# Position:  [1998.759, -424.578, 6772.598] km
# Velocity:  [-6.940832, -2.132153, 1.920398] km/s

# Verify by transforming back to RTN
x_rel_rtn_verify = bh.state_eci_to_rtn(x_chief, x_deputy)

print("Round-trip verification (RTN -> ECI -> RTN):")
print(f"Original RTN:  [{x_rel_rtn[0]:.3f}, {x_rel_rtn[1]:.3f}, {x_rel_rtn[2]:.3f}] m")
print(
    f"Recovered RTN: [{x_rel_rtn_verify[0]:.3f}, {x_rel_rtn_verify[1]:.3f}, {x_rel_rtn_verify[2]:.3f}] m"
)
print(f"Difference:    {np.linalg.norm(x_rel_rtn - x_rel_rtn_verify):.9f} m")
# Original RTN:  [1000.000, 500.000, -300.000] m
# Recovered RTN: [1000.000, 500.000, -300.000] m
# Difference:    0.000000000 m
