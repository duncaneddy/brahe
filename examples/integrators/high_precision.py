# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
High-precision orbit propagation using RKN1210 integrator.

This example demonstrates RKN1210's capability for very tight tolerances
on a highly elliptical orbit.
"""

import brahe as bh
import numpy as np

# Initialize EOP data
bh.initialize_eop()

# Define HEO orbit (Molniya-type)
a = 26554e3  # Semi-major axis (m)
e = 0.74  # Eccentricity
i = 63.4  # Inclination

# Convert to Cartesian state
oe = np.array([a, e, i, 0.0, 0.0, 0.0])
state0 = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)

# Orbital period
period = bh.orbital_period(a)


# Two-body dynamics
def dynamics(t, state):
    mu = bh.GM_EARTH
    r = state[0:3]
    v = state[3:6]
    r_norm = np.linalg.norm(r)
    a = -mu / r_norm**3 * r
    return np.concatenate([v, a])


print("High-Precision HEO Orbit Propagation")
print(f"Semi-major axis: {a / 1e3:.1f} km")
print(f"Eccentricity: {e}")
print(f"Period: {period / 3600:.2f} hours\n")

# Create RKN1210 integrator with very tight tolerances
abs_tol = 1e-14
rel_tol = 1e-13
config = bh.IntegratorConfig.adaptive(abs_tol=abs_tol, rel_tol=rel_tol)
integrator = bh.RKN1210Integrator(6, dynamics, config=config)

print(f"Using RKN1210 with tol={abs_tol:.0e}")
print("Propagating for one orbital period...\n")

# Propagate for one orbit
t = 0.0
state = state0.copy()
dt = 60.0
steps = 0
total_error = 0.0

while t < period:
    result = integrator.step(t, state, min(dt, period - t))

    t += result.dt_used
    state = result.state
    dt = result.dt_next
    steps += 1
    total_error += result.error_estimate

    # Print at apogee and perigee
    r_norm = np.linalg.norm(state[0:3])
    if steps % 10 == 1:
        print(
            f"t={t / 3600:6.2f}h  r={r_norm / 1e3:8.1f}km  dt={result.dt_used:6.1f}s  err={result.error_estimate:.2e}"
        )

print("\nPropagation complete!")
print(f"Total steps: {steps}")
print(f"Average step: {period / steps:.1f} s")
print(f"Cumulative error estimate: {total_error:.2e}")

# Verify orbit closure (should return close to initial state)
final_oe = bh.state_eci_to_koe(state, bh.AngleFormat.DEGREES)
initial_oe = bh.state_eci_to_koe(state0, bh.AngleFormat.DEGREES)

print("\nOrbit element errors after one period:")
print(f"  Semi-major axis: {abs(final_oe[0] - initial_oe[0]):.3e} m")
print(f"  Eccentricity:    {abs(final_oe[1] - initial_oe[1]):.3e}")

# Expected Output:
# High-Precision HEO Orbit Propagation
# Semi-major axis: 26554.0 km
# Eccentricity: 0.74
# Period: 11.96 hours

# Using RKN1210 with tol=1e-14
# Propagating for one orbital period...

# t=  0.02h  r=  6915.2km  dt=  60.0s  err=1.14e-03
# t=  0.61h  r= 14336.8km  dt= 308.4s  err=2.37e-02
# t=  2.54h  r= 34813.8km  dt= 900.0s  err=1.52e-03
# t=  5.04h  r= 45404.2km  dt= 900.0s  err=1.65e-03
# t=  7.54h  r= 44004.3km  dt= 900.0s  err=0.00e+00
# t= 10.04h  r= 29862.5km  dt= 900.0s  err=6.33e-03
# t= 11.66h  r=  9739.4km  dt= 291.4s  err=2.59e-01

# Propagation complete!
# Total steps: 67
# Average step: 642.7 s
# Cumulative error estimate: 5.85e+00

# Orbit element errors after one period:
#   Semi-major axis: 3.725e-08 m
#   Eccentricity:    4.441e-16
