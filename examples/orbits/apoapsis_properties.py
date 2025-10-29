import brahe as bh

a = bh.R_EARTH + 500.0e3
e = 0.01

# Compute periapsis velocity
apoapsis_velocity = bh.apoapsis_velocity(a, e, bh.GM_EARTH)
print(f"Apoapsis velocity: {apoapsis_velocity:.3f}")
# Apoapsis velocity: 7536.859

# Compute as a apogee velocity
apogee_velocity = bh.apogee_velocity(a, e)
print(f"Apogee velocity:   {apogee_velocity:.3f}")
assert apoapsis_velocity == apogee_velocity
# Perigee velocity:   7536.859

# Compute apoapsis distance
apoapsis_distance = bh.apoapsis_distance(a, e)
print(f"Apoapsis distance: {apoapsis_distance:.3f}")
# Periapsis distance: 6946917.663
