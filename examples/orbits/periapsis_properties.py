import brahe as bh

a = bh.R_EARTH + 500.0e3
e = 0.01

# Compute periapsis velocity
periapsis_velocity = bh.periapsis_velocity(a, e, bh.GM_EARTH)
print(f"Periapsis velocity: {periapsis_velocity:.3f}")
# Periapsis velocity: 7689.119

# Compute as a perigee velocity
perigee_velocity = bh.perigee_velocity(a, e)
print(f"Perigee velocity:   {perigee_velocity:.3f}")
assert periapsis_velocity == perigee_velocity
# Perigee velocity:   7689.119

# Compute periapsis distance
periapsis_distance = bh.periapsis_distance(a, e)
print(f"Periapsis distance: {periapsis_distance:.3f}")
# Periapsis distance: 6809354.937
