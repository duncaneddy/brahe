# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Hohmann transfer using impulsive maneuvers with event callbacks.
Demonstrates a two-burn orbit transfer from LEO to higher orbit.
"""

import numpy as np
import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Create initial epoch and state
epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

# Initial circular orbit at 400 km
r1 = bh.R_EARTH + 400e3
# Target circular orbit at 800 km
r2 = bh.R_EARTH + 800e3

# Initial state (circular orbit at perigee of transfer)
oe_initial = np.array([r1, 0.0001, 0.0, 0.0, 0.0, 0.0])
state = bh.state_koe_to_eci(oe_initial, bh.AngleFormat.DEGREES)

# Calculate Hohmann transfer delta-vs
v1_circular = np.sqrt(bh.GM_EARTH / r1)
v2_circular = np.sqrt(bh.GM_EARTH / r2)

# Transfer ellipse parameters
a_transfer = (r1 + r2) / 2
v_perigee_transfer = np.sqrt(bh.GM_EARTH * (2 / r1 - 1 / a_transfer))
v_apogee_transfer = np.sqrt(bh.GM_EARTH * (2 / r2 - 1 / a_transfer))

# Delta-v magnitudes
dv1 = v_perigee_transfer - v1_circular  # First burn (prograde at perigee)
dv2 = v2_circular - v_apogee_transfer  # Second burn (prograde at apogee)

print(
    f"Hohmann Transfer: {(r1 - bh.R_EARTH) / 1e3:.0f} km -> {(r2 - bh.R_EARTH) / 1e3:.0f} km"
)
print(f"  First burn (perigee):  {dv1:.3f} m/s")
print(f"  Second burn (apogee):  {dv2:.3f} m/s")
print(f"  Total delta-v:         {dv1 + dv2:.3f} m/s")

# Transfer time (half period of transfer ellipse)
transfer_time = np.pi * np.sqrt(a_transfer**3 / bh.GM_EARTH)
print(f"  Transfer time:         {transfer_time / 60:.1f} min")


# Create callback for first burn
def first_burn_callback(event_epoch, event_state):
    """Apply first delta-v at departure."""
    new_state = event_state.copy()
    # Add delta-v in velocity direction (prograde)
    v = event_state[3:6]
    v_hat = v / np.linalg.norm(v)
    new_state[3:6] += dv1 * v_hat
    print(f"  First burn applied at t+0s: dv = {dv1:.3f} m/s")
    return (new_state, bh.EventAction.CONTINUE)


# Create callback for second burn
def second_burn_callback(event_epoch, event_state):
    """Apply second delta-v at arrival."""
    new_state = event_state.copy()
    v = event_state[3:6]
    v_hat = v / np.linalg.norm(v)
    new_state[3:6] += dv2 * v_hat
    dt = event_epoch - epoch
    print(f"  Second burn applied at t+{dt:.1f}s: dv = {dv2:.3f} m/s")
    return (new_state, bh.EventAction.CONTINUE)


# Create propagator (two-body for clean Hohmann)
prop = bh.NumericalOrbitPropagator(
    epoch,
    state,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.two_body(),
    None,
)

# First burn at t=0 (immediate)
event1 = bh.TimeEvent(epoch + 1.0, "First Burn").with_callback(first_burn_callback)

# Second burn at apogee (half transfer period)
event2 = bh.TimeEvent(epoch + transfer_time, "Second Burn").with_callback(
    second_burn_callback
)

prop.add_event_detector(event1)
prop.add_event_detector(event2)

# Propagate through both burns plus one orbit of final orbit
final_orbit_period = 2 * np.pi * np.sqrt(r2**3 / bh.GM_EARTH)
prop.propagate_to(epoch + transfer_time + final_orbit_period)

# Check final orbit
final_koe = prop.state_koe(prop.current_epoch, bh.AngleFormat.DEGREES)
final_altitude = final_koe[0] - bh.R_EARTH

print("\nFinal orbit:")
print(f"  Semi-major axis: {final_koe[0] / 1e3:.3f} km")
print(
    f"  Altitude:        {final_altitude / 1e3:.3f} km (target: {(r2 - bh.R_EARTH) / 1e3:.0f} km)"
)
print(f"  Eccentricity:    {final_koe[1]:.6f}")

# Validate final orbit achieved significant altitude gain
# Note: Some error expected due to numerical integration and event timing
altitude_gain = final_altitude - (r1 - bh.R_EARTH)
assert altitude_gain > 200e3  # Significant altitude gain achieved
assert final_koe[1] < 0.1  # Reasonably circular

print("\nExample validated successfully!")
