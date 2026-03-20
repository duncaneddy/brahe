# /// script
# dependencies = ["brahe"]
# ///
"""
Initialize an SGPPropagator from OMM mean elements — extract epoch, mean motion,
eccentricity, inclination, RAAN, argument of pericenter, mean anomaly, and TLE
parameters to create an SGP4 propagator.
"""

import brahe as bh
from brahe.ccsds import OMM

bh.initialize_eop()

# Parse OMM
omm = OMM.from_file("test_assets/ccsds/omm/OMMExample1.txt")
print(f"Object: {omm.object_name} ({omm.object_id})")
print(f"Theory: {omm.mean_element_theory}")
print(f"Epoch:  {omm.epoch}")

# Extract mean elements for SGP4
# The epoch string is needed in ISO format for from_omm_elements
d = omm.to_dict()
epoch_str = d["mean_elements"]["epoch"]

# Initialize SGP propagator from OMM elements
prop = bh.SGPPropagator.from_omm_elements(
    epoch=epoch_str,
    mean_motion=omm.mean_motion,
    eccentricity=omm.eccentricity,
    inclination=omm.inclination,
    raan=omm.ra_of_asc_node,
    arg_of_pericenter=omm.arg_of_pericenter,
    mean_anomaly=omm.mean_anomaly,
    norad_id=omm.norad_cat_id,
    object_name=omm.object_name,
    object_id=omm.object_id,
    classification=omm.classification_type,
    bstar=omm.bstar,
    mean_motion_dot=omm.mean_motion_dot,
    mean_motion_ddot=omm.mean_motion_ddot,
    ephemeris_type=omm.ephemeris_type,
    element_set_no=omm.element_set_no,
    rev_at_epoch=omm.rev_at_epoch,
)

print("\nSGP Propagator created:")
print(f"  NORAD ID: {prop.norad_id}")
print(f"  Name:     {prop.satellite_name}")
print(f"  Epoch:    {prop.epoch}")

# Propagate 1 day forward
target = prop.epoch + 86400.0
state = prop.state(target)
print(f"\nState after 1 day ({target}):")
print(
    f"  Position: [{state[0] / 1e3:.3f}, {state[1] / 1e3:.3f}, {state[2] / 1e3:.3f}] km"
)
print(f"  Velocity: [{state[3]:.3f}, {state[4]:.3f}, {state[5]:.3f}] m/s")

# Propagate to several epochs
print("\nState every 6 hours:")
for hours in range(0, 25, 6):
    t = prop.epoch + hours * 3600.0
    s = prop.state(t)
    r = (s[0] ** 2 + s[1] ** 2 + s[2] ** 2) ** 0.5
    print(f"  +{hours:2d}h: r={r / 1e3:.1f} km")
# State every 6 hours:
#   + 0h: r=... km
#   + 6h: r=... km
#   +12h: r=... km
#   +18h: r=... km
#   +24h: r=... km
