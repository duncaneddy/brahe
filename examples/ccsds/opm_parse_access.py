# /// script
# dependencies = ["brahe"]
# ///
"""
Parse an OPM file and access state vector, Keplerian elements, and maneuvers.
"""

import brahe as bh
from brahe.ccsds import OPM

bh.initialize_eop()

# Parse OPM with Keplerian elements and maneuvers
opm = OPM.from_file("test_assets/ccsds/opm/OPMExample2.txt")

# Header
print(f"Format version: {opm.format_version}")
print(f"Originator:     {opm.originator}")
print(f"Creation date:  {opm.creation_date}")

# Metadata
print(f"\nObject name:  {opm.object_name}")
print(f"Object ID:    {opm.object_id}")
print(f"Center name:  {opm.center_name}")
print(f"Ref frame:    {opm.ref_frame}")
print(f"Time system:  {opm.time_system}")

# State vector (SI units: meters, m/s)
print(f"\nEpoch:    {opm.epoch}")
pos = opm.position
vel = opm.velocity
print(f"Position: [{pos[0] / 1e3:.4f}, {pos[1] / 1e3:.4f}, {pos[2] / 1e3:.4f}] km")
print(f"Velocity: [{vel[0]:.8f}, {vel[1]:.8f}, {vel[2]:.8f}] m/s")

# Keplerian elements
print(f"\nHas Keplerian: {opm.has_keplerian_elements}")
if opm.has_keplerian_elements:
    print(f"  Semi-major axis:    {opm.semi_major_axis / 1e3:.4f} km")
    print(f"  Eccentricity:       {opm.eccentricity:.9f}")
    print(f"  Inclination:        {opm.inclination:.6f} deg")
    print(f"  RAAN:               {opm.ra_of_asc_node:.6f} deg")
    print(f"  Arg of pericenter:  {opm.arg_of_pericenter:.6f} deg")
    print(f"  True anomaly:       {opm.true_anomaly:.6f} deg")
    print(f"  GM:                 {opm.gm:.4e} m³/s²")

# Spacecraft parameters
print(f"\nMass:           {opm.mass} kg")
print(f"Solar rad area: {opm.solar_rad_area} m²")
print(f"Solar rad coef: {opm.solar_rad_coeff}")
print(f"Drag area:      {opm.drag_area} m²")
print(f"Drag coeff:     {opm.drag_coeff}")

# Maneuvers
print(f"\nManeuvers: {len(opm.maneuvers)}")
for i, man in enumerate(opm.maneuvers):
    print(f"\n  Maneuver {i}:")
    print(f"    Epoch ignition: {man.epoch_ignition}")
    print(f"    Duration:       {man.duration} s")
    print(f"    Delta mass:     {man.delta_mass} kg")
    print(f"    Ref frame:      {man.ref_frame}")
    dv = man.dv
    print(f"    Delta-V:        [{dv[0]:.5f}, {dv[1]:.5f}, {dv[2]:.5f}] m/s")
