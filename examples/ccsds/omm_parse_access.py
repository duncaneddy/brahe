# /// script
# dependencies = ["brahe"]
# ///
"""
Parse an OMM file and access mean elements, TLE parameters, and metadata.
"""

import brahe as bh
from brahe.ccsds import OMM

bh.initialize_eop()

# Parse OMM file
omm = OMM.from_file("test_assets/ccsds/omm/OMMExample1.txt")

# Header
print(f"Format version: {omm.format_version}")
print(f"Originator:     {omm.originator}")
print(f"Creation date:  {omm.creation_date}")

# Metadata
print(f"\nObject name:          {omm.object_name}")
print(f"Object ID:            {omm.object_id}")
print(f"Center name:          {omm.center_name}")
print(f"Ref frame:            {omm.ref_frame}")
print(f"Time system:          {omm.time_system}")
print(f"Mean element theory:  {omm.mean_element_theory}")

# Mean orbital elements (CCSDS/TLE-native units)
print(f"\nEpoch:               {omm.epoch}")
print(f"Mean motion:         {omm.mean_motion} rev/day")
print(f"Eccentricity:        {omm.eccentricity}")
print(f"Inclination:         {omm.inclination} deg")
print(f"RAAN:                {omm.ra_of_asc_node} deg")
print(f"Arg of pericenter:   {omm.arg_of_pericenter} deg")
print(f"Mean anomaly:        {omm.mean_anomaly} deg")
print(f"GM:                  {omm.gm:.4e} m³/s²")

# TLE parameters
print(f"\nNORAD catalog ID:    {omm.norad_cat_id}")
print(f"Classification:      {omm.classification_type}")
print(f"Ephemeris type:      {omm.ephemeris_type}")
print(f"Element set no:      {omm.element_set_no}")
print(f"Rev at epoch:        {omm.rev_at_epoch}")
print(f"BSTAR:               {omm.bstar}")
print(f"Mean motion dot:     {omm.mean_motion_dot} rev/day²")
print(f"Mean motion ddot:    {omm.mean_motion_ddot} rev/day³")

# Serialization
d = omm.to_dict()
print(f"\nDict keys: {list(d.keys())}")
