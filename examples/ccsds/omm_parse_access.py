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
# Format version: 3.0
# Originator:     NOAA/USA
# Creation date:  2007-065T16:00:00.000 UTC

# Metadata
print(f"\nObject name:          {omm.object_name}")
print(f"Object ID:            {omm.object_id}")
print(f"Center name:          {omm.center_name}")
print(f"Ref frame:            {omm.ref_frame}")
print(f"Time system:          {omm.time_system}")
print(f"Mean element theory:  {omm.mean_element_theory}")
# Object name:          GOES 9
# Object ID:            1995-025A
# Center name:          EARTH
# Ref frame:            TEME
# Time system:          UTC
# Mean element theory:  SGP/SGP4

# Mean orbital elements (CCSDS/TLE-native units)
print(f"\nEpoch:               {omm.epoch}")
print(f"Mean motion:         {omm.mean_motion} rev/day")
print(f"Eccentricity:        {omm.eccentricity}")
print(f"Inclination:         {omm.inclination} deg")
print(f"RAAN:                {omm.ra_of_asc_node} deg")
print(f"Arg of pericenter:   {omm.arg_of_pericenter} deg")
print(f"Mean anomaly:        {omm.mean_anomaly} deg")
print(f"GM:                  {omm.gm:.4e} m³/s²")
# Epoch:               2007-064T10:34:41.426 UTC
# Mean motion:         1.00273272 rev/day
# Eccentricity:        0.0005013
# Inclination:         3.0539 deg
# RAAN:                81.7939 deg
# Arg of pericenter:   249.2363 deg
# Mean anomaly:        150.1602 deg
# GM:                  3.9860e+14 m³/s²

# TLE parameters
print(f"\nNORAD catalog ID:    {omm.norad_cat_id}")
print(f"Classification:      {omm.classification_type}")
print(f"Ephemeris type:      {omm.ephemeris_type}")
print(f"Element set no:      {omm.element_set_no}")
print(f"Rev at epoch:        {omm.rev_at_epoch}")
print(f"BSTAR:               {omm.bstar}")
print(f"Mean motion dot:     {omm.mean_motion_dot} rev/day²")
print(f"Mean motion ddot:    {omm.mean_motion_ddot} rev/day³")
# NORAD catalog ID:    23581
# Classification:      U
# Ephemeris type:      0
# Element set no:      925
# Rev at epoch:        4316
# BSTAR:               0.0001
# Mean motion dot:     -1.13e-06 rev/day²
# Mean motion ddot:    0.0 rev/day³

# Serialization
d = omm.to_dict()
print(f"\nDict keys: {list(d.keys())}")
# Dict keys: ['header', 'metadata', 'mean_elements', 'tle_parameters']
