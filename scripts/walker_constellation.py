import pathlib
import math
import json

from brahe.constants import R_EARTH
from brahe.epoch import Epoch
from brahe.tle import tle_string_from_elements
from brahe.astro import sun_sync_inclination, mean_motion
from brahe.data_models import Spacecraft, SpacecraftModel

# Output Directory
OUTPUT_DIR = 'script_outputs/satellite_constellation'

# Time of TLEs
EPOCH = Epoch(2020, 1, 1)

# Satellite Orbit
ALTITUDE = 500
SMA = R_EARTH + ALTITUDE*1.0e3
ECCENTRICITY = 0.001
INCL = sun_sync_inclination(SMA, ECCENTRICITY, use_degrees=True)
INIT_RAAN = 0.0
ARGP = 0.0
INIT_MANM = 0.0

MEAN_MOTION = mean_motion(SMA, use_degrees=False)/(2*math.pi)*86400.0

# Walker constellation specification
# WALKER_SPEC = (36, 12, 1)
WALKER_SPEC = (3, 3, 1)

# Spacecraft Properties
SLEW_RATE = 1.0
SETTLING_TIME = 15.0

model = SpacecraftModel(
    slew_rate=SLEW_RATE,
    settling_time=SETTLING_TIME,
)

# Ensure Output Directory Exists
OUTPUT_DIR = pathlib.Path(OUTPUT_DIR)
if not pathlib.Path(OUTPUT_DIR).exists():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True) 

# Generate Spacecraft JSON
# Iterate over planes
spacecraft = []
from brahe.tle import TLE

# Number of satellites in planes
s = int(WALKER_SPEC[0]//WALKER_SPEC[1])
f = 360 / (s * WALKER_SPEC[1]) * WALKER_SPEC[2]

idx = 0
for p in range(0, WALKER_SPEC[1]):
    # Get Plane Offset
    raan_offset = 360.0 / WALKER_SPEC[1]
    RAAN = INIT_RAAN + raan_offset*p

    # Mean anomaly offsets in 
    mean_anm_offset = 360 / s

    if p > 0:
        INIT_MANM += f

    # Iterate over satellites in plane
    for slot in range(0, s):
        MANM = INIT_MANM + slot*mean_anm_offset

        # Create TLE
        idx += 1
        OE = [MEAN_MOTION, ECCENTRICITY, INCL, RAAN, ARGP, MANM, 0.0, 0.0, 0.0]
        line1, line2 = tle_string_from_elements(EPOCH, OE, norad_id=idx)

        spacecraft.append(
            Spacecraft(line1=line1, line2=line2, id=idx, name=f'Spacecraft {idx}', model=model)
        )


# Save to Output
json.dump([sc.dict() for sc in spacecraft], open(f'{OUTPUT_DIR}/spacecraft.json', 'w'), indent=4)