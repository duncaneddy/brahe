# /// script
# dependencies = ["brahe"]
# ///
"""
Read a Modified ITC ephemeris file and inspect its contents.

Parses a Starlink public ephemeris, prints the header, the record count,
the first state vector in SI units, and the diagonal of the first
covariance matrix.
"""

import numpy as np

import brahe as bh

PATH = "test_assets/starlink/MEME_100001_STARLINK-38128_2540142_Operational_1473385380_UNCLASSIFIED.txt"

itc = bh.ITC.from_file(PATH)
header = itc.header

print(f"Source file:      {itc.source_name}")
print(f"Created:          {header.created}")
print(f"Ephemeris start:  {header.ephemeris_start}")
print(f"Ephemeris stop:   {header.ephemeris_stop}")
print(f"Step size:        {header.step_size} s")
print(f"State frame:      {header.state_frame}")
print(f"Covariance frame: {header.covariance_frame}")
print(f"Records:          {len(itc)}")
print(f"Has covariance:   {itc.has_covariance}")

first = itc.states[0]
print(f"First epoch:      {first.epoch}")
print(f"Position [m]:     {np.array2string(first.position, precision=3)}")
print(f"Velocity [m/s]:   {np.array2string(first.velocity, precision=6)}")

sigma = np.sqrt(np.diag(itc.covariances[0]))
print(f"1-sigma RTN position [m]:   {np.array2string(sigma[:3], precision=3)}")
print(f"1-sigma RTN velocity [m/s]: {np.array2string(sigma[3:], precision=6)}")
