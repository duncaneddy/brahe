# /// script
# dependencies = ["brahe"]
# ///
"""
Build a Modified ITC message from a trajectory and write it to a file.

Takes an existing ephemeris, converts it to a trajectory, rebuilds an ITC
message with a new header, generates a Space-Track compliant file name,
writes the file, and reads it back.
"""

import tempfile
from pathlib import Path

import brahe as bh

bh.initialize_eop()

PATH = "test_assets/starlink/MEME_100002_STARLINK-37711_2540149_Operational_1473385800_UNCLASSIFIED.txt"

trajectory = bh.ITC.from_file(PATH).to_trajectory()

header = bh.ITCHeader(
    created=bh.Epoch(2026, 9, 11, 2, 0, 0.0, 0.0, time_system=bh.TimeSystem.UTC),
    ephemeris_source="brahe example",
)
itc = bh.ITC.from_trajectory(trajectory, header)
print(f"Records: {len(itc)}, covariance: {itc.has_covariance}")
print(f"Span: {itc.start_epoch} to {itc.end_epoch}")

name = itc.file_name(
    100002, "STARLINK-37711", bh.EphemerisFileCategory.OPERATIONAL, "nomnvr"
)
print(f"File name: {name}")

with tempfile.TemporaryDirectory() as directory:
    path = Path(directory) / str(name)
    itc.to_file(str(path))
    reread = bh.ITC.from_file(str(path))
    print(f"Re-read {len(reread)} records; state frame {reread.header.state_frame}")
    print(f"First line: {path.read_text().splitlines()[0]}")
