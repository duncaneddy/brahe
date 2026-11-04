# /// script
# dependencies = ["brahe"]
# ///
"""
Read a gzip-compressed SP3 file, write a gzip copy, and read the bytes directly.
"""

import os
import tempfile

import brahe as bh

path = "test_assets/sp3/IGS0OPSFIN_20243150000_01D_15M_ORB_subset.SP3.gz"

# The gzip data is detected from the first bytes of the file
sp3 = bh.SP3.from_file(path)
print(f"Read {sp3.num_satellites} satellites and {sp3.num_epochs} epochs from {path}")

# Bytes already in memory, for example a downloaded product, are parsed directly
with open(path, "rb") as f:
    from_memory = bh.SP3.from_bytes(f.read())
print(f"Same content from bytes: {from_memory.to_string() == sp3.to_string()}")

# A path ending in .gz selects gzip output
with tempfile.TemporaryDirectory() as directory:
    out_path = os.path.join(directory, "orbits.sp3.gz")
    sp3.to_file(out_path)
    with open(out_path, "rb") as f:
        print(f"First bytes of the written file: {f.read(2).hex()}")
    print(f"Round trip: {bh.SP3.from_file(out_path).to_string() == sp3.to_string()}")
