# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Build an OEM message from scratch and write it to KVN format.
"""

import brahe as bh
import numpy as np
from brahe.ccsds import OEM

bh.initialize_eop()

# Create a new OEM with header info
oem = OEM(originator="BRAHE_EXAMPLE")
oem.classification = "unclassified"
oem.message_id = "OEM-2024-001"

# Define a LEO orbit and propagate with KeplerianPropagator (two-body)
epoch = bh.Epoch.from_datetime(2024, 6, 15, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.001, 51.6, 15.0, 30.0, 0.0])
prop = bh.KeplerianPropagator.from_keplerian(epoch, oe, bh.AngleFormat.DEGREES, 60.0)

# Add a segment with metadata
step = 60.0  # 60-second spacing
n_states = 5
stop_epoch = epoch + step * (n_states - 1)

seg_idx = oem.add_segment(
    object_name="LEO SAT",
    object_id="2024-100A",
    center_name="EARTH",
    ref_frame="EME2000",
    time_system="UTC",
    start_time=epoch,
    stop_time=stop_epoch,
    interpolation="LAGRANGE",
    interpolation_degree=7,
)

# Propagate to build trajectory, then bulk-add states to segment
prop.propagate_to(stop_epoch)
seg = oem.segments[seg_idx]
seg.add_trajectory(prop.trajectory)

print(f"Created OEM with {len(oem.segments)} segment, {seg.num_states} states")
# Created OEM with 1 segment, 5 states

# Write to KVN string
kvn = oem.to_string("KVN")
print(f"\nKVN output ({len(kvn)} chars):")
print(kvn[:500])

# Write to file
oem.to_file("/tmp/brahe_example_oem.txt", "KVN")
print("\nWritten to /tmp/brahe_example_oem.txt")

# Verify round-trip
oem2 = OEM.from_file("/tmp/brahe_example_oem.txt")
print(f"Round-trip: {len(oem2.segments)} segment, {oem2.segments[0].num_states} states")
# Round-trip: 1 segment, 5 states
