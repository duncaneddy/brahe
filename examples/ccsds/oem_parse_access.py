# /// script
# dependencies = ["brahe"]
# ///
"""
Parse an OEM file and access header, segment metadata, and state vectors.
"""

import brahe as bh
from brahe.ccsds import OEM

bh.initialize_eop()

# Parse from file (auto-detects KVN, XML, or JSON format)
oem = OEM.from_file("test_assets/ccsds/oem/OEMExample1.txt")

# Header properties
print(f"Format version: {oem.format_version}")
print(f"Originator:     {oem.originator}")
print(f"Classification: {oem.classification}")
print(f"Creation date:  {oem.creation_date}")

# Segments — OEM can contain multiple trajectory arcs
print(f"\nNumber of segments: {len(oem.segments)}")

# Access segment metadata
seg = oem.segments[0]
print("\nSegment 0:")
print(f"  Object name:   {seg.object_name}")
print(f"  Object ID:     {seg.object_id}")
print(f"  Center name:   {seg.center_name}")
print(f"  Ref frame:     {seg.ref_frame}")
print(f"  Time system:   {seg.time_system}")
print(f"  Start time:    {seg.start_time}")
print(f"  Stop time:     {seg.stop_time}")
print(f"  Interpolation: {seg.interpolation}")
print(f"  States:        {seg.num_states}")
print(f"  Covariances:   {seg.num_covariances}")

# Access individual state vectors
sv = seg.states[0]
print("\nFirst state vector:")
print(f"  Epoch:    {sv.epoch}")
print(
    f"  Position: [{sv.position[0]:.3f}, {sv.position[1]:.3f}, {sv.position[2]:.3f}] m"
)
print(
    f"  Velocity: [{sv.velocity[0]:.5f}, {sv.velocity[1]:.5f}, {sv.velocity[2]:.5f}] m/s"
)

# Iterate over all states in a segment
print("\nAll states in segment 0:")
for i, sv in enumerate(seg.states):
    print(
        f"  [{i}] {sv.epoch}  pos=({sv.position[0] / 1e3:.3f}, {sv.position[1] / 1e3:.3f}, {sv.position[2] / 1e3:.3f}) km"
    )

# Serialization
kvn = oem.to_string("KVN")
print(f"\nKVN output length: {len(kvn)} characters")
d = oem.to_dict()
print(f"Dict keys: {list(d.keys())}")
