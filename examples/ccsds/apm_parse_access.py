# /// script
# dependencies = ["brahe"]
# ///
"""
Parse an APM file and access header, metadata, and quaternion attitude data.
"""

import brahe as bh
from brahe.ccsds import APM

bh.initialize_eop()

# Parse APM with a single attitude quaternion block
apm = APM.from_file("test_assets/ccsds/apm/APMExampleG1.txt")

# Header
print(f"Format version: {apm.format_version}")
print(f"Originator:     {apm.originator}")
print(f"Creation date:  {apm.creation_date}")
print(f"Message ID:     {apm.message_id}")

# Metadata
print(f"\nObject name:  {apm.object_name}")
print(f"Object ID:    {apm.object_id}")
print(f"Center name:  {apm.center_name}")
print(f"Time system:  {apm.time_system}")

# Epoch (shared by all blocks except maneuvers)
print(f"\nEpoch: {apm.epoch}")

# Attitude quaternion blocks
print(f"\nQuaternion blocks: {len(apm.quaternion_states)}")
for i, q in enumerate(apm.quaternion_states):
    print(f"\n  Block {i}:")
    print(f"    Ref frame A: {q.ref_frame_a}")
    print(f"    Ref frame B: {q.ref_frame_b}")
    wire = q.quaternion.to_vector(scalar_first=False)
    print(
        f"    Quaternion [Q1, Q2, Q3, QC]: [{wire[0]:.5f}, {wire[1]:.5f}, {wire[2]:.5f}, {wire[3]:.5f}]"
    )
