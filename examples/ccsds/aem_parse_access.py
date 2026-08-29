# /// script
# dependencies = ["brahe"]
# ///
"""
Parse an AEM file and access header, metadata, and attitude states.
"""

import brahe as bh
from brahe.ccsds import AEM

bh.initialize_eop()

# Parse an AEM with two quaternion segments
aem = AEM.from_file("test_assets/ccsds/aem/AEMExampleG4.txt")

# Header
print(f"Format version: {aem.format_version}")
print(f"Originator:     {aem.originator}")
print(f"Creation date:  {aem.creation_date}")
print(f"Message ID:     {aem.message_id}")

print(f"\nSegments: {len(aem.segments)}")
for i, segment in enumerate(aem.segments):
    print(f"\n  Segment {i}:")
    print(f"    Object name:   {segment.object_name}")
    print(f"    Ref frame A:   {segment.ref_frame_a}")
    print(f"    Ref frame B:   {segment.ref_frame_b}")
    print(f"    Attitude type: {segment.attitude_type}")
    print(f"    Interpolation: {segment.interpolation_method}")
    print(f"    States:        {len(segment.states)}")

    first = segment.states[0]
    wire = first.quaternion.to_vector(scalar_first=False)
    print(
        f"    First quaternion [Q1, Q2, Q3, QC] @ {first.epoch}: "
        f"[{wire[0]:.5f}, {wire[1]:.5f}, {wire[2]:.5f}, {wire[3]:.5f}]"
    )
