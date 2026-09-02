# /// script
# dependencies = ["brahe"]
# ///
"""
Build an AEM message from scratch and write it to KVN format.
"""

import math

import brahe as bh
from brahe.ccsds import AEM, AEMAttitudeState, AEMSegment

bh.initialize_eop()

# One segment spanning 60 seconds, carrying the rotation from EME2000 into the
# spacecraft body frame at each epoch.
t0 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
t1 = t0 + 60.0

segment = AEMSegment(
    "SAT1", "2024-001A", "EME2000", "SC_BODY_1", "UTC", t0, t1, "QUATERNION"
)

# The body starts aligned with EME2000 and rotates 2 degrees about its Z axis
# over the segment. A quaternion stores the half-angle, so the sample at t1
# uses 1 degree.
half_angle = math.radians(1.0)
segment.add_state(
    AEMAttitudeState.from_quaternion(t0, bh.Quaternion(1.0, 0.0, 0.0, 0.0))
)
segment.add_state(
    AEMAttitudeState.from_quaternion(
        t1, bh.Quaternion(math.cos(half_angle), 0.0, 0.0, math.sin(half_angle))
    )
)

aem = AEM("BRAHE_EXAMPLE")
aem.message_id = "AEM-2024-001"
aem.add_segment(segment)

print(
    f"Created AEM with {len(aem.segments)} segment, "
    f"{len(aem.segments[0].states)} attitude states"
)

# Write to KVN string
kvn = aem.to_string("KVN")
print(f"\nKVN output ({len(kvn)} chars):")
print(kvn)

# Write to file
aem.to_file("/tmp/brahe_example_aem.txt", "KVN")
print("\nWritten to /tmp/brahe_example_aem.txt")

# Verify round-trip
aem2 = AEM.from_file("/tmp/brahe_example_aem.txt")
print(
    f"Round-trip: {len(aem2.segments)} segment, "
    f"{len(aem2.segments[0].states)} attitude states"
)
