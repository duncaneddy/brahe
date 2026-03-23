# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrate OEM construction using add_state, append, extend, and del operations
on segments and state vectors.
"""

import brahe as bh
from brahe.ccsds import OEM, OEMSegment, OEMStateVector

bh.initialize_eop()

# ── Standalone OEMSegment construction ──
epoch1 = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
seg = OEMSegment(
    object_name="TEST SAT",
    object_id="2024-999A",
    center_name="EARTH",
    ref_frame="EME2000",
    time_system="UTC",
    start_time=epoch1,
    stop_time=epoch1 + 120.0,
)
print(f"Standalone segment: {seg.object_name}, {seg.num_states} states")

# ── Add states to standalone segment via add_state ──
seg.add_state(epoch=epoch1, position=[7000e3, 0.0, 0.0], velocity=[0.0, 7500.0, 0.0])
seg.add_state(
    epoch=epoch1 + 60.0, position=[6999e3, 450e3, 0.0], velocity=[0.0, 7499.0, 0.0]
)
seg.add_state(
    epoch=epoch1 + 120.0, position=[6996e3, 900e3, 0.0], velocity=[0.0, 7497.0, 0.0]
)
print(f"After add_state x3: {seg.num_states} states")

# Verify standalone states
states = seg.states  # Returns a list copy for standalone segments
print(f"  States list length: {len(states)}")
print(f"  First: epoch={states[0].epoch}, pos={states[0].position}")

# ── Build OEM and add segment ──
oem = OEM(originator="EXAMPLE")

# Add an empty segment via the builder method
idx = oem.add_segment(
    object_name="SAT-A",
    object_id="2024-001A",
    center_name="EARTH",
    ref_frame="GCRF",
    time_system="UTC",
    start_time=epoch1,
    stop_time=epoch1 + 300.0,
)
print(f"\nOEM segments: {len(oem.segments)}")

# Append states to the OEM segment (proxy mode — mutations reflect back)
proxy_seg = oem.segments[idx]
proxy_seg.states.append(OEMStateVector(epoch1, [8000e3, 0.0, 0.0], [0.0, 7200.0, 0.0]))
proxy_seg.states.append(
    OEMStateVector(epoch1 + 60.0, [7999e3, 432e3, 0.0], [0.0, 7199.0, 0.0])
)
print(f"Proxy segment states: {proxy_seg.num_states}")

# Extend with more states
proxy_seg.states.extend(
    [
        OEMStateVector(epoch1 + 120.0, [7996e3, 864e3, 0.0], [0.0, 7197.0, 0.0]),
    ]
)
print(f"After extend: {proxy_seg.num_states}")

# Delete a state
del proxy_seg.states[1]  # Remove the middle state
print(f"After del [1]: {proxy_seg.num_states} states")

# ── Append standalone segment to OEM ──
oem.segments.append(seg)
print(f"\nAfter append standalone segment: {len(oem.segments)} segments")

# Access the appended segment (now in proxy mode)
appended = oem.segments[1]
print(f"  Segment 1: {appended.object_name}, {appended.num_states} states")

# ── Delete a segment ──
del oem.segments[0]  # Remove the first segment
print(f"After del segment[0]: {len(oem.segments)} segment(s)")
print(f"  Remaining: {oem.segments[0].object_name}")

# ── Extend with multiple segments ──
seg2 = OEMSegment(
    object_name="TEST SAT",
    object_id="2024-999A",
    center_name="EARTH",
    ref_frame="EME2000",
    time_system="UTC",
    start_time=epoch1 + 300.0,
    stop_time=epoch1 + 600.0,
)
oem.segments.extend([seg2])
print(f"After extend segments: {len(oem.segments)}")

# ── Verify round-trip ──
kvn = oem.to_string("KVN")
oem2 = OEM.from_str(kvn)
print(f"\nRound-trip: {len(oem2.segments)} segments")
for i in range(len(oem2.segments)):
    s = oem2.segments[i]
    print(f"  [{i}] {s.object_name}: {s.num_states} states")
