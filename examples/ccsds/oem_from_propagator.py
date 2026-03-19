# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Generate an OEM from NumericalOrbitPropagator output — propagate a LEO orbit,
extract the trajectory, and build an OEM message.
"""

import brahe as bh
import numpy as np
from brahe.ccsds import OEM, OEMStateVector

bh.initialize_eop()
bh.initialize_sw()

# Define initial state
epoch = bh.Epoch.from_datetime(2024, 6, 15, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
oe = np.array([bh.R_EARTH + 500e3, 0.001, 51.6, 15.0, 30.0, 45.0])
state = bh.state_koe_to_eci(oe, bh.AngleFormat.DEGREES)
params = np.array([500.0, 2.0, 2.2, 2.0, 1.3])

# Create propagator with default force model
prop = bh.NumericalOrbitPropagator(
    epoch,
    state,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.default(),
    params,
)

# Propagate for 90 minutes
target_epoch = epoch + 5400.0
prop.propagate_to(target_epoch)
print(f"Propagated from {epoch} to {prop.current_epoch()}")

# Get the accumulated trajectory
traj = prop.trajectory
print(f"Trajectory: {len(traj)} states, span={traj.timespan():.0f}s")

# Build an OEM from the trajectory states
oem = OEM(originator="BRAHE_PROP")
stop_epoch = prop.current_epoch()
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

# Extract states from trajectory and add to OEM
seg = oem.segments[seg_idx]
for i in range(len(traj)):
    epc, s = traj.get(i)
    seg.states.append(OEMStateVector(epc, s[:3], s[3:6]))

print(f"\nOEM: {len(oem.segments)} segment, {seg.num_states} states")

# Write to KVN
kvn = oem.to_string("KVN")
print(f"KVN output: {len(kvn)} characters")

# Verify by re-parsing
oem2 = OEM.from_str(kvn)
print(f"Round-trip: {oem2.segments[0].num_states} states")
