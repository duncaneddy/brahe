# /// script
# dependencies = ["brahe", "numpy"]
# ///
"""
Read an OPM with maneuvers, initialize a propagator, and apply each maneuver
as an impulsive delta-V at the specified ignition epoch using TimeEvent callbacks.
"""

import brahe as bh
import numpy as np
from brahe.ccsds import OPM

bh.initialize_eop()
bh.initialize_sw()

# Parse OPM with maneuvers
opm = OPM.from_file("test_assets/ccsds/opm/OPMExample2.txt")
print(f"Object: {opm.object_name}")
print(f"Epoch:  {opm.epoch}")
print(f"Maneuvers: {len(opm.maneuvers)}")

# Extract initial state (OPM is in TOD frame, convert to ECI)
state_eci = bh.state_ecef_to_eci(opm.epoch, opm.state)

# Spacecraft parameters from OPM
mass = opm.mass or 500.0
params = np.array(
    [
        mass,
        opm.drag_area or 10.0,
        opm.drag_coeff or 2.3,
        opm.solar_rad_area or 10.0,
        opm.solar_rad_coeff or 1.3,
    ]
)

# Create propagator
prop = bh.NumericalOrbitPropagator(
    opm.epoch,
    state_eci,
    bh.NumericalPropagationConfig.default(),
    bh.ForceModelConfig.default(),
    params,
)

# Add event detectors for each maneuver with inertial delta-V
for i, man in enumerate(opm.maneuvers):
    dv = man.dv  # [dvx, dvy, dvz] in m/s in the maneuver's ref frame
    frame = man.ref_frame

    # For this example, only apply inertial-frame maneuvers (J2000/EME2000)
    # RTN maneuvers would require frame rotation which adds complexity
    if frame in ("J2000", "EME2000"):

        def make_callback(dv_vec, man_idx):
            """Create a closure that applies the delta-V."""

            def apply_dv(epoch, state):
                new_state = state.copy()
                new_state[3] += dv_vec[0]
                new_state[4] += dv_vec[1]
                new_state[5] += dv_vec[2]
                dv_mag = np.linalg.norm(dv_vec)
                print(f"  Applied maneuver {man_idx} at {epoch}: |dv|={dv_mag:.3f} m/s")
                return (new_state, bh.EventAction.CONTINUE)

            return apply_dv

        event = bh.TimeEvent(man.epoch_ignition, f"Maneuver-{i}")
        event = event.with_callback(make_callback(dv, i))
        prop.add_event_detector(event)
        print(
            f"  Registered maneuver {i}: epoch={man.epoch_ignition}, frame={frame}, "
            f"|dv|={np.linalg.norm(dv):.3f} m/s"
        )
    else:
        print(f"  Skipping maneuver {i} (RTN frame — requires frame rotation)")

# Propagate past all maneuvers
last_man = opm.maneuvers[-1]
target = last_man.epoch_ignition + 3600.0  # 1 hour after last maneuver
print(f"\nPropagating to {target}...")
prop.propagate_to(target)

# Report final state
final = prop.current_state()
print(f"\nFinal state at {prop.current_epoch()}:")
print(
    f"  Position: [{final[0] / 1e3:.3f}, {final[1] / 1e3:.3f}, {final[2] / 1e3:.3f}] km"
)
print(f"  Velocity: [{final[3]:.3f}, {final[4]:.3f}, {final[5]:.3f}] m/s")

# Check event log
events = prop.event_log()
print(f"\nEvent log: {len(events)} events triggered")
for e in events:
    print(f"  {e}")
