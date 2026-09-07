# /// script
# dependencies = ["brahe", "numpy"]
# ///
# FLAGS = [SLOW]
"""
Read an OPM with maneuvers, initialize a propagator, and apply each maneuver
as an impulsive delta-V at the specified ignition epoch using TimeEvent callbacks.

Each delta-V is expressed in the frame named by its MAN_REF_FRAME. Inertial
vectors (J2000/EME2000) are rotated into GCRF by the frame bias, and RTN
vectors are rotated by the RTN-to-inertial matrix built from the spacecraft
state at the ignition epoch.
"""

import numpy as np

import brahe as bh
from brahe.ccsds import OPM

bh.initialize_eop()
bh.initialize_sw()

# Parse OPM with maneuvers
opm = OPM.from_file("test_assets/ccsds/opm/OPMExample2.txt")
print(f"Object: {opm.object_name}")
print(f"Epoch:  {opm.epoch}")
print(f"Maneuvers: {len(opm.maneuvers)}")

# Extract initial state; the OPM declares its state in the TOD frame
state_eci = opm.state_in_frame(bh.CelestialFrame.GCRF)

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

# The message's ignition epochs precede its state epoch, so maneuvers are
# scheduled relative to the state epoch, preserving the spacing between them
first_ignition = opm.maneuvers[0].epoch_ignition
scheduled_epochs = [
    opm.epoch + 3600.0 + (man.epoch_ignition - first_ignition) for man in opm.maneuvers
]


def make_callback(dv_vec, man_idx, is_rtn):
    """Create a closure that rotates the delta-V into GCRF and applies it.

    Args:
        dv_vec (numpy.ndarray): Delta-V [dv1, dv2, dv3] in the maneuver frame (m/s)
        man_idx (int): Index of the maneuver within the OPM
        is_rtn (bool): True if `dv_vec` is expressed in the RTN frame

    Returns:
        callable: Event callback returning the post-maneuver state and action
    """

    def apply_dv(epoch, state):
        dv_gcrf = bh.rotation_rtn_to_eci(state) @ dv_vec if is_rtn else dv_vec
        new_state = state.copy()
        new_state[3] += dv_gcrf[0]
        new_state[4] += dv_gcrf[1]
        new_state[5] += dv_gcrf[2]
        dv_mag = np.linalg.norm(dv_gcrf)
        print(f"  Applied maneuver {man_idx} at {epoch}: |dv|={dv_mag:.3f} m/s")
        return (new_state, bh.EventAction.CONTINUE)

    return apply_dv


# The frame bias between EME2000 (alias J2000) and GCRF is epoch-independent
r_eme2000_to_gcrf = bh.rotation_eme2000_to_gcrf()

# Add an event detector for each maneuver
for i, (man, sched_epoch) in enumerate(zip(opm.maneuvers, scheduled_epochs)):
    dv = man.dv  # [dv1, dv2, dv3] in m/s in the maneuver's ref frame
    frame = man.ref_frame

    if frame in ("J2000", "EME2000"):
        # Inertial delta-V: rotate into GCRF once, ahead of propagation
        callback = make_callback(r_eme2000_to_gcrf @ dv, i, False)
    elif frame == "RTN":
        # RTN delta-V: the rotation depends on the state at the ignition epoch
        callback = make_callback(dv, i, True)
    else:
        raise ValueError(f"Unsupported maneuver reference frame: {frame}")

    event = bh.TimeEvent(sched_epoch, f"Maneuver-{i}")
    event = event.with_callback(callback)
    prop.add_event_detector(event)
    print(
        f"  Registered maneuver {i}: epoch={sched_epoch}, frame={frame}, "
        f"|dv|={np.linalg.norm(dv):.3f} m/s"
    )

# Propagate past all maneuvers
target = scheduled_epochs[-1] + 3600.0  # 1 hour after last maneuver
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
