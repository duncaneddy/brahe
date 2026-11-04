# /// script
# dependencies = ["brahe"]
# ///
"""
Chain consecutive SP3 files into one continuous series of samples for each satellite.
"""

import brahe as bh

files = [
    bh.SP3.from_file(f"test_assets/sp3/gbm19500_{name}.sp3")
    for name in ("truncated", "after_drop", "after_no_drop")
]
for sp3 in files:
    print(f"{sp3.start_epoch} to {sp3.end_epoch}: {sp3.num_epochs} epochs")

# The files may be given in any order; neighbors share a boundary epoch
satellites = bh.SP3Satellite.from_files(files)
print(f"\nChained satellites: {len(satellites)}")

for sat in satellites[:3]:
    print(
        f"{sat.get_name()}: {len(sat)} samples from {sat.epochs()[0]} "
        f"to {sat.epochs()[-1]}, gaps: {sat.has_gaps()}"
    )

# Every satellite is one continuous run, so each converts to a single trajectory
traj = satellites[0].to_trajectory()
print(f"\n{satellites[0].get_name()} trajectory: {len(traj)} states in {traj.frame}")

# A file that starts more than 1.5 intervals after the previous one leaves a gap
later = bh.SP3.from_file("test_assets/sp3/gbm19500_large_gap.sp3")
sat = bh.SP3Satellite.from_files([files[0], later])[0]
print(f"\n{sat.get_name()} across a gap: {len(sat)} samples, gaps: {sat.has_gaps()}")
for start, end in sat.segments():
    print(f"  segment {start} to {end}")
print(f"  trajectories: {len(sat.to_trajectory_segments())}")
