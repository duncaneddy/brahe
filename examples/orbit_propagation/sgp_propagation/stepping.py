# /// script
# dependencies = ["brahe"]
# ///
"""
Single and multiple step propagation with SGPPropagator
"""

import brahe as bh

bh.initialize_eop()

line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
prop = bh.SGPPropagator.from_tle(line1, line2, 60.0)

# Single step (60 seconds)
prop.step()
print(f"After 1 step: {prop.current_epoch}")

# Multiple steps
prop.propagate_steps(10)
print(f"After 11 total steps: {len(prop.trajectory)} states")

# Step by custom duration
prop.step_by(120.0)
print(f"After custom step: {prop.current_epoch}")

# Expected output:
# After 1 step: 2008-09-20 12:26:40.104 UTC
# After 11 total steps: 12 states
# After custom step: 2008-09-20 12:38:40.104 UTC
