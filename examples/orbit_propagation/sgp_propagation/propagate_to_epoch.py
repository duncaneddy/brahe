# /// script
# dependencies = ["brahe"]
# ///
"""
Propagate to a specific target epoch
"""

import brahe as bh

bh.initialize_eop()

line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
prop = bh.SGPPropagator.from_tle(line1, line2, 60.0)

# Propagate to specific epoch
target = prop.epoch + 7200.0  # 2 hours later
prop.propagate_to(target)

print(f"Target epoch: {target}")
print(f"Current epoch: {prop.current_epoch}")
print(f"Trajectory contains {len(prop.trajectory)} states")

# Expected output:
# Target epoch: 2008-09-20 14:25:40.104 UTC
# Current epoch: 2008-09-20 14:25:40.104 UTC
# Trajectory contains 121 states
