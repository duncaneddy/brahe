# /// script
# dependencies = ["brahe"]
# ///
"""
Create an off-nadir constraint for side-looking radar requiring specific viewing geometry.
"""

import brahe as bh

# Side-looking radar requiring specific geometry
constraint = bh.OffNadirConstraint(min_off_nadir_deg=20.0, max_off_nadir_deg=45.0)

print(f"Created: {constraint}")
# Created: OffNadirConstraint(20.0° - 45.0°)
