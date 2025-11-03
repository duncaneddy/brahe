# /// script
# dependencies = ["brahe"]
# ///
"""
Create an off-nadir constraint for imaging payloads with a 30° maximum viewing angle.
"""

import brahe as bh

# Imaging payload with 30° maximum off-nadir
constraint = bh.OffNadirConstraint(max_off_nadir_deg=30.0)

print(f"Created: {constraint}")
# Created: OffNadirConstraint(<= 30.0°)
