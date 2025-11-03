# /// script
# dependencies = ["brahe"]
# ///
"""
Create an ascending/descending constraint for ascending passes only.
"""

import brahe as bh
from brahe import AscDsc

# Only ascending passes
constraint = bh.AscDscConstraint(allowed=AscDsc.ASCENDING)

print(f"Created: {constraint}")
# Created: AscDscConstraint(Ascending)
