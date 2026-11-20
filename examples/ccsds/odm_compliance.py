# /// script
# dependencies = ["brahe"]
# ///
"""
Read an OMM with an empty header and report its missing mandatory keywords.
"""

import brahe as bh
from brahe.ccsds import OMM

bh.initialize_eop()

# CelesTrak publishes OMMs with empty CREATION_DATE and ORIGINATOR values
omm = OMM.from_file("test_assets/ccsds/omm/OMM-celestrak-empty-header.txt")

print(f"Object name:   {omm.object_name}")
print(f"Creation date: {omm.creation_date}")
print(f"Originator:    {omm.originator}")
print(f"Compliant:     {omm.is_compliant()}")
print(f"Issues:        {omm.compliance_issues()}")

# Filling in the missing keywords makes the message compliant
omm.creation_date = bh.Epoch.from_datetime(
    2024, 1, 15, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC
)
omm.originator = "CelesTrak"
print(f"Compliant after update: {omm.is_compliant()}")
