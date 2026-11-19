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

# Writing is strict by default and refuses a non-compliant message
try:
    omm.to_string("KVN")
except bh.BraheError as err:
    print(f"Strict write:  {err}")
kvn = omm.to_string("KVN", strict=False)
print(f"Lenient write: {len(kvn)} characters")

# Filling in the missing keywords makes the message compliant
omm.creation_date = bh.Epoch.from_datetime(
    2024, 1, 15, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC
)
omm.originator = "CelesTrak"
print(f"Compliant after update: {omm.is_compliant()}")
