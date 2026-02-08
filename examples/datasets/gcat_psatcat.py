# /// script
# dependencies = ["brahe"]
# FLAGS = ["CI-ONLY"]
# ///
"""
Download and explore the GCAT PSATCAT (payload) catalog.

This example demonstrates downloading the PSATCAT catalog and using
payload-specific filters like category, class, and active status.
"""

import brahe as bh

# Download the PSATCAT catalog
psatcat = bh.datasets.gcat.get_psatcat()
print(f"Loaded {len(psatcat)} PSATCAT records")

# Filter for active payloads (result="S" and no end date)
active = psatcat.filter_active()
print(f"Active payloads: {len(active)}")

# Filter by mission category (COM=communications, IMG=imaging, NAV=navigation, etc.)
comms = psatcat.filter_by_category("COM")
print(f"\nCommunications payloads: {len(comms)}")

# Filter by mission class (A=amateur, B=business, C=civil, D=defense)
civil = psatcat.filter_by_class("C")
print(f"Civil payloads: {len(civil)}")

# Look up a specific payload (ISS Zarya module)
iss = psatcat.get_by_jcat("S25544")
if iss:
    print("\nISS Payload Details:")
    print(f"  Name:       {iss.name}")
    print(f"  Program:    {iss.program}")
    print(f"  Category:   {iss.category}")
    print(f"  Class:      {iss.class_}")
    print(f"  Result:     {iss.result}")

# Expected output:
# Loaded NNNNN PSATCAT records
# Active payloads: NNNN
#
# Communications payloads: NNNNN
# Civil payloads: NNNN
#
# ISS Payload Details:
#   Name:       Zarya Cargo Block
#   Program:    TsM
#   Category:   SS
#   Class:      C
#   Result:     S
