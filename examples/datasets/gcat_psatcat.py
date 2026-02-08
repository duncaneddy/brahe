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

# Filter by mission category
comms = psatcat.filter_by_category("Communications")
print(f"\nCommunications payloads: {len(comms)}")

# Filter by mission class
stations = psatcat.filter_by_class("Station")
print(f"Space stations: {len(stations)}")

# Look up a specific payload
iss = psatcat.get_by_jcat("S049652")
if iss:
    print("\nISS Payload Details:")
    print(f"  Name:       {iss.name}")
    print(f"  Program:    {iss.program}")
    print(f"  Category:   {iss.category}")
    print(f"  Class:      {iss.class_}")
    print(f"  Result:     {iss.result}")
    print(f"  Discipline: {iss.discipline}")

# Expected output:
# Loaded NNNNN PSATCAT records
# Active payloads: NNNN
#
# Communications payloads: NNNN
# Space stations: NN
#
# ISS Payload Details:
#   Name:       ISS (Zarya)
#   Program:    ISS
#   Category:   Human spaceflight
#   Class:      Station
#   Result:     S
#   Discipline: Life sci
