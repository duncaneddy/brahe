# /// script
# dependencies = ["brahe"]
# ///
"""
Load groundstation data from embedded providers.

This example demonstrates loading groundstation locations from the
embedded datasets. All data is offline-capable.
"""

import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Load groundstations from a single provider
ksat_stations = bh.datasets.groundstations.load("ksat")
print(f"KSAT stations: {len(ksat_stations)}")

# Load all available providers at once
all_stations = bh.datasets.groundstations.load_all()
print(f"Total stations (all providers): {len(all_stations)}")

# List available providers
providers = bh.datasets.groundstations.list_providers()
print(f"\nAvailable providers: {', '.join(providers)}")

# Load multiple specific providers
aws_stations = bh.datasets.groundstations.load("aws")
ssc_stations = bh.datasets.groundstations.load("ssc")
combined = aws_stations + ssc_stations
print(f"\nCombined AWS + SSC: {len(combined)} stations")

# Expected output:
# KSAT stations: 36
# Total stations (all providers): 96

# Available providers: atlas, aws, ksat, leaf, ssc, viasat

# Combined AWS + SSC: 22 stations
