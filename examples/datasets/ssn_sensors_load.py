# /// script
# dependencies = ["brahe"]
# ///
"""
Load and filter the SSN sensor dataset.

This example demonstrates loading the Vallado SSN sensor sites, filtering
by sensor type, and inspecting a site's properties.
"""

import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Load all SSN sensor sites
sites = bh.datasets.ssn_sensors.load()
print(f"Total SSN sites: {len(sites)}")

# Filter by sensor type: radar/phased-array/mechanical trackers report
# az/el/range, optical trackers report angles-only az/el
radars = [s for s in sites if s.properties["sensor_type"] == "azel_range"]
optical = [s for s in sites if s.properties["sensor_type"] == "azel"]
print(f"Radar/phased-array/mechanical sites: {len(radars)}")
print(f"Optical (angles-only) sites: {len(optical)}")

# Inspect one site's properties
eglin = next(s for s in sites if s.get_name() == "Eglin")
props = eglin.properties
print(f"\n{eglin.get_name()}")
print(f"Location: ({eglin.lat:.2f}, {eglin.lon:.2f})")
print(f"System: {props['system']}")
print(f"Category: {props['category']}")
print(f"Elevation limits: {props.get('el_min_deg')} - {props.get('el_max_deg')} deg")
print(f"Range max: {props.get('range_max_m') / 1e3:.0f} km")
print(f"Azimuth noise: {props.get('az_noise_deg')} deg")

assert len(sites) == 21
assert len(radars) + len(optical) == len(sites)
assert eglin.properties["sensor_type"] == "azel_range"
print("\nExample validated successfully!")
