# /// script
# dependencies = ["brahe"]
# ///
"""
Compute ground station access windows directly from the satellites in an SP3 file.
"""

import brahe as bh

bh.initialize_eop()

sp3 = bh.SP3.from_file("test_assets/sp3/COD0OPSFIN_20243150000_01D_05M_ORB_subset.SP3")

# One SP3Satellite per satellite with valid positions, named by its identifier
satellites = sp3.satellites()
for sat in satellites:
    print(f"{sat.get_name()}: {len(sat)} samples, gaps: {sat.has_gaps()}")

station = bh.PointLocation(-105.27, 40.01, 1655.0).with_name("Boulder")
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)

windows = bh.location_accesses(
    station, satellites, sp3.start_epoch, sp3.end_epoch, constraint
)

print(f"\nAccess windows over {sp3.end_epoch - sp3.start_epoch:.0f} s: {len(windows)}")
for window in windows:
    print(
        f"{window.satellite_name}: {window.start} to {window.end} "
        f"({window.duration / 60.0:.1f} min)"
    )
