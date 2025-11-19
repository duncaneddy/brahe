# /// script
# dependencies = ["brahe"]
# ///
import brahe as bh

bh.initialize_eop()
passes = bh.location_accesses(
    bh.PointLocation(-122.4194, 37.7749, 0.0),  # San Francisco
    bh.celestrak.get_tle_by_id_as_propagator(25544, 60.0, "active"),  # ISS
    bh.Epoch.now(),
    bh.Epoch.now() + 24 * 3600.0,  # Next 24 hours
    bh.ElevationConstraint(min_elevation_deg=10.0),
)

print("Upcoming passes for ISS over San Francisco in the next 24 hours:")
for p in passes:
    print(
        f"Start: {p.t_start}, End: {p.t_end}, Max Elevation: {p.elevation_max:.2f} deg"
    )
