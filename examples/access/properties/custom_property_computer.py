# /// script
# dependencies = ["brahe"]
# ///
"""
Compute custom properties during access searches using property computers
"""

import brahe as bh
import numpy as np

bh.initialize_eop()


class MinRangeComputer(bh.AccessPropertyComputer):
    """Compute minimum range during access window."""

    def sampling_config(self):
        """Sample at window midpoint."""
        return bh.SamplingConfig.midpoint()

    def compute(
        self,
        window,
        sample_epochs,
        sample_states_ecef,
        location_ecef,
        location_geodetic,
    ):
        """Calculate range at window midpoint.

        Args:
            window: AccessWindow with timing information
            sample_epochs: Sample epochs in MJD [N]
            sample_states_ecef: Satellite states [N x 6] in ECEF (m, m/s)
            location_ecef: Location position [x,y,z] in ECEF (m)
            location_geodetic: Location geodetic coords [lon, lat, alt] in (degrees, degrees, m)

        Returns:
            dict: Property values
        """
        # Compute range using sampled satellite state at midtime (first/only sample)
        satellite_state = sample_states_ecef[0]
        range_vec = satellite_state[:3] - location_ecef
        range_km = np.linalg.norm(range_vec) / 1000.0

        # Return as dictionary with raw Python values
        return {"min_range_km": float(range_km)}

    def property_names(self):
        """List properties this computer provides."""
        return ["min_range_km"]


# Setup for access computation
location = bh.PointLocation(-122.4194, 37.7749, 0.0)

# Use TLE for propagator
tle_line1 = "1 25544U 98067A   25306.42331346  .00010070  00000-0  18610-3 0  9999"
tle_line2 = "2 25544  51.6344 342.0717 0004969   8.9436 351.1640 15.49700017536601"
propagator = bh.SGPPropagator.from_tle(tle_line1, tle_line2, 60.0)

epoch_start = propagator.epoch
epoch_end = epoch_start + 86400.0 * 7
constraint = bh.ElevationConstraint(min_elevation_deg=10.0)

# Pass property computer to location_accesses
property_computers = [MinRangeComputer()]

windows = bh.location_accesses(
    [location],
    [propagator],
    epoch_start,
    epoch_end,
    constraint,
    property_computers=property_computers,
)

# Access custom property
print(f"Found {len(windows)} access windows\n")
for i, window in enumerate(windows[:3], 1):  # Show first 3
    range_km = window.properties.additional.get("min_range_km")
    print(f"Window {i}: Min range = {range_km:.1f} km")

# Expected output:
# Found 35 access windows

# Window 1: Min range = 1064.3 km
# Window 2: Min range = 638.1 km
# Window 3: Min range = 1454.7 km
