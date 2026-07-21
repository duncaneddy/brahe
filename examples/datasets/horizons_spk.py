# /// script
# dependencies = ["brahe"]
# FLAGS = ["NETWORK"]
# ///
"""
Generate and load a targeted SPK from JPL Horizons.

Requests an SPK for Ceres (SPK-ID 20000001) over a time span, loads it into
the SPICE registry, and prints the cached kernel path. The .bsp is cached
under $BRAHE_CACHE/horizons and reused for the same request on repeat runs.
"""

import brahe as bh

t0 = bh.Epoch.from_datetime(2015, 12, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.TDB)
t1 = bh.Epoch.from_datetime(2016, 3, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.TDB)

request = bh.datasets.horizons.HorizonsSPKRequest.for_spkid(20000001, t0, t1)
response = bh.datasets.horizons.HorizonsClient().get_spk(request)

response.load()
print(f"Cached at: {response.path}")
