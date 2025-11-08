#!/usr/bin/env python
# /// script
# dependencies = ["brahe", "plotly"]
# FLAGS = ["CI-ONLY"]
# ///

import time
import brahe as bh

ts = time.time()

bh.initialize_eop()

starlink = bh.datasets.celestrak.get_tles_as_propagators("starlink", 60.0)

for sat in starlink:
    sat.propagate_to(sat.epoch + 86400.0)  # Propagate one orbit (24 hours)

print(
    f"Propagated {len(starlink)} Starlink satellites to one orbit in {bh.format_time_string(time.time() - ts)}."
)
