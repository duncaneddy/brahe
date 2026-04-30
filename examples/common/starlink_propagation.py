#!/usr/bin/env python
# /// script
# dependencies = ["brahe"]
# FLAGS = ["IGNORE"]
# ///

import time
import brahe as bh

ts = time.time()

bh.initialize_eop()

client = bh.celestrak.CelestrakClient()
records = client.get_gp(group="starlink")
starlink = [record.to_sgp_propagator(60.0) for record in records]

for sat in starlink:
    sat.propagate_to(sat.epoch + 86400.0)  # Propagate one orbit (24 hours)

print(
    f"Propagated {len(starlink)} Starlink satellites to one orbit in {bh.format_time_string(time.time() - ts)}."
)
