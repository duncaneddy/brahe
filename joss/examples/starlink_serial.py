# /// script
# dependencies = ["brahe"]
# ///
import time
import brahe as bh

ts = time.time()

bh.initialize_eop()
client = bh.celestrak.CelestrakClient()
gp_records = client.get_gp(group="starlink")
starlink = [rec.to_sgp_propagator(step_size=60.0) for rec in gp_records]
for sat in starlink:
    sat.propagate_to(bh.Epoch.now() + 86400.0)

te = time.time()
print(f"Took {bh.format_time_string(te - ts)} to run test_parallel.py")
