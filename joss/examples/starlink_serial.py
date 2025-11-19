# /// script
# dependencies = ["brahe"]
# ///
import time
import brahe as bh

ts = time.time()

bh.initialize_eop()
starlink = bh.datasets.celestrak.get_tles_as_propagators("starlink", 60.0)
for sat in starlink:
    sat.propagate_to(bh.Epoch.now() + 86400.0)

te = time.time()
print(f"Took {bh.format_time_string(te - ts)} to run test_parallel.py")
