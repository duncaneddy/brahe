# /// script
# dependencies = ["brahe"]

import brahe as bh

# Default initializers use caching providers that automatically download new
# data if the local data is more than 7 days old. Only updates on initialization.
bh.initialize_eop()
bh.initialize_sw()

# Print last date of data
print("EOP data available through:", bh.Epoch(bh.get_global_eop_mjd_max()))
print("SW data available through:", bh.Epoch(bh.get_global_sw_mjd_max()))
