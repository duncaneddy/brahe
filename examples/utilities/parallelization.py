# /// script
# dependencies = ["brahe"]
# ///
"""
Demonstrate thread pool configuration utilities.

This example shows how to configure and query the global thread pool
used by Brahe for parallel computation operations.
"""

import brahe as bh

bh.initialize_eop()

# Query the default number of threads
# By default, Brahe uses 90% of available CPU cores
default_threads = bh.get_max_threads()
print(f"Default thread count: {default_threads}")

# Set a specific number of threads
bh.set_num_threads(4)
threads_after_set = bh.get_max_threads()
print(f"Thread count after setting to 4: {threads_after_set}")

# Set to maximum available (100% of CPU cores)
bh.set_max_threads()
max_threads = bh.get_max_threads()
print(f"Maximum thread count: {max_threads}")

# Alternative: use the fun alias!
bh.set_ludicrous_speed()
ludicrous_threads = bh.get_max_threads()
print(f"Ludicrous speed thread count: {ludicrous_threads}")

# The thread pool can be reconfigured at any time
bh.set_num_threads(2)
final_threads = bh.get_max_threads()
print(f"Final thread count: {final_threads}")

# Note: Thread pool is used for parallelizable operations like:
# - Computing access windows between satellites and ground locations
# - Processing large batches of orbital calculations

# Expected output (actual numbers vary by system):
# Default thread count: 7
# Thread count after setting to 4: 4
# Maximum thread count: 8
# Ludicrous speed thread count: 8
# Final thread count: 2
