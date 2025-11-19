# /// script
# dependencies = ["brahe"]
# FLAGS = ["CI-ONLY"]
# ///
"""
Download a NAIF DE kernel for planetary ephemeris data.

This example demonstrates how to download and cache DE (Development Ephemeris)
kernels from NASA JPL's NAIF archive.
"""

import brahe as bh

# Initialize EOP data
bh.initialize_eop()

# Download de440s kernel (smaller variant, ~33MB)
# This will download once and cache for future use
kernel_path = bh.datasets.naif.download_de_kernel("de440s")

print(f"Kernel cached at: {kernel_path}")

# Subsequent calls use the cached file - no re-download
kernel_path_again = bh.datasets.naif.download_de_kernel("de440s")
print(f"Retrieved from cache: {kernel_path_again}")

# Optionally copy to a specific location
output_path = "/tmp/my_kernel.bsp"
copied_path = bh.datasets.naif.download_de_kernel("de440s", output_path)
print(f"Copied to: {copied_path}")

# Expected output:
# Kernel cached at: /Users/username/.cache/brahe/naif/de440s.bsp
# Retrieved from cache: /Users/username/.cache/brahe/naif/de440s.bsp
# Copied to: /tmp/my_kernel.bsp
