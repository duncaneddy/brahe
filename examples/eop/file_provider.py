# /// script
# dependencies = ["brahe"]
# ///
"""
Using FileEOPProvider with downloaded EOP data.

Demonstrates:
- Downloading EOP files from IERS
- Loading EOP from file
- Setting global file-based provider
- Interpolation and extrapolation options
"""

import brahe as bh
import tempfile
import os

if __name__ == "__main__":
    # Create temporary directory for EOP data
    temp_dir = tempfile.mkdtemp()
    eop_file = os.path.join(temp_dir, "finals.all.iau2000.txt")

    # Download latest EOP file
    print("Downloading EOP file...")
    bh.download_standard_eop_file(eop_file)
    print(f"Downloaded to {eop_file}")

    # Load from file with interpolation
    provider = bh.FileEOPProvider.from_file(
        eop_file, interpolate=True, extrapolate="Hold"
    )
    bh.set_global_eop_provider_from_file_provider(provider)

    print("File EOP provider initialized")
    print("Use case: Production applications with current EOP data")

    # Cleanup
    os.remove(eop_file)
    os.rmdir(temp_dir)
