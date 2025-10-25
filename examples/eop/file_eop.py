# /// script
# dependencies = ["brahe"]
# ///
"""
This example demonstrates ways to initialize EOP from files using Brahe.
"""

import brahe as bh

# Method 1: Default Providers -> These are packaged data files within Brahe

# File-based EOP Provider - Default IERS Standard with Hold Extrapolation
eop_file_default = bh.FileEOPProvider.from_default_standard(
    True,  # Interpolation -> if True times between data points are interpolated
    "Hold",  # Extrapolation method -> How accesses outside data range are handled
)
bh.set_global_eop_provider(eop_file_default)

# File-based EOP Provider - Default C04 Standard with Zero Extrapolation
eop_file_c04 = bh.FileEOPProvider.from_default_c04(False, "Zero")
bh.set_global_eop_provider(eop_file_c04)

# Method 2: Custom File Paths -> Replace 'path_to_file.txt' with actual file paths

if False:  # Change to True to enable custom file examples
    # File-based EOP Provider - Custom Standard File
    eop_file_custom = bh.FileEOPProvider.from_standard_file(
        "path_to_standard_file.txt",  # Replace with actual file path
        True,  # Interpolation
        "Hold",  # Extrapolation
    )
    bh.set_global_eop_provider(eop_file_custom)

    # File-based EOP Provider - Custom C04 File
    eop_file_custom_c04 = bh.FileEOPProvider.from_c04_file(
        "path_to_c04_file.txt",  # Replace with actual file path
        True,  # Interpolation
        "Hold",  # Extrapolation
    )
    bh.set_global_eop_provider(eop_file_custom_c04)
