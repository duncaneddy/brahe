# /// script
# dependencies = ["brahe"]
# ///
"""
Create tessellation configurations with default and custom parameters.
"""

import brahe as bh

bh.initialize_eop()

# Default configuration
config = bh.OrbitGeometryTessellatorConfig()
print(f"Default image_width: {config.image_width} m")
print(f"Default image_length: {config.image_length} m")
print(f"Default crosstrack_overlap: {config.crosstrack_overlap} m")
print(f"Default alongtrack_overlap: {config.alongtrack_overlap} m")
print(f"Default min_image_length: {config.min_image_length} m")
print(f"Default max_image_length: {config.max_image_length} m")

# Custom configuration for ascending passes with larger tiles
custom_config = bh.OrbitGeometryTessellatorConfig(
    image_width=10000,
    image_length=15000,
    crosstrack_overlap=300,
    alongtrack_overlap=300,
    asc_dsc=bh.AscDsc.ASCENDING,
    min_image_length=8000,
    max_image_length=25000,
)
print(f"\nCustom image_width: {custom_config.image_width} m")
print(f"Custom image_length: {custom_config.image_length} m")
print(f"Custom asc_dsc: {custom_config.asc_dsc}")

# Expected output:
# Default image_width: 5000.0 m
# Default image_length: 5000.0 m
# Default crosstrack_overlap: 200.0 m
# Default alongtrack_overlap: 200.0 m
# Default min_image_length: 5000.0 m
# Default max_image_length: 5000.0 m
#
# Custom image_width: 10000.0 m
# Custom image_length: 15000.0 m
# Custom asc_dsc: AscDsc.ASCENDING
