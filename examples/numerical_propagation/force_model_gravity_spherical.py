# /// script
# dependencies = ["brahe"]
# ///
"""
Configuring spherical harmonic gravity with different model sources.
Shows packaged models (EGM2008, GGM05S, JGM3) and custom file loading.
"""

import brahe as bh

# ==============================================================================
# Packaged Gravity Models
# ==============================================================================

# EGM2008 - High-fidelity NGA model (360x360 max)
gravity_egm2008 = bh.GravityConfiguration.spherical_harmonic(
    degree=20, order=20, model_type=bh.GravityModelType.EGM2008_360
)

# GGM05S - GRACE mission model (180x180 max)
gravity_ggm05s = bh.GravityConfiguration.spherical_harmonic(
    degree=20, order=20, model_type=bh.GravityModelType.GGM05S
)

# JGM3 - Legacy model, fast computation (70x70 max)
gravity_jgm3 = bh.GravityConfiguration.spherical_harmonic(
    degree=20, order=20, model_type=bh.GravityModelType.JGM3
)

# ==============================================================================
# Custom Model from File
# ==============================================================================

# Load custom gravity model from GFC format file
# GravityModelType.from_file validates the path exists
custom_model_type = bh.GravityModelType.from_file("data/gravity_models/EGM2008_360.gfc")
gravity_custom = bh.GravityConfiguration.spherical_harmonic(
    degree=20, order=20, model_type=custom_model_type
)
