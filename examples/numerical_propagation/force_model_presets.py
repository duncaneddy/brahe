# /// script
# dependencies = ["brahe"]
# ///
"""
Overview of all preset force model configurations.
Shows what each preset includes and when to use them.
"""

import brahe as bh

# Brahe provides several preset configurations for common scenarios

# 1. two_body() - Point mass gravity only
# Use for: Validation, comparison with Keplerian, quick estimates
two_body = bh.ForceModelConfig.two_body()

# 2. earth_gravity() - Spherical harmonic gravity only (20x20)
# Use for: Studying gravity perturbations in isolation
earth_gravity = bh.ForceModelConfig.earth_gravity()

# 3. conservative_forces() - Gravity + third-body + relativity (no drag/SRP)
# Use for: Long-term orbit evolution, conservative dynamics studies
conservative = bh.ForceModelConfig.conservative_forces()

# 4. default() - Balanced configuration for LEO to GEO
# Use for: General mission analysis, initial studies
default = bh.ForceModelConfig.default()

# 5. leo_default() - Optimized for Low Earth Orbit
# Use for: LEO missions where drag is dominant
leo = bh.ForceModelConfig.leo_default()

# 6. geo_default() - Optimized for Geostationary Orbit
# Use for: GEO missions where SRP and third-body dominate
geo = bh.ForceModelConfig.geo_default()

# 7. high_fidelity() - Maximum precision
# Use for: Precision orbit determination, research applications
high_fidelity = bh.ForceModelConfig.high_fidelity()
