# /// script
# dependencies = ["brahe"]
# ///
"""
Configuring third-body perturbations with different ephemeris sources.
Shows how to include Sun, Moon, and planetary gravitational effects.
"""

import brahe as bh

# Third-body perturbations configuration: one entry per perturbing body,
# each carrying its own ephemeris source and gravity model (point-mass by
# default).

# Option 1: Low-precision analytical ephemerides
# Fast but less accurate (~km level errors for Sun/Moon)
# Only Sun and Moon are available
third_bodies_low = [
    bh.ThirdBodyConfiguration(
        bh.ThirdBody.SUN, ephemeris_source=bh.EphemerisSource.LowPrecision
    ),
    bh.ThirdBodyConfiguration(
        bh.ThirdBody.MOON, ephemeris_source=bh.EphemerisSource.LowPrecision
    ),
]

# Option 2: DE440s high-precision ephemerides (recommended)
# Uses JPL Development Ephemeris 440 (small bodies version)
# ~m level accuracy, valid 1550-2650 CE
# All planets available, ~17 MB file. DE440s is the default source, so bare
# bodies can be passed directly and become point-mass entries.
third_bodies_de440s = [bh.ThirdBody.SUN, bh.ThirdBody.MOON]

# Option 3: DE440 full-precision ephemerides
# Highest accuracy (~mm level), valid 13200 BCE-17191 CE
# All planets available, ~114 MB file
third_bodies_de440 = [
    bh.ThirdBodyConfiguration(
        bh.ThirdBody.SUN, ephemeris_source=bh.EphemerisSource.DE440
    ),
    bh.ThirdBodyConfiguration(
        bh.ThirdBody.MOON, ephemeris_source=bh.EphemerisSource.DE440
    ),
]

# Option 4: Include all major planets (high-fidelity). The *_BARYCENTER
# variants use the planetary-system barycenters with system GMs — the
# classical third-body formulation, resolvable from the DE kernel alone.
third_bodies_all_planets = [
    bh.ThirdBody.SUN,
    bh.ThirdBody.MOON,
    bh.ThirdBody.MERCURY,
    bh.ThirdBody.VENUS,
    bh.ThirdBody.MARS_BARYCENTER,
    bh.ThirdBody.JUPITER_BARYCENTER,
    bh.ThirdBody.SATURN_BARYCENTER,
    bh.ThirdBody.URANUS_BARYCENTER,
    bh.ThirdBody.NEPTUNE_BARYCENTER,
]

# Create force model with Sun/Moon perturbations (common case)
force_config = bh.ForceModelConfig(
    gravity=bh.GravityConfiguration(degree=20, order=20),
    third_body=third_bodies_de440s,
)
print(f"Third bodies: {force_config.third_body}")
