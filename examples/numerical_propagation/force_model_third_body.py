# /// script
# dependencies = ["brahe"]
# ///
"""
Configuring third-body perturbations with different ephemeris sources.
Shows how to include Sun, Moon, and planetary gravitational effects.
"""

import brahe as bh

# Third-body perturbations configuration
# Gravitational attraction from other celestial bodies

# Option 1: Low-precision analytical ephemerides
# Fast but less accurate (~km level errors for Sun/Moon)
# Only Sun and Moon are available
third_body_low = bh.ThirdBodyConfiguration(
    ephemeris_source=bh.EphemerisSource.LowPrecision,
    bodies=[bh.ThirdBody.SUN, bh.ThirdBody.MOON],
)

# Option 2: DE440s high-precision ephemerides (recommended)
# Uses JPL Development Ephemeris 440 (small bodies version)
# ~m level accuracy, valid 1550-2650 CE
# All planets available, ~17 MB file
third_body_de440s = bh.ThirdBodyConfiguration(
    ephemeris_source=bh.EphemerisSource.DE440s,
    bodies=[bh.ThirdBody.SUN, bh.ThirdBody.MOON],
)

# Option 3: DE440 full-precision ephemerides
# Highest accuracy (~mm level), valid 13200 BCE-17191 CE
# All planets available, ~114 MB file
third_body_de440 = bh.ThirdBodyConfiguration(
    ephemeris_source=bh.EphemerisSource.DE440,
    bodies=[bh.ThirdBody.SUN, bh.ThirdBody.MOON],
)

# Option 4: Include all major planets (high-fidelity)
third_body_all_planets = bh.ThirdBodyConfiguration(
    ephemeris_source=bh.EphemerisSource.DE440s,
    bodies=[
        bh.ThirdBody.SUN,
        bh.ThirdBody.MOON,
        bh.ThirdBody.MERCURY,
        bh.ThirdBody.VENUS,
        bh.ThirdBody.MARS,
        bh.ThirdBody.JUPITER,
        bh.ThirdBody.SATURN,
        bh.ThirdBody.URANUS,
        bh.ThirdBody.NEPTUNE,
    ],
)
