# /// script
# dependencies = ["brahe", "numpy"]
# FLAGS = ["IGNORE"]
# ///
"""
Load SPICE kernels and query ephemeris data from the global registry.

This example demonstrates the generic NAIF-ID queries (spk_position/velocity/state),
the kernel-scoped variants, and the per-body *_de convenience functions. Downloads
the de440s kernel (~33 MB) on first run.
"""

import brahe as bh
import numpy as np

bh.initialize_eop()

epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)

# Loading is idempotent and explicit; spk_* queries also auto-load de440s if
# no kernel has been loaded yet.
bh.load_kernel("de440s")
print(f"Loaded kernels: {bh.loaded_kernels()}")

# Generic queries take NAIF IDs and resolve across all loaded SPK kernels.
r_moon = bh.spk_position(bh.NAIF_MOON, bh.NAIF_EARTH, epc)
v_moon = bh.spk_velocity(bh.NAIF_MOON, bh.NAIF_EARTH, epc)
x_sun = bh.spk_state(bh.NAIF_SUN, bh.NAIF_EARTH, epc)

print(f"\nMoon position rel. Earth (km): {r_moon / 1e3}")
print(f"Moon velocity rel. Earth (m/s): {v_moon}")
print(f"Sun distance rel. Earth (AU): {np.linalg.norm(x_sun[0:3]) / bh.AU:.6f}")

# Kernel-scoped variants query a single named kernel directly, bypassing
# cross-kernel chaining and precedence.
r_moon_de440s = bh.spk_position_from_kernel("de440s", bh.NAIF_MOON, bh.NAIF_EARTH, epc)
print(f"\nMoon position from de440s directly (km): {r_moon_de440s / 1e3}")

# Per-body convenience functions wrap the same queries for the ten most
# commonly used bodies, selecting the kernel via EphemerisSource.
r_mars = bh.mars_position_de(epc, bh.EphemerisSource.DE440s)
v_mars = bh.mars_velocity_de(epc, bh.EphemerisSource.DE440s)
x_mars = bh.mars_state_de(epc, bh.EphemerisSource.DE440s)

print(f"\nMars position rel. Earth (km): {r_mars / 1e3}")
print(f"Mars velocity rel. Earth (m/s): {v_mars}")
print(f"Position/state consistency check: {np.allclose(r_mars, x_mars[0:3])}")
