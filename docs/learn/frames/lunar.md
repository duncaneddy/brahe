# Lunar Reference Frames

Lunar reference frames enable orbit propagation and analysis for Moon-orbiting spacecraft. Brahe implements lunar inertial frames parallel to the Earth frame structure.

## Frame Definitions

### LCRF (Lunar Celestial Reference Frame)

The **Lunar Celestial Reference Frame** is the lunar equivalent of GCRF, aligned with the International Celestial Reference Frame (ICRF). This is the primary modern inertial reference frame for lunar orbit analysis.

- **Alignment**: ICRF-aligned (same as GCRF for Earth)
- **Origin**: Moon center of mass
- **Use case**: Modern lunar mission planning and analysis

**Alias**: `LCI` (Lunar-Centered Inertial) is provided as an alternative name for LCRF, following the same pattern as ECI for GCRF.

### MOON_J2000 (Lunar Mean Equator and Equinox of J2000.0)

The **Lunar Mean Equator and Equinox of J2000.0** frame is the lunar equivalent of EME2000, aligned with the J2000.0 mean equatorial plane.

- **Alignment**: J2000.0 mean equator and equinox
- **Origin**: Moon center of mass
- **Use case**: Legacy systems and consistency with J2000-based Earth systems

## Transformation Between Lunar Frames

The transformation between LCRF and MOON_J2000 is a **constant frame bias** (does not depend on time), identical to the EME2000 ↔ GCRF transformation for Earth. This bias accounts for the ~23 milliarcsecond offset between the ICRF and J2000.0 frames.

## Available Functions

### Rotation Matrices

```python
import brahe as bh
import numpy as np

# Get constant bias matrix
B = bh.bias_moon_j2000()

# Get rotation matrices (LCRF ↔ MOON_J2000)
R_lcrf_to_j2000 = bh.rotation_lcrf_to_moon_j2000()
R_j2000_to_lcrf = bh.rotation_moon_j2000_to_lcrf()

# Using LCI alias
R_lci_to_j2000 = bh.rotation_lci_to_moon_j2000()  # Same as LCRF version
```

### Position Transformations

Transform 3D position vectors between lunar frames:

```python
import brahe as bh
import numpy as np

# Position 100 km above lunar surface in LCRF
r_lcrf = np.array([bh.R_MOON + 100e3, 0.0, 0.0])

# Transform to MOON_J2000
r_j2000 = bh.position_lcrf_to_moon_j2000(r_lcrf)

# Transform back
r_lcrf_back = bh.position_moon_j2000_to_lcrf(r_j2000)

# Using LCI alias
r_j2000_alt = bh.position_lci_to_moon_j2000(r_lcrf)
```

### State Transformations

Transform 6D state vectors (position + velocity) between lunar frames:

```python
import brahe as bh
import numpy as np

# Circular orbit state in LCRF [x, y, z, vx, vy, vz] (m, m/s)
state_lcrf = np.array([
    bh.R_MOON + 100e3, 0.0, 0.0,  # Position
    0.0, 1700.0, 0.0               # Velocity (~1.7 km/s orbital speed)
])

# Transform to MOON_J2000
state_j2000 = bh.state_lcrf_to_moon_j2000(state_lcrf)

# Transform back
state_lcrf_back = bh.state_moon_j2000_to_lcrf(state_j2000)
```

## Naming Conventions

Brahe provides two equivalent naming schemes for lunar frames:

| Primary Name | Alias | Description |
|-------------|-------|-------------|
| LCRF | LCI | Lunar Celestial Reference Frame (ICRF-aligned) |
| MOON_J2000 | - | Lunar Mean Equator and Equinox of J2000.0 |

The **LCI** (Lunar-Centered Inertial) alias follows the same pattern as the ECI/ECEF naming for Earth frames, providing familiar terminology for users transitioning from Earth to lunar analysis.

## Practical Usage

### When to Use Each Frame

- **Use LCRF/LCI**: For modern lunar missions and when consistency with ICRF-based systems is required
- **Use MOON_J2000**: For legacy systems or when consistency with J2000-based Earth propagation is needed

### Integration with Orbit Propagation

!!! note "Future Feature"
    Full integration with `OrbitFrame` enum and trajectory propagation will be added in a future release. Currently, lunar frames are available for direct coordinate transformations but cannot yet be used as propagation frames in `NumericalOrbitPropagator` or trajectory objects.

### Example: Coordinate Transformation

```python
import brahe as bh
import numpy as np

# Define a lunar orbit state in LCRF
# 100 km circular equatorial orbit
r_orbit = bh.R_MOON + 100e3
v_orbit = np.sqrt(bh.GM_MOON / r_orbit)  # Circular orbit velocity

state_lcrf = np.array([r_orbit, 0.0, 0.0, 0.0, v_orbit, 0.0])

# Convert to MOON_J2000 for comparison with legacy data
state_j2000 = bh.state_lcrf_to_moon_j2000(state_lcrf)

print(f"State in LCRF: {state_lcrf}")
print(f"State in MOON_J2000: {state_j2000}")

# The difference is very small (~23 milliarcseconds rotation)
diff = np.linalg.norm(state_lcrf[:3] - state_j2000[:3])
print(f"Position difference: {diff:.6f} m")  # Sub-meter difference
```

## Technical Details

### Frame Bias Matrix

The frame bias between LCRF and MOON_J2000 uses the same bias matrix as the Earth EME2000 ↔ GCRF transformation:

```
η₀ = -6.8192 mas
ξ₀ = -16.617 mas  
da₀ = -14.6 mas
```

Where `mas` denotes milliarcseconds. This bias corrects for the offset between the ICRF and J2000.0 mean equator/equinox definitions.

### Precision

The constant frame bias provides milliarcsecond-level precision, suitable for most lunar mission applications. For sub-milliarcsecond precision, more complex time-varying transformations incorporating lunar libration would be required (not currently implemented).

## See Also

- [EME2000 ↔ GCRF Transformations](eme2000_gcrf.md) - Earth equivalent transformation
- [Constants](../constants.md) - `GM_MOON`, `R_MOON` physical constants
- [Orbit Propagation](../orbit_propagation/index.md) - Numerical orbit propagation (future lunar support)
