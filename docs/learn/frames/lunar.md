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

=== "Python"

    ```python
    --8<-- "./examples/frames/lunar_rotation_matrices.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/frames/lunar_rotation_matrices.rs:4"
    ```

### Position Transformations

Transform 3D position vectors between lunar frames:

=== "Python"

    ```python
    --8<-- "./examples/frames/lunar_position_transform.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/frames/lunar_position_transform.rs:4"
    ```

### State Transformations

Transform 6D state vectors (position + velocity) between lunar frames:

=== "Python"

    ```python
    --8<-- "./examples/frames/lunar_state_transform.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/frames/lunar_state_transform.rs:4"
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

=== "Python"

    ```python
    --8<-- "./examples/frames/lunar_orbit_conversion.py:8"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/frames/lunar_orbit_conversion.rs:4"
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
