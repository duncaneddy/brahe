# SPICE Kernels

Native SPICE kernel registry: loading/unloading SPK and binary PCK kernels,
and generic NAIF-ID ephemeris/orientation queries. For the per-body `*_spice`
convenience functions, see the
[Ephemerides reference](../orbit_dynamics/ephemerides.md).

## Kernel Registry

::: brahe.load_spice_kernel

::: brahe.unload_spice_kernel

::: brahe.clear_spice_kernels

::: brahe.loaded_spice_kernels

::: brahe.load_common_spice_kernels

::: brahe.load_all_spice_kernels

## Generic SPK Queries

::: brahe.spk_position

::: brahe.spk_velocity

::: brahe.spk_state

## Kernel-Scoped SPK Queries

::: brahe.spk_position_from_kernel

::: brahe.spk_velocity_from_kernel

::: brahe.spk_state_from_kernel

## PCK Orientation Queries

::: brahe.pck_euler_angles

::: brahe.pck_euler_angle

::: brahe.pck_euler_rates

::: brahe.pck_euler_angle_and_rates

::: brahe.pck_quaternion

::: brahe.pck_rotation_matrix

## NAIFId

Full NAIF integer ID listing: [NAIF Integer ID Codes](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html).

::: brahe.NAIFId
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - SOLAR_SYSTEM_BARYCENTER
        - MERCURY_BARYCENTER
        - VENUS_BARYCENTER
        - EARTH_MOON_BARYCENTER
        - MARS_BARYCENTER
        - JUPITER_BARYCENTER
        - SATURN_BARYCENTER
        - URANUS_BARYCENTER
        - NEPTUNE_BARYCENTER
        - PLUTO_BARYCENTER
        - SUN
        - MERCURY
        - VENUS
        - EARTH
        - MOON
        - MARS
        - JUPITER
        - SATURN
        - URANUS
        - NEPTUNE
        - PLUTO
        - PHOBOS
        - DEIMOS
        - IO
        - EUROPA
        - GANYMEDE
        - CALLISTO
        - TITAN
        - ARIEL
        - UMBRIEL
        - TITANIA
        - OBERON
        - MIRANDA
        - TRITON
        - CHARON
      show_bases: false
      heading_level: 3

Any other NAIF ID present in a loaded kernel (e.g. a spacecraft or minor
body) also works — pass the raw integer directly. In Rust the equivalent
catch-all is `NAIFId::Id(i32)`.

## FrameId

::: brahe.FrameId
    options:
      show_root_heading: true
      show_root_full_path: false
      members:
        - MOON_PA_DE440
      show_bases: false
      heading_level: 3

Any other frame class ID present in a loaded binary PCK also works — pass
the raw integer directly. In Rust the equivalent catch-all is
`FrameId::Id(i32)`. Full frame reference:
[NAIF Frames Required Reading](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html).

## See Also

- [SPICE Kernels Guide](../../learn/spice/index.md) - Kernel loading, registry behavior, and PCK orientation
- [Ephemerides](../orbit_dynamics/ephemerides.md) - Per-body position/velocity/state functions
- [NAIF Functions](../datasets/naif.md) - Downloading and caching kernel files
- [EulerAngle](../attitude/euler_angles.md) - Type returned by `pck_euler_angle`/`pck_euler_angle_and_rates`
- [Quaternion](../attitude/quaternion.md) - Type returned by `pck_quaternion`
- [RotationMatrix](../attitude/rotation_matrix.md) - Type returned by `pck_rotation_matrix`
