# SPICE Kernels

Native SPICE kernel registry: loading/unloading SPK and binary PCK kernels,
and generic NAIF-ID ephemeris/orientation queries. For the per-body `*_de`
convenience functions, see the
[Ephemerides reference](../orbit_dynamics/ephemerides.md).

## Kernel Registry

### load_kernel

::: brahe.load_kernel

### unload_kernel

::: brahe.unload_kernel

### clear_kernels

::: brahe.clear_kernels

### loaded_kernels

::: brahe.loaded_kernels

### load_common_kernels

::: brahe.load_common_kernels

### load_all_kernels

::: brahe.load_all_kernels

## Generic SPK Queries

### spk_position

::: brahe.spk_position

### spk_velocity

::: brahe.spk_velocity

### spk_state

::: brahe.spk_state

## Kernel-Scoped SPK Queries

### spk_position_from_kernel

::: brahe.spk_position_from_kernel

### spk_velocity_from_kernel

::: brahe.spk_velocity_from_kernel

### spk_state_from_kernel

::: brahe.spk_state_from_kernel

## PCK Orientation Queries

### pck_euler_angles

::: brahe.pck_euler_angles

### pck_euler_angle

::: brahe.pck_euler_angle

### pck_euler_rates

::: brahe.pck_euler_rates

### pck_euler_angle_and_rates

::: brahe.pck_euler_angle_and_rates

### pck_quaternion

::: brahe.pck_quaternion

### pck_rotation_matrix

::: brahe.pck_rotation_matrix

## NAIFId

::: brahe.NAIFId
    options:
      show_root_heading: true
      show_root_full_path: false

Full NAIF integer ID listing: [NAIF Integer ID Codes](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html).

**Members:**

| Member | NAIF ID | Body |
|---|---|---|
| `NAIFId.SOLAR_SYSTEM_BARYCENTER` | 0 | Solar System Barycenter |
| `NAIFId.MERCURY_BARYCENTER` | 1 | Mercury barycenter |
| `NAIFId.VENUS_BARYCENTER` | 2 | Venus barycenter |
| `NAIFId.EARTH_MOON_BARYCENTER` | 3 | Earth-Moon Barycenter |
| `NAIFId.MARS_BARYCENTER` | 4 | Mars barycenter |
| `NAIFId.JUPITER_BARYCENTER` | 5 | Jupiter barycenter |
| `NAIFId.SATURN_BARYCENTER` | 6 | Saturn barycenter |
| `NAIFId.URANUS_BARYCENTER` | 7 | Uranus barycenter |
| `NAIFId.NEPTUNE_BARYCENTER` | 8 | Neptune barycenter |
| `NAIFId.PLUTO_BARYCENTER` | 9 | Pluto barycenter |
| `NAIFId.SUN` | 10 | Sun |
| `NAIFId.MERCURY` | 199 | Mercury body center |
| `NAIFId.VENUS` | 299 | Venus body center |
| `NAIFId.EARTH` | 399 | Earth body center |
| `NAIFId.MOON` | 301 | Moon body center |
| `NAIFId.MARS` | 499 | Mars body center |
| `NAIFId.JUPITER` | 599 | Jupiter body center |
| `NAIFId.SATURN` | 699 | Saturn body center |
| `NAIFId.URANUS` | 799 | Uranus body center |
| `NAIFId.NEPTUNE` | 899 | Neptune body center |
| `NAIFId.PLUTO` | 999 | Pluto body center |
| `NAIFId.PHOBOS` | 401 | Phobos, moon of Mars |
| `NAIFId.DEIMOS` | 402 | Deimos, moon of Mars |
| `NAIFId.IO` | 501 | Io, moon of Jupiter |
| `NAIFId.EUROPA` | 502 | Europa, moon of Jupiter |
| `NAIFId.GANYMEDE` | 503 | Ganymede, moon of Jupiter |
| `NAIFId.CALLISTO` | 504 | Callisto, moon of Jupiter |
| `NAIFId.TITAN` | 606 | Titan, moon of Saturn |
| `NAIFId.ARIEL` | 701 | Ariel, moon of Uranus |
| `NAIFId.UMBRIEL` | 702 | Umbriel, moon of Uranus |
| `NAIFId.TITANIA` | 703 | Titania, moon of Uranus |
| `NAIFId.OBERON` | 704 | Oberon, moon of Uranus |
| `NAIFId.MIRANDA` | 705 | Miranda, moon of Uranus |
| `NAIFId.TRITON` | 801 | Triton, moon of Neptune |
| `NAIFId.CHARON` | 901 | Charon, moon of Pluto |

Any other NAIF ID present in a loaded kernel (e.g. a spacecraft or minor
body) also works — pass the raw integer directly. In Rust the equivalent
catch-all is `NAIFId::Id(i32)`.

## FrameId

::: brahe.FrameId
    options:
      show_root_heading: true
      show_root_full_path: false

**Members:**

| Member | Frame Class ID | Frame |
|---|---|---|
| `FrameId.MOON_PA_DE440` | 31008 | Lunar principal-axis body-fixed frame (from `moon_pa_de440`) |

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
