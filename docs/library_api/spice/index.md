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

## Generic SPK Queries

### spk_position

::: brahe.spk_position

### spk_velocity

::: brahe.spk_velocity

### spk_state

::: brahe.spk_state

## Kernel-Scoped SPK Queries

### spk_position_in_kernel

::: brahe.spk_position_in_kernel

### spk_velocity_in_kernel

::: brahe.spk_velocity_in_kernel

### spk_state_in_kernel

::: brahe.spk_state_in_kernel

## PCK Orientation Queries

### pck_euler_angles

::: brahe.pck_euler_angles

### pck_rotation_matrix

::: brahe.pck_rotation_matrix

## NAIF ID Constants

| Constant | NAIF ID | Body |
|---|---|---|
| `NAIF_SSB` | 0 | Solar System Barycenter |
| `NAIF_MERCURY_BARYCENTER` | 1 | Mercury barycenter |
| `NAIF_VENUS_BARYCENTER` | 2 | Venus barycenter |
| `NAIF_EMB` | 3 | Earth-Moon Barycenter |
| `NAIF_MARS_BARYCENTER` | 4 | Mars barycenter |
| `NAIF_JUPITER_BARYCENTER` | 5 | Jupiter barycenter |
| `NAIF_SATURN_BARYCENTER` | 6 | Saturn barycenter |
| `NAIF_URANUS_BARYCENTER` | 7 | Uranus barycenter |
| `NAIF_NEPTUNE_BARYCENTER` | 8 | Neptune barycenter |
| `NAIF_PLUTO_BARYCENTER` | 9 | Pluto barycenter |
| `NAIF_SUN` | 10 | Sun |
| `NAIF_MERCURY` | 199 | Mercury body center |
| `NAIF_VENUS` | 299 | Venus body center |
| `NAIF_EARTH` | 399 | Earth body center |
| `NAIF_MOON` | 301 | Moon body center |
| `NAIF_MARS` | 499 | Mars body center |

## See Also

- [SPICE Kernels Guide](../../learn/spice/index.md) - Kernel loading, registry semantics, and PCK orientation
- [Ephemerides](../orbit_dynamics/ephemerides.md) - Per-body position/velocity/state functions
- [NAIF Functions](../datasets/naif.md) - Downloading and caching kernel files
