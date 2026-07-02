# SPICE Kernels

Brahe includes a native reader for NASA JPL NAIF SPICE kernels: binary SPK
ephemeris kernels (Chebyshev position/velocity, Types 2 and 3) and binary
PCK orientation kernels (Chebyshev Euler angles, Type 2). Kernels are parsed
directly by brahe — there is no dependency on the CSPICE toolkit or another
SPICE binding at runtime.

Two ways to query loaded kernels are available:

- **Generic queries** (`spk_position`/`spk_velocity`/`spk_state`,
  `pck_euler_angles`/`pck_rotation_matrix`) take NAIF IDs or frame class IDs
  directly and resolve against a process-wide kernel registry.
- **Per-body convenience functions** (`sun_position_de`, `moon_state_de`, ...)
  wrap the same registry for the ten most commonly used bodies.

For downloading and caching the underlying kernel files, see
[NAIF Ephemeris Kernels](../datasets/naif.md).

## Loading Kernels

`load_kernel` accepts either a known kernel name or a filesystem path, and
auto-detects SPK vs. binary PCK from the file header:

=== "Python"

    ```python
    --8<-- "./examples/spice/spice_kernel_registry.py:13"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/spice/spice_kernel_registry.rs:10"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/spice/spice_kernel_registry.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/spice/spice_kernel_registry.rs.txt"
        ```

Known DE kernel names (`de430`, `de432s`, `de435`, `de438`, `de440`,
`de440s`, `de442`, `de442s`) and the binary PCK name `moon_pa_de440` are
downloaded and cached automatically; any other string is treated as a path
to a local `.bsp` or `.bpc` file. See
[NAIF Ephemeris Kernels](../datasets/naif.md) for kernel sizes and caching
details.

Registry semantics:

- **Idempotent**: calling `load_kernel` with a name/path that is already
  loaded is a no-op.
- **Persistent**: kernels stay resident until `unload_kernel` or
  `clear_kernels` is called.
- **Precedence**: `spk_*` queries resolve across every loaded SPK kernel; for
  body pairs covered by more than one kernel, the most recently loaded
  kernel wins (matching SPICE's own "last loaded wins" convention). `pck_*`
  queries search loaded PCK kernels newest-first for a frame with coverage
  at the requested epoch.
- **Auto-initialization**: `spk_position`/`spk_velocity`/`spk_state` load
  `de440s` automatically if no SPK kernel has been loaded yet. Binary PCKs
  are never auto-loaded — `load_kernel("moon_pa_de440")` (or an explicit
  path) must be called first.

## Querying Ephemeris Data

### Generic NAIF-ID Queries

`spk_position`, `spk_velocity`, and `spk_state` take a target NAIF ID, a
center NAIF ID, and an `Epoch`, and resolve across all loaded SPK kernels:

| Function | Returns | Units |
|---|---|---|
| `spk_position(target, center, epc)` | Position of `target` rel. `center` | m |
| `spk_velocity(target, center, epc)` | Velocity of `target` rel. `center` | m/s |
| `spk_state(target, center, epc)` | `[x, y, z, vx, vy, vz]` of `target` rel. `center` | m, m/s |

Common NAIF IDs are exposed as constants:

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

Any other NAIF ID present in a loaded kernel (e.g. a specific outer-planet
moon) also works; these constants only cover the bodies used elsewhere in
brahe.

### Kernel-Scoped Queries

`spk_position_in_kernel`, `spk_velocity_in_kernel`, and `spk_state_in_kernel`
take an additional `kernel_name` argument and query **that kernel only** —
no cross-kernel chaining is performed and the registry's precedence rules do
not apply. The named kernel is auto-loaded if not already resident. Use
these when a query must come from a specific kernel regardless of what else
is loaded.

### Per-Body Functions

`sun_position_de`, `moon_position_de`, `mercury_position_de`,
`venus_position_de`, `mars_position_de`, `jupiter_position_de`,
`saturn_position_de`, `uranus_position_de`, `neptune_position_de`, and
`solar_system_barycenter_position_de` (alias: `ssb_position_de`) each have
`_velocity_de` and `_state_de` counterparts, all queried relative to Earth
(`NAIF_EARTH`). They take an `Epoch` and an ephemeris source
(`EphemerisSource.DE440s` / `EphemerisSource.DE440` in Python,
`SPKKernel::DE440s` / `SPKKernel::DE440` in Rust) selecting which kernel to
query. Computing the state shares a single record lookup between position
and velocity, so prefer `*_state_de` over separate position/velocity calls
when both are needed.

The Mars/Jupiter/Saturn/Uranus/Neptune functions query the corresponding
`NAIF_*_BARYCENTER` ID (see the table above), so they return the
planetary-system barycenter, not the body center — for these five outer
planets the two differ by up to a few hundred km due to large moons
(e.g. ~290 km for Saturn from Titan, ~230 km for Jupiter from the Galilean
moons). Sun, Moon, Mercury, and Venus have no
significant satellites, so their functions' body-center and barycenter
positions coincide.

## PCK Orientation

`pck_euler_angles` and `pck_rotation_matrix` query body orientation from a
loaded binary PCK, given a frame class ID. The lunar principal-axis frame
from the `moon_pa_de440` kernel is registered under frame class ID `31008`
(`MOON_PA_DE440`):

=== "Python"

    ```python
    --8<-- "./examples/spice/spice_pck_orientation.py:14"
    ```

=== "Rust"

    ```rust
    --8<-- "./examples/spice/spice_pck_orientation.rs:11"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/spice/spice_pck_orientation.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/spice/spice_pck_orientation.rs.txt"
        ```

`pck_euler_angles` returns 3-1-3 Euler angles `[phi, delta, w]` (rad) and
their rates (rad/s); `pck_rotation_matrix` returns the corresponding 3x3
rotation matrix from the segment's reference frame to the body-fixed frame.

## Reference Frame

SPK and PCK outputs are expressed in the kernel's inertial reference frame.
For DE4xx-era kernels (DE430 and later) that frame is ICRF, even though NAIF
labels it "J2000" in kernel metadata and documentation — see the
[NAIF Frames Required Reading](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html)
for the frame-ID definitions and the historical "J2000" naming. Brahe
applies no additional bias rotation between the kernel output and GCRF:
values from `spk_*`/`*_de` queries are GCRF-compatible directly.

## Performance

SPK and PCK kernels are parsed once, at `load_kernel` time, into an
in-memory segment table. Queries take a short-lived read lock on the shared
registry only to look up the relevant segment chain; the Chebyshev
evaluation itself runs outside any lock. Repeated queries for the same
target/center pair reuse a cached chain, so sequential access patterns (e.g.
a propagator stepping through epochs) avoid re-resolving kernel coverage on
every call.

## See Also

- [Ephemerides API Reference](../../library_api/orbit_dynamics/ephemerides.md) - Per-body `*_de` function reference
- [SPICE Kernels API Reference](../../library_api/spice/index.md) - Kernel registry and generic query reference
- [NAIF Ephemeris Kernels](../datasets/naif.md) - Downloading and caching DE/PCK kernel files
- [Third-Body Perturbations](../orbital_dynamics/third_body.md) - Using DE ephemerides in force models
- [Epoch](../time/epoch.md) - Converting to SPICE ephemeris time (ET)
