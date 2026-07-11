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

Known kernel names — the eight planetary DE kernels (`de430`, `de432s`,
`de435`, `de438`, `de440`, `de440s`, `de442`, `de442s`), the seven
satellite-system kernels (`mar099`, `mar099s`, `jup365`, `sat441`, `ura184`,
`nep097`, `plu060`), and the binary PCK `moon_pa_de440` — are downloaded and
cached automatically; any other string is treated as a path to a local
`.bsp` or `.bpc` file (bring-your-own kernels). The `moon_pa_de440` binary
PCK is derived from the same DE440 integration as `de440.bsp` but is
distributed by NAIF as a separate file — SPK kernels carry only
translational states. See [NAIF Ephemeris Kernels](../datasets/naif.md) for
caching details.

### Downloadable Kernels

| Name | File | Size | Coverage | Contents |
|---|---|---|---|---|
| `de440s` | `de440s.bsp` | ~31 MB | 1849–2150 | Sun, Moon, all planets and planetary-system barycenters (default for `spk_*`/`*_de` auto-init) |
| `de440` | `de440.bsp` | ~114 MB | 1550–2650 | Same content as `de440s`, wider time span |
| `de430` | `de430.bsp` | ~114 MB | — | Standard precision, extended time span |
| `de432s` | `de432s.bsp` | ~11 MB | — | Small variant tuned for New Horizons Pluto targeting |
| `de435` | `de435.bsp` | ~114 MB | — | Higher accuracy for inner planets |
| `de438` | `de438.bsp` | ~114 MB | — | Standard precision |
| `de442` | `de442.bsp` | ~114 MB | — | Intended for the MESSENGER mission to Mercury |
| `de442s` | `de442s.bsp` | ~31 MB | — | Small variant of `de442` |
| `mar099s` | `mar099s.bsp` | ~64 MB | 1995–2050 | Mars, Phobos, Deimos (default Mars satellite kernel for body-center auto-download) |
| `mar099` | `mar099.bsp` | ~1.1 GB | 1600–2600 | Mars, Phobos, Deimos, wider time span |
| `jup365` | `jup365.bsp` | ~1.1 GB | 1600–2200 | Jupiter, Io, Europa, Ganymede, Callisto |
| `sat441` | `sat441.bsp` | ~631 MB | 1750–2250 | Saturn, Titan and the mid-size moons |
| `ura184` | `ura184_part-3.bsp` | ~387 MB | 1600–2399 | Uranus, Miranda, Ariel, Umbriel, Titania, Oberon |
| `nep097` | `nep097.bsp` | ~100 MB | 1600–2400 | Neptune, Triton |
| `plu060` | `plu060.bsp` | ~129 MB | 1800–2200 | Pluto, Charon |
| `moon_pa_de440` | `moon_pa_de440_200625.bpc` | ~13 MB | matches `de440`/`de440s` | Lunar principal-axis orientation (binary PCK, not an SPK) |

Any string that does not match a name in this table is treated as a
filesystem path to a local `.bsp` or `.bpc` file.

Registry behavior:

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

### Loading Multiple Kernels at Once

`load_common_kernels` and `load_all_kernels` pre-load a curated set of
kernels in one call, so a session does not pay per-kernel download latency
the first time each body is queried:

| Function | Loads | Download size |
|---|---|---|
| `load_common_kernels()` | `de440s`, `moon_pa_de440` | ~46 MB |
| `load_all_kernels()` | `de440s`, `moon_pa_de440`, `mar099s`, `jup365`, `sat441`, `ura184`, `nep097`, `plu060` | ~2.4 GB |

Each kernel load within these calls is idempotent, so calling either
alongside individual `load_kernel` calls is safe. Prefer
`load_common_kernels` unless outer-planet body centers, their moons, or
Pluto are needed — `load_all_kernels` downloads over 2 GB on first use.

## Querying Ephemeris Data

### Generic NAIF-ID Queries

`spk_position`, `spk_velocity`, and `spk_state` take a target NAIF ID, a
center NAIF ID, and an `Epoch`, and resolve across all loaded SPK kernels:

| Function | Returns | Units |
|---|---|---|
| `spk_position(target, center, epc)` | Position of `target` rel. `center` | m |
| `spk_velocity(target, center, epc)` | Velocity of `target` rel. `center` | m/s |
| `spk_state(target, center, epc)` | `[x, y, z, vx, vy, vz]` of `target` rel. `center` | m, m/s |

Common NAIF IDs are exposed through the `NAIFId` enum (Python: `IntEnum`;
Rust: `enum NAIFId` with an `Id(i32)` catch-all variant), covering the ten
planetary-system barycenters, the Sun, the eight planet body centers, and
the major natural satellites used elsewhere in brahe (Phobos, Deimos, the
four Galilean moons, Titan, the five major Uranian moons, Triton, Charon).
`NAIFId` values compare and pass equal to the equivalent raw integer, so
either form works everywhere a NAIF ID is expected:

=== "Python"

    ```python
    import brahe as bh

    epc = bh.Epoch.from_date(2025, 1, 1, bh.TimeSystem.UTC)
    r1 = bh.spk_position(bh.NAIFId.MOON, bh.NAIFId.EARTH, epc)
    r2 = bh.spk_position(301, 399, epc)  # equivalent raw NAIF IDs
    ```

=== "Rust"

    ```rust
    use brahe::spice::{spk_position, NAIFId};

    let r1 = spk_position(NAIFId::Moon, NAIFId::Earth, epc).unwrap();
    let r2 = spk_position(301, 399, epc).unwrap(); // equivalent raw NAIF IDs
    ```

Any NAIF ID present in a loaded kernel but not named by the enum (e.g. a
minor body or spacecraft) is passed as a raw integer (Python) or
`NAIFId::Id(...)` (Rust). Full ID listing: [NAIF Integer ID Codes](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html).

### Kernel-Scoped Queries

`spk_position_from_kernel`, `spk_velocity_from_kernel`, and `spk_state_from_kernel`
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
(`NAIFId.EARTH` / `NAIFId::Earth`, ID 399). They take an `Epoch` and an
ephemeris source (`EphemerisSource.DE440s` / `EphemerisSource.DE440` in
Python, `EphemerisSource::DE440s` / `EphemerisSource::DE440` in Rust)
selecting which DE kernel to query. Computing the state shares a single
record lookup between position and velocity, so prefer `*_state_de` over
separate position/velocity calls when both are needed.

The Mars/Jupiter/Saturn/Uranus/Neptune functions return the planet **body
center**. The DE kernel only carries the planetary-system barycenter for these
outer planets, so the body center is computed as a two-leg sum: the
barycenter relative to Earth from the DE kernel, plus the body center relative
to the barycenter from the planet's satellite-system kernel. That satellite
kernel is auto-downloaded and loaded on first use (mar099s ~64 MB, jup365
~1.1 GB, sat441 ~631 MB, ura184 ~387 MB, nep097 ~100 MB).

Each of the five outer planets also has `mars_barycenter_position_de`,
`jupiter_barycenter_position_de`, and so on (with `_velocity_de` / `_state_de`
counterparts) that return the planetary-system barycenter using **only** the
DE kernel — no satellite-kernel download. The barycenter and body center
differ by up to a few hundred km due to large moons (e.g. ~290 km for Saturn
from Titan, ~230 km for Jupiter from the Galilean moons; only ~0.2 m for Mars).
Prefer the `_barycenter_` variants for third-body force modeling, which uses
the system barycenter with the system GM. Sun, Moon, Mercury, and Venus have
no significant satellites, so their body-center and barycenter positions
coincide and no barycenter variant is provided.

## PCK Orientation

PCK queries take a frame class ID — either a raw NAIF frame class ID or a
`FrameId` enum value. The lunar principal-axis frame from the
`moon_pa_de440` kernel is registered under frame class ID `31008`
(`FrameId.MOON_PA_DE440` / `FrameId::MoonPaDe440`).

| Function | Returns |
|---|---|
| `pck_euler_angles(frame_id, epc)` | `(angles, rates)` as raw arrays: `[phi, delta, w]` (rad) and their rates (rad/s) |
| `pck_euler_angle(frame_id, epc)` | `EulerAngle` (order ZXZ, radians) |
| `pck_euler_rates(frame_id, epc)` | `[phi_dot, delta_dot, w_dot]` (rad/s) as a raw array |
| `pck_euler_angle_and_rates(frame_id, epc)` | `(EulerAngle, rates)` from a single shared segment lookup |
| `pck_quaternion(frame_id, epc)` | `Quaternion` (unit quaternion, ICRF to body-fixed) |
| `pck_rotation_matrix(frame_id, epc)` | `RotationMatrix` (ICRF to body-fixed) |

`pck_euler_angles` (the original, tuple-of-arrays form) is unchanged and
still available alongside the typed functions. `pck_rotation_matrix`
returns a `RotationMatrix` object rather than a raw array — index it
directly (`r[(0, 0)]`) or call `.to_matrix()` for a numpy array in Python;
in Rust call `.to_matrix()` to get an `SMatrix3<f64>` indexable as
`r.to_matrix()[(0, 0)]`. See the
[EulerAngle](../../library_api/attitude/euler_angles.md),
[Quaternion](../../library_api/attitude/quaternion.md), and
[RotationMatrix](../../library_api/attitude/rotation_matrix.md) references
for the full type APIs.

The example below uses the original `pck_euler_angles`/`pck_rotation_matrix`
pair:

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
- [NAIF Integer ID Codes](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/naif_ids.html) - Full NAIF body ID reference
- [NAIF DAF Required Reading](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/daf.html) - Double Precision Array File format underlying SPK and PCK
- [NAIF SPK Required Reading](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/spk.html) - SPK ephemeris kernel format and data types
- [NAIF PCK Required Reading](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/pck.html) - Binary PCK orientation kernel format
- [NAIF Frames Required Reading](https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/frames.html) - Reference frame definitions and the "J2000"/ICRF naming
