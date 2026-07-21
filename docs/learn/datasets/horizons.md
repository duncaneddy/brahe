# JPL Horizons SPK Generation

The `brahe.datasets.horizons` module generates, caches, and loads targeted SPK
(`.bsp`) kernels for small bodies without packaged DE-kernel coverage via the
[JPL Horizons API](https://ssd-api.jpl.nasa.gov/doc/horizons.html). This
unblocks third-body and solar radiation pressure perturbations for bodies such
as Ceres, which are not contained in the DE kernels and therefore must be
loaded separately.

`HorizonsClient.get_spk(request)` returns a `HorizonsSPKResponse`. The `.bsp`
is cached under `$BRAHE_CACHE/horizons` (default `~/.cache/brahe/horizons`) and
reused whenever a kernel for the same request already exists, so a given
body/span is downloaded only once.

## Requesting and Loading an SPK

=== "Python"

    ``` python
    --8<-- "./examples/datasets/horizons_spk.py:13"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/datasets/horizons_spk.rs:10"
    ```

## Request and Center

`HorizonsSPKRequest.for_spkid(spkid, start, stop)` targets a small body by SPK
ID; `HorizonsSPKRequest(command, start, stop)` accepts a raw Horizons `COMMAND`.
`with_center(center)` sets the requested ephemeris center (default `"500@0"`).
In practice Horizons returns small-body SPKs centered on the Sun (NAIF 10) in
the J2000/ICRF frame; loading `de440s` alongside supplies the Sun's position
relative to the Solar System barycenter, so the two kernels chain and positions
of the Sun and planets relative to the small body resolve for third-body and
SRP forces.

## Response

`HorizonsSPKResponse` exposes `.path` (cached `.bsp` path), `.spk_file_id`, and
methods `.bytes()` (raw kernel bytes) and `.load()` (load into the SPICE
registry).

## See Also

- [SBDB Lookup (Learn)](sbdb.md) - Resolve a name to the NAIF ID this request needs
- [Horizons SPK (Library API)](../../library_api/datasets/horizons.md) - Full class reference
- [Propagation Around Other Central Bodies (Learn)](../orbit_propagation/numerical_propagation/other_central_bodies.md) - Using a custom body with perturbations
- [Dawn at Ceres (Example)](../../examples/dawn_ceres_orbit.md) - End-to-end workflow
