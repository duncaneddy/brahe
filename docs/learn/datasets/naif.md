# NAIF Ephemeris Kernels

[NAIF (Navigation and Ancillary Information Facility)](https://naif.jpl.nasa.gov/) is NASA JPL's archive for planetary ephemeris data. Brahe provides functions to download DE (Development Ephemeris), satellite-system, and lunar-orientation kernels from the NAIF archive.

!!! info "What are DE Kernels?"
    DE kernels are binary SPK (SPICE Kernel) files containing numerical integration results for planetary positions and velocities. Each version represents a different JPL Development Ephemeris model, with newer versions incorporating improved observations and models.

## Supported Kernels

Brahe supports downloading the following kernel files, enumerated by
`NAIFKernel` (Rust). The Python `bh.datasets.naif.download_de_kernel(name, ...)`
function and the Rust `brahe::datasets::naif::download_kernel(kernel, ...)`
function both accept any of these — by name (`str`) in Python, by
`NAIFKernel` variant in Rust:

<div class="center-table" markdown="1">
| Kernel    | File Size | Description |
|-----------|-----------|-------------|
| `de430`   | ~114 MB   | Standard precision, extended time span |
| `de432s`  | ~11 MB    | Designed for New Horizons targeting Pluto |
| `de435`   | ~114 MB   | Higher accuracy for inner planets |
| `de438`   | ~114 MB   | Standard precision |
| `de440`   | ~114 MB   | Latest standard precision (1550-2650) |
| `de440s`  | ~31 MB    | Small variant of DE440 (1849-2150) |
| `de442`   | ~114 MB   | Intended for the MESSENGER mission to Mercury |
| `de442s`  | ~31 MB    | Small variant of DE442 |
| `mar099s` | ~68 MB    | Mars satellite ephemeris — Phobos, Deimos (1995-2050) |
| `mar099`  | ~1.1 GB   | Mars satellite ephemeris, wider time span (1600-2600) |
| `jup365`  | ~1.1 GB   | Jupiter satellite ephemeris — Io, Europa, Ganymede, Callisto (1600-2200) |
| `sat441`  | ~662 MB   | Saturn satellite ephemeris — Titan and mid-size moons (1750-2250) |
| `ura184`  | ~387 MB   | Uranus satellite ephemeris — Miranda, Ariel, Umbriel, Titania, Oberon (1600-2399) |
| `nep097`  | ~105 MB   | Neptune satellite ephemeris — Triton (1600-2400) |
| `plu060`  | ~135 MB   | Pluto-system ephemeris — Charon (1800-2200) |
| `moon_pa_de440` | ~13 MB | Lunar principal-axis binary PCK (orientation, not SPK) |
</div>

!!! tip "Choosing a Kernel"
    For most applications, `de440s` provides a good balance between file size and accuracy. The "s" (small) variants cover a shorter time span but are significantly smaller files. The satellite-system kernels (`mar099s`, `jup365`, `sat441`, `ura184`, `nep097`) are downloaded automatically the first time a planet body-center `*_de` function (e.g. `mars_position_de`) is called — see [Ephemerides](../../library_api/orbit_dynamics/ephemerides.md).

!!! info "Binary PCK Kernels"
    Brahe also downloads and caches the `moon_pa_de440` binary PCK (lunar
    principal-axis orientation) the same way. Load it with
    `bh.load_kernel("moon_pa_de440")` — see [SPICE Kernels](../spice/index.md)
    for orientation queries against it.

## Caching Behavior

DE kernels are large files that do not change over time. Brahe implements permanent caching:

- **Cache location**: `~/.cache/brahe/naif/` (or `$BRAHE_CACHE/naif/` if set)
- **Cache duration**: Permanent (kernels are stable long-term products)
- **Cache check**: Simple file existence check - no age validation

Once a kernel is downloaded, it remains cached indefinitely and subsequent calls return the cached file path without re-downloading.

## Usage

### Basic Download

Download a kernel and use the cached location:

=== "Python"

    ``` python
    --8<-- "./examples/datasets/naif_download_kernel.py:12"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/datasets/naif_download_kernel.rs:9"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/datasets/naif_download_kernel.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/datasets/naif_download_kernel.rs.txt"
        ```

The first call downloads and caches the kernel. Subsequent calls immediately return the cached file path.

## Error Handling

The Python binding takes a kernel name as a string and validates it before attempting a download; invalid names raise an error immediately:

=== "Python"

    ``` python
    try:
        bh.datasets.naif.download_de_kernel("de999")
    except RuntimeError as e:
        print(e)
        # "Unsupported kernel name 'de999'. Supported kernels: de430, de432s, ..."
    ```

In Rust, `download_kernel` takes a `NAIFKernel` enum value directly rather
than a string, so an unsupported kernel name cannot be passed at all — the
enum only has the 16 valid variants. To validate a name obtained at runtime
(e.g. from user input), resolve it through `NAIFKernel::from_name` first:

=== "Rust"

    ``` rust
    match bh::spice::NAIFKernel::from_name("de999") {
        Some(kernel) => {
            let path = bh::datasets::naif::download_kernel(kernel, None).unwrap();
            println!("Downloaded to {}", path.display());
        }
        None => println!("Unsupported kernel name 'de999'"),
    }
    ```

## See Also

- [SPICE Kernels](../spice/index.md) - Loading, querying, and PCK orientation
- [NAIF SPICE System](https://naif.jpl.nasa.gov/naif/toolkit.html) - Full SPICE toolkit documentation
- [DE Kernel Details](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/aa_summaries.txt) - Detailed descriptions of each DE version
- [Library API Reference](../../library_api/datasets/naif.md) - Complete function documentation
