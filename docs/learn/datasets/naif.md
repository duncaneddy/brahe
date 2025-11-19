# NAIF Ephemeris Kernels

[NAIF (Navigation and Ancillary Information Facility)](https://naif.jpl.nasa.gov/) is NASA JPL's archive for planetary ephemeris data. Brahe provides functions to download DE (Development Ephemeris) kernels, which contain high-precision position and velocity data for solar system bodies.

!!! info "What are DE Kernels?"
    DE kernels are binary SPK (SPICE Kernel) files containing numerical integration results for planetary positions and velocities. Each version represents a different JPL Development Ephemeris model, with newer versions incorporating improved observations and models.

## Supported Kernels

Brahe supports downloading the following DE kernel files:


<div class="center-table" markdown="1">
| Kernel    | File Size | Description |
|-----------|-----------|-------------|
| `de430`   | ~114 MB   | Standard precision, extended time span |
| `de432s`  | ~32 MB    | Designed for New Horizons Targeting Pluto |
| `de435`   | ~114 MB   | Higher accuracy for inner planets |
| `de440`   | ~114 MB   | Latest standard precision |
| `de440s`  | ~33 MB    | Latest small variant of DE440 |
| `de442`   | ~114 MB   | Intendended for MESSENGER mission to Mercury |
| `de442s`  | ~33 MB    | Small variant of DE442 |
</div>

!!! tip "Choosing a Kernel"
    For most applications, `de440s` provides a good balance between file size and accuracy. The "s" (small) variants cover a shorter time span but are significantly smaller files.

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

The first call downloads and caches the kernel. Subsequent calls immediately return the cached file path.

## Error Handling

The function validates kernel names before attempting downloads. Invalid kernel names raise an error immediately:

=== "Python"

    ``` python
    try:
        bh.datasets.naif.download_de_kernel("de999")
    except RuntimeError as e:
        print(e)
        # "Unsupported kernel name 'de999'. Supported kernels: de430, de432s, ..."
    ```

=== "Rust"

    ``` rust
    match bh::datasets::naif::download_de_kernel("de999", None) {
        Ok(_) => println!("Success"),
        Err(e) => println!("Error: {}", e),
        // Error: Unsupported kernel name 'de999'. Supported kernels: de430, de432s, ...
    }
    ```

## See Also

- [NAIF SPICE System](https://naif.jpl.nasa.gov/naif/toolkit.html) - Full SPICE toolkit documentation
- [DE Kernel Details](https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/aa_summaries.txt) - Detailed descriptions of each DE version
- [Library API Reference](../../library_api/datasets/naif.md) - Complete function documentation
