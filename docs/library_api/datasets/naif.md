# NAIF Functions

Functions for downloading planetary DE, satellite ephemeris, and lunar-orientation kernels from NASA JPL's NAIF archive.

All functions are available via `brahe.datasets.naif.<function_name>`.

## download_spice_kernel

::: brahe._brahe.download_spice_kernel

This function accepts any of the 16 kernels brahe knows how to download —
the eight planetary DE kernels, the seven satellite ephemeris kernels, and the
`moon_pa_de440` binary PCK. See [NAIF Ephemeris Kernels](../../learn/datasets/naif.md#supported-kernels)
for the full list. The equivalent Rust function is
`brahe::datasets::naif::download_spice_kernel(kernel: SPICEKernel, output_path: Option<PathBuf>)`,
which takes a typed `SPICEKernel` variant rather than a string.

## See Also

- [NAIF Ephemeris Kernels](../../learn/datasets/naif.md) - Supported kernels, caching behavior, and usage examples
- [SPICE Kernels](../../learn/spice/index.md) - Loading, querying, and PCK orientation
- [NASA NAIF Website](https://naif.jpl.nasa.gov/) - Official NAIF data archive
