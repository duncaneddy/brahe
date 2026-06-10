# ICGEM Functions

Functions and types for discovering and downloading spherical harmonic gravity models from the [International Centre for Global Earth Models (ICGEM)](https://icgem.gfz.de).

All functions are available via `brahe.datasets.icgem.<function_name>`.

!!! tip "Loading a downloaded model into a propagator"
    To use an ICGEM model as the central-body field in a numerical propagator,
    build a `GravityModelType` from the same `(body, name)` pair via
    [`GravityModelType.icgem`](../orbit_dynamics/gravity.md#gravity-model-type).
    The download and cache happen transparently on first use.

## list_models

::: brahe._brahe.icgem_list_models

## download_model

::: brahe._brahe.icgem_download_model

## refresh_index

::: brahe._brahe.icgem_refresh_index

## refresh_all_indexes

::: brahe._brahe.icgem_refresh_all_indexes

## ICGEMIndexEntry

::: brahe._brahe.ICGEMIndexEntry
    options:
      show_root_heading: true
      show_root_full_path: false

---

## See Also

- [ICGEM Gravity Models (Learn)](../../learn/datasets/icgem.md) - Overview, caching, and usage guide
- [Gravity Models (Learn)](../../learn/orbital_dynamics/gravity.md) - Geopotential theory and Brahe's gravity API
- [Force Models (Learn)](../../learn/orbit_propagation/numerical_propagation/force_models.md) - Wiring an ICGEM model into a numerical propagator
- [ICGEM Website](https://icgem.gfz.de) - Official ICGEM model catalog
