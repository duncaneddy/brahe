# Star Catalogs

Brahe provides three fixed-epoch star catalogs - **FK5**, **Hipparcos**, and **Tycho-2** - for reference-frame realization and star-based attitude determination. Unlike other datasets, star catalogs are static: once published they are not expected to change, so cached copies never go stale by default.

For complete API details, see the [Star Catalogs API Reference](../../library_api/datasets/star_catalogs.md).

## Available Catalogs

### FK5

The Fifth Fundamental Catalogue (FK5) is a fixed catalog of **1,535** bright fundamental stars at epoch **J2000.0**. It is the classical realization of the mean equator/equinox reference system that preceded ICRS-based catalogs.

### Hipparcos

The Hipparcos Catalogue is a fixed astrometric catalog of **~118,000** stars derived from the ESA Hipparcos satellite mission, referred to the **ICRS at epoch J1991.25**. The `hip_main` source file does not carry a radial velocity column, so [`StarRecord::radial_velocity`](../../library_api/datasets/star_catalogs.md) always returns `None`/`null` for Hipparcos records.

### Tycho-2

The Tycho-2 Catalogue is a fixed astrometric catalog of **~2.54 million** stars derived from the Hipparcos satellite's star mapper data, referred to the **ICRS**. Tycho-2 does not carry a parallax or radial velocity column.

!!! warning "Tycho-2 download size"
    The Tycho-2 source file is large (**~526 MB**, ~2.54 million records), so the first call to `get_tycho2()` may take some time. Subsequent calls use the cached copy.

A small fraction of Tycho-2 entries (`pflag == "X"`) have no mean astrometric solution: their `ra`/`dec`/`pm_ra`/`pm_dec`/`epoch_ra`/`epoch_dec` fields are all missing. For these records, the catalog's derived quantities (`id`, `name`, `unit_vector`, `radec_at_epoch`) fall back to the always-present **observed position** (`ra_observed`/`dec_observed`, epoch ~1991.5) instead.

## Caching Behavior

Star catalog data is downloaded from `https://www.simplespacedata.org/star_catalog/cds` with file-based caching:

- **Cache location**: `~/.cache/brahe/star_catalogs/` (or `$BRAHE_CACHE/star_catalogs/` if set)
- **Default TTL**: none - the cached copy **never goes stale**, since published star catalogs are not expected to change
- **Force refresh**: Pass `cache_max_age=0` (Python) / `Some(0.0)` (Rust) to bypass the cache and download fresh data

This differs from GCAT's 24-hour TTL: FK5, Hipparcos, and Tycho-2 are fixed, one-time publications, so there is no staleness to guard against by default.

## Usage

### Downloading Catalogs

Download the Hipparcos catalog, filter to naked-eye-bright stars, and inspect the result:

=== "Python"

    ``` python
    --8<-- "./examples/datasets/hipparcos_catalog.py:12"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/datasets/hipparcos_catalog.rs:8"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/datasets/hipparcos_catalog.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/datasets/hipparcos_catalog.rs.txt"
        ```

### Filtering

Every catalog supports lookup by identifier, magnitude filtering, and cone-search filtering. Filter methods return a new catalog instance (immutable pattern), so the original catalog is never modified and filters can be chained. The example below downloads FK5, looks up a star by its running number, then filters by magnitude and cone search, chaining both:

=== "Python"

    ``` python
    --8<-- "./examples/datasets/fk5_catalog.py:10"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/datasets/fk5_catalog.rs:6"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/datasets/fk5_catalog.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/datasets/fk5_catalog.rs.txt"
        ```

### DataFrame Export

All three catalogs support conversion to [Polars](https://pola.rs/) DataFrames for analysis. In Python, `to_dataframe()` returns a `polars.DataFrame`; in Rust, it returns a `Result<polars::DataFrame, BraheError>`:

=== "Python"

    ``` python
    --8<-- "./examples/datasets/star_catalogs_dataframe.py:10"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/datasets/star_catalogs_dataframe.rs:6"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/datasets/star_catalogs_dataframe.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/datasets/star_catalogs_dataframe.rs.txt"
        ```

### Proper Motion

Catalog positions are only valid at the catalog's reference epoch (J2000.0 for FK5, J1991.25 for Hipparcos, J2000.0 for Tycho-2). Every record exposes `radec_at_epoch` to propagate its position to a different epoch using proper motion (and parallax/radial velocity, when known). The example below filters Hipparcos by magnitude and cone search to locate Sirius, then propagates its catalog position forward to J2030.0:

=== "Python"

    ``` python
    --8<-- "./examples/datasets/star_catalogs_filtering.py:10"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/datasets/star_catalogs_filtering.rs:7"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/datasets/star_catalogs_filtering.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/datasets/star_catalogs_filtering.rs.txt"
        ```

This uses the same proper-motion transformation as [`apply_proper_motion`](../../library_api/coordinates/radec.md) - IAU SOFA's `iauPmsafe` space-motion routine - see [RA/Dec Transformations](../coordinates/radec_transformations.md#proper-motion) for details.

## Reference Frames

Catalog positions are returned in the catalog's **native reference frame**, exactly as published. Brahe applies no frame correction: `ra`/`dec`, `unit_vector`, and `radec_at_epoch` all return raw catalog-frame values. Which frame that is differs by catalog, and this matters when catalog directions are combined with a GCRF orbit state or fed into the IAU 2006/2000A (CIO-based) [frame transformations](../../library_api/frames/index.md).

<div class="center-table" markdown="1">

| Catalog | Native frame | Relation to Brahe's GCRF/ECI |
|---|---|---|
| Hipparcos | ICRS | Same axes (aligned to ~10 µas); no rotation needed |
| Tycho-2 | ICRS | Same axes; no rotation needed |
| FK5 | Mean equator/equinox of J2000.0 (EME2000) | Offset by the ~23 mas frame bias; rotate EME2000 → GCRF |

</div>

!!! tip "ICRS catalogs are GCRF-ready; FK5 is not"
    The "epoch" in a catalog's description refers to two different things depending on the catalog. For Hipparcos, **ICRS at epoch J1991.25** means the ICRS *axes* (which are fixed and epoch-independent, tied to extragalactic radio sources) with each star's *position* measured at J1991.25. Because Brahe's GCRF is aligned to the ICRS by construction, Hipparcos and Tycho-2 directions are already GCRF directions - the only step needed to use them at another date is proper-motion propagation via `radec_at_epoch`.

    For **FK5 at epoch J2000.0**, J2000.0 pins both the position epoch *and* the axes: FK5 realizes the dynamical mean equator and equinox of J2000.0, which is the same frame Brahe exposes as EME2000, **not** the ICRS/GCRF axes. FK5 directions therefore differ from GCRF by the ~23 mas frame bias and must be rotated with [`position_eme2000_to_gcrf`](../../library_api/frames/eme2000_gcrf.md) (or [`rotation_eme2000_to_gcrf`](../../library_api/frames/eme2000_gcrf.md)) before they are mixed with a GCRF state or passed to the GCRF → ITRF transform. (To FK5's own systematic accuracy this identification is exact; it does not carry FK5's residual zonal/equinox errors.)

The example below propagates an FK5 star to an observation epoch, rotates the resulting direction from EME2000 into GCRF, and then into the Earth-fixed ITRF frame with the IAU 2006/2000A transform - the full catalog → GCRF → ITRF chain:

=== "Python"

    ``` python
    --8<-- "./examples/datasets/fk5_frame_correction.py:17"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/datasets/fk5_frame_correction.rs:13"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/datasets/fk5_frame_correction.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/datasets/fk5_frame_correction.rs.txt"
        ```

## Field Reference

### FK5Record

<div class="center-table" markdown="1">

| Field | Units | Description |
|-------|-------|--------------|
| `fk5_id` | - | FK5 catalog running number |
| `ra` | deg | Right ascension, J2000.0 |
| `dec` | deg | Declination, J2000.0 |
| `pm_ra` | mas/yr | Proper motion in right ascension ($\mu_{\alpha*} = \mu_\alpha \cos\delta$), J2000.0 |
| `pm_dec` | mas/yr | Proper motion in declination, J2000.0 |
| `epoch_ra_1900` | yr | Mean epoch of right ascension observations, minus 1900 (optional) |
| `epoch_dec_1900` | yr | Mean epoch of declination observations, minus 1900 (optional) |
| `vmag` | mag | Visual magnitude (optional) |
| `vmag_flag` | - | Visual magnitude quality/note flag (optional) |
| `spectral_type` | - | Spectral type (optional) |
| `parallax` | mas | Trigonometric parallax (optional) |
| `radial_velocity` | km/s | Radial velocity (optional) |
| `hd_id` | - | Henry Draper (HD) catalog identifier (optional) |
| `dm_id` | - | Durchmusterung (DM) catalog identifier (optional) |
| `gc_id` | - | Groombridge Catalogue (GC) identifier (optional) |

</div>

### HipparcosRecord

<div class="center-table" markdown="1">

| Field | Units | Description |
|-------|-------|--------------|
| `hip_id` | - | Hipparcos catalog identifier |
| `vmag` | mag | Visual magnitude (optional) |
| `var_flag` | - | Magnitude uncertainty/variability flag (optional) |
| `ra` | deg | Right ascension, ICRS, epoch J1991.25 |
| `dec` | deg | Declination, ICRS, epoch J1991.25 |
| `parallax` | mas | Trigonometric parallax (optional) |
| `pm_ra` | mas/yr | Proper motion in right ascension ($\mu_{\alpha*} = \mu_\alpha \cos\delta$), ICRS (optional) |
| `pm_dec` | mas/yr | Proper motion in declination, ICRS (optional) |
| `e_ra`, `e_dec` | mas | Standard error in right ascension/declination (optional) |
| `e_parallax` | mas | Standard error in parallax (optional) |
| `e_pm_ra`, `e_pm_dec` | mas/yr | Standard error in proper motion (optional) |
| `bt_mag`, `vt_mag` | mag | Mean Tycho BT/VT magnitude (optional) |
| `b_v` | mag | Johnson B-V colour (optional) |
| `hp_mag` | mag | Hipparcos-system magnitude (optional) |
| `hvar_type` | - | Variability type flag (optional) |
| `mult_flag` | - | Double/multiple system flag (optional) |
| `hd_id` | - | Henry Draper (HD) catalog identifier (optional) |
| `bd_id`, `cod_id`, `cpd_id` | - | Raw BD/CoD/CPD Durchmusterung identifiers; see `name()` for the expanded form (optional) |
| `spectral_type` | - | Spectral type (optional) |

</div>

Hipparcos records have no `radial_velocity` field/column: `StarRecord::radial_velocity` always returns `None`/`null`.

### Tycho2Record

<div class="center-table" markdown="1">

| Field | Units | Description |
|-------|-------|--------------|
| `tyc1`, `tyc2`, `tyc3` | - | Tycho-2 identifier triple (GSC region, running number, component) |
| `pflag` | - | Mean position flag: blank for a normal entry, `"P"` for a photocenter solution, `"X"` for no mean position (optional) |
| `ra`, `dec` | deg | Mean right ascension/declination, ICRS; `None`/`null` when `pflag == "X"` (optional) |
| `pm_ra` | mas/yr | Proper motion in right ascension ($\mu_{\alpha*} = \mu_\alpha \cos\delta$) (optional) |
| `pm_dec` | mas/yr | Proper motion in declination (optional) |
| `epoch_ra`, `epoch_dec` | yr | Mean epoch of the right ascension/declination (optional) |
| `bt_mag`, `vt_mag` | mag | Tycho-2 BT (blue)/VT (visual) magnitude (optional) |
| `vmag` | mag | Johnson V-band approximation (optional; see below) |
| `tycho1_flag` | - | Set (`"T"`) if this entry also has a Tycho-1 record (optional) |
| `hip_id` | - | Hipparcos catalog identifier, if this star is also in Hipparcos (optional) |
| `ra_observed`, `dec_observed` | deg | Observed right ascension/declination, epoch ~1991.5; **always present**, even when `ra`/`dec` are missing |

</div>

Tycho-2 has no `parallax` or `radial_velocity` column: both always return `None`/`null` via `StarRecord`. The `vmag` field is computed from the catalog's BT/VT photometry as $V_T - 0.090\,(B_T - V_T)$ when both are present, falling back to `vt_mag` alone, per the Tycho-2 catalog documentation.

---

## See Also

- [Star Catalogs API Reference](../../library_api/datasets/star_catalogs.md) - Complete function and class documentation
- [RA/Dec Transformations](../coordinates/radec_transformations.md) - Proper motion equations and RA/Dec coordinate conversions
- [Datasets Overview](index.md) - Understanding datasets in Brahe
- [GCAT Satellite Catalogs](gcat.md) - Artificial object catalogs (for comparison with these star catalogs)
