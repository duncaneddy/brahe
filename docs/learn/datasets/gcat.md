# GCAT Satellite Catalogs

[GCAT (General Catalog of Artificial Space Objects)](https://planet4589.org/space/gcat/) is Jonathan McDowell's comprehensive catalog of all known artificial objects in space. Brahe provides functions to download and query two GCAT catalogs: SATCAT (satellite catalog) and PSATCAT (payload satellite catalog), with automatic file-based caching.

!!! info "What is GCAT?"
    GCAT is an independent catalog maintained by astrophysicist Jonathan McDowell at the Harvard-Smithsonian Center for Astrophysics. It provides detailed metadata for every cataloged space object, including physical dimensions, orbital parameters, ownership, and mission details. Unlike the US Space Command catalog (which focuses on tracking), GCAT emphasizes comprehensive metadata about each object's identity and purpose.

## Available Catalogs

### SATCAT

The SATCAT catalog contains physical, orbital, and administrative metadata for all cataloged artificial space objects. Each record includes 42 fields organized into several categories:

<div class="center-table" markdown="1">

| Category | Fields | Description |
|----------|--------|-------------|
| **Identification** | `jcat`, `satcat`, `launch_tag`, `piece`, `name`, `pl_name`, `alt_names` | Catalog IDs, designations, and names |
| **Classification** | `object_type`, `status`, `dest`, `op_orbit`, `oqual` | Object type (P=payload, R=rocket body), status (O=operational, D=decayed), orbit class |
| **Physical** | `mass`, `dry_mass`, `tot_mass`, `length`, `diameter`, `span`, `shape` | Dimensions and mass properties (kg, meters) |
| **Orbital** | `perigee`, `apogee`, `inc`, `odate` | Perigee/apogee altitude (km), inclination (degrees) |
| **Administrative** | `owner`, `state`, `manufacturer`, `bus`, `motor` | Owner, country, manufacturer, spacecraft bus |
| **Timeline** | `ldate`, `sdate`, `ddate`, `parent`, `primary` | Launch, separation, and decay dates |

</div>

### PSATCAT

The PSATCAT catalog contains payload-specific metadata for missions, extending the SATCAT with operational and registry information. Each record includes 28 fields:

<div class="center-table" markdown="1">

| Category | Fields | Description |
|----------|--------|-------------|
| **Mission** | `program`, `class` (Python: `class_`), `category`, `discipline`, `result` | Program name, mission class/category, outcome |
| **Operations** | `top`, `tdate`, `tlast`, `tf`, `att`, `mvr`, `control` | Operational dates, attitude control, maneuver capability |
| **UN Registry** | `un_state`, `un_reg`, `un_period`, `un_perigee`, `un_apogee`, `un_inc` | UN registration details and registered orbital parameters |
| **Disposal** | `disp_epoch`, `disp_peri`, `disp_apo`, `disp_inc` | End-of-life orbit parameters |

</div>

## Caching Behavior

GCAT data is updated regularly as new objects are cataloged. Brahe implements time-based file caching:

- **Cache location**: `~/.cache/brahe/gcat/` (or `$BRAHE_CACHE/gcat/` if set)
- **Default TTL**: 24 hours (86400 seconds)
- **Force refresh**: Pass `cache_max_age=0` to bypass the cache and download fresh data

Once downloaded, the TSV files are cached locally. Subsequent calls within the TTL window return the cached data without a network request.

## Usage

### Downloading Catalogs

Download the SATCAT catalog and look up records by SATCAT number or JCAT identifier:

=== "Python"

    ``` python
    --8<-- "./examples/datasets/gcat_get_satcat.py:12"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/datasets/gcat_get_satcat.rs:8"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/datasets/gcat_get_satcat.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/datasets/gcat_get_satcat.rs.txt"
        ```

### Searching and Filtering

Use name search and filter chaining to narrow down the catalog:

=== "Python"

    ``` python
    --8<-- "./examples/datasets/gcat_search_filter.py:12"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/datasets/gcat_search_filter.rs:8"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/datasets/gcat_search_filter.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/datasets/gcat_search_filter.rs.txt"
        ```

All filter methods return new catalog instances (immutable pattern), so the original catalog is never modified. This enables chaining multiple filters to progressively narrow results.

### Payload Catalog (PSATCAT)

Download the PSATCAT catalog and use payload-specific filters:

=== "Python"

    ``` python
    --8<-- "./examples/datasets/gcat_psatcat.py:12"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/datasets/gcat_psatcat.rs:8"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/datasets/gcat_psatcat.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/datasets/gcat_psatcat.rs.txt"
        ```

### DataFrame Export

Both catalogs support conversion to [Polars](https://pola.rs/) DataFrames for analysis. In Python, `to_dataframe()` returns a `polars.DataFrame`; in Rust, it returns a `polars::DataFrame`:

```python
import brahe as bh

satcat = bh.datasets.gcat.get_satcat()

# Convert to Polars DataFrame
df = satcat.to_dataframe()
print(df.shape)       # (rows, columns)
print(df.columns[:5]) # ['jcat', 'satcat', 'launch_tag', 'piece', 'object_type']

# Use Polars operations for analysis
operational = df.filter(df["status"] == "O")
print(f"Operational objects: {operational.shape[0]}")
```

## Field Code Reference

Many GCAT fields use abbreviated codes. The tables below document the most common values. For full definitions, see the [GCAT column definitions](https://planet4589.org/space/gcat/web/cat/cols.html).

### SATCAT Codes

**Object Type** (`object_type`):

| Code | Meaning |
|------|---------|
| `P` | Payload |
| `R` | Rocket body |
| `D` | Debris |
| `C` | Component |

**Status** (`status`) — see [GCAT phases](https://planet4589.org/space/gcat/web/intro/phases.html) for full list:

| Code | Meaning |
|------|---------|
| `O` | Operational (in orbit) |
| `D` | Decayed (re-entered) |
| `L` | Landed (on surface) |
| `AR` | Attached/Recovered |

**JCAT Prefix**:

| Prefix | Meaning |
|--------|---------|
| `S` | Standard catalog |
| `A` | Auxiliary catalog |
| `D` | Deep space |
| `F` | Failed to orbit |

### PSATCAT Codes

**Mission Class** (`class` / Python: `class_`):

| Code | Meaning |
|------|---------|
| `A` | Amateur / academic / non-profit |
| `B` | Business (commercial) |
| `C` | Civil (government, non-military) |
| `D` | Defense (military / intelligence) |

Two-letter codes (e.g. `BD`, `CD`) indicate shared management across categories.

**Mission Category** (`category`) — see [GCAT payload categories](https://planet4589.org/space/gcat/web/cat/pcols.html) for full list:

| Code | Meaning |
|------|---------|
| `COM` | Communications |
| `IMG` | Imaging (optical) |
| `IMG-R` | Imaging (radar) |
| `NAV` | Navigation |
| `MET` | Meteorology |
| `SCI` | Science |
| `TECH` | Technology demonstration |
| `AST` | Astronomy |
| `SS` | Space station / crewed spaceflight |
| `CAL` | Calibration |
| `SIG` | Signals intelligence |
| `EW` | Early warning |
| `EOSCI` | Earth observation science |
| `GEOD` | Geodesy |

**Result** (`result`):

| Code | Meaning |
|------|---------|
| `S` | Success |
| `F` | Failure |
| `U` | Unknown |

**Date Conventions**: A `tdate` value of `*` means "still active" (no end-of-operations date).

---

## See Also

- [GCAT API Reference](../../library_api/datasets/gcat.md) - Complete function and class documentation
- [GCAT Website](https://planet4589.org/space/gcat/) - Jonathan McDowell's catalog home page
- [Datasets Overview](index.md) - Understanding datasets in Brahe
- [CelesTrak Data Source](../ephemeris/celestrak.md) - Alternative satellite catalog from CelesTrak
