# Tessellation

Tessellation divides geographic areas of interest (AOIs) into smaller rectangular tiles. These tiles are normally sized to match the sensor field-of-view for Earth-imaging satellites. This enables larger areas, ones too big to be collected in a single imaging action, to be broken down into smaller parts which can be feasibly collected.

There are infinitely many ways to tile a large area if entirely unconstrained in the tile placement. Brahe implements an orbit-geometry based tessalator that generates tiles aligned with the orbital ground-track of a satellite. This approach is particular well-suited to satellites with push-broom imaging modes such as radar imaging satellites. The `OrbitGeometryTessellator` uses a satellite's orbital elements and a reference epoch to determine ground-track directions at any latitude. It then tiles the target location perpendicular and parallel to the ground track. Output tiles are `PolygonLocation` instances with metadata properties describing the tile geometry, making them compatible with the rest of the access computation system. The tesselation configuration should be setup such that the maximum width and length remain feasible to collect in a single imaging pass.

For complete API details, see the [API Reference: Tessellation](../../library_api/access/tessellation.md).

## Configuration

The `OrbitGeometryTessellatorConfig` controls tile dimensions, overlap, and ascending/descending pass selection. All dimensions are in meters.

<div class="center-table" markdown="1">
| Parameter | Default | Description |
|-----------|---------|-------------|
| `image_width` | 5000 m | Cross-track tile width |
| `image_length` | 5000 m | Along-track tile length |
| `crosstrack_overlap` | 200 m | Cross-track overlap between adjacent strips |
| `alongtrack_overlap` | 200 m | Along-track overlap between adjacent tiles |
| `asc_dsc` | Either | Ascending/descending pass selection |
| `min_image_length` | 5000 m | Minimum tile length (tiles shorter than this are discarded) |
| `max_image_length` | 5000 m | Maximum tile length (tiles longer than this are split) |
</div>

=== "Python"

    ``` python
    --8<-- "./examples/access/tessellation/config_creation.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/tessellation/config_creation.rs:5"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/access/tessellation/config_creation.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/access/tessellation/config_creation.rs.txt"
        ```

## Point Tessellation

Tessellating a `PointLocation` creates one tile per pass direction, centered on the point. With `AscDsc.ASCENDING`, a single tile is created; with `AscDsc.EITHER`, up to two tiles are created (one per direction). At high latitudes where ascending and descending ground tracks converge, redundant tiles may be automatically merged.

=== "Python"

    ``` python
    --8<-- "./examples/access/tessellation/point_tessellation.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/tessellation/point_tessellation.rs:5"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/access/tessellation/point_tessellation.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/access/tessellation/point_tessellation.rs.txt"
        ```

The figure below shows the difference between ascending-only and ascending+descending tessellation for a single point near San Francisco. Each tile direction produces a rectangle aligned to the satellite ground track at that latitude.

<figure markdown="span">
    ![Point tessellation](../../figures/tessellation_point_light.png#only-light)
    ![Point tessellation](../../figures/tessellation_point_dark.png#only-dark)
</figure>

??? "Figure Source"

    ``` python
    --8<-- "./plots/learn/access_computation/tessellation_figures.py:point_figure"
    ```

## Polygon Tessellation

Tessellating a `PolygonLocation` divides the area into cross-track strips perpendicular to the satellite ground track, then subdivides each strip along-track into individual tiles. The algorithm handles concave polygons by detecting gaps in the along-track direction. Tiles at polygon edges may have adjusted lengths to fit the boundary.

=== "Python"

    ``` python
    --8<-- "./examples/access/tessellation/polygon_tessellation.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/tessellation/polygon_tessellation.rs:5"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/access/tessellation/polygon_tessellation.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/access/tessellation/polygon_tessellation.rs.txt"
        ```

The figure below shows England tessellated with 50 km tiles. Tiles are colored by `tile_group_id` — each color represents tiles sharing the same ground-track direction (ascending vs descending). The dashed line is the input polygon boundary.

<figure markdown="span">
    ![Polygon tessellation](../../figures/tessellation_polygon_light.png#only-light)
    ![Polygon tessellation](../../figures/tessellation_polygon_dark.png#only-dark)
</figure>

??? "Figure Source"

    ``` python
    --8<-- "./plots/learn/access_computation/tessellation_figures.py:polygon_figure"
    ```

## Effect of Configuration Parameters

### Tile Length

Increasing `image_width` and `image_length` produces fewer, larger tiles. The left panel uses 5 km tiles and the right uses 15 km tiles for the same region near San Francisco.

<figure markdown="span">
    ![Tile length comparison](../../figures/tessellation_tile_length_light.png#only-light)
    ![Tile length comparison](../../figures/tessellation_tile_length_dark.png#only-dark)
</figure>

### Overlap

Increasing `crosstrack_overlap` and `alongtrack_overlap` causes adjacent tiles to share more area, which produces more tiles for the same region. The left panel uses 0 m overlap; the right uses 1000 m overlap.

<figure markdown="span">
    ![Overlap comparison](../../figures/tessellation_overlap_light.png#only-light)
    ![Overlap comparison](../../figures/tessellation_overlap_dark.png#only-dark)
</figure>

??? "Figure Source"

    ``` python
    --8<-- "./plots/learn/access_computation/tessellation_figures.py:config_figures"
    ```

## Tile Metadata Properties

Each output tile is a `PolygonLocation` with metadata properties stored in its `properties` dictionary. These properties describe the tile geometry and ownership.

<div class="center-table" markdown="1">
| Property | Type | Description |
|----------|------|-------------|
| `tile_direction` | `[x, y, z]` | Along-track unit vector in ECEF coordinates |
| `tile_width` | `float` | Cross-track dimension in meters |
| `tile_length` | `float` | Along-track dimension in meters |
| `tile_area` | `float` | Tile area ($\text{width} \times \text{length}$) in m$^2$ |
| `tile_group_id` | `str` | UUID shared by all tiles in the same tiling direction |
| `spacecraft_ids` | `list[str]` | Spacecraft identifiers that can collect this tile |
</div>

=== "Python"

    ``` python
    --8<-- "./examples/access/tessellation/tile_properties.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/tessellation/tile_properties.rs:5"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/access/tessellation/tile_properties.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/access/tessellation/tile_properties.rs.txt"
        ```

## Merging Tiles from Multiple Spacecraft

When multiple spacecraft have similar orbital planes, their ground-track directions at a given latitude will be similar. The `tile_merge_orbit_geometry` function clusters tiles by direction and merges groups whose directions fall within a configurable angular threshold. Rather than creating duplicate tiles, it adds the additional spacecraft's ID to the base tile's `spacecraft_ids` list.

=== "Python"

    ``` python
    --8<-- "./examples/access/tessellation/tile_merging.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/tessellation/tile_merging.rs:5"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/access/tessellation/tile_merging.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/access/tessellation/tile_merging.rs.txt"
        ```

The figure below shows tiles from two spacecraft with slightly different inclinations (~1.4° offset). Before merging, the tiles from SC-1 and SC-2 are visibly offset; after merging with a 2° angular threshold, overlapping tiles are combined with both spacecraft IDs in the `spacecraft_ids` list.

<figure markdown="span">
    ![Merging before/after](../../figures/tessellation_merging_light.png#only-light)
    ![Merging before/after](../../figures/tessellation_merging_dark.png#only-dark)
</figure>

??? "Figure Source"

    ``` python
    --8<-- "./plots/learn/access_computation/tessellation_figures.py:merging_figure"
    ```

---

## See Also

- [Locations](locations.md) - Ground location types used as tessellation inputs
- [Computation](computation.md) - Access algorithms for finding observation windows
- [API Reference: Tessellation](../../library_api/access/tessellation.md)
- [Example: Collection Planning with Tessellation](../../examples/tessellation_visualization.md)
