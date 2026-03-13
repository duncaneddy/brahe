# Collection Planning with Tessellation

End-to-end example: tessellate Ireland for satellite imaging using the
NISAR SAR satellite (242 km swath width), then compute collection
opportunities over a 7-day period.

NISAR (NASA-ISRO SAR) is an L-band and S-band synthetic aperture radar
satellite in a sun-synchronous orbit with a 242 km swath width. This example
demonstrates the full tessellation workflow — from defining an area of
interest through to finding imaging windows — using NISAR's real orbit
downloaded from CelesTrak.

---

## Setup

Import libraries and initialize Earth orientation parameters:

``` python
--8<-- "./examples/examples/tessellation_visualization.py:preamble"
```

Download the NISAR TLE from CelesTrak:

``` python
--8<-- "./examples/examples/tessellation_visualization.py:download_nisar"
```

## Define Area of Interest

Define an approximate polygon boundary for Ireland:

``` python
--8<-- "./examples/examples/tessellation_visualization.py:define_ireland"
```

<figure markdown="span">
    ![Ireland AOI](../figures/tessellation_ireland_aoi_light.png#only-light)
    ![Ireland AOI](../figures/tessellation_ireland_aoi_dark.png#only-dark)
</figure>

## Tessellate for Collection

Create tiles matching NISAR's 242 km swath width with 500 km along-track
length. Using `AscDsc.EITHER` produces tiles for both ascending and
descending passes:

``` python
--8<-- "./examples/examples/tessellation_visualization.py:tessellate"
```

<figure markdown="span">
    ![Ireland tessellation](../figures/tessellation_ireland_tiles_light.png#only-light)
    ![Ireland tessellation](../figures/tessellation_ireland_tiles_dark.png#only-dark)
</figure>

## Compute Collection Opportunities

Use an off-nadir constraint (10°–45°) to find all collection windows over a
7-day period:

``` python
--8<-- "./examples/examples/tessellation_visualization.py:compute_accesses"
```

``` python
--8<-- "./examples/examples/tessellation_visualization.py:results"
```

## Full Code Example

```python title="tessellation_visualization.py"
--8<-- "./examples/examples/tessellation_visualization.py:all"
```

---

## See Also

- [Learn: Tessellation](../learn/access_computation/tessellation.md) - Conceptual overview and configuration guide
- [Access Computation](../learn/access_computation/index.md) - Full access computation system
- [Imaging Data Latency](imaging_data_latency.md) - Related end-to-end analysis with Capella
- [API Reference: Tessellation](../library_api/access/tessellation.md)
