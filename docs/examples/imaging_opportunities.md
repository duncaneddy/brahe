# Earth Observation Imaging Opportunities

In this example we'll find upcoming imaging opportunities for the ICEYE constellation over San Francisco (lon: -122.4194, lat: 37.7749), subject to specific imaging constraints.

---

## Setup

First, we'll import the necessary libraries, initialize Earth orientation parameters, download the latest TLE data for all active spacecraft and filter it to select just the ICEYE spacecraft:

``` python
--8<-- "./examples/examples/imaging_opportunities.py:preamble"
```

We download all active satellites from CelesTrak and filter for ICEYE spacecraft:

``` python
--8<-- "./examples/examples/imaging_opportunities.py:ephemeris_download"
```

## Constellation Visualization

Before getting further into the analysis, it's useful to visualize the 3D geometry of the constellation. We propagate each satellite for one orbital period and create a 3D visualization:

``` python
--8<-- "./examples/examples/imaging_opportunities.py:constellation_propagation"
```

The resulting plot shows the ICEYE constellation orbits in 3D space:

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/imaging_opportunities_constellation_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/imaging_opportunities_constellation_dark.html"  loading="lazy"></iframe>
</div>

## Target Location Definition

We define San Francisco as our imaging target:

``` python
--8<-- "./examples/examples/imaging_opportunities.py:target_definition"
```

## Constraint Specification

In this case, we want to collect a descending-pass, right-looking image collected from between 35 and 45 degrees off-nadir angle. We compose these requirements using Brahe's constraint system:

``` python
--8<-- "./examples/examples/imaging_opportunities.py:constraint_definition"
```

This creates a composite constraint that requires **all three conditions** to be satisfied simultaneously:

- `AscDscConstraint`: Filters for descending passes only
- `LookDirectionConstraint`: Requires right-looking geometry
- `OffNadirConstraint`: Limits imaging angle to 35-45° off-nadir

!!! note "Look direction and pass direction jointly fix the imaged geometry"
    A side-looking SAR images to one side of its ground track, so the
    right-looking [`LookDirectionConstraint`](../learn/access_computation/constraints.md)
    ([API](../library_api/access/constraints.md#brahe.LookDirectionConstraint))
    combined with a descending
    [`AscDscConstraint`](../learn/access_computation/constraints.md)
    ([API](../library_api/access/constraints.md#brahe.AscDscConstraint))
    selects a specific illumination geometry: which side of the track, and
    from which heading, the target is viewed. Composed with an
    [`OffNadirConstraint`](../learn/access_computation/constraints.md)
    ([API](../library_api/access/constraints.md#brahe.OffNadirConstraint))
    into an all-of constraint, a pass is reported only when it satisfies every
    condition at once - relaxing any one changes which opportunities survive,
    not merely how many.

## Compute Collection Opportunities

Now we'll compute all imaging opportunities between the constellation and San Francisco over a 7-day period:

``` python
--8<-- "./examples/examples/imaging_opportunities.py:opportunity_computation"
```

Below is a table of the first 10 imaging opportunities. Click on any column header to sort:

<div class="center-table" markdown="1">
{{ read_csv('figures/imaging_opportunities_windows.csv') }}
</div>

## Full Code Example

??? "Full Code"

    ```python title="imaging_opportunities.py"
    --8<-- "./examples/examples/imaging_opportunities.py:all"
    ```

---

## See Also

- [Access Computation](../learn/access_computation/index.md)
- [Locations](../library_api/access/locations.md)
- [Constraints](../library_api/access/constraints.md)