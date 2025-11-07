# Maximum Communications Gap

In this example we'll analyze the communication gaps between a satellite constellation and supporting ground station network. For this work we'll use the Umbra constellation and 5 KSAT ground stations (Svalbard, Punta Arenas, Hartebeesthoek, Awarua, and Athens).

The maximum contact gap is a significant factor in the reactivity (speed from request to uplink) and latency (time from collection to delivery) for satellite imaging constellations.

---

## Setup

First, we'll import the necessary libraries, initialize Earth orientation parameters, download the latest TLE data for all active spacecraft, and filter to select just the Umbra satellites:

``` python
--8<-- "./examples/examples/max_communications_gap.py:21:32"
```

We download all active satellite TLEs from CelesTrak as propagators and filter for satellites with "UMBRA" in their name:

``` python
--8<-- "./examples/examples/max_communications_gap.py:42:45"
```

Next, we load the 5 specific KSAT ground stations that will support communications:

``` python
--8<-- "./examples/examples/max_communications_gap.py:52:56"
```

## Constellation Visualization

Before getting further into the analysis, it's useful to visualize the 3D geometry of the constellation. We propagate each satellite for one orbit and plot their trajectories:

``` python
--8<-- "./examples/examples/max_communications_gap.py:65:86"
```

The resulting plot shows the complete Umbra constellation orbiting Earth:

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/max_communications_gap_constellation_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/max_communications_gap_constellation_dark.html"  loading="lazy"></iframe>
</div>

## Access Computation

To figure out the contact gaps, we first need to compute all ground contacts over the 7-day propagation window. We reset the propagators and compute access windows with a 5° minimum elevation constraint:

``` python
--8<-- "./examples/examples/max_communications_gap.py:95:110"
```

## Max Gap Computation

Next we'll compute the contact gaps over the course of the simulation. The contact gap is defined as the time between the last contact for a spacecraft and the next contact for that spacecraft. The gap is always computed on a per-spacecraft basis:

``` python
--8<-- "./examples/examples/max_communications_gap.py:118:148"
```

The 10 longest contact gaps are shown below:

<div class="center-table" markdown="1">
{{ read_csv('figures/max_communications_gap_gaps.csv') }}
</div>

The distribution of gaps for the constellation is shown in this histogram:

``` python
--8<-- "./examples/examples/max_communications_gap.py:205:256"
```

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/max_communications_gap_distribution_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/max_communications_gap_distribution_dark.html"  loading="lazy"></iframe>
</div>

To better understand what percentage of gaps fall below a certain duration, we create a cumulative distribution plot. This shows the percentage of gaps that are less than or equal to each duration value:

``` python
--8<-- "./examples/examples/max_communications_gap.py:261:337"
```

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/max_communications_gap_cumulative_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/max_communications_gap_cumulative_dark.html"  loading="lazy"></iframe>
</div>

The cumulative distribution plot includes reference lines at the 25th, 50th, 75th, and 90th percentiles, making it easy to determine what fraction of gaps are below a specific threshold.

## Contact Gap Visualization

Finally, we'll visualize the 3 longest gaps on a ground track plot to see where they occur. For each gap, we extract the satellite's ground track during that time period and plot it as a colored segment. We also interpolate to the ±180° edges to avoid visual gaps at the antimeridian. This type of visualization can be helpful in understanding ground network design and where additional ground stations might help:

``` python
--8<-- "./examples/examples/max_communications_gap.py:343:503"
```

<figure markdown="span">
    ![Maximum Communication Gaps](../figures/max_communications_gap_groundtrack_light.svg#only-light)
    ![Maximum Communication Gaps](../figures/max_communications_gap_groundtrack_dark.svg#only-dark)
</figure>

## Full Code Example

```python title="max_communications_gap.py"
--8<-- "./examples/examples/max_communications_gap.py:21:506"
```

---

## See Also

- [Access Computation](../learn/access_computation/index.md) - Understanding access windows and constraints
- [KSAT Ground Stations](../learn/datasets/groundstations.md) - Ground station dataset documentation
- [CelesTrak Dataset](../learn/datasets/celestrak.md) - Downloading TLE data
- [String Formatting](../learn/utilities/string_formatting.md) - Formatting time durations
- [Predicting Ground Contacts](ground_contacts.md) - Related ground contact analysis example
