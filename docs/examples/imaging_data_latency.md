# Calculating Imaging Data Latency

In this example we'll analyze the imaging data latency for a satellite constellation. Data latency is defined as the time between a satellite exiting an imaging region (Area of Interest) and the start of its next ground station contact for data downlink.

For this analysis we'll use the Capella constellation (a commercial SAR imaging constellation) and the KSAT ground station network. The Area of Interest is the continental United States.

This metric is critical for understanding how quickly collected imagery can be delivered to end users, which is a key performance indicator for Earth observation missions.

---

## Setup

First, we'll import the necessary libraries and initialize Earth orientation parameters:

``` python
--8<-- "./examples/examples/imaging_data_latency.py:preamble"
```

We download all active satellite TLEs from CelesTrak and filter for Capella satellites:

``` python
--8<-- "./examples/examples/imaging_data_latency.py:download_capella"
```

Next, we load the KSAT ground station network:

``` python
--8<-- "./examples/examples/imaging_data_latency.py:load_ksat"
```

## Define the Area of Interest

We define the AOI as a polygon covering the continental United States using a detailed GeoJSON outline:

``` python
--8<-- "./examples/examples/imaging_data_latency.py:define_aoi"
```

## Filter Ground Stations

We filter out ground stations that are inside the AOI. The reasoning is that a satellite cannot begin a downlink pass while it is still over the imaging region - it must first exit the AOI:

``` python
--8<-- "./examples/examples/imaging_data_latency.py:filter_stations"
```

## Compute AOI Exit Events

Using the `AOIExitEvent` detector with `SGPPropagator`, we detect every time a Capella satellite exits the US imaging region:

``` python
--8<-- "./examples/examples/imaging_data_latency.py:compute_aoi_exits"
```

## Compute Ground Contacts

We reset the propagators and use the access computation pipeline to find all ground station contacts over the 7-day period:

``` python
--8<-- "./examples/examples/imaging_data_latency.py:compute_ground_contacts"
```

## Calculate Latencies

For each AOI exit event, we find the next ground contact for that satellite and compute the latency (time difference):

``` python
--8<-- "./examples/examples/imaging_data_latency.py:compute_latencies"
```

## Results

### Top 5 Worst Latencies

The table below shows the 5 longest imaging data latencies - these represent the worst-case scenarios for data delivery:

<div class="center-table" markdown="1">
{{ read_csv('figures/imaging_data_latency_top5.csv') }}
</div>

### Latency Statistics

Summary statistics for all imaging data latencies over the 7-day period:

<div class="center-table" markdown="1">
{{ read_csv('figures/imaging_data_latency_stats.csv') }}
</div>

``` python
--8<-- "./examples/examples/imaging_data_latency.py:statistics"
```

## Visualization

The ground track plot below shows the satellite paths during the top 3 worst-case latency periods. The green dashed line indicates the US AOI boundary, and the blue circles show the ground station communication cones:

``` python
--8<-- "./examples/examples/imaging_data_latency.py:visualization"
```

<figure markdown="span">
    ![Imaging Data Latency Ground Tracks](../figures/imaging_data_latency_groundtrack_light.svg#only-light)
    ![Imaging Data Latency Ground Tracks](../figures/imaging_data_latency_groundtrack_dark.svg#only-dark)
</figure>

The colored tracks show the satellite ground paths from the moment of AOI exit until the start of the next ground contact:

- **Red**: Longest latency (worst case)
- **Orange**: Second longest latency
- **Yellow**: Third longest latency

This visualization helps identify geographic regions where additional ground stations might reduce data latency.

## Full Code Example

```python title="imaging_data_latency.py"
--8<-- "./examples/examples/imaging_data_latency.py:all"
```

---

## See Also

- [Maximum Communications Gap](max_communications_gap.md) - Related analysis of communication gaps
- [Predicting Ground Contacts](ground_contacts.md) - Ground contact analysis example
- [Access Computation](../learn/access_computation/index.md) - Understanding access windows and constraints
- [AOI Events](../library_api/events/premade.md#area-of-interest-events) - AOIEntryEvent and AOIExitEvent API reference
- [KSAT Ground Stations](../learn/datasets/groundstations.md) - Ground station dataset documentation
