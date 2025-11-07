# Predicting Ground Contacts

In this example we'll predict ground contacts between the NISAR radar satellite and the NASA Near Earth Network ground stations using Brahe. We'll download the latest TLE data for the satellite from CelesTrak, load the NASA Near Earth Network ground station data, and compute the ground contacts between the satellite and ground stations over a 7-day period. We'll then analyze the statistics of the contact duration and number of contacts per ground station.

---

## Setup

First, we'll import the necessary libraries, initialize Earth orientation parameters, download the TLE for NISAR (NORAD ID 65053) from CelesTrak, and load the NASA Near Earth Network ground station network.

``` python
--8<-- "./examples/examples/ground_contacts.py:19:28"
```

We download the NISAR TLE directly by NORAD ID and load all NASA NEN ground stations:

``` python
--8<-- "./examples/examples/ground_contacts.py:40:41"
```

``` python
--8<-- "./examples/examples/ground_contacts.py:50:50"
```

We then propagate NISAR for 3 orbital periods to prepare for ground track visualization:

``` python
--8<-- "./examples/examples/ground_contacts.py:57:58"
```

## Ground Track Visualization

Next we'll visualize the ground track and communication cones for NISAR over a 3-orbit period. The communication cones show the coverage area of each ground station based on a 5° minimum elevation angle:

``` python
--8<-- "./examples/examples/ground_contacts.py:66:91"
```

The resulting plot shows NISAR's ground track in red and the NASA Near Earth Network stations with their communication cones in blue:

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/ground_contacts_groundtrack_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/ground_contacts_groundtrack_dark.html"  loading="lazy"></iframe>
</div>

## Compute Ground Contacts

We'll compute the ground contacts between NISAR and the NASA Near Earth Network ground stations over a 7-day period using Brahe's access computation tools. We use an elevation constraint of 5° minimum elevation:

``` python
--8<-- "./examples/examples/ground_contacts.py:98:110"
```

Below is the table of the first 20 contact windows. Click on any column header to sort:

<div class="center-table" markdown="1">
{{ read_csv('figures/ground_contacts_windows.csv') }}
</div>

## Analyze Contact Statistics

Finally, we'll analyze the contact statistics, including the average daily contacts per station and distribution of contact durations.

We group the contact windows by station and compute the average daily contacts:

``` python
--8<-- "./examples/examples/ground_contacts.py:164:190"
```

Then we create two visualizations: a bar chart of average daily contacts per station, and a histogram of contact duration distribution:

``` python
--8<-- "./examples/examples/ground_contacts.py:188:251"
```

The daily contacts bar chart shows which stations have the most frequent visibility to NISAR:

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/ground_contacts_daily_contacts_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/ground_contacts_daily_contacts_dark.html"  loading="lazy"></iframe>
</div>

The duration histogram shows the distribution of contact lengths, with statistics overlay:

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/ground_contacts_duration_dist_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/ground_contacts_duration_dist_dark.html"  loading="lazy"></iframe>
</div>

## Full Code Example

```python title="ground_contacts.py"
--8<-- "./examples/examples/ground_contacts.py:19:257"
```

## See Also

- [Access Computation](../learn/access_computation/index.md)
- [Locations](../library_api/access/locations.md)
- [Constraints](../library_api/access/constraints.md)
