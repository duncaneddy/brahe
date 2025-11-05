# Calculating Doppler Compensation


In this example we'll predict ground contacts between the NISA radar satellite constellation and a ground station network using Brahe. We'll download the lastest TLE data for the satellite from CelesTrak, load the NASA Near Earth Network ground station network, and compute the ground station between the satellite and ground station over a 7-day period. We'll then compute the statistics of the contact duration and number of contacts per ground station.
---

## Setup

First, we'll import the necessary libraries, initialize Earth orientation parameters, download the TLE for NISAR (65053) from CelesTrak, and load the NASA Near Earth Network ground station network.


## Ground Track Visualization

Next we'll visualize the ground track and communication cones for NISAR over over a 3-orbit period.

<!-- Use matplotlib plotting from grond_track with natural_earth background, set altitude based on NISAR semi-major axis -->

## Compute Ground Contacts

We'll compute the ground contacts between NISAR and the NASA Near Earth Network ground stations over a 7-day period using Brahe's access computation tools.


Here is a sample of the contact windows computed for NISAR and the ground stations:

<!-- Have an optional print statement of markdown formatted, station name, time start, time end, duration, maximum elevation -->

## Analyze Contact Statistics

Finally, we'll analyze the contact statistics, including the average daily contact per station and distribution of contact durations.

<!-- present both results as a histogram -->


## Full Code Example

---

## See Also