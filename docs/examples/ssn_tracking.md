# SSN Radar Tracking: EKF, UKF, and Batch Least Squares

In this example we'll simulate Space Surveillance Network (SSN) radar tracking of a LEO object and estimate its orbit with three different estimators: an Extended Kalman Filter (EKF), an Unscented Kalman Filter (UKF), and Batch Least Squares (BLS). We'll load the Vallado SSN sensor dataset, find the passes visible to each sensor over a 6-hour tracking arc, simulate az/el/range measurements during those passes, and compare how the three estimators converge on the true orbit.

---

## Setup

First, we'll import the necessary libraries, initialize Earth orientation parameters, and configure the tracking arc:

``` python
--8<-- "./examples/examples/ssn_tracking.py:preamble"
```

## Load the Sensor Network

We load the Vallado SSN sensor sites and build az/el/range sensors from them. Of the 21 sites in the dataset, 13 construct as `SimpleSSNSensor` instances -- the remaining 8 are optical (`radec`) trackers or sites (HAX, Haystack) that Table 4-2 lists without the az/el/range calibration values `SimpleSSNSensor` requires:

``` python
--8<-- "./examples/examples/ssn_tracking.py:load_sensors"
```

## Visualize the Sensor Network

Next we propagate a truth trajectory -- a 700 km, 72&deg; inclination LEO orbit -- and plot its ground track against the sensor network's coverage cones, using each sensor's minimum elevation angle:

``` python
--8<-- "./examples/examples/ssn_tracking.py:visualize_network"
```

The truth trajectory is generated with Hermite-cubic interpolation. The propagator stores states at its adaptive-step cadence (roughly 60 s); linear interpolation between those points would be off by kilometers when sampled at the much finer measurement cadence used below, so Hermite-cubic interpolation is required to keep the simulated measurements consistent with the true orbit.

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/ssn_tracking_network_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/ssn_tracking_network_dark.html"  loading="lazy"></iframe>
</div>

## Simulate Measurements

Access windows bound the simulation: for each sensor we compute the elevation-constrained access windows against the truth trajectory, then call `simulate_observations()` only within those windows. This keeps the observation set to what the sensor geometry actually supports, rather than sampling continuously and discarding invisible epochs:

``` python
--8<-- "./examples/examples/ssn_tracking.py:simulate_measurements"
```

Over the 6-hour arc, the network produces 740 observations across 22 passes. Each measurement is `[azimuth, elevation, range]`, generated with each sensor's own bias and noise from Vallado Table 4-4:

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/ssn_tracking_measurements_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/ssn_tracking_measurements_dark.html"  loading="lazy"></iframe>
</div>

## Run the Estimators

Each sensor exposes a `measurement_model()` that carries the same bias and noise used to generate its measurements. Because the model applies bias inside `predict()`, a filter built from these models is consistent with the simulated observations -- this is the calibrated-sensor assumption: the filter is told about the same bias the sensor introduces, rather than having to estimate it away.

The EKF and UKF process observations sequentially in epoch order. Passes are separated by long gaps with no sensor visibility, so between passes the filter is stepped forward with `propagate_to()` at a fixed 60 s cadence rather than jumping straight to the next observation. This is prediction-only -- no measurement update -- and it records a `FilterRecord` at each step (named `"Propagation"`, with empty measurement fields) so covariance growth during the gap is visible in the diagnostic output:

``` python
--8<-- "./examples/examples/ssn_tracking.py:run_filters"
```

BLS, by contrast, solves the full 740-observation set at once rather than processing arc-by-arc. Its default state-correction convergence threshold (`1e-8`, in mixed position/velocity units) is far tighter than the centimeter-level precision this measurement set actually supports, so the example loosens `state_correction_threshold` to `1e-3` and adds a `cost_convergence_threshold` -- this stops the solver once corrections are physically insignificant instead of exhausting `max_iterations` on progressively smaller, noise-driven corrections.

## Analyze Results

Finally, we compare true position error against each filter's formal 3-sigma uncertainty, and check that the uncertainty envelope actually bounds the error (filter consistency):

``` python
--8<-- "./examples/examples/ssn_tracking.py:analyze_results"
```

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/ssn_tracking_filters_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/ssn_tracking_filters_dark.html"  loading="lazy"></iframe>
</div>

The EKF and UKF converge to nearly identical results here (6.6 m final position error for both) since the two-body dynamics and az/el/range geometry are only mildly nonlinear over this arc; the UKF's sigma-point propagation does not offer a material advantage at this level of nonlinearity. Note that the UKF's sigma-point measurement *mean* can be distorted when sigma points straddle the azimuth 0/360&deg; wrap during a pass -- `AzElRangeMeasurementModel.residual()` wraps the final residual, but this does not correct a biased sigma-point mean, so it remains a documented limitation of applying the UKF to wrap-prone angular measurements. BLS, solving the full arc in one batch rather than propagating through gaps, reaches a 9.5 m final position error -- higher than the sequential filters here because it minimizes the weighted residual cost over the whole arc rather than tracking the most recent state, but with a comparable formal 3-sigma uncertainty (39.5 m vs. 42.1 m for the EKF/UKF).

## Full Code Example

```python title="ssn_tracking.py"
--8<-- "./examples/examples/ssn_tracking.py:all"
```

## See Also

- [Estimation](../learn/estimation/index.md)
- [Measurement Models](../learn/estimation/measurement_models.md)
- [Extended Kalman Filter](../learn/estimation/extended_kalman_filter.md)
- [Unscented Kalman Filter](../learn/estimation/unscented_kalman_filter.md)
- [Batch Least Squares](../learn/estimation/batch_least_squares.md)
- [SSN Sensor Datasets](../learn/datasets/ssn_sensors.md)
