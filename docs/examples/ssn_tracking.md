# SSN Radar Tracking: EKF, UKF, and Batch Least Squares

In this example we'll simulate Space Surveillance Network (SSN) radar tracking of a LEO object and estimate its orbit sequentially with an Extended Kalman Filter (EKF) and an Unscented Kalman Filter (UKF). We'll load the Vallado SSN sensor dataset, find the passes visible to each sensor over a 6-hour tracking arc, simulate az/el/range measurements during those passes, and compare how the two filters converge on the true orbit. A separate script, `ssn_tracking_bls.py`, solves the same scenario with Batch Least Squares (BLS) instead; we walk through it near the end of this page.

---

## Setup

First, we'll import the necessary libraries, initialize Earth orientation parameters, and configure the tracking arc:

``` python
--8<-- "./examples/examples/ssn_tracking.py:preamble"
```

## Load the Sensor Network

We load the Vallado SSN sensor sites and build sensors from the radar sites. The dataset's sites split into two sensor types: optical (`radec`) trackers (Socorro, Maui, Diego Garcia, MSSS, MOTIF, Moron), which `SimpleSSNSensor` doesn't support, and radar/phased-array/mechanical (`azel_range`) sites. `SimpleSSNSensor.from_locations()` builds a sensor for every radar site, defaulting the two that lack Table 4-3/4-4 calibration (Haystack, HAX) to zero noise and bias -- overridable with `with_noise()`/`with_bias()`. `from_locations_calibrated()` restricts to the fully-calibrated sites; this example uses the calibrated set so simulated measurement noise always matches Table 4-4:

``` python
--8<-- "./examples/examples/ssn_tracking.py:load_sensors"
```

## Visualize the Sensor Network

Next we propagate a truth trajectory -- a 700 km, 72&deg; inclination LEO orbit -- and plot its ground track against the sensor network's coverage cones, using each sensor's minimum elevation angle. Each sensor gets a stable color, reused for its measurement traces in the next figure, so a comm cone's color on the map identifies the same sensor in the az/el/range plot below:

``` python
--8<-- "./examples/examples/ssn_tracking.py:visualize_network"
```

!!! tip "Interpolation"
    Normally sampling measurements at a finer time step (15 s) than the propagation step (~60 s) would introduce interpolation error; brahe's orbit propagators use Hermite-cubic interpolation by default, which avoids these artifacts. See [InterpolationMethod](../library_api/orbits/enums.md#interpolationmethod) for the available interpolation algorithms.

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/ssn_tracking_network_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/ssn_tracking_network_dark.html"  loading="lazy"></iframe>
</div>

## Simulate Measurements

Access windows bound the simulation: for each sensor we compute the elevation-constrained access windows against the truth trajectory, then call `simulate_observations()` only within those windows. This keeps the observation set to what the sensor geometry actually supports, rather than sampling continuously and discarding invisible epochs:

``` python
--8<-- "./examples/examples/ssn_tracking.py:simulate_measurements"
```

Over the 6-hour arc, the network produces hundreds of observations across dozens of passes. Each measurement is `[azimuth, elevation, range]`, generated with each sensor's own bias and noise from Vallado Table 4-4, and colored to match its sensor in the network figure above:

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/ssn_tracking_measurements_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/ssn_tracking_measurements_dark.html"  loading="lazy"></iframe>
</div>

## Run the Estimators

Each sensor exposes a `measurement_model()` that carries the same bias and noise used to generate its measurements. Because the model applies bias inside `predict()`, a filter built from these models is consistent with the simulated observations -- this is the calibrated-sensor assumption: the filter is told about the same bias the sensor introduces, rather than having to estimate it away.

Passes from different sensors can overlap -- Cavalier and Millstone, for instance, both track the object around the same time -- so the per-sensor pass windows are first merged into non-overlapping tracking intervals. The EKF and UKF then each walk the intervals in order: the filter is stepped forward through the gap before an interval with `propagate_to()` at a fixed 60 s cadence (prediction-only, no measurement update, recording a `FilterRecord` named `"Propagation"` at each step so covariance growth during the gap is visible in the diagnostic output), and every observation within the interval is then processed in time order:

``` python
--8<-- "./examples/examples/ssn_tracking.py:run_filters"
```

## Analyze Results

Finally, we check filter consistency -- whether the formal uncertainty actually bounds the true error -- by rotating each record's position error and position-covariance block into the radial/along-track/cross-track (RTN) frame of the truth state at that epoch with `bh.rotation_eci_to_rtn()`. Each per-axis row shades a &plusmn;3&sigma; band from the rotated covariance diagonal and draws the signed error on top, so a consistent filter keeps its error line inside the shaded band; a bottom row shows the unsigned total position error on a linear scale:

``` python
--8<-- "./examples/examples/ssn_tracking.py:analyze_results"
```

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/ssn_tracking_filters_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/ssn_tracking_filters_dark.html"  loading="lazy"></iframe>
</div>

Along-track error typically dominates the other two axes between passes: along-track position error grows from an along-track velocity or timing error integrated over time, while radial and cross-track errors stay comparatively small once the orbit plane and semi-major axis are well observed. The shaded bands widen visibly during the gaps between passes -- reflecting the same covariance growth visible in the `"Propagation"` records -- and narrow again once a new pass starts feeding measurements back in.

The EKF and UKF converge to nearly identical results here, both reaching meter-level final position error, since the two-body dynamics and az/el/range geometry are only mildly nonlinear over this arc; the UKF's sigma-point propagation does not offer a material advantage at this level of nonlinearity. Azimuth wrapping is handled consistently: `AzElRangeMeasurementModel.residual()` wraps the residual, the model's Jacobian differences through it, and the UKF forms its predicted measurement mean and innovation deviations through `residual()` as well, so a pass whose sigma-point azimuths straddle the 0/360&deg; wrap stays well-behaved.

## Batch Least Squares

BLS solves for the orbit in one batch, minimizing the weighted residual cost over the entire observation set at once, rather than the sequential filters' approach of recursively updating a running state estimate and letting covariance grow between passes. Each Gauss-Newton iteration re-propagates the full arc together with the state transition matrix, which makes BLS substantially slower than the sequential filters above, so this variant lives in its own script, `ssn_tracking_bls.py`.

The script reproduces the same scenario as `ssn_tracking.py` -- the sensor dataset, the truth orbit, and the simulated measurements -- before solving with BLS. Its default state-correction convergence threshold (`1e-8`, in mixed position/velocity units) is far tighter than the centimeter-level precision this measurement set actually supports, so the script loosens `state_correction_threshold` and adds a `cost_convergence_threshold`; this stops the solver once corrections are physically insignificant instead of exhausting `max_iterations` on progressively smaller, noise-driven corrections:

``` python
--8<-- "./examples/examples/ssn_tracking_bls.py:run_bls"
```

???+ example "Output"
    ```
    --8<-- "./docs/outputs/examples/ssn_tracking_bls.py.txt"
    ```

BLS's final position error is somewhat higher than the sequential filters reach, since it minimizes the residual cost over the whole arc rather than tracking the most recent state, though its formal uncertainty remains comparable to the sequential filters', as shown in the output above.

??? "Full Code"

    ```python title="ssn_tracking_bls.py"
    --8<-- "./examples/examples/ssn_tracking_bls.py:all"
    ```

## Full Code Example

??? "Full Code"

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
