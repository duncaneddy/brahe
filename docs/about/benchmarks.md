## Benchmarks

To check Brahe's consistency with other astrodynamics software, we've built a benchmark suite that exercises a wide range of astrodynamics tasks across multiple libraries. It's extensible, so new tasks and libraries can be added over time. The current set covers 32 tasks across 8 modules: time, coordinates, attitude, frames, orbits, propagation, force model, and access computation.

The primary "source of truth" is [**OreKit 13.1.5**](https://github.com/CS-SI/Orekit) (Java), arguably the most widely used open-source astrodynamics library. We also compare against [**GMAT R2026a**](https://github.com/nasa/GMAT) (NASA Goddard's General Mission Analysis Tool) on 31 of the 32 tasks, [**Basilisk**](https://github.com/AVSLab/basilisk) (AVS Lab) on the 14 tasks across attitude, orbits, frames, coordinates, and propagation where the two libraries' user-facing APIs overlap, and [Nyx](https://github.com/nyx-space/nyx)/[ANISE](https://github.com/nyx-space/anise) (Nyx Space) on 26 tasks where the libraries have comparable APIs. The benchmarks report both performance and accuracy metrics for each library on each task, with OreKit as the reference baseline for accuracy comparisons.

!!! tip

    These benchmarks show how Brahe agrees with other astrodynamics libraries and help users compare options. They are not meant to declare a winner. Brahe itself wouldn't exist without the work of other projects building out an open astrodynamics ecosystem, and we think it's healthy for users to have multiple viable tools to pick from.

Libraries have different design goals and use cases. Basilisk, for example, is a full spacecraft simulation environment with a Python interface; its API is not designed for atomic conversion calls or single-step propagation, since it computes those internally during simulation. GMAT is a script-driven mission analysis tool with a Python API layered on top of the script interface, so some tasks are implemented via script execution rather than direct API calls. Brahe's benchmark suite exercises atomic conversion calls (frame transforms, force-model accelerations, time-scale conversions, single-step Keplerian propagation) that aren't always exposed at the user-API level in other libraries; the suite only includes tasks where the API call being timed is apples-to-apples.

A useful side effect of comparing this many libraries is that broad agreement across them gives confidence that the implementations are correct and that results should be similar regardless of which library you pick. The performance numbers are also informative, but they should be read in light of each library's design goals and use cases.

!!! warning

    Benchmarks are difficult and nuanced to put together. They're highly sensitive to the specific hardware and software environment they run in, and small implementation details can shift performance significantly. These comparisons tried to be faithful to each library's "out-of-the-box" capabilities; better ways to set up the tasks or use the libraries may well exist. Treat the numbers here as a starting point for comparison, not a final word. If you're choosing a library for a particular use case, it's worth running your own benchmarks on your target hardware and software.

### Methodology

**Languages and Libraries**:

- **Java**: [OreKit 13.1.5](https://github.com/CS-SI/Orekit) on OpenJDK 21
- **Basilisk**: [AVS Lab Basilisk](https://github.com/AVSLab/basilisk) (`bsk` Python wheel from PyPI), imported in-process by the Python runner; participates on a 14-task subset.
- **GMAT**: [GMAT R2026a](https://github.com/nasa/GMAT) (`gmatpy` API), accessed via a local GMAT install pointed to by `GMAT_ROOT_PATH`; participates on a 31-task subset.
- **Nyx Space**: [Nyx 2.4.0](https://github.com/nyx-space/nyx), [ANISE 0.10.1](https://github.com/nyx-space/anise), and [hifitime](https://github.com/nyx-space/hifitime) Rust libraries. Nyx/ANISE participate on a 26-task subset.
- **Brahe (Python)**: Brahe Python bindings (PyO3)
- **Brahe (Rust)**: Brahe native Rust library
**Test Environment**: 2021 MacBook Pro, Apple M1 Max, 64 GB RAM

**Protocol**: Two independent harnesses share the same task registry:

- **Performance** — each task runs 200 iterations of a single fixed input with a fixed random seed. Mean / median / std / min / max execution time is reported per language. Per-module bar charts plot the mean time per task; the error bars span **±1 standard deviation** of the iteration timings (raw spread of a single run, not a confidence interval on the mean).
- **Accuracy** — each task runs once across a sweep of 200 independent initial conditions (configurable via `--samples`). For every non-baseline language the per-sample max-abs and RMS errors are computed against OreKit, then aggregated to p50 / p95 / p99 / max distributional statistics. Per-module CDFs visualize the full error distribution; per-task scatter plots are emitted where a single scalar (altitude, epoch, …) usefully indexes the sample. Accuracy comparisons take OreKit as the single reference baseline.

### Performance Overview

The table below summarizes average speedup relative to different baselines. Values greater than 1.0× indicate the library is faster than the baseline; values less than 1.0× indicate the baseline is faster.

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_overview.csv') }}

</div>

<div class="plotly-embed x-tall">
  <iframe class="only-light" src="../figures/fig_bench_speedup_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_bench_speedup_dark.html"  loading="lazy"></iframe>
</div>

OreKit is the single performance reference. We don't show per-baseline speedup charts using Basilisk or GMAT as the denominator because each ratio is derivable from this chart, and the extra framings tend to read as "who won" comparisons rather than the consistency check we're after.

---

### Time

Five tasks covering epoch creation and time system conversions (UTC → TAI, TT, GPS, UT1).

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_perf_time.csv') }}

</div>

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_bench_time_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_bench_time_dark.html"  loading="lazy"></iframe>
</div>

**Accuracy** (initial-condition sweep; sample count per row; p50 / p95 / p99 / max max-abs error vs Orekit):

*Sampling range*: each sample is a random calendar datetime with year uniformly in [2000, 2030], month [1, 12], day [1, 28], hour [0, 23], minute [0, 59], second [0, 60) s, and nanosecond [0, 10⁹) ns. Components are drawn independently per sample.

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_accuracy_time.csv') }}

</div>

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_bench_accuracy_time_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_bench_accuracy_time_dark.html"  loading="lazy"></iframe>
</div>

---

### Coordinates

Five tasks covering coordinate system transformations: geodetic/geocentric to/from ECEF, and ECEF to azimuth-elevation.

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_perf_coordinates.csv') }}

</div>

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_bench_coordinates_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_bench_coordinates_dark.html"  loading="lazy"></iframe>
</div>

**Accuracy** (initial-condition sweep; sample count per row; p50 / p95 / p99 / max max-abs error vs Orekit):

*Sampling range*: geodetic/geocentric inputs use longitude uniform in [-180°, 180°], latitude uniform in [-90°, 90°], and altitude (or radius offset above the sphere) uniform in [0, 1000] km. The `ecef_to_azel` task additionally samples a satellite point above each station with longitude/latitude offsets in ±10° and satellite altitude uniform in [200, 1000] km; station latitude is restricted to [-70°, 70°] to avoid degeneracies near the poles.

Geodetic and geocentric outputs combine angular and altitude components in different units; the benchmark converts angle residuals to surface-arc distance so each row reports a single position-equivalent error in meters. The Orekit-vs-GMAT row on the geodetic conversions is dominated by GMAT's choice of equatorial radius (6378.1363 km vs WGS84's 6378.137 km, a ~0.7 m offset); on the WGS84 baselines the implementation-level agreement is sub-nanometer. The Orekit-vs-Nyx geodetic rows show ~350–415 mm residuals because ANISE's `frame_info` populates the Earth ellipsoid from `pck11.pca` (IAU PCK, $a$ = 6378.1366 km) rather than WGS84 ($a$ = 6378.137 km); the ~0.4 m equatorial-radius difference is the source of the offset. ANISE 0.10.1 doesn't expose a WGS84-specific ellipsoid override, so this is fixed in the library's coordinate conversion path.

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_accuracy_coordinates.csv') }}

</div>

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_bench_accuracy_coordinates_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_bench_accuracy_coordinates_dark.html"  loading="lazy"></iframe>
</div>

---

### Attitude

Four tasks covering conversions between quaternions, rotation matrices, and Euler angles.

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_perf_attitude.csv') }}

</div>

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_bench_attitude_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_bench_attitude_dark.html"  loading="lazy"></iframe>
</div>

**Accuracy**: Quaternion comparisons are performed in rotation-matrix space (Frobenius norm of $R_a - R_b$, where $R$ is the rotation matrix induced by each quaternion). This removes the $q \equiv -q$ sign ambiguity at the comparison boundary, so any residual is a real rotation difference rather than a representation artifact.

*Sampling range*: quaternion inputs are drawn uniformly from the unit 3-sphere by sampling each component i.i.d. from a unit Gaussian and normalizing — this yields a uniform distribution on SO(3) rotations. Rotation-matrix inputs are derived from the same quaternion samples. The `euler_angle_to_quaternion` task samples Euler angles directly with $\phi \in [-\pi, \pi]$, $\theta \in [-\pi/2, \pi/2]$, $\psi \in [-\pi, \pi]$ (ZYX convention).

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_accuracy_attitude.csv') }}

</div>

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_bench_accuracy_attitude_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_bench_accuracy_attitude_dark.html"  loading="lazy"></iframe>
</div>

---

### Frames

Two tasks covering full 6-DOF state vector transformations between ECEF and ECI reference frames using the IAU 2006/2000A precession-nutation model.

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_perf_frames.csv') }}

</div>

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_bench_frames_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_bench_frames_dark.html"  loading="lazy"></iframe>
</div>

**Accuracy** (initial-condition sweep; sample count per row; p50 / p95 / p99 / max max-abs error vs Orekit):

*Sampling range*: each sample is a 6-DOF Cartesian state derived from random Keplerian elements with semi-major axis uniform in $R_\oplus + [200, 36{,}000]$ km, eccentricity uniform in [0.001, 0.3], inclination uniform in [0°, 180°], and RAAN / argument of periapsis / true anomaly uniform in [0°, 360°). Epochs are drawn uniformly in [2024-01-01, 2025-01-01) (Basilisk's bundled ITRF93 PCK does not extend far past the 2024 window, which constrains the swept range).

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_accuracy_frames.csv') }}

</div>

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_bench_accuracy_frames_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_bench_accuracy_frames_dark.html"  loading="lazy"></iframe>
</div>

---

### Orbits

Two tasks covering conversions between Keplerian orbital elements and Cartesian state vectors.

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_perf_orbits.csv') }}

</div>

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_bench_orbits_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_bench_orbits_dark.html"  loading="lazy"></iframe>
</div>

**Accuracy** (initial-condition sweep; sample count per row; p50 / p95 / p99 / max max-abs error vs Orekit):

*Sampling range*: Keplerian elements are drawn with semi-major axis uniform in $R_\oplus + [200, 36{,}000]$ km (LEO through GEO), eccentricity uniform in [0.001, 0.5], inclination uniform in [0°, 180°], and RAAN / argument of periapsis / mean anomaly (or true anomaly for the Cartesian-input task) uniform in [0°, 360°). Cartesian-input samples are produced by converting these elements to a state vector via the standard perifocal-to-ECI rotation.

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_accuracy_orbits.csv') }}

</div>

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_bench_accuracy_orbits_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_bench_accuracy_orbits_dark.html"  loading="lazy"></iframe>
</div>

---

### Propagation

Five tasks covering Keplerian (two-body analytical), numerical (RK4/RK78 two-body), and SGP4/SDP4 propagation.

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_perf_propagation.csv') }}

</div>

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_bench_propagation_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_bench_propagation_dark.html"  loading="lazy"></iframe>
</div>

**Accuracy** (initial-condition sweep; sample count per row; p50 / p95 / p99 / max max-abs error vs Orekit):

*Sampling range*:

- `keplerian_single`: random orbit with $a \in R_\oplus + [200, 36{,}000]$ km, $e \in [0.001, 0.3]$, $i \in [0°, 180°]$, RAAN / $\omega$ / $M$ uniform in [0°, 360°); per-sample propagation duration $\Delta t$ uniform in [1 h, 24 h]; epoch JD spread uniformly over calendar year 2024.
- `keplerian_trajectory`, `numerical_twobody`, and the three RK4 force-model tasks (`grav5x5`, `grav20x20_sun_moon`, `grav80x80_full`): LEO IC sweep with $a \in R_\oplus + [400, 1500]$ km, $e \in [0.001, 0.02]$, $i \in [0°, 180°]$, angles uniform in [0°, 360°), epoch JD spread over 2024. Altitude floor of 400 km and eccentricity ceiling of 0.02 keep every sampled perigee above ~250 km, where fixed-step RK4 remains well-behaved on every backend; the 80×80 + drag + SRP case otherwise aborted with "accuracy settings violated" errors in GMAT and produced 10¹¹ m state offsets in brahe. Each case is propagated over one nominal LEO orbital period (~90 minutes); only the final state is compared.
- `sgp4_single`: ISS TLE evaluated at $n$ random time offsets uniformly distributed in [0, 86400] s (sorted).
- `sgp4_trajectory`: ISS TLE sampled at $n$ uniformly spaced points across a fixed 48-hour horizon — points along a single trajectory rather than statistically independent draws, so this row reads as an error-growth profile rather than an IC distribution.

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_accuracy_propagation.csv') }}

</div>

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_bench_accuracy_propagation_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_bench_accuracy_propagation_dark.html"  loading="lazy"></iframe>
</div>

---

### Force Model

Five tasks evaluating a single acceleration term at a fixed spacecraft state and epoch (no integrator, no time stepping). This isolates the force-model implementations from the propagator, so any disagreement here comes from the force-model calculation itself — gravity coefficients, third-body ephemeris, frame transformations — rather than from numerical integration.

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_perf_force_model.csv') }}

</div>

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_bench_force_model_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_bench_force_model_dark.html"  loading="lazy"></iframe>
</div>

**Accuracy** (LEO IC sweep; per-pair max-abs error distribution vs Orekit):

*Sampling range*: force-model accuracy is swept over the same LEO range used for the propagation accuracy sweep — $a \in R_\oplus + [400, 1500]$ km, $e \in [0.001, 0.02]$, $i \in [0°, 180°]$, angles uniform in [0°, 360°), epoch JD spread over 2024. Each case's Keplerian elements are converted to a Cartesian GCRF state in the task generator so every backend sees the identical state per case (avoiding implementation-drift in KOE→ECI conversion as a source of residual). Each task therefore yields one residual per case, so the harness reports a distribution rather than a single number.

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_accuracy_force_model.csv') }}

</div>

---

### Access (Comparative)

One task: computing all satellite-to-ground-station access windows over a 48-hour period using SGP4 propagation.

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_perf_access.csv') }}

</div>

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_bench_access_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_bench_access_dark.html"  loading="lazy"></iframe>
</div>

**Accuracy** (per-location contact-count and per-window timing residuals vs Orekit):

*Sampling range*: a single fixed SGP4 propagator (ISS TLE) is evaluated against 100 random ground locations sampled with longitude uniform in [-180°, 180°], latitude uniform in [-90°, 90°], altitude fixed at 0 m, 10° minimum elevation, over a 24-hour search window starting at the TLE epoch. The samples here index ground locations rather than orbit initial conditions, so the accuracy distribution describes how access-window detection varies geographically for one orbit, not how it varies across orbits.

The table reports, for each backend pair: how many ground locations produced any contacts, the total contact count each backend found across those locations, the worst per-location count mismatch, and the distribution of matched-window start/end time residuals across all locations. Windows are matched greedily by nearest start time within a 120 s tolerance; unmatched windows contribute to the count difference but not the timing residuals.

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_accuracy_access.csv') }}

</div>

---

### Access Computation (Brahe vs Skyfield)

In addition to the OreKit comparison above, Brahe is also benchmarked against **Skyfield**, a popular Python astronomy library, for access computation. This benchmark focuses on Brahe's serial vs parallel execution modes and Python bindings vs native Rust performance.

The benchmark propagates the ISS (from a single, static TLE) over a 2-day (48-hour) window and computes all access intervals against 100 randomly sampled ground station locations with a 5° minimum elevation. Access start and end times agree to within one second between Brahe and Skyfield.

<div class="center-table" markdown="1">
| Implementation         | Avg Time  | vs Skyfield     |
|------------------------|-----------|-----------------|
| Skyfield               |  32.78ms  | baseline        |
| Brahe-Python (serial)  |   2.99ms  | 11.0x faster    |
| Brahe-Python (parallel)|   3.08ms  | 10.6x faster    |
| Brahe-Rust (serial)    |   2.47ms  | 13.3x faster    |
| Brahe-Rust (parallel)  |   2.00ms  | 16.4x faster    |
</div>

The parallel rows report the per-location amortized time when all 100 locations are batched into a single call; the serial rows report the mean time for a single location computed in isolation. Brahe's Python bindings dispatch into the same Rust core as the native Rust path, so the serial Python row sits within ~20% of the serial Rust row. For the Python path at this problem size, batching does not improve per-location time because the per-location work is already short relative to the call setup amortized in the serial measurement.

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_access_benchmark_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_access_benchmark_dark.html"  loading="lazy"></iframe>
</div>

---

### Notes on Basilisk Comparisons

Basilisk participates in 14 of 32 tasks. The gap reflects API surface, not capability: brahe's suite exercises atomic conversion calls (frame transforms, force-model accelerations, time-scale conversions, single-step Keplerian propagation), and Basilisk doesn't expose all of those at the user-API level since its dynamics modules compute them internally during simulation.

**Where Basilisk participates**: 4 attitude conversions (`RigidBodyKinematics`), 2 orbital-element conversions (`orbitalMotion.elem2rv` / `rv2elem`), 2 frame transformations (via the bundled `pyswice` SPICE Toolkit, J2000 ↔ ITRF93), 2 geodetic/ECEF coordinate conversions (also `pyswice`), and 4 numerical orbit propagation cases (`spacecraft.Spacecraft` + `simIncludeGravBody` with the RK4 default integrator).

**Where Basilisk does not participate**: SGP4 propagation (Basilisk has no SGP4 in its core), analytical Keplerian propagation (`orbitalMotion.elem2rv` is a single closed-form function call, not a propagator object), atomic force-model accelerations (computed inside dynamics modules), time-scale conversions (Basilisk uses SPICE's `unitim_c`, which would measure SPICE Toolkit rather than a peer time-scale library), and access calculations (depend on SGP4).

**Frame definitions**: Basilisk-via-pyswice transforms between `J2000` and `ITRF93` using NAIF's high-precision Earth orientation binary PCK (`earth_latest_high_prec.bpc`, downloaded automatically by `bench-compare-setup`). OreKit uses `EME2000` (≡ J2000 to sub-meter precision) and `ITRF` (IERS 2010 conventions). ITRF93 follows the IERS 1996 conventions, so expect kilometer-scale ECEF position differences vs. the Java baseline.

**Propagation methodology**: Basilisk's `SimBaseClass` setup cost is included in the per-iteration timing, matching the OreKit/Rust pattern of timing the full `run`. Basilisk's inertial output (`r_BN_N`, `v_BN_N`, J2000-equatorial) is captured raw inside the timed region; the J2000 → GCRF frame-bias transform via `brahe.state_eme2000_to_gcrf` runs after `time_iterations` returns, so the timed work is purely Basilisk. Basilisk's default Earth `mu` is overridden from `3.986004360e14` to `3.986004418e14` so all three baselines see the same central-body parameter. Gravity coefficients come from `GGM03S` (Basilisk-bundled, up to degree 180) versus EIGEN-5C (Orekit) and brahe's native EGM family. The numerical two-body and three RK4 force-model tasks all run fixed-step Classical Runge-Kutta 4 (RK4) at the same step size on every implementation, so the residual reflects gravity-coefficient sources and floating-point ordering rather than integrator precision.

**Anomaly convention**: Basilisk's `orbitalMotion.ClassicElements` uses true anomaly in radians; brahe/Orekit use mean anomaly in degrees per the existing benchmark convention. Each library is timed on its native call; conversion happens outside the timed region. The scalar-first `[w, x, y, z]` quaternion convention is already consistent across all baselines, and Basilisk's `RigidBodyKinematics.EP2C` / `C2EP` / `EP2Euler321` / `euler3212EP` use the same passive-rotation (DCM) convention as brahe's `Quaternion` primitives, so all four attitude conversions run on native library calls.

**Atmospheric drag inputs**: For the 80×80 full task, Basilisk's `msisAtmosphere` is driven with representative quiet-Sun space-weather values (Ap=8, F10.7=110) rather than by interpolating `SpaceWeather-All-v1.2.txt` the way Orekit's `CssiSpaceWeatherData` does. That's roughly what a Basilisk user would do without bespoke SW preparation, and it adds some bounded divergence from the Java baseline on that task.

---

### Notes on GMAT Comparisons

[GMAT R2026a](https://github.com/nasa/GMAT) (the General Mission Analysis Tool, developed by NASA Goddard) participates in 31 of the 32 tasks. The one exclusion is `coordinates.ecef_to_azel`: GMAT's `GroundStation` exposes no azimuth/elevation accessor via the `gmatpy` API, and the script-based `Topocentric` coordinate system path segfaults under `gmat.RunScript()`. The only working route is a pure-Python ENZ rotation, which doesn't exercise GMAT code and would be misleading to label as a GMAT comparison.

**Where GMAT participates**: all 5 time tasks, 4 of 5 coordinates tasks, all 4 attitude tasks, both frames tasks, both orbits tasks, all 8 propagation tasks, all 5 force-model tasks, and the access task — 31 tasks in total.

**Setup**: Install GMAT R2026a, export `GMAT_ROOT_PATH` pointing at the install root (the directory containing `bin/`, `data/`, `api/`), and run `just bench-compare-setup`. The setup recipe generates `api_startup_file.txt` via GMAT's bundled `BuildApiStartupFile.py`. When `GMAT_ROOT_PATH` is unset, GMAT comparisons are silently skipped.

**Frame and coordinate disclosure**: GMAT's `EarthFixed` body-fixed frame uses FK5 reference theory with IAU 1980 nutation; brahe and OreKit use the IERS 2010 ITRF realization. Position errors of ~10 m at LEO are attributable to frame-definition choice, not implementation precision. For geodetic coordinate conversions, GMAT uses an Earth equatorial radius of 6378.1363 km versus WGS84's 6378.137 km — accounting for ~0.7 m position error.

**Quaternion convention**: GMAT uses scalar-last `[q1, q2, q3, q4]`; the benchmark canonical is scalar-first `[w, x, y, z]`. Reordering is applied outside the timed region. All four attitude tasks (quaternion ↔ rotation matrix, quaternion ↔ Euler ZYX) call GMAT's native `AttitudeConversionUtility` primitives — `ToCosineMatrix`, `ToQuaternion`, `ToEulerAngles(rv, 3, 2, 1)`.

**Time scales**: GMAT's `A1ModJulian` epoch is rooted at 1941-01-05 (JD 2430000.0), not the standard MJD epoch of 1858-11-17 (JD 2400000.5). The benchmark applies the appropriate offset for time-conversion inputs. GMAT has no GPS time-system enum; GPS times are derived as TAI − 19 s. GMAT's Gregorian time parser has millisecond precision, contributing ~0.5 ms to time-conversion accuracy comparisons.

**Propagation methodology**: Per-iteration construction of `Spacecraft` + `ForceModel` + `Propagator` is included in the timing, which reflects how a GMAT user actually exercises the API. This matches Basilisk's per-iteration setup methodology. The `EarthMJ2000Eq` → GCRF transform is applied after timing using `brahe.state_eme2000_to_gcrf`.

**Units**: GMAT uses km / km/s / km/s² internally for position, velocity, and acceleration; the benchmark canonical is SI (m / m/s / m/s²). Scaling by 10³ is applied outside the timed region.

**Gravity coefficients and atmosphere sources**: GMAT uses bundled JGM-2 / EGM-96 coefficients depending on degree; the JGM-2 file supports up to degree 70, so 80×80 tasks use EGM-96. Drag uses MSISE-90 from GMAT's bundled space-weather data. OreKit uses EIGEN-6S, Basilisk uses GGM03S, and brahe uses per-task bundled data, so differences reflect data-source variation rather than implementation precision.

**SGP4 propagation and access tasks**: These run through GMAT's script-based interface (`gmat.LoadScript` + `gmat.RunScript`) because the SGP4 and ContactLocator plugins don't activate cleanly through the direct `gmatpy` API path. The results are still correct; it's a GMAT integration detail.

---

### Notes on Nyx / ANISE Comparisons

[Nyx 2.4.0](https://github.com/nyx-space/nyx) is a Rust astrodynamics library designed for mission design, trajectory optimization, and orbit determination. [ANISE 0.10.1](https://github.com/nyx-space/anise) is its companion library for reference-frame and ephemeris operations; it provides the `Almanac` abstraction that loads NAIF SPICE-compatible kernels (SPK, PCK, FK) and exposes attitude, rotation, and ephemeris queries in native Rust. [hifitime 4.x](https://github.com/nyx-space/hifitime) is the precision time library shared by both and underpins all epoch and duration arithmetic.

The Nyx/ANISE baseline runs as a separate Rust binary (compiled from `benchmarks/nyx/`), invoked as a subprocess by the Python runner the same way as the Java OreKit binary. This mirrors the brahe-Rust execution model: the Rust binary receives task parameters via stdin JSON and writes results to stdout JSON, while the Python harness handles timing and statistical aggregation identically to all other baselines.

**Where Nyx/ANISE participates**: 26 of 32 tasks. The 6 dropped tasks are listed below.

**Dropped tasks**:

- `coordinates.geocentric_to_ecef` and `coordinates.ecef_to_geocentric`: Brahe defines these as pure spherical-coordinate conversions (latitude-longitude-radius ↔ XYZ without ellipsoidal flattening). ANISE 0.10.1 has no equivalent API that we found; its only coordinate conversion path is geodetic (WGS84 ellipsoid). Implementing this with hand-rolled trigonometry wouldn't exercise any Nyx/ANISE code, so we drop these tasks.
- `force_model.accel_point_mass_gravity`: Brahe exposes a standalone single-body Newtonian evaluation. Nyx 2.4.0 doesn't expose this in a public API; the two-body term is internal to `OrbitalDynamics::eom`. A Nyx implementation would reduce to hand-rolled `μ·r/r³` rather than library code, so we drop this task too.
- `access.sgp4_access`: Brahe's `location_accesses` API finds all satellite-to-ground-station contact windows over an interval. Nyx 2.4.0 has no direct equivalent: the `eclipse` module is unrelated, and a window-finder built on top of Nyx's propagator would have to reproduce brahe's search logic rather than compare library APIs. Dropped.
- `propagation.sgp4_single` and `propagation.sgp4_trajectory`: Nyx 2.4.0 doesn't expose an SGP4 propagator in its public API.

**Earth orientation parameters**: ANISE 0.10.1 doesn't load IERS `finals2000A.all` data. Earth rotation is driven by ANISE's own `earth_latest_high_prec.bpc` binary PCK, which is downloaded fresh from JPL at setup time by `bench-compare-setup`. Frame-transform residuals of roughly 0.5–1.5 m relative to OreKit (which uses IERS finals2000A.all) reflect this BPC-vs-IERS-finals2000A algorithmic difference rather than a precision defect — that difference is the variable under test for the frames tasks.

**Gravity coefficient file format**: Nyx supports SHADR and COF gravity-coefficient formats but not ICGEM. Brahe ships `EGM2008_120.gfc` in ICGEM format. The Nyx benchmark implementation parses the ICGEM file and writes a temporary SHADR file at benchmark startup; coefficient values are numerically identical to the source, so no accuracy impact is expected from the format translation.

**Atmospheric drag model substitution**: Nyx 2.4.0's public atmosphere models are `ConstantDrag`, an exponential model, and US Standard Atmosphere 1976. Neither NRLMSISE-00 nor Harris-Priester is exposed in a public API. The `propagation.numerical_rk4_grav80x80_full` task uses US Standard Atmosphere 1976 for drag, while OreKit uses NRLMSISE-00, so accuracy residuals on this task are larger than on the other propagation tasks. That's expected given the different atmosphere models and not a numerical precision difference.

**Time tasks**: All five time tasks run directly against hifitime's Epoch API, with no extra library layers. hifitime uses TAI internally with UT1 obtained from a bundled IERS file; conversions to GPS, TT, UTC, and UT1 match OreKit at nanosecond precision across the swept epoch range.

**Attitude tasks**: Attitude conversions use ANISE's `DCM`, `EulerParameter`, and matrix-product primitives. The DCM ↔ Euler-angle path goes through ANISE's ZYX decomposition; the quaternion ↔ DCM path uses ANISE's `EulerParameter::to_dcm` / `DCM::to_euler_parameters`. All four tasks run on native ANISE calls.

**Frame tasks**: The two frame-transformation tasks call `Almanac::transform_to` between `EARTH_J2000` and `EARTH_ITRF93` frames. ANISE 0.10.1 provides `DynamicFrame` with five variants: `EarthMeanOfDate` (MOD, precession only), `EarthTrueOfDate` (TOD, precession + nutation), `EarthTrueEquatorMeanEquinox` (TEME, precession + nutation + equation of the equinoxes), `BodyMeanOfDate`, and `BodyTrueOfDate`. All three Earth variants compute rotations relative to the ICRS/J2000 inertial parent using SOFA precession-nutation models, covering the GCRF → MOD → TOD segment of the IAU 2006/2000A chain. None of them include the Earth Rotation Angle (ERA) step that rotates the true-of-date pole to the Terrestrial Intermediate Reference System (TIRS), or the polar-motion step that maps TIRS to ITRF. So `EARTH_ITRF93` is the only high-precision Earth body-fixed frame in ANISE 0.10.1; `IAU_EARTH_FRAME` uses a low-fidelity polynomial PCK rotation and isn't used here. The `earth_latest_high_prec.bpc` kernel is downloaded fresh from JPL at setup time and contains current Earth orientation parameters (UT1, polar motion). The NAIF orientation ID "ITRF93" refers to the frame definition standard used by NAIF/SPICE (IERS 1996 conventions), not to stale 1993-era data.

**Performance characteristics**: hifitime time-conversion tasks have no I/O or kernel-lookup overhead and are among the fastest tasks in the suite. ANISE attitude and orbit tasks carry per-call kernel-parsing overhead on the first call; subsequent calls within a benchmark run reuse the loaded `Almanac`. Numerical propagation tasks use Nyx's `RungeKutta4` fixed-step integrator at the same step size as the other Rust and Java baselines, so integrator-introduced residuals are comparable.

---

## GPU Comparison Suite (Brahe vs Astrojax)

A separate benchmark suite at `benchmarks/gpu_comparison/` looks at a different question: at what batch size does running the same astrodynamics computation on a GPU start to beat Brahe's Rust + rayon CPU implementation? The comparison target is [Astrojax](https://github.com/duncaneddy/astrojax), an experimental JAX-native astrodynamics library aimed at GPUs. Astrojax is derived from brahe's design but built on JAX so that GPU parallelization is direct. The design assumes batched work, since the GPU only pays off above some task-dependent crossover batch size.

The suite sweeps a geometric batch-size ladder (1 … 10^8 depending on task) for every task and reports throughput in operations per second. Multi-GPU cells are gated by a per-task `multigpu_min_batch()` below which `pmap` overhead dominates.

!!! note "Different from the comparative suite"

    The comparative suite above measures per-iteration latency on a fixed input with Orekit as the accuracy baseline. The GPU comparison suite measures throughput vs batch size across two libraries with no accuracy baseline. The two suites use different harnesses, different result schemas, and different reporting conventions, and aren't meant to be merged.

**Configurations**:

| Config | Backend | dtype | Parallelism |
|---|---|---|---|
| `brahe-rust-rayon` | Rust subprocess (`bench_gpu_rust`) | f64 | rayon, all CPU cores |
| `astrojax-cpu` | JAX in a spawned Python child (`JAX_PLATFORMS=cpu`) | f64 | `jit(vmap(...))` on one CPU device |
| `astrojax-gpu` | in-process JAX | f32 | `jit(vmap(...))` on one GPU |
| `astrojax-multigpu` | in-process JAX | f32 | `jit(vmap(...))` with `NamedSharding` across every visible GPU |

Brahe runs in f64 (its only mode). Astrojax on CPU is f64 for apples-to-apples comparison. Astrojax on GPU is f32 because that's the realistic deployment choice on tensor-core hardware, where you trade precision for throughput.

**Test environment**: AMD EPYC 7713 (64 physical / 128 logical cores), 503 GB RAM, Linux 6.17. 2× NVIDIA A100 80 GB PCIe (driver 595.71.05, CUDA 13.2). Brahe 1.5.2 (commit `715192ec`), Astrojax 0.8.0 (commit `98454ab`), JAX 0.10.1, Rust 1.93.1, Python 3.14.

!!! info "What an 'op' means"

    Throughput is reported in **operations per second**, where an *operation* is one user-facing call of the task, not a single integration step or low-level math op. The unit therefore varies by task family:

    - **Time conversions** (`time.utc_mjd_to_tt_mjd`): 1 op = 1 MJD-UTC → MJD-TT conversion.
    - **Coordinate transformations** (`coordinates.*`): 1 op = 1 conversion of one input vector (3-vector for geodetic / ECEF / ENZ; 6-vector for Keplerian / Cartesian state).
    - **Frame transformations** (`frames.gcrf_to_itrf`): 1 op = 1 `(epoch, state)` → ITRF state transformation, including the underlying precession / nutation / polar-motion rotations.
    - **SGP4 propagation** (`propagation.sgp4_iss_sweep`): 1 op = 1 propagation of the ISS TLE from its epoch to one `tsince_minutes` offset (one full SGP4 evaluation per element of the batch).
    - **Numerical / force-model propagation** (`propagation.numerical_twobody_j2`, `force_model.grav_5x5`): 1 op = 1 *complete orbit propagation* — i.e. **180 RK4 steps** spanning one ~90-minute LEO period at 30-second cadence.

### Peak Speedup vs Brahe-Rayon

For each (task, non-baseline-config) pair, this is the peak speedup of that config's throughput over `brahe-rust-rayon` at any batch size on the ladder. Values greater than 1 mean the config beats Brahe somewhere along its ladder.

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_gpu_speedup.csv') }}

</div>

<div class="plotly-embed medium">
  <iframe class="only-light" src="../figures/fig_gpu_peak_speedup_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_gpu_peak_speedup_dark.html"  loading="lazy"></iframe>
</div>

### Per-Task Throughput vs Batch Size

Each chart plots throughput (ops/s, log-log) for every config that completed at least one cell. The crossover point, where an Astrojax curve crosses above `brahe-rust-rayon`, is the batch size at which that config starts outperforming Brahe.

#### Time conversions

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_gpu_time_utc_mjd_to_tt_mjd.csv') }}

</div>

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_gpu_time_utc_mjd_to_tt_mjd_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_gpu_time_utc_mjd_to_tt_mjd_dark.html"  loading="lazy"></iframe>
</div>

#### Coordinate transformations

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_gpu_coordinates_geodetic_to_ecef.csv') }}

</div>

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_gpu_coordinates_geodetic_to_ecef_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_gpu_coordinates_geodetic_to_ecef_dark.html"  loading="lazy"></iframe>
</div>

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_gpu_coordinates_keplerian_to_cartesian.csv') }}

</div>

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_gpu_coordinates_keplerian_to_cartesian_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_gpu_coordinates_keplerian_to_cartesian_dark.html"  loading="lazy"></iframe>
</div>

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_gpu_coordinates_enz_to_azel.csv') }}

</div>

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_gpu_coordinates_enz_to_azel_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_gpu_coordinates_enz_to_azel_dark.html"  loading="lazy"></iframe>
</div>

#### Frame transformations

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_gpu_frames_gcrf_to_itrf.csv') }}

</div>

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_gpu_frames_gcrf_to_itrf_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_gpu_frames_gcrf_to_itrf_dark.html"  loading="lazy"></iframe>
</div>

#### SGP4 propagation

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_gpu_propagation_sgp4_iss_sweep.csv') }}

</div>

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_gpu_propagation_sgp4_iss_sweep_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_gpu_propagation_sgp4_iss_sweep_dark.html"  loading="lazy"></iframe>
</div>

#### Numerical two-body / J2 propagation

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_gpu_propagation_numerical_twobody_j2.csv') }}

</div>

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_gpu_propagation_numerical_twobody_j2_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_gpu_propagation_numerical_twobody_j2_dark.html"  loading="lazy"></iframe>
</div>

#### Full force model (5×5 spherical harmonic)

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_gpu_force_model_grav_5x5.csv') }}

</div>

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_gpu_force_model_grav_5x5_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_gpu_force_model_grav_5x5_dark.html"  loading="lazy"></iframe>
</div>

`force_model.grav_5x5` is the clearest example of the crossover. Brahe's hand-tuned spherical-harmonic gravity saturates at ~11k orbits/s above batch ~1k (limited by CPU thread count). Astrojax's RK4 + SH(5×5) graph is much slower per element at small batches because of JIT-compile overhead and a heavier per-step XLA graph, but it scales nearly linearly with batch size and crosses Brahe at roughly batch 30k. By batch 100k single-GPU is **6.5×** faster than Brahe and dual-GPU is **9.2×** faster. The gravity model is closed over inside `_propagate_one`, so each device JIT-compiles its own copy and only the per-orbit initial states are sharded.

### Notes on the GPU Comparison

**dtype heterogeneity**: The Brahe (f64) vs Astrojax-GPU (f32) mismatch is intentional. F32 on A100 tensor cores is the realistic deployment choice; forcing astrojax-GPU to f64 would halve its throughput and turn the comparison into "JAX overhead vs Rust" instead of "what's realistically achievable on A100s". Astrojax-CPU stays at f64 so the JAX-vs-Rust comparison on a CPU device is apples-to-apples.

**Spawn-isolated `astrojax-cpu`**: JAX can't host CPU and GPU devices in the same Python process once it's initialized. The runner launches every astrojax-cpu cell in a `multiprocessing.spawn`-ed child with `JAX_PLATFORMS=cpu` set on the child, which isolates the CPU JAX init from the parent (which has CUDA jaxlib loaded) at the cost of one process spawn plus JIT-compile per cell.

**Input generation is vectorized**: Per-row inputs come from `numpy.random.default_rng(seed)` (one `uniform(low, high, batch_size)` call per column) rather than Python loops. The same seed produces the same N distinct rows across runs. At batch 10^7 generation takes ~2 s; at batch 10^8 (the largest cell, used by the time tasks) it takes ~4 s. This is setup overhead and is not part of the timed window.

**Per-task batch ladders are tuned**: cheap tasks (time conversion, coordinate transforms) sweep up to 10^7 or 10^8 to find the GPU crossover; heavier per-element tasks (frames, numerical propagation, full force model) cap at 10^4–10^6 so individual cells stay within the 90 s wall-clock budget.

**Astrojax data alignment**: both backends use Brahe's bundled `data/eop/finals.all.iau2000.txt` and `data/space_weather/sw19571001.txt`. Astrojax's `load_eop_from_file` and `load_sw_from_file` accept Brahe's files directly with no conversion step. JAX 64-bit mode is enabled via `JAX_ENABLE_X64=1` so the EOP table preserves f64 precision, and integrator-using kernels also call `astrojax.set_dtype(jnp.float64)` so the RK4 scan body's carry input and output dtypes match.

---

## Reproducing These Results

**Comparative benchmarks** (Java/Python/Rust/Basilisk/GMAT):

> **GMAT setup is opt-in via environment variable.** To include GMAT in the comparison, install [GMAT R2026a](https://github.com/nasa/GMAT) locally and export `GMAT_ROOT_PATH` pointing at the install root (the directory containing `bin/`, `data/`, `api/`) before running `just bench-compare-setup`. Example (macOS): `export GMAT_ROOT_PATH="/Applications/GMAT R2026a"`. If `GMAT_ROOT_PATH` is unset, the setup recipe and all benchmark runs skip GMAT silently — the other four baselines run normally.

> **Prerequisites**: `brew install just uv openjdk` (macOS). The `just` recipes auto-detect Homebrew's openjdk and Rust's `~/.cargo/bin/cargo`, so no manual `JAVA_HOME` export or symlink is needed. If Rust isn't installed yet, follow the [rustup instructions](https://rustup.rs/). The Rust toolchain is required to build the Brahe and Nyx benchmark binaries.

```bash
# Optional: point at a local GMAT install to include GMAT in the comparison.
export GMAT_ROOT_PATH="/Applications/GMAT R2026a"

# One-time setup: build Java/Rust implementations, install Basilisk wheel,
# download OreKit data, and (if GMAT_ROOT_PATH is set) generate GMAT's
# api_startup_file.txt via the bundled BuildApiStartupFile.py.
just bench-compare-setup

# Run benchmarks, generate figures + CSV tables, and stage artifacts for commit
# (perf takes --iterations, accuracy takes --samples; both default to 100, but
#  200 gives noticeably tighter ±1σ bars on the performance charts.)
just bench-compare --iterations 200 --seed 42
just bench-compare-accuracy --samples 200 --seed 42
BRAHE_FIGURE_OUTPUT_DIR=./docs/figures/ uv run python plots/fig_comparative_benchmarks.py

# Review staged changes and commit
git status
git commit -m "Update benchmark data"
```

Individual steps can also be run separately:

```bash
# Run benchmarks only (results saved to benchmarks/comparative/results/)
just bench-compare --iterations 200 --seed 42
just bench-compare-accuracy --samples 200 --seed 42

# Regenerate figures and tables from existing results (without re-running benchmarks)
python plots/fig_comparative_benchmarks.py
```

**Access benchmark** (Brahe vs Skyfield):

```bash
uv run scripts/benchmark_access_three_way.py --n-locations 100 --seed 42 --output chart.html --plot-style scatter --csv accesses.csv
```

**GPU comparison benchmarks** (Brahe-Rust vs Astrojax on CPU / single GPU / multi-GPU):

> **Prerequisites**: CUDA-capable GPU(s) and an installed CUDA driver. The `gpu-comparison` extra in `pyproject.toml` carries only the Python tooling deps (`typer`, `rich`, `psutil`). Astrojax and the CUDA-capable jaxlib are installed by the `just bench-gpu-install` recipe — astrojax editable from `~/repos/astrojax` when present, otherwise from PyPI; jaxlib matched to the detected CUDA major version. They are intentionally NOT pinned in `pyproject.toml` so `uv sync --all-extras --frozen` works in CI without a sibling astrojax checkout.

```bash
# One-time setup: install brahe[gpu-comparison] + astrojax + a CUDA-matched
# jaxlib. NO_LOCAL=1 forces the PyPI astrojax install regardless of
# ~/repos/astrojax presence.
just bench-gpu-install
just bench-gpu-build                          # compiles bench_gpu_rust

# Run the full suite — 8 tasks × 4 configs × geometric batch ladder.
# JAX_ENABLE_X64=1 is required so EOP-backed tasks preserve f64 precision
# on the astrojax-CPU side.
JAX_ENABLE_X64=1 just bench-gpu --budget 90 --iterations 5

# Regenerate figures + CSVs from the most recent results JSON.
BRAHE_FIGURE_OUTPUT_DIR=./docs/figures/ uv run python plots/fig_gpu_comparison.py

# Inspect a results file as a per-task table without re-running:
just bench-gpu-inspect benchmarks/gpu_comparison/results/run_<timestamp>.json
```

Individual cells can be triaged with `just bench-gpu-cell <task> <config> <batch>`. Use `just bench-gpu-list` to see registered tasks.
