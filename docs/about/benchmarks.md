## Benchmarks

To validate Brahe's consistency with other astrodynamics software, we've developed a benchmark suite that exercises a broad range of astrodynamics-related tasks across multiple libraries. The suite is designed to be extensible, so new tasks and libraries can be added over time. The current iteration includes 32 tasks spanning 8 modules: time, coordinates, attitude, frames, orbits, propagation, force model, and access computation.

The primary "source of truth" is  [**OreKit 13.1.5**](https://github.com/CS-SI/Orekit) (Java), one of (if not the) most widely used open-source astrodynamics library. We additionally compare against [**GMAT R2026a**](https://github.com/nasa/GMAT) (NASA Goddard's General Mission Analysis Tool) on 31 of the 32 benchmark tasks, [**Basilisk**](https://github.com/AVSLab/basilisk) (AVS Lab) on the 14 tasks across four modules (attitude, orbits, frames, coordinates, and propagation) where the two libraries' user-facing APIs overlap, and [Nyx](https://github.com/nyx-space/nyx)/[ANISE](https://github.com/nyx-space/anise) (Nyx Space) on 28 tasks where the libraries have comparable APIs. The benchmarks report both performance and accuracy metrics for each library on each task, with OreKit as the single reference baseline for accuracy comparisons.

!!! tip

    These benchmarks are meant to highlight the consistency and agreement of Brahe with other astrodynamics software libraries and enable users to make informed selection trades. The purpose is NOT state that one offering is superior to another. These benchmarks, and even brahe itself wouldn't exist without the excellent work of other projects trail-blazing an open astrodynamics software ecosystem. As programming and technology evolves, it is helpful to have multiple viable solutions so that users have flexibility to select a tool that works well for their system and problem.

Libraries have different design goals, philosophies, and use cases. For example, Basilisk is a full spacecraft simulation environment with a Python interface, so its API is not designed for atomic conversion calls or single-step propagation — it computes those internally during simulation. GMAT is a script-driven mission analysis tool with a Python API layered on top of the script interface, so some tasks are implemented via script execution rather than direct API calls. Brahe's benchmark suite exercises atomic conversion calls (frame transforms, force-model accelerations, time-scale conversions, single-step Keplerian propagation) that are not necessarily exposed at the user-API level in these other libraries; the benchmarks only include tasks where the comparison is apples-to-apples on the API call being timed.

One of the benefits of these benchmarks is that they show broad consistency in terms of outputs and accuracy across multiple libraries, which gives users confidence that the implementations are correct and that they can expect similar results regardless of which library you choose. The performance comparisons are also informative, but they should be interpreted in the context of the different design goals and use cases of each library.

### Methodology

**Languages and Libraries**:

- **Java**: [OreKit 13.1.5](https://github.com/CS-SI/Orekit) on OpenJDK 21
- **Basilisk**: [AVS Lab Basilisk](https://github.com/AVSLab/basilisk) (`bsk` Python wheel from PyPI), imported in-process by the Python runner; participates on a 14-task subset.
- **GMAT**: [GMAT R2026a](https://github.com/nasa/GMAT) (`gmatpy` API), accessed via a local GMAT install pointed to by `GMAT_ROOT_PATH`; participates on a 31-task subset.
- **Nyx Space**: [Nyx 2.3.1](https://github.com/nyx-space/nyx), [ANISE 0.9.6](https://github.com/nyx-space/anise), and [hifitime](https://github.com/nyx-space/hifitime) Rust libraries. Nyx/ANISE participate on a 28-task subset.
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

Per-module average Python speedups (relative to OreKit) range from ~0.85× (force-model) to ~22× (time); Rust ranges from ~5× (access) to ~210× (attitude). Force-model and access tasks are closest to parity; time and attitude show the largest separation.

<div class="plotly-embed x-tall">
  <iframe class="only-light" src="../figures/fig_bench_speedup_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_bench_speedup_dark.html"  loading="lazy"></iframe>
</div>

OreKit is the single performance reference. Per-baseline speedup charts using Basilisk or GMAT as the denominator are not shown because each ratio is derivable from this chart and the additional framings tend to read as "winner" comparisons rather than the consistency check these benchmarks are intended to support.

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

Geodetic and geocentric outputs combine angular and altitude components in different units; the benchmark converts angle residuals to surface-arc distance so each row reports a single position-equivalent error in meters. The Orekit-vs-GMAT row on the geodetic conversions is dominated by GMAT's choice of equatorial radius (6378.1363 km vs WGS84's 6378.137 km, a ~0.7 m offset documented in `docs/about/benchmark-deviations/coordinates_geodetic_to_ecef.md`); the implementation-level agreement on the WGS84 baselines is sub-nanometer.

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

Five tasks evaluating a single acceleration term at a fixed spacecraft state and epoch (no integrator, no time stepping). These benchmarks isolate the force-model implementations from the propagator: any disagreement here is attributable to the force-model calculation itself (gravity coefficients, third-body ephemeris, frame transformations), not to numerical integration.

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

Basilisk participates in 14 of 32 tasks. The gap is API-driven, not capability-driven: brahe's benchmark suite exercises atomic conversion calls (frame transforms, force-model accelerations, time-scale conversions, single-step Keplerian propagation), and Basilisk does not expose all of those at the user-API level — its dynamics modules compute them internally during simulation.

**Where Basilisk participates**: 4 attitude conversions (`RigidBodyKinematics`), 2 orbital-element conversions (`orbitalMotion.elem2rv` / `rv2elem`), 2 frame transformations (via the bundled `pyswice` SPICE Toolkit, J2000 ↔ ITRF93), 2 geodetic/ECEF coordinate conversions (also `pyswice`), and 4 numerical orbit propagation cases (`spacecraft.Spacecraft` + `simIncludeGravBody` with RK4 default integrator).

**Where Basilisk does not participate**: SGP4 propagation (no SGP4 in Basilisk's core), analytical Keplerian propagation (`orbitalMotion.elem2rv` is a single closed-form function call, not a propagator object), atomic force-model accelerations (computed inside dynamics modules), time-scale conversions (Basilisk relies on SPICE's `unitim_c` — would measure SPICE Toolkit, not a peer time-scale library), and access calculations (depend on SGP4).

**Frame definitions**: Basilisk-via-pyswice transforms between `J2000` and `ITRF93` using NAIF's high-precision Earth orientation binary PCK (`earth_latest_high_prec.bpc`, downloaded automatically by `bench-compare-setup`). OreKit uses `EME2000` (≡ J2000 to sub-meter precision) and `ITRF` (IERS 2010 conventions). ITRF93 follows the IERS 1996 conventions, so expect kilometer-scale ECEF position differences vs. the Java baseline.

**Propagation methodology**: Basilisk's `SimBaseClass` setup cost is included in the per-iteration timing (matches the OreKit / Rust pattern of timing the full `run`). Basilisk's inertial output (`r_BN_N`, `v_BN_N`, J2000-equatorial) is captured raw inside the timed region; the J2000 → GCRF frame-bias transform via `brahe.state_eme2000_to_gcrf` is applied after `time_iterations` returns, so the timed work is purely Basilisk (this matches the GMAT adapter's `mj2000_to_gcrf` post-processing). Basilisk's default Earth `mu` is overridden from `3.986004360e14` to `3.986004418e14` so all three baselines see the same central-body parameter. Gravity coefficients come from `GGM03S` (Basilisk-bundled, up to degree 180) versus EIGEN-5C (Orekit) and brahe's native EGM family. The numerical two-body and three RK4 force-model tasks all use fixed-step Classical Runge-Kutta 4 (RK4) at the same step size in every implementation, so the residual reflects gravity-coefficient sources and floating-point ordering rather than integrator-precision differences.

**Anomaly convention**: Basilisk's `orbitalMotion.ClassicElements` uses true anomaly in radians; brahe/Orekit use mean anomaly in degrees per the existing benchmark convention. Each library is timed on its native call; conversion happens outside the timed region. Quaternion convention (scalar-first `[w, x, y, z]`) is already consistent across all baselines; Basilisk's `RigidBodyKinematics.EP2C` / `C2EP` / `EP2Euler321` / `euler3212EP` and brahe's `Quaternion` primitives use the same passive-rotation (DCM) convention, so all four attitude conversions are timed on native library calls.

**Atmospheric drag inputs**: For the 80×80 full task, Basilisk's `msisAtmosphere` is driven with representative quiet-Sun space-weather values (Ap=8, F10.7=110), not by interpolating `SpaceWeather-All-v1.2.txt` the way Orekit's `CssiSpaceWeatherData` does. This is what a typical Basilisk user would do without bespoke SW preparation; it contributes additional bounded divergence from the Java baseline on that task.

---

### Notes on GMAT Comparisons

[GMAT R2026a](https://github.com/nasa/GMAT) (the General Mission Analysis Tool, developed by NASA Goddard) participates in 31 of the 32 benchmark tasks. The single exclusion is `coordinates.ecef_to_azel`: GMAT's `GroundStation` exposes no azimuth/elevation accessor via the `gmatpy` API, and the script-based `Topocentric` coordinate system path segfaults under `gmat.RunScript()`. The only working approach is pure-Python ENZ rotation, which doesn't exercise GMAT code and would be dishonest to label as a GMAT comparison.

**Where GMAT participates**: all 5 time tasks, 4 of 5 coordinates tasks, all 4 attitude tasks, both frames tasks, both orbits tasks, all 8 propagation tasks, all 5 force-model tasks, and the access task — 31 tasks in total.

**Setup**: Install GMAT R2026a, export `GMAT_ROOT_PATH` pointing at the install root (the directory containing `bin/`, `data/`, `api/`), and run `just bench-compare-setup`. The setup recipe generates `api_startup_file.txt` via GMAT's bundled `BuildApiStartupFile.py`. When `GMAT_ROOT_PATH` is unset, GMAT comparisons are silently skipped.

**Frame and coordinate disclosure**: GMAT's `EarthFixed` body-fixed frame uses FK5 reference theory with IAU 1980 nutation; brahe and OreKit use the IERS 2010 ITRF realization. Position errors of ~10 m at LEO are attributable to frame-definition choice, not implementation precision. For geodetic coordinate conversions, GMAT uses an Earth equatorial radius of 6378.1363 km versus WGS84's 6378.137 km — accounting for ~0.7 m position error.

**Quaternion convention**: GMAT uses scalar-last `[q1, q2, q3, q4]`; the benchmark canonical is scalar-first `[w, x, y, z]`. Reordering is applied outside the timed region. All four attitude tasks (quaternion ↔ rotation matrix, quaternion ↔ Euler ZYX) call GMAT's native `AttitudeConversionUtility` primitives — `ToCosineMatrix`, `ToQuaternion`, `ToEulerAngles(rv, 3, 2, 1)`.

**Time scales**: GMAT's `A1ModJulian` epoch is rooted at 1941-01-05 (JD 2430000.0), not the standard MJD epoch of 1858-11-17 (JD 2400000.5). The benchmark applies the appropriate offset for time-conversion inputs. GMAT has no GPS time-system enum; GPS times are derived as TAI − 19 s. GMAT's Gregorian time parser has millisecond precision, contributing ~0.5 ms to time-conversion accuracy comparisons.

**Propagation methodology**: Per-iteration construction of `Spacecraft` + `ForceModel` + `Propagator` is included in the timing — an accurate representation of how a GMAT user would use the API. This matches Basilisk's per-iteration setup methodology. The `EarthMJ2000Eq` → GCRF transform is applied post-timing using `brahe.state_eme2000_to_gcrf`.

**Units**: GMAT uses km / km/s / km/s² internally for position, velocity, and acceleration; the benchmark canonical is SI (m / m/s / m/s²). Scaling by 10³ is applied outside the timed region.

**Gravity coefficients and atmosphere sources**: GMAT uses bundled JGM-2 / EGM-96 coefficients depending on degree; the JGM-2 file supports up to degree 70, so 80×80 tasks use EGM-96. Drag uses MSISE-90 from GMAT's bundled space-weather data. OreKit uses EIGEN-6S, Basilisk uses GGM03S, brahe uses per-task bundled data — differences reflect data-source variation, not implementation precision.

**SGP4 propagation and access tasks**: These tasks are implemented via GMAT's script-based interface (`gmat.LoadScript` + `gmat.RunScript`) because the SGP4 and ContactLocator plugins do not activate cleanly through the direct `gmatpy` API path. This is a GMAT integration detail; results are correct.

---

### Notes on Nyx / ANISE Comparisons

[Nyx 2.3.1](https://github.com/nyx-space/nyx) is a Rust astrodynamics library designed for mission design, trajectory optimization, and orbit determination. [ANISE 0.9.6](https://github.com/nyx-space/anise) is its companion library for reference-frame and ephemeris operations; it provides the `Almanac` abstraction that loads NAIF SPICE-compatible kernels (SPK, PCK, FK) and exposes attitude, rotation, and ephemeris queries in native Rust. [hifitime 4.x](https://github.com/nyx-space/hifitime) is the precision time library shared by both and underpins all epoch and duration arithmetic.

The Nyx/ANISE baseline runs as a separate Rust binary (compiled from `benchmarks/nyx/`), invoked as a subprocess by the Python runner in the same way that the Java OreKit binary is invoked. This mirrors the brahe-Rust execution model — the Rust binary receives task parameters via stdin JSON and writes results to stdout JSON; the Python harness handles timing and statistical aggregation identically to all other baselines.

**Where Nyx/ANISE participates**: 28 of 32 tasks. The 4 dropped tasks are listed below.

**Dropped tasks**:

- `coordinates.geocentric_to_ecef` and `coordinates.ecef_to_geocentric`: Brahe defines these as pure spherical-coordinate conversions (latitude-longitude-radius ↔ XYZ without ellipsoidal flattening). ANISE 0.9.6 has no equivalent API that we found — its only coordinate conversion path is geodetic (WGS84 ellipsoid). Implementing this with hand-rolled trigonometry would not exercise any Nyx/ANISE code, so these tasks are dropped.
- `force_model.accel_point_mass_gravity`: Brahe exposes a standalone single-body Newtonian evaluation. Nyx 2.3.1 does not expose this as a public API — the two-body term is internal to `OrbitalDynamics::eom`. Implementing it for Nyx would reduce to hand-rolled `μ·r/r³` rather than library code, so this task is dropped.
- `access.sgp4_access`: Brahe's `location_accesses` API finds all satellite-to-ground-station contact windows over an interval. Nyx 2.3.1 has no direct equivalent — the `eclipse` module is unrelated, and building a window-finder on top of Nyx's propagator would require reproducing brahe's search logic rather than comparing library APIs. This task is dropped.

**Earth orientation parameters**: ANISE 0.9.6 does not load IERS `finals2000A.all` data. Earth rotation is driven by ANISE's own `earth_latest_high_prec.bpc` binary PCK, which is downloaded at setup time by `bench-compare-setup`. Frame-transform residuals of roughly 1–3 m relative to OreKit (which uses IERS finals2000A.all) reflect this BPC-vs-IERS-finals2000A algorithmic difference rather than a precision defect — it is the variable under test for the frames tasks.

**Gravity coefficient file format**: Nyx supports SHADR and COF gravity coefficient formats but not ICGEM. Brahe ships `EGM2008_360.gfc` in ICGEM format. The Nyx benchmark implementation parses the ICGEM file and writes a temporary SHADR file at benchmark startup; coefficient values are numerically identical to the source. No accuracy impact is expected from this format translation.

**Atmospheric drag model substitution**: Nyx 2.3.1's public atmosphere models are `ConstantDrag`, an exponential model, and US Standard Atmosphere 1976. Neither NRLMSISE-00 nor Harris-Priester is exposed as a public API. The `propagation.numerical_rk4_grav80x80_full` task uses US Standard Atmosphere 1976 for drag. OreKit uses NRLMSISE-00 for the same task, so accuracy residuals on this specific task are larger than on the other propagation tasks — this is expected given the different atmosphere models and not indicative of a numerical precision difference.

**Time tasks**: All five time tasks are implemented directly against hifitime's Epoch API without additional library layers. hifitime uses TAI as its internal representation with UT1 obtained from a bundled IERS file; conversions to GPS, TT, UTC, and UT1 match OreKit at nanosecond precision across the swept epoch range.

**Attitude tasks**: Attitude conversions use ANISE's `DCM`, `EulerParameter`, and matrix-product primitives. The DCM ↔ Euler-angle path goes through ANISE's ZYX decomposition; the quaternion ↔ DCM path uses ANISE's `EulerParameter::to_dcm` / `DCM::to_euler_parameters`. All four tasks are implemented on native ANISE calls.

**Frame tasks**: The two frame-transformation tasks use `Almanac::transform_to` between `EARTH_J2000` and `EARTH_ITRF93` frames, loading the same `earth_latest_high_prec.bpc` PCK used by Basilisk. The frame-definition note from the Basilisk section applies here as well: ITRF93 follows IERS 1996 conventions versus OreKit's IERS 2010 ITRF, contributing kilometer-scale ECEF position differences on the accuracy sweep.

**Performance characteristics**: hifitime time-conversion tasks have no I/O or kernel-lookup overhead and are among the fastest tasks in the suite. ANISE attitude and orbit tasks carry per-call kernel-parsing overhead on the first call; subsequent calls within a benchmark run reuse the loaded `Almanac`. Numerical propagation tasks involve Nyx's `RungeKutta4` fixed-step integrator with the same step size used by the other Rust and Java baselines, so integrator-introduced residuals are comparable. SGP4 tasks use the `sgp4` Rust crate directly — the same crate used by brahe-Rust.

---

### Reproducing These Results

**Comparative benchmarks** (Java/Python/Rust/Basilisk/GMAT):

> **GMAT setup is opt-in via environment variable.** To include GMAT in the comparison, install [GMAT R2026a](https://github.com/nasa/GMAT) locally and export `GMAT_ROOT_PATH` pointing at the install root (the directory containing `bin/`, `data/`, `api/`) before running `just bench-compare-setup`. Example (macOS): `export GMAT_ROOT_PATH="/Applications/GMAT R2026a"`. If `GMAT_ROOT_PATH` is unset, the setup recipe and all benchmark runs skip GMAT silently — the other four baselines run normally.

> **Java prerequisite (macOS)**: The OreKit baseline requires Java. Install via [Amazon Corretto](https://aws.amazon.com/corretto/) (links `/Library/Java/JavaVirtualMachines/` automatically) or via Homebrew: `brew install openjdk`. With Homebrew, the system Java shim (`/usr/bin/java`) may not activate automatically — run `sudo ln -sfn $(brew --prefix openjdk)/libexec/openjdk.jdk /Library/Java/JavaVirtualMachines/openjdk.jdk` to link it. The `just bench-compare-setup` and `just bench-compare` recipes auto-detect the Homebrew path as a fallback if the shim is missing.

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
