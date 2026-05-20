## Benchmarks

Brahe is benchmarked against **OreKit 12.2** (Java), the most widely used open-source astrodynamics library, across 32 tasks spanning 8 modules. All three implementations — Java (OreKit), Python (Brahe), and Rust (Brahe) — are given identical inputs (seed=42, 100 iterations) and their outputs are compared for both performance and numerical accuracy. Brahe is additionally compared against [**Basilisk**](https://github.com/AVSLab/basilisk) (AVS Lab) on the 14 tasks across four modules (attitude, orbits, frames, coordinates, and propagation) where the two libraries' user-facing APIs overlap.

!!! tip

    These benchmarks are meant to highlight the consistency and agreement of Brahe with other astrodynamics software libraries and enable users to make informed selection trades. The purpose is NOT state that one offering is superior to another. These benchmarks, and even brahe itself wouldn't exist without the excellent work of other projects trail-blazing an open astrodynamics software ecosystem. As programming and technology evolves, it is helpful to have multiple viable solutions so that users have flexibility to select a tool that works well for their system and problem.

### Methodology

**Languages and Libraries**:

- **Java**: OreKit 12.2 on OpenJDK 21
- **Python**: Brahe Python bindings (PyO3)
- **Rust**: Brahe native Rust library
- **Basilisk**: AVS Lab Basilisk (`bsk` Python wheel from PyPI), imported in-process by the Python runner; participates on a 14-task subset.

**Test Environment**: 2021 MacBook Pro, Apple M1 Max, 64 GB RAM

**Protocol**: Each task is run 100 iterations with a fixed random seed. Mean execution time is reported. Accuracy is measured by comparing outputs element-wise against the Java (OreKit) reference implementation.

### Performance Overview

The table below summarizes average speedup relative to Java (OreKit) for each module. Values greater than 1.0× indicate Brahe is faster; values less than 1.0× indicate OreKit is faster.

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_overview.csv') }}

</div>

Per-module average Python speedups (relative to OreKit) range from 1.3× to 46×; Rust ranges from 4.7× to 341×. Force-model and access tasks are closest to parity; time and attitude show the largest separation.

<div class="plotly-embed x-tall">
  <iframe class="only-light" src="../figures/fig_bench_speedup_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_bench_speedup_dark.html"  loading="lazy"></iframe>
</div>

The chart below restricts the comparison to the 14 tasks where Basilisk participates and uses Basilisk as the baseline. See "Notes on Basilisk Comparisons" below for caveats specific to that comparison (gravity coefficient sources, frame definitions, default integrator differences).

<div class="plotly-embed tall">
  <iframe class="only-light" src="../figures/fig_bench_speedup_vs_basilisk_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_bench_speedup_vs_basilisk_dark.html"  loading="lazy"></iframe>
</div>

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

**Accuracy**: TAI, TT, and GPS conversions show **zero error** — all three implementations use identical offset constants. UT1 shows a maximum absolute error of ~1.0 µs, attributable to different Earth Orientation Parameter (EOP) sources and interpolation methods between OreKit and Brahe.

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

**Accuracy**: All coordinate transformations agree to **sub-nanometer** precision (< 2 nm). Maximum absolute errors are on the order of $10^{-9}$ m, reflecting only floating-point representation differences.

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_accuracy_coordinates.csv') }}

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

**Accuracy**: Quaternion ↔ rotation matrix conversions agree to **machine epsilon** ($< 10^{-15}$). Quaternion ↔ Euler angle conversions also agree to machine epsilon.

The `euler_angle_to_quaternion` task shows a large apparent max absolute error of 0.67 in the raw comparison data. This is a **quaternion sign convention artifact**, not a real error — quaternions $q$ and $-q$ represent the same rotation, so implementations may validly return either sign. The actual rotations are equivalent.

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

**Accuracy**:

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_accuracy_frames.csv') }}

</div>

Both implementations use the IAU 2006/2000A model. Residuals are on the order of tens of centimeters, reflecting EOP interpolation and nutation series truncation differences.

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

**Accuracy**: Both conversion directions agree to **sub-millimeter** precision. Maximum absolute errors are on the order of $10^{-8}$ m (~24 nm), with relative errors below $10^{-11}$.

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_accuracy_orbits.csv') }}

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

Propagation task speedups (relative to OreKit) range from roughly parity (RK4 + 20×20 + Sun/Moon) to ~32× (SGP4 trajectory generation, Rust). Keplerian and SGP4 trajectory generation are in the 4–32× range; the RK4 + 5×5 case is ~3.6×, and RK4 + 80×80 + drag + SRP is ~1.4×.

**Accuracy**:

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_accuracy_propagation.csv') }}

</div>

Keplerian propagation agrees at the nanometer level. Numerical propagation diverges at the centimeter level, reflecting different integrator implementations and step-size strategies. SGP4 divergence is on the order of tens of meters; the original Fortran, OreKit Java, and Brahe Rust SGP4 implementations make slightly different numerical choices in the deep-space and near-Earth branch logic.

For the high-fidelity RK4 cases (`RK4 + 5x5 Gravity`, `RK4 + 20x20 + Sun/Moon`, `RK4 + 80x80 + Drag + SRP`), both implementations are fed identical EGM2008 gravity coefficients (brahe's packaged `EGM2008_360.gfc`, loaded on the OreKit side via `ICGEMFormatReader`), DE-440 ephemerides for third-body perturbers, identical spacecraft parameters (1000 kg, 10 m² area, Cd=2.2, Cr=1.3), and matched GCRF↔ITRF rotations (IAU 2006/2000A on both sides). The 5×5 and 20×20+Sun/Moon cases agree at the µm level over one LEO revolution — small enough that the residual reflects only floating-point summation differences. The 80×80 + drag + SRP case diverges at the meter level, dominated by NRLMSISE-00 implementation and SRP eclipse-model differences rather than gravity.

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

**Accuracy**:

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_accuracy_force_model.csv') }}

</div>

Point-mass gravity agrees to machine epsilon (10⁻¹⁶ m/s²) — both implementations use identical $GM_\\oplus$. Spherical-harmonic gravity agrees to ~10⁻¹² m/s² for both 20×20 and 80×80. Third-body Sun and Moon accelerations agree to ~10⁻¹⁴ m/s², with DE-440 ephemerides used on both sides.

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

**Accuracy**: There are slight variations in access window start/end time. Both implementations agree on the total number of access windows.

---

### Access Computation (Brahe vs Skyfield)

In addition to the OreKit comparison above, Brahe is also benchmarked against **Skyfield**, a popular Python astronomy library, for access computation. This benchmark focuses on Brahe's serial vs parallel execution modes and Python bindings vs native Rust performance.

The benchmark randomly samples 100 ground station locations and computes all satellite accesses over a 48-hour window. Access start and end times agree to within one second between Brahe and Skyfield.

<div class="center-table" markdown="1">
| Implementation         | Avg Time  | vs Skyfield    | vs Brahe-Py-Serial |
|------------------------|-----------|----------------|---------------------|
| Brahe-Rust (parallel)  |   1.37ms  | 3.2x faster    | 23.0x faster        |
| Brahe-Python (parallel)|   2.40ms  | 1.8x faster    | 13.1x faster        |
| Brahe-Rust (serial)    |   2.79ms  | 1.6x faster    | 11.2x faster        |
| Skyfield               |   4.44ms  | baseline       | 7.1x faster         |
| Brahe-Python (serial)  |  31.41ms  | 7.1x slower    | baseline            |
</div>

The parallel implementations leverage multiple CPU cores to handle multiple ground stations simultaneously. Skyfield's performance is impressive, being only marginally slower than Brahe's serial Rust implementation despite being written in pure Python.

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_access_benchmark_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_access_benchmark_dark.html"  loading="lazy"></iframe>
</div>

---

### Notes on Basilisk Comparisons

Basilisk participates in 14 of 32 tasks. The gap is API-driven, not capability-driven: brahe's benchmark suite exercises atomic conversion calls (frame transforms, force-model accelerations, time-scale conversions, single-step Keplerian propagation), and Basilisk does not expose all of those at the user-API level — its dynamics modules compute them internally during simulation.

**Where Basilisk participates**: 4 attitude conversions (`RigidBodyKinematics`), 2 orbital-element conversions (`orbitalMotion.elem2rv` / `rv2elem`), 2 frame transformations (via the bundled `pyswice` SPICE Toolkit, J2000 ↔ ITRF93), 2 geodetic/ECEF coordinate conversions (also `pyswice`), and 4 numerical orbit propagation cases (`spacecraft.Spacecraft` + `simIncludeGravBody` with RK4 default integrator).

**Where Basilisk does not participate**: SGP4 propagation (no SGP4 in Basilisk's core), analytical Keplerian propagation (`orbitalMotion.elem2rv` is a single closed-form function call, not a propagator object), atomic force-model accelerations (computed inside dynamics modules), time-scale conversions (Basilisk relies on SPICE's `unitim_c` — would measure SPICE Toolkit, not a peer time-scale library), and access calculations (depend on SGP4).

**Frame definitions**: Basilisk-via-pyswice transforms between `J2000` and `ITRF93` using NAIF's high-precision Earth orientation binary PCK (`earth_latest_high_prec.bpc`, downloaded automatically by `bench-compare-setup`). OreKit uses `EME2000` (≡ J2000 to sub-meter precision) and `ITRF` (IERS 2010 conventions). ITRF93 follows the IERS 1996 conventions, so expect kilometer-scale ECEF position differences vs. the Java baseline — this is a real difference between IERS conventions, not implementation noise.

**Propagation methodology**: Basilisk's `SimBaseClass` setup cost is included in the per-iteration timing (matches the OreKit / Rust pattern of timing the full `run`). Basilisk's inertial output (`r_BN_N`, `v_BN_N`, J2000-equatorial) is transformed to GCRF inside the benchmark using `brahe.state_eme2000_to_gcrf` before the accuracy comparison. Basilisk's default Earth `mu` is overridden from `3.986004360e14` to `3.986004418e14` so all three baselines see the same central-body parameter. Gravity coefficients come from `GGM03S` (Basilisk-bundled, up to degree 180) versus EIGEN-5C (Orekit) and brahe's native EGM family. The two-body task uses each library's default high-accuracy integrator (Java: DP8(5,3) adaptive; brahe: DP54 adaptive) but Basilisk's default is fixed-step RK4, so the basilisk row shows accumulated truncation error of ~tens of meters over one LEO orbit — intrinsic to RK4-at-60s, not a Basilisk bug. The three RK4 force-model tasks use RK4 at the same step in all four implementations.

**Anomaly convention**: Basilisk's `orbitalMotion.ClassicElements` uses true anomaly in radians; brahe/Orekit use mean anomaly in degrees per the existing benchmark convention. Each library is timed on its native call; conversion happens outside the timed region. Quaternion convention (scalar-first `[w, x, y, z]`) is already consistent across all baselines; Basilisk's `RigidBodyKinematics.EP2C` and brahe's `Quaternion.to_rotation_matrix()` use the same passive-rotation (DCM) convention.

**Atmospheric drag inputs**: For the 80×80 full task, Basilisk's `msisAtmosphere` is driven with representative quiet-Sun space-weather values (Ap=8, F10.7=110), not by interpolating `SpaceWeather-All-v1.2.txt` the way Orekit's `CssiSpaceWeatherData` does. This is what a typical Basilisk user would do without bespoke SW preparation; it contributes additional bounded divergence from the Java baseline on that task.

---

### Reproducing These Results

**Comparative benchmarks** (Java/Python/Rust):

```bash
# One-time setup: build Java/Rust implementations, download OreKit data
just bench-compare-setup

# Run benchmarks, generate figures + CSV tables, and stage artifacts for commit
just bench-compare-publish --iterations 100 --seed 42

# Review staged changes and commit
git status
git commit -m "Update benchmark data"
```

Individual steps can also be run separately:

```bash
# Run benchmarks only (results saved to benchmarks/comparative/results/)
just bench-compare --iterations 100 --seed 42

# Regenerate figures and tables from existing results (without re-running benchmarks)
python plots/fig_comparative_benchmarks.py
```

**Access benchmark** (Brahe vs Skyfield):

```bash
uv run scripts/benchmark_access_three_way.py --n-locations 100 --seed 42 --output chart.html --plot-style scatter --csv accesses.csv
```
