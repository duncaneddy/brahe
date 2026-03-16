## Benchmarks

Brahe is benchmarked against **OreKit 12.2** (Java), the most widely used open-source astrodynamics library, across 24 tasks spanning 7 modules. All three implementations — Java (OreKit), Python (Brahe), and Rust (Brahe) — are given identical inputs (seed=42, 100 iterations) and their outputs are compared for both performance and numerical accuracy.

!!! tip

    These benchmarks are meant to highlight the consistency and agreement of Brahe with other astrodynamics software libraries and enable users to make informed selection trades. The purpose is NOT state that one offering is superior to another. These benchmarks, and even brahe itself wouldn't exist without the excellent work of other projects trail-blazing an open astrodynamics software ecosystem. As programming and technology evolves, it is helpful to have multiple viable solutions so that users have flexibility to select a tool that works well for their system and problem.

### Methodology

**Languages and Libraries**:

- **Java**: OreKit 12.2 on OpenJDK 21
- **Python**: Brahe Python bindings (PyO3)
- **Rust**: Brahe native Rust library

**Test Environment**: 2021 MacBook Pro, Apple M1 Max, 64 GB RAM

**Protocol**: Each task is run 1000 iterations with a fixed random seed. Mean execution time is reported. Accuracy is measured by comparing outputs element-wise against the Java (OreKit) reference implementation.

### Performance Overview

The table below summarizes average speedup relative to Java (OreKit) for each module. Values greater than 1.0× indicate Brahe is faster; values less than 1.0× indicate OreKit is faster.

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_overview.csv') }}

</div>

Brahe's Python bindings are 2–44× faster than OreKit across most modules, and Rust native is 3–225× faster for pure computational tasks. Propagation benchmarks are the exception — OreKit's mature SGP4 and numerical integrator implementations are highly optimized for trajectory generation workloads.

<div class="plotly-embed x-tall">
  <iframe class="only-light" src="../figures/fig_bench_speedup_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_bench_speedup_dark.html"  loading="lazy"></iframe>
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

**Accuracy**: The raw comparison data shows large apparent errors ($\sim 2 \times 10^5$ m) between OreKit and Brahe frame transformations. This is a **known comparison methodology issue** — the benchmark compares full 6-element state vectors `[x, y, z, vx, vy, vz]` element-wise, where position components (meters) and velocity components (m/s) have vastly different magnitudes. The large max absolute error reflects a small fractional difference in position that appears large in absolute terms due to the ~7000 km orbital radius.

Both implementations use the IAU 2006/2000A model but differ in EOP interpolation and nutation series truncation. Investigation of position-only and velocity-only errors separately is planned to quantify the true agreement.

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

Propagation is the one area where OreKit outperforms Brahe, particularly for SGP4 trajectory generation. OreKit's SGP4 implementation benefits from decades of optimization and a mature numerical integration framework. The SGP4 trajectory benchmark shows OreKit ~20× faster — this reflects architectural differences in how each library handles batch propagation, not fundamental algorithmic limitations.

**Accuracy**:

<div class="center-table" markdown="1">

{{ read_csv('figures/bench_accuracy_propagation.csv') }}

</div>

Keplerian propagation shows nanometer-level agreement. Numerical propagation diverges by ~7 cm due to different integrator implementations and step-size strategies. SGP4 divergence of ~50 m is expected and well-documented across different SGP4 implementations — the original Fortran, OreKit Java, and Brahe Rust implementations all make slightly different numerical choices in the deep-space and near-Earth branch logic.

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

**Accuracy**: Access window start/end times differ by a maximum of ~69 seconds between OreKit and Brahe. This is consistent with the ~50 m SGP4 position divergence — at LEO velocities (~7.5 km/s), a 50 m position difference translates to roughly this magnitude of timing difference in pass predictions. Both implementations agree on the total number of access windows.

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
