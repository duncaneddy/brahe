## Benchmarks

While Brahe prioritizes usability and flexibility, it also aims to provide competitive performance for orbit propagation tasks and analysis.

### Access Computation Benchmark

We benchmarked Brahe's orbit propagation and access computation performance against Skyfield, a popular Python library for satellite tracking and orbit propagation and access computation. The benchmark involved randomly sampling 100 different locations and computing all accesses between a satellite in low Earth orbit and ground stations over a 48-hour period.

The access start and end times all agreed to within one second between Brahe and Skyfield, indicating consistent results to within the level of error of differing Earth orientation models used by each library.

We tested five implementations:

1. **Skyfield** - Popular Python astronomy library (baseline)
2. **Brahe-Python (serial)** - Python bindings, one location per call
3. **Brahe-Python (parallel)** - Python bindings, all locations in single call to leverage internal parallelization
4. **Brahe-Rust (serial)** - Native Rust, one location per call
5. **Brahe-Rust (parallel)** - Native Rust, all locations in single call to leverage internal parallelization

<div class="center-table" markdown="1">
| Implementation         | Avg Time  | vs Skyfield    | vs Brahe-Py-Serial |
|------------------------|-----------|----------------|---------------------|
| Brahe-Rust (parallel)  |   1.37ms  | 3.2x faster    | 23.0x faster        |
| Brahe-Python (parallel)|   2.40ms  | 1.8x faster    | 13.1x faster        |
| Brahe-Rust (serial)    |   2.79ms  | 1.6x faster    | 11.2x faster        |
| Skyfield               |   4.44ms  | baseline       | 7.1x faster         |
| Brahe-Python (serial)  |  31.41ms  | 7.1x slower    | baseline            |
</div>

Unsurprisingly, the parallel implementations significantly outperform their serial counterparts by leveraging multiple CPU cores to handle multiple ground station locations simultaneously. The Rust serial implementation still outperforms both the Skyfield and Brahe-Python serial implementations. Skyfield's performance is still impressive being only marginally slower than Brahe's serial Rust implementation despite being written in pure Python. These tests were run on a 2021 MacBook Pro with an Apple M1 Max chip and 64GB of RAM.

<div class="plotly-embed">
  <iframe class="only-light" src="../figures/fig_access_benchmark_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../figures/fig_access_benchmark_dark.html"  loading="lazy"></iframe>
</div>

You can reproduce these tests by running:

```bash
uv run scripts/benchmark_access_three_way.py --n-locations 100 --seed 42 --output chart.html --plot-style scatter --csv accesses.csv
```
