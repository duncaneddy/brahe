## Benchmarks

While Brahe prioritizes usability and flexibility, it also aims to provide competitive performance for orbit propagation tasks and analysis. 

### Access Computation Benchmark

We benchmarked Brahe's orbit propagation and access computation performance against Skyfield, a popular Python library for satellite tracking and orbit propagation and access computation. The benchmark involved randomly sampling different locations and computing all accesses between a satellite in low Earth orbit and ground stations over a 48-hour period.

The access start and end times all agreed to within one second between Brahe and Skyfield, indicating consistent results to within the level of error of differing Earth orientation models used by each library.

Generally, we found that Brahe's Rust implementation provided the best performance, followed by Skyfield, and then Brahe's Python implementation. The results are summarized in the table below:

<div class="center-table" markdown="1">
| Implementation | Avg Time  | vs Skyfield    | vs Brahe-Python |
|----------------|-----------|----------------|-----------------|
| Brahe-Rust     |   2.78ms  | 1.4x faster    | 10.8x faster    |
| Skyfield       |   4.01ms  | baseline       | 7.5x faster     |
| Brahe-Python   |  29.93ms  | 7.5x slower    | baseline        |
</div>

You can reproduce these tests by running:

```bash
uv run scripts/benchmark_access_three_way.py --n-locations 1000 --seed 42 --output chart.html --plot-style scatter --csv accesses.csv
```
