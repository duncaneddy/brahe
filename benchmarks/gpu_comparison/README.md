# Brahe vs. Astrojax GPU Benchmark Suite

Throughput-vs-batch-size benchmarks comparing brahe (Rust + rayon, f64)
against astrojax (JAX on CPU f64, single GPU f32, multi-GPU f32) across
time, coordinate, frame, SGP4, numerical, and full-force-model tasks.

See `docs/superpowers/specs/2026-05-22-brahe-vs-astrojax-gpu-benchmarks-design.md`
for the design.

## Setup

```bash
just bench-gpu-install   # installs brahe[gpu-comparison] + astrojax + CUDA-matched jaxlib
                         # (NO_LOCAL=1 forces astrojax from PyPI)
just bench-gpu-build     # compiles the bench_gpu_rust subprocess
```

## Running

```bash
# Full suite (default flags)
uv run python -m benchmarks.gpu_comparison run

# A single task family
uv run python -m benchmarks.gpu_comparison run --module coordinates

# A single (task, config, batch) cell — useful for triage
uv run python -m benchmarks.gpu_comparison run-cell \
    coordinates.geodetic_to_ecef brahe-rust-rayon 100000
```

## Inspecting results

```bash
uv run python -m benchmarks.gpu_comparison inspect \
    benchmarks/gpu_comparison/results/run_<timestamp>.json
```

## Hardware reporting

Every run records CPU model + cores, RAM, GPU model + driver + CUDA version,
and the git SHAs / versions of brahe and astrojax. See the `system` field
of any results JSON.

## Adding a task

1. Add a subclass of `BatchTask` to the appropriate file under `tasks/`.
2. Implement the Rust side in `implementations/rust/src/main.rs` and wire
   it into the dispatch `match`.
3. Implement the astrojax side as a builder registered in
   `implementations/astrojax_kernels.py`.
4. Call `register(YourTask)` from `tasks/__init__.py`.
