# Profiles

Performance and allocation profiling harness for brahe.

## Quick Start

```bash
# One-time setup (installs samply + py-spy if missing)
just dev-setup

# CPU profile (Rust, opens Firefox Profiler when done)
just profile-rust rk4_pointmass

# Heap profile (Rust, opens dh_view; drag the .dhat.json file into it)
just profile-rust rk4_80x80_full --heap

# CPU profile (Python, opens SVG flamegraph)
just profile-python sgp4_access

# Run both Rust and Python flavors of the same task
just profile-compare sgp4_trajectory

# Discover available tasks
just profile-list
```

Flags accepted by `profile-rust` and `profile-python`:
- `--duration N` — sampling window in seconds (default `10`).
- `--no-open` — write the artifact but don't open the browser.

Additionally, `profile-rust` accepts:
- `--heap` — use `dhat-heap` allocation profiling instead of CPU sampling.

## Default Workloads

| Name | What it measures |
|---|---|
| `rk4_pointmass` | RK4 numerical propagation, point-mass gravity only. |
| `rk4_5x5_gravity` | RK4 numerical propagation, 5×5 spherical-harmonic gravity. |
| `rk4_20x20_thirdbody` | RK4 numerical propagation, 20×20 gravity + sun/moon third-body. |
| `rk4_80x80_full` | RK4 numerical propagation, 80×80 gravity + sun/moon + NRLMSISE-00 drag + SRP. |
| `sgp4_trajectory` | SGP4 propagation of an ISS-like TLE, 24h horizon, 60s steps. |
| `sgp4_access` | SGP4 + `location_accesses` over a 7-day window, SF ground station, 10° elevation. |

All workloads share a fixed initial state (500 km sun-sync LEO at 2024-01-01 UTC for numerical propagation; a pinned ISS TLE for SGP4) so Rust ↔ Python flamegraphs of the same task are directly comparable.

## Adding a New Task

### Rust

1. Copy `profiles/rust/src/bin/_template.rs.txt` to `profiles/rust/src/bin/<your_task>.rs`.
2. Replace `<TASK_NAME>` in the final `eprintln!` with your task name.
3. Customise the `force` config (or replace the entire workload).
4. Run `just profile-rust <your_task>`.

`src/bin/*.rs` is a Cargo convention — every `.rs` file there becomes a bin target automatically. No manifest edits needed. The template includes the `#[cfg(feature = "dhat-heap")]` blocks so `--heap` works for new tasks out of the box.

### Python

1. Copy `profiles/python/_template.py.txt` to `profiles/python/<your_task>.py`.
2. Replace `<TASK_NAME>` in the final `print()` with your task name.
3. Customise the `force` config and `workload` closure.
4. Run `just profile-python <your_task>`.

## Reading a CPU Profile (Rust, samply)

samply opens the Firefox Profiler with your `.json.gz` loaded:
- **Flame graph (default view):** bar width = self time + descendants. Look for unexpectedly wide leaves.
- **Call tree:** drill down hierarchically; filter by `brahe::` to focus on library frames.
- **Time range selection:** drag in the timeline to isolate a portion of the run.

The artifact in `profiles/results/<timestamp>_<task>.json.gz` can be re-opened anytime with `samply load <file>`.

## Reading a Heap Profile (Rust, dhat)

The recipe opens <https://nnethercote.github.io/dh_view/dh_view.html>. Drag the `.dhat.json` file into the page to load.
- **Block-graph view:** allocation sites by total bytes / count / max-live.
- **Filter by lifetime:** distinguish short-lived (per-step) from long-lived (per-propagator) allocations.

## Reading a CPU Profile (Python, py-spy)

py-spy emits an SVG flamegraph that opens directly in any browser.
- `--native` is on by default in our recipe, so `_brahe` extension frames expand into real Rust call stacks.
- If you see lots of time in `_brahe::*` but the matching Rust task's flamegraph shows the same time in pure-Rust functions, the work is real numerics, not binding overhead. Conversely, if Python time concentrates in `_brahe::*` frames that the Rust task does not exhibit, you're looking at PyO3 marshaling overhead.

### macOS Limitations

py-spy `--native` is **not supported on arm64 macOS** as of py-spy 0.4.x — `just profile-python` will fail with "Collecting stack traces from native extensions is not supported on your platform." Workarounds:
- Run the Python profile in a Linux VM, Docker container, or on Linux CI.
- Drop `--native` and profile the Python frames only (requires editing the just recipe; not the default).
- py-spy on macOS also generally requires `sudo`; consider that even on x86_64 macOS hosts where `--native` may work.

## Why the Outputs Are Gitignored

Profiles are machine-specific (CPU, frequency scaling, kernel scheduler, page cache state all affect timing). Committing them would invite false "this regressed!" interpretations from environment drift. Keep them local; share artifacts via direct file transfer or by uploading to the Firefox Profiler hosted service when you want a permanent link.

## Why CPU and Heap Are Separate Modes

`dhat` instruments every allocation through a wrapped global allocator. CPU samples taken during a `dhat` run measure dhat-overhead-plus-workload, not the workload — so the modes share task definitions but execute in different builds (with vs. without `--features dhat-heap`).

## Reference

The full design rationale lives in `docs/superpowers/specs/2026-05-21-flamegraph-profiling-design.md` (local-only; gitignored per project convention).
