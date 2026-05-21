# propagation.numerical_rk4_grav80x80_full

_Last updated from accuracy run: 2026-05-21_

## Observed residual

- p99 max-abs error: 2.234e+02 (vs basilisk)
- Worst-sample max-abs error: 2.853e+02
- Samples in sweep: 30
- Module threshold (p99): 1.000e+02

## Cause

After aligning SW and gravity coefficients between brahe and Orekit, the ~55 m max residual is unchanged — which means neither input is its dominant source. Specifically:

**Space weather** — brahe loads Orekit's CSSI ``SpaceWeather-All-v1.2.txt`` via ``brahe.FileSpaceWeatherProvider.from_file`` (see ``implementations/python/base.py::ensure_sw``). Ap and F10.7 are byte-identical between brahe and Orekit at every tested epoch.

**Gravity coefficients** — Orekit's ``GravityFieldFactory`` is reconfigured at the start of every Java propagation run to read brahe's bundled ``data/gravity_models/EGM2008_360.gfc`` via an explicit ``ICGEMFormatReader``. The benchmark orchestrator sets ``BRAHE_GRAVITY_FILE`` (see ``benchmarks/comparative/config.py``) which the Java adapter picks up; the adapter calls ``clearPotentialCoefficientsReaders()`` first so Orekit's default EIGEN-6S file is no longer in the search chain. Rerunning with vs. without the env var produced numerically identical results — confirming that EGM2008 and EIGEN-6S agree closely enough at degree 80 to be invisible at this fidelity.

**What's left** — the residual reflects implementation-level differences in the dynamics that aren't expressible as shared data: independent NRLMSISE-00 ports (brahe in Rust, Orekit in Java) accumulate slightly different intermediate values at LEO densities; SRP eclipse-model details and the third-body ephemeris source (brahe DE440s vs Orekit bundled JPL DE) contribute smaller amounts. None of these are configuration knobs the benchmark adapter can reach from the outside; closing the gap further would mean changing one library's algorithm to match the other, which is out of scope for a comparison benchmark.
