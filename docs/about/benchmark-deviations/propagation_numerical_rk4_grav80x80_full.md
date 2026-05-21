# propagation.numerical_rk4_grav80x80_full

_Last updated from accuracy run: 2026-05-21_

## Observed residual

- p99 max-abs error: 2.234e+02 (vs basilisk)
- Worst-sample max-abs error: 2.853e+02
- Samples in sweep: 30
- Module threshold (p99): 1.000e+02

## Cause

brahe-vs-Orekit ~55 m max over 100 LEO orbits is consistent with **atmospheric drag space-weather input differences** — EIGEN-6S and EGM2008 truncated to 80x80 agree to sub-metre over one orbit, so gravity is not the dominant term. brahe reads space weather from its bundled NRLMSISE-00 driver configuration; Orekit reads CSSI's `SpaceWeather-All-v1.2.txt`; GMAT uses MSISE-90 with its own SW table; Basilisk uses representative quiet-Sun values (Ap=8, F10.7=110). These SW differences integrate to metre-to-tens-of-metres position errors at LEO over one orbit, with the spread widening at lower altitudes. SRP models are also slightly different across backends but contribute much less than drag at LEO. brahe already exposes ``from_file`` for both gravity (.gfc) and space-weather inputs, so aligning the backends on a single CSSI SW table and a single EGM2008 / EIGEN-6S coefficient file would collapse this residual; the work is the benchmark adapter side, not the brahe library side.
