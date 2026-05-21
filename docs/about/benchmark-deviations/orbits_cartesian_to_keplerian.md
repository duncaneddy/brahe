# orbits.cartesian_to_keplerian

_Last updated from accuracy run: 2026-05-21_

## Observed residual

- p99 max-abs error: 6.156e-02 (vs basilisk)
- Worst-sample max-abs error: 6.160e-02
- Samples in sweep: 100
- Module threshold (p99): 1.000e-03

## Cause

Keplerian element vector ``[a, e, i, raan, argp, M]`` mixes a length (meters) with five angles (degrees). The current element-wise compare reports the worst-of-six in unit-free meters, which can be dominated by an angle disagreement near 360° wrap or by a small semi-major-axis residual. The future task-specific comparison (analogous to the geodetic surface-arc conversion) should split angle and length components before reporting.
