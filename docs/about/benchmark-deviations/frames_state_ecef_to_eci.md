# frames.state_ecef_to_eci

_Last updated from accuracy run: 2026-05-21_

## Observed residual

- p99 max-abs error: 9.566e+02 (vs basilisk)
- Worst-sample max-abs error: 1.204e+03
- Samples in sweep: 100
- Module threshold (p99): 1.000e+00

## Cause

See state_eci_to_ecef — same EOP-source explanation. The inverse transformation reuses the same rotation matrix.
