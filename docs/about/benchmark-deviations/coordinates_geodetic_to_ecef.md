# coordinates.geodetic_to_ecef

_Last updated from accuracy run: 2026-05-22_

## Observed residual

- p99 max-abs error: 7.181e-01 (vs gmat)
- Worst-sample max-abs error: 7.186e-01
- Samples in sweep: 200
- Module threshold (p99): 1.000e-03

## Cause

GMAT uses an Earth equatorial radius of 6378.1363 km versus WGS84's 6378.137 km, which accounts for ~0.7 m position error in geodetic ↔ ECEF conversions. The non-GMAT residuals are sub-nanometer (WGS84 vs WGS84) and not flagged.
