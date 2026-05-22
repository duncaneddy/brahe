# frames.state_eci_to_ecef

_Last updated from accuracy run: 2026-05-22_

## Observed residual

- p99 max-abs error: 1.412e+03 (vs basilisk)
- Worst-sample max-abs error: 1.557e+03
- Samples in sweep: 200
- Module threshold (p99): 1.000e+00

## Cause

Earth orientation parameters: brahe and Orekit use the same IERS 2010 conventions but may load slightly different EOP files. Sub-meter residuals are expected; >10 m residuals would warrant investigation.
