# Sensitivity Computation

Sensitivity matrix computation utilities for parameter estimation and consider covariance analysis.

## Overview

Brahe provides both analytical and numerical sensitivity computation through a unified interface. Sensitivity matrices describe how a function's output changes with respect to consider parameters ($\partial f/\partial p$), which is essential for orbit determination with consider parameters and covariance analysis.

---

::: brahe.NumericalSensitivity
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.AnalyticSensitivity
    options:
      show_root_heading: true
      show_root_full_path: false

## See Also

- [Sensitivity Matrix Guide](../../learn/integrators/sensitivity_matrix.md) - Detailed usage examples and theory
- [Jacobian Computation](jacobian.md) - Related Jacobian computation utilities
- [Mathematics Module](index.md) - Mathematics module overview
