# /// script
# dependencies = ["brahe"]
# ///
"""
Configuring the RKF78 adaptive integrator method.
"""

import brahe as bh

# RKF78: Runge-Kutta-Fehlberg 7(8), useful for tight tolerances
config = (
    bh.NumericalPropagationConfig.with_method(bh.IntegrationMethod.RKF78)
    .with_abs_tol(1e-10)
    .with_rel_tol(1e-8)
)

print(f"Method: {config.method}")
print(f"abs_tol: {config.abs_tol}")
print(f"rel_tol: {config.rel_tol}")
