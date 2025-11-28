# /// script
# dependencies = ["brahe"]
# ///
"""
Configuring the DP54 adaptive integrator method (default).
"""

import brahe as bh

# DP54: Dormand-Prince 5(4) - the default integrator
config = bh.NumericalPropagationConfig.default()

# Customize tolerances using builder pattern
config_tight = (
    bh.NumericalPropagationConfig.default().with_abs_tol(1e-9).with_rel_tol(1e-6)
)

print(f"Method: {config.method}")
print(f"abs_tol: {config.abs_tol}")
print(f"rel_tol: {config.rel_tol}")
# Method: IntegrationMethod.DP54
# abs_tol: 1e-06
# rel_tol: 0.001
