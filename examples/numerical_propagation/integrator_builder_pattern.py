# /// script
# dependencies = ["brahe"]
# ///
"""
Using the builder pattern to customize integrator configuration.
"""

import brahe as bh

# Chain with_* methods to customize from a preset
config = (
    bh.NumericalPropagationConfig.default()
    .with_abs_tol(1e-9)
    .with_rel_tol(1e-6)
    .with_max_step(300.0)
    .with_initial_step(60.0)
)

print(f"Method: {config.method}")
print(f"abs_tol: {config.abs_tol}")
print(f"rel_tol: {config.rel_tol}")
# Method: IntegrationMethod.DP54
# abs_tol: 1e-09
# rel_tol: 1e-06
