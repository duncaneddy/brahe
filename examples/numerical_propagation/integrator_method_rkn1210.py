# /// script
# dependencies = ["brahe"]
# ///
"""
Configuring the RKN1210 high-precision integrator method.
"""

import brahe as bh

# RKN1210: High-order adaptive integrator for maximum precision
config = bh.NumericalPropagationConfig.high_precision()

# Or manually configure with custom tolerances
config_custom = (
    bh.NumericalPropagationConfig.with_method(bh.IntegrationMethod.RKN1210)
    .with_abs_tol(1e-12)
    .with_rel_tol(1e-10)
)

print(f"Method: {config.method}")
print(f"abs_tol: {config.abs_tol}")
print(f"rel_tol: {config.rel_tol}")
# Method: IntegrationMethod.RKN1210
# abs_tol: 1e-10
# rel_tol: 1e-08
