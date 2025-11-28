# /// script
# dependencies = ["brahe"]
# ///
"""
Setting custom error tolerances for adaptive integrators.
"""

import brahe as bh

# Different tolerance levels for various use cases
config_quick = (
    bh.NumericalPropagationConfig.default().with_abs_tol(1e-3).with_rel_tol(1e-1)
)
config_standard = bh.NumericalPropagationConfig.default()  # abs=1e-6, rel=1e-3
config_precision = (
    bh.NumericalPropagationConfig.default().with_abs_tol(1e-9).with_rel_tol(1e-6)
)
config_maximum = bh.NumericalPropagationConfig.high_precision()  # abs=1e-10, rel=1e-8

print(f"Quick:     abs={config_quick.abs_tol}, rel={config_quick.rel_tol}")
print(f"Standard:  abs={config_standard.abs_tol}, rel={config_standard.rel_tol}")
print(f"Precision: abs={config_precision.abs_tol}, rel={config_precision.rel_tol}")
print(f"Maximum:   abs={config_maximum.abs_tol}, rel={config_maximum.rel_tol}")
# Quick:     abs=0.001, rel=0.1
# Standard:  abs=1e-06, rel=0.001
# Precision: abs=1e-09, rel=1e-06
# Maximum:   abs=1e-10, rel=1e-08
