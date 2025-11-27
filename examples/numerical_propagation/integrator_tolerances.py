# /// script
# dependencies = ["brahe"]
# ///
"""
Setting custom error tolerances for adaptive integrators.
Shows how tolerances affect accuracy and performance.
"""

import brahe as bh

# Error tolerances control the accuracy of adaptive integrators
# The integrator adjusts step size to keep error within:
#   error < abs_tol + rel_tol * |state|

# Default configuration (general purpose)
config_default = bh.NumericalPropagationConfig.default()

# Quick analysis (looser tolerances, faster)
config_quick = (
    bh.NumericalPropagationConfig.default().with_abs_tol(1e-3).with_rel_tol(1e-1)
)

# Standard precision
config_standard = (
    bh.NumericalPropagationConfig.default().with_abs_tol(1e-6).with_rel_tol(1e-3)
)

# High precision
config_precision = (
    bh.NumericalPropagationConfig.default().with_abs_tol(1e-9).with_rel_tol(1e-6)
)

# Maximum precision (use with RKN1210)
config_maximum = (
    bh.NumericalPropagationConfig.with_method(bh.IntegrationMethod.RKN1210)
    .with_abs_tol(1e-12)
    .with_rel_tol(1e-10)
)

print("Integrator Error Tolerances")
print("=" * 60)

print("\nError Control Formula:")
print("  error < abs_tol + rel_tol * |state|")
print("\n  - abs_tol: Bounds error when state is small")
print("  - rel_tol: Bounds error proportional to state magnitude")

print("\nTolerance Presets:")
print("\n| Level          | abs_tol | rel_tol | Use Case                   |")
print("|----------------|---------|---------|----------------------------|")
print("| Quick analysis | 1e-3    | 1e-1    | Rough estimates            |")
print("| Standard       | 1e-6    | 1e-3    | General mission analysis   |")
print("| High precision | 1e-9    | 1e-6    | Precision applications     |")
print("| Maximum        | 1e-12   | 1e-10   | POD, research (use RKN1210)|")

print("\nGuidelines for Choosing Tolerances:")

print("\nAbsolute Tolerance (abs_tol):")
print("  - Controls error when state components are small")
print("  - For position: set to desired position accuracy (meters)")
print("  - For velocity: set to desired velocity accuracy (m/s)")
print("  - Default: 1e-6 (~1 micrometer position)")

print("\nRelative Tolerance (rel_tol):")
print("  - Controls error as fraction of state magnitude")
print("  - 1e-3 = 0.1% relative error")
print("  - 1e-6 = 0.0001% relative error")
print("  - Default: 1e-3 (0.1% relative accuracy)")

print("\nTrade-offs:")
print("  Tighter tolerances -> Smaller steps -> More computation")
print("  Looser tolerances -> Larger steps -> Less computation")
print("  Find the minimum accuracy that meets your requirements")

print("\nExample validated successfully!")
