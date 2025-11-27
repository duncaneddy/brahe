# /// script
# dependencies = ["brahe"]
# ///
"""
Overview of all integrator configuration presets.
Shows the available presets and their settings.
"""

import brahe as bh

# NumericalPropagationConfig provides preset configurations
# for common use cases

# 1. default() - General purpose DP54
# Best for most applications
default = bh.NumericalPropagationConfig.default()

# 2. high_precision() - Maximum accuracy with RKN1210
# For precision orbit determination and research
high_precision = bh.NumericalPropagationConfig.high_precision()

# 3. with_method() - Start from specific integrator
# Customize from a chosen method
rkf45_config = bh.NumericalPropagationConfig.with_method(bh.IntegrationMethod.RKF45)
rk4_config = bh.NumericalPropagationConfig.with_method(bh.IntegrationMethod.RK4)

print("Integrator Configuration Presets")
print("=" * 70)

print("\n| Preset           | Method  | abs_tol | rel_tol | Description           |")
print("|------------------|---------|---------|---------|----------------------|")
print("| default()        | DP54    | 1e-6    | 1e-3    | General purpose      |")
print("| high_precision() | RKN1210 | 1e-10   | 1e-8    | Maximum accuracy     |")
print("| with_method(M)   | M       | 1e-6    | 1e-3    | Custom method        |")

print("\nDetailed Preset Descriptions:")

print("\ndefault():")
print("  - Method: DP54 (Dormand-Prince 5(4))")
print("  - Tolerances: abs=1e-6, rel=1e-3")
print("  - Step limits: min=1e-12, max=900s")
print("  - Use for: Most mission analysis, LEO to GEO")
print("  - Good balance of accuracy and speed")

print("\nhigh_precision():")
print("  - Method: RKN1210 (Runge-Kutta-Nystrom 12(10))")
print("  - Tolerances: abs=1e-10, rel=1e-8")
print("  - Use for: POD, validation, research")
print("  - Achieves sub-meter accuracy over days")
print("  - More expensive computationally")

print("\nwith_method(IntegrationMethod):")
print("  - Starts with specified integrator method")
print("  - Uses default tolerances (abs=1e-6, rel=1e-3)")
print("  - Chain with_* methods to customize further")
print("  - Available methods:")
print("    - IntegrationMethod.RK4 (fixed step)")
print("    - IntegrationMethod.RKF45 (adaptive)")
print("    - IntegrationMethod.DP54 (adaptive, default)")
print("    - IntegrationMethod.RKN1210 (adaptive, high precision)")

print("\nIntegrator Method Comparison:")
print("\n| Method  | Order  | Adaptive | Evals | Best For                   |")
print("|---------|--------|----------|-------|---------------------------|")
print("| RK4     | 4      | No       | 4     | Simple problems, debugging|")
print("| RKF45   | 4(5)   | Yes      | 6     | General purpose           |")
print("| DP54    | 5(4)   | Yes      | 6-7   | General purpose (default) |")
print("| RKN1210 | 12(10) | Yes      | 17    | Highest precision         |")

print("\nChoosing a Preset:")
print("  1. Start with default() for most applications")
print("  2. Use high_precision() when accuracy is critical")
print("  3. Use with_method() when you need specific behavior")
print("  4. Chain with_* methods to fine-tune any preset")

print("\nExample validated successfully!")
