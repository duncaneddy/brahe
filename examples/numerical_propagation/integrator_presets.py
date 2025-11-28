# /// script
# dependencies = ["brahe"]
# ///
"""
Overview of integrator configuration presets.
"""

import brahe as bh

# Preset configurations for common use cases
default = bh.NumericalPropagationConfig.default()
high_precision = bh.NumericalPropagationConfig.high_precision()
rkf45 = bh.NumericalPropagationConfig.with_method(bh.IntegrationMethod.RKF45)
rk4 = bh.NumericalPropagationConfig.with_method(bh.IntegrationMethod.RK4)

print(f"default():        {default.method}")
print(f"high_precision(): {high_precision.method}")
print(f"with_method(RKF45): {rkf45.method}")
print(f"with_method(RK4):   {rk4.method}")
# default():        DP54
# high_precision(): RKN1210
# with_method(RKF45): RKF45
# with_method(RK4):   RK4
