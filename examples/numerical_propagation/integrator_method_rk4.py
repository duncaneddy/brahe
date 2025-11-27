# /// script
# dependencies = ["brahe"]
# ///
"""
Configuring the RK4 fixed-step integrator method.
Classic 4th-order Runge-Kutta for simple problems.
"""

import brahe as bh

# RK4 (Classical Runge-Kutta 4th order)
# - Fixed step size (no adaptive error control)
# - 4 function evaluations per step
# - Good for simple problems or when you need predictable timing
# - Requires careful step size selection

# Create configuration with RK4 method
config = bh.NumericalPropagationConfig.with_method(bh.IntegrationMethod.RK4)

# For fixed-step integrators, use IntegratorConfig.fixed_step()
# Create full configuration with fixed step
int_config = bh.IntegratorConfig.fixed_step(60.0)  # 60 second steps

# Full configuration using NumericalPropagationConfig.new()
config_with_step = bh.NumericalPropagationConfig.new(
    bh.IntegrationMethod.RK4,
    bh.IntegratorConfig.fixed_step(60.0),
    bh.VariationalConfig(),
)

print("RK4 Fixed-Step Integrator Configuration:")
print("  Method: RK4 (Classical 4th-order Runge-Kutta)")
print("  Adaptive: No (fixed step size)")
print("  Function evaluations per step: 4")
print("  Order: 4")

print("\nCharacteristics:")
print("  - Predictable step timing")
print("  - No error estimation/control")
print("  - Fast per-step computation")
print("  - Requires manual step size tuning")

print("\nWhen to use RK4:")
print("  - Simple dynamical systems")
print("  - When consistent timing matters more than accuracy")
print("  - Initial prototyping and testing")
print("  - Educational/demonstration purposes")

print("\nWhen NOT to use RK4:")
print("  - High-precision requirements")
print("  - Stiff differential equations")
print("  - Long-duration propagations")
print("  - When dynamics vary significantly over orbit")

print("\nStep size guidelines for orbital mechanics:")
print("  - LEO: 30-60 seconds")
print("  - MEO: 60-120 seconds")
print("  - GEO: 120-300 seconds")
print("  - Highly eccentric: Smaller near perigee")

print("\nExample validated successfully!")
