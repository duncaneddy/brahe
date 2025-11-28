# /// script
# dependencies = ["brahe"]
# ///
"""
Configuring the RK4 fixed-step integrator method.
"""

import brahe as bh

# RK4: Fixed-step 4th-order Runge-Kutta
config = bh.NumericalPropagationConfig(
    bh.IntegrationMethod.RK4,
    bh.IntegratorConfig.fixed_step(60.0),  # 60 second fixed steps
    bh.VariationalConfig(),
)

print(f"Method: {config.method}")
# Method: IntegrationMethod.RK4
