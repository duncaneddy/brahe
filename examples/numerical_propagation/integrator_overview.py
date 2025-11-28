# /// script
# dependencies = ["brahe"]
# ///
"""
Complete overview of NumericalPropagationConfig showing all configuration fields.
This example demonstrates every configurable option for integrator configuration.
"""

import brahe as bh

# Create a fully-configured integrator configuration
config = bh.NumericalPropagationConfig(
    # Integration method: Dormand-Prince 5(4)
    bh.IntegrationMethod.DP54,
    # Integrator settings: tolerances and step control
    bh.IntegratorConfig(
        abs_tol=1e-9,
        rel_tol=1e-6,
        initial_step=60.0,  # 60 second initial step
        min_step=1e-6,  # Minimum step size
        max_step=300.0,  # Maximum step size (5 minutes)
        step_safety_factor=0.9,  # Safety margin for step control
        min_step_scale_factor=0.2,  # Minimum step reduction
        max_step_scale_factor=10.0,  # Maximum step growth
        max_step_attempts=10,  # Max attempts per step
    ),
    # Variational configuration: STM and sensitivity settings
    bh.VariationalConfig(
        enable_stm=True,
        enable_sensitivity=False,
        store_stm_history=True,
        store_sensitivity_history=False,
    ),
)

print(f"Method: {config.method}")
print(f"abs_tol: {config.abs_tol}")
print(f"rel_tol: {config.rel_tol}")
print(f"Variational: {config.variational}")
# Method: IntegrationMethod.DP54
# abs_tol: 1e-09
# rel_tol: 1e-06
# Variational: VariationalConfig(enable_stm=true, ...)
