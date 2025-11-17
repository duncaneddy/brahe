# Integrators Module

**Module**: `brahe.integrators`

The integrators module provides numerical integration methods for solving ordinary differential equations (ODEs), with specialized support for orbital mechanics and variational equation propagation.

## Module Contents

- [Configuration](config.md) - Integrator configuration and adaptive step control
- [RK4 Integrator](rk4.md) - Classical 4th-order Runge-Kutta method
- [RKF45 Integrator](rkf45.md) - Runge-Kutta-Fehlberg 4(5) adaptive method
- [DP54 Integrator](dp54.md) - Dormand-Prince 5(4) adaptive method
- [RKN1210 Integrator](rkn1210.md) - Runge-Kutta-Nyström 12(10) high-precision method

## Common Interface

All integrators implement one of these trait interfaces:

- `FixedStepIntegrator`: Fixed time-step integration
- `AdaptiveStepIntegrator`: Adaptive time-step integration with error control

Borth of which provide common methods:

- `step()`: Advance the state by one time step
- `step_with_varmat()`: Advance state and state transition matrix together

## See Also

- [Numerical Integration User Guide](../../learn/integrators/index.md) - Conceptual introduction and examples
- [Choosing Integrators](../../learn/integrators/fixed_step.md) - Guide to selecting the right integrator
- [Configuration Guide](../../learn/integrators/configuration.md) - Tuning integrator parameters
