"""
Math Module

Mathematical utilities and numerical methods.

This module provides numerical methods for mathematical computations used in
orbital mechanics and satellite dynamics:

**Jacobian Computation:**
- Numerical Jacobian providers using finite differences
- Analytical Jacobian providers for user-supplied functions
- Multiple finite difference methods (forward, central, backward)
- Adaptive perturbation strategies for accuracy

Jacobian computation is essential for:
- Numerical integration with variational equations
- State transition matrix propagation
- Sensitivity analysis and uncertainty quantification
"""

from brahe._brahe import (
    # Jacobian enums
    DifferenceMethod,
    PerturbationStrategy,
    # Jacobian providers
    NumericalJacobian,
    AnalyticJacobian,
    # Sensitivity providers
    NumericalSensitivity,
    AnalyticSensitivity,
)

__all__ = [
    # Jacobian enums
    "DifferenceMethod",
    "PerturbationStrategy",
    # Jacobian providers
    "NumericalJacobian",
    "AnalyticJacobian",
    # Sensitivity providers
    "NumericalSensitivity",
    "AnalyticSensitivity",
]
