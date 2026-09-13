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

**Linear Algebra:**
- Skew-symmetric cross-product matrices
- Block-diagonal matrix construction
- Matrix symmetrization and symmetry checks
"""

from brahe._brahe import (
    AnalyticJacobian,
    AnalyticSensitivity,
    # Jacobian enums
    DifferenceMethod,
    # Jacobian providers
    NumericalJacobian,
    # Sensitivity providers
    NumericalSensitivity,
    PerturbationStrategy,
    # Linear algebra
    block_diagonal,
    is_symmetric,
    skew_symmetric,
    symmetrize,
)

__all__ = [
    "AnalyticJacobian",
    "AnalyticSensitivity",
    # Jacobian enums
    "DifferenceMethod",
    # Jacobian providers
    "NumericalJacobian",
    # Sensitivity providers
    "NumericalSensitivity",
    "PerturbationStrategy",
    # Linear algebra
    "block_diagonal",
    "is_symmetric",
    "skew_symmetric",
    "symmetrize",
]
