"""
Propagators Module

Orbit propagators for predicting satellite positions over time.

This module provides:

**Propagators:**
- SGPPropagator: SGP4/SDP4 propagator for TLE-based orbit prediction
- KeplerianPropagator: Analytical two-body orbit propagator
- NumericalOrbitPropagator: High-fidelity numerical orbit propagator with force models
- NumericalPropagator: Generic numerical propagator for arbitrary dynamics

**Configuration Classes:**
- IntegrationMethod: Integration method selection (RK4, RKF45, DP54, RKN1210)
- AtmosphericModel: Atmospheric density model selection
- EclipseModel: Eclipse model for SRP calculations
- NumericalPropagationConfig: Configuration for numerical integration
- VariationalConfig: STM/sensitivity configuration
- ForceModelConfig: Force model configuration for orbit propagation
- ParameterSource: Source for parameter values (fixed or from vector)
- GravityConfiguration: Gravity model configuration
- DragConfiguration: Atmospheric drag configuration
- SolarRadiationPressureConfiguration: SRP configuration
- ThirdBody: Third-body perturber enum
- ThirdBodyConfiguration: Third-body perturbation configuration

**Functions:**
- par_propagate_to: Propagate multiple propagators in parallel to a target epoch

The propagators implement both the StateProvider trait (for direct state computation
at any epoch) and the OrbitPropagator trait (for stepped propagation with trajectory
accumulation).
"""

from brahe._brahe import (
    # Propagators
    SGPPropagator,
    KeplerianPropagator,
    NumericalOrbitPropagator,
    NumericalPropagator,
    # Configuration Classes
    IntegrationMethod,
    AtmosphericModel,
    EclipseModel,
    NumericalPropagationConfig,
    VariationalConfig,
    ForceModelConfig,
    ParameterSource,
    GravityConfiguration,
    DragConfiguration,
    SolarRadiationPressureConfiguration,
    ThirdBody,
    ThirdBodyConfiguration,
    TrajectoryMode,
    # Functions
    par_propagate_to,
)

__all__ = [
    # Propagators
    "SGPPropagator",
    "KeplerianPropagator",
    "NumericalOrbitPropagator",
    "NumericalPropagator",
    # Configuration Classes
    "IntegrationMethod",
    "AtmosphericModel",
    "EclipseModel",
    "NumericalPropagationConfig",
    "VariationalConfig",
    "ForceModelConfig",
    "ParameterSource",
    "GravityConfiguration",
    "DragConfiguration",
    "SolarRadiationPressureConfiguration",
    "ThirdBody",
    "ThirdBodyConfiguration",
    "TrajectoryMode",
    # Functions
    "par_propagate_to",
]
