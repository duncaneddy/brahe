"""
Propagators Module

Orbit propagators for predicting satellite positions over time.

This module provides:

**Propagators:**
- SGPPropagator: SGP4/SDP4 propagator for TLE-based orbit prediction
- KeplerianPropagator: Analytical two-body orbit propagator
- NumericalOrbitPropagator: High-fidelity numerical orbit propagator with force models
- NumericalOrbitPropagatorBuilder: Builder for NumericalOrbitPropagator
- NumericalPropagator: Generic numerical propagator for arbitrary dynamics

**Configuration Classes:**
- IntegrationMethod: Integration method selection (RK4, RKF45, RKF78, DP54, RKN1210)
- AtmosphericModel: Atmospheric density model selection
- EclipseModel: Eclipse model for SRP calculations
- ZonalHarmonicsDegree: Maximum zonal harmonic degree (J2..=J6)
- ParallelMode: Parallelization mode for spherical harmonic gravity (Auto, Always, Never)
- FrameTransformationModel: ECI-to-body-fixed rotation precision selector
- NumericalPropagationConfig: Configuration for numerical integration
- VariationalConfig: STM/sensitivity configuration
- CentralBody: Central body an orbit is propagated relative to (Earth, Moon, Mars, EMB, SSB, Custom)
- ForceModelConfig: Force model configuration for orbit propagation
- ParameterSource: Source for parameter values (fixed or from vector)
- GravityConfiguration: Gravity model configuration
- DragConfiguration: Atmospheric drag configuration
- SolarRadiationPressureConfiguration: SRP configuration
- OccultingBody: Occulting body for eclipse/shadow modeling in SRP calculations
- ThirdBody: Third-body perturber enum
- ThirdBodyConfiguration: Third-body perturbation configuration
- SolidTideConfig: Solid Earth tide configuration
- OceanTideConfig: FES2004 ocean tide configuration
- PermanentTideConfig: Permanent (zero-frequency) tide handling
- TidesConfiguration: Tidal correction configuration

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
    NumericalOrbitPropagatorBuilder,
    NumericalPropagator,
    # Configuration Classes
    IntegrationMethod,
    AtmosphericModel,
    EclipseModel,
    ZonalHarmonicsDegree,
    ParallelMode,
    FrameTransformationModel,
    NumericalPropagationConfig,
    VariationalConfig,
    CentralBody,
    ForceModelConfig,
    ParameterSource,
    GravityConfiguration,
    DragConfiguration,
    SolarRadiationPressureConfiguration,
    OccultingBody,
    ThirdBody,
    ThirdBodyConfiguration,
    SolidTideConfig,
    OceanTideConfig,
    PermanentTideConfig,
    TidesConfiguration,
    TrajectoryMode,
    # Functions
    par_propagate_to,
)

__all__ = [
    # Propagators
    "SGPPropagator",
    "KeplerianPropagator",
    "NumericalOrbitPropagator",
    "NumericalOrbitPropagatorBuilder",
    "NumericalPropagator",
    # Configuration Classes
    "IntegrationMethod",
    "AtmosphericModel",
    "EclipseModel",
    "ZonalHarmonicsDegree",
    "ParallelMode",
    "FrameTransformationModel",
    "NumericalPropagationConfig",
    "VariationalConfig",
    "CentralBody",
    "ForceModelConfig",
    "ParameterSource",
    "GravityConfiguration",
    "DragConfiguration",
    "SolarRadiationPressureConfiguration",
    "OccultingBody",
    "ThirdBody",
    "ThirdBodyConfiguration",
    "SolidTideConfig",
    "OceanTideConfig",
    "PermanentTideConfig",
    "TidesConfiguration",
    "TrajectoryMode",
    # Functions
    "par_propagate_to",
]
