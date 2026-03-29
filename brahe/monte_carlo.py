"""
Monte Carlo Simulation Module

Monte Carlo simulation framework for astrodynamics analysis.

This module provides:

**Distribution Types:**
- Gaussian: Univariate normal distribution
- UniformDist: Continuous uniform distribution
- TruncatedGaussian: Truncated Gaussian distribution
- MultivariateNormal: Multivariate normal distribution

**Variable Identification:**
- MonteCarloVariableId: Typed identifier for simulation variables

**Configuration:**
- MonteCarloStoppingCondition: Stopping condition (fixed runs or convergence)

**Simulation:**
- MonteCarloSimulation: Monte Carlo simulation engine

**Results:**
- MonteCarloResults: Results with statistical accessors
- MonteCarloRun: Data from a single simulation run

**Thread-Local Providers (for Monte Carlo orbit propagation):**
- TableEOPProvider: Table-based EOP provider
- TableSpaceWeatherProvider: Table-based space weather provider
- set_thread_local_eop_provider: Set thread-local EOP override
- clear_thread_local_eop_provider: Clear thread-local EOP override
- set_thread_local_space_weather_provider: Set thread-local SW override
- clear_thread_local_space_weather_provider: Clear thread-local SW override
"""

from brahe._brahe import (
    # Distribution types
    Gaussian,
    UniformDist,
    TruncatedGaussian,
    MultivariateNormal,
    # Variable identification
    MonteCarloVariableId,
    # Configuration
    MonteCarloStoppingCondition,
    # Simulation engine
    MonteCarloSimulation,
    # Results
    MonteCarloResults,
    MonteCarloRun,
    # Thread-local providers
    TableEOPProvider,
    TableSpaceWeatherProvider,
    set_thread_local_eop_provider,
    clear_thread_local_eop_provider,
    set_thread_local_space_weather_provider,
    clear_thread_local_space_weather_provider,
)

__all__ = [
    # Distribution types
    "Gaussian",
    "UniformDist",
    "TruncatedGaussian",
    "MultivariateNormal",
    # Variable identification
    "MonteCarloVariableId",
    # Configuration
    "MonteCarloStoppingCondition",
    # Simulation engine
    "MonteCarloSimulation",
    # Results
    "MonteCarloResults",
    "MonteCarloRun",
    # Thread-local providers
    "TableEOPProvider",
    "TableSpaceWeatherProvider",
    "set_thread_local_eop_provider",
    "clear_thread_local_eop_provider",
    "set_thread_local_space_weather_provider",
    "clear_thread_local_space_weather_provider",
]
