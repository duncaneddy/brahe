"""
Propagators Module

Orbit propagators for predicting satellite positions over time.

This module provides:

**Propagators:**
- SGPPropagator: SGP4/SDP4 propagator for TLE-based orbit prediction
- KeplerianPropagator: Analytical two-body orbit propagator

The propagators implement both the StateProvider trait (for direct state computation
at any epoch) and the OrbitPropagator trait (for stepped propagation with trajectory
accumulation).
"""

from brahe._brahe import (
    # Propagators
    SGPPropagator,
    KeplerianPropagator,
)

__all__ = [
    # Propagators
    "SGPPropagator",
    "KeplerianPropagator",
]
