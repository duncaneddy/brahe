# SGP Propagator

The SGP4/SDP4 propagator for satellite orbit propagation using Two-Line Element (TLE) data. The SGP (Simplified General Perturbations) propagator implements the SGP4/SDP4 models for propagating satellites using TLE orbital data. This is the standard model used for tracking objects in Earth orbit and is maintained by NORAD/Space Force.

**Module**: `brahe.orbits`

::: brahe.SGPPropagator
    options:
      show_root_heading: true
      show_root_full_path: false

## See Also

- [SGPPropagatorBuilder](sgp_propagator_builder.md) - Builder for constructing this propagator from OMM elements
- [KeplerianPropagator](keplerian_propagator.md) - Analytical two-body propagator
- [TLE](../orbits/tle.md) - Two-Line Element format details
- [Keplerian Elements](../orbits/keplerian.md) - Orbital element functions
