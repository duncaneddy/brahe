# SGP Propagator Builder

Builder for `SGPPropagator` from CCSDS OMM (Orbit Mean-elements Message) elements. `builder()` takes the eight required OMM inputs (`epoch`, `mean_motion`, `eccentricity`, `inclination`, `raan`, `arg_of_pericenter`, `mean_anomaly`, `norad_id`) directly as arguments. Optional inputs are set through chained setters and default to `None`, except `step_size` which defaults to 60 seconds.

**Module**: `brahe.orbits`

::: brahe.SGPPropagatorBuilder
    options:
      show_root_heading: true
      show_root_full_path: false

---

## See Also

- [SGPPropagator](sgp_propagator.md) - SGP4/SDP4 propagator for TLE-based satellite orbit propagation
- [SGP Propagation Guide](../../learn/orbit_propagation/sgp_propagation.md) - User guide documentation
