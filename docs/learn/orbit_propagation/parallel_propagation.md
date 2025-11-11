# Parallel Orbit Propagation

When working with multiple satellites (constellations, Monte Carlo simulations, etc.), propagating each satellite sequentially can be slow. The `par_propagate_to` function enables efficient parallel propagation by utilizing multiple CPU cores. The parallel propagation function uses Rayon's work-stealing thread pool, configured via `brahe.set_num_threads()`.

!!! tip "When to Use Parallel Propagation"

    - Propagating constellations (10s to 1000s of satellites)
    - Running Monte Carlo simulations
    - Batch processing orbital predictions
    - You have multiple CPU cores available

See the [threading documentation](../../learn/threading.md) for more details on configuring threading in Brahe.

## Basic Example

This example creates a constellation of 10 satellites and propagates them 24 hours forward in parallel:

=== "Python"
    ```python
    --8<-- "./examples/orbit_propagation/functions/parallel_propagation.py:12"
    ```

=== "Rust"
    ```rust
    --8<-- "./examples/orbit_propagation/functions/parallel_propagation.rs:8"
    ```

## Mixing Propagator Types

All propagators in the list must be the same type (either all `KeplerianPropagator` or all `SGPPropagator`). Mixing types will raise a `TypeError`:

=== "Python"
    ```python
    import brahe as bh

    # Example propgator intiailization
    kep_prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)
    sgp_prop = bh.SGPPropagator.from_tle(line1, line2, 60.0)

    # This will raise TypeError
    bh.par_propagate_to([kep_prop, sgp_prop], target)
    ```

## Memory Considerations

The parallel function clones each propagator before propagation, then updates the originals with final states. Memory usage scales linearly with the number of propagators.

For very large constellations (1000s of satellites), consider processing in batches and monitoring memory usage to avoid crashes from memory exhaustion.

## Error Handling

If any propagator fails during parallel propagation, the function will panic (Rust) or raise an exception (Python). This can occur with SGP4 propagators when satellites decay below Earth's surface.

## See Also

- [Keplerian Propagator](../../library_api/propagators/keplerian_propagator.md) - Two-body orbital propagation
- [SGP4 Propagator](../../library_api/propagators/sgp_propagator.md) - TLE-based propagation
- [API Reference: par_propagate_to](../../library_api/propagators/functions.md#par_propagate_to)
