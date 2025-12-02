# Propagator Integration

SpaceTrack GP data can be converted directly to SGP propagators for orbit propagation. This enables seamless integration between data retrieval and trajectory computation.

## Converting GP to Propagators

Use `gp_as_propagators()` to fetch GP data and convert it to SGP propagators in one step:

=== "Python"

    ``` python
    --8<-- "./examples/spacetrack/gp_to_propagators.py:8"
    ```

## Propagating Multiple Satellites

Query multiple satellites and propagate them together:

=== "Python"

    ``` python
    --8<-- "./examples/spacetrack/multiple_propagators.py:8"
    ```

## Working with Constellations

Query entire constellations by international designator pattern:

=== "Python"

    ``` python
    --8<-- "./examples/spacetrack/constellation_query.py:8"
    ```

## See Also

- [Client Usage](client.md) - Basic client operations
- [Query Filters](queries.md) - Building filtered queries
- [SGP Propagator](../orbit_propagation/sgp_propagation.md) - SGP propagator usage
- [SpaceTrackClient API Reference](../../library_api/spacetrack/index.md)
