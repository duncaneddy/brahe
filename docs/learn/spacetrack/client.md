# Client Usage

The `SpaceTrackClient` provides authenticated access to the Space-Track.org API. Authentication happens automatically when the client is created.

## Creating a Client

Create a client with your Space-Track.org credentials:

=== "Python"

    ``` python
    --8<-- "./examples/spacetrack/client_creation.py:8"
    ```

## Querying GP Data

The GP (General Perturbations) endpoint returns orbital elements for cataloged objects. Query by NORAD ID for specific satellites:

=== "Python"

    ``` python
    --8<-- "./examples/spacetrack/query_gp.py:8"
    ```

## Querying Satellite Catalog

The SATCAT endpoint provides metadata about satellites including launch date, decay status, and ownership:

=== "Python"

    ``` python
    --8<-- "./examples/spacetrack/query_satcat.py:8"
    ```

## Other Data Types

The client provides methods for additional data types:

| Method | Description |
|--------|-------------|
| `tle()` | Legacy TLE data (deprecated, use `gp()`) |
| `decay()` | Re-entry predictions and actual decay data |
| `tip()` | Tracking and Impact Prediction messages |
| `cdm_public()` | Public conjunction data messages |
| `boxscore()` | Catalog statistics by country |
| `launch_site()` | Launch facility information |
| `satcat_change()` | Catalog change history |
| `satcat_debut()` | New catalog entries |
| `announcement()` | Space-Track announcements |

## Generic Requests

For advanced use cases not covered by typed methods, use `generic_request()`:

=== "Python"

    ``` python
    --8<-- "./examples/spacetrack/generic_request.py:8"
    ```

## See Also

- [Query Filters](queries.md) - Building filtered queries with operators
- [Propagator Integration](propagators.md) - Converting to SGP propagators
- [SpaceTrackClient API Reference](../../library_api/spacetrack/index.md)
