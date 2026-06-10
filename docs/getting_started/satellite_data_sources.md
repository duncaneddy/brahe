# Satellite Data Sources

For many modeling tasks it is useful to access satellite ephemeris (orbit) data. This data is made available by a number of sources, including public sources such as [Celestrak](https://celestrak.com/) and [Space-Track](https://www.space-track.org/). Brahe provides functions for accessing satellite data from both of these sources, as well as initializing SGP4 propagators from data.

Both clients have integrated, default rate-limiting and caching to ensure efficient and responsible access to the data. For more information on the configuration of the clients, see the respective language API documentation.

!!! tip "Moving Beyond TLEs"
    While TLEs have been historically used for satellite ephemeris data, we will soon encounter the problem of catalog number exhaustion. Brahe supports both alpha-5 and GP Record formats that are being adopted to address the issue.

## Celestrak

The celestrack client 

=== "Python"

    ``` python
    --8<-- "./examples/getting_started/clients_celestrak.py:4"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/getting_started/clients_celestrak.rs:1"
    ```

???+ example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/getting_started/clients_celestrak.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/getting_started/clients_celestrak.rs.txt"
        ```

## Space-Track

The space-track client requires a user account to access the data. Once authenticated, the client provides access to a variety of satellite data, including TLEs, GP Records, and other records.

=== "Python"

    ``` python
    --8<-- "./examples/getting_started/clients_spacetrack.py:5"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/getting_started/clients_spacetrack.rs:2"
    ```

???+ example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/getting_started/clients_spacetrack.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/getting_started/clients_spacetrack.rs.txt"
        ```
