# Coordinate Transformations

In dynamics we are primarily concerned with the object _state_, which encodes the location and often the rate of change of location of an object. This state is often expressed in a particular coordinate system, such as Cartesian coordinates (position and velocity) or Keplerian orbital elements. It also supports transforming to topocentric coordinates such as Geodetic coordinates. Brahe provides functions for transforming between different coordinate systems.

## Keplerian and Cartesian Transformations

We can take a state in Keplerian orbital elements and transform it into Cartesian orbital elements using `state_koe_to_eci`. We can invert this transformation using `state_eci_to_koe` to get back the original Keplerian orbital elements.

=== "Python"

    ``` python
    --8<-- "./examples/getting_started/coordinates_keplerian_cartesian.py:4"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/getting_started/coordinates_keplerian_cartesian.rs:1"
    ```

???+ example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/getting_started/coordinates_keplerian_cartesian.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/getting_started/coordinates_keplerian_cartesian.rs.txt"
        ```


## Cartesian to Geodetic

Similarly we can convert from a Cartesian state in ECEF to geodetic coordinates (longitude, latitude, altitude) using `state_ecef_to_geodetic`.

=== "Python"

    ``` python
    --8<-- "./examples/getting_started/coordinates_cartesian_geodetic.py:4"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/getting_started/coordinates_cartesian_geodetic.rs:1"
    ```

???+ example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/getting_started/coordinates_cartesian_geodetic.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/getting_started/coordinates_cartesian_geodetic.rs.txt"
        ```
