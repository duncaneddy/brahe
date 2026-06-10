# Orbit Propagation

Orbit propagation is the process of calculating the future state of an orbiting object based on its current state. Brahe provides functions for propagating orbits with both high-precision numerical propagation.

## Numerical Propagation

=== "Python"

    ``` python
    --8<-- "./examples/getting_started/propagation_numerical.py:4"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/getting_started/propagation_numerical.rs:1"
    ```

???+ example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/getting_started/propagation_numerical.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/getting_started/propagation_numerical.rs.txt"
        ```


## SGP4 Propagation

=== "Python"

    ``` python
    --8<-- "./examples/getting_started/orbits_period.py:4"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/getting_started/orbits_period.rs:1"
    ```

???+ example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/getting_started/orbits_period.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/getting_started/orbits_period.rs.txt"
        ```
