# Orbit Propagation

Orbit propagation is the process of calculating the future state of an orbiting object based on its current state. Brahe provides functions for propagating orbits with different orbit propagators. Two of the most commonly used are numerical propagation and SGP4 propagation.

## Numerical Propagation

Brahe provides an easy-to-use interface for numerical propgation. It supports a variety of integrators and force models, in addition to supporting integration of the variational equations to compute state transition and sensitivity matrices.

For additional information on the configuration of the numerical propagator see the [Force Model Configuration](../learn/orbit_propagation/numerical_propagation/force_models.md) and [Integrator Configuration](../learn/orbit_propagation/numerical_propagation/integrator_configuration.md) documetnation pages.

There are also a number of default configuration options available for both the force model and integrator, for common use cases. For the varities of default configurations available, see the respective language API documentation.

There are also additional capabilities such as [event detection](../learn/orbit_propagation/numerical_propagation/event_detection.md), [control inputs](../learn/orbit_propagation/numerical_propagation/maneuvers.md), and even [extending the state vector](../learn/orbit_propagation/numerical_propagation/extending_state.md) that can be found in the documentation.

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