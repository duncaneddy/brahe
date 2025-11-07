# Access Properties

Access properties are geometric and temporal measurements computed for each access window. Brahe automatically calculates core properties during access searches, and provides both built-in and custom property computers for mission-specific analysis.

## Core Properties

Brahe automatically computes these temporal and geometric properties for every access window:

| Name | Type | Description |
|------|------|-------------|
| `window_open` | [`Epoch`](../../library_api/time/epoch.md) | UTC time when access window starts |
| `window_close` | [`Epoch`](../../library_api/time/epoch.md) | UTC time when access window ends |
| `duration` | `float` | Total duration of access window in seconds |
| `midtime` | [`Epoch`](../../library_api/time/epoch.md) | UTC time at midpoint of access window |
| `azimuth_open` | `float` | Azimuth angle from location to satellite at window start (degrees) |
| `azimuth_close` | `float` | Azimuth angle from location to satellite at window end (degrees) |
| `elevation_min` | `float` | Minimum elevation angle during access window (degrees) |
| `elevation_max` | `float` | Maximum elevation angle during access window (degrees) |
| `local_time` | `float` | Local solar time at window midpoint in seconds $\left[0, 86400\right)$ |
| `look_direction` | [`LookDirection`](../../library_api/access/enums.md#lookdirection) | Satellite look direction relative to velocity |
| `asc_dsc` | [`AscDsc`](../../library_api/access/enums.md#ascdsc) | Pass classification based on satellite motion |

Core properties are attributes of the `AccessWindow` object returned by access computations and can be accessed directly like `window.window_open` or `window.elevation_max`.

Below are examples of accessing core properties in Python and Rust.

=== "Python"

    ``` python
    --8<-- "./examples/access/properties/accessing_core_properties.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/properties/accessing_core_properties.rs:4"
    ```

## Property Computers

Property computers allow users to extend the access computation system to define and compute custom properties for each access window beyond the core set. These computations are performed after access windows are identified and refined. 

Python users can implement property computers by subclassing [`AccessPropertyComputer`](../../library_api/access/properties.md#accesspropertycomputer), while in Rust you implement the `AccessPropertyComputer` trait. These traits require the implementation of the `sampling_config` and `compute` methods. `sampling_config` defines how satellite states are sampled during the access window, and `compute` performs the actual property calculation using those sampled states.

Brahe defines a few built-in property computers for common use cases, and users can create custom property computers for application-specific needs.

## Sampling Configuration

Property computers use [`SamplingConfig`](../../library_api/access/properties.md#samplingconfig) to determine when satellite states are sampled within the access window. That is, what `epoch, state` pairs are provided to the computer for its calculations.

You can choose from several sampling modes:

- `relative_points([0.0, 0.5, 1.0])` - Samples at specified fractions of the window duration with 0.0 being the start and 1.0 being the end
- `fixed_count(n)` - Samples a fixed number of evenly spaced points within the window
- `fixed_interval(interval, offset)` - Samples at regular time intervals (defined by seconds between samples) throughout the window with an optional offset
- `midpoint` - Samples only at the midpoint of the window

This allows you to compute time-series data at specific intervals or points.

### Sampling Modes

=== "Python"

    ``` python
    --8<-- "./examples/access/properties/sampling_config_examples.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/properties/sampling_config_examples.rs:4"
    ```

## Built-in Property Computers

Brahe provides three commonly-used property computers optimized in Rust:

### DopplerComputer

Computes Doppler frequency shifts for uplink and/or downlink communications:

=== "Python"

    ``` python
    --8<-- "./examples/access/properties/builtin_doppler_computer.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/properties/builtin_doppler_computer.rs:4"
    ```

**Doppler Physics:**

- **Uplink**: $\Delta f = f_0\frac{v_{los}}{c - v_{los}}$ - Ground station pre-compensates transmit frequency
- **Downlink**: $\Delta f = -f_0\frac{v_{los}}{c}$ - Ground station adjusts receive frequency
- Where $v_{los}$ is the velocity of the object along the line of sight from the observer. With $v_{los} < 0$ when approaching and $v_{los} > 0$ when receding.

### RangeComputer

Computes slant range (distance) from the location to the satellite:

=== "Python"

    ``` python
    --8<-- "./examples/access/properties/builtin_range_computer.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/properties/builtin_range_computer.rs:4"
    ```

### RangeRateComputer

Computes line-of-sight velocity (range rate) with the convention that positive values indicate increasing range (satellite receding) and negative values indicate decreasing range (satellite approaching):

=== "Python"

    ``` python
    --8<-- "./examples/access/properties/builtin_range_rate_computer.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/properties/builtin_range_rate_computer.rs:4"
    ```


## Custom Property Computers

You can also create your own property computer to compute application-specific properties values. The system will pre-sample the satellite state at the specified times defined by your [`SamplingConfig`](../../library_api/access/properties.md#samplingconfig), so you don't need to manually propagate the trajectory.

This section provides examples of custom property computers in both Python and Rust.

### Python Implementation

In python you subclass [`AccessPropertyComputer`](../../library_api/access/properties.md#accesspropertycomputer) and implement three methods:

=== "Python"

    ``` python
    --8<-- "./examples/access/properties/custom_max_speed.py:8"
    ```

### Combining Multiple Computers

Pass multiple computers to compute different properties simultaneously:

=== "Python"

    ``` python
    --8<-- "./examples/access/properties/combining_multiple_computers.py:8"
    ```

### Rust Implementation

To implement a custom property computer in Rust, create a struct that implements the `AccessPropertyComputer` trait by defining the `sampling_config` and `compute` methods.

=== "Rust"

    ``` rust
    --8<-- "./examples/access/properties/custom_max_speed.rs:8"
    ```

---

## See Also

- [Access Computation Overview](index.md)
- [Constraints](constraints.md)
- [Locations](locations.md)
- [Computation Configuration](computation.md)
