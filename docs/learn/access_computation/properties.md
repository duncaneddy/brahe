# Access Properties

Access properties are geometric and temporal measurements computed for each access window. Brahe automatically calculates six core properties during access searches, and users can add custom properties for mission-specific metadata or derived quantities.

Properties are stored using the `PropertyValue` enum, which supports multiple data types including scalars, vectors, time series, booleans, strings, and JSON objects. This flexible system allows you to attach any information needed for downstream analysis and mission planning.

## Core Properties

Brahe automatically computes these temporal and geometric properties for every access window:

- `window_open` - UTC time when access window starts
- `window_close` - UTC time when access window ends
- `duration` - Total duration of access window in seconds
- `midtime` - UTC time at midpoint of access window
- `azimuth_open` - Azimuth angle from location to satellite at window start (degrees)
- `azimuth_close` - Azimuth angle from location to satellite at window end (degrees)
- `elevation_min` - Minimum elevation angle during access window (degrees)
- `elevation_max` - Maximum elevation angle during access window (degrees)
- `local_time` - Local solar time at window midpoint in seconds \[0, 86400\)
- `look_direction` - Satellite look direction relative to velocity
- `asc_dsc` - Pass classification based on satellite motion

## Accessing Properties

Retrieve properties from `AccessWindow` objects:

=== "Python"

    ``` python
    --8<-- "./examples/access/properties/accessing_core_properties.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/properties/accessing_core_properties.rs:4"
    ```

## Custom Property Computers (Python)

Automatically compute custom properties for each access window during the search:

``` python
--8<-- "./examples/access/properties/custom_property_computer.py:11"
```

!!! tip "Property Computer Performance"
    Property computers are called once per access window after boundaries are refined. For performance-critical applications, minimize expensive calculations like coordinate transformations.

## See Also

- [Access Computation Overview](index.md)
- [Constraints](constraints.md)
- [Locations](locations.md)
