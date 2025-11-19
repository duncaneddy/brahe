# Managing Space Weather Data

Brahe provides a global space weather provider that supplies geomagnetic indices and solar flux data when needed. If you want to skip the details for now, initialize the global provider with defaults:

=== "Python"

    ``` python
    --8<-- "./examples/space_weather/initialize_sw.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/space_weather/initialize_sw.rs:4"
    ```

!!! warning

    Space weather data **MUST** be initialized before using any functionality that requires it. If no data is initialized, brahe will panic and terminate the program.

    The data is used by atmospheric drag models to compute density variations.

## Space Weather Providers

Brahe defines three provider types with different use cases.

### StaticSpaceWeatherProvider

A static provider uses fixed values for all space weather parameters. This is useful for testing or when you want reproducible results with known conditions.

=== "Python"

    ``` python
    --8<-- "./examples/space_weather/static_sw.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/space_weather/static_sw.rs:3"
    ```

### FileSpaceWeatherProvider

Load space weather data from CSSI format files. Brahe includes a default data file that is updated with each release.

=== "Python"

    ``` python
    --8<-- "./examples/space_weather/file_sw.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/space_weather/file_sw.rs:3"
    ```

### CachingSpaceWeatherProvider

The caching provider automatically downloads and manages space weather data files from CelesTrak. It checks file age and updates when the cache becomes stale.

=== "Python"

    ``` python
    --8<-- "./examples/space_weather/caching_sw.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/space_weather/caching_sw.rs:3"
    ```

## Extrapolation Options

When querying dates outside the available data range, the provider behavior depends on the extrapolation setting:

- **`"Zero"`**: Return zero values for all parameters
- **`"Hold"`**: Return the last (or first) available value
- **`"Error"`**: Panic and terminate the program

## Accessing Space Weather Data

Query space weather data using the global functions:

=== "Python"

    ``` python
    --8<-- "./examples/space_weather/accessing_sw_data.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/space_weather/accessing_sw_data.rs:3"
    ```

## Range Data Access

The space weather providers also support querying data over a date range, returning a vector of values from before the specific time. This is useful to providing the weather history for drag models.

=== "Python"

    ``` python
    --8<-- "./examples/space_weather/historical_sw_data.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/space_weather/historical_sw_data.rs:3"
    ```

---

## See Also

- [StaticSpaceWeatherProvider API Reference](../../library_api/space_weather/static_provider.md)
- [FileSpaceWeatherProvider API Reference](../../library_api/space_weather/file_provider.md)
- [CachingSpaceWeatherProvider API Reference](../../library_api/space_weather/caching_provider.md)
- [Space Weather Functions](../../library_api/space_weather/functions.md)
