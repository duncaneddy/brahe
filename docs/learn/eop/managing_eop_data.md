# Managing EOP Data

Generally, users of brahe will not need to directly manage Earth orientation data. The package provides default data files and the `CachingEOPProvider` to automatically update data as needed. However, for advanced users or those with specific data requirements, brahe provides functionality to load and manage Earth orientation data manually.

To make the package interface ergonommic, brahe functions do not explicitly accept Earth orientation data as input parameters. Instead, there is a single, global Earth orientation provider used internally by brahe functions. This global provider can be initialized using one of the provided loading functions.

If you want to skip understanding Earth orientation data for now, you can initialize the global provider with zeroed values using the `initialize_eop()` function:

=== "Python"

    ``` python
    --8<-- "./examples/eop/initialize_eop.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/eop/initialize_eop.rs:4"
    ```

!!! warning 

    Earth orientation data **MUST** be initialized before using any functionality in brahe that requires Earth orientation data. If no data is initialized, brahe will panic and terminate the program when Earth orientation data is requested.

## Earth Orientation Providers

Brahe defines the `EarthOrientationProvider` trait to provide a common interface for accessing Earth orientation data. There are multiple different types of providers, each with their own use cases. The package includes default data files for ease of use that are sufficient for most purposes.

For the most accurate Earth orientation data modeling in scripts, you should download the latest available Earth orientation data for the desired model and the using the file-based loading methods. Alternatively you can the `CachingEOPProvider` to initialize the Earth orientation data which will automatically download and update the latest data files as needed.

### StaticEOPProvider

A static provider is one that just uses fixed values for Earth orientation parameters. This provider is useful for testing and development or if your application only requires low accuracy.

=== "Python"

    ``` python
    --8<-- "./examples/eop/static_eop.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/eop/static_eop.rs:3"
    ```

### FileEOPProvider

If you want to use high-accuracy Earth orientation data, you can load data from IERS files using the `FileEOPProvider`. Brahe provides functions to load default IERS data files provided with the package, or you can specify your own file paths.

When creating any new file-based data provider there are two parameters that are set at loading time which will determine how the EOP instances handles data returns for times not in the loaded data.

The first parameter is the `interpolate` setting. When `interpolate` is set to `True` and data set will be linearly interpolated to the desired time. When set to `False`, the function call will return the last value prior to the requested data. Given that IERS data is typically provided at daily intervals, it is generally recommended to enable interpolation for most applications.

The second parameter is the `extrapolate` parameter, which can have a value of `Zero`, `Hold`, or `Error`. This value will determine how requests for data points beyond the end of the loaded data are handled. The possible behaviors are

- `Zero`: Returned values will be `0.0` where data is not available
- `Hold`: Will return the last available returned value when data is not available
- `Error`: Data access attempts where data is not present will panic and terminate the program

You can create a file-based Earth orientation provider by specifying the file paths to the desired data files as follows:

=== "Python"

    ``` python
    --8<-- "./examples/eop/file_eop.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/eop/file_eop.rs:3"
    ```


### CachingEOPProvider

The `CachingEOPProvider` is a `FileEOPProvider` that automatically downloads and caches the latest Earth orientation data files from the IERS website as needed. It checks the age of the cached data and if the data is older than a specified threshold, it downloads the latest files, then loads them for use. This provider can also be configured to check for a stale cache on use and update the data if needed, which is useful for long-running applications.

The `CachingEOPProvider` is the recommended provider for most applications as it provides high-accuracy Earth orientation data without requiring manual management of data files. `initialize_eop()` uses this provider by default.

The interpolation and extrapolation parameters are also available when creating a `CachingEOPProvider`, with the same behavior as described for the `FileEOPProvider`.

=== "Python"

    ``` python
    --8<-- "./examples/eop/caching_eop.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/eop/caching_eop.rs:3"
    ```


## Downloading EOP Data Files

If you want to manually download Earth orientation data files to store or save them, brahe provides two means of doing so. The first is through the command-line interface (CLI) tool included with brahe. The second is through direct function calls in either the Rust or Python APIs.

### CLI

The brahe CLI command includes an `eop download` subcommand which can be used to download the latest Earth orientation data files from IERS servers.

To download the latest standard product file, use the following command:

```bash
brahe eop download --product standard <output_filepath>
```

To download the latest C04 final product file, use the following command:

```bash
brahe eop download --product c04 <output_filepath>
```

### Functions

You can also download Earth orientation data files directly using the `download_standard_eop_file` and `download_c04_eop_file` functions in the `brahe.eop` module.

You can download the latest standard EOP data file as follows:

=== "Python"

    ``` python
    import brahe as bh

    # Download latest standard EOP data
    bh.download_standard_eop_file("./eop_data/standard_eop.txt")
    ```

=== "Rust"

    ``` rust
    use brahe::eop::download_standard_eop_file;

    // Download latest C04 EOP data
    download_standard_eop_file("./eop_data/finals2000A.all.csv")
    ```


Or download the latest C04 final product file as follows:

=== "Python"

    ``` python
    import brahe as bh

    # Download latest C04 EOP data
    bh.download_c04_eop_file("./eop_data/finals2000A.all.csv")
    ```

=== "Rust"

    ``` rust
    use brahe::eop::download_c04_eop_file;

    // Download latest C04 EOP data
    download_c04_eop_file("./eop_data/finals2000A.all.csv")
    ```


## Accessing EOP Parameters

While not common it is possible to directly access Earth orientation parameters from the currently loaded global Earth orientation provider. This can be useful for debugging or analysis purposes.

=== "Python"

    ``` python
    --8<-- "./examples/eop/accesing_eop_data.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/eop/accesing_eop_data.rs:3"
    ```

You can find more functions to access specific subsets of Earth orientation data in the [API Reference](../../library_api/eop/functions.md).

## See Also

- [StaticEOPProvider API Reference](../../library_api/eop/static_provider.md)
- [FileEOPProvider API Reference](../../library_api/eop/file_provider.md)
- [CachingEOPProvider API Reference](../../library_api/eop/caching_provider.md)
