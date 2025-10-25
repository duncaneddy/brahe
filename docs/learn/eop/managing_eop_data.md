# Managing EOP Data

Generally, users of brahe will not need to directly manage Earth orientation data. The package provides default data files and the `CachingEOPProvider` to automatically update data as needed. However, for advanced users or those with specific data requirements, brahe provides functionality to load and manage Earth orientation data manually.

To make the package interface ergonommic, brahe functions do not explicitly accept Earth orientation data as input parameters. Instead, there is a single, global Earth orientation provider used internally by brahe functions. This global provider can be initialized using one of the provided loading functions. See the [Loading Data Sets](#loading-data-data-sets) section for more information on loading and managing Earth orientation data in brahe.

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

### Functions


## Accessing EOP Parameters



<!-- Most software using this library requires upfront, explicit initialization of the  static Earth orientation data. Earth orientation data is loaded globally by calling one of the provided loading methods: `set_global_eop_from_zero`, `set_global_eop_from_static_values`, `set_global_eop_from_c04_file`, `set_global_eop_from_default_c04`, `set_global_eop_from_standard_file`, or `set_global_eop_from_default_standard`. These methods can be called multiple times to reset or override the currently loaded data.

The `set_global_eop_from_zero` will initialize the global data with zeroed values. This enables usage of other module functionality, but does not provide the most accurate modeling of Earth or time systems. It should be used when a quick, approximately correct answer is needed. `set_global_eop_from_static_values` is a similar initialization method which configures the module with a single set of Earth orientation data used for all transformations.

To configure more accurate Earth orientation data to use in the module, `set_global_eop_from_c04_file` can be used to load long-term IERS C04 products and `set_global_eop_from_standard_file` to load either Bulletin A or Bulletin B data from the IERS standard file product format.

brahe distributions also include packaged IERS C04 and Bulletin A/B data. These can be configured using `set_global_eop_from_default_c04` or `set_global_eop_from_default_standard`, respectively. While not updated regularly.

For the most accurate Earth orientation data modeling in scripts, you should download the latest available Earth orientation data for the desired model and the using the file-based loading methods (`set_global_eop_from_c04_file` or `set_global_eop_from_standard_file`) to initialize the Earth orientation data based on the file.

When creating any new Earth Orientation data instance there are two parameters that are set at loading time which will determine how the EOP instances handles data returns for certain cases. The first parameter is the `extrapolate` parameter, which can have a value of `Zero`, `Hold`, or `Error`. This value will determine how requests for data points beyond the end of the loaded data are handled. The possible behaviors are
- `Zero`: Returned values will be `0.0` where data is not available
- `Hold`: Will return the last available returned value when data is not available
- `Error`: Data access attempts where data is not present will panic and terminate the program

The second parameter the `interpolate` setting. When `interpolate` is set to true and data requests made for a point that wasn't explicitly loaded as part of the input data set will be linearly interpolated to the desired time. When set to `false`, the function call will return the last value prior to the requested data.

Below is an example of loading C04 data

=== "Rust"

    ``` rust
    --8<-- "../examples/eop_c04_loading.rs"
    ```

=== "Python"

    ``` python
    --8<-- "../examples/eop_c04_loading.py"
    ```

The process for loading standard data is similar. However, when loading standard files there is one other parameter which comes into play, the Earth Orientation Type. This type-setting determines whether the Bulletin A or Bulletin B data is loaded into the object when parsing the file. In rust

=== "Rust"

    ``` rust
    --8<-- "../examples/eop_standard_loading.rs"
    ```

=== "Python"

    ``` python
    --8<-- "../examples/eop_standard_loading.py"
    ```

!!! note

    For applications where the time is in the future it is recommended to use standard EOP data as standard files contain predictions for approximately 1 year into the future and will increase accuracy of analysis by accounting for Earth orientation corrections.

    For analysis for scenarios in the past it is recommended to use the final C04 products as they contain the highest accress estimates of Earth orientation data.

### Accessing Earth Orientation Data

Most of the time the data stored by the Earth orientation object is not used directly. If your application calls for accessing the `EarthOrientationProvider` object provides a number of methods for accessing different Earth orientation Parameters stored by the object. However, in mostcases, it is best to use the data for the crate's loaded static Earth orientation data. In these cases the following methods can be used to access the loaded static Earth orientation data:
- `get_global_ut1_utc`
- `get_global_pm`
- `get_global_dxdy`
- `get_global_lod`
- `get_global_eop`

The following methods return information on the currently loaded Earth orientation data:
- `get_global_eop_initialization`
- `get_global_eop_len`
- `get_global_eop_type`
- `get_global_eop_extrapolate`
- `get_global_eop_interpolate`
- `get_global_eop_mjd_min`
- `get_global_eop_mjd_max`
- `get_global_eop_mjd_last_lod`
- `get_global_eop_mjd_last_dxdy`

=== "Rust"

    ``` rust
    --8<-- "../examples/eop_data_access.rs"
    ```

=== "Python"

    ``` python
    --8<-- "../examples/eop_data_access.py"
    ```

### Downloading updated Earth Orientation Data

The final functionality that Brahe provides is the ability to download new Earth orientation parameter data files.

The functions `download_c04_eop_file` and `download_standard_eop_file` can be used to downloaded the latest product files from IERS servers and store them locally at the specified filepath. The download functions will attempt to create the necessary directory structure if required.

=== "Rust"

    ``` rust
    use brahe::eop::{download_c04_eop_file, download_standard_eop_file};

    fn main() {
        // Download latest C04 final product file
        download_c04_eop_file("./c04_file.txt").unwrap();
    
        // Download latest standard product file
        download_standard_eop_file("./standard_file.txt").unwrap();
    }
    ```

=== "Python"

    ``` python
    import brahe

    if __name__ == '__main__':
        # // Download latest C04 final product file
        brahe.eop.download_c04_eop_file("./c04_file_py.txt")
    
        # // Download latest standard product file
        brahe.eop.download_standard_eop_file("./standard_file_py.txt")
    ```

If using the brahe CLI, product files can be download with

```bash
brahe eop download --product final final_c04_eop_file.txt
```

or 

```bash
brahe eop download --product standard standard_eop_file.txt
```
 -->

## See Also

- [StaticEOPProvider API Reference](../../library_api/eop/static_provider.md)
- [FileEOPProvider API Reference](../../library_api/eop/file_provider.md)
- [CachingEOPProvider API Reference](../../library_api/eop/caching_provider.md)
