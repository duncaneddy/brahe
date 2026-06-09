# Managing External Data

Modern astrodynamics applications often require the use of external data for precision modeling of common phenomena. In particular, information on Earth reference frames, space weather, and planetary ephemerides (positions) are commonly provided through external data files calculated regularly by _product centers_ (e.g. NASA JPL, ESA, etc.) and made available to the public. While some applications don't need the level of accuracy provided by these data files, they are necessary for working with the current de facto standards in astrodynamics and are needed to perform calculations accurately and make them predictable.

<!-- Say you want to compare your orbit predictions with those of another operator to check for potential collisions. To accurately compare the predicted positions, they need to be expressed in the same reference frame. In most cases, that will be the GCRF (Geocentric Celestial Reference Frame) defined by the IAU and maintained by the IERS. However, if the other operator is using the ITRF reference frame, which is a rotating Earth-fixed frame, you would need to convert between the frames. This conversion requires knowledge of the Earth's orientation at each time step, which changes over time due to phenomena such as precession, nutation, and polar motion. While these values can be predicted, they cannot be known with perfect accuracy, so are continuously measured and updated by the IERS. This allows for reprocessing of past data with improved accuracy, as well as providing the most accurate data for current and future dates. -->

Historically, managing these data files has been a significant source of friction for users of astrodynamics software. You need to know where to get the data, how to download it, configure your software to use it, as well as regularly check for updates and download new versions of the data as they are released. For long-running applications, if you don't regularly update the data, the results degrade in accuracy over time or the application may stop working entirely if there are no data entries for the needed date.

Brahe provides a wide set of capabilities for downloading, setting, updating, and manging EOP and other external data files. However, the library still requires the user to be _aware_ of these data because the selection of which data to use is a _**modeling choice**_ that affects the accuracy and correctness of the results, which can ultimately only be determined by the user. As such, Brahe requires the user to explicitly load the data files before use, but generally the library handles the rest of the management process for you.

!!! tip "Do the Rightest Thing"
    One of the [core design principles](../about/design.md) of Brahe is to "Do the Rightest Thing". This means that the library should make it easy for users to do the most typically correct and accurate thing per current conventions in the field. However, it should not force users to only do the default choice. It should enable informed users to be able to make different modeling choices if they want to. However, since in most cases the users just want the "standard" choice, that should be able to be done with as little friction and boilerplate code as possible.

## Default Caching Provider

If you just want to get started quickly you can use the following two lines of code to initialize the default EOP and Space Weather (SW) data providers. Internally, brahe initializes the global gravity using the [CachingEOPProvider](../library_api/eop/caching_provider.md) and [CachingSpaceWeatherProvider](../library_api/space_weather/caching_provider.md) data providers, which use locally cached copies of the data files. If the data files are not present, or the data is too old (default 7 days), the providers will automatically download the latest versions of the data files from the source websites and cache them locally for future use.

The age of the data is only checked when the provider is initialized, so if you want to check for updates more frequently, or even whenever the data is accessed, you need to configure a different provider.

=== "Python"

    ``` python
    --8<-- "./examples/getting_started/load_external_data.py:4"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/getting_started/load_external_data.rs:1"
    ```

!!! tip "Note"
    Remember to add the necessary headers to your script. See the [First Script](first_script.md) page for more details.

???+ example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/getting_started/load_external_data.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/getting_started/load_external_data.rs.txt"
        ```

!!! note "Global Providers"
    One design choice of Brahe is that EOP and SW data providers are global entities, shared across all threads. This choice was made becaue the data is normally used extensively across many functions and calculations, so it would make for a repetitive API to pass around an EOP provider to nearly every function. Additionally, the data is only typically loaded once and then read many times, so it's possible to safely share a single copy across many threads reading it.

## Other Providers

If you don't want to use the default providers, such as needing to check for updates regularly (for long-running processes), you want to use a local file to avoid any network calls or guarantee reproducibility, or you want to ignore the data entirely, Brahe provides a wide variety of other providers that you can use. See the [EOP Providers](../learn/eop/index.md) and [Space Weather](../learn/space_weather/index.md) sections of the user guide for more information.

For quick reference the major provides you might want to use are shown below:



## Learning More