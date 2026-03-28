# Magnetic Field Models

Brahe computes Earth's geomagnetic field using spherical harmonic models. You provide a geodetic position and an epoch, and get back a three-component magnetic field vector in your choice of output frame. Two models are available: **IGRF-14** for broad historical coverage and **WMMHR-2025** for high spatial resolution near the current epoch.

For a complete listing of all function signatures and parameters, see the [IGRF API Reference](../../library_api/earth_models/igrf.md) and [WMMHR API Reference](../../library_api/earth_models/wmmhr.md).

## Computing the Field

The simplest call takes an epoch, a geodetic position `(longitude, latitude, altitude)`, and an angle format. The result is a three-element vector `[B_east, B_north, B_zenith]` in nanoTesla:

=== "Python"
    ``` python
    --8<-- "./examples/earth_models/igrf_field.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/earth_models/igrf_field.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/earth_models/igrf_field.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/earth_models/igrf_field.rs.txt"
        ```

The output components are:

- **$B_\text{east}$** -- eastward component (positive east)
- **$B_\text{north}$** -- northward component (positive north, tangent to the reference surface)
- **$B_\text{zenith}$** -- vertical component (positive upward, perpendicular to the reference surface)

From these you can derive the standard magnetic elements: horizontal intensity $H = \sqrt{B_e^2 + B_n^2}$, total intensity $F = |B|$, inclination $I = \arctan(-B_z / H)$ (positive when the field dips below horizontal), and declination $D = \arctan(B_e / B_n)$ (the compass deviation from true north).

## Output Frames

Each model offers three output frame variants. All take the same geodetic input -- only the frame of the returned field vector changes.

The **geodetic ENZ** functions (`igrf_geodetic_enz`, `wmmhr_geodetic_enz`) return the field relative to the WGS84 ellipsoid surface. "Zenith" points along the ellipsoid normal. This is the standard frame for geomagnetic applications and matches the convention used by NOAA's magnetic field calculators.

The **geocentric ENZ** functions (`igrf_geocentric_enz`, `wmmhr_geocentric_enz`) return the field relative to a geocentric sphere. "Zenith" points radially outward from Earth's center. At the equator the two frames coincide; at high latitudes they differ by up to ~0.2 degrees due to Earth's oblateness.

The **ECEF** functions (`igrf_ecef`, `wmmhr_ecef`) return the field in the Earth-Centered Earth-Fixed frame. This is useful when you need the field expressed in the same frame as satellite position vectors, for example when computing magnetic torques on a spacecraft.

## IGRF vs WMMHR

**IGRF-14** (International Geomagnetic Reference Field) covers 1900 to 2030. It models spherical harmonic degrees 1 through 13, capturing Earth's core field at ~3000 km spatial resolution. Coefficients are provided every 5 years and interpolated linearly for dates in between. Use IGRF when you need magnetic field values over long time spans or at historical epochs.

**WMMHR-2025** (World Magnetic Model High Resolution) covers approximately 2025 to 2030. It extends to spherical harmonic degree 133, adding crustal magnetic anomalies at ~300 km resolution on top of the core field. Use WMMHR when you need the most accurate current field values, particularly at or near Earth's surface where crustal contributions matter.

=== "Python"
    ``` python
    --8<-- "./examples/earth_models/wmmhr_field.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/earth_models/wmmhr_field.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/earth_models/wmmhr_field.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/earth_models/wmmhr_field.rs.txt"
        ```

The `nmax` parameter on WMMHR functions controls the maximum spherical harmonic degree. Setting `nmax=13` gives results comparable to standard WMM/IGRF resolution. The default (`None` / `None`) uses the full 133 degrees.

## Using with Satellite ECEF Positions

Satellite positions are often available in ECEF or ECI coordinates rather than geodetic. The typical workflow is: convert the ECEF position to geodetic using `position_ecef_to_geodetic`, then call the magnetic field function with the geodetic result.

=== "Python"
    ``` python
    --8<-- "./examples/earth_models/magnetic_field_from_ecef.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/earth_models/magnetic_field_from_ecef.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/earth_models/magnetic_field_from_ecef.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/earth_models/magnetic_field_from_ecef.rs.txt"
        ```

!!! info
    The input altitude is always in **meters** (SI), consistent with all other Brahe functions. The models work internally in kilometers but handle the conversion automatically.

## See Also

- [IGRF API Reference](../../library_api/earth_models/igrf.md) -- Full function signatures for IGRF-14
- [WMMHR API Reference](../../library_api/earth_models/wmmhr.md) -- Full function signatures for WMMHR-2025
- [Geodetic Coordinates](../coordinates/geodetic_transformations.md) -- Converting between ECEF and geodetic
- [Reference Frame Transformations](../frames/eci_ecef.md) -- ECI to ECEF conversion for satellite positions
