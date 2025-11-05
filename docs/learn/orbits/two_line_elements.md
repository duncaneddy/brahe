# Two-Line Elements (TLE)

Two-Line Element sets (TLEs) are a standardized format for representing satellite orbital data. Originally developed by NORAD (North American Aerospace Defense Command), TLEs encode an epoch, Keplerian orbital elements, and additional parameters needed for SGP4/SDP4 propagation into two 69-character ASCII lines.

An example of a TLE set for the International Space Station (ISS) is:

```
1 25544U 98067A   25302.48953544  .00013618  00000-0  24977-3 0  9995
2 25544  51.6347   1.5519 0004808 353.3325   6.7599 15.49579513535999
```

TLEs are still widely used for satellite tracking and orbit prediction, distributed by organizations like [CelesTrak](https://celestrak.org) and [Space-Track](https://www.space-track.org).

For additional information on the TLE format and field definitions, see the [CelesTrak TLE documentation](https://celestrak.org/columns/v04n03/) or the [Wikipedia TLE article](https://en.wikipedia.org/wiki/Two-line_element_set).

For complete API documentation, see the [TLE reference](../../library_api/orbits/tle.md).

!!! warning "TLE Accuracy Limitations"
    TLEs are designed for near-Earth satellites and have limited accuracy due to simplifications in the SGP4/SDP4 models. They **ARE NOT** suitable for high-precision orbit determination or long-term predictions.


!!! warning "NORAID ID Exhaustion"
    TLEs were originally designed for a maximum of 99,999 cataloged objects. However with the rise of mega-constellations and recent anti-satellite tests by Russia and China, the number of tracked objects is rapidly approaching this limit. 
    
    The Alpha-5 NORAD ID format extends the range by using letters A-Z (excluding I and O) as the leading character, allowing for up to 339,999 objects. This is a temporary solution however, and generally organizations should plan to transition to using formats like General Perturbations (GP) elements, CCSDS Orbit Ephemeris Messages (OEM), or other modern representations.

A common variant of TLEs is the Three-Line Element set (3LE), which adds a title line above the standard two lines for the object name. Brahe's TLE functions work with both TLE and 3LE formats interchangeably.

The same TLE data in 3LE format would be:

```
ISS (ZARYA)             
1 25544U 98067A   25302.48953544  .00013618  00000-0  24977-3 0  9995
2 25544  51.6347   1.5519 0004808 353.3325   6.7599 15.49579513535999
```

## Validating TLEs

Before parsing TLE data, you can validate the format and checksums to ensure data integrity.

### Validating a TLE Set

=== "Python"

    ```python
    --8<-- "examples/orbits/tle_validate_set.py:7"
    ```

=== "Rust"

    ```rust
    --8<-- "examples/orbits/tle_validate_set.rs:4"
    ```

The `validate_tle_lines` function checks that both lines have the correct format, valid checksums, and matching NORAD catalog numbers.

### Calculating Checksums

Each TLE line ends with a modulo-10 checksum. You can calculate this checksum to verify data integrity or when creating custom TLEs:

=== "Python"

    ```python
    --8<-- "examples/orbits/tle_calculate_checksum.py:7"
    ```

=== "Rust"

    ```rust
    --8<-- "examples/orbits/tle_calculate_checksum.rs:4"
    ```

!!! info "Checksum Algorithm"
    The checksum is calculated by summing all digits in the line (treating minus signs as 1) and taking the result modulo 10. All other characters (letters, spaces, periods) are ignored in the checksum calculation.

## Parsing TLEs

### Extracting Orbital Elements

The most common operation is parsing a TLE to extract the epoch and orbital elements:

=== "Python"

    ```python
    --8<-- "examples/orbits/tle_extract_elements.py:7"
    ```

=== "Rust"

    ```rust
    --8<-- "examples/orbits/tle_extract_elements.rs:4"
    ```

The returned elements follow the standard Brahe order: `[a, e, i, Ω, ω, M]` where:

- $a$ - Semi-major axis (meters)
- $e$ - Eccentricity (dimensionless)
- $i$ - Inclination (degrees)
- $\Omega$ - Right Ascension of Ascending Node (degrees)
- $\omega$ - Argument of Periapsis (degrees)
- $M$ - Mean Anomaly (degrees)

!!! tip "Angle Units Convention"
    TLE functions use **degrees** for all angles. This matches the TLE format standard and makes it easier to work with TLE data directly.

### Extracting Just the Epoch

If you only need the epoch timestamp without the full orbital elements:

=== "Python"

    ```python
    --8<-- "examples/orbits/tle_extract_epoch.py:7"
    ```

=== "Rust"

    ```rust
    --8<-- "examples/orbits/tle_extract_epoch.rs:4"
    ```

The epoch is always returned in the UTC time system.

## Creating TLEs

### From Keplerian Elements

You can generate TLE lines from an epoch and mean orbital elements:

=== "Python"

    ```python
    --8<-- "examples/orbits/tle_create_from_elements.py:7"
    ```

=== "Rust"

    ```rust
    --8<-- "examples/orbits/tle_create_from_elements.rs:4"
    ```

!!! info "Default Values"
    The `keplerian_elements_to_tle` function uses zero for fields like drag terms and derivatives. For complete control over all TLE fields, use the `create_tle_lines` function with its full parameter set.

!!! warning "Mean Element Representation"
    The TLE format encodes the orbital state as _mean orbital elements_ estimated from orbit propgation using the SGP4/SDP4 models.

    While the package allows for direclty creating TLEs from arbitrary Keplerian elements, the resulting TLEs **WILL NOT** accurate propagation results with the SGP4/SDP4 models unless the input elements are already mean elements derived from those models.

    If you need to create TLEs for real satellites it's best to estimate the mean elements from observed data using orbit determination techniques using the SGP4/SDP4 models.

You can verify generated TLEs by parsing them back:

=== "Python"

    ```python
    --8<-- "examples/orbits/tle_verify_roundtrip.py:10:25"
    ```

=== "Rust"

    ```rust
    --8<-- "examples/orbits/tle_verify_roundtrip.rs:9:40"
    ```

## NORAD ID Formats

TLEs support two formats for NORAD catalog numbers:

- **Numeric**: 5-digit numbers (00001-99999)
- **Alpha-5**: 5-character alphanumeric (A0000-Z9999)

The Alpha-5 format extends the catalog beyond 99,999 satellites by using letters A-Z (excluding I and O to avoid confusion with 1 and 0).

### Converting Between Formats

=== "Python"

    ```python
    --8<-- "examples/orbits/tle_convert_norad_formats.py:7:40"
    ```

=== "Rust"

    ```rust
    --8<-- "examples/orbits/tle_convert_norad_formats.rs:7:43"
    ```

!!! info "Alpha-5 Range"
    Alpha-5 format is only valid for NORAD IDs >= 100,000. The range is 100,000 (A0000) to 339,999 (Z9999).

### Parsing Mixed Formats

The `parse_norad_id` function automatically detects whether a NORAD ID is in numeric or Alpha-5 format:

=== "Python"

    ```python
    --8<-- "examples/orbits/tle_parse_norad_id.py:7:15"
    ```

=== "Rust"

    ```rust
    --8<-- "examples/orbits/tle_parse_norad_id.rs:7:17"
    ```

---

## See Also

- [SGP Propagator](../../library_api/propagators/sgp_propagator.md) - Use TLEs with SGP4/SDP4 propagation
- [Keplerian Elements](../../library_api/orbits/keplerian.md) - Working with orbital elements
- [Downloading TLE Data](../../examples/visualizing_starlink.md) - How to fetch current TLEs from online sources
- [Epoch](../time/epoch.md) - Understanding time representation in Brahe
