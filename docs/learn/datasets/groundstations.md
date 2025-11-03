# Groundstation Datasets

## Overview

Groundstation datasets provide geographic locations and metadata for commercial satellite ground facilities worldwide. This data is essential for:

- **Computing contact opportunities**: Determine when satellites are visible from ground stations
- **Network planning**: Analyze coverage and redundancy across multiple providers
- **Mission design**: Evaluate downlink opportunities for different orbit configurations

Brahe includes embedded GeoJSON data for 6 major commercial groundstation providers, totaling 50+ facilities globally. All data is:

- **Offline-capable**: No network requests required
- **Comprehensive**: Global coverage across multiple providers
- **Standardized**: Consistent format with geographic coordinates and metadata
- **Up-to-date**: Maintained as provider networks evolve

### When to Use

Use groundstation datasets when you need to:

- Compute visibility windows for satellite-to-ground contacts
- Plan downlink schedules for data collection
- Analyze network coverage and redundancy
- Compare provider capabilities across different locations

## Available Providers

Brahe includes groundstation data from six major commercial providers:

| Provider | Description |
|----------|-------------|
| **Atlas** | Atlas Space Operations |
| **AWS** | Amazon Web Services Ground Station |
| **KSAT** | Kongsberg Satellite Services |
| **Leaf** | Leaf Space |
| **SSC** | Swedish Space Corporation |
| **Viasat** | Viasat |

## Usage

### Loading Groundstations

Load groundstation data from one or more providers:

=== "Python"

    ``` python
    --8<-- "./examples/datasets/groundstations_load.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/datasets/groundstations_load.rs:7"
    ```

### Accessing Properties

Each groundstation includes geographic coordinates and metadata:

=== "Python"

    ``` python
    --8<-- "./examples/datasets/groundstations_properties.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/datasets/groundstations_properties.rs:7"
    ```

### Computing Access Windows

Use groundstation data with brahe's access computation to find contact opportunities:

=== "Python"

    ``` python
    --8<-- "./examples/datasets/groundstations_access.py:11"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/datasets/groundstations_access.rs:7"
    ```

## Data Format

Each groundstation is represented as a `PointLocation` with standardized properties:

```python
import brahe as bh

stations = bh.datasets.groundstations.load("ksat")
station = stations[0]

# Geographic coordinates (WGS84)
lon = station.lon()      # Longitude in degrees
lat = station.lat()      # Latitude in degrees
alt = station.alt()      # Altitude in meters

# Metadata properties
props = station.properties
name = station.get_name()              # Station name
provider = props["provider"]            # Provider name (e.g., "KSAT")
bands = props["frequency_bands"]        # Supported bands (e.g., ["S", "X"])
```

All groundstations include these standard properties:

- **`provider`**: Provider name (string, e.g., "KSAT", "Atlas")
- **`frequency_bands`**: List of supported frequency bands (e.g., `["S", "X", "Ka"]`)

Additional properties may be included in future releases as data becomes available.

## See Also

- [Datasets Overview](index.md) - Understanding datasets in Brahe
- [Groundstation API Reference](../../library_api/datasets/groundstations.md) - Complete function documentation
