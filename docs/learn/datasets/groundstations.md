# Groundstation Datasets

<!-- Fill in overview -->

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

### Combining Networks

For comprehensive coverage, combine multiple providers:

```python
import brahe as bh

# Load multiple providers
primary = bh.datasets.groundstations.load("ksat")
backup = bh.datasets.groundstations.load("ssc")

# Combine into single network
combined = primary + backup

# Compute access with redundant coverage
accesses = bh.location_accesses(
    combined,
    propagator,
    start,
    end,
    constraint
)
```

## See Also

- [Datasets Overview](index.md) - Understanding datasets in Brahe
- [Groundstation API Reference](../../library_api/datasets/groundstations.md) - Complete function documentation
