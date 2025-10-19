# Groundstation Datasets

Brahe includes curated groundstation location datasets from commercial satellite communication providers. These datasets provide standardized geographic coordinates and metadata for ground stations used in satellite operations, tracking, and communications.

## Overview

Groundstation data in Brahe is embedded directly in the library, requiring no external files or network access. The data represents real commercial groundstation networks maintained by major satellite service providers.

**Data Source**: Publicly available information compiled from provider websites and documentation

**Format**: GeoJSON FeatureCollections parsed into `PointLocation` objects

**Update Frequency**: Updated periodically with Brahe releases

## Available Providers

Brahe includes groundstation data from six major commercial providers:

| Provider | Description | Network Type |
|----------|-------------|--------------|
| **Atlas** | Atlas Space Operations | Commercial ground network |
| **AWS** | Amazon Web Services Ground Station | Cloud-based ground services |
| **KSAT** | Kongsberg Satellite Services | Global polar network |
| **Leaf** | Leaf Space | Commercial ground network |
| **SSC** | Swedish Space Corporation | Commercial and institutional |
| **Viasat** | Viasat | Communication services |

### Provider Characteristics

**Atlas Space Operations**
- Modern cloud-based ground station network
- Focus on automated operations and API access
- Growing global coverage
- Multiple frequency band support

**AWS Ground Station**
- Cloud-native ground station service
- Pay-as-you-go pricing model
- Integration with AWS services
- Strategic global placement

**KSAT (Kongsberg Satellite Services)**
- Extensive polar coverage (Arctic and Antarctic)
- Long operational history
- Large established network
- Strong Earth observation support

**Leaf Space**
- European-based commercial provider
- Focus on LEO satellite support
- Flexible antenna sharing
- Modern infrastructure

**SSC (Swedish Space Corporation)**
- Mix of commercial and institutional facilities
- Strong presence in northern latitudes
- Launch support capabilities
- Experienced operator

**Viasat**
- Primarily communications-focused
- Global coverage
- High-capacity infrastructure
- Commercial satellite services

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

### Properties Dictionary

All groundstations include these standard properties:

- **`provider`**: Provider name (string, e.g., "KSAT", "Atlas")
- **`frequency_bands`**: List of supported frequency bands (e.g., `["S", "X", "Ka"]`)

Additional properties may be included in future releases as data becomes available.

### Frequency Bands

Common frequency bands in groundstation data:

| Band | Frequency Range | Typical Use |
|------|----------------|-------------|
| S    | 2-4 GHz       | TT&C, telemetry |
| X    | 8-12 GHz      | High-rate downlink, radar |
| Ku   | 12-18 GHz     | Communications |
| Ka   | 26-40 GHz     | High-bandwidth communications |

## Use Cases

### Access Analysis

Compute visibility windows between satellites and ground networks:

```python
import brahe as bh

# Load ground network
stations = bh.datasets.groundstations.load("ksat")

# Create satellite propagator
tle1 = "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997"
tle2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"
propagator = bh.SGPPropagator.from_tle(tle1, tle2)

# Define search period
start = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0, tsys="UTC")
end = bh.Epoch.from_datetime(2024, 1, 2, 0, 0, 0, tsys="UTC")

# Compute access windows with minimum elevation constraint
constraint = bh.ElevationConstraint(5.0)  # 5 degrees
accesses = bh.location_accesses(
    stations,
    propagator,
    start,
    end,
    constraint
)

# Analyze coverage
for access in accesses:
    station_name = access.location.get_name()
    duration = (access.end - access.start) / 60.0
    print(f"{station_name}: {duration:.1f} minutes")
```

### Network Coverage Analysis

Evaluate geographic distribution and coverage:

```python
import brahe as bh

# Load all providers
all_stations = bh.datasets.groundstations.load_all()

# Analyze by latitude band
arctic = [s for s in all_stations if s.lat() > 66.5]
temperate = [s for s in all_stations if -66.5 <= s.lat() <= 66.5]
antarctic = [s for s in all_stations if s.lat() < -66.5]

print(f"Arctic stations: {len(arctic)}")
print(f"Temperate stations: {len(temperate)}")
print(f"Antarctic stations: {len(antarctic)}")

# Find stations by capability
x_band_stations = [
    s for s in all_stations
    if "X" in s.properties["frequency_bands"]
]
print(f"X-band capable: {len(x_band_stations)}")
```

### Mission Planning

Select appropriate ground network for mission requirements:

```python
import brahe as bh

# Requirements
required_bands = ["S", "X"]
min_elevation = 5.0
preferred_regions = ["arctic", "temperate"]

# Evaluate providers
providers = bh.datasets.groundstations.list_providers()

for provider in providers:
    stations = bh.datasets.groundstations.load(provider)

    # Filter by capability
    capable = [
        s for s in stations
        if all(band in s.properties["frequency_bands"] for band in required_bands)
    ]

    # Check geographic distribution
    arctic_count = len([s for s in capable if s.lat() > 60])

    print(f"\n{provider.upper()}")
    print(f"  Capable stations: {len(capable)}")
    print(f"  Arctic coverage: {arctic_count}")
```

## Data Quality and Limitations

### Accuracy

**Geographic Coordinates**:
- Based on publicly available information
- Typically accurate to ~100-1000 meters
- Sufficient for access analysis and mission planning
- Not suitable for precision pointing or RF link budgets

**Frequency Bands**:
- Reflects general provider capabilities
- May not represent all antennas at a site
- Subject to provider updates and changes
- Verify specific capabilities with provider for operational use

### What's Included

**Included Data**:
- Geographic location (latitude, longitude, altitude)
- Provider identification
- Supported frequency bands
- Station names (where available)

**Not Included**:
- Antenna parameters (gain, beamwidth, etc.)
- Operational schedules or availability
- Pricing or commercial terms
- Real-time status or health
- Contact information

### Data Currency

Groundstation data is updated periodically as part of Brahe releases:

- **Sources**: Provider websites, press releases, public documentation
- **Update cycle**: As providers announce new stations or changes
- **Verification**: Manual review of provider information
- **No guarantee**: Networks change; verify operational details with providers

## Best Practices

### When to Use Embedded Data

**Appropriate Uses**:
- Access analysis and link budget studies
- Mission planning and concept development
- Geographic coverage visualization
- Network comparison studies
- Educational purposes

**Verify Before**:
- Contract negotiations
- Operational mission planning
- RF link budget calculations
- Regulatory filings
- Critical mission operations

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

### Custom Groundstation Data

For proprietary or specialized groundstations, create custom data:

```python
import brahe as bh

# Create custom groundstation
custom_station = bh.PointLocation(
    lon=-122.4,  # degrees
    lat=37.8,    # degrees
    alt=100.0    # meters
).add_property("provider", "Custom") \
 .add_property("frequency_bands", ["S", "X", "Ka"])

# Combine with commercial network
ksat_stations = bh.datasets.groundstations.load("ksat")
all_stations = [custom_station] + ksat_stations
```

Or load from GeoJSON file:

```python
import brahe as bh

# Load custom network from file
custom_network = bh.datasets.groundstations.load_from_file("my_stations.geojson")
```

## API Access

### Loading Data

```python
import brahe as bh

# Single provider
ksat = bh.datasets.groundstations.load("ksat")

# All providers
all_stations = bh.datasets.groundstations.load_all()

# List available providers
providers = bh.datasets.groundstations.list_providers()
```

See the [Groundstation Functions Reference](../../library_api/datasets/groundstations.md) for complete API documentation.

## See Also

- [Datasets Overview](index.md) - Understanding datasets in Brahe
- [Groundstation API Reference](../../library_api/datasets/groundstations.md) - Complete function documentation
