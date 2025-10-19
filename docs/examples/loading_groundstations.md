# Loading Groundstation Datasets

This guide shows how to load and work with groundstation location datasets in Brahe.

## Overview

Brahe provides curated groundstation datasets from multiple commercial providers. The data is embedded in the library, so no internet connection or external files are required.

Available providers:

- **atlas**: Atlas Space Operations
- **aws**: Amazon Web Services Ground Station
- **ksat**: Kongsberg Satellite Services (KSAT)
- **leaf**: Leaf Space
- **ssc**: Swedish Space Corporation
- **viasat**: Viasat

## Loading Groundstations by Provider

Load groundstations from a specific provider:

```python
import brahe as bh

# Load KSAT groundstations
ksat_stations = bh.datasets.groundstations.load("ksat")

print(f"Loaded {len(ksat_stations)} KSAT stations")

# Display first few stations
for station in ksat_stations[:5]:
    print(f"{station.get_name()}: ({station.lon():.2f}°, {station.lat():.2f}°)")
```

Provider names are case-insensitive:

```python
# These all work the same
stations1 = bh.datasets.groundstations.load("KSAT")
stations2 = bh.datasets.groundstations.load("ksat")
stations3 = bh.datasets.groundstations.load("KsAt")
```

## Loading All Groundstations

Load groundstations from all providers at once:

```python
import brahe as bh

# Load all groundstations
all_stations = bh.datasets.groundstations.load_all()
print(f"Total stations: {len(all_stations)}")

# Group by provider
by_provider = {}
for station in all_stations:
    props = station.properties
    provider = props.get("provider", "Unknown")

    if provider not in by_provider:
        by_provider[provider] = []
    by_provider[provider].append(station)

# Display counts
for provider, stations in sorted(by_provider.items()):
    print(f"{provider}: {len(stations)} stations")
```

## Listing Available Providers

Get a list of all available providers:

```python
import brahe as bh

providers = bh.datasets.groundstations.list_providers()
print(f"Available providers: {', '.join(providers)}")

# Load stations for each provider
for provider in providers:
    stations = bh.datasets.groundstations.load(provider)
    print(f"{provider}: {len(stations)} stations")
```

## Working with Groundstation Properties

Each groundstation has metadata stored in properties:

```python
import brahe as bh

stations = bh.datasets.groundstations.load("atlas")
station = stations[0]

# Access basic info
print(f"Name: {station.get_name()}")
print(f"Location: {station.lon():.3f}°, {station.lat():.3f}°, {station.alt():.0f} m")

# Access properties
props = station.properties

# Provider name
print(f"Provider: {props['provider']}")

# Supported frequency bands
bands = props['frequency_bands']
print(f"Frequency bands: {', '.join(bands)}")
```

All groundstations have these properties:

- `provider`: Provider name (e.g., "KSAT", "Atlas")
- `frequency_bands`: List of supported frequency bands (e.g., `["S", "X", "Ka"]`)

## Accessing Coordinates

Groundstations provide convenient coordinate accessors:

```python
import brahe as bh

station = bh.datasets.groundstations.load("ksat")[0]

# Quick accessors (always in degrees)
lon = station.lon()    # Longitude in degrees
lat = station.lat()    # Latitude in degrees
alt = station.alt()    # Altitude in meters

print(f"Position: {lon:.4f}°E, {lat:.4f}°N, {alt:.1f}m")
```

## Computing Satellite Access

Use groundstations with the access module to compute visibility windows:

```python
import brahe as bh
import numpy as np

# Load groundstations
stations = bh.datasets.groundstations.load("ksat")

# Create satellite propagator from TLE
tle_line1 = "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997"
tle_line2 = "2 25544  51.6461 306.0234 0003417  88.1267  25.5695 15.48919103000003"
propagator = bh.SGPPropagator.from_tle(tle_line1, tle_line2)

# Define time range
start_epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0, tsys="UTC")
end_epoch = bh.Epoch.from_datetime(2024, 1, 2, 0, 0, 0, tsys="UTC")

# Compute access windows with elevation constraint
constraint = bh.ElevationConstraint(5.0)  # 5 degree minimum elevation

accesses = bh.location_accesses(
    stations,      # All KSAT stations
    propagator,
    start_epoch,
    end_epoch,
    constraint,
    step_size=60.0  # 60 second time step
)

# Print access windows
for access in accesses:
    station_name = access.location.name
    duration = (access.end - access.start) / 60.0  # Convert to minutes
    print(f"{station_name}: {access.start} ({duration:.1f} min)")
```

## Filtering Stations by Frequency Band

Find stations that support specific frequency bands:

```python
import brahe as bh

# Load all stations
all_stations = bh.datasets.groundstations.load_all()

# Filter for X-band capable stations
x_band_stations = []
for station in all_stations:
    props = station.properties
    if "X" in props["frequency_bands"]:
        x_band_stations.append(station)

print(f"Found {len(x_band_stations)} X-band capable stations")

# Display by provider
by_provider = {}
for station in x_band_stations:
    provider = station.properties["provider"]
    if provider not in by_provider:
        by_provider[provider] = 0
    by_provider[provider] += 1

for provider, count in sorted(by_provider.items()):
    print(f"  {provider}: {count} stations")
```

## Filtering by Geographic Region

Find stations in a specific geographic region:

```python
import brahe as bh

# Load all stations
all_stations = bh.datasets.groundstations.load_all()

# Filter for Arctic stations (latitude > 60°N)
arctic_stations = [
    s for s in all_stations
    if s.lat() > 60.0
]

print(f"Found {len(arctic_stations)} Arctic stations:")
for station in arctic_stations:
    print(f"  {station.get_name()}: {station.lat():.2f}°N")
```

## Loading from Custom GeoJSON Files

You can also load groundstations from custom GeoJSON files:

```python
import brahe as bh

# Load from custom file
custom_stations = bh.datasets.groundstations.load_from_file("my_stations.geojson")

print(f"Loaded {len(custom_stations)} custom stations")
```

The GeoJSON file must be a FeatureCollection with Point geometries:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Point",
        "coordinates": [15.4, 78.2, 0.0]
      },
      "properties": {
        "name": "My Ground Station",
        "provider": "Custom",
        "frequency_bands": ["S", "X"]
      }
    }
  ]
}
```

## Using the CLI

Brahe provides command-line tools for working with groundstations:

### List Available Providers

```bash
brahe datasets groundstations list
```

### Show Stations

```bash
# Show KSAT stations
brahe datasets groundstations show ksat

# Show with properties
brahe datasets groundstations show atlas --properties
```

### Show All Stations

```bash
brahe datasets groundstations show-all

# With properties
brahe datasets groundstations show-all --properties
```

## Common Patterns

### Find Nearest Station

Find the groundstation nearest to a given point:

```python
import brahe as bh
import numpy as np

# Target location
target_lon, target_lat = 10.0, 60.0  # degrees

# Load all stations
all_stations = bh.datasets.groundstations.load_all()

# Find nearest (simple great circle distance)
min_distance = float('inf')
nearest_station = None

for station in all_stations:
    # Simple distance approximation
    dlon = station.lon() - target_lon
    dlat = station.lat() - target_lat
    distance = np.sqrt(dlon**2 + dlat**2)

    if distance < min_distance:
        min_distance = distance
        nearest_station = station

print(f"Nearest station: {nearest_station.get_name()}")
print(f"Location: {nearest_station.lon():.2f}°, {nearest_station.lat():.2f}°")
print(f"Provider: {nearest_station.properties['provider']}")
```

### Create Coverage Map

Create a simple coverage map showing station distribution:

```python
import brahe as bh
import matplotlib.pyplot as plt

# Load all stations
all_stations = bh.datasets.groundstations.load_all()

# Extract coordinates
lons = [s.lon() for s in all_stations]
lats = [s.lat() for s in all_stations]

# Get colors by provider
providers = [s.properties["provider"] for s in all_stations]
unique_providers = sorted(set(providers))
colors = [unique_providers.index(p) for p in providers]

# Plot
plt.figure(figsize=(12, 6))
plt.scatter(lons, lats, c=colors, cmap='tab10', alpha=0.6, s=50)
plt.xlabel('Longitude (degrees)')
plt.ylabel('Latitude (degrees)')
plt.title(f'Global Groundstation Coverage ({len(all_stations)} stations)')
plt.grid(True, alpha=0.3)
plt.colorbar(label='Provider', ticks=range(len(unique_providers)))
plt.tight_layout()
plt.show()
```

## See Also

- [Computing Ground Contacts](computing_ground_contacts.md)
- [Groundstation Functions Reference](../library_api/datasets/groundstations.md)
