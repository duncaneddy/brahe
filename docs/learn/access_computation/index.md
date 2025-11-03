# Access Computation

Access computation determines when and under what conditions satellites can view, observe, or "access" ground locations. This is fundamental for mission planning, ground station contact scheduling, imaging opportunity planning, and other mission operations planning tasks.

An **access** occurs when a satellite has a clear geometric line-of-sight to a ground location and meets all constraints (e.g., minimum elevation angle, local time of day, look direction). Brahe's access computation system identifies time windows where all constraints are met and computes relevant properties (e.g., azimuth, elevation, off-nadir angle) for each access window. The system is designed to be flexible, allowing users to define custom locations, constraints, and properties as needed. Access computation is also parallelized by default to efficiently handle large numbers of locations and satellites.

## System Architecture

Brahe's access computation system is built around four major components:

### 1. Locations

**Locations** define *where* to check for access. Brahe supports two primary location types:

- **`PointLocation`** - Single geodetic point (e.g., ground station, city)
- **`PolygonLocation`** - Closed polygon area (e.g., imaging region, coverage zone)

Both implement the `AccessibleLocation` trait, which provides coordinate access, property management, and GeoJSON import/export capabilities. Locations can be created from coordinates or loaded from GeoJSON files, and the type supports custom properties for metadata storage.

[Learn more about Locations →](locations.md)

### 2. Constraints

**Constraints** define *what conditions* must be satisfied for an access to occur. Brahe provides several built-in constraint types including:

- **`ElevationConstraint`** - Minimum/maximum elevation above horizon
- **`ElevationMaskConstraint`** - Azimuth-dependent elevation masks (terrain profiles)
- **`OffNadirConstraint`** - Minimum/maximum off-nadir angle (imaging satellites)
- **`LocalTimeConstraint`** - Local solar time windows (e.g., daylight imaging)
- **`LookDirectionConstraint`** - Left/right/either relative to velocity vector
- **`AscDscConstraint`** - Ascending/descending pass filter

Constraints can be combined using the `ConstraintComposite` system to express sophisticated requirements like "elevation > 10° AND (daylight OR look-right)". Python users can create custom constraints by implementing the `AccessConstraintComputer` interface.

[Learn more about Constraints →](constraints.md)

### 3. Properties

**Properties** define *what information* to compute during each access window. Brahe automatically computes six core geometric properties:

- `azimuth_open`, `azimuth_close` - Azimuth angles at window start/end
- `elevation_min`, `elevation_max` - Minimum/maximum elevation during access
- `off_nadir_min`, `off_nadir_max` - Minimum/maximum off-nadir angle
- `local_time` - Local solar time at access midpoint
- `look_direction` - Satellite look direction (Left/Right)
- `asc_dsc` - Pass type (Ascending/Descending)

Properties are stored as `PropertyValue` enums supporting scalar, vector, time-series, boolean, string, and JSON data types. Users can add custom properties or implement `AccessPropertyComputer` for automated property calculation during access searches.

[Learn more about Properties →](properties.md)

### 4. Computation

**Computation** is the algorithm that ties everything together. The primary function `location_accesses()` performs a two-phase search:

1. **Coarse search** - Evaluate access at regular time steps to identify candidate windows
2. **Refinement** - Use binary search to precisely locate window boundaries

The `AccessSearchConfig` struct controls algorithm behavior (initial time step, adaptive stepping, etc.) for optimal performance across different scenarios. Results are returned as `AccessWindow` objects containing start/end times, identifiers, and computed properties.

[Learn more about Computation →](computation.md)

## Module Catalog

This section provides a complete reference of all types, traits, and functions in the access computation module.

### Location Types

**`PointLocation`** - Single geodetic point location

- Create from coordinates: `new(lat, lon, alt)`
- Load from GeoJSON: `from_geojson(geojson_str)`
- Access coordinates: `lat()`, `lon()`, `alt()`, `longitude()`, `latitude()`, `altitude()`
- Manage properties: `add_property(name, value)`
- Export: `to_geojson()`

**`PolygonLocation`** - Closed polygon area location

- Create from vertices: `new(vertices)`
- Load from GeoJSON: `from_geojson(geojson_str)`
- Access geometry: `vertices()`, `num_vertices()`, center via `lat()`, `lon()`, `alt()`
- Manage properties: `add_property(name, value)`
- Export: `to_geojson()`

**`AccessibleLocation` trait** - Common interface for all location types

- Get center coordinates: `center_geodetic()`, `center_ecef()`
- Access properties: `properties()`, `properties_mut()`
- Export: `to_geojson()`

### Constraint Types

**Built-in Constraints:**

- **`ElevationConstraint`** - Enforce minimum/maximum elevation angles above horizon
- **`ElevationMaskConstraint`** - Apply azimuth-dependent elevation masks for terrain modeling
- **`OffNadirConstraint`** - Limit off-nadir viewing angles for imaging payloads
- **`LocalTimeConstraint`** - Filter by local solar time windows (e.g., daylight-only imaging)
- **`LookDirectionConstraint`** - Require left/right/either look direction relative to velocity
- **`AscDscConstraint`** - Filter by ascending/descending pass type
- **`ConstraintComposite`** - Combine constraints with Boolean logic (All/Any/Not)

**Constraint Traits:**

- **`AccessConstraint` trait** - Interface for evaluating constraints at specific times
  - `evaluate(epoch, location, propagator) -> bool` - Check if constraint satisfied
  - `name()` - Get constraint name for debugging
  - `format_string()` - Get human-readable constraint description

- **`AccessConstraintComputer` trait** - Python interface for custom user-defined constraints
  - `evaluate(epoch, location, propagator) -> bool` - Custom constraint logic
  - `name()` - Constraint identifier

### Property Types

**`PropertyValue` enum** - Strongly-typed property values

- `Scalar(f64)` - Single floating-point value
- `Vector(Vec<f64>)` - Array of values
- `TimeSeries(Vec<(Epoch, f64)>)` - Time-indexed measurements
- `Boolean(bool)` - True/false flag
- `String(String)` - Text data
- `Json(String)` - Arbitrary JSON data

**`AccessProperties` struct** - Container for access window properties

- Core properties: `azimuth_open`, `azimuth_close`, `elevation_min`, `elevation_max`, `off_nadir_min`, `off_nadir_max`, `local_time`, `look_direction`, `asc_dsc`
- Custom properties stored in HashMap
- Methods: `new()`, `add_property(name, value)`, `get_property(name)`

**`AccessPropertyComputer` trait** - Python interface for custom property calculation

- `compute(window, location, propagator) -> HashMap<String, PropertyValue>` - Calculate properties
- `property_names() -> Vec<String>` - List computed property names

### Window and Configuration Types

**`AccessWindow` struct** - Represents a single access opportunity

- Time bounds: `window_open`, `window_close`
- Identifiers: `location_name/id/uuid`, `satellite_name/id/uuid`, `name/id/uuid`
- Properties: `properties` (AccessProperties)
- Methods: `new()`, `start()`, `end()`, `midtime()`, `duration()`

**`AccessSearchConfig` struct** - Controls access computation algorithm

- `initial_time_step` - Coarse search step size (seconds)
- `adaptive_step` - Enable adaptive refinement
- `adaptive_fraction` - Refinement step fraction (0.0-1.0)
- `parallel` - Enable parallel location/satellite processing
- `num_threads` - Thread pool size (0 = auto)

### Enumerations

**`LookDirection`** - Satellite look direction

- `Left` - Looking left relative to velocity vector
- `Right` - Looking right relative to velocity vector
- `Either` - Either direction acceptable

**`AscDsc`** - Pass type classification

- `Ascending` - Satellite moving from south to north
- `Descending` - Satellite moving from north to south
- `Either` - Either direction acceptable

## See Also

- [Locations](locations.md) - Ground location types and GeoJSON support
- [Constraints](constraints.md) - Built-in and custom constraint types
- [Computation](computation.md) - Access algorithms and property computation
- [Example: Predicting Ground Contacts](../../examples/ground_contacts.md) - Complete ground station example
- [Example: Computing Imaging Opportunities](../../examples/imaging_opportunities.md) - Imaging scenario
- [API Reference: Access Module](../../library_api/access/index.md) - Complete API documentation
