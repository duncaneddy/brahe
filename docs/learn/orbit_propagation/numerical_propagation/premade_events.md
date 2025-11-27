# Premade Events

Brahe provides built-in event detectors for common orbital conditions. These premade events handle the underlying value function implementation, making it easy to detect frequently-needed conditions without writing custom detection logic.

## Available Events

The following table summarizes all premade event detectors:

| Category | Event | Description | Parameters |
|----------|-------|-------------|------------|
| **State-Derived** | `AltitudeEvent` | Geodetic altitude threshold | altitude (m), name, direction |
| | `SpeedEvent` | Velocity magnitude threshold | speed (m/s), name, direction |
| | `LongitudeEvent` | Geodetic longitude threshold | longitude (rad), name, direction |
| | `LatitudeEvent` | Geodetic latitude threshold | latitude (rad), name, direction |
| **Orbital Elements** | `SemiMajorAxisEvent` | Semi-major axis threshold | sma (m), name, direction |
| | `EccentricityEvent` | Eccentricity threshold | eccentricity, name, direction |
| | `InclinationEvent` | Inclination threshold | inclination (rad), name, direction |
| | `ArgumentOfPerigeeEvent` | Argument of perigee threshold | aop (rad), name, direction |
| | `MeanAnomalyEvent` | Mean anomaly threshold | M (rad), name, direction |
| | `EccentricAnomalyEvent` | Eccentric anomaly threshold | E (rad), name, direction |
| | `TrueAnomalyEvent` | True anomaly threshold | $\nu$ (rad), name, direction |
| | `ArgumentOfLatitudeEvent` | Argument of latitude threshold | u (rad), name, direction |
| **Node Crossings** | `AscendingNodeEvent` | Ascending node crossing (u = 0) | name |
| | `DescendingNodeEvent` | Descending node crossing (u = $\pi$) | name |
| **Eclipse/Shadow** | `UmbraEvent` | Full shadow (umbra) entry/exit | name, edge_type, ephemeris_source |
| | `PenumbraEvent` | Partial shadow (penumbra) entry/exit | name, edge_type, ephemeris_source |
| | `EclipseEvent` | Any shadow entry/exit | name, edge_type, ephemeris_source |
| | `SunlitEvent` | Full sunlight entry/exit | name, edge_type, ephemeris_source |

### Parameter Types

**Value Events** (threshold crossing):

- `direction`: `EventDirection.INCREASING`, `EventDirection.DECREASING`, or `EventDirection.ANY`

**Binary Events** (boolean condition transitions):

- `edge_type`: `EdgeType.RISING_EDGE`, `EdgeType.FALLING_EDGE`, or `EdgeType.ANY_EDGE`
- `ephemeris_source`: `None` (low precision), `EphemerisSource.DE440s`, or `EphemerisSource.DE440`

## Altitude Event

The `AltitudeEvent` detects when a spacecraft crosses a specified geodetic altitude threshold. This is one of the most commonly needed event types in orbital mechanics.

### Configuration

`AltitudeEvent` accepts three parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `altitude` | float | Threshold altitude in meters (above WGS84 ellipsoid) |
| `name` | str | Identifier for the event in the event log |
| `direction` | EventDirection | Which crossings to detect |

The `EventDirection` options are:

- `INCREASING` - Detect only when ascending through the altitude
- `DECREASING` - Detect only when descending through the altitude
- `ANY` - Detect crossings in both directions

### Example

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/event_altitude.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/event_altitude.rs:4"
    ```

### Applications

| Use Case | Configuration |
|----------|---------------|
| Atmospheric interface detection | `AltitudeEvent(120e3, "Karman line", DECREASING)` |
| Orbit raising trigger | `AltitudeEvent(target_alt, "Target", INCREASING)` |
| Perigee passage | `AltitudeEvent(perigee_alt, "Perigee", ANY)` |
| Re-entry monitoring | `AltitudeEvent(100e3, "Re-entry", DECREASING)` |
| Payload deployment altitude | `AltitudeEvent(deploy_alt, "Deploy", INCREASING)` |

### Implementation Details

The altitude event internally computes geodetic altitude using the WGS84 ellipsoid model. The event value function is:

$$g(\mathbf{x}) = h_{\text{geodetic}}(\mathbf{r}) - h_{\text{threshold}}$$

where $h_{\text{geodetic}}$ is the geodetic altitude computed from the position vector $\mathbf{r}$.

## Orbital Element Events

These events detect when orbital elements cross specified threshold values. All orbital element events use the same constructor pattern:

```python
event = ElementEvent(threshold, name, direction)
```

| Event | Element | Units | Typical Use Cases |
|-------|---------|-------|-------------------|
| `SemiMajorAxisEvent` | $a$ | meters | Orbit decay monitoring, altitude maintenance |
| `EccentricityEvent` | $e$ | dimensionless | Circularization detection, orbit stability |
| `InclinationEvent` | $i$ | radians | Plane change monitoring, SSO maintenance |
| `ArgumentOfPerigeeEvent` | $\omega$ | radians | Apsidal rotation, frozen orbit maintenance |
| `MeanAnomalyEvent` | $M$ | radians | Orbit phasing, synchronization |
| `EccentricAnomalyEvent` | $E$ | radians | Precise anomaly detection |
| `TrueAnomalyEvent` | $\nu$ | radians | Apoapsis/periapsis detection ($\nu = 0$ or $\pi$) |
| `ArgumentOfLatitudeEvent` | $u = \omega + \nu$ | radians | Latitude-based triggers |

## Node Crossing Events

Node crossing events detect when a spacecraft passes through the equatorial plane:

```python
# Ascending node: spacecraft crosses equator heading north (u = 0)
asc_event = bh.AscendingNodeEvent("Ascending Node")

# Descending node: spacecraft crosses equator heading south (u = π)
desc_event = bh.DescendingNodeEvent("Descending Node")
```

These events are useful for:

- Ground track analysis
- Orbit determination campaigns
- RAAN drift monitoring
- Conjunction screening at nodes

## Eclipse Events

Eclipse events detect shadow conditions using the conical shadow model. They accept an optional `EphemerisSource` parameter for sun position computation:

```python
# Using low-precision sun position (default)
eclipse = bh.EclipseEvent("Eclipse", bh.EdgeType.ANY_EDGE, None)

# Using high-precision DE440s ephemeris
eclipse = bh.EclipseEvent("Eclipse", bh.EdgeType.RISING_EDGE, bh.EphemerisSource.DE440s)
```

| Event | Condition | `RISING_EDGE` | `FALLING_EDGE` |
|-------|-----------|---------------|----------------|
| `UmbraEvent` | Full shadow (illumination = 0) | Enter umbra | Exit umbra |
| `PenumbraEvent` | Partial shadow (0 < illumination < 1) | Enter penumbra | Exit penumbra |
| `EclipseEvent` | Any shadow (illumination < 1) | Enter eclipse | Exit eclipse |
| `SunlitEvent` | Full sunlight (illumination = 1) | Exit eclipse | Enter eclipse |

### Ephemeris Sources

| Source | Description | Accuracy |
|--------|-------------|----------|
| `None` / `LowPrecision` | Analytical approximation | ~0.01° |
| `DE440s` | JPL DE440s ephemeris (short-term) | ~1 arcsec |
| `DE440` | JPL DE440 ephemeris (long-term) | ~1 arcsec |

---

## See Also

- [Event Detection](event_detection.md) - Core event detection concepts
- [Event Callbacks](event_callbacks.md) - Responding to detected events
- [Maneuvers](maneuvers.md) - Using events for orbit maneuvers
