# Premade Events

Brahe provides built-in event detectors for common orbital conditions. These premade events handle the underlying value function implementation, making it easy to detect frequently-needed conditions without writing custom detection logic.

## Event Categories

Premade events fall into four categories based on what they detect:

<div class="center-table" markdown="1">
| Category | What They Detect | Event Type |
|----------|------------------|------------|
| **Orbital Elements** | Threshold crossings of Keplerian elements | Value |
| **State-Derived** | Altitude, speed, geodetic position | Value |
| **Eclipse/Shadow** | Shadow transitions (umbra, penumbra, sunlit) | Binary |
| **Node Crossings** | Equatorial plane crossings | Value |
</div>

The distinction between **value events** and **binary events** is important:

- **Value events** detect when a continuously-varying quantity crosses a threshold (e.g., altitude = 400 km)
- **Binary events** detect when a boolean condition changes state (e.g., enters shadow)

## Orbital Element Events

Orbital element events detect when Keplerian elements cross threshold values. These are value events with configurable thresholds and directions.

### Available Events

<div class="center-table" markdown="1">
| Event | Element | Units |
|-------|---------|-------|
| `SemiMajorAxisEvent` | $a$ | meters |
| `EccentricityEvent` | $e$ | dimensionless |
| `InclinationEvent` | $i$ | degrees or radians |
| `ArgumentOfPerigeeEvent` | $\omega$ | degrees or radians |
| `MeanAnomalyEvent` | $M$ | degrees or radians |
| `EccentricAnomalyEvent` | $E$ | degrees or radians |
| `TrueAnomalyEvent` | $\nu$ | degrees or radians |
| `ArgumentOfLatitudeEvent` | $u = \omega + \nu$ | degrees or radians |
</div>

### Configuration

Orbital element events take up to four parameters:

- `threshold` - Target value to detect
- `name` - Identifier for the event in the event log
- `direction` - Which crossings to detect (`INCREASING`, `DECREASING`, or `ANY`)
- `angle_format` - For angle-based events: `AngleFormat.DEGREES` or `AngleFormat.RADIANS`

Non-angle events (`SemiMajorAxisEvent`, `EccentricityEvent`) omit the `angle_format` parameter.

### Example

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/event_orbital_elements.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/event_orbital_elements.rs:4"
    ```

### Applications

| Event | Use Cases |
|-------|-----------|
| `TrueAnomalyEvent` | Apoapsis detection ($\nu = 180°$), periapsis detection ($\nu = 0°$) |
| `SemiMajorAxisEvent` | Orbit decay monitoring, altitude maintenance |
| `EccentricityEvent` | Circularization detection, orbit stability |
| `InclinationEvent` | Plane change monitoring, SSO maintenance |

## State-Derived Events

State-derived events compute quantities from the instantaneous state vector rather than orbital elements.

### Available Events

<div class="center-table" markdown="1">
| Event | Quantity | Units |
|-------|----------|-------|
| `AltitudeEvent` | Geodetic altitude (WGS84) | meters |
| `SpeedEvent` | Velocity magnitude | m/s |
| `LongitudeEvent` | Geodetic longitude | degrees or radians |
| `LatitudeEvent` | Geodetic latitude | degrees or radians |
</div>

### Configuration

State-derived events follow the same pattern as orbital element events:

- `threshold` - Target value to detect
- `name` - Identifier for the event in the event log
- `direction` - Which crossings to detect (`INCREASING`, `DECREASING`, or `ANY`)
- `angle_format` - For geodetic events: `AngleFormat.DEGREES` or `AngleFormat.RADIANS`

### Example: Altitude Event

The `AltitudeEvent` is one of the most commonly used premade events. It detects when a spacecraft crosses a specified geodetic altitude.

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

## Eclipse/Shadow Events

Eclipse events detect shadow conditions using the conical shadow model. These are binary events that trigger on state transitions.

### Available Events

<div class="center-table" markdown="1">
| Event | Condition | `RISING_EDGE` | `FALLING_EDGE` |
|-------|-----------|---------------|----------------|
| `EclipseEvent` | Any shadow (illumination < 1) | Enter eclipse | Exit eclipse |
| `UmbraEvent` | Full shadow (illumination = 0) | Enter umbra | Exit umbra |
| `PenumbraEvent` | Partial shadow (0 < illumination < 1) | Enter penumbra | Exit penumbra |
| `SunlitEvent` | Full sunlight (illumination = 1) | Exit eclipse | Enter eclipse |
</div>

### Configuration

Eclipse events take three parameters:

- `name` - Identifier for the event in the event log
- `edge_type` - Which transition to detect (`RISING_EDGE`, `FALLING_EDGE`, or `ANY_EDGE`)
- `ephemeris_source` - Sun position source (`None` for analytical, or `EphemerisSource.DE440s`/`DE440`)

### Example

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/event_eclipse.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/event_eclipse.rs:4"
    ```

### Ephemeris Sources

<div class="center-table" markdown="1">
| Source | Description |
|--------|-------------|
| `None` / `LowPrecision` | Analytical approximation (fastest) |
| `DE440s` | JPL DE440s ephemeris (short-term, high precision) |
| `DE440` | JPL DE440 ephemeris (long-term, high precision) |
</div>

## Node Crossing Events

Node crossing events detect when a spacecraft passes through the equatorial plane. These are specialized value events with fixed thresholds.

### Available Events

<div class="center-table" markdown="1">
| Event | Trigger Condition | Direction |
|-------|-------------------|-----------|
| `AscendingNodeEvent` | Argument of latitude = 0 (northward crossing) | Increasing |
| `DescendingNodeEvent` | Argument of latitude = $\pi$ (southward crossing) | Increasing |
</div>

### Configuration

Node events take only a name parameter:

```python
asc_event = bh.AscendingNodeEvent("Ascending Node")
desc_event = bh.DescendingNodeEvent("Descending Node")
```

### Example

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/event_node_crossing.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/event_node_crossing.rs:4"
    ```

### Applications

- Ground track analysis
- Orbit determination campaigns
- RAAN drift monitoring
- Conjunction screening at nodes

## Quick Reference

### All Premade Events

<div class="center-table" markdown="1">
| Category | Event | Parameters |
|----------|-------|------------|
| **Eclipse/Shadow** | `EclipseEvent` | name, edge_type, ephemeris_source |
| | `UmbraEvent` | name, edge_type, ephemeris_source |
| | `PenumbraEvent` | name, edge_type, ephemeris_source |
| | `SunlitEvent` | name, edge_type, ephemeris_source |
| **Node Crossings** | `AscendingNodeEvent` | name |
| | `DescendingNodeEvent` | name |
| **Orbital Elements** | `SemiMajorAxisEvent` | threshold (m), name, direction |
| | `EccentricityEvent` | threshold, name, direction |
| | `InclinationEvent` | threshold, name, direction, angle_format |
| | `ArgumentOfPerigeeEvent` | threshold, name, direction, angle_format |
| | `MeanAnomalyEvent` | threshold, name, direction, angle_format |
| | `EccentricAnomalyEvent` | threshold, name, direction, angle_format |
| | `TrueAnomalyEvent` | threshold, name, direction, angle_format |
| | `ArgumentOfLatitudeEvent` | threshold, name, direction, angle_format |
| **State-Derived** | `AltitudeEvent` | threshold (m), name, direction |
| | `SpeedEvent` | threshold (m/s), name, direction |
| | `LongitudeEvent` | threshold, name, direction, angle_format |
| | `LatitudeEvent` | threshold, name, direction, angle_format |
</div>

### Parameter Types

**Value Events** (threshold crossing):

- `direction`: `EventDirection.INCREASING`, `EventDirection.DECREASING`, or `EventDirection.ANY`
- `angle_format`: `AngleFormat.DEGREES` or `AngleFormat.RADIANS` (angle-based events only)

**Binary Events** (boolean condition transitions):

- `edge_type`: `EdgeType.RISING_EDGE`, `EdgeType.FALLING_EDGE`, or `EdgeType.ANY_EDGE`
- `ephemeris_source`: `None` (low precision), `EphemerisSource.DE440s`, or `EphemerisSource.DE440`

---

## See Also

- [Event Detection](event_detection.md) - Core event detection concepts
- [Event Callbacks](event_callbacks.md) - Responding to detected events
- [Maneuvers](maneuvers.md) - Using events for orbit maneuvers
