# Orbit Enumerations

Enumerations for specifying orbit representation types and reference frames.

## OrbitRepresentation

::: brahe.OrbitRepresentation
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

Specifies the type of orbital elements being used.

**Module**: `brahe.orbits`

### Values

- **MEAN_ELEMENTS** - Mean orbital elements with mean anomaly
- **OSCULATING_ELEMENTS** - Osculating orbital elements with true anomaly

### Usage

```python
import brahe as bh
import numpy as np

# Mean elements (with mean anomaly)
mean_elements = np.array([7000000.0, 0.001, 0.9, 0.5, 0.3, 0.2])
prop_mean = bh.KeplerianPropagator(
    epoch=epoch,
    elements=mean_elements,
    element_type=bh.OrbitRepresentation.MEAN_ELEMENTS
)

# Osculating elements (with true anomaly)
osc_elements = np.array([7000000.0, 0.001, 0.9, 0.5, 0.3, 0.15])
prop_osc = bh.KeplerianPropagator(
    epoch=epoch,
    elements=osc_elements,
    element_type=bh.OrbitRepresentation.OSCULATING_ELEMENTS
)
```

---

## OrbitFrame

::: brahe.OrbitFrame
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

Specifies the reference frame for orbit state vectors.

**Module**: `brahe.orbits`

### Values

- **ECI** - Earth-Centered Inertial frame
- **ECEF** - Earth-Centered Earth-Fixed frame

### Usage

```python
import brahe as bh

# Create propagator with ECI frame output
prop_eci = bh.KeplerianPropagator(
    epoch=epoch,
    elements=elements,
    frame=bh.OrbitFrame.ECI
)

# Create propagator with ECEF frame output
prop_ecef = bh.KeplerianPropagator(
    epoch=epoch,
    elements=elements,
    frame=bh.OrbitFrame.ECEF
)
```

---

## InterpolationMethod

::: brahe.InterpolationMethod
    options:
      show_root_heading: true
      show_root_full_path: false
      heading_level: 3

Specifies the interpolation method for trajectory state queries.

**Module**: `brahe.trajectories`

### Values

- **LINEAR** - Linear interpolation between states
- **LAGRANGE** - Lagrange polynomial interpolation

### Usage

```python
import brahe as bh

# Create trajectory with linear interpolation
traj_linear = bh.DTrajectory(
    dimension=6,
    interpolation_method=bh.InterpolationMethod.LINEAR
)

# Create trajectory with Lagrange interpolation
traj_lagrange = bh.STrajectory6(
    interpolation_method=bh.InterpolationMethod.LAGRANGE
)
```

## See Also

- [KeplerianPropagator](keplerian_propagator.md)
- [Trajectories](../trajectories/index.md)
