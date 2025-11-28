# Covariance and Sensitivity

The `NumericalOrbitPropagator` can propagate additional quantities alongside the orbital state, enabling covariance propagation and sensitivity analysis. This is essential for uncertainty quantification, orbit determination, and mission analysis.

## State Transition Matrices (STMs)

The State Transition Matrix (STM) is the fundamental quantity for linear uncertainty propagation. It describes how small perturbations in the initial state map to perturbations at a later time:

$$\delta \mathbf{x}(t) = \Phi(t, t_0) \delta \mathbf{x}(t_0)$$

where $\Phi(t, t_0)$ is the 6×6 STM from epoch $t_0$ to time $t$.

### Enabling STM Propagation

Enable STM computation through the propagation configuration:

```python
# Method 1: Using with_stm() builder
prop_config = bh.NumericalPropagationConfig.default().with_stm()

# Method 2: Using VariationalConfig directly
var_config = bh.VariationalConfig(enable_stm=True)
prop_config = bh.NumericalPropagationConfig.new(
    bh.IntegrationMethod.DP54,
    bh.IntegratorConfig.adaptive(1e-10, 1e-12),
    var_config
)

# Create propagator with STM-enabled config
prop = bh.NumericalOrbitPropagator(
    epoch, state, prop_config, force_config, params
)
```

### STM Structure

For orbital mechanics with state $\mathbf{x} = [x, y, z, v_x, v_y, v_z]^T$, the 6×6 STM has structure:

$$\Phi = \begin{bmatrix} \frac{\partial \mathbf{r}}{\partial \mathbf{r}_0} & \frac{\partial \mathbf{r}}{\partial \mathbf{v}_0} \\ \frac{\partial \mathbf{v}}{\partial \mathbf{r}_0} & \frac{\partial \mathbf{v}}{\partial \mathbf{v}_0} \end{bmatrix}$$

Each 3×3 submatrix represents:

| Submatrix | Location | Physical Meaning |
|-----------|----------|------------------|
| $\frac{\partial \mathbf{r}}{\partial \mathbf{r}_0}$ | Upper left | Position sensitivity to initial position |
| $\frac{\partial \mathbf{r}}{\partial \mathbf{v}_0}$ | Upper right | Position sensitivity to initial velocity |
| $\frac{\partial \mathbf{v}}{\partial \mathbf{r}_0}$ | Lower left | Velocity sensitivity to initial position |
| $\frac{\partial \mathbf{v}}{\partial \mathbf{v}_0}$ | Lower right | Velocity sensitivity to initial velocity |

### Accessing the STM

After propagation, access the STM via the `stm()` method:

```python
prop.propagate_to(target_epoch)

# Get current STM (from initial epoch to current epoch)
stm = prop.stm()

# STM is a 6x6 numpy array (or nalgebra matrix in Rust)
print(f"STM shape: {stm.shape}")
```

For STM at intermediate times (requires `store_stm_history=True`):

```python
# Enable STM history storage
prop_config = bh.NumericalPropagationConfig.default().with_stm().with_stm_history()

# After propagation, query STM at any epoch in the trajectory
stm_at_t = prop.stm_at(intermediate_epoch)
```

### STM Properties

The STM has several important properties:

1. **Identity at initial time**: $\Phi(t_0, t_0) = I$

2. **Composition**: STMs can be composed to span longer intervals:
   $$\Phi(t_2, t_0) = \Phi(t_2, t_1) \Phi(t_1, t_0)$$

3. **Determinant preservation**: For Hamiltonian systems (conservative forces), $\det(\Phi) = 1$

---

## Covariance Propagation

The primary application of the STM is propagating uncertainty. Given an initial covariance $P_0$, the propagated covariance is:

$$P(t) = \Phi(t, t_0) P_0 \Phi(t, t_0)^T$$

### Creating a Propagator with Initial Covariance

You can provide an initial covariance when creating the propagator:

```python
# Define initial covariance (6x6 matrix)
# Position uncertainty: 10 m, Velocity uncertainty: 0.01 m/s
P0 = np.diag([100.0, 100.0, 100.0, 0.0001, 0.0001, 0.0001])

# Create propagator with initial covariance (automatically enables STM)
prop = bh.NumericalOrbitPropagator(
    epoch, state,
    prop_config, force_config,
    params=params,
    initial_covariance=P0
)
```

### Covariance Retrieval

After propagation, retrieve covariance in different reference frames:

```python
# Covariance in ECI frame
cov_eci = prop.covariance_eci(target_epoch)

# Covariance in ECEF frame
cov_ecef = prop.covariance_ecef(target_epoch)

# Covariance in RTN (Radial-Tangential-Normal) frame
cov_rtn = prop.covariance_rtn(target_epoch)

# Covariance in GCRF frame
cov_gcrf = prop.covariance_gcrf(target_epoch)
```

### Example

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/covariance_propagation.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/covariance_propagation.rs:4"
    ```

### Interpreting Results

The propagated covariance reveals how initial uncertainties evolve:

- **Position uncertainty** typically grows over time due to velocity uncertainty
- **Velocity uncertainty** may grow due to along-track position uncertainty coupling with gravity gradients
- **Cross-correlations** develop between position and velocity components
- The RTN frame is useful for interpreting errors relative to the orbit: radial (altitude), tangential (along-track), and normal (cross-track)

### Covariance Evolution Visualization

The following plot shows how position uncertainty evolves over three orbital periods:

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/covariance_evolution_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/covariance_evolution_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="covariance_evolution.py"
    --8<-- "./plots/learn/numerical_propagation/covariance_evolution.py:12"
    ```

---

## Sensitivity Propagation

In addition to STM propagation (which tracks sensitivity to initial state), the propagator can also compute **parameter sensitivity** - how the state changes with respect to configuration parameters.

### Parameter Sensitivity

The sensitivity matrix $S(t)$ describes how the state at time $t$ depends on the parameters:

$$S(t) = \frac{\partial \mathbf{x}(t)}{\partial \mathbf{p}}$$

where $\mathbf{p}$ is the parameter vector. This enables answering questions like:

- "How does a 1% uncertainty in drag coefficient affect position prediction?"
- "What is the impact of mass uncertainty on orbit determination?"

### Configuration Parameters

The default parameter vector contains spacecraft physical properties:

| Index | Parameter | Units | Description |
|-------|-----------|-------|-------------|
| 0 | mass | kg | Spacecraft mass |
| 1 | drag_area | m² | Cross-sectional area for drag |
| 2 | Cd | - | Drag coefficient |
| 3 | srp_area | m² | Cross-sectional area for SRP |
| 4 | Cr | - | Solar radiation pressure coefficient |

### Enabling Sensitivity Propagation

```python
# Enable sensitivity computation
prop_config = (
    bh.NumericalPropagationConfig.default()
    .with_sensitivity()
    .with_sensitivity_history()
)

# Parameters are required for sensitivity computation
params = np.array([1000.0, 10.0, 2.2, 10.0, 1.3])

prop = bh.NumericalOrbitPropagator(
    epoch, state,
    prop_config, force_config,
    params=params
)
```

### Accessing Sensitivity

After propagation, access the sensitivity matrix:

```python
prop.propagate_to(target_epoch)

# Get current sensitivity matrix (6 x num_params)
sens = prop.sensitivity()

# Query sensitivity at intermediate times (requires sensitivity_history)
sens_at_t = prop.sensitivity_at(intermediate_epoch)
```

### Example: Parameter Sensitivity Analysis

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/sensitivity_analysis.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/sensitivity_analysis.rs:4"
    ```

### Interpreting Sensitivity Results

The sensitivity matrix $S$ is 6×5 (state dimension × parameter count):

- **Column 0**: Sensitivity to mass - affects drag acceleration ($a \propto 1/m$)
- **Column 1**: Sensitivity to drag area - affects drag force
- **Column 2**: Sensitivity to Cd - drag coefficient uncertainty
- **Column 3**: Sensitivity to SRP area - solar radiation pressure
- **Column 4**: Sensitivity to Cr - SRP coefficient uncertainty

Physical insights:

- For LEO orbits, drag parameters (mass, drag_area, Cd) typically dominate
- For GEO orbits, SRP parameters (srp_area, Cr) become more important
- Two-body propagation shows zero sensitivity (no force depends on parameters)
- Sensitivity grows over time as perturbation effects accumulate

### Sensitivity Evolution Visualization

The following plot shows how position sensitivity to each parameter evolves over time for a LEO orbit:

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/sensitivity_evolution_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/sensitivity_evolution_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="sensitivity_evolution.py"
    --8<-- "./plots/learn/numerical_propagation/sensitivity_evolution.py:12"
    ```

---

## Computational Considerations

STM and sensitivity computation significantly increase computational cost:

| Configuration | State dimension | Memory per step |
|---------------|-----------------|-----------------|
| State only | 6 | ~48 bytes |
| With STM | 42 (6 + 36) | ~336 bytes |
| With Sensitivity | 36 (6 + 30) | ~288 bytes |
| With Both | 72 (6 + 36 + 30) | ~576 bytes |

For long propagations, consider:

1. **Checkpointing**: Compute STM over segments and compose them
   $$\Phi(t_2, t_0) = \Phi(t_2, t_1) \Phi(t_1, t_0)$$

2. **Tighter tolerances**: Variational equation accuracy requires careful integrator settings

3. **Selective computation**: Enable only what you need (STM vs sensitivity vs both)

---

## See Also

- [Integrator Configuration](integrator_configuration.md) - Variational equation settings
- [Numerical Orbit Propagator](numerical_orbit_propagator.md) - Propagator fundamentals
- [Force Models](force_models.md) - Force model configuration
