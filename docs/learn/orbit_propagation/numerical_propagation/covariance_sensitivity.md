# Covariance and Sensitivity

The `NumericalOrbitPropagator` can propagate additional quantities alongside the orbital state, enabling covariance propagation and sensitivity analysis. This is essential for uncertainty quantification, orbit determination, and mission analysis.

## Full Example

Here is a complete example demonstrating STM, covariance, and sensitivity propagation together:

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/variational_overview.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/variational_overview.rs:4"
    ```

## Architecture Overview

### Configuration Hierarchy

Variational propagation is configured through the `VariationalConfig` within `NumericalPropagationConfig`:

``` .no-linenums
NumericalPropagationConfig
├── method: IntegrationMethod
├── integrator: IntegratorConfig
└── variational: VariationalConfig
    ├── enable_stm: bool
    ├── enable_sensitivity: bool
    ├── store_stm_history: bool
    ├── store_sensitivity_history: bool
    ├── jacobian_method: DifferenceMethod
    └── sensitivity_method: DifferenceMethod
```

### Auto-Enable Behavior

Providing `initial_covariance` when creating the propagator automatically enables STM propagation, even without explicitly setting `enable_stm = true`.

---

## State Transition Matrices (STM)

The State Transition Matrix (STM) is a foundational tool for linear uncertainty propagation. It describes how small perturbations in the initial state map to perturbations at a later time:

$$\delta \mathbf{x}(t) = \Phi(t, t_0) \delta \mathbf{x}(t_0)$$

where $\Phi(t, t_0)$ is the 6x6 STM from epoch $t_0$ to time $t$.

### STM Structure

For orbital mechanics with state $\mathbf{x} = [x, y, z, v_x, v_y, v_z]^T$, the 6x6 STM has structure:

$$\Phi = \begin{bmatrix} \frac{\partial \mathbf{r}}{\partial \mathbf{r}_0} & \frac{\partial \mathbf{r}}{\partial \mathbf{v}_0} \\ \frac{\partial \mathbf{v}}{\partial \mathbf{r}_0} & \frac{\partial \mathbf{v}}{\partial \mathbf{v}_0} \end{bmatrix}$$

<div class="center-table" markdown="1">
| Submatrix | Location | Physical Meaning |
|-----------|----------|------------------|
| $\frac{\partial \mathbf{r}}{\partial \mathbf{r}_0}$ | Upper left | Position sensitivity to initial position |
| $\frac{\partial \mathbf{r}}{\partial \mathbf{v}_0}$ | Upper right | Position sensitivity to initial velocity |
| $\frac{\partial \mathbf{v}}{\partial \mathbf{r}_0}$ | Lower left | Velocity sensitivity to initial position |
| $\frac{\partial \mathbf{v}}{\partial \mathbf{v}_0}$ | Lower right | Velocity sensitivity to initial velocity |
</div>

### STM Properties

The STM has several important mathematical properties:

1. **Identity at initial time**: $\Phi(t_0, t_0) = I$

2. **Composition**: STMs can be composed to span longer intervals:
   $\Phi(t_2, t_0) = \Phi(t_2, t_1) \Phi(t_1, t_0)$

3. **Determinant preservation**: For Hamiltonian systems (conservative forces only), $\det(\Phi) = 1$

### Enabling STM Propagation

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/stm_propagation.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/stm_propagation.rs:4"
    ```

---

## Covariance Propagation

The primary application of the STM is propagating uncertainty. Given an initial covariance $P_0$, the propagated covariance is:

$$P(t) = \Phi(t, t_0) P_0 \Phi(t, t_0)^T$$

### Creating a Propagator with Initial Covariance

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/covariance_propagation.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/covariance_propagation.rs:4"
    ```

### Covariance in RTN Frame

The RTN (Radial-Tangential-Normal) frame provides physical insight into how uncertainty evolves relative to the orbit.

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/covariance_rtn.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/covariance_rtn.rs:4"
    ```

### RTN Frame Interpretation

<div class="center-table" markdown="1">
| Component | Physical Meaning | Typical Behavior |
|-----------|------------------|------------------|
| Radial (R) | Altitude uncertainty | Bounded oscillation |
| Tangential (T) | Along-track timing | Unbounded growth |
| Normal (N) | Cross-track offset | Bounded oscillation |
</div>

The along-track (tangential) uncertainty grows fastest because velocity uncertainty causes timing errors that accumulate over time. After one orbit, the T/R ratio is typically 20-30x.

### Covariance Evolution Visualization

The following plot shows how position uncertainty evolves over three orbital periods in the ECI frame:

<div class="plotly-embed">
  <iframe class="only-light" src="../../../figures/covariance_evolution_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../../figures/covariance_evolution_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="covariance_evolution.py"
    --8<-- "./plots/learn/numerical_propagation/covariance_evolution.py:12"
    ```

### RTN Covariance Evolution

The RTN frame clearly shows why along-track error dominates:

<div class="plotly-embed">
  <iframe class="only-light" src="../../../figures/covariance_rtn_evolution_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../../figures/covariance_rtn_evolution_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="covariance_rtn_evolution.py"
    --8<-- "./plots/learn/numerical_propagation/covariance_rtn_evolution.py:12"
    ```

---

## Sensitivity Propagation

In addition to STM propagation (which tracks sensitivity to initial state), the propagator can compute **parameter sensitivity** - how the state changes with respect to configuration parameters.

### Parameter Sensitivity

The sensitivity matrix $S(t)$ describes how the state at time $t$ depends on the parameters:

$$S(t) = \frac{\partial \mathbf{x}(t)}{\partial \mathbf{p}}$$

where $\mathbf{p}$ is the parameter vector. This enables answering questions like:

- "How does a 1% uncertainty in drag coefficient affect position prediction?"
- "What is the impact of mass uncertainty on orbit determination?"

### Configuration Parameters

The default parameter vector contains spacecraft physical properties:

<div class="center-table" markdown="1">
| Index | Parameter | Units | Description |
|-------|-----------|-------|-------------|
| 0 | mass | kg | Spacecraft mass |
| 1 | drag_area | m² | Cross-sectional area for drag |
| 2 | Cd | - | Drag coefficient |
| 3 | srp_area | m² | Cross-sectional area for SRP |
| 4 | Cr | - | Solar radiation pressure coefficient |
</div>

### Enabling Sensitivity Propagation

=== "Python"

    ``` python
    --8<-- "./examples/numerical_propagation/sensitivity_analysis.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/numerical_propagation/sensitivity_analysis.rs:4"
    ```

### Interpreting Sensitivity Results

The sensitivity matrix $S$ is 6x5 (state dimension x parameter count):

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
  <iframe class="only-light" src="../../../figures/sensitivity_evolution_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../../figures/sensitivity_evolution_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="sensitivity_evolution.py"
    --8<-- "./plots/learn/numerical_propagation/sensitivity_evolution.py:12"
    ```

---

## Configuration Reference

### VariationalConfig Options

<div class="center-table" markdown="1">
| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `enable_stm` | bool | false | Enable State Transition Matrix computation |
| `enable_sensitivity` | bool | false | Enable parameter sensitivity computation |
| `store_stm_history` | bool | false | Store STM at each trajectory point |
| `store_sensitivity_history` | bool | false | Store sensitivity at each trajectory point |
| `jacobian_method` | DifferenceMethod | Central | Finite difference method for Jacobian |
| `sensitivity_method` | DifferenceMethod | Central | Finite difference method for sensitivity |
</div>

### DifferenceMethod Options

<div class="center-table" markdown="1">
| Method | Accuracy | Cost | Description |
|--------|----------|------|-------------|
| Forward | O(h) | S+1 evaluations | First-order forward differences |
| Central | O(h²) | 2S evaluations | Second-order central differences (default) |
| Backward | O(h) | S+1 evaluations | First-order backward differences |
</div>

### Computational Considerations

STM and sensitivity computation significantly increase computational cost:

<div class="center-table" markdown="1">
| Configuration | State dimension | Memory per step |
|---------------|-----------------|-----------------|
| State only | 6 | ~48 bytes |
| With STM | 42 (6 + 36) | ~336 bytes |
| With Sensitivity | 36 (6 + 30) | ~288 bytes |
| With Both | 72 (6 + 36 + 30) | ~576 bytes |
</div>

---

## See Also

- [Integrator Configuration](integrator_configuration.md) - Variational equation settings
- [Numerical Orbit Propagator](numerical_orbit_propagator.md) - Propagator fundamentals
- [Force Models](force_models.md) - Force model configuration
