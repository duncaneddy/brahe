# Monte Carlo Simulation

Monte Carlo simulation quantifies how uncertainty in inputs propagates through a dynamical system. Brahe's Monte Carlo framework samples uncertain parameters from probability distributions, runs many instances of a simulation in parallel, and collects statistical summaries of the outputs.

The framework supports two execution modes: a **declarative orbit propagation** mode that runs entirely in Rust with rayon-based parallelism, and a **Python callable** mode where a user-provided function defines the simulation logic. Both modes share the same variable sampling, configuration, and results infrastructure.

## Core Concepts

### Simulation Workflow

A Monte Carlo simulation follows four steps:

1. **Configure** -- Create a `MonteCarloSimulation` with a stopping condition and random seed.
2. **Register variables** -- Attach probability distributions, pre-sampled values, or callbacks to typed variable identifiers.
3. **Run** -- Execute the simulation. Each run receives its sampled parameters and produces named outputs.
4. **Analyze** -- Query the `MonteCarloResults` for means, standard deviations, percentiles, covariances, and convergence status.

### Variable Identifiers

`MonteCarloVariableId` provides typed identifiers for simulation variables. Built-in identifiers map to well-known spacecraft parameters:

| Identifier | Description | Params Index |
|---|---|---|
| `INITIAL_STATE` | State vector (position + velocity) | -- |
| `MASS` | Spacecraft mass | `params[0]` |
| `DRAG_AREA` | Drag reference area | `params[1]` |
| `DRAG_COEFFICIENT` | Drag coefficient $C_d$ | `params[2]` |
| `SRP_AREA` | Solar radiation pressure area | `params[3]` |
| `REFLECTIVITY_COEFFICIENT` | Reflectivity coefficient $C_r$ | `params[4]` |
| `EOP_TABLE` | Earth orientation parameter table | -- |
| `SPACE_WEATHER_TABLE` | Space weather data table | -- |

For user-defined variables, use `MonteCarloVariableId.custom("name")`.

### Distributions

Four probability distribution types are available for variable sampling:

| Distribution | Value Type | Description |
|---|---|---|
| `Gaussian(mean, std)` | Scalar | Univariate normal $N(\mu, \sigma^2)$ |
| `UniformDist(low, high)` | Scalar | Continuous uniform over $[\text{low}, \text{high})$ |
| `TruncatedGaussian(mean, std, low, high)` | Scalar | Gaussian truncated to $[\text{low}, \text{high}]$ via rejection sampling |
| `MultivariateNormal(mean, cov)` | Vector | Multivariate normal $N(\boldsymbol{\mu}, \boldsymbol{\Sigma})$ with Cholesky decomposition |

The `MultivariateNormal` distribution is useful for correlated initial state uncertainty, where the covariance matrix encodes correlations between position and velocity components.

### Sampling Sources

Variables can be populated in three ways:

- **Distribution** -- Sample from a built-in probability distribution each run. Added via `add_variable(var_id, distribution)`.
- **Pre-sampled** -- Provide an explicit list of values, one per run. Added via `add_presampled(var_id, samples)`. The list must contain at least as many values as the number of runs.
- **Callback** -- Supply a Python function `(run_index, seed) -> float | np.ndarray` that generates values with custom logic. Added via `add_callback(var_id, callback)`.

All sampling happens up front before execution begins, using deterministic per-run seeds derived from the master seed. This guarantees reproducible results regardless of thread scheduling.

## Stopping Conditions

Two stopping strategies control when the simulation terminates:

### Fixed Runs

Execute exactly $N$ simulation runs:

```python
stop = bh.MonteCarloStoppingCondition.fixed_runs(1000)
```

### Convergence-Based

Run until the standard error of the mean for all monitored outputs drops below a threshold. The standard error is computed as:

$$SE = \frac{\sigma}{\sqrt{n}}$$

where $\sigma$ is the sample standard deviation and $n$ is the number of successful runs.

```python
stop = bh.MonteCarloStoppingCondition.convergence(
    targets=["final_altitude"],
    threshold=0.01,     # SE threshold (meters, for altitude)
    min_runs=100,       # minimum runs before checking
    max_runs=10000,     # safety limit
    check_interval=50,  # check every 50 runs
)
```

Convergence checking runs simulations in batches. After each batch of `check_interval` runs, the standard error is computed. The simulation stops when all targets have $SE <$ `threshold`, or when `max_runs` is reached.

## Execution Modes

### Declarative Orbit Propagation

The `run_orbit_propagation()` method provides a built-in template for orbit propagation Monte Carlo analysis. Each run builds a `NumericalOrbitPropagator`, applies sampled parameter overrides, propagates to a target epoch, and extracts configured outputs. This mode runs entirely in Rust with rayon parallelism.

Available outputs:

| Output Key | Description |
|---|---|
| `"final_state"` | Final ECI Cartesian state vector (6 elements) |
| `"final_epoch"` | Final epoch as MJD |
| `"simulation_duration"` | Duration in seconds |
| `"final_altitude"` | Final geodetic altitude in meters |
| `"final_semi_major_axis"` | Final semi-major axis in meters |
| `"final_eccentricity"` | Final eccentricity |
| `"first_event_duration_<name>"` | Time to first occurrence of named event (seconds) |
| `"last_event_duration_<name>"` | Time to last occurrence of named event (seconds) |
| `"event_count_<name>"` | Number of occurrences of named event |

Event detectors can be added to monitor conditions during propagation. Currently, altitude crossing events are supported, configured as dictionaries:

```python
events = [
    {
        "type": "altitude",
        "altitude": 100e3,       # meters
        "name": "reentry",
        "terminal": True,
        "direction": "decreasing",
    }
]
```

### Python Callable

The `run()` method accepts a Python function that defines arbitrary simulation logic. The function receives `(run_index, variables)` where `variables` is a dictionary mapping variable names to sampled values, and returns a dictionary of output values:

```python
def simulation_fn(run_index, variables):
    x = variables["x"]
    return {"x_squared": x * x}

results = sim.run(simulation_fn)
```

This mode runs sequentially on the Python thread. Use it for custom dynamics, non-orbital simulations, or when Python libraries are needed in the simulation loop.

## Thread-Local EOP and Space Weather

When running orbit propagation Monte Carlo simulations, each parallel worker thread needs its own Earth orientation parameter (EOP) and space weather data. Brahe provides thread-local providers that override the global providers on a per-thread basis.

Register `MonteCarloVariableId.EOP_TABLE` or `MonteCarloVariableId.SPACE_WEATHER_TABLE` variables to supply per-run environmental data. When using `run_orbit_propagation()`, thread-local providers are automatically set from the sampled table data at the start of each run and cleaned up afterward.

For the Python callable mode, thread-local providers can be managed manually:

```python
from brahe import (
    TableEOPProvider,
    set_thread_local_eop_provider,
    clear_thread_local_eop_provider,
)
```

## Results and Statistics

`MonteCarloResults` provides statistical analysis methods across all successful runs:

| Method | Description |
|---|---|
| `mean(name)` | Arithmetic mean of a scalar output |
| `std(name)` | Sample standard deviation ($\text{ddof}=1$) |
| `percentile(name, p)` | Percentile via linear interpolation ($p \in [0, 1]$) |
| `min(name)` / `max(name)` | Extrema |
| `standard_error(name)` | Standard error of the mean: $\sigma / \sqrt{n}$ |
| `scalar_values(name)` | Raw array of all scalar values |
| `mean_vector(name)` | Element-wise mean of a vector output |
| `covariance(name)` | Sample covariance matrix of a vector output |

Additional properties:

- `num_successful` / `num_failed` -- counts of completed and errored runs
- `num_runs` -- total runs executed
- `converged` -- `True`/`False`/`None` for convergence mode
- `final_standard_errors` -- dictionary of final SE values (convergence mode only)
- `runs` -- list of individual `MonteCarloRun` objects with per-run inputs and outputs

### Deterministic Seeding

All simulations are reproducible given the same master seed. Per-run seeds are derived deterministically from the master seed and run index using a hash function from the PCG family. The same `(master_seed, run_index)` pair always produces the same per-run seed, regardless of execution order or parallelism.

## See Also

- [Monte Carlo Orbit Propagation Example](../../examples/monte_carlo_orbit.md) -- Step-by-step orbit reentry analysis
- [Monte Carlo API Reference](../../library_api/monte_carlo/index.md) -- Full Python API documentation
- [Numerical Orbit Propagator](../orbit_propagation/numerical_propagation/numerical_orbit_propagator.md) -- Propagator used in declarative mode
- [Event Detection](../orbit_propagation/numerical_propagation/event_detection.md) -- Event system used during propagation
