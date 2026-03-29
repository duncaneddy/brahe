# Monte Carlo Orbit Reentry Analysis

In this example we'll use Brahe's Monte Carlo framework to analyze the uncertainty in atmospheric reentry timing for a low Earth orbit satellite. We'll define a LEO orbit, add uncertainty to the initial state and drag coefficient, configure a 100 km altitude event as the reentry trigger, and analyze the statistical spread of reentry times across many simulation runs.

---

## Setup

First, import the necessary libraries and initialize Earth orientation parameters. We use a static EOP provider for simplicity.

```python
import brahe as bh
import numpy as np

# Initialize EOP (required for numerical propagation)
eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
bh.set_global_eop_provider(eop)

# Define the initial epoch
epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.UTC)
```

## Define the Nominal Orbit

We set up a low Earth orbit at 250 km altitude with near-circular eccentricity. At this altitude, atmospheric drag causes orbital decay over days to weeks depending on spacecraft properties and solar activity.

```python
# Nominal orbital elements: [a, e, i, raan, argp, M] in meters, degrees
a = bh.R_EARTH + 250e3  # 250 km altitude
e = 0.001
i = 51.6    # ISS-like inclination
raan = 0.0
argp = 0.0
M = 0.0

# Convert to ECI Cartesian state
nominal_state = bh.koe_to_state_eci(a, e, i, raan, argp, M)

# Spacecraft parameters: [mass, drag_area, Cd, srp_area, Cr]
params = np.array([500.0, 2.0, 2.2, 2.0, 1.8])
```

## Configure the Monte Carlo Simulation

We create a simulation with 200 fixed runs and register two uncertain variables:

1. **Initial state** -- Sampled from a multivariate normal distribution with 100 m position uncertainty and 0.1 m/s velocity uncertainty per axis.
2. **Drag coefficient** -- Sampled from a truncated Gaussian centered at 2.2 with bounds [1.8, 2.6].

```python
# Create the simulation
stop = bh.MonteCarloStoppingCondition.fixed_runs(200)
sim = bh.MonteCarloSimulation(stop, seed=42)

# Initial state uncertainty: 100m position, 0.1 m/s velocity (1-sigma)
state_cov = np.diag([100.0**2, 100.0**2, 100.0**2, 0.1**2, 0.1**2, 0.1**2])

# Use a callback to add perturbation to the nominal state
def sample_initial_state(run_index, seed):
    rng = np.random.default_rng(seed)
    perturbation = rng.multivariate_normal(np.zeros(6), state_cov)
    return nominal_state + perturbation

sim.add_callback(bh.MonteCarloVariableId.INITIAL_STATE, sample_initial_state)

# Drag coefficient uncertainty
sim.add_variable(
    bh.MonteCarloVariableId.DRAG_COEFFICIENT,
    bh.TruncatedGaussian(mean=2.2, std=0.1, low=1.8, high=2.6),
)
```

## Configure Propagation

Define the force model, integrator settings, target epoch, event detectors, and which outputs to extract. The 100 km altitude event acts as a terminal condition -- propagation stops when the satellite descends below this altitude.

```python
# Force model with two-body gravity and atmospheric drag
force_config = bh.ForceModelConfig.two_body_with_drag()

# Integrator configuration
prop_config = bh.NumericalPropagationConfig()

# Target epoch: 30 days from start (enough time for reentry from 250 km)
target_epoch = epoch + 30.0 * 86400.0

# Event: stop propagation when altitude drops below 100 km
events = [
    {
        "type": "altitude",
        "altitude": 100e3,
        "name": "reentry",
        "terminal": True,
        "direction": "decreasing",
    }
]

# Outputs to collect
outputs = [
    "simulation_duration",
    "final_altitude",
    "final_semi_major_axis",
    "final_eccentricity",
    "first_event_duration_reentry",
]
```

## Run the Simulation

Call `run_orbit_propagation()` to execute all runs. This runs entirely in Rust with parallel execution via rayon.

```python
results = sim.run_orbit_propagation(
    epoch=epoch,
    propagation_config=prop_config,
    force_config=force_config,
    target_epoch=target_epoch,
    outputs=outputs,
    params=params,
    events=events,
)

print(f"Completed: {results.num_successful} successful, {results.num_failed} failed")
```

## Analyze Results

Extract statistical summaries from the results. The `first_event_duration_reentry` output gives the time from epoch to the first 100 km altitude crossing for each run.

```python
# Reentry duration statistics
mean_duration = results.mean("first_event_duration_reentry")
std_duration = results.std("first_event_duration_reentry")
se_duration = results.standard_error("first_event_duration_reentry")

print(f"Reentry duration:")
print(f"  Mean:   {mean_duration / 86400:.2f} days")
print(f"  Std:    {std_duration / 86400:.2f} days")
print(f"  SE:     {se_duration / 86400:.4f} days")
print(f"  Min:    {results.min('first_event_duration_reentry') / 86400:.2f} days")
print(f"  Max:    {results.max('first_event_duration_reentry') / 86400:.2f} days")
print(f"  Median: {results.percentile('first_event_duration_reentry', 0.5) / 86400:.2f} days")

# 5th and 95th percentiles
p05 = results.percentile("first_event_duration_reentry", 0.05)
p95 = results.percentile("first_event_duration_reentry", 0.95)
print(f"  90% CI: [{p05 / 86400:.2f}, {p95 / 86400:.2f}] days")
```

### Final Altitude Spread

The spread in final altitude shows how initial state and drag uncertainty affect the terminal state:

```python
# Final altitude statistics
mean_alt = results.mean("final_altitude")
std_alt = results.std("final_altitude")

print(f"\nFinal altitude:")
print(f"  Mean: {mean_alt / 1e3:.2f} km")
print(f"  Std:  {std_alt / 1e3:.2f} km")
```

### Inspect Individual Runs

Access individual run data to examine specific cases or build custom analyses:

```python
# Examine the first few runs
for run in results.runs[:5]:
    if run.succeeded:
        duration_days = run.outputs["first_event_duration_reentry"] / 86400
        alt_km = run.outputs["final_altitude"] / 1e3
        print(f"  Run {run.run_index}: reentry at {duration_days:.2f} days, "
              f"final alt {alt_km:.2f} km")
    else:
        print(f"  Run {run.run_index}: FAILED - {run.error_message}")
```

### Collect Raw Values

For custom plotting or further analysis, extract the raw arrays of output values:

```python
# Get all reentry durations as a numpy array
durations = results.scalar_values("first_event_duration_reentry")
print(f"\nCollected {len(durations)} reentry durations")
```

## Using Convergence-Based Stopping

For production analysis where you want the simulation to run until results stabilize, use convergence-based stopping instead of a fixed run count:

```python
stop = bh.MonteCarloStoppingCondition.convergence(
    targets=["first_event_duration_reentry"],
    threshold=100.0,    # stop when SE < 100 seconds
    min_runs=50,
    max_runs=5000,
    check_interval=25,
)

sim_conv = bh.MonteCarloSimulation(stop, seed=123)
# ... register the same variables as above ...

# After running:
# results.converged   -> True/False
# results.final_standard_errors -> {"first_event_duration_reentry": <SE value>}
```

## Using the Python Callable Mode

For simulations that need Python libraries in the loop (custom dynamics, plotting per-run, etc.), use the `run()` method with a Python callable instead:

```python
stop = bh.MonteCarloStoppingCondition.fixed_runs(100)
sim = bh.MonteCarloSimulation(stop, seed=42)
sim.add_variable(
    bh.MonteCarloVariableId.custom("cd"),
    bh.Gaussian(mean=2.2, std=0.1),
)

def custom_sim(run_index, variables):
    cd = variables["cd"]
    # Custom simulation logic using cd
    decay_rate = cd * 0.5  # simplified model
    return {"decay_rate": decay_rate}

results = sim.run(custom_sim)
print(f"Mean decay rate: {results.mean('decay_rate'):.4f}")
```

---

## See Also

- [Monte Carlo Simulation Guide](../learn/monte_carlo/index.md) -- Concepts and architecture
- [Monte Carlo API Reference](../library_api/monte_carlo/index.md) -- Full Python API documentation
- [Numerical Orbit Propagator](../learn/orbit_propagation/numerical_propagation/numerical_orbit_propagator.md) -- Propagator used in declarative mode
