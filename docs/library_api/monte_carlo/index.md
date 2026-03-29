# Monte Carlo

Monte Carlo simulation framework for astrodynamics uncertainty analysis.

!!! note
    For conceptual explanations and usage examples, see [Monte Carlo Simulation](../../learn/monte_carlo/index.md) in the User Guide and the [Monte Carlo Orbit Reentry Example](../../examples/monte_carlo_orbit.md).

## Simulation Engine

---

::: brahe.MonteCarloSimulation
    options:
      show_root_heading: true
      show_root_full_path: false

---

## Configuration

---

::: brahe.MonteCarloStoppingCondition
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.MonteCarloVariableId
    options:
      show_root_heading: true
      show_root_full_path: false

---

## Distributions

---

::: brahe.Gaussian
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.UniformDist
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.TruncatedGaussian
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.MultivariateNormal
    options:
      show_root_heading: true
      show_root_full_path: false

---

## Results

---

::: brahe.MonteCarloResults
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.MonteCarloRun
    options:
      show_root_heading: true
      show_root_full_path: false

---

## Thread-Local Providers

---

::: brahe.TableEOPProvider
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.TableSpaceWeatherProvider
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.set_thread_local_eop_provider
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.clear_thread_local_eop_provider
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.set_thread_local_space_weather_provider
    options:
      show_root_heading: true
      show_root_full_path: false

---

::: brahe.clear_thread_local_space_weather_provider
    options:
      show_root_heading: true
      show_root_full_path: false

---

## See Also

- [Monte Carlo Simulation Guide](../../learn/monte_carlo/index.md) -- Concepts and architecture
- [Monte Carlo Orbit Reentry Example](../../examples/monte_carlo_orbit.md) -- Step-by-step example
- [Numerical Orbit Propagator](../propagators/numerical_orbit_propagator.md) -- Propagator used in declarative mode
- [Force Model Config](../propagators/force_model_config.md) -- Force model configuration
- [Event Detection](../events/index.md) -- Event system for propagation
