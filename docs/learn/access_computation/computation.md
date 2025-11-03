# Access Computation

Access computation finds time windows when satellites can observe or communicate with ground locations, subject to geometric and operational constraints. Brahe provides the `location_accesses()` function as the primary function for finding accesses, with optional search configuration parameters to tune performance and accuracy.

## Basic Workflow

The simplest access computation requires: a location, a propagator, time bounds, and a constraint.

=== "Python"

    ``` python
    --8<-- "./examples/access/computation/basic_workflow.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/computation/basic_workflow.rs:4"
    ```

## Multiple Locations and Satellites

Compute access for multiple locations and satellites simultaneously:

=== "Python"

    ``` python
    --8<-- "./examples/access/computation/multiple_locations_satellites.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/computation/multiple_locations_satellites.rs:4"
    ```

## Algorithm Explanation

Brahe uses a two-step search algorithm to balance accuracy and performance:

### Phase 1: Coarse Search

The algorithm evaluates the constraint at regular time intervals (`initial_time_step`) across the entire search period. When the constraint transitions from `false` to `true`, a candidate access window has been found. This phase identifies periods of potential access quickly.

Optionally, adaptive stepping can be enabled to speed up the search by increasing by increasing the first step after an access window is found. The step size is based on a fraction of the satellite's orbital period (`adaptive_fraction`). For LEO satellites, this can significantly reduce the number of evaluations needed, as at most one access window occurs per orbit.

**Example:** With a 60-second time step over 24 hours, the algorithm performs ~1,440 constraint evaluations to identify candidate windows.

### Phase 2: Refinement

For each candidate window, the algorithm uses binary search to precisely locate the boundary times:

1. Start at the coarse boundary estimate
2. Take steps backward/forward at half the previous step size until the constraint changes
3. Evaluate constraint at each step
4. When constraint changes, reduce step size, change direction, and repeat
5. Continue until boundary is located to desired precision

## Configuration

The `AccessSearchConfig` struct controls algorithm behavior:

=== "Python"

    ``` python
    --8<-- "./examples/access/computation/custom_configuration.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/computation/custom_configuration.rs:4"
    ```

### Parameter Guidance

**`initial_time_step`** - Coarse search step size (seconds)

- **Smaller values** (10-60s): More accurate, slower, for complex constraints or short windows
- **Larger values** (60-180s): Faster, risk missing brief access windows
- **Rule of thumb**: Use 1/10th of expected minimum window duration

**`adaptive_step`** - Enable adaptive stepping to speed up coarse search

- **`true`**: Enabled, faster for LEO satellites with regular orbits
- **`false`**: Disabled, standard fixed-step search

**`adaptive_fraction`** - Fraction of orbital period for adaptive step size

- **Smaller values** (0.3-0.6): Smaller adaptive step, less risk of missing windows
- **Larger values** (0.6-0.8): Larger adaptive step, faster but riskier
- **Recommended**: 0.5-0.75 for LEO satellites

**`parallel`** - Enable parallel processing

- **`true`**: Process location-satellite pairs in parallel (recommended)
- **`false`**: Sequential processing, lower memory usage

**`num_threads`** - Thread pool size

- **0**: Auto-detect CPU cores (recommended)
- **N > 0**: Use exactly N threads for parallel work

## See Also

- [Locations](locations.md) - Ground location types and properties
- [Constraints](constraints.md) - Constraint system and composition
- [Access Computation Index](index.md) - Overview and usage examples
- [Example: Predicting Ground Contacts](../../examples/ground_contacts.md) - Complete workflow
- [API Reference: Access Module](../../library_api/access/index.md) - Complete API documentation
