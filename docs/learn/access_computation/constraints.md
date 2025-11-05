# Constraints

Constraints define the criteria that must be satisfied for satellite access to ground locations. Brahe provides a comprehensive constraint system with built-in geometric constraints, logical composition operators, and support for custom user-defined constraints.

A constraint evaluates to `true` when access conditions are met and `false` otherwise. During access computation, the algorithm searches for continuous time periods where constraints remain `true`, identifying these as access windows.

## Built-in Constraints

### Elevation Constraint

The most common constraint - requires satellites to be above a minimum elevation angle. This accounts for terrain obstructions, atmospheric effects, and antenna pointing limits.

**Basic elevation constraint:**

=== "Python"

    ``` python
    --8<-- "./examples/access/constraints/elevation_basic.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/constraints/elevation_basic.rs:4"
    ```

**With maximum elevation:**

=== "Python"

    ``` python
    --8<-- "./examples/access/constraints/elevation_with_max.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/constraints/elevation_with_max.rs:4"
    ```

### Elevation Mask Constraint

Models azimuth-dependent elevation masks for terrain profiles, mountains, or buildings blocking low-elevation views in specific directions.

=== "Python"

    ``` python
    --8<-- "./examples/access/constraints/elevation_mask.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/constraints/elevation_mask.rs:4"
    ```

### Off-Nadir Constraint

Limits the off-nadir viewing angle for imaging satellites. Off-nadir angle is measured from the satellite's nadir (straight down) to the target location.

**Imaging payload:**

=== "Python"

    ``` python
    --8<-- "./examples/access/constraints/off_nadir_imaging.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/constraints/off_nadir_imaging.rs:4"
    ```

**Side-looking radar:**

=== "Python"

    ``` python
    --8<-- "./examples/access/constraints/off_nadir_side_looking.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/constraints/off_nadir_side_looking.rs:4"
    ```

### Local Time Constraint

Filters access windows by local solar time at the ground location. Useful for daylight-only imaging or night-time astronomy observations.

**Single time window:**

=== "Python"

    ``` python
    --8<-- "./examples/access/constraints/local_time_single_window.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/constraints/local_time_single_window.rs:4"
    ```

**Multiple time windows:**

=== "Python"

    ``` python
    --8<-- "./examples/access/constraints/local_time_multiple_windows.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/constraints/local_time_multiple_windows.rs:4"
    ```

**Using decimal hours:**

=== "Python"

    ``` python
    --8<-- "./examples/access/constraints/local_time_decimal_hours.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/constraints/local_time_decimal_hours.rs:4"
    ```

!!! warning "Local Solar Time"

    Local solar time is based on the Sun's position relative to the location, not clock time zones. Noon (1200) is when the Sun is highest in the sky.

### Look Direction Constraint

Requires the satellite to look in a specific direction relative to its velocity vector - left, right, or either side.

**Left-looking:**

=== "Python"

    ``` python
    --8<-- "./examples/access/constraints/look_direction_left.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/constraints/look_direction_left.rs:4"
    ```

### Ascending-Descending Constraint

Filters passes by whether the satellite is ascending (moving south-to-north) or descending (north-to-south) over the location.

**Ascending passes:**

=== "Python"

    ``` python
    --8<-- "./examples/access/constraints/asc_dsc_ascending.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/constraints/asc_dsc_ascending.rs:4"
    ```

## Constraint Composition

Combine constraints using Boolean logic to express complex requirements.

### ConstraintAll (AND Logic)

All child constraints must be satisfied simultaneously:

=== "Python"

    ``` python
    --8<-- "./examples/access/constraints/constraint_all.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/constraints/constraint_all.rs:4"
    ```

### ConstraintAny (OR Logic)

At least one child constraint must be satisfied:

=== "Python"

    ``` python
    --8<-- "./examples/access/constraints/constraint_any.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/constraints/constraint_any.rs:4"
    ```

### ConstraintNot (Negation)

Inverts a constraint - access occurs when the child constraint is NOT satisfied:

=== "Python"

    ``` python
    --8<-- "./examples/access/constraints/constraint_not.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/constraints/constraint_not.rs:4"
    ```

### Complex Composition

Build complex logic by combining multiple constraints:

=== "Python"

    ``` python
    --8<-- "./examples/access/constraints/nested_constraints.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/access/constraints/nested_constraints.rs:4"
    ```

## Custom Constraints (Python)

Python users can create fully custom constraints by implementing the `AccessConstraintComputer` interface:

``` python
--8<-- "./examples/access/constraints/custom_constraint.py:8"
```

!!! note "Custom Constraints in Rust"
    Rust users implement the `AccessConstraint` trait directly. This provides maximum performance but requires recompiling the library.

---

## See Also

- [Locations](locations.md) - Ground location types
- [Computation](computation.md) - How constraints are evaluated during access search
- [API Reference: Constraints](../../library_api/access/constraints.md)
- [Example: Predicting Ground Contacts](../../examples/ground_contacts.md)
- [Example: Computing Imaging Opportunities](../../examples/imaging_opportunities.md)
