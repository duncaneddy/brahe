# CDM — Conjunction Data Message

A Conjunction Data Message (CDM) describes a close approach between two space objects, providing state vectors, covariance matrices, and collision probability data at the Time of Closest Approach (TCA). It is the standard format used by conjunction assessment services (e.g., 18th Space Defense Squadron) to communicate collision risk to satellite operators.

## Parsing a CDM

Parse from KVN, XML, or JSON files and access conjunction data:

=== "Python"
    ``` python
    from brahe.ccsds import CDM

    cdm = CDM.from_file("conjunction.cdm")

    # Conjunction-level data
    print(f"TCA: {cdm.tca}")
    print(f"Miss distance: {cdm.miss_distance} m")
    print(f"Collision probability: {cdm.collision_probability}")

    # Object states (in meters and m/s)
    print(f"Object 1: {cdm.object1_name}")
    print(f"Object 1 state: {cdm.object1_state}")
    print(f"Object 2: {cdm.object2_name}")
    print(f"Object 2 state: {cdm.object2_state}")

    # Covariance matrix (6x6, in m², m²/s, m²/s²)
    cov = cdm.object1_covariance
    print(f"Position variance (R,R): {cov[0][0]} m²")
    ```

=== "Rust"
    ``` rust
    use brahe::ccsds::CDM;

    let cdm = CDM::from_file("conjunction.cdm").unwrap();

    println!("TCA: {:?}", cdm.tca());
    println!("Miss distance: {} m", cdm.miss_distance());
    println!("Collision probability: {:?}", cdm.collision_probability());

    // Object states as 6-element vectors [x, y, z, vx, vy, vz]
    let s1 = cdm.object1_state();
    let s2 = cdm.object2_state();

    // 6x6 RTN covariance submatrix
    let cov = cdm.object1_rtn_covariance_6x6();
    ```

## Creating a CDM

Build a CDM programmatically by constructing state vectors, covariance matrices, and object metadata, then combining them into a message:

=== "Python"
    ``` python
    --8<-- "./examples/ccsds/cdm_create_write.py:8"
    ```

=== "Rust"
    ``` rust
    --8<-- "./examples/ccsds/cdm_create_write.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/ccsds/cdm_create_write.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/ccsds/cdm_create_write.rs.txt"
        ```

## What a CDM Contains

Every CDM has a **header** (version, creation date, originator, message ID), **relative metadata** (TCA, miss distance, optional collision probability and screening volume), and exactly **two object sections**.

Each object section contains **metadata** (object identity, reference frame, covariance method, force model info), **OD parameters** (observation spans, residuals), **additional parameters** (mass, drag/SRP areas, hard-body radius), a **state vector** at TCA, and a **covariance matrix** in the RTN frame.

The covariance matrix is always specified in the Radial-Transverse-Normal (RTN) frame centered on the object. The standard 6$\times$6 matrix covers position and velocity uncertainty. CDM also supports extended 7$\times$7 through 9$\times$9 matrices that include drag coefficient, solar radiation pressure, and thrust uncertainty correlations.

## Format Support

CDM supports three encoding formats:

- **KVN** (`.cdm`, `.txt`) — keyword=value text, the most common format from conjunction screening services
- **XML** (`.xml`) — structured XML following the CCSDS NDM XML schema
- **JSON** — programmatic convenience format (not in the CCSDS standard)

All three formats are auto-detected on parse. Specify the format explicitly when writing:

```python
cdm = CDM.from_file("conjunction.cdm")  # Auto-detect
kvn = cdm.to_string("KVN")
xml = cdm.to_string("XML")
json_str = cdm.to_string("JSON")
```

## See Also

- [CDM API Reference](../../library_api/ccsds/cdm.md) — Full API documentation
- [OPM Format Guide](opm.md) — Single-state messages (closest analog)
- [CCSDS Module](index.md) — Module overview
