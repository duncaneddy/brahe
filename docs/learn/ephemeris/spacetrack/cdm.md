# Conjunction Data Messages

Conjunction Data Messages (CDMs) are collision risk assessments published by the 18th Space Defense Squadron. Each CDM describes a predicted close approach between two cataloged objects, including the time of closest approach (TCA), miss distance, and probability of collision ($P_c$).

Space-Track publishes CDMs through the `CDMPublic` request class, which uses the `expandedspacedata` controller. Unlike GP or SATCAT data, CDMs do not have a dedicated typed response struct -- use `query_json()` to receive results as a list of dictionaries (Python) or `Vec<serde_json::Value>` (Rust).

## CDM Fields

The following table lists commonly used CDM fields for filtering and analysis:

| Field | Description |
|-------|-------------|
| `CDM_ID` | Unique CDM identifier |
| `CREATED` | CDM creation timestamp |
| `EMERGENCY_REPORTABLE` | Emergency reportable flag (`Y`/`N`) |
| `TCA` | Time of Closest Approach |
| `MIN_RNG` | Minimum range at TCA (km) |
| `PC` | Probability of Collision |
| `SAT_1_ID` | NORAD catalog ID of the first object |
| `SAT_2_ID` | NORAD catalog ID of the second object |
| `SAT_1_NAME` | Name of the first object |
| `SAT_2_NAME` | Name of the second object |
| `SAT1_OBJECT_TYPE` | Object type of the first object (e.g., `PAYLOAD`, `DEBRIS`) |
| `SAT2_OBJECT_TYPE` | Object type of the second object |
| `SAT1_RCS` | Radar cross-section of the first object |
| `SAT2_RCS` | Radar cross-section of the second object |
| `SAT_1_EXCL_VOL` | Exclusion volume of the first object |
| `SAT_2_EXCL_VOL` | Exclusion volume of the second object |

!!! note "JSON-Only Responses"
    CDM queries return unstructured JSON. Use `query_json()` on the client to parse the response as `list[dict]` (Python) or `Vec<serde_json::Value>` (Rust). There is no typed `CDMRecord` struct.

## Query Examples

The examples below demonstrate building CDM queries. Each query constructs a URL path; execute it with `SpaceTrackClient.query_json()`.

=== "Python"

    ``` python
    --8<-- "./examples/spacetrack/cdm_high_probability.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/spacetrack/cdm_high_probability.rs:4"
    ```

### Working with CDM Results

After executing a CDM query with `query_json()`, each element in the returned list is a dictionary with string keys matching the field names above:

```python
# Example: iterating over CDM results (after query execution)
for cdm in results:
    tca = cdm["TCA"]
    pc = float(cdm["PC"])
    sat1 = cdm["SAT_1_NAME"]
    sat2 = cdm["SAT_2_NAME"]
    miss_km = float(cdm["MIN_RNG"])
    print(f"{tca}: {sat1} vs {sat2}, Pc={pc:.2e}, miss={miss_km:.1f} km")
```

---

## See Also

- [Common Queries](common_queries.md) -- GP, SATCAT, and Decay query patterns
- [Query Builder](query_builder.md) -- Filters, ordering, limits, and output formats
- [Client](client.md) -- Authentication and query execution
- [Operators Reference](../../../library_api/ephemeris/spacetrack/operators.md) -- All operator functions
