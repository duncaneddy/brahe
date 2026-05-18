# Shared Types

Types and functions shared by both the CelesTrak and Space-Track ephemeris clients.

## GPRecord

::: brahe.GPRecord
    options:
      show_root_heading: true
      show_root_full_path: false

## Operator Functions

Filter operator functions for constructing query filter values. In Python, access these via `brahe.spacetrack.operators`:

```python
from brahe.spacetrack import operators as op
op.greater_than("25544")  # ">25544"
```

### Comparison Operators

::: brahe._brahe.spacetrack_greater_than

::: brahe._brahe.spacetrack_less_than

::: brahe._brahe.spacetrack_not_equal

### Range and Pattern Operators

::: brahe._brahe.spacetrack_inclusive_range

::: brahe._brahe.spacetrack_like

::: brahe._brahe.spacetrack_startswith

### Time References

::: brahe._brahe.spacetrack_now

::: brahe._brahe.spacetrack_now_offset

### Special Values

::: brahe._brahe.spacetrack_null_val

::: brahe._brahe.spacetrack_or_list

---

## See Also

- [Ephemeris Data Sources Overview](../../learn/ephemeris/index.md) -- GPRecord concepts and data source comparison
- [CelesTrak](celestrak.md) -- CelesTrak client that returns GPRecord
- [Space-Track](spacetrack/index.md) -- Space-Track client that returns GPRecord
