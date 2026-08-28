# Relative Motion

The relative motion module provides tools for working with satellite state representations,
relative motion frames, and relative dynamics models. 

Every relative-motion function accepts batches: passing `(n, 6)` arrays for the chief and deputy transforms each pair, and a single chief (or single deputy) broadcasts across the batch. The `axis` keyword (default `-1`) names the dimension holding the vector components, following the rules described in [Vectorized Transformations](../frames/vectorized.md).

### See Also

- [RTN Transformations API Reference](../../library_api/relative_motion/rtn_transformations.md) - Detailed API documentation
