# Vectorized Transformations

Frame transformation functions accept a batch of vectors in a single call. Passing a two-dimensional array such as `(n, 6)` to `state_eci_to_ecef` transforms every row and returns an array of the same shape; the loop, the Earth-orientation lookups, and the thread scheduling all happen inside the Rust core. A batch that shares one epoch runs hundreds of times faster than a Python loop over the scalar function because the IAU 2006/2000A rotation matrices are computed once and applied to every vector.

## Batch Input

A vectorized function is the same function as the scalar one. It decides how to evaluate based on the shape of its inputs:

| Input | Behavior |
|---|---|
| 1-D vector, single `Epoch` | Scalar transformation, unchanged output shape `(3,)` or `(6,)` |
| Array with the components along `axis` (default `-1`), single `Epoch` | Every vector is transformed with one shared epoch; the output keeps the input layout |
| 1-D vector, sequence of `n` epochs | The vector is transformed at each epoch; the output has shape `(n, k)` (or `(k, n)` with `axis=0`) |
| Array of `n` vectors, sequence of `n` epochs | Vector `i` is transformed at epoch `i`; the output keeps the input layout |

`axis` names the array dimension that holds the vector components, following the numpy convention used by functions such as `np.linalg.norm`. The default `-1` matches the common `(n, 6)` layout where each row is a state; `axis=0` selects a `(6, n)` layout where each column is a state. Any number of leading batch dimensions is accepted, so a `(2, 3, 6)` array is transformed element by element and returned as `(2, 3, 6)`.

Epoch arguments accept an `Epoch` or any sequence of `Epoch` objects (list, tuple, or object array). The epoch count must be `1` or equal to the number of vectors; other combinations raise `ValueError`.

Rotation-matrix functions such as `rotation_eci_to_ecef` accept a sequence of epochs and return an `(n, 3, 3)` array.

The same rules apply to every frame family: ECI/ECEF and GCRF/ITRF, EME2000, the lunar (LCI, LFPA, LFME), Mars (MCI, MCMF), and Earth-Moon barycenter (EMBI) frames, the EMR/SER/GSE synodic frames, `rotation_icrf_to_body_fixed_iau`, and the generic `rotation_frame_to_frame`, `position_frame_to_frame`, and `state_frame_to_frame` router. Functions that can fail for a single input (synodic and router transforms, IAU rotations) raise the same `RuntimeError` for a batch.

=== "Python"

    ``` python
    --8<-- "./examples/frames/vectorized_transforms.py:8"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/frames/vectorized_transforms.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/frames/vectorized_transforms.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/frames/vectorized_transforms.rs.txt"
        ```

## Rust API

The Rust core exposes each batch operation as a plural function alongside its scalar counterpart: `rotation_eci_to_ecef` and `rotations_eci_to_ecef`, `position_eci_to_ecef` and `positions_eci_to_ecef`, `state_eci_to_ecef` and `states_eci_to_ecef`. Plural functions take slices and return a `Vec`. Epoch-dependent functions take `&[Epoch]` as their first argument and follow one broadcast rule: every slice argument has length `1` or the common batch length, and the output has the common length. A length mismatch returns `Err(BraheError::Error)`.

Batches evaluate sequentially for small inputs and on the global thread pool for large ones. The pool size is controlled with `set_num_threads`; see [Threading](../utilities/threading.md).

## Two Regimes

The batch speed-up comes from two different sources, and which one applies depends on the epochs.

When all vectors share one epoch, the transformation context is computed once and applied to every vector: for ECI/ECEF the bias-precession-nutation, Earth-rotation, and polar-motion matrices; for the lunar and Mars body-fixed frames the PCK or IAU orientation and angular velocity; for the translation frames (LCI, MCI, EMBI) the ephemeris offset; for the synodic frames the rotating axes, their rate, and the origin offset; and for `position_frame_to_frame` the source and target rotation matrices and center offset. `state_frame_to_frame` is the exception: it evaluates the scalar transformation for each element, because the velocity transport terms are resolved through each frame's own state routine, so a batch through the state router gains from thread-pool evaluation but not from shared-epoch hoisting. The frame-specific `states_*` functions hoist. The per-vector work is a handful of matrix products, so the cost of the batch is dominated by the single context evaluation.

When each vector has its own epoch, the context must be evaluated per vector and the batch runs those evaluations across the thread pool. The speed-up over a Python loop is then the sum of avoiding per-call overhead and using multiple cores.

Results are identical to calling the scalar function in a loop in both regimes; the plural functions apply the same expressions in the same order.
