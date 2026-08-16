// Batch argument parsing and dispatch helpers for vectorized transformation
// bindings.
//
// Vectorized Python functions accept either a single vector (1-D input, the
// scalar path) or an array of vectors whose component axis is selected by
// `axis` (numpy convention, default `-1`). Epoch arguments accept a single
// `Epoch` or a sequence of `Epoch`. Batch evaluation calls the plural core
// function with the GIL released and restores the input layout on output.

/// Shape bookkeeping for a batched vector argument: the original array shape
/// and the (normalized) component axis.
struct BatchLayout {
    shape: Vec<usize>,
    axis: usize,
}

impl BatchLayout {
    /// Layout describing a batch of `n` vectors laid out along a fresh batch
    /// dimension, with the component axis placed according to `axis` (`-1`
    /// or `1` puts components last, giving `(n, N)`; `0` gives `(N, n)`).
    fn for_broadcast<const N: usize>(n: usize, axis: isize) -> PyResult<Self> {
        let axis = normalize_axis(axis, 2)?;
        let shape = if axis == 1 { vec![n, N] } else { vec![N, n] };
        Ok(BatchLayout { shape, axis })
    }

    /// Number of vectors described by this layout.
    fn batch_len(&self) -> usize {
        self.shape
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != self.axis)
            .map(|(_, d)| d)
            .product()
    }
}

/// A vector argument parsed from Python: a single vector or a batch.
enum VecArg<const N: usize> {
    Single(SVector<f64, N>),
    Batch {
        vecs: Vec<SVector<f64, N>>,
        layout: BatchLayout,
    },
}

/// An epoch argument parsed from Python: a single epoch or a sequence.
enum EpochArg {
    Single(time::Epoch),
    Many(Vec<time::Epoch>),
}

/// Resolve a possibly negative axis against `ndim`.
fn normalize_axis(axis: isize, ndim: usize) -> PyResult<usize> {
    let n = ndim as isize;
    if axis < -n || axis >= n {
        return Err(exceptions::PyValueError::new_err(format!(
            "axis {} is out of bounds for array of dimension {}",
            axis, ndim
        )));
    }
    Ok(if axis < 0 { (axis + n) as usize } else { axis as usize })
}

/// Parse a Python object into a single vector or a batch of vectors with the
/// component axis at `axis`.
fn parse_vec_arg<const N: usize>(obj: &Bound<'_, PyAny>, axis: isize) -> PyResult<VecArg<N>> {
    let py = obj.py();
    let np = py
        .import("numpy")
        .map_err(|_| exceptions::PyImportError::new_err("Failed to import numpy"))?;
    let float64 = np.getattr("float64")?;
    let arr = np
        .call_method1("asarray", (obj, float64))
        .map_err(|_| {
            exceptions::PyTypeError::new_err("Expected a numpy array or Python list of floats")
        })?;
    let arr = arr
        .cast::<PyArrayDyn<f64>>()
        .map_err(|_| exceptions::PyTypeError::new_err("Expected a numpy array or Python list"))?;
    let readonly = arr.readonly();
    let view = readonly.as_array();
    let ndim = view.ndim();

    if ndim == 0 {
        return Err(exceptions::PyValueError::new_err(
            "Expected an array or list of vectors, got a scalar",
        ));
    }

    if ndim == 1 {
        let len = view.len();
        if len != N {
            return Err(exceptions::PyValueError::new_err(format!(
                "Expected array or list of length {}, got {}",
                N, len
            )));
        }
        return Ok(VecArg::Single(SVector::<f64, N>::from_iterator(
            view.iter().copied(),
        )));
    }

    let axis_norm = normalize_axis(axis, ndim)?;
    let shape = view.shape().to_vec();
    if shape[axis_norm] != N {
        return Err(exceptions::PyValueError::new_err(format!(
            "Expected axis {} to have length {}, got shape {:?}",
            axis, N, shape
        )));
    }

    // Move the component axis last so each lane along it is one vector.
    let mut perm: Vec<usize> = (0..ndim).filter(|&i| i != axis_norm).collect();
    perm.push(axis_norm);
    let moved = view.permuted_axes(perm);
    let vecs: Vec<SVector<f64, N>> = moved
        .lanes(ndarray::Axis(ndim - 1))
        .into_iter()
        .map(|lane| SVector::<f64, N>::from_iterator(lane.iter().copied()))
        .collect();

    Ok(VecArg::Batch {
        vecs,
        layout: BatchLayout {
            shape,
            axis: axis_norm,
        },
    })
}

/// Parse a Python object into a single epoch or a sequence of epochs.
fn parse_epoch_arg(obj: &Bound<'_, PyAny>) -> PyResult<EpochArg> {
    if let Ok(epc) = obj.extract::<PyRef<'_, PyEpoch>>() {
        return Ok(EpochArg::Single(epc.obj));
    }
    if let Ok(epochs) = obj.extract::<Vec<PyRef<'_, PyEpoch>>>() {
        return Ok(EpochArg::Many(epochs.iter().map(|e| e.obj).collect()));
    }
    Err(exceptions::PyTypeError::new_err(
        "Expected an Epoch or a sequence of Epoch",
    ))
}

/// Validate the epoch/batch broadcast rule before calling the core.
fn check_batch_lengths(n_epochs: usize, n_vecs: usize) -> PyResult<()> {
    if n_epochs == 1 || n_vecs == 1 || n_epochs == n_vecs {
        Ok(())
    } else {
        Err(exceptions::PyValueError::new_err(format!(
            "Epoch count {} does not match batch length {}; expected 1 or equal lengths",
            n_epochs, n_vecs
        )))
    }
}

/// Convert a batch result back to a numpy array in the layout of the input.
///
/// When the core broadcast a length-1 batch across `n` epochs, the output is
/// laid out along a fresh batch dimension instead.
fn vecs_to_numpy<'py, const N: usize>(
    py: Python<'py>,
    layout: &BatchLayout,
    axis: isize,
    vecs: Vec<SVector<f64, N>>,
) -> PyResult<Bound<'py, PyAny>> {
    let n = vecs.len();
    let layout_owned;
    let layout = if layout.batch_len() == n {
        layout
    } else {
        layout_owned = BatchLayout::for_broadcast::<N>(n, axis)?;
        &layout_owned
    };
    let ndim = layout.shape.len();

    // Build (batch dims..., N) then move the component axis back into place.
    let mut moved_shape: Vec<usize> = layout
        .shape
        .iter()
        .enumerate()
        .filter(|(i, _)| *i != layout.axis)
        .map(|(_, d)| *d)
        .collect();
    moved_shape.push(N);
    let flat: Vec<f64> = vecs.iter().flat_map(|v| v.iter().copied()).collect();
    let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&moved_shape), flat)
        .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;

    let mut inv: Vec<usize> = (0..ndim - 1).collect();
    inv.insert(layout.axis, ndim - 1);
    let restored = arr.view().permuted_axes(inv);
    Ok(restored.to_pyarray(py).into_any())
}

/// Convert a batch of 3x3 matrices to an `(n, 3, 3)` numpy array.
fn matrices_to_numpy<'py>(py: Python<'py>, mats: Vec<SMatrix3>) -> Bound<'py, PyAny> {
    let n = mats.len();
    let flat: Vec<f64> = mats
        .iter()
        .flat_map(|m| (0..3).flat_map(move |i| (0..3).map(move |j| m[(i, j)])))
        .collect();
    flat.into_pyarray(py).reshape([n, 3, 3]).unwrap().into_any()
}

/// Inputs of a batched epoch-dependent transform: epochs, vectors, and the
/// output layout.
struct EpochVecBatch<const N: usize> {
    epochs: Vec<time::Epoch>,
    vecs: Vec<SVector<f64, N>>,
    layout: BatchLayout,
}

/// Parsed epoch/vector arguments: either the scalar pair or a batch.
enum EpochVecArgs<const N: usize> {
    Single(time::Epoch, SVector<f64, N>),
    Batch(EpochVecBatch<N>),
}

/// Parse and validate the epoch and vector arguments of a transform.
fn parse_epoch_vec_args<const N: usize>(
    epc: &Bound<'_, PyAny>,
    x: &Bound<'_, PyAny>,
    axis: isize,
) -> PyResult<EpochVecArgs<N>> {
    let epochs = parse_epoch_arg(epc)?;
    let vecs = parse_vec_arg::<N>(x, axis)?;
    let (epochs, vecs, layout) = match (epochs, vecs) {
        (EpochArg::Single(e), VecArg::Single(v)) => return Ok(EpochVecArgs::Single(e, v)),
        (EpochArg::Single(e), VecArg::Batch { vecs, layout }) => (vec![e], vecs, layout),
        (EpochArg::Many(epochs), VecArg::Single(v)) => {
            let layout = BatchLayout::for_broadcast::<N>(1, axis)?;
            (epochs, vec![v], layout)
        }
        (EpochArg::Many(epochs), VecArg::Batch { vecs, layout }) => (epochs, vecs, layout),
    };
    check_batch_lengths(epochs.len(), vecs.len())?;
    Ok(EpochVecArgs::Batch(EpochVecBatch {
        epochs,
        vecs,
        layout,
    }))
}

/// Dispatch an epoch-dependent vector transform on a single vector or a batch.
fn dispatch_epoch_vec<'py, const N: usize>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    axis: isize,
    scalar: impl Fn(time::Epoch, SVector<f64, N>) -> SVector<f64, N>,
    batch: impl Fn(&[time::Epoch], &[SVector<f64, N>]) -> Result<Vec<SVector<f64, N>>, RustBraheError>
    + Sync,
) -> PyResult<Bound<'py, PyAny>> {
    match parse_epoch_vec_args::<N>(epc, x, axis)? {
        EpochVecArgs::Single(e, v) => Ok(vector_to_numpy!(py, scalar(e, v), N, f64).into_any()),
        EpochVecArgs::Batch(b) => {
            let out = py.detach(|| batch(&b.epochs, &b.vecs))?;
            vecs_to_numpy(py, &b.layout, axis, out)
        }
    }
}

/// Dispatch a fallible epoch-dependent vector transform on a single vector or
/// a batch. Errors from the core are raised as `RuntimeError`.
fn try_dispatch_epoch_vec<'py, const N: usize>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    x: &Bound<'py, PyAny>,
    axis: isize,
    scalar: impl Fn(time::Epoch, SVector<f64, N>) -> Result<SVector<f64, N>, RustBraheError>,
    batch: impl Fn(&[time::Epoch], &[SVector<f64, N>]) -> Result<Vec<SVector<f64, N>>, RustBraheError>
    + Sync,
) -> PyResult<Bound<'py, PyAny>> {
    match parse_epoch_vec_args::<N>(epc, x, axis)? {
        EpochVecArgs::Single(e, v) => {
            let out = scalar(e, v).map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(vector_to_numpy!(py, out, N, f64).into_any())
        }
        EpochVecArgs::Batch(b) => {
            let out = py
                .detach(|| batch(&b.epochs, &b.vecs))
                .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
            vecs_to_numpy(py, &b.layout, axis, out)
        }
    }
}

/// Dispatch an epoch-free vector transform on a single vector or a batch.
fn dispatch_vec<'py, const N: usize>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    axis: isize,
    scalar: impl Fn(SVector<f64, N>) -> SVector<f64, N>,
    batch: impl Fn(&[SVector<f64, N>]) -> Vec<SVector<f64, N>> + Sync,
) -> PyResult<Bound<'py, PyAny>> {
    match parse_vec_arg::<N>(x, axis)? {
        VecArg::Single(v) => Ok(vector_to_numpy!(py, scalar(v), N, f64).into_any()),
        VecArg::Batch { vecs, layout } => {
            let out = py.detach(|| batch(&vecs));
            vecs_to_numpy(py, &layout, axis, out)
        }
    }
}

/// Dispatch an epoch-dependent rotation on a single epoch or a sequence.
fn dispatch_epoch_rotation<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    scalar: impl Fn(time::Epoch) -> SMatrix3,
    batch: impl Fn(&[time::Epoch]) -> Vec<SMatrix3> + Sync,
) -> PyResult<Bound<'py, PyAny>> {
    match parse_epoch_arg(epc)? {
        EpochArg::Single(e) => {
            let mat = scalar(e);
            Ok(matrix_to_numpy!(py, mat, 3, 3, f64).into_any())
        }
        EpochArg::Many(epochs) => {
            let out = py.detach(|| batch(&epochs));
            Ok(matrices_to_numpy(py, out))
        }
    }
}

/// Dispatch a fallible epoch-dependent rotation on a single epoch or a
/// sequence. Errors from the core are raised as `RuntimeError`.
fn try_dispatch_epoch_rotation<'py>(
    py: Python<'py>,
    epc: &Bound<'py, PyAny>,
    scalar: impl Fn(time::Epoch) -> Result<SMatrix3, RustBraheError>,
    batch: impl Fn(&[time::Epoch]) -> Result<Vec<SMatrix3>, RustBraheError> + Sync,
) -> PyResult<Bound<'py, PyAny>> {
    match parse_epoch_arg(epc)? {
        EpochArg::Single(e) => {
            let mat = scalar(e).map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(matrix_to_numpy!(py, mat, 3, 3, f64).into_any())
        }
        EpochArg::Many(epochs) => {
            let out = py
                .detach(|| batch(&epochs))
                .map_err(|e| exceptions::PyRuntimeError::new_err(e.to_string()))?;
            Ok(matrices_to_numpy(py, out))
        }
    }
}

/// Dispatch a fallible epoch-free vector transform on a single vector or a
/// batch. The closures return `PyResult` so each binding keeps its own error
/// mapping.
fn try_dispatch_vec<'py, const N: usize>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    axis: isize,
    scalar: impl Fn(SVector<f64, N>) -> PyResult<SVector<f64, N>>,
    batch: impl Fn(&[SVector<f64, N>]) -> PyResult<Vec<SVector<f64, N>>> + Sync,
) -> PyResult<Bound<'py, PyAny>> {
    match parse_vec_arg::<N>(x, axis)? {
        VecArg::Single(v) => {
            let out = scalar(v)?;
            Ok(vector_to_numpy!(py, out, N, f64).into_any())
        }
        VecArg::Batch { vecs, layout } => {
            let out = py.detach(|| batch(&vecs))?;
            vecs_to_numpy(py, &layout, axis, out)
        }
    }
}

/// Dispatch a vector-to-rotation-matrix function on a single vector or a
/// batch. A batch returns the batch dimensions followed by `(3, 3)`.
fn dispatch_vec_rotation<'py, const N: usize>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    axis: isize,
    scalar: impl Fn(SVector<f64, N>) -> SMatrix3,
    batch: impl Fn(&[SVector<f64, N>]) -> Vec<SMatrix3> + Sync,
) -> PyResult<Bound<'py, PyAny>> {
    match parse_vec_arg::<N>(x, axis)? {
        VecArg::Single(v) => {
            let mat = scalar(v);
            Ok(matrix_to_numpy!(py, mat, 3, 3, f64).into_any())
        }
        VecArg::Batch { vecs, layout } => {
            let out = py.detach(|| batch(&vecs));
            let mut shape: Vec<usize> = layout
                .shape
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != layout.axis)
                .map(|(_, d)| *d)
                .collect();
            shape.extend([3, 3]);
            let flat: Vec<f64> = out
                .iter()
                .flat_map(|m| (0..3).flat_map(move |i| (0..3).map(move |j| m[(i, j)])))
                .collect();
            Ok(flat
                .into_pyarray(py)
                .reshape(shape)
                .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?
                .into_any())
        }
    }
}

/// Dispatch a two-vector transform (for example site and target) on scalar
/// or batched arguments. Either argument may be a batch; when both are
/// batched their lengths must be equal or one of them must be 1. The output
/// takes the layout of the batched argument with the common length (the first
/// when both have it).
fn dispatch_vec_pair<'py, const N: usize>(
    py: Python<'py>,
    a: &Bound<'py, PyAny>,
    b: &Bound<'py, PyAny>,
    axis: isize,
    scalar: impl Fn(SVector<f64, N>, SVector<f64, N>) -> SVector<f64, N>,
    batch: impl Fn(&[SVector<f64, N>], &[SVector<f64, N>]) -> Result<Vec<SVector<f64, N>>, RustBraheError>
    + Sync,
) -> PyResult<Bound<'py, PyAny>> {
    let (a_vecs, a_layout) = match parse_vec_arg::<N>(a, axis)? {
        VecArg::Single(v) => (vec![v], None),
        VecArg::Batch { vecs, layout } => (vecs, Some(layout)),
    };
    let (b_vecs, b_layout) = match parse_vec_arg::<N>(b, axis)? {
        VecArg::Single(v) => (vec![v], None),
        VecArg::Batch { vecs, layout } => (vecs, Some(layout)),
    };
    let layout = match (a_layout, b_layout) {
        (None, None) => {
            return Ok(vector_to_numpy!(py, scalar(a_vecs[0], b_vecs[0]), N, f64).into_any());
        }
        (Some(la), Some(lb)) => {
            if a_vecs.len() == b_vecs.len() || b_vecs.len() == 1 {
                la
            } else if a_vecs.len() == 1 {
                lb
            } else {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Batch lengths {} and {} do not match; expected equal lengths or a single vector",
                    a_vecs.len(),
                    b_vecs.len()
                )));
            }
        }
        (Some(la), None) => la,
        (None, Some(lb)) => lb,
    };
    let out = py.detach(|| batch(&a_vecs, &b_vecs))?;
    vecs_to_numpy(py, &layout, axis, out)
}

/// A numeric argument parsed from Python: a scalar or an array of any shape.
enum NumArg {
    Scalar(f64),
    Array(ndarray::ArrayD<f64>),
}

/// Parse a Python float/int or array-like into a `NumArg`.
fn parse_num_arg(obj: &Bound<'_, PyAny>) -> PyResult<NumArg> {
    if let Ok(v) = obj.extract::<f64>() {
        return Ok(NumArg::Scalar(v));
    }
    let py = obj.py();
    let np = py
        .import("numpy")
        .map_err(|_| exceptions::PyImportError::new_err("Failed to import numpy"))?;
    let float64 = np.getattr("float64")?;
    let arr = np
        .call_method1("asarray", (obj, float64))
        .map_err(|_| exceptions::PyTypeError::new_err("Expected a number or an array of numbers"))?;
    let arr = arr
        .cast::<PyArrayDyn<f64>>()
        .map_err(|_| exceptions::PyTypeError::new_err("Expected a number or an array of numbers"))?;
    Ok(NumArg::Array(arr.readonly().as_array().to_owned()))
}

/// Broadcast numeric arguments against each other with numpy rules and return
/// the common shape plus each argument flattened to that shape (C order).
fn broadcast_num_args(py: Python<'_>, args: &[&NumArg]) -> PyResult<(Vec<usize>, Vec<Vec<f64>>)> {
    let np = py
        .import("numpy")
        .map_err(|_| exceptions::PyImportError::new_err("Failed to import numpy"))?;
    let py_args: Vec<Bound<'_, PyAny>> = args
        .iter()
        .map(|a| match a {
            NumArg::Scalar(v) => np.call_method1("asarray", (*v,)),
            NumArg::Array(arr) => Ok(arr.to_pyarray(py).into_any()),
        })
        .collect::<PyResult<Vec<_>>>()?;
    let broadcast = np
        .call_method1("broadcast_arrays", pyo3::types::PyTuple::new(py, &py_args)?)
        .map_err(|_| {
            exceptions::PyValueError::new_err("Arguments could not be broadcast to a common shape")
        })?;
    let mut shape: Vec<usize> = Vec::new();
    let mut flats: Vec<Vec<f64>> = Vec::new();
    for item in broadcast.try_iter()? {
        let arr = item?.cast_into::<PyArrayDyn<f64>>().map_err(|_| {
            exceptions::PyTypeError::new_err("Expected a number or an array of numbers")
        })?;
        let ro = arr.readonly();
        let view = ro.as_array();
        shape = view.shape().to_vec();
        flats.push(view.iter().copied().collect());
    }
    Ok((shape, flats))
}

/// Evaluate a scalar kernel over numeric arguments with numpy broadcasting.
/// All-scalar arguments return a Python float; otherwise an array of the
/// broadcast shape.
fn ufunc<'py>(
    py: Python<'py>,
    args: &[&Bound<'py, PyAny>],
    f: impl Fn(&[f64]) -> PyResult<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let parsed: Vec<NumArg> = args
        .iter()
        .map(|a| parse_num_arg(a))
        .collect::<PyResult<Vec<_>>>()?;
    if parsed.iter().all(|a| matches!(a, NumArg::Scalar(_))) {
        let vals: Vec<f64> = parsed
            .iter()
            .map(|a| match a {
                NumArg::Scalar(v) => *v,
                NumArg::Array(_) => unreachable!(),
            })
            .collect();
        return Ok(f(&vals)?.into_pyobject(py)?.into_any());
    }
    let refs: Vec<&NumArg> = parsed.iter().collect();
    let (shape, flats) = broadcast_num_args(py, &refs)?;
    let n = flats.first().map(|v| v.len()).unwrap_or(0);
    let mut out = Vec::with_capacity(n);
    let mut vals = vec![0.0; flats.len()];
    for i in 0..n {
        for (k, flat) in flats.iter().enumerate() {
            vals[k] = flat[i];
        }
        out.push(f(&vals)?);
    }
    let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), out)
        .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
    Ok(arr.into_pyarray(py).into_any())
}

/// Dispatch a Keplerian scalar function whose first argument is either a
/// value (with an optional second numeric argument `e`) or an element set.
///
/// - scalar first argument with no `e` or a scalar `e`: `elem_fn(x, e)`
/// - `(6,)` array: `oe_fn(elements)`; `e` is ignored
/// - `(n, 6)` array: `oe_fn` per row, returning `(n,)`; `e` is ignored
/// - any other array with no `e`: `ValueError`
/// - array `e` with a scalar first argument, or any other array first
///   argument with `e` given: element-wise `elem_fn` with numpy broadcasting
fn dispatch_oe_or_scalar<'py>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    e: Option<&Bound<'py, PyAny>>,
    oe_fn: impl Fn(&[f64]) -> PyResult<f64>,
    elem_fn: impl Fn(f64, Option<f64>) -> PyResult<f64>,
) -> PyResult<Bound<'py, PyAny>> {
    let x_arg = parse_num_arg(x)?;
    let e_arg = e.map(parse_num_arg).transpose()?;
    match (&x_arg, &e_arg) {
        (NumArg::Scalar(v), None) => Ok(elem_fn(*v, None)?.into_pyobject(py)?.into_any()),
        (NumArg::Scalar(v), Some(NumArg::Scalar(ecc))) => {
            Ok(elem_fn(*v, Some(*ecc))?.into_pyobject(py)?.into_any())
        }
        (NumArg::Array(arr), _) if arr.ndim() == 1 && arr.len() == 6 => {
            let oe: Vec<f64> = arr.iter().copied().collect();
            Ok(oe_fn(&oe)?.into_pyobject(py)?.into_any())
        }
        (NumArg::Array(arr), _) if arr.ndim() == 2 && arr.shape()[1] == 6 => {
            let out: Vec<f64> = arr
                .rows()
                .into_iter()
                .map(|row| {
                    let oe: Vec<f64> = row.iter().copied().collect();
                    oe_fn(&oe)
                })
                .collect::<PyResult<Vec<_>>>()?;
            Ok(out.into_pyarray(py).into_any())
        }
        (NumArg::Array(arr), None) => {
            if arr.ndim() == 1 {
                return Err(exceptions::PyValueError::new_err(format!(
                    "Expected array or list of length 6, got {}",
                    arr.len()
                )));
            }
            Err(exceptions::PyValueError::new_err(format!(
                "Expected an element set of shape (6,) or a batch of shape (n, 6), got shape {:?}",
                arr.shape()
            )))
        }
        (_, Some(ea)) => {
            let refs: Vec<&NumArg> = vec![&x_arg, ea];
            let (shape, flats) = broadcast_num_args(py, &refs)?;
            let out: Vec<f64> = flats[0]
                .iter()
                .zip(flats[1].iter())
                .map(|(x, ecc)| elem_fn(*x, Some(*ecc)))
                .collect::<PyResult<Vec<_>>>()?;
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), out)
                .map_err(|err| exceptions::PyValueError::new_err(err.to_string()))?;
            Ok(arr.into_pyarray(py).into_any())
        }
    }
}

/// Dispatch a vector-to-scalar function on a single vector or a batch. A batch
/// returns an array of the batch dimensions.
fn dispatch_vec_to_scalar<'py, const N: usize>(
    py: Python<'py>,
    x: &Bound<'py, PyAny>,
    axis: isize,
    f: impl Fn(SVector<f64, N>) -> f64,
) -> PyResult<Bound<'py, PyAny>> {
    match parse_vec_arg::<N>(x, axis)? {
        VecArg::Single(v) => Ok(f(v).into_pyobject(py)?.into_any()),
        VecArg::Batch { vecs, layout } => {
            let shape: Vec<usize> = layout
                .shape
                .iter()
                .enumerate()
                .filter(|(i, _)| *i != layout.axis)
                .map(|(_, d)| *d)
                .collect();
            let out: Vec<f64> = vecs.iter().map(|v| f(*v)).collect();
            let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), out)
                .map_err(|e| exceptions::PyValueError::new_err(e.to_string()))?;
            Ok(arr.into_pyarray(py).into_any())
        }
    }
}
