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
    let epochs = parse_epoch_arg(epc)?;
    let vecs = parse_vec_arg::<N>(x, axis)?;
    let (epochs, vecs, layout) = match (epochs, vecs) {
        (EpochArg::Single(e), VecArg::Single(v)) => {
            return Ok(vector_to_numpy!(py, scalar(e, v), N, f64).into_any());
        }
        (EpochArg::Single(e), VecArg::Batch { vecs, layout }) => (vec![e], vecs, layout),
        (EpochArg::Many(epochs), VecArg::Single(v)) => {
            let layout = BatchLayout::for_broadcast::<N>(1, axis)?;
            (epochs, vec![v], layout)
        }
        (EpochArg::Many(epochs), VecArg::Batch { vecs, layout }) => (epochs, vecs, layout),
    };
    check_batch_lengths(epochs.len(), vecs.len())?;
    let out = py.detach(|| batch(&epochs, &vecs))?;
    vecs_to_numpy(py, &layout, axis, out)
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
