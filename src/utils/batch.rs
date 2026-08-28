/*!
 * Batch evaluation primitives shared by the plural transformation functions
 * (`states_*`, `positions_*`, `rotations_*`, ...).
 *
 * All broadcast, epoch-context hoisting, and threading policy for batch
 * evaluation lives here so that each plural function is a one-line
 * composition of its scalar counterpart with one of these primitives.
 */

#![allow(dead_code)]

use rayon::prelude::*;

use crate::time::Epoch;
use crate::utils::errors::BraheError;
use crate::utils::threading::get_thread_pool;

/// Minimum batch length at which batch primitives evaluate on the global
/// rayon thread pool. Below this length evaluation is sequential, since the
/// per-element kernels are sub-microsecond and thread hand-off would dominate.
pub(crate) const PARALLEL_THRESHOLD: usize = 1024;

/// Resolve the common batch length of a set of slice lengths under the
/// broadcast rule: every length must be `1` or the common length `N`.
///
/// # Arguments
///
/// * `lens` - Lengths of each slice argument of a batch function
///
/// # Returns
///
/// The common length `N` (`1` if every length is `1`, `0` if any length is `0`
/// and the rest are `1`), or an error naming the offending lengths.
///
/// # Examples
///
/// ```ignore
/// use crate::utils::batch::broadcast_len;
///
/// assert_eq!(broadcast_len(&[1, 4]).unwrap(), 4);
/// assert!(broadcast_len(&[2, 4]).is_err());
/// ```
pub(crate) fn broadcast_len(lens: &[usize]) -> Result<usize, BraheError> {
    let n = lens.iter().copied().find(|&l| l != 1).unwrap_or(1);
    if lens.iter().any(|&l| l != 1 && l != n) {
        return Err(BraheError::Error(format!(
            "Batch inputs must have length 1 or a common length; got lengths {:?}",
            lens
        )));
    }
    Ok(n)
}

/// Select the `i`-th element of a broadcast slice argument.
///
/// # Arguments
///
/// * `slice` - Slice of length `1` (broadcast) or the common batch length
/// * `i` - Batch index
///
/// # Returns
///
/// `&slice[0]` when the slice has length `1`, otherwise `&slice[i]`.
///
/// # Examples
///
/// ```ignore
/// use crate::utils::batch::pick;
///
/// let one = [1.0];
/// let many = [1.0, 2.0, 3.0];
/// assert_eq!(*pick(&one, 2), 1.0);
/// assert_eq!(*pick(&many, 2), 3.0);
/// ```
pub(crate) fn pick<T>(slice: &[T], i: usize) -> &T {
    if slice.len() == 1 {
        &slice[0]
    } else {
        &slice[i]
    }
}

/// Evaluate `f` for every index in `0..n`, preserving order.
///
/// Runs on the global rayon thread pool when `n >= PARALLEL_THRESHOLD` and
/// sequentially otherwise.
///
/// # Arguments
///
/// * `f` - Kernel evaluated at each index
/// * `n` - Number of elements to evaluate
///
/// # Returns
///
/// Vector of `n` results in index order.
///
/// # Examples
///
/// ```ignore
/// use crate::utils::batch::map_indices;
///
/// let squares = map_indices(|i| i * i, 4);
/// assert_eq!(squares, vec![0, 1, 4, 9]);
/// ```
pub(crate) fn map_indices<U: Send>(f: impl Fn(usize) -> U + Sync, n: usize) -> Vec<U> {
    if n >= PARALLEL_THRESHOLD {
        get_thread_pool().install(|| (0..n).into_par_iter().map(&f).collect())
    } else {
        (0..n).map(&f).collect()
    }
}

/// Apply `f` to every element of `inputs`.
///
/// # Arguments
///
/// * `f` - Element-wise kernel
/// * `inputs` - Elements to transform
///
/// # Returns
///
/// Vector with one output per input, in input order.
///
/// # Examples
///
/// ```ignore
/// use crate::utils::batch::batch_map;
///
/// let doubled = batch_map(|x| 2.0 * x, &[1.0, 2.0, 3.0]);
/// assert_eq!(doubled, vec![2.0, 4.0, 6.0]);
/// ```
pub(crate) fn batch_map<T: Sync, U: Send>(f: impl Fn(&T) -> U + Sync, inputs: &[T]) -> Vec<U> {
    map_indices(|i| f(&inputs[i]), inputs.len())
}

/// Apply `f` pairwise across two slice arguments under the broadcast rule.
///
/// # Arguments
///
/// * `f` - Pairwise kernel
/// * `a` - First slice argument, length `1` or `N`
/// * `b` - Second slice argument, length `1` or `N`
///
/// # Returns
///
/// Vector of `N` results in index order, or an error if the lengths do not
/// satisfy the broadcast rule.
///
/// # Examples
///
/// ```ignore
/// use crate::utils::batch::batch_zip;
///
/// // A single left operand broadcasts across the right batch
/// let sums = batch_zip(|a, b| a + b, &[10.0], &[1.0, 2.0]).unwrap();
/// assert_eq!(sums, vec![11.0, 12.0]);
/// ```
pub(crate) fn batch_zip<A: Sync, B: Sync, U: Send>(
    f: impl Fn(&A, &B) -> U + Sync,
    a: &[A],
    b: &[B],
) -> Result<Vec<U>, BraheError> {
    let n = broadcast_len(&[a.len(), b.len()])?;
    Ok(map_indices(|i| f(pick(a, i), pick(b, i)), n))
}

/// Apply an epoch-dependent kernel across a batch, hoisting the epoch
/// context when the batch shares a single epoch.
///
/// When `epochs.len() == 1` the context is computed once and applied to every
/// input. Otherwise the context is computed for each element. `epochs` and
/// `inputs` follow the broadcast rule (each has length `1` or `N`).
///
/// # Arguments
///
/// * `context` - Builds the per-epoch context (rotation matrices, angular
///   rates, ephemeris lookups) that the scalar transform would otherwise
///   recompute on every call
/// * `apply` - Applies a context to one input
/// * `epochs` - Epochs, length `1` or `N`
/// * `inputs` - Elements to transform, length `1` or `N`
///
/// # Returns
///
/// Vector of `N` results in index order, or an error if the lengths do not
/// satisfy the broadcast rule.
///
/// # Examples
///
/// ```ignore
/// use crate::time::Epoch;
/// use crate::utils::batch::batch_map_epochs;
///
/// let epc = Epoch::from_gps_seconds(0.0);
/// // One epoch: the context closure runs once for the whole batch
/// let out = batch_map_epochs(|e| e.gps_seconds(), |t, x| t + x, &[epc], &[1.0, 2.0]).unwrap();
/// assert_eq!(out, vec![1.0, 2.0]);
/// ```
pub(crate) fn batch_map_epochs<C: Sync, T: Sync, U: Send>(
    context: impl Fn(Epoch) -> C + Sync,
    apply: impl Fn(&C, &T) -> U + Sync,
    epochs: &[Epoch],
    inputs: &[T],
) -> Result<Vec<U>, BraheError> {
    let n = broadcast_len(&[epochs.len(), inputs.len()])?;
    if n == 0 {
        return Ok(Vec::new());
    }
    if epochs.len() == 1 {
        let c = context(epochs[0]);
        Ok(map_indices(|i| apply(&c, pick(inputs, i)), n))
    } else {
        Ok(map_indices(
            |i| apply(&context(epochs[i]), pick(inputs, i)),
            n,
        ))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use approx::assert_abs_diff_eq;
    use serial_test::parallel;

    use super::*;
    use crate::time::Epoch;

    #[test]
    #[parallel]
    fn test_broadcast_len_common_length() {
        assert_eq!(broadcast_len(&[3, 3]).unwrap(), 3);
        assert_eq!(broadcast_len(&[1, 3]).unwrap(), 3);
        assert_eq!(broadcast_len(&[3, 1]).unwrap(), 3);
        assert_eq!(broadcast_len(&[1, 1]).unwrap(), 1);
        assert_eq!(broadcast_len(&[7]).unwrap(), 7);
    }

    #[test]
    #[parallel]
    fn test_broadcast_len_empty() {
        assert_eq!(broadcast_len(&[0, 0]).unwrap(), 0);
        assert_eq!(broadcast_len(&[1, 0]).unwrap(), 0);
        assert_eq!(broadcast_len(&[0, 1]).unwrap(), 0);
    }

    #[test]
    #[parallel]
    fn test_broadcast_len_mismatch() {
        let err = broadcast_len(&[3, 5]).unwrap_err();
        assert!(matches!(err, BraheError::Error(_)));
        assert!(err.to_string().contains("[3, 5]"));
        assert!(broadcast_len(&[0, 5]).is_err());
        assert!(broadcast_len(&[2, 1, 4]).is_err());
    }

    #[test]
    #[parallel]
    fn test_pick_broadcast_and_indexed() {
        let one = [10.0];
        let many = [1.0, 2.0, 3.0];
        assert_eq!(*pick(&one, 0), 10.0);
        assert_eq!(*pick(&one, 2), 10.0);
        assert_eq!(*pick(&many, 0), 1.0);
        assert_eq!(*pick(&many, 2), 3.0);
    }

    #[test]
    #[parallel]
    fn test_map_indices_sequential_preserves_order() {
        let out = map_indices(|i| i * 10, 5);
        assert_eq!(out, vec![0, 10, 20, 30, 40]);
    }

    #[test]
    #[parallel]
    fn test_map_indices_parallel_preserves_order() {
        let n = PARALLEL_THRESHOLD + 7;
        let out = map_indices(|i| i * 10, n);
        let expected: Vec<usize> = (0..n).map(|i| i * 10).collect();
        assert_eq!(out, expected);
    }

    #[test]
    #[parallel]
    fn test_map_indices_empty() {
        let out: Vec<usize> = map_indices(|i| i, 0);
        assert!(out.is_empty());
    }

    #[test]
    #[parallel]
    fn test_batch_map_small_and_large() {
        let small: Vec<f64> = (0..3).map(|i| i as f64).collect();
        assert_eq!(batch_map(|x| x * 2.0, &small), vec![0.0, 2.0, 4.0]);

        let large: Vec<f64> = (0..PARALLEL_THRESHOLD + 3).map(|i| i as f64).collect();
        let out = batch_map(|x| x * 2.0, &large);
        let expected: Vec<f64> = large.iter().map(|x| x * 2.0).collect();
        assert_eq!(out, expected);

        let empty: Vec<f64> = Vec::new();
        assert!(batch_map(|x| x * 2.0, &empty).is_empty());
    }

    #[test]
    #[parallel]
    fn test_batch_zip_paired() {
        let a = [1.0, 2.0, 3.0];
        let b = [10.0, 20.0, 30.0];
        assert_eq!(
            batch_zip(|x, y| x + y, &a, &b).unwrap(),
            vec![11.0, 22.0, 33.0]
        );
    }

    #[test]
    #[parallel]
    fn test_batch_zip_broadcast_either_side() {
        let one = [100.0];
        let many = [1.0, 2.0, 3.0];
        assert_eq!(
            batch_zip(|x, y| x + y, &one, &many).unwrap(),
            vec![101.0, 102.0, 103.0]
        );
        assert_eq!(
            batch_zip(|x, y| x - y, &many, &one).unwrap(),
            vec![-99.0, -98.0, -97.0]
        );
        assert_eq!(batch_zip(|x, y| x * y, &one, &one).unwrap(), vec![10000.0]);
    }

    #[test]
    #[parallel]
    fn test_batch_zip_mismatch_and_empty() {
        let a = [1.0, 2.0];
        let b = [1.0, 2.0, 3.0];
        assert!(batch_zip(|x, y| x + y, &a, &b).is_err());

        let empty: [f64; 0] = [];
        assert!(batch_zip(|x, y| x + y, &empty, &empty).unwrap().is_empty());
        assert!(batch_zip(|x, y| x + y, &[1.0], &empty).unwrap().is_empty());
    }

    #[test]
    #[parallel]
    fn test_batch_zip_parallel_path() {
        let n = PARALLEL_THRESHOLD + 1;
        let a: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let b = [0.5];
        let out = batch_zip(|x, y| x + y, &a, &b).unwrap();
        let expected: Vec<f64> = a.iter().map(|x| x + 0.5).collect();
        assert_eq!(out, expected);
    }

    fn epochs(n: usize) -> Vec<Epoch> {
        (0..n)
            .map(|i| Epoch::from_gps_seconds(60.0 * i as f64))
            .collect()
    }

    #[test]
    #[parallel]
    fn test_batch_map_epochs_shared_epoch_hoists_context() {
        let calls = AtomicUsize::new(0);
        let epc = epochs(1);
        let inputs = [1.0, 2.0, 3.0];
        let out = batch_map_epochs(
            |e| {
                calls.fetch_add(1, Ordering::SeqCst);
                e.gps_seconds()
            },
            |c, x| c + x,
            &epc,
            &inputs,
        )
        .unwrap();
        assert_eq!(out, vec![1.0, 2.0, 3.0]);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    #[parallel]
    fn test_batch_map_epochs_shared_epoch_parallel_still_hoists() {
        let calls = AtomicUsize::new(0);
        let epc = epochs(1);
        let inputs: Vec<f64> = (0..PARALLEL_THRESHOLD + 1).map(|i| i as f64).collect();
        let out = batch_map_epochs(
            |e| {
                calls.fetch_add(1, Ordering::SeqCst);
                e.gps_seconds()
            },
            |c, x| c + x,
            &epc,
            &inputs,
        )
        .unwrap();
        assert_eq!(out, inputs);
        assert_eq!(calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    #[parallel]
    fn test_batch_map_epochs_per_epoch() {
        let calls = AtomicUsize::new(0);
        let epc = epochs(3);
        let inputs = [1.0, 2.0, 3.0];
        let out = batch_map_epochs(
            |e| {
                calls.fetch_add(1, Ordering::SeqCst);
                e.gps_seconds()
            },
            |c, x| c + x,
            &epc,
            &inputs,
        )
        .unwrap();
        assert_eq!(out.len(), 3);
        for (got, want) in out.iter().zip([1.0, 62.0, 123.0]) {
            assert_abs_diff_eq!(*got, want, epsilon = 1e-9);
        }
        assert_eq!(calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    #[parallel]
    fn test_batch_map_epochs_single_input_many_epochs() {
        let epc = epochs(4);
        let inputs = [0.5];
        let out = batch_map_epochs(|e| e.gps_seconds(), |c, x| c + x, &epc, &inputs).unwrap();
        assert_eq!(out.len(), 4);
        for (got, want) in out.iter().zip([0.5, 60.5, 120.5, 180.5]) {
            assert_abs_diff_eq!(*got, want, epsilon = 1e-9);
        }
    }

    #[test]
    #[parallel]
    fn test_batch_map_epochs_mismatch_and_empty() {
        let epc = epochs(2);
        let inputs = [1.0, 2.0, 3.0];
        assert!(batch_map_epochs(|e| e.gps_seconds(), |c, x| c + x, &epc, &inputs).is_err());

        let calls = AtomicUsize::new(0);
        let none: [f64; 0] = [];
        let out = batch_map_epochs(
            |e| {
                calls.fetch_add(1, Ordering::SeqCst);
                e.gps_seconds()
            },
            |c, x| c + x,
            &epochs(1),
            &none,
        )
        .unwrap();
        assert!(out.is_empty());
        assert_eq!(calls.load(Ordering::SeqCst), 0);

        let out = batch_map_epochs(|e| e.gps_seconds(), |c, x| c + x, &epochs(0), &none).unwrap();
        assert!(out.is_empty());
    }
}
