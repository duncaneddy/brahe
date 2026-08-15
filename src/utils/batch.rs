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
/// * `n` - Number of elements to evaluate
/// * `f` - Kernel evaluated at each index
///
/// # Returns
///
/// Vector of `n` results in index order.
pub(crate) fn map_indices<U: Send>(n: usize, f: impl Fn(usize) -> U + Sync) -> Vec<U> {
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
/// * `inputs` - Elements to transform
/// * `f` - Element-wise kernel
///
/// # Returns
///
/// Vector with one output per input, in input order.
pub(crate) fn batch_map<T: Sync, U: Send>(inputs: &[T], f: impl Fn(&T) -> U + Sync) -> Vec<U> {
    map_indices(inputs.len(), |i| f(&inputs[i]))
}

/// Apply `f` pairwise across two slice arguments under the broadcast rule.
///
/// # Arguments
///
/// * `a` - First slice argument, length `1` or `N`
/// * `b` - Second slice argument, length `1` or `N`
/// * `f` - Pairwise kernel
///
/// # Returns
///
/// Vector of `N` results in index order, or an error if the lengths do not
/// satisfy the broadcast rule.
pub(crate) fn batch_zip<A: Sync, B: Sync, U: Send>(
    a: &[A],
    b: &[B],
    f: impl Fn(&A, &B) -> U + Sync,
) -> Result<Vec<U>, BraheError> {
    let n = broadcast_len(&[a.len(), b.len()])?;
    Ok(map_indices(n, |i| f(pick(a, i), pick(b, i))))
}

#[cfg(test)]
mod tests {
    use serial_test::parallel;

    use super::*;

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
        let out = map_indices(5, |i| i * 10);
        assert_eq!(out, vec![0, 10, 20, 30, 40]);
    }

    #[test]
    #[parallel]
    fn test_map_indices_parallel_preserves_order() {
        let n = PARALLEL_THRESHOLD + 7;
        let out = map_indices(n, |i| i * 10);
        let expected: Vec<usize> = (0..n).map(|i| i * 10).collect();
        assert_eq!(out, expected);
    }

    #[test]
    #[parallel]
    fn test_map_indices_empty() {
        let out: Vec<usize> = map_indices(0, |i| i);
        assert!(out.is_empty());
    }

    #[test]
    #[parallel]
    fn test_batch_map_small_and_large() {
        let small: Vec<f64> = (0..3).map(|i| i as f64).collect();
        assert_eq!(batch_map(&small, |x| x * 2.0), vec![0.0, 2.0, 4.0]);

        let large: Vec<f64> = (0..PARALLEL_THRESHOLD + 3).map(|i| i as f64).collect();
        let out = batch_map(&large, |x| x * 2.0);
        let expected: Vec<f64> = large.iter().map(|x| x * 2.0).collect();
        assert_eq!(out, expected);

        let empty: Vec<f64> = Vec::new();
        assert!(batch_map(&empty, |x| x * 2.0).is_empty());
    }

    #[test]
    #[parallel]
    fn test_batch_zip_paired() {
        let a = [1.0, 2.0, 3.0];
        let b = [10.0, 20.0, 30.0];
        assert_eq!(
            batch_zip(&a, &b, |x, y| x + y).unwrap(),
            vec![11.0, 22.0, 33.0]
        );
    }

    #[test]
    #[parallel]
    fn test_batch_zip_broadcast_either_side() {
        let one = [100.0];
        let many = [1.0, 2.0, 3.0];
        assert_eq!(
            batch_zip(&one, &many, |x, y| x + y).unwrap(),
            vec![101.0, 102.0, 103.0]
        );
        assert_eq!(
            batch_zip(&many, &one, |x, y| x - y).unwrap(),
            vec![-99.0, -98.0, -97.0]
        );
        assert_eq!(batch_zip(&one, &one, |x, y| x * y).unwrap(), vec![10000.0]);
    }

    #[test]
    #[parallel]
    fn test_batch_zip_mismatch_and_empty() {
        let a = [1.0, 2.0];
        let b = [1.0, 2.0, 3.0];
        assert!(batch_zip(&a, &b, |x, y| x + y).is_err());

        let empty: [f64; 0] = [];
        assert!(batch_zip(&empty, &empty, |x, y| x + y).unwrap().is_empty());
        assert!(batch_zip(&[1.0], &empty, |x, y| x + y).unwrap().is_empty());
    }

    #[test]
    #[parallel]
    fn test_batch_zip_parallel_path() {
        let n = PARALLEL_THRESHOLD + 1;
        let a: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let b = [0.5];
        let out = batch_zip(&a, &b, |x, y| x + y).unwrap();
        let expected: Vec<f64> = a.iter().map(|x| x + 0.5).collect();
        assert_eq!(out, expected);
    }
}
