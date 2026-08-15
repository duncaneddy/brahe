/*!
 * Batch evaluation primitives shared by the plural transformation functions
 * (`states_*`, `positions_*`, `rotations_*`, ...).
 *
 * All broadcast, epoch-context hoisting, and threading policy for batch
 * evaluation lives here so that each plural function is a one-line
 * composition of its scalar counterpart with one of these primitives.
 */

#![allow(dead_code)]

use crate::utils::errors::BraheError;

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
}
