/*!
 * Covariance matrix construction and validation utilities.
 *
 * Provides common patterns for building covariance matrices from standard
 * deviation values, diagonal vectors, upper-triangular packed arrays, or
 * full matrices with validation. These utilities are used by measurement
 * models, process noise configuration, and other estimation components.
 */

use nalgebra::{DMatrix, DVector};

use crate::utils::errors::BraheError;

/// Create an isotropic covariance matrix: σ² · I.
///
/// Builds a `dim × dim` diagonal matrix where every diagonal element is
/// `sigma²`. This represents uncorrelated noise with equal variance on
/// all axes.
///
/// # Arguments
///
/// * `dim` - Matrix dimension (number of rows/columns)
/// * `sigma` - Standard deviation applied to all axes
///
/// # Returns
///
/// * `DMatrix<f64>` - `dim × dim` diagonal covariance matrix
///
/// # Examples
///
/// ```
/// use brahe::math::covariance::isotropic_covariance;
///
/// let r = isotropic_covariance(3, 10.0);
/// assert_eq!(r.nrows(), 3);
/// assert_eq!(r[(0, 0)], 100.0);
/// assert_eq!(r[(0, 1)], 0.0);
/// ```
pub fn isotropic_covariance(dim: usize, sigma: f64) -> DMatrix<f64> {
    DMatrix::from_diagonal(&DVector::from_element(dim, sigma * sigma))
}

/// Create a diagonal covariance matrix from per-axis standard deviations.
///
/// Each element of `sigmas` is squared to produce the corresponding
/// diagonal element. Off-diagonal elements are zero.
///
/// # Arguments
///
/// * `sigmas` - Slice of standard deviations, one per axis
///
/// # Returns
///
/// * `DMatrix<f64>` - `n × n` diagonal covariance matrix where `n = sigmas.len()`
///
/// # Examples
///
/// ```
/// use brahe::math::covariance::diagonal_covariance;
///
/// let r = diagonal_covariance(&[5.0, 10.0, 15.0]);
/// assert_eq!(r[(0, 0)], 25.0);
/// assert_eq!(r[(1, 1)], 100.0);
/// assert_eq!(r[(2, 2)], 225.0);
/// ```
pub fn diagonal_covariance(sigmas: &[f64]) -> DMatrix<f64> {
    let variances: Vec<f64> = sigmas.iter().map(|s| s * s).collect();
    DMatrix::from_diagonal(&DVector::from_vec(variances))
}

/// Validate a user-provided covariance matrix.
///
/// Checks that the matrix is square and symmetric within a relative
/// tolerance. Returns the validated matrix on success.
///
/// # Arguments
///
/// * `matrix` - Matrix to validate
///
/// # Returns
///
/// * `Ok(DMatrix<f64>)` - The validated matrix (unchanged)
/// * `Err(BraheError)` - If the matrix is not square or not symmetric
///
/// # Examples
///
/// ```
/// use brahe::math::covariance::validate_covariance;
/// use nalgebra::DMatrix;
///
/// let r = DMatrix::from_diagonal_element(3, 3, 100.0);
/// assert!(validate_covariance(r).is_ok());
/// ```
pub fn validate_covariance(matrix: DMatrix<f64>) -> Result<DMatrix<f64>, BraheError> {
    let n = matrix.nrows();
    let m = matrix.ncols();

    if n != m {
        return Err(BraheError::Error(format!(
            "Covariance matrix must be square, got {}x{}",
            n, m
        )));
    }

    // Check symmetry: |a_ij - a_ji| <= tol * max(|a_ij|, |a_ji|, 1.0)
    let tol = 1e-10;
    for i in 0..n {
        for j in (i + 1)..n {
            let aij = matrix[(i, j)];
            let aji = matrix[(j, i)];
            let scale = aij.abs().max(aji.abs()).max(1.0);
            if (aij - aji).abs() > tol * scale {
                return Err(BraheError::Error(format!(
                    "Covariance matrix is not symmetric: element ({},{})={} != ({},{})={}",
                    i, j, aij, j, i, aji
                )));
            }
        }
    }

    Ok(matrix)
}

/// Build a symmetric covariance matrix from upper-triangular elements.
///
/// Elements are provided in row-major packed upper-triangular order.
/// For a 3×3 matrix the expected order is:
/// `[c₀₀, c₀₁, c₀₂, c₁₁, c₁₂, c₂₂]` (6 elements).
///
/// The required number of elements is `dim * (dim + 1) / 2`.
///
/// # Arguments
///
/// * `dim` - Matrix dimension (number of rows/columns)
/// * `upper` - Upper-triangular elements in row-major packed order
///
/// # Returns
///
/// * `Ok(DMatrix<f64>)` - The symmetric `dim × dim` matrix
/// * `Err(BraheError)` - If the element count does not match `dim * (dim + 1) / 2`
///
/// # Examples
///
/// ```
/// use brahe::math::covariance::covariance_from_upper_triangular;
///
/// // [100, 5, 0, 225, 10, 400] → 3×3 symmetric matrix
/// let r = covariance_from_upper_triangular(3, &[100.0, 5.0, 0.0, 225.0, 10.0, 400.0]).unwrap();
/// assert_eq!(r[(0, 0)], 100.0);
/// assert_eq!(r[(0, 1)], 5.0);
/// assert_eq!(r[(1, 0)], 5.0);  // mirrored
/// assert_eq!(r[(2, 2)], 400.0);
/// ```
pub fn covariance_from_upper_triangular(
    dim: usize,
    upper: &[f64],
) -> Result<DMatrix<f64>, BraheError> {
    let expected = dim * (dim + 1) / 2;
    if upper.len() != expected {
        return Err(BraheError::Error(format!(
            "Upper-triangular covariance for {}x{} matrix requires {} elements, got {}",
            dim,
            dim,
            expected,
            upper.len()
        )));
    }

    let mut matrix = DMatrix::zeros(dim, dim);
    let mut idx = 0;
    for i in 0..dim {
        for j in i..dim {
            matrix[(i, j)] = upper[idx];
            matrix[(j, i)] = upper[idx];
            idx += 1;
        }
    }

    Ok(matrix)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;

    #[test]
    fn test_isotropic_covariance() {
        let r = isotropic_covariance(3, 10.0);
        assert_eq!(r.nrows(), 3);
        assert_eq!(r.ncols(), 3);
        assert_abs_diff_eq!(r[(0, 0)], 100.0, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(1, 1)], 100.0, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(2, 2)], 100.0, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(0, 1)], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(1, 2)], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_isotropic_covariance_1d() {
        let r = isotropic_covariance(1, 5.0);
        assert_eq!(r.nrows(), 1);
        assert_abs_diff_eq!(r[(0, 0)], 25.0, epsilon = 1e-15);
    }

    #[test]
    fn test_diagonal_covariance() {
        let r = diagonal_covariance(&[5.0, 10.0, 15.0]);
        assert_eq!(r.nrows(), 3);
        assert_abs_diff_eq!(r[(0, 0)], 25.0, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(1, 1)], 100.0, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(2, 2)], 225.0, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(0, 1)], 0.0, epsilon = 1e-15);
    }

    #[test]
    fn test_diagonal_covariance_6d() {
        let sigmas = [5.0, 10.0, 15.0, 0.05, 0.1, 0.15];
        let r = diagonal_covariance(&sigmas);
        assert_eq!(r.nrows(), 6);
        assert_abs_diff_eq!(r[(0, 0)], 25.0, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(3, 3)], 0.0025, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(5, 5)], 0.0225, epsilon = 1e-15);
    }

    #[test]
    fn test_validate_covariance_valid() {
        let r = DMatrix::from_diagonal_element(3, 3, 100.0);
        assert!(validate_covariance(r).is_ok());
    }

    #[test]
    fn test_validate_covariance_symmetric_with_offdiag() {
        let mut r = DMatrix::zeros(3, 3);
        r[(0, 0)] = 100.0;
        r[(1, 1)] = 200.0;
        r[(2, 2)] = 300.0;
        r[(0, 1)] = 5.0;
        r[(1, 0)] = 5.0;
        assert!(validate_covariance(r).is_ok());
    }

    #[test]
    fn test_validate_covariance_non_square() {
        let r = DMatrix::zeros(3, 4);
        let result = validate_covariance(r);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("square"));
    }

    #[test]
    fn test_validate_covariance_asymmetric() {
        let mut r = DMatrix::zeros(3, 3);
        r[(0, 0)] = 100.0;
        r[(1, 1)] = 200.0;
        r[(2, 2)] = 300.0;
        r[(0, 1)] = 5.0;
        r[(1, 0)] = 50.0; // asymmetric
        let result = validate_covariance(r);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("symmetric"));
    }

    #[test]
    fn test_covariance_from_upper_triangular_3d() {
        // [c00, c01, c02, c11, c12, c22]
        let upper = [100.0, 5.0, 0.0, 225.0, 10.0, 400.0];
        let r = covariance_from_upper_triangular(3, &upper).unwrap();

        assert_eq!(r.nrows(), 3);
        assert_eq!(r.ncols(), 3);

        // Diagonal
        assert_abs_diff_eq!(r[(0, 0)], 100.0, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(1, 1)], 225.0, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(2, 2)], 400.0, epsilon = 1e-15);

        // Off-diagonal (symmetric)
        assert_abs_diff_eq!(r[(0, 1)], 5.0, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(1, 0)], 5.0, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(0, 2)], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(2, 0)], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(1, 2)], 10.0, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(2, 1)], 10.0, epsilon = 1e-15);
    }

    #[test]
    fn test_covariance_from_upper_triangular_6d() {
        // 6x6 → 21 elements, packed row-major upper triangle:
        // Row 0: indices 0..5 → (0,0),(0,1),(0,2),(0,3),(0,4),(0,5)
        // Row 1: indices 6..10 → (1,1),(1,2),(1,3),(1,4),(1,5)
        // Row 2: indices 11..14 → (2,2),(2,3),(2,4),(2,5)
        // Row 3: indices 15..17 → (3,3),(3,4),(3,5)
        // Row 4: indices 18..19 → (4,4),(4,5)
        // Row 5: index 20 → (5,5)
        let mut upper = vec![0.0; 21];
        upper[0] = 1.0; // (0,0)
        upper[6] = 4.0; // (1,1)
        upper[11] = 9.0; // (2,2)
        upper[15] = 16.0; // (3,3)
        upper[18] = 25.0; // (4,4)
        upper[20] = 36.0; // (5,5)
        // Set one off-diagonal: (0,3) at index 3
        upper[3] = 7.5;

        let r = covariance_from_upper_triangular(6, &upper).unwrap();
        assert_eq!(r.nrows(), 6);

        // Diagonal
        assert_abs_diff_eq!(r[(0, 0)], 1.0, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(1, 1)], 4.0, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(4, 4)], 25.0, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(5, 5)], 36.0, epsilon = 1e-15);

        // Off-diagonal symmetry
        assert_abs_diff_eq!(r[(0, 3)], 7.5, epsilon = 1e-15);
        assert_abs_diff_eq!(r[(3, 0)], 7.5, epsilon = 1e-15);
    }

    #[test]
    fn test_covariance_from_upper_triangular_wrong_count() {
        let result = covariance_from_upper_triangular(3, &[1.0, 2.0, 3.0]);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("6 elements"));
    }

    #[test]
    fn test_covariance_from_upper_triangular_1d() {
        let r = covariance_from_upper_triangular(1, &[42.0]).unwrap();
        assert_eq!(r.nrows(), 1);
        assert_abs_diff_eq!(r[(0, 0)], 42.0, epsilon = 1e-15);
    }
}
