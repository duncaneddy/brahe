/*!
 * Linear algebra utilities and type definitions.
 */

use nalgebra as na;
use nalgebra::linalg::SymmetricEigen;
use num_traits::float::Float;

/// 3-dimensional static vector type for Cartesian state vectors
pub type SVector3 = na::SVector<f64, 3>;

/// 6-dimensional static vector type for Cartesian state vectors
pub type SVector6 = na::SVector<f64, 6>;

/// 3x3 static matrix type for rotation matrices and transformations
pub type SMatrix3 = na::SMatrix<f64, 3, 3>;

/// 6x6 static matrix type for rotation matrices and transformations
pub type SMatrix6 = na::SMatrix<f64, 6, 6>;

/// Split a floating point number into its integer and fractional parts.
///
/// # Arguments
/// - `num`: The number to split. Can be `f32` or `f64`.
///
/// # Examples
/// ```
/// use brahe::math::linalg::split_float;
///
/// assert!(split_float(1.5_f32) == (1.0, 0.5));
/// assert!(split_float(-1.5_f32) == (-1.0, -0.5));
/// assert!(split_float(0.0_f32) == (0.0, 0.0));
/// assert!(split_float(1.0_f32) == (1.0, 0.0));
///
/// assert!(split_float(1.5_f64) == (1.0, 0.5));
/// assert!(split_float(-1.5_f64) == (-1.0, -0.5));
/// assert!(split_float(0.0_f64) == (0.0, 0.0));
/// assert!(split_float(1.0_f64) == (1.0, 0.0));
/// ```
pub fn split_float<T: Float>(num: T) -> (T, T) {
    (T::trunc(num), T::fract(num))
}

/// Convert a 3-element array to a `na::Vector3<f64>`.
///
/// # Arguments
/// - `vec`: The 3-element array to convert.
///
/// # Examples
/// ```
/// use nalgebra as na;
/// use brahe::math::linalg::vector3_from_array;
///
/// let vec = [1.0, 2.0, 3.0];
/// let v = vector3_from_array(vec);
/// assert_eq!(v, na::Vector3::new(1.0, 2.0, 3.0));
/// ```
pub fn vector3_from_array(vec: [f64; 3]) -> na::Vector3<f64> {
    na::Vector3::new(vec[0], vec[1], vec[2])
}

/// Convert a 6-element array to a `na::SVector<f64, 6>`.
///
/// # Arguments
/// - `vec`: The 6-element array to convert.
///
/// # Examples
/// ```
/// use nalgebra as na;
/// use brahe::math::linalg::vector6_from_array;
///
/// let vec = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
/// let v = vector6_from_array(vec);
/// assert_eq!(v, na::SVector::<f64, 6>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0));
/// ```
pub fn vector6_from_array(vec: [f64; 6]) -> na::SVector<f64, 6> {
    na::SVector::<f64, 6>::new(vec[0], vec[1], vec[2], vec[3], vec[4], vec[5])
}

/// Convert a 3x3 array to a `na::SMatrix<f64, 3, 3>`.
///
/// # Arguments
/// - `mat`: The 3x3 array to convert.
///
/// # Examples
/// ```
/// use nalgebra as na;
/// use brahe::math::linalg::matrix3_from_array;
///
/// let mat = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
/// let m = matrix3_from_array(&mat);
/// assert_eq!(m, na::SMatrix::<f64, 3, 3>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0));
/// ```
pub fn matrix3_from_array(mat: &[[f64; 3]; 3]) -> na::SMatrix<f64, 3, 3> {
    na::SMatrix::<f64, 3, 3>::new(
        mat[0][0], mat[0][1], mat[0][2], mat[1][0], mat[1][1], mat[1][2], mat[2][0], mat[2][1],
        mat[2][2],
    )
}

/// Compute the Kronecker delta function.
///
/// # Arguments
///
/// - `i`: The first index.
/// - `j`: The second index.
///
/// # Returns
///
/// - `u8`: The value of the Kronecker delta function.
///
/// # Examples
///
/// ```
/// use brahe::math::linalg::kronecker_delta;
///
/// assert_eq!(kronecker_delta(0, 0), 1);
/// assert_eq!(kronecker_delta(0, 1), 0);
/// assert_eq!(kronecker_delta(1, 0), 0);
/// assert_eq!(kronecker_delta(1, 1), 1);
/// ```
pub fn kronecker_delta(i: usize, j: usize) -> u8 {
    if i == j { 1 } else { 0 }
}

/// Compute the matrix square root of a symmetric positive-definite matrix.
///
/// This function computes the square root of a symmetric positive-definite matrix
/// using eigenvalue decomposition. For a symmetric positive-definite matrix M,
/// the square root is computed as:
///
/// M^(1/2) = V * D^(1/2) * V^T
///
/// where M = V * D * V^T is the eigendecomposition with V containing the eigenvectors
/// as columns and D being a diagonal matrix of eigenvalues.
///
/// This function is optimized for symmetric matrices (such as covariance matrices)
/// and uses `SymmetricEigen` for efficient computation.
///
/// # Arguments
///
/// * `matrix` - A symmetric positive-definite matrix of size N×N
///
/// # Returns
///
/// * `Result<SMatrix<f64, N, N>, String>` - The matrix square root, or an error if:
///   - The matrix has negative eigenvalues (not positive-definite)
///   - The eigendecomposition fails
///
/// # Examples
///
/// ```
/// use nalgebra::SMatrix;
/// use brahe::math::linalg::spd_sqrtm;
///
/// // Identity matrix
/// let identity = SMatrix::<f64, 2, 2>::identity();
/// let sqrt_identity = spd_sqrtm(identity).unwrap();
/// assert!((sqrt_identity - identity).norm() < 1e-10);
///
/// // Diagonal matrix
/// let diag = SMatrix::<f64, 2, 2>::new(4.0, 0.0, 0.0, 9.0);
/// let sqrt_diag = spd_sqrtm(diag).unwrap();
/// let expected = SMatrix::<f64, 2, 2>::new(2.0, 0.0, 0.0, 3.0);
/// assert!((sqrt_diag - expected).norm() < 1e-10);
/// ```
pub fn spd_sqrtm<const N: usize>(
    matrix: na::SMatrix<f64, N, N>,
) -> Result<na::SMatrix<f64, N, N>, String>
where
    na::Const<N>: na::DimName,
{
    // Convert to DMatrix for eigendecomposition
    let dmatrix = na::DMatrix::from_iterator(N, N, matrix.iter().cloned());

    // Compute symmetric eigendecomposition
    let eigen = SymmetricEigen::new(dmatrix);

    // Check for negative eigenvalues
    for &eigenvalue in eigen.eigenvalues.iter() {
        if eigenvalue < 0.0 {
            return Err(format!(
                "Matrix is not positive-definite: found negative eigenvalue {}",
                eigenvalue
            ));
        }
    }

    // Compute square root of eigenvalues
    let sqrt_eigenvalues = eigen.eigenvalues.map(|x: f64| x.sqrt());

    // Reconstruct: M^(1/2) = V * sqrt(D) * V^T
    // where V is the eigenvector matrix and D is the diagonal eigenvalue matrix
    let v = &eigen.eigenvectors;
    let sqrt_d = na::DMatrix::<f64>::from_diagonal(&sqrt_eigenvalues);
    let result_dmatrix = v * sqrt_d * v.transpose();

    // Convert back to SMatrix
    let mut result = na::SMatrix::<f64, N, N>::zeros();
    for i in 0..N {
        for j in 0..N {
            result[(i, j)] = result_dmatrix[(i, j)];
        }
    }

    Ok(result)
}

/// Compute the matrix square root of a general square matrix.
///
/// This function computes the square root of a general (possibly non-symmetric) square matrix
/// using Denman-Beavers iteration. See [Denman-Beavers iteration](https://en.wikipedia.org/wiki/Square_root_of_a_matrix#By_Denman%E2%80%93Beavers_iteration) for additional details.
///
/// # Arguments
///
/// * `matrix` - A square matrix of size N×N
///
/// # Returns
///
/// * `Result<SMatrix<f64, N, N>, String>` - The matrix square root, or an error if:
///   - The matrix has complex eigenvalues
///   - The matrix has negative real eigenvalues
///   - The eigendecomposition fails
///
/// # Examples
///
/// ```
/// use nalgebra::SMatrix;
/// use brahe::math::linalg::sqrtm;
///
/// // Test case: A = [33 24; 48 57], sqrtm(A) = [5 2; 4 7]
/// let a = SMatrix::<f64, 2, 2>::new(33.0, 24.0, 48.0, 57.0);
/// let sqrt_a = sqrtm(a).unwrap();
/// let expected = SMatrix::<f64, 2, 2>::new(5.0, 2.0, 4.0, 7.0);
/// assert!((sqrt_a - expected).norm() < 1e-10);
///
/// // Verify: sqrtm(A) * sqrtm(A) = A
/// let reconstructed = sqrt_a * sqrt_a;
/// assert!((reconstructed - a).norm() < 1e-10);
/// ```
pub fn sqrtm<const N: usize>(
    matrix: na::SMatrix<f64, N, N>,
) -> Result<na::SMatrix<f64, N, N>, String>
where
    na::Const<N>: na::DimName,
{
    // Use Denman-Beavers iteration for computing matrix square root
    // This works for any matrix with eigenvalues in the open right half-plane
    // Iterations: Y_{k+1} = (Y_k + Z_k^{-1}) / 2
    //             Z_{k+1} = (Z_k + Y_k^{-1}) / 2
    // Starting with Y_0 = A, Z_0 = I

    // Convert to DMatrix for computation
    let a = na::DMatrix::from_iterator(N, N, matrix.iter().cloned());

    let mut y = a.clone();
    let mut z = na::DMatrix::<f64>::identity(N, N);

    const MAX_ITERATIONS: usize = 50;
    const TOLERANCE: f64 = 1e-10;

    for _ in 0..MAX_ITERATIONS {
        // Compute inverses
        let y_inv = y.clone().try_inverse().ok_or_else(|| {
            "Matrix became singular during iteration; cannot compute matrix square root".to_string()
        })?;

        let z_inv = z.clone().try_inverse().ok_or_else(|| {
            "Iteration matrix became singular; cannot compute matrix square root".to_string()
        })?;

        // Update Y and Z
        let y_new = (&y + &z_inv) * 0.5;
        let z_new = (&z + &y_inv) * 0.5;

        // Check convergence: ||Y_{k+1} - Y_k|| < tolerance
        let diff = (&y_new - &y).norm();
        if diff < TOLERANCE {
            y = y_new;
            break;
        }

        y = y_new;
        z = z_new;
    }

    // Verify the result: Y * Y should equal A
    let check = &y * &y;
    let error = (&check - &a).norm();
    if error > 1e-8 {
        return Err(format!(
            "Matrix square root did not converge to sufficient accuracy (error: {})",
            error
        ));
    }

    // Convert back to SMatrix
    let mut result = na::SMatrix::<f64, N, N>::zeros();
    for i in 0..N {
        for j in 0..N {
            result[(i, j)] = y[(i, j)];
        }
    }

    Ok(result)
}

/// Compute the matrix square root of a general dynamic-sized square matrix.
///
/// This function computes the square root of a general (possibly non-symmetric) square matrix
/// using Denman-Beavers iteration. See [Denman-Beavers iteration](https://en.wikipedia.org/wiki/Square_root_of_a_matrix#By_Denman%E2%80%93Beavers_iteration) for additional details.
///
/// # Arguments
///
/// * `matrix` - A square DMatrix
///
/// # Returns
///
/// * `Result<DMatrix<f64>, String>` - The matrix square root, or an error if:
///   - The matrix is not square
///   - The matrix has complex eigenvalues
///   - The matrix has negative real eigenvalues
///   - The eigendecomposition fails
///
/// # Examples
///
/// ```
/// use nalgebra::DMatrix;
/// use brahe::math::linalg::sqrtm_dmatrix;
///
/// // Test case: A = [33 24; 48 57], sqrtm(A) = [5 2; 4 7]
/// let a = DMatrix::from_row_slice(2, 2, &[33.0, 24.0, 48.0, 57.0]);
/// let sqrt_a = sqrtm_dmatrix(&a).unwrap();
/// let expected = DMatrix::from_row_slice(2, 2, &[5.0, 2.0, 4.0, 7.0]);
/// assert!((sqrt_a.clone() - expected).norm() < 1e-10);
///
/// // Verify: sqrtm(A) * sqrtm(A) = A
/// let reconstructed = &sqrt_a * &sqrt_a;
/// assert!((reconstructed - a).norm() < 1e-10);
/// ```
pub fn sqrtm_dmatrix(matrix: &na::DMatrix<f64>) -> Result<na::DMatrix<f64>, String> {
    // Check matrix is square
    if matrix.nrows() != matrix.ncols() {
        return Err(format!(
            "Matrix must be square, got {}x{}",
            matrix.nrows(),
            matrix.ncols()
        ));
    }

    let n = matrix.nrows();

    // Use Denman-Beavers iteration for computing matrix square root
    // This works for any matrix with eigenvalues in the open right half-plane
    // Iterations: Y_{k+1} = (Y_k + Z_k^{-1}) / 2
    //             Z_{k+1} = (Z_k + Y_k^{-1}) / 2
    // Starting with Y_0 = A, Z_0 = I

    let mut y = matrix.clone();
    let mut z = na::DMatrix::<f64>::identity(n, n);

    const MAX_ITERATIONS: usize = 50;
    const TOLERANCE: f64 = 1e-10;

    for _ in 0..MAX_ITERATIONS {
        // Compute inverses
        let y_inv = y.clone().try_inverse().ok_or_else(|| {
            "Matrix became singular during iteration; cannot compute matrix square root".to_string()
        })?;

        let z_inv = z.clone().try_inverse().ok_or_else(|| {
            "Iteration matrix became singular; cannot compute matrix square root".to_string()
        })?;

        // Update Y and Z
        let y_new = (&y + &z_inv) * 0.5;
        let z_new = (&z + &y_inv) * 0.5;

        // Check convergence: ||Y_{k+1} - Y_k|| < tolerance
        let diff = (&y_new - &y).norm();
        if diff < TOLERANCE {
            y = y_new;
            break;
        }

        y = y_new;
        z = z_new;
    }

    // Verify the result: Y * Y should equal A
    let check = &y * &y;
    let error = (&check - matrix).norm();
    if error > 1e-8 {
        return Err(format!(
            "Matrix square root did not converge to sufficient accuracy (error: {})",
            error
        ));
    }

    Ok(y)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_split_float_f32() {
        assert_eq!(split_float(1.5_f32), (1.0, 0.5));
        assert_eq!(split_float(-1.5_f32), (-1.0, -0.5));
        assert_eq!(split_float(0.0_f32), (0.0, 0.0));
        assert_eq!(split_float(1.0_f32), (1.0, 0.0));
        assert_eq!(split_float(-1.0_f32), (-1.0, 0.0));
    }

    #[test]
    fn test_split_float_f64() {
        assert_eq!(split_float(1.5_f64), (1.0, 0.5));
        assert_eq!(split_float(-1.5_f64), (-1.0, -0.5));
        assert_eq!(split_float(0.0_f64), (0.0, 0.0));
        assert_eq!(split_float(1.0_f64), (1.0, 0.0));
        assert_eq!(split_float(-1.0_f64), (-1.0, 0.0));
    }

    #[test]
    fn test_vector3_from_array() {
        let vec = [1.0, 2.0, 3.0];
        let v = vector3_from_array(vec);
        assert_eq!(v, na::Vector3::new(1.0, 2.0, 3.0));
    }

    #[test]
    fn test_vector6_from_array() {
        let vec = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = vector6_from_array(vec);
        assert_eq!(v, na::SVector::<f64, 6>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0));
    }

    #[test]
    fn test_matrix3_from_array() {
        let mat = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
        let m = matrix3_from_array(&mat);
        assert_eq!(
            m,
            na::SMatrix::<f64, 3, 3>::new(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0)
        );

        assert_eq!(m[(0, 0)], 1.0);
        assert_eq!(m[(0, 1)], 2.0);
        assert_eq!(m[(0, 2)], 3.0);

        assert_eq!(m[(1, 0)], 4.0);
        assert_eq!(m[(1, 1)], 5.0);
        assert_eq!(m[(1, 2)], 6.0);

        assert_eq!(m[(2, 0)], 7.0);
        assert_eq!(m[(2, 1)], 8.0);
        assert_eq!(m[(2, 2)], 9.0);
    }

    #[test]
    fn test_kronecker_delta() {
        assert_eq!(kronecker_delta(0, 0), 1);
        assert_eq!(kronecker_delta(0, 1), 0);
        assert_eq!(kronecker_delta(1, 0), 0);
        assert_eq!(kronecker_delta(1, 1), 1);
    }

    #[test]
    fn test_spd_sqrtm_identity() {
        // Test identity matrix
        let identity = na::SMatrix::<f64, 3, 3>::identity();
        let sqrt_identity = spd_sqrtm(identity).unwrap();

        // sqrt(I) = I
        assert!((sqrt_identity - identity).norm() < 1e-10);
    }

    #[test]
    fn test_spd_sqrtm_diagonal() {
        // Test diagonal matrix
        let diag = na::SMatrix::<f64, 3, 3>::new(4.0, 0.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 16.0);

        let sqrt_diag = spd_sqrtm(diag).unwrap();
        let expected = na::SMatrix::<f64, 3, 3>::new(2.0, 0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0, 4.0);

        assert!((sqrt_diag - expected).norm() < 1e-10);

        // Verify: sqrt(D) * sqrt(D) = D
        let reconstructed = sqrt_diag * sqrt_diag;
        assert!((reconstructed - diag).norm() < 1e-10);
    }

    #[test]
    fn test_spd_sqrtm_covariance() {
        // Test a realistic 6x6 covariance matrix
        let mut cov = na::SMatrix::<f64, 6, 6>::identity() * 100.0;
        // Make it slightly non-diagonal but still symmetric positive-definite
        cov[(0, 1)] = 10.0;
        cov[(1, 0)] = 10.0;
        cov[(2, 3)] = 5.0;
        cov[(3, 2)] = 5.0;

        let sqrt_cov = spd_sqrtm(cov).unwrap();

        // Verify: sqrt(C) * sqrt(C) = C
        let reconstructed = sqrt_cov * sqrt_cov;
        assert!((reconstructed - cov).norm() < 1e-8);

        // Verify sqrt is also symmetric
        let sqrt_cov_t = sqrt_cov.transpose();
        assert!((sqrt_cov - sqrt_cov_t).norm() < 1e-10);
    }

    #[test]
    fn test_spd_sqrtm_error_negative_eigenvalue() {
        // Create a matrix with a negative eigenvalue
        // This is not positive-definite
        let mat = na::SMatrix::<f64, 2, 2>::new(1.0, 0.0, 0.0, -1.0);

        let result = spd_sqrtm(mat);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("negative eigenvalue"));
    }

    #[test]
    fn test_sqrtm_wiki_test_case() {
        // Wikipedia test case: A = [33 24; 48 57], sqrtm(A) = [5 2; 4 7]
        let a = na::SMatrix::<f64, 2, 2>::new(33.0, 24.0, 48.0, 57.0);

        let sqrt_a = sqrtm(a).unwrap();
        let expected = na::SMatrix::<f64, 2, 2>::new(5.0, 2.0, 4.0, 7.0);

        // Check result matches expected
        assert!((sqrt_a - expected).norm() < 1e-10);

        // Verify: sqrtm(A) * sqrtm(A) = A
        let reconstructed = sqrt_a * sqrt_a;
        assert!((reconstructed - a).norm() < 1e-10);
    }

    #[test]
    fn test_sqrtm_3x3_general() {
        // Test a 3x3 non-symmetric matrix with real eigenvalues
        let mat = na::SMatrix::<f64, 3, 3>::new(5.0, 2.0, 1.0, 0.0, 3.0, 1.0, 0.0, 0.0, 2.0);

        let sqrt_mat = sqrtm(mat).unwrap();

        // Verify: sqrtm(M) * sqrtm(M) = M
        let reconstructed = sqrt_mat * sqrt_mat;
        assert!((reconstructed - mat).norm() < 1e-10);
    }

    #[test]
    fn test_sqrtm_symmetric_matches_spd() {
        // For a symmetric positive-definite matrix, both functions should give same result
        let mat = na::SMatrix::<f64, 3, 3>::new(4.0, 2.0, 0.0, 2.0, 3.0, 0.0, 0.0, 0.0, 5.0);

        let sqrt_spd = spd_sqrtm(mat).unwrap();
        let sqrt_gen = sqrtm(mat).unwrap();

        // Results should be very close (within numerical precision)
        assert!((sqrt_spd - sqrt_gen).norm() < 1e-8);
    }

    #[test]
    fn test_sqrtm_error_negative_eigenvalue() {
        // Matrix with negative eigenvalue
        // The Denman-Beavers iteration will fail to converge for this matrix
        let mat = na::SMatrix::<f64, 2, 2>::new(-1.0, 0.0, 0.0, 4.0);

        let result = sqrtm(mat);
        assert!(result.is_err());
        // The error could be about singular matrix or convergence failure
        let err_msg = result.unwrap_err();
        assert!(
            err_msg.contains("singular")
                || err_msg.contains("converge")
                || err_msg.contains("accuracy")
        );
    }

    // =========================================================================
    // sqrtm_dmatrix Tests
    // =========================================================================

    #[test]
    fn test_sqrtm_dmatrix_wiki_test_case() {
        // Wikipedia test case: A = [33 24; 48 57], sqrtm(A) = [5 2; 4 7]
        let a = na::DMatrix::from_row_slice(2, 2, &[33.0, 24.0, 48.0, 57.0]);

        let sqrt_a = sqrtm_dmatrix(&a).unwrap();
        let expected = na::DMatrix::from_row_slice(2, 2, &[5.0, 2.0, 4.0, 7.0]);

        // Check result matches expected
        assert!((&sqrt_a - &expected).norm() < 1e-10);

        // Verify: sqrtm(A) * sqrtm(A) = A
        let reconstructed = &sqrt_a * &sqrt_a;
        assert!((reconstructed - &a).norm() < 1e-10);
    }

    #[test]
    fn test_sqrtm_dmatrix_identity() {
        // sqrtm(I) = I
        let identity = na::DMatrix::<f64>::identity(4, 4);
        let sqrt_identity = sqrtm_dmatrix(&identity).unwrap();
        assert!((sqrt_identity - &identity).norm() < 1e-10);
    }

    #[test]
    fn test_sqrtm_dmatrix_diagonal() {
        // Diagonal matrix: sqrt([4 0; 0 9]) = [2 0; 0 3]
        let diag = na::DMatrix::from_row_slice(2, 2, &[4.0, 0.0, 0.0, 9.0]);
        let sqrt_diag = sqrtm_dmatrix(&diag).unwrap();
        let expected = na::DMatrix::from_row_slice(2, 2, &[2.0, 0.0, 0.0, 3.0]);
        assert!((&sqrt_diag - &expected).norm() < 1e-10);
    }

    #[test]
    fn test_sqrtm_dmatrix_non_square_error() {
        let mat = na::DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let result = sqrtm_dmatrix(&mat);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("square"));
    }
}
