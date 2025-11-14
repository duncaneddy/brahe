/*!
 * Math utility functions and type definitions.
 */

use nalgebra as na;
use nalgebra::linalg::SymmetricEigen;
use num_traits::float::Float;

use crate::{AngleFormat, constants};

/// 3-dimensional static vector type for Cartesian state vectors
pub type SVector3 = na::SVector<f64, 3>;

/// 6-dimensional static vector type for Cartesian state vectors
pub type SVector6 = na::SVector<f64, 6>;

/// 3x3 static matrix type for rotation matrices and transformations
pub type SMatrix3 = na::SMatrix<f64, 3, 3>;

/// 6x6 static matrix type for rotation matrices and transformations
pub type SMatrix6 = na::SMatrix<f64, 6, 6>;

/// Convert a number to radians, if `as_degrees` is `true` the number is assumed to be in degrees.
/// If `false` the number is assumed to be in radians already and is passed through.
///
/// # Arguments
/// - `num`: The number to convert.
/// - `as_degrees`: If `true`, the number is assumed to be in degrees.
///
/// # Returns
/// - `f64`: The number in radians.
///
/// # Examples
/// ```
/// use brahe::utils::math::from_degrees;
///
/// assert!(from_degrees(180.0, true) == std::f64::consts::PI);
/// assert!(from_degrees(std::f64::consts::PI, false) == std::f64::consts::PI);
/// ```
pub fn from_degrees(num: f64, as_degrees: bool) -> f64 {
    if as_degrees {
        num * constants::DEG2RAD
    } else {
        num
    }
}

/// Transform angular value, to desired output format. Input is expected to be in radians.  If `as_degrees` is `true`
/// the number will be converted to be in degrees. If false, the value will be directly passed
/// through and returned in radians.
///
/// # Arguments
/// - `num`: The number to convert.
/// - `as_degrees`: If `true`, the number will be converted to degrees.
///
/// # Returns
/// - `f64`: The number in degrees.
///
/// # Examples
/// ```
/// use std::f64::consts::PI;
/// use brahe::utils::math::to_degrees;
///
/// assert!(to_degrees(PI, false) == PI);
/// assert!(to_degrees(PI, true) == 180.0);
/// ```
pub fn to_degrees(num: f64, as_degrees: bool) -> f64 {
    if as_degrees {
        num * constants::RAD2DEG
    } else {
        num
    }
}

/// Convert orbital elements to degrees if `angle_format` is `Degrees`, otherwise pass through.
///
/// # Arguments
/// - `oe`: Orbital elements vector [a, e, i, RAAN, arg_perigee, mean_anomaly]
/// - `angle_format`: Angle format of the input.
///
/// # Returns
/// - `SVector6`: Orbital elements with angles in degrees if specified.
pub fn oe_to_degrees(oe: SVector6, angle_format: AngleFormat) -> SVector6 {
    match angle_format {
        AngleFormat::Radians => SVector6::new(
            oe[0],
            oe[1],
            oe[2] * constants::RAD2DEG,
            oe[3] * constants::RAD2DEG,
            oe[4] * constants::RAD2DEG,
            oe[5] * constants::RAD2DEG,
        ),
        AngleFormat::Degrees => oe,
    }
}

/// Convert orbital elements to radians if `angle_format` is `Degrees`, otherwise pass through.
///
/// # Arguments
/// - `oe`: Orbital elements vector [a, e, i, RAAN, arg_perigee, mean_anomaly]
/// - `angle_format`: Angle format of the input.
///
/// # Returns
/// - `SVector6`: Orbital elements with angles in radians if specified.
pub fn oe_to_radians(oe: SVector6, angle_format: AngleFormat) -> SVector6 {
    match angle_format {
        AngleFormat::Degrees => SVector6::new(
            oe[0],
            oe[1],
            oe[2] * constants::DEG2RAD,
            oe[3] * constants::DEG2RAD,
            oe[4] * constants::DEG2RAD,
            oe[5] * constants::DEG2RAD,
        ),
        AngleFormat::Radians => oe,
    }
}

/// Split a floating point number into its integer and fractional parts.
///
/// # Arguments
/// - `num`: The number to split. Can be `f32` or `f64`.
///
/// # Examples
/// ```
/// use brahe::utils::math::split_float;
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
/// use brahe::utils::math::vector3_from_array;
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
/// use brahe::utils::math::vector6_from_array;
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
/// use brahe::utils::math::matrix3_from_array;
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
/// use brahe::utils::math::kronecker_delta;
///
/// assert_eq!(kronecker_delta(0, 0), 1);
/// assert_eq!(kronecker_delta(0, 1), 0);
/// assert_eq!(kronecker_delta(1, 0), 0);
/// assert_eq!(kronecker_delta(1, 1), 1);
/// ```
pub fn kronecker_delta(i: usize, j: usize) -> u8 {
    if i == j { 1 } else { 0 }
}

/// Wrap an angle to the range \[0, 2π\].
///
/// # Arguments
///
/// - `angle`: The angle to wrap.
///
/// # Returns
///
/// - `f64`: The wrapped angle.
///
/// # Examples
///
/// ```
/// use brahe::utils::math::wrap_to_2pi;
///
/// assert_eq!(wrap_to_2pi(2.0 * std::f64::consts::PI), 0.0);
/// assert_eq!(wrap_to_2pi(3.0 * std::f64::consts::PI), std::f64::consts::PI);
/// ```
pub fn wrap_to_2pi(angle: f64) -> f64 {
    let two_pi = 2.0 * std::f64::consts::PI;
    angle.rem_euclid(two_pi)
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
/// use brahe::utils::math::spd_sqrtm;
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
/// use brahe::utils::math::sqrtm;
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

#[cfg(test)]
mod tests {
    use approx::assert_abs_diff_eq;
    use std::f64::consts::PI;

    use super::*;

    #[test]
    fn test_from_degrees() {
        assert_eq!(from_degrees(180.0, true), PI);
        assert_eq!(from_degrees(PI, false), PI);
    }

    #[test]
    fn test_to_degrees() {
        assert_eq!(to_degrees(PI, false), PI);
        assert_eq!(to_degrees(PI, true), 180.0);
    }

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
    fn test_wrap_to_2pi() {
        assert_eq!(wrap_to_2pi(PI), PI);
        assert_eq!(wrap_to_2pi(2.0 * PI), 0.0);
        assert_eq!(wrap_to_2pi(3.0 * PI), PI);

        assert_eq!(wrap_to_2pi(-PI), PI);
        assert_eq!(wrap_to_2pi(-3.0 / 2.0 * PI), PI / 2.0);
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

    #[test]
    fn test_oe_to_radians() {
        let oe_deg = SVector6::new(7000.0, 0.001, 45.0, 120.0, 90.0, 30.0);
        let oe_rad = oe_to_radians(oe_deg, AngleFormat::Degrees);

        assert_abs_diff_eq!(oe_rad[0], 7000.0);
        assert_abs_diff_eq!(oe_rad[1], 0.001);
        assert_abs_diff_eq!(oe_rad[2], 45.0 * constants::DEG2RAD);
        assert_abs_diff_eq!(oe_rad[3], 120.0 * constants::DEG2RAD);
        assert_abs_diff_eq!(oe_rad[4], 90.0 * constants::DEG2RAD);
        assert_abs_diff_eq!(oe_rad[5], 30.0 * constants::DEG2RAD);

        // Test with Radians input
        let oe_rad_input =
            SVector6::new(7000.0, 0.001, PI / 4.0, 2.0 * PI / 3.0, PI / 2.0, PI / 6.0);
        let oe_rad_output = oe_to_radians(oe_rad_input, AngleFormat::Radians);

        assert_eq!(oe_rad_output, oe_rad_input);
    }

    #[test]
    fn test_oe_to_degrees() {
        let oe_rad = SVector6::new(7000.0, 0.001, PI / 4.0, 2.0 * PI / 3.0, PI / 2.0, PI / 6.0);
        let oe_deg = oe_to_degrees(oe_rad, AngleFormat::Radians);

        assert_abs_diff_eq!(oe_deg[0], 7000.0);
        assert_abs_diff_eq!(oe_deg[1], 0.001);
        assert_abs_diff_eq!(oe_deg[2], PI / 4.0 * constants::RAD2DEG);
        assert_abs_diff_eq!(oe_deg[3], 2.0 * PI / 3.0 * constants::RAD2DEG);
        assert_abs_diff_eq!(oe_deg[4], PI / 2.0 * constants::RAD2DEG);
        assert_abs_diff_eq!(oe_deg[5], PI / 6.0 * constants::RAD2DEG);

        // Test with Degrees input
        let oe_deg_input = SVector6::new(7000.0, 0.001, 45.0, 120.0, 90.0, 30.0);
        let oe_deg_output = oe_to_degrees(oe_deg_input, AngleFormat::Degrees);

        assert_eq!(oe_deg_output, oe_deg_input);
    }
}
