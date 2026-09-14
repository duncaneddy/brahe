/*!
 * Conversions between compile-time sized `SMatrix` and runtime-sized `DMatrix` matrices.
 */

use nalgebra::{DMatrix, SMatrix};

use crate::utils::BraheError;

/// Copies a fixed-size `SMatrix` into a dynamically-sized `DMatrix`.
///
/// # Arguments
/// * `m` - Fixed-size `R x C` matrix to copy
///
/// # Returns
/// * `DMatrix<f64>`: Dynamically-sized `R x C` matrix with the same entries
///
/// # Examples
/// ```
/// use nalgebra::{DMatrix, SMatrix};
/// use brahe::utils::dmatrix_from_smatrix;
///
/// let m = SMatrix::<f64, 2, 3>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let d = dmatrix_from_smatrix(&m);
///
/// assert_eq!(d, DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
/// ```
pub fn dmatrix_from_smatrix<const R: usize, const C: usize>(
    m: &SMatrix<f64, R, C>,
) -> DMatrix<f64> {
    DMatrix::from_iterator(R, C, m.iter().copied())
}

/// Copies a dynamically-sized `DMatrix` into a fixed-size `SMatrix`.
///
/// # Arguments
/// * `m` - Dynamically-sized matrix to copy, must have shape `(R, C)`
///
/// # Returns
/// * `SMatrix<f64, R, C>`: Fixed-size matrix with the same entries
///
/// # Errors
/// Returns `BraheError::Error` if `m` does not have shape `(R, C)`.
///
/// # Examples
/// ```
/// use nalgebra::{DMatrix, SMatrix};
/// use brahe::utils::smatrix_from_dmatrix;
///
/// let d = DMatrix::from_row_slice(2, 3, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
/// let m: SMatrix<f64, 2, 3> = smatrix_from_dmatrix(&d).unwrap();
///
/// assert_eq!(m, SMatrix::<f64, 2, 3>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
/// assert!(smatrix_from_dmatrix::<2, 3>(&DMatrix::zeros(3, 2)).is_err());
/// ```
pub fn smatrix_from_dmatrix<const R: usize, const C: usize>(
    m: &DMatrix<f64>,
) -> Result<SMatrix<f64, R, C>, BraheError> {
    if m.shape() != (R, C) {
        return Err(BraheError::Error(format!(
            "expected a {}x{} matrix, got {}x{}",
            R,
            C,
            m.nrows(),
            m.ncols()
        )));
    }
    Ok(SMatrix::<f64, R, C>::from_iterator(m.iter().copied()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::parallel;

    #[test]
    #[parallel]
    fn test_dmatrix_from_smatrix_smatrix_from_dmatrix_round_trip() {
        let m = SMatrix::<f64, 6, 6>::from_iterator((0..36).map(|v| v as f64));
        let d = dmatrix_from_smatrix(&m);
        let back: SMatrix<f64, 6, 6> = smatrix_from_dmatrix(&d).unwrap();

        assert_eq!(d.nrows(), 6);
        assert_eq!(d.ncols(), 6);
        assert_eq!(back, m);
    }

    #[test]
    #[parallel]
    fn test_dmatrix_from_smatrix_non_square() {
        let m = SMatrix::<f64, 3, 2>::from_row_slice(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
        let d = dmatrix_from_smatrix(&m);

        assert_eq!(d.shape(), (3, 2));
        assert_eq!(
            d,
            DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        );

        let back: SMatrix<f64, 3, 2> = smatrix_from_dmatrix(&d).unwrap();
        assert_eq!(back, m);
    }

    #[test]
    #[parallel]
    fn test_smatrix_from_dmatrix_shape_error() {
        let d = DMatrix::<f64>::zeros(3, 2);
        let err = smatrix_from_dmatrix::<6, 6>(&d).unwrap_err();

        assert_eq!(err.to_string(), "expected a 6x6 matrix, got 3x2");
    }
}
