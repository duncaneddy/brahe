/*!
Implements the Butcher tableau for Runge-Kutta methods.
 */

use nalgebra::{SMatrix, SVector};

use crate::utils::BraheError;

/// Defines a Butcher tableau for a Runge-Kutta method.
///
/// The Butcher tableau is a matrix representation of the coefficients of a Runge-Kutta method.
///
#[derive(Debug)]
pub struct ButcherTableau<const S: usize> {
    /// Matrix of RK coefficients (S×S). Lower triangular for explicit methods.
    /// Element a[i,j] weights the j-th stage derivative in computing the i-th stage.
    pub a: SMatrix<f64, S, S>,
    /// Vector of output weights (length S). Used to combine stage derivatives for final step.
    /// Must sum to 1.0 for consistency. Each element weights corresponding stage's contribution.
    pub b: SVector<f64, S>,
    /// Vector of node times (length S). Fractional timesteps where stages are evaluated.
    /// First element must be 0.0. Element c[i] determines time offset for i-th stage.
    pub c: SVector<f64, S>,
}

impl<const S: usize> ButcherTableau<S> {
    /// Create a new Butcher tableau for a Runge-Kutta method.
    ///
    /// # Arguments
    ///
    /// - `a`: The matrix of coefficients for the Runge-Kutta method. Should be a square matrix of size `S`.
    ///   The upper diagonal of the matrix should be zero.
    /// - `b`: The vector of coefficients for the Runge-Kutta method. Should be a vector of size `S`.
    /// - `c`: The vector of coefficients for the Runge-Kutta method. Should be a vector of size `S`. The first
    ///   element of the vector should be zero.
    ///
    /// # Returns
    ///
    /// - A `Result` containing the Butcher tableau if the tableau is valid, or a `BraheError` if the tableau is invalid.
    ///
    /// # Example
    ///
    /// ```
    /// use nalgebra::{SMatrix, SVector};
    /// use brahe::integrators::ButcherTableau;
    ///
    /// // Define the Butcher tableau for the standard Runge-Kutta 4th order method
    /// let a = SMatrix::<f64, 4, 4>::new(
    ///     0.0, 0.0, 0.0, 0.0,
    ///     0.5, 0.0, 0.0, 0.0,
    ///     0.0, 0.5, 0.0, 0.0,
    ///     0.0, 0.0, 1.0, 0.0,
    /// );
    /// let b = SVector::<f64, 4>::new(1.0/6.0, 1.0/3.0, 1.0/3.0, 1.0/6.0);
    /// let c = SVector::<f64, 4>::new(0.0, 0.5, 0.5, 1.0);
    ///
    /// let rk4_bt = ButcherTableau::new(a, b, c);
    /// assert!(rk4_bt.is_ok());
    /// ```
    pub fn new(
        a: SMatrix<f64, S, S>,
        b: SVector<f64, S>,
        c: SVector<f64, S>,
    ) -> Result<Self, BraheError> {
        // Validate that the Butcher tableau is consistent
        validate_explicit_butcher_tableau(a, b, c)?;

        Ok(Self { a, b, c })
    }

    /// Validate the Butcher tableau for a Runge-Kutta method.
    ///
    /// This is a convenience method to implement tests that validate pre-defined Butcher tableaus.
    #[allow(dead_code)] // For some reason, this is being flagged as dead code
    fn validate(&self) -> Result<(), BraheError> {
        validate_explicit_butcher_tableau(self.a, self.b, self.c)
    }
}

/// Validate a Butcher tableau for a Runge-Kutta method.
///
/// The Butcher tableau is a matrix representation of the coefficients of a Runge-Kutta method.
/// We perform some basic validation to ensure that the Butcher tableau is consistent and valid.
///
/// # Arguments
///
/// - `a`: The matrix of coefficients for the Runge-Kutta method.
/// - `b`: The vector of coefficients for the Runge-Kutta method.
/// - `c`: The vector of coefficients for the Runge-Kutta method.
///
/// # Returns
///
/// - A `Result` containing `()` if the Butcher tableau is valid, or a `BraheError` if the Butcher tableau is invalid.
fn validate_explicit_butcher_tableau<const S: usize>(
    a: SMatrix<f64, S, S>,
    b: SVector<f64, S>,
    c: SVector<f64, S>,
) -> Result<(), BraheError> {
    // Validate that the Butcher tableau is consistent
    let b_sum = b.sum();
    if (b_sum - 1.0).abs() > 1.0e-14 {
        return Err(BraheError::Error(format!(
            "Invalid Butcher tableau: sum of b coefficients must be 1.0. Found {}",
            b_sum
        )));
    }

    if c[0] != 0.0 {
        return Err(BraheError::Error(format!(
            "Invalid Butcher tableau: c[0] must be 0.0. Found {}",
            c[0]
        )));
    }

    // Check that upper diagonal of 'a' matrix is all zeros (explicit method)
    for i in 0..S {
        for j in (i + 1)..S {
            if a[(i, j)] != 0.0 {
                // Return immediately if we found a non-zero value in the upper diagonal
                return Err(BraheError::Error(
                    "Invalid Butcher tableau: upper-diagonal of a must be 0.0.".to_string(),
                ));
            }
        }
    }

    Ok(())
}

/// Standard Runge-Kutta 4th order method
pub(crate) const RK4_TABLEAU: ButcherTableau<4> = ButcherTableau {
    a: SMatrix::<f64, 4, 4>::new(
        0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ),
    b: SVector::<f64, 4>::new(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0),
    c: SVector::<f64, 4>::new(0.0, 0.5, 0.5, 1.0),
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_butcher_tableau() {
        let a = SMatrix::<f64, 4, 4>::new(
            0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        );

        let b = SVector::<f64, 4>::new(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0);
        let c = SVector::<f64, 4>::new(0.0, 0.5, 0.5, 1.0);

        let bt = ButcherTableau::new(a, b, c);

        assert!(bt.is_ok());
    }

    #[test]
    fn test_validate_rk4_butcher_tableau() {
        assert!(RK4_TABLEAU.validate().is_ok());
    }

    #[test]
    fn test_butcher_tableau_invalid_b_sum() {
        // Test b coefficients that don't sum to 1.0
        let a = SMatrix::<f64, 4, 4>::new(
            0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        );
        let b = SVector::<f64, 4>::new(0.2, 0.2, 0.2, 0.2); // Sum = 0.8, not 1.0
        let c = SVector::<f64, 4>::new(0.0, 0.5, 0.5, 1.0);

        let result = ButcherTableau::new(a, b, c);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("sum of b coefficients")
        );
    }

    #[test]
    fn test_butcher_tableau_invalid_c_first() {
        // Test c[0] that is not 0.0
        let a = SMatrix::<f64, 4, 4>::new(
            0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        );
        let b = SVector::<f64, 4>::new(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0);
        let c = SVector::<f64, 4>::new(0.1, 0.5, 0.5, 1.0); // c[0] should be 0.0

        let result = ButcherTableau::new(a, b, c);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("c[0] must be 0.0"));
    }

    #[test]
    fn test_butcher_tableau_invalid_upper_diagonal() {
        // Test upper diagonal of 'a' that contains non-zero
        let a = SMatrix::<f64, 4, 4>::new(
            0.0, 0.1, 0.0, 0.0, // Non-zero at position (0,1) - upper diagonal
            0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        );
        let b = SVector::<f64, 4>::new(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0);
        let c = SVector::<f64, 4>::new(0.0, 0.5, 0.5, 1.0);

        let result = ButcherTableau::new(a, b, c);
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("upper-diagonal of a must be 0.0")
        );
    }

    #[test]
    fn test_butcher_tableau_valid_with_correct_values() {
        // Verify that a valid tableau passes all checks
        let a = SMatrix::<f64, 4, 4>::new(
            0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        );
        let b = SVector::<f64, 4>::new(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0);
        let c = SVector::<f64, 4>::new(0.0, 0.5, 0.5, 1.0);

        let result = ButcherTableau::new(a, b, c);
        assert!(result.is_ok());
    }
}
