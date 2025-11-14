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

/// Embedded Butcher tableau for adaptive Runge-Kutta methods.
///
/// Contains two sets of b coefficients: one for high-order solution and one for
/// low-order solution. The difference provides error estimation for adaptive stepping.
#[derive(Debug)]
pub struct EmbeddedButcherTableau<const S: usize> {
    /// Matrix of RK coefficients (S×S). Lower triangular for explicit methods.
    pub a: SMatrix<f64, S, S>,
    /// High-order output weights (e.g., 5th order in RKF45)
    pub b_high: SVector<f64, S>,
    /// Low-order output weights (e.g., 4th order in RKF45)
    pub b_low: SVector<f64, S>,
    /// Vector of node times (length S)
    pub c: SVector<f64, S>,
    /// Order of high-order method
    pub order_high: usize,
    /// Order of low-order method
    pub order_low: usize,
}

impl<const S: usize> EmbeddedButcherTableau<S> {
    /// Create a new embedded Butcher tableau.
    ///
    /// # Arguments
    /// - `a`: Coefficient matrix
    /// - `b_high`: High-order solution weights
    /// - `b_low`: Low-order solution weights
    /// - `c`: Node times
    /// - `order_high`: Order of high-order method
    /// - `order_low`: Order of low-order method
    ///
    /// # Returns
    /// Result containing the tableau if valid, or error
    pub fn new(
        a: SMatrix<f64, S, S>,
        b_high: SVector<f64, S>,
        b_low: SVector<f64, S>,
        c: SVector<f64, S>,
        order_high: usize,
        order_low: usize,
    ) -> Result<Self, BraheError> {
        // Validate both b vectors
        validate_explicit_butcher_tableau(a, b_high, c)?;
        validate_explicit_butcher_tableau(a, b_low, c)?;

        Ok(Self {
            a,
            b_high,
            b_low,
            c,
            order_high,
            order_low,
        })
    }
}

/// Standard Runge-Kutta 4th order method
pub(crate) const RK4_TABLEAU: ButcherTableau<4> = ButcherTableau {
    a: SMatrix::<f64, 4, 4>::new(
        0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
    ),
    b: SVector::<f64, 4>::new(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0),
    c: SVector::<f64, 4>::new(0.0, 0.5, 0.5, 1.0),
};

/// Runge-Kutta-Fehlberg 4(5) method - 6 stages, 5th/4th order embedded
///
/// Coefficients from Fehlberg (1969). Provides 5th order accurate solution with
/// embedded 4th order solution for error estimation.
pub(crate) const RKF45_TABLEAU: EmbeddedButcherTableau<6> = EmbeddedButcherTableau {
    a: SMatrix::<f64, 6, 6>::new(
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0 / 4.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        3.0 / 32.0,
        9.0 / 32.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1932.0 / 2197.0,
        -7200.0 / 2197.0,
        7296.0 / 2197.0,
        0.0,
        0.0,
        0.0,
        439.0 / 216.0,
        -8.0,
        3680.0 / 513.0,
        -845.0 / 4104.0,
        0.0,
        0.0,
        -8.0 / 27.0,
        2.0,
        -3544.0 / 2565.0,
        1859.0 / 4104.0,
        -11.0 / 40.0,
        0.0,
    ),
    b_high: SVector::<f64, 6>::new(
        16.0 / 135.0,
        0.0,
        6656.0 / 12825.0,
        28561.0 / 56430.0,
        -9.0 / 50.0,
        2.0 / 55.0,
    ),
    b_low: SVector::<f64, 6>::new(
        25.0 / 216.0,
        0.0,
        1408.0 / 2565.0,
        2197.0 / 4104.0,
        -1.0 / 5.0,
        0.0,
    ),
    c: SVector::<f64, 6>::new(0.0, 1.0 / 4.0, 3.0 / 8.0, 12.0 / 13.0, 1.0, 1.0 / 2.0),
    order_high: 5,
    order_low: 4,
};

/// Create Dormand-Prince 5(4) tableau.
///
/// Coefficients from Dormand & Prince (1980). This is MATLAB's ode45 method.
/// More efficient than RKF45 due to better error constants and FSAL property.
/// The 7th stage evaluation becomes the 1st stage of the next step (FSAL).
pub(crate) fn dp54_tableau() -> EmbeddedButcherTableau<7> {
    EmbeddedButcherTableau {
        a: SMatrix::<f64, 7, 7>::from_row_slice(&[
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.0 / 5.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3.0 / 40.0,
            9.0 / 40.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            44.0 / 45.0,
            -56.0 / 15.0,
            32.0 / 9.0,
            0.0,
            0.0,
            0.0,
            0.0,
            19372.0 / 6561.0,
            -25360.0 / 2187.0,
            64448.0 / 6561.0,
            -212.0 / 729.0,
            0.0,
            0.0,
            0.0,
            9017.0 / 3168.0,
            -355.0 / 33.0,
            46732.0 / 5247.0,
            49.0 / 176.0,
            -5103.0 / 18656.0,
            0.0,
            0.0,
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
            0.0,
        ]),
        b_high: SVector::<f64, 7>::from_column_slice(&[
            35.0 / 384.0,
            0.0,
            500.0 / 1113.0,
            125.0 / 192.0,
            -2187.0 / 6784.0,
            11.0 / 84.0,
            0.0,
        ]),
        b_low: SVector::<f64, 7>::from_column_slice(&[
            5179.0 / 57600.0,
            0.0,
            7571.0 / 16695.0,
            393.0 / 640.0,
            -92097.0 / 339200.0,
            187.0 / 2100.0,
            1.0 / 40.0,
        ]),
        c: SVector::<f64, 7>::from_column_slice(&[
            0.0,
            1.0 / 5.0,
            3.0 / 10.0,
            4.0 / 5.0,
            8.0 / 9.0,
            1.0,
            1.0,
        ]),
        order_high: 5,
        order_low: 4,
    }
}

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
