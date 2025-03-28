/*!
Implements the Butcher tableau for Runge-Kutta methods.
 */

use nalgebra::{SMatrix, SVector};

use crate::utils::BraheError;

/// Defines a Butcher tableau for a Runge-Kutta method.
///
/// The Butcher tableau is a matrix representation of the coefficients of a Runge-Kutta method.
///
pub struct ButcherTableau<const S: usize> {
    pub a: SMatrix<f64, S, S>,
    pub b: SVector<f64, S>,
    pub c: SVector<f64, S>,
}

impl<const S: usize> ButcherTableau<S> {
    /// Create a new Butcher tableau for a Runge-Kutta method.
    ///
    /// # Arguments
    ///
    /// - `a`: The matrix of coefficients for the Runge-Kutta method. Should be a square matrix of size `S`.
    ///       The upper diagonal of the matrix should be zero.
    /// - `b`: The vector of coefficients for the Runge-Kutta method. Should be a vector of size `S`.
    /// - `c`: The vector of coefficients for the Runge-Kutta method. Should be a vector of size `S`. The first
    ///     element of the vector should be zero.
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
    if (b_sum - 1.0).abs() <= 1.0e-16 {
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

    for i in 0..S {
        for j in 0..i {
            if j >= i && a[(i, j)] != 0.0 {
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
}
