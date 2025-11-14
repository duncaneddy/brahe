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

/// Embedded Butcher tableau for Runge-Kutta-Nyström (RKN) methods.
///
/// RKN methods are specialized for second-order ODEs of the form `y'' = f(t, y)`.
/// They require separate coefficient sets for position and velocity updates,
/// making them more efficient than standard RK methods for such problems.
#[derive(Debug)]
pub struct EmbeddedRKNButcherTableau<const S: usize> {
    /// Matrix of RKN coefficients (S×S). Used for position updates.
    pub a: SMatrix<f64, S, S>,
    /// High-order position weights (e.g., 12th order in RKN1210)
    pub b_pos_high: SVector<f64, S>,
    /// Low-order position weights (e.g., 10th order in RKN1210)
    pub b_pos_low: SVector<f64, S>,
    /// High-order velocity weights (e.g., 12th order in RKN1210)
    pub b_vel_high: SVector<f64, S>,
    /// Low-order velocity weights (e.g., 10th order in RKN1210)
    pub b_vel_low: SVector<f64, S>,
    /// Vector of node times (length S)
    pub c: SVector<f64, S>,
    /// Order of high-order method
    pub order_high: usize,
    /// Order of low-order method
    pub order_low: usize,
}

impl<const S: usize> EmbeddedRKNButcherTableau<S> {
    /// Create a new embedded RKN Butcher tableau.
    ///
    /// # Arguments
    /// - `a`: Coefficient matrix for position
    /// - `b_pos_high`: High-order position solution weights
    /// - `b_pos_low`: Low-order position solution weights
    /// - `b_vel_high`: High-order velocity solution weights
    /// - `b_vel_low`: Low-order velocity solution weights
    /// - `c`: Node times
    /// - `order_high`: Order of high-order method
    /// - `order_low`: Order of low-order method
    ///
    /// # Returns
    /// Result containing the tableau if valid, or error
    #[allow(dead_code)]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        a: SMatrix<f64, S, S>,
        b_pos_high: SVector<f64, S>,
        b_pos_low: SVector<f64, S>,
        b_vel_high: SVector<f64, S>,
        b_vel_low: SVector<f64, S>,
        c: SVector<f64, S>,
        order_high: usize,
        order_low: usize,
    ) -> Result<Self, BraheError> {
        // Validate all b vectors - both position and velocity must sum to 1.0
        validate_explicit_butcher_tableau(a, b_pos_high, c)?;
        validate_explicit_butcher_tableau(a, b_pos_low, c)?;
        validate_explicit_butcher_tableau(a, b_vel_high, c)?;
        validate_explicit_butcher_tableau(a, b_vel_low, c)?;

        Ok(Self {
            a,
            b_pos_high,
            b_pos_low,
            b_vel_high,
            b_vel_low,
            c,
            order_high,
            order_low,
        })
    }
}

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

/// Create RKN12(10) tableau for Runge-Kutta-Nyström integration.
///
/// Coefficients from Dormand, El-Mikkawy, & Prince (1987): "High-Order Embedded
/// Runge-Kutta-Nyström Formulae". 17 stages, 12th/10th order embedded method.
///
/// Based on implementation by Rody Oldenhuis (FEX-RKN1210), used under BSD 2-Clause License.
/// Coefficients obtained from http://www.tampa.phys.ucl.ac.uk/rmat/test/rknint.f
///
/// This is a very high-order method designed for problems with extremely stringent
/// error tolerances (< 1e-10). Optimal for second-order ODEs like orbital mechanics.
///
/// # ⚠️ Experimental
///
/// This tableau and the associated RKN1210 integrator are experimental. While coefficient
/// consistency has been verified, the implementation requires more extensive validation
/// across diverse problem types before production use.
#[allow(clippy::excessive_precision)]
pub(crate) fn rkn1210_tableau() -> EmbeddedRKNButcherTableau<17> {
    EmbeddedRKNButcherTableau {
        // Node times (c)
        c: SVector::<f64, 17>::from_column_slice(&[
            0.0,
            2.0e-2,
            4.0e-2,
            1.0e-1,
            1.333333333333333333333e-1,
            1.6e-1,
            5.0e-2,
            2.0e-1,
            2.5e-1,
            3.333333333333333333333e-1,
            5.0e-1,
            5.555555555555555555556e-1,
            7.5e-1,
            8.571428571428571428571e-1,
            9.452162222720143401300e-1,
            1.0,
            1.0,
        ]),

        // Coefficient matrix A (17x17, stored row-wise then transposed)
        a: {
            let mut a = SMatrix::<f64, 17, 17>::zeros();

            // Row 2
            a[(1, 0)] = 2.0e-4;

            // Row 3
            a[(2, 0)] = 2.666666666666666666667e-4;
            a[(2, 1)] = 5.333333333333333333333e-4;

            // Row 4
            a[(3, 0)] = 2.916666666666666666667e-3;
            a[(3, 1)] = -4.166666666666666666667e-3;
            a[(3, 2)] = 6.25e-3;

            // Row 5
            a[(4, 0)] = 1.646090534979423868313e-3;
            a[(4, 2)] = 5.486968449931412894376e-3;
            a[(4, 3)] = 1.755829903978052126200e-3;

            // Row 6
            a[(5, 0)] = 1.9456e-3;
            a[(5, 2)] = 7.151746031746031746032e-3;
            a[(5, 3)] = 2.912711111111111111111e-3;
            a[(5, 4)] = 7.899428571428571428571e-4;

            // Row 7
            a[(6, 0)] = 5.6640625e-4;
            a[(6, 2)] = 8.809730489417989417989e-4;
            a[(6, 3)] = -4.369212962962962962963e-4;
            a[(6, 4)] = 3.390066964285714285714e-4;
            a[(6, 5)] = -9.946469907407407407407e-5;

            // Row 8
            a[(7, 0)] = 3.083333333333333333333e-3;
            a[(7, 3)] = 1.777777777777777777778e-3;
            a[(7, 4)] = 2.7e-3;
            a[(7, 5)] = 1.578282828282828282828e-3;
            a[(7, 6)] = 1.086060606060606060606e-2;

            // Row 9
            a[(8, 0)] = 3.651839374801129713751e-3;
            a[(8, 2)] = 3.965171714072343066176e-3;
            a[(8, 3)] = 3.197258262930628223501e-3;
            a[(8, 4)] = 8.221467306855435369687e-3;
            a[(8, 5)] = -1.313092695957237983620e-3;
            a[(8, 6)] = 9.771586968064867815626e-3;
            a[(8, 7)] = 3.755769069232833794879e-3;

            // Row 10
            a[(9, 0)] = 3.707241068718500810196e-3;
            a[(9, 2)] = 5.082045854555285980761e-3;
            a[(9, 3)] = 1.174708002175412044736e-3;
            a[(9, 4)] = -2.114762991512699149962e-2;
            a[(9, 5)] = 6.010463698107880812226e-2;
            a[(9, 6)] = 2.010573476850618818467e-2;
            a[(9, 7)] = -2.835075012293358084304e-2;
            a[(9, 8)] = 1.487956891858193275559e-2;

            // Row 11
            a[(10, 0)] = 3.512537656073344153113e-2;
            a[(10, 2)] = -8.615749195138479103406e-3;
            a[(10, 3)] = -5.791448051007916521676e-3;
            a[(10, 4)] = 1.945554823782615842394e0;
            a[(10, 5)] = -3.435123867456513596368e0;
            a[(10, 6)] = -1.093070110747522175839e-1;
            a[(10, 7)] = 2.349638311899516639432e0;
            a[(10, 8)] = -7.560094086870229780272e-1;
            a[(10, 9)] = 1.095289722215692642465e-1;

            // Row 12
            a[(11, 0)] = 2.052779253748249665097e-2;
            a[(11, 2)] = -7.286446764480179917782e-3;
            a[(11, 3)] = -2.115355607961840240693e-3;
            a[(11, 4)] = 9.275807968723522242568e-1;
            a[(11, 5)] = -1.652282484425736679073e0;
            a[(11, 6)] = -2.107956300568656981919e-2;
            a[(11, 7)] = 1.206536432620787154477e0;
            a[(11, 8)] = -4.137144770010661413247e-1;
            a[(11, 9)] = 9.079873982809653759568e-2;
            a[(11, 10)] = 5.355552600533985049169e-3;

            // Row 13
            a[(12, 0)] = -1.432407887554551504589e-1;
            a[(12, 2)] = 1.252870377309181727785e-2;
            a[(12, 3)] = 6.826019163969827128681e-3;
            a[(12, 4)] = -4.799555395574387265502e0;
            a[(12, 5)] = 5.698625043951941433792e0;
            a[(12, 6)] = 7.553430369523645222494e-1;
            a[(12, 7)] = -1.275548785828108371754e-1;
            a[(12, 8)] = -1.960592605111738432891e0;
            a[(12, 9)] = 9.185609056635262409762e-1;
            a[(12, 10)] = -2.388008550528443105348e-1;
            a[(12, 11)] = 1.591108135723421551387e-1;

            // Row 14
            a[(13, 0)] = 8.045019205520489486972e-1;
            a[(13, 2)] = -1.665852706701124517785e-2;
            a[(13, 3)] = -2.141583404262973481173e-2;
            a[(13, 4)] = 1.682723592896246587020e1;
            a[(13, 5)] = -1.117283535717609792679e1;
            a[(13, 6)] = -3.377159297226323741489e0;
            a[(13, 7)] = -1.524332665536084564618e1;
            a[(13, 8)] = 1.717983573821541656202e1;
            a[(13, 9)] = -5.437719239823994645354e0;
            a[(13, 10)] = 1.387867161836465575513e0;
            a[(13, 11)] = -5.925827732652811653477e-1;
            a[(13, 12)] = 2.960387317129735279616e-2;

            // Row 15
            a[(14, 0)] = -9.132967666973580820963e-1;
            a[(14, 2)] = 2.411272575780517839245e-3;
            a[(14, 3)] = 1.765812269386174198207e-2;
            a[(14, 4)] = -1.485164977972038382461e1;
            a[(14, 5)] = 2.158970867004575600308e0;
            a[(14, 6)] = 3.997915583117879901153e0;
            a[(14, 7)] = 2.843415180023223189845e1;
            a[(14, 8)] = -2.525936435494159843788e1;
            a[(14, 9)] = 7.733878542362237365534e0;
            a[(14, 10)] = -1.891302894847867461038e0;
            a[(14, 11)] = 1.001484507022471780367e0;
            a[(14, 12)] = 4.641199599109051905105e-3;
            a[(14, 13)] = 1.121875502214895703398e-2;

            // Row 16
            a[(15, 0)] = -2.751962972055939382061e-1;
            a[(15, 2)] = 3.661188877915492013423e-2;
            a[(15, 3)] = 9.789519688231562624651e-3;
            a[(15, 4)] = -1.229306234588621030421e1;
            a[(15, 5)] = 1.420722645393790269429e1;
            a[(15, 6)] = 1.586647690678953683225e0;
            a[(15, 7)] = 2.457773532759594543903e0;
            a[(15, 8)] = -8.935193694403271905523e0;
            a[(15, 9)] = 4.373672731613406948393e0;
            a[(15, 10)] = -1.834718176544949163043e0;
            a[(15, 11)] = 1.159208528906149120781e0;
            a[(15, 12)] = -1.729025316538392215180e-2;
            a[(15, 13)] = 1.932597790446076667276e-2;
            a[(15, 14)] = 5.204442937554993111849e-3;

            // Row 17
            a[(16, 0)] = 1.307639184740405758800e0;
            a[(16, 2)] = 1.736410918974584186709e-2;
            a[(16, 3)] = -1.854445645426579502436e-2;
            a[(16, 4)] = 1.481152203286772689685e1;
            a[(16, 5)] = 9.383176308482470907879e0;
            a[(16, 6)] = -5.228426199944542254147e0;
            a[(16, 7)] = -4.895128052584765080401e1;
            a[(16, 8)] = 3.829709603433792256258e1;
            a[(16, 9)] = -1.058738133697597970916e1;
            a[(16, 10)] = 2.433230437622627635851e0;
            a[(16, 11)] = -1.045340604257544428487e0;
            a[(16, 12)] = 7.177320950867259451982e-2;
            a[(16, 13)] = 2.162210970808278269055e-3;
            a[(16, 14)] = 7.009595759602514236993e-3;

            a
        },

        // High-order position weights (Bhat)
        b_pos_high: SVector::<f64, 17>::from_column_slice(&[
            1.212786851718541497689e-2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            8.629746251568874443638e-2,
            2.525469581187147194323e-1,
            -1.974186799326823033583e-1,
            2.031869190789725908093e-1,
            -2.077580807771491661219e-2,
            1.096780487450201362501e-1,
            3.806513252646650573449e-2,
            1.163406880432422964409e-2,
            4.658029704024878686936e-3,
            0.0,
            0.0,
        ]),

        // High-order velocity weights (Bphat)
        b_vel_high: SVector::<f64, 17>::from_column_slice(&[
            1.212786851718541497689e-2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            9.083943422704078361724e-2,
            3.156836976483933992904e-1,
            -2.632249065769097378111e-1,
            3.047803786184588862139e-1,
            -4.155161615542983322439e-2,
            2.467756096762953065628e-1,
            1.522605301058660229380e-1,
            8.143848163026960750865e-2,
            8.502571193890811280080e-2,
            -9.155189630077962873141e-3,
            2.5e-2,
        ]),

        // Low-order position weights (B)
        b_pos_low: SVector::<f64, 17>::from_column_slice(&[
            1.700870190700699175275e-2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            7.225933593083140694886e-2,
            3.720261773267530453882e-1,
            -4.018211450093035214393e-1,
            3.354550683013516666966e-1,
            -1.313065010753318084303e-1,
            1.894319066160486527227e-1,
            2.684080204002904790537e-2,
            1.630566560591792389352e-2,
            3.799988356696594561666e-3,
            0.0,
            0.0,
        ]),

        // Low-order velocity weights (Bp)
        b_vel_low: SVector::<f64, 17>::from_column_slice(&[
            1.700870190700699175275e-2,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            7.606245887455937573564e-2,
            4.650327216584413067353e-1,
            -5.357615266790713619191e-1,
            5.031826024520275000449e-1,
            -2.626130021506636168606e-1,
            4.262217898861094686260e-1,
            1.073632081601161916215e-1,
            1.141396592414254672546e-1,
            6.936338665004867700906e-2,
            2.0e-2,
            0.0,
        ]),

        order_high: 12,
        order_low: 10,
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
