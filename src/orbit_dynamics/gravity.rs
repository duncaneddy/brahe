/*!
Implement central-body gravity force models.
 */

use std::io::{BufRead, BufReader, Read};
use std::path::Path;
use std::sync::{Arc, RwLock, RwLockReadGuard};

use nalgebra::{DMatrix, Vector3};

use crate::math::{SMatrix3, traits::IntoPosition};
use once_cell::sync::Lazy;

use crate::math::kronecker_delta;
use crate::utils::BraheError;

/// Packaged EGM2008_360 Data File
static PACKAGED_EGM2008_360: &[u8] = include_bytes!("../../data/gravity_models/EGM2008_360.gfc");

/// Packaged GGM05S Data File
static PACKAGED_GGM05S: &[u8] = include_bytes!("../../data/gravity_models/GGM05S.gfc");

/// Packaged JGM3
static PACKAGED_JGM3: &[u8] = include_bytes!("../../data/gravity_models/JGM3.gfc");
static GLOBAL_GRAVITY_MODEL: Lazy<Arc<RwLock<Box<GravityModel>>>> =
    Lazy::new(|| Arc::new(RwLock::new(Box::new(GravityModel::new()))));

/// Set the global gravity model to a new gravity model. A global gravity model is useful as it
/// allows for a single gravity model to be used throughout a program. This is useful when multiple
/// objects are being propagated, as it allows for the gravity model to only be loaded into memory
/// once, reducing memory usage and improving performance.
///
/// # Arguments
///
/// - `gravity_model` : New gravity model to set as the global gravity model.
///
/// # Examples
///
/// ```
/// use brahe::gravity::{GravityModel, set_global_gravity_model, DefaultGravityModel};
///
/// let gravity_model = GravityModel::from_default(DefaultGravityModel::EGM2008_360);
/// set_global_gravity_model(gravity_model);
/// ```
pub fn set_global_gravity_model(gravity_model: GravityModel) {
    **GLOBAL_GRAVITY_MODEL.write().unwrap() = gravity_model;
}

/// Get the global gravity model.
///
/// # Returns
///
/// - `GravityModel`: Gravity model object.
///
/// # Examples
///
/// ```
/// use brahe::gravity::{GravityModel, set_global_gravity_model, get_global_gravity_model, DefaultGravityModel};
///
/// let gravity_model = GravityModel::from_default(DefaultGravityModel::EGM2008_360);
/// set_global_gravity_model(gravity_model);
///
/// let model = get_global_gravity_model();
///
/// assert_eq!(model.model_name, "EGM2008");
/// ```
pub fn get_global_gravity_model() -> RwLockReadGuard<'static, Box<GravityModel>> {
    GLOBAL_GRAVITY_MODEL.read().unwrap()
}

/// Helper function to aid in denormalization of gravity field coefficients.
/// This method computes the factorial ratio (n-m)!/(n+m)! in an efficient
/// manner, without computing the full factorial products.
fn factorial_product(n: usize, m: usize) -> f64 {
    let mut p = 1.0;

    // TODO: Confirm range termination of n+m+1 vs n+m
    for i in n - m + 1..n + m + 1 {
        p /= i as f64;
    }

    p
}

/// Compute the acceleration due to point-mass gravity.
///
/// This function accepts either a 3D position vector or a 6D state vector for `r_object`.
/// When a state vector is provided, only the position component is used.
///
/// # Arguments
///
/// - `r_object`: Position vector of the object, or state vector (position + velocity).
/// - `r_central_body`: Position vector of the central body. If the central body is at the origin, this is the zero vector.
/// - `gm`: Product of the gravitational parameter and the mass of the central body.
///
/// # Returns
///
/// - `a_grav` : Acceleration due to gravity of the central body.
///
/// # Examples
///
/// Using a position vector:
/// ```
/// use brahe::constants::{R_EARTH, GM_EARTH};
/// use brahe::orbit_dynamics::accel_point_mass_gravity;
/// use nalgebra::Vector3;
///
/// let r_object = Vector3::new(R_EARTH, 0.0, 0.0);
/// let r_central_body = Vector3::new(0.0, 0.0, 0.0);
///
/// let a_grav = accel_point_mass_gravity(r_object, r_central_body, GM_EARTH);
///
/// // Acceleration should be in the negative x-direction and magnitude should be GM_EARTH / R_EARTH^2
/// // Roughly -9.81 m/s^2
/// assert!((a_grav - Vector3::new(-GM_EARTH / R_EARTH.powi(2), 0.0, 0.0)).norm() < 1e-12);
/// ```
///
/// Using a state vector:
/// ```
/// use brahe::constants::{R_EARTH, GM_EARTH};
/// use brahe::orbit_dynamics::accel_point_mass_gravity;
/// use nalgebra::{SVector, Vector3};
///
/// let x_object = SVector::<f64, 6>::new(R_EARTH, 0.0, 0.0, 0.0, 7500.0, 0.0);
/// let r_central_body = Vector3::new(0.0, 0.0, 0.0);
///
/// let a_grav = accel_point_mass_gravity(x_object, r_central_body, GM_EARTH);
///
/// // Acceleration should be in the negative x-direction
/// assert!((a_grav - Vector3::new(-GM_EARTH / R_EARTH.powi(2), 0.0, 0.0)).norm() < 1e-12);
/// ```
///
/// # References
///
/// - TODO: Add references
pub fn accel_point_mass_gravity<P: IntoPosition>(
    r_object: P,
    r_central_body: Vector3<f64>,
    gm: f64,
) -> Vector3<f64> {
    let r_obj = r_object.position();
    let d = r_obj - r_central_body;

    let d_norm = d.norm();
    let r_central_body_norm = r_central_body.norm();

    if r_central_body_norm != 0.0 {
        -gm * (d / d_norm.powi(3) + r_central_body / r_central_body_norm.powi(3))
    } else {
        -gm * d / d_norm.powi(3)
    }
}

/// Enumeration of the tide system used in a gravity model.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum GravityModelTideSystem {
    /// Zero-tide system: includes permanent tidal deformation from Sun and Moon.
    /// C₂₀ coefficient includes indirect effect of Earth's centrifugal potential.
    ZeroTide,
    /// Tide-free system: permanent tidal effects removed. Most commonly used in
    /// modern gravity models. Represents Earth's shape without tidal deformation.
    TideFree,
    /// Mean-tide system: includes time-averaged permanent tide. Historical convention
    /// used in older gravity models and some geodetic applications.
    MeanTide,
    /// Unknown or unspecified tide system. Default value when tide system is not declared.
    Unknown,
}

/// Enumeration of the error handling used in a gravity model.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum GravityModelErrors {
    /// No error estimates provided with gravity coefficients. Model provides only
    /// nominal C and S coefficient values without uncertainty information.
    No,
    /// Calibrated error estimates: empirically derived uncertainties based on
    /// comparison with independent data sources or solution comparisons.
    Calibrated,
    /// Formal errors: statistical uncertainties from least-squares adjustment or
    /// estimation process. May underestimate true uncertainties.
    Formal,
    /// Both calibrated and formal error estimates provided, allowing comparison
    /// between statistical and empirical uncertainty assessments.
    CalibratedAndFormal,
}

/// Enumeration of the normalization used in a gravity model.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum GravityModelNormalization {
    /// Fully normalized spherical harmonics (4π normalization). Standard in modern
    /// gravity models (EGM2008, GRACE, etc.). Coefficients have similar magnitudes.
    FullyNormalized,
    /// Unnormalized spherical harmonics. Used in older models and some theoretical
    /// applications. Coefficients decrease rapidly with increasing degree/order.
    Unnormalized,
}

/// Enumeration of the default gravity models available in Brahe.
#[derive(Debug, PartialEq, Clone, Copy)]
pub enum DefaultGravityModel {
    /// Earth Gravitational Model 2008, truncated to degree/order 360. High-accuracy
    /// global model developed by NGA. Best for precision orbit determination.
    EGM2008_360,
    /// Goddard Earth Model from GRACE mission, degree/order 180. Derived from
    /// satellite gravity measurements. Good balance of accuracy and computation speed.
    GGM05S,
    /// Joint Gravity Model 3, degree/order 70. Legacy model from 1990s. Included
    /// for compatibility and applications not requiring modern accuracy.
    JGM3,
}

/// The `GravityModel` struct is for storing spherical harmonic gravity models.
///
/// # Fields
///
/// - `data` : Matrix of gravity model coefficients. The matrix is (n_max + 1) x (m_max + 1) in size.
/// - `tide_system` : Tide system used in the gravity model.
/// - `n_max` : Maximum degree of the gravity model.
/// - `m_max` : Maximum order of the gravity model.
/// - `gm` : Gravitational parameter of the central body. Units are m^3/s^2.
/// - `radius` : Radius of the central body. Units are meters.
/// - `model_name` : Name of the gravity model.
/// - `model_errors` : Error handling used in the gravity model.
/// - `normalization` : Normalization used in the gravity model.
pub struct GravityModel {
    data: DMatrix<f64>,
    /// Tide system convention used in the model (zero-tide, tide-free, or mean-tide).
    pub tide_system: GravityModelTideSystem,
    /// Maximum degree (n) of spherical harmonic expansion. Higher degrees capture
    /// finer spatial resolution of gravity field. Typical values: 70-2190.
    pub n_max: usize,
    /// Maximum order (m) of spherical harmonic expansion. Often equal to n_max.
    /// Determines resolution in longitude direction.
    pub m_max: usize,
    /// Gravitational parameter (GM) of the central body. Product of gravitational
    /// constant and body mass. Units: m³/s².
    pub gm: f64,
    /// Reference radius of the central body used for normalization of spherical harmonics.
    /// For Earth models, typically the equatorial radius. Units: meters.
    pub radius: f64,
    /// Human-readable name of the gravity model (e.g., "EGM2008", "GGM05S").
    pub model_name: String,
    /// Type of error/uncertainty estimates included with the model coefficients.
    pub model_errors: GravityModelErrors,
    /// Normalization convention used for spherical harmonic coefficients (fully normalized or unnormalized).
    pub normalization: GravityModelNormalization,
}

impl GravityModel {
    /// Create a new gravity model. This is an internal method used to aid in initializing the
    /// global gravity model. It should not be used directly.
    fn new() -> Self {
        Self {
            data: DMatrix::zeros(1, 1),
            tide_system: GravityModelTideSystem::Unknown,
            n_max: 0,
            m_max: 0,
            gm: 0.0,
            radius: 0.0,
            model_name: String::from("Unknown"),
            model_errors: GravityModelErrors::No,
            normalization: GravityModelNormalization::FullyNormalized,
        }
    }

    fn from_bufreader<T: Read>(reader: BufReader<T>) -> Result<Self, BraheError> {
        let mut lines = reader.lines();

        // Read the header
        let mut line = lines.next().unwrap()?;

        let mut model_name = String::from("Unknown");
        let mut gm = 0.0;
        let mut radius = 0.0;
        let mut n_max = 0;
        let mut m_max = 0;
        let mut tide_system = GravityModelTideSystem::Unknown;
        let mut model_errors = GravityModelErrors::No;
        let mut normalization = GravityModelNormalization::FullyNormalized;

        while !line.starts_with("end_of_head") {
            // Read the header line
            line = lines.next().unwrap()?;

            // Parse the header
            if line.starts_with("modelname") {
                model_name = String::from(line.split_whitespace().last().unwrap());
            } else if line.starts_with("earth_gravity_constant") {
                gm = line.split_whitespace().last().unwrap().parse::<f64>()?;
            } else if line.starts_with("radius") {
                radius = line.split_whitespace().last().unwrap().parse::<f64>()?;
            } else if line.starts_with("max_degree") {
                n_max = line.split_whitespace().last().unwrap().parse::<usize>()?;
                m_max = n_max;
            } else if line.starts_with("tide_system") {
                tide_system = match line.split_whitespace().last().unwrap() {
                    "zero_tide" => GravityModelTideSystem::ZeroTide,
                    "tide_free" => GravityModelTideSystem::TideFree,
                    "mean_tide" => GravityModelTideSystem::MeanTide,
                    _ => GravityModelTideSystem::Unknown,
                };
            } else if line.starts_with("errors") {
                model_errors = match line.split_whitespace().last().unwrap() {
                    "no" => GravityModelErrors::No,
                    "calibrated" => GravityModelErrors::Calibrated,
                    "formal" => GravityModelErrors::Formal,
                    "calibrated_and_formal" => GravityModelErrors::CalibratedAndFormal,
                    _ => {
                        return Err(BraheError::ParseError(format!(
                            "Invalid model_errors value: \"{}\". Expected \"no\", \"calibrated\", \"formal\", or \"calibrated_and_formal\".",
                            line.split_whitespace().last().unwrap()
                        )));
                    }
                };
            } else if line.starts_with("normalization") {
                normalization = match line.split_whitespace().last().unwrap() {
                    "fully_normalized" => GravityModelNormalization::FullyNormalized,
                    "unnormalized" => GravityModelNormalization::Unnormalized,
                    _ => {
                        return Err(BraheError::ParseError(format!(
                            "Invalid normalization value: \"{}\". Expected \"fully_normalized\" or \"unnormalized\".",
                            line.split_whitespace().last().unwrap()
                        )));
                    }
                };
            }
        }

        // Confirm that the header contained all required fields
        if gm == 0.0 {
            return Err(BraheError::ParseError(
                "Gravity model file header missing required field: \"earth_gravity_constant\""
                    .to_string(),
            ));
        }
        if radius == 0.0 {
            return Err(BraheError::ParseError(
                "Gravity model file header missing required field: \"radius\"".to_string(),
            ));
        }
        if n_max == 0 {
            return Err(BraheError::ParseError(
                "Gravity model file header missing required field: \"max_degree\"".to_string(),
            ));
        }

        // Read the data
        let mut data = DMatrix::zeros(n_max + 1, m_max + 1);

        for line in lines {
            let l = line?.replace("D", "e").replace("d", "e");
            let mut values = l.split_whitespace();

            // Convert values from string to numeric types
            values.next(); // Skip the first value
            let n = values.next().unwrap().parse::<usize>()?;
            let m = values.next().unwrap().parse::<usize>()?;
            let c = values.next().unwrap().parse::<f64>()?;
            let s = values.next().unwrap().parse::<f64>()?;
            // let sig_C = values.next().unwrap().parse::<f64>()?;
            // let sig_S = values.next().unwrap().parse::<f64>()?;

            // Store the values in the data matrix
            data[(n, m)] = c;

            if m > 0 {
                data[(m - 1, n)] = s;
            }
        }

        Ok(Self {
            data,
            tide_system,
            n_max,
            m_max,
            gm,
            radius,
            model_name,
            model_errors,
            normalization,
        })
    }

    /// Load a gravity model from a file.
    ///
    /// # Arguments
    ///
    /// - `filepath` : Path to the gravity model file.
    ///
    /// # Returns
    ///
    /// - `Self` : Gravity model object.
    ///
    /// # Examples
    ///
    /// ```ignore
    /// use brahe::gravity::GravityModel;
    /// use std::path::Path;
    ///
    /// let filepath = Path::new("./data/gravity_models/EGM2008_360.gfc");
    /// let gravity_model = GravityModel::from_file(filepath).unwrap();
    /// ```
    pub fn from_file(filepath: &Path) -> Result<Self, BraheError> {
        let file = std::fs::File::open(filepath)?;
        let reader = BufReader::new(file);

        Self::from_bufreader(reader)
    }

    /// Load a gravity model from default models included with Brahe. The available default models
    /// are defined by the `DefaultGravityModel` enum. Currently, the available default models are:
    ///
    /// - `EGM2008_360` - a truncated 360x360 version of the full 2190x2190 EGM2008_360 model.
    /// - `GGM05S` - The full 180x180 GGM05S model.
    /// - `JGM3` - The full 70x70 JGM3 model.
    ///
    /// # Arguments
    ///
    /// - `model` : Default gravity model to load. This is a `DefaultGravityModel` enum.
    pub fn from_default(model: DefaultGravityModel) -> Self {
        match model {
            DefaultGravityModel::EGM2008_360 => {
                let reader = BufReader::new(PACKAGED_EGM2008_360);
                Self::from_bufreader(reader).unwrap()
            }
            DefaultGravityModel::GGM05S => {
                let reader = BufReader::new(PACKAGED_GGM05S);
                Self::from_bufreader(reader).unwrap()
            }
            DefaultGravityModel::JGM3 => {
                let reader = BufReader::new(PACKAGED_JGM3);
                Self::from_bufreader(reader).unwrap()
            }
        }
    }

    /// Get the gravity model coefficients for a given degree and order.
    ///
    /// # Arguments
    ///
    /// - `n` : Degree of the gravity model.
    /// - `m` : Order of the gravity model.
    ///
    /// # Returns
    ///
    /// - `Result<(f64, f64), BraheError>` : Tuple of the gravity model coefficients. The first value
    ///   is the C coefficient and the second value is the S coefficient.
    pub fn get(&self, n: usize, m: usize) -> Result<(f64, f64), BraheError> {
        if n > self.n_max || m > self.m_max {
            return Err(BraheError::OutOfBoundsError(format!(
                "Requested gravity model coefficients (n={}, m={}) are out of bounds (n_max={}, m_max={}).",
                n, m, self.n_max, self.m_max
            )));
        }

        if m == 0 {
            Ok((self.data[(n, m)], 0.0))
        } else {
            Ok((self.data[(n, m)], self.data[(m - 1, n)]))
        }
    }

    /// Compute gravitational acceleration from spherical harmonic expansion.
    ///
    /// Evaluates gravity field using recursively-computed associated Legendre functions.
    /// Higher degrees/orders provide more accurate representation of Earth's gravitational
    /// field but increase computational cost.
    ///
    /// # Arguments
    /// - `r_body`: Position vector in body-fixed frame (e.g., ECEF). Units: meters.
    /// - `n_max`: Maximum degree of expansion (zonal terms). Must not exceed model's n_max.
    /// - `m_max`: Maximum order of expansion (tesseral/sectoral terms). Must satisfy m_max <= n_max.
    ///
    /// # Returns
    /// Acceleration vector in body-fixed frame. Units: m/s².
    ///
    /// # Errors
    /// - OutOfBoundsError if requested n_max or m_max exceeds loaded model's limits
    /// - OutOfBoundsError if m_max > n_max
    #[allow(non_snake_case)]
    pub fn compute_spherical_harmonics(
        &self,
        r_body: Vector3<f64>,
        n_max: usize,
        m_max: usize,
    ) -> Result<Vector3<f64>, BraheError> {
        if n_max > self.n_max || m_max > self.m_max {
            return Err(BraheError::OutOfBoundsError(format!(
                "Requested gravity model coefficients (n_max={}, m_max={}) are out of bounds for the input model (n_max={}, m_max={}).",
                n_max, m_max, self.n_max, self.m_max
            )));
        }

        if m_max > n_max {
            return Err(BraheError::OutOfBoundsError(format!(
                "Requested gravity model coefficients (n_max={}, m_max={}) are out of bounds. m_max must be less than or equal to n_max.",
                n_max, m_max
            )));
        }

        // Auxiliary quantities
        let r_sqr = r_body.dot(&r_body); // Square of distance
        let rho = self.radius * self.radius / r_sqr;
        // Normalized coordinates
        let x0 = self.radius * r_body[0] / r_sqr;
        let y0 = self.radius * r_body[1] / r_sqr;
        let z0 = self.radius * r_body[2] / r_sqr;

        // Initialize V and W intemetidary matrices
        let mut V = DMatrix::<f64>::zeros(n_max + 2, n_max + 2);
        let mut W = DMatrix::<f64>::zeros(n_max + 2, n_max + 2);

        // Calculate zonal terms V(n,0); set W(n,0)=0.0
        V[(0, 0)] = self.radius / r_sqr.sqrt();
        W[(0, 0)] = 0.0;

        V[(1, 0)] = z0 * V[(0, 0)];
        W[(1, 0)] = 0.0;

        for n in 2..(n_max + 2) {
            let nf = n as f64;
            V[(n, 0)] =
                ((2.0 * nf - 1.0) * z0 * V[(n - 1, 0)] - (nf - 1.0) * rho * V[(n - 2, 0)]) / nf;
            W[(n, 0)] = 0.0
        }

        // Calculate tesseral and sectorial terms
        for m in 1..m_max + 2 {
            let mf = m as f64;
            // Calculate V(m,m) .. V(n_max+1,m)
            V[(m, m)] = (2.0 * mf - 1.0) * (x0 * V[(m - 1, m - 1)] - y0 * W[(m - 1, m - 1)]);
            W[(m, m)] = (2.0 * mf - 1.0) * (x0 * W[(m - 1, m - 1)] + y0 * V[(m - 1, m - 1)]);

            if m <= n_max {
                V[(m + 1, m)] = (2.0 * mf + 1.0) * z0 * V[(m, m)];
                W[(m + 1, m)] = (2.0 * mf + 1.0) * z0 * W[(m, m)];
            }

            for n in m + 2..n_max + 2 {
                let nf = n as f64;
                V[(n, m)] = ((2.0 * nf - 1.0) * z0 * V[(n - 1, m)]
                    - (nf + mf - 1.0) * rho * V[(n - 2, m)])
                    / (nf - mf);
                W[(n, m)] = ((2.0 * nf - 1.0) * z0 * W[(n - 1, m)]
                    - (nf + mf - 1.0) * rho * W[(n - 2, m)])
                    / (nf - mf);
            }
        }

        // Calculate accelerations ax,ay,az
        let mut ax = 0.0;
        let mut ay = 0.0;
        let mut az = 0.0;

        for m in 0..m_max + 1 {
            let mf = m as f64;
            for n in m..n_max + 1 {
                let nf = n as f64;
                if m == 0 {
                    // Denormalize coefficients, if required
                    let C = if self.normalization == GravityModelNormalization::FullyNormalized {
                        let N = (2.0 * nf + 1.0).sqrt();
                        N * self.data[(n, 0)]
                    } else {
                        self.data[(n, 0)]
                    };

                    ax -= C * V[(n + 1, 1)];
                    ay -= C * W[(n + 1, 1)];
                    az -= (nf + 1.0) * C * V[(n + 1, 0)];
                } else {
                    let C;
                    let S;
                    // Denormalize coefficients, if required
                    if self.normalization == GravityModelNormalization::FullyNormalized {
                        let N = ((2 - kronecker_delta(0, m)) as f64
                            * (2.0 * nf + 1.0)
                            * factorial_product(n, m))
                        .sqrt();
                        C = N * self.data[(n, m)];
                        S = N * self.data[(m - 1, n)];
                    } else {
                        C = self.data[(n, m)];
                        S = self.data[(m - 1, n)];
                    }

                    let Fac = 0.5 * (nf - mf + 1.0) * (nf - mf + 2.0);
                    ax += 0.5 * (-C * V[(n + 1, m + 1)] - S * W[(n + 1, m + 1)])
                        + Fac * (C * V[(n + 1, m - 1)] + S * W[(n + 1, m - 1)]);
                    ay += 0.5 * (-C * W[(n + 1, m + 1)] + S * V[(n + 1, m + 1)])
                        + Fac * (-C * W[(n + 1, m - 1)] + S * V[(n + 1, m - 1)]);
                    az += (nf - mf + 1.0) * (-C * V[(n + 1, m)] - S * W[(n + 1, m)]);
                }
            }
        }

        // Body-fixed acceleration
        Ok((self.gm / (self.radius * self.radius)) * Vector3::new(ax, ay, az))
    }
}

impl std::fmt::Display for GravityModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GravityModel: {}", self.model_name)
    }
}

impl std::fmt::Debug for GravityModel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GravityModel")
            .field("model_name", &self.model_name)
            .field("gm", &self.gm)
            .field("radius", &self.radius)
            .field("n_max", &self.n_max)
            .field("m_max", &self.m_max)
            .field("tide_system", &self.tide_system)
            .field("model_errors", &self.model_errors)
            .field("normalization", &self.normalization)
            .finish()
    }
}

/// Compute the acceleration due to gravity using a spherical harmonic gravity model. The gravity
/// model is defined by the `GravityModel` struct. The acceleration is computed in the body-fixed
/// frame of the central body, and returned in the inertial frame.
///
/// This function accepts either a 3D position vector or a 6D state vector for `r_eci`.
/// When a state vector is provided, only the position component is used.
///
/// # Arguments
///
/// - `r_eci` : Position vector of the object in the ECI frame, or state vector (position + velocity).
/// - `R_i2b` : Transformation matrix from the ECI frame to the body-fixed frame of the central body.
/// - `gravity_model` : Gravity model to use for the computation.
/// - `n_max` : Maximum degree of the gravity model to evaluate.
/// - `m_max` : Maximum order of the gravity model to evaluate.
///
/// # Returns
///
/// - `Vector3<f64>` : Acceleration due to gravity in the ECI frame.
///
/// # Examples
///
/// Using a position vector:
/// ```
/// use nalgebra::{Vector3, Vector6};
/// use brahe::gravity::{GravityModel, DefaultGravityModel};
/// use brahe::frames::rotation_eci_to_ecef;
/// use brahe::time::Epoch;
/// use brahe::eop::{set_global_eop_provider, FileEOPProvider, EOPExtrapolation};
/// use brahe::{R_EARTH, state_osculating_to_cartesian, TimeSystem, AngleFormat};
///
/// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
/// set_global_eop_provider(eop);
///
/// // Compute the rotation matrix from ECI to ECEF
/// let epoch = Epoch::from_datetime(2024, 2, 25, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let R_i2b = rotation_eci_to_ecef(epoch);
///
/// // Create a gravity model
/// let gravity_model = GravityModel::from_default(DefaultGravityModel::EGM2008_360);
///
/// // Compute the acceleration due to gravity
/// let oe = Vector6::new(R_EARTH + 500.0e3, 0.01, 97.3, 0.0, 0.0, 0.0);
/// let x_eci = state_osculating_to_cartesian(oe, AngleFormat::Degrees);
/// let r_eci: Vector3<f64> = x_eci.fixed_rows::<3>(0).into();
///
/// // Compute the acceleration due to gravity
/// let a_grav = brahe::gravity::accel_gravity_spherical_harmonics(r_eci, R_i2b, &gravity_model, 20, 20);
/// ```
///
/// Using a state vector:
/// ```
/// use nalgebra::Vector6;
/// use brahe::gravity::{GravityModel, DefaultGravityModel};
/// use brahe::frames::rotation_eci_to_ecef;
/// use brahe::time::Epoch;
/// use brahe::eop::{set_global_eop_provider, FileEOPProvider, EOPExtrapolation};
/// use brahe::{R_EARTH, state_osculating_to_cartesian, TimeSystem, AngleFormat};
///
/// let eop = FileEOPProvider::from_default_standard(true, EOPExtrapolation::Hold).unwrap();
/// set_global_eop_provider(eop);
///
/// // Compute the rotation matrix from ECI to ECEF
/// let epoch = Epoch::from_datetime(2024, 2, 25, 12, 0, 0.0, 0.0, TimeSystem::UTC);
/// let R_i2b = rotation_eci_to_ecef(epoch);
///
/// // Create a gravity model
/// let gravity_model = GravityModel::from_default(DefaultGravityModel::EGM2008_360);
///
/// // Compute the acceleration due to gravity using state vector directly
/// let oe = Vector6::new(R_EARTH + 500.0e3, 0.01, 97.3, 0.0, 0.0, 0.0);
/// let x_eci = state_osculating_to_cartesian(oe, AngleFormat::Degrees);
///
/// // Pass state vector directly - no need to extract position
/// let a_grav = brahe::gravity::accel_gravity_spherical_harmonics(x_eci, R_i2b, &gravity_model, 20, 20);
/// ```
#[allow(non_snake_case)]
pub fn accel_gravity_spherical_harmonics<P: IntoPosition>(
    r_eci: P,
    R_i2b: SMatrix3,
    gravity_model: &GravityModel,
    n_max: usize,
    m_max: usize,
) -> Vector3<f64> {
    // Extract position and compute body-fixed position
    let r = r_eci.position();
    let r_bf = R_i2b * r;

    // Compute spherical harmonic acceleration
    let a_ecef = gravity_model
        .compute_spherical_harmonics(r_bf, n_max, m_max)
        .unwrap();

    // Inertial acceleration
    R_i2b.transpose() * a_ecef
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;
    use rstest::rstest;

    use crate::constants::{GM_EARTH, R_EARTH};
    use crate::utils::testing::setup_global_test_eop;

    use super::*;

    #[test]
    fn test_gravity_model_from_file() {
        let filepath = Path::new("data/gravity_models/EGM2008_360.gfc");
        let gravity_model = GravityModel::from_file(filepath).unwrap();

        assert_eq!(gravity_model.model_name, "EGM2008");
        assert_eq!(gravity_model.gm, GM_EARTH);
        assert_eq!(gravity_model.radius, R_EARTH);
        assert_eq!(gravity_model.n_max, 360);
        assert_eq!(gravity_model.m_max, 360);
        assert_eq!(gravity_model.tide_system, GravityModelTideSystem::TideFree);
        assert_eq!(gravity_model.model_errors, GravityModelErrors::Calibrated);
        assert_eq!(
            gravity_model.normalization,
            GravityModelNormalization::FullyNormalized
        );
    }

    #[test]
    fn test_gravity_model_from_default_egm2008_360() {
        let gravity_model = GravityModel::from_default(DefaultGravityModel::EGM2008_360);

        assert_eq!(gravity_model.model_name, "EGM2008");
        assert_eq!(gravity_model.gm, GM_EARTH);
        assert_eq!(gravity_model.radius, R_EARTH);
        assert_eq!(gravity_model.n_max, 360);
        assert_eq!(gravity_model.m_max, 360);
        assert_eq!(gravity_model.tide_system, GravityModelTideSystem::TideFree);
        assert_eq!(gravity_model.model_errors, GravityModelErrors::Calibrated);
        assert_eq!(
            gravity_model.normalization,
            GravityModelNormalization::FullyNormalized
        );
    }

    #[test]
    fn test_gravity_model_from_default_ggm05s() {
        let gravity_model = GravityModel::from_default(DefaultGravityModel::GGM05S);

        assert_eq!(gravity_model.model_name, "GGM05S");
        assert_eq!(gravity_model.gm, GM_EARTH);
        assert_eq!(gravity_model.radius, R_EARTH);
        assert_eq!(gravity_model.n_max, 180);
        assert_eq!(gravity_model.m_max, 180);
        assert_eq!(gravity_model.tide_system, GravityModelTideSystem::ZeroTide);
        assert_eq!(gravity_model.model_errors, GravityModelErrors::Calibrated);
        assert_eq!(
            gravity_model.normalization,
            GravityModelNormalization::FullyNormalized
        );
    }

    #[test]
    fn test_gravity_model_from_default_jgm3() {
        let gravity_model = GravityModel::from_default(DefaultGravityModel::JGM3);

        assert_eq!(gravity_model.model_name, "JGM3");
        assert_eq!(gravity_model.gm, GM_EARTH);
        assert_eq!(gravity_model.radius, R_EARTH);
        assert_eq!(gravity_model.n_max, 70);
        assert_eq!(gravity_model.m_max, 70);
        assert_eq!(gravity_model.tide_system, GravityModelTideSystem::Unknown);
        assert_eq!(gravity_model.model_errors, GravityModelErrors::Formal);
        assert_eq!(
            gravity_model.normalization,
            GravityModelNormalization::FullyNormalized
        );
    }

    #[test]
    fn test_gravity_model_get() {
        let gravity_model = GravityModel::from_default(DefaultGravityModel::EGM2008_360);

        let (c, s) = gravity_model.get(2, 0).unwrap();
        assert_abs_diff_eq!(c, -0.484165143790815e-03, epsilon = 1e-12);
        assert_abs_diff_eq!(s, 0.0, epsilon = 1e-12);

        let (c, s) = gravity_model.get(3, 3).unwrap();
        assert_abs_diff_eq!(c, 0.721321757121568e-06, epsilon = 1e-12);
        assert_abs_diff_eq!(s, 0.141434926192941e-05, epsilon = 1e-12);

        let (c, s) = gravity_model.get(360, 360).unwrap();
        assert_abs_diff_eq!(c, 0.200046056782130e-10, epsilon = 1e-12);
        assert_abs_diff_eq!(s, -0.958653755280305e-10, epsilon = 1e-12);

        let result = gravity_model.get(361, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_accel_point_mass_gravity() {
        let r_object = Vector3::new(R_EARTH, 0.0, 0.0);
        let r_central_body = Vector3::new(0.0, 0.0, 0.0);

        let a_grav = accel_point_mass_gravity(r_object, r_central_body, GM_EARTH);

        // Acceleration should be in the negative x-direction and magnitude should be GM_EARTH / R_EARTH^2
        // Roughly -9.8 m/s^2
        assert_abs_diff_eq!(a_grav[0], -GM_EARTH / R_EARTH.powi(2), epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[2], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav.norm(), 9.798, epsilon = 1e-3);

        let r_object = Vector3::new(0.0, R_EARTH, 0.0);
        let a_grav = accel_point_mass_gravity(r_object, r_central_body, GM_EARTH);

        // Acceleration should be in the negative y-direction and magnitude should be GM_EARTH / R_EARTH^2
        // Roughly -9.8 m/s^2
        assert_abs_diff_eq!(a_grav[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[1], -GM_EARTH / R_EARTH.powi(2), epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[2], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav.norm(), 9.798, epsilon = 1e-3);

        let r_object = Vector3::new(0.0, 0.0, R_EARTH);
        let a_grav = accel_point_mass_gravity(r_object, r_central_body, GM_EARTH);

        // Acceleration should be in the negative z-direction and magnitude should be GM_EARTH / R_EARTH^2
        // Roughly -9.8 m/s^2
        assert_abs_diff_eq!(a_grav[0], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[2], -GM_EARTH / R_EARTH.powi(2), epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav.norm(), 9.798, epsilon = 1e-3);
    }

    #[test]
    fn test_gravity_model_compute_spherical_harmonics() {
        setup_global_test_eop();

        let gravity_model = GravityModel::from_default(DefaultGravityModel::EGM2008_360);

        let r_body = Vector3::new(R_EARTH, 0.0, 0.0);

        // Simple test confirming point-mass equivalence
        let a_grav = gravity_model
            .compute_spherical_harmonics(r_body, 0, 0)
            .unwrap();
        assert_abs_diff_eq!(a_grav[0], -GM_EARTH / R_EARTH.powi(2), epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[1], 0.0, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[2], 0.0, epsilon = 1e-12);

        // Test a more complex case
        let a_grav = gravity_model
            .compute_spherical_harmonics(r_body, 60, 60)
            .unwrap();
        assert_abs_diff_eq!(a_grav[0], -9.81433239, epsilon = 1e-8);
        assert_abs_diff_eq!(a_grav[1], 1.813976e-6, epsilon = 1e-12);
        assert_abs_diff_eq!(a_grav[2], -7.29925652190e-5, epsilon = 1e-12);
    }

    #[rstest]
    #[case(2, 2, - 6.97922756436, - 1.8292810538, - 2.69001658552)]
    #[case(3, 3, - 6.97926211185, - 1.82929165145, - 2.68998602761)]
    #[case(4, 4, - 6.97931189287, - 1.82931487069, - 2.6899914012)]
    #[case(5, 5, - 6.9792700471, - 1.82929795164, - 2.68997917147)]
    #[case(6, 6, - 6.979220667, - 1.8292787808, - 2.68997263887)]
    #[case(7, 7, - 6.97925478463, - 1.82926946742, - 2.68999296889)]
    #[case(8, 8, - 6.97927699747, - 1.82928186346, - 2.68998582282)]
    #[case(9, 9, - 6.97925893036, - 1.82928170212, - 2.68997442046)]
    #[case(10, 10, - 6.97924447943, - 1.82928331386, - 2.68997524437)]
    #[case(11, 11, - 6.9792517591, - 1.82928094754, - 2.68998382906)]
    #[case(12, 12, - 6.97924725688, - 1.82928130662, - 2.68998625958)]
    #[case(13, 13, - 6.97924858679, - 1.82928591192, - 2.6899891726)]
    #[case(14, 14, - 6.97924919386, - 1.82928546814, - 2.68999164569)]
    #[case(15, 15, - 6.97925490319, - 1.82928469874, - 2.68999376747)]
    #[case(16, 16, - 6.97926211023, - 1.82928438361, - 2.68999719587)]
    #[case(17, 17, - 6.97926308133, - 1.82928484644, - 2.68999716187)]
    #[case(18, 18, - 6.97926208121, - 1.829284918, - 2.6899952379)]
    #[case(19, 19, - 6.97926229494, - 1.82928369323, - 2.68999256236)]
    #[case(20, 20, - 6.979261862, - 1.82928315091, - 2.68999053339)]
    fn test_accel_gravity_jgm3_validation(
        #[case] n: usize,
        #[case] m: usize,
        #[case] ax: f64,
        #[case] ay: f64,
        #[case] az: f64,
    ) {
        let rot = SMatrix3::identity();

        let gravity_model = GravityModel::from_default(DefaultGravityModel::JGM3);
        let r_body = Vector3::new(6525.919e3, 1710.416e3, 2508.886e3);

        let a_grav = accel_gravity_spherical_harmonics(r_body, rot, &gravity_model, n, m);

        // This could potentially be validated to a higher degree of accuracy, but currently the
        // parameters provided by the Satellite Orbits book are only accurate to seven decimal
        // places, so without using the exact same parameters, it's difficult to validate to a higher
        // degree of accuracy.
        let tol = 1e-7;
        assert_abs_diff_eq!(a_grav[0], ax, epsilon = tol);
        assert_abs_diff_eq!(a_grav[1], ay, epsilon = tol);
        assert_abs_diff_eq!(a_grav[2], az, epsilon = tol);
    }
}
