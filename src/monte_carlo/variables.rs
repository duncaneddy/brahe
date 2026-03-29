/*!
 * Variable definitions and sampling types for Monte Carlo simulations.
 *
 * Provides typed variable identifiers ([`MonteCarloVariableId`]), sampled value
 * containers ([`MonteCarloSampledValue`]), and a per-run parameter collection
 * ([`MonteCarloSampledParameters`]) with type-safe accessors.
 */

use std::collections::HashMap;
use std::fmt;

use nalgebra::{DMatrix, DVector};

use crate::utils::BraheError;

/// Typed identifier for a simulation variable.
///
/// Well-known spacecraft parameters map to specific indices in the propagator's
/// `params` vector via [`param_index`](MonteCarloVariableId::param_index).
/// Use [`Custom`](MonteCarloVariableId::Custom) for user-defined variables.
///
/// # Examples
///
/// ```
/// use brahe::monte_carlo::MonteCarloVariableId;
///
/// let id = MonteCarloVariableId::Mass;
/// assert_eq!(id.param_index(), Some(0));
///
/// let custom = MonteCarloVariableId::Custom("my_param".to_string());
/// assert_eq!(custom.param_index(), None);
/// ```
#[derive(Clone, Debug, Hash, Eq, PartialEq)]
pub enum MonteCarloVariableId {
    /// Initial state vector (position + velocity).
    InitialState,
    /// Spacecraft mass. Maps to params\[0\].
    Mass,
    /// Drag reference area. Maps to params\[1\].
    DragArea,
    /// Drag coefficient (Cd). Maps to params\[2\].
    DragCoefficient,
    /// Solar radiation pressure area. Maps to params\[3\].
    SrpArea,
    /// Reflectivity coefficient (Cr). Maps to params\[4\].
    ReflectivityCoefficient,
    /// Earth orientation parameter table.
    EopTable,
    /// Space weather data table.
    SpaceWeatherTable,
    /// User-defined variable with a custom name.
    Custom(String),
}

impl MonteCarloVariableId {
    /// Returns the params vector index for well-known spacecraft parameters.
    ///
    /// # Returns
    ///
    /// `Option<usize>`: The index into the propagator params vector, or `None`
    /// for variables that are not scalar spacecraft parameters.
    pub fn param_index(&self) -> Option<usize> {
        match self {
            Self::Mass => Some(0),
            Self::DragArea => Some(1),
            Self::DragCoefficient => Some(2),
            Self::SrpArea => Some(3),
            Self::ReflectivityCoefficient => Some(4),
            _ => None,
        }
    }
}

impl fmt::Display for MonteCarloVariableId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InitialState => write!(f, "InitialState"),
            Self::Mass => write!(f, "Mass"),
            Self::DragArea => write!(f, "DragArea"),
            Self::DragCoefficient => write!(f, "DragCoefficient"),
            Self::SrpArea => write!(f, "SrpArea"),
            Self::ReflectivityCoefficient => write!(f, "ReflectivityCoefficient"),
            Self::EopTable => write!(f, "EopTable"),
            Self::SpaceWeatherTable => write!(f, "SpaceWeatherTable"),
            Self::Custom(name) => write!(f, "Custom({})", name),
        }
    }
}

/// A sampled value produced by a Monte Carlo distribution.
///
/// Supports scalar, vector, matrix, and tabular data types to cover the
/// range of parameters needed in astrodynamics simulations.
#[derive(Clone, Debug)]
pub enum MonteCarloSampledValue {
    /// A single scalar value.
    Scalar(f64),
    /// A column vector (e.g., state perturbation).
    Vector(DVector<f64>),
    /// A matrix (e.g., covariance or rotation).
    Matrix(DMatrix<f64>),
    /// A table of (epoch, values) pairs (e.g., EOP or space weather data).
    Table(Vec<(f64, Vec<f64>)>),
}

impl fmt::Display for MonteCarloSampledValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Scalar(v) => write!(f, "Scalar({})", v),
            Self::Vector(v) => write!(f, "Vector({}x1)", v.nrows()),
            Self::Matrix(m) => write!(f, "Matrix({}x{})", m.nrows(), m.ncols()),
            Self::Table(t) => write!(f, "Table({} rows)", t.len()),
        }
    }
}

/// All sampled values for a single simulation run.
///
/// Stores the run index, the per-run random seed, and a map of variable
/// identifiers to their sampled values. Provides type-safe accessors that
/// return [`BraheError`] on type mismatch.
///
/// # Examples
///
/// ```
/// use brahe::monte_carlo::{MonteCarloSampledParameters, MonteCarloVariableId, MonteCarloSampledValue};
///
/// let mut params = MonteCarloSampledParameters::new(0, 42);
/// params.insert(MonteCarloVariableId::Mass, MonteCarloSampledValue::Scalar(500.0));
///
/// let mass = params.get_scalar(&MonteCarloVariableId::Mass).unwrap();
/// assert_eq!(mass, 500.0);
/// ```
#[derive(Clone, Debug)]
pub struct MonteCarloSampledParameters {
    /// Zero-based index of this run within the simulation.
    pub run_index: usize,
    /// Random seed used for this specific run.
    pub run_seed: u64,
    values: HashMap<MonteCarloVariableId, MonteCarloSampledValue>,
}

impl MonteCarloSampledParameters {
    /// Create an empty parameter set for a given run.
    ///
    /// # Arguments
    ///
    /// - `run_index` - Zero-based index of this simulation run
    /// - `run_seed` - Random seed for this run
    ///
    /// # Returns
    ///
    /// `MonteCarloSampledParameters`: Empty parameter collection
    pub fn new(run_index: usize, run_seed: u64) -> Self {
        Self {
            run_index,
            run_seed,
            values: HashMap::new(),
        }
    }

    /// Insert a sampled value for the given variable.
    ///
    /// # Arguments
    ///
    /// - `id` - Variable identifier
    /// - `value` - Sampled value
    pub fn insert(&mut self, id: MonteCarloVariableId, value: MonteCarloSampledValue) {
        self.values.insert(id, value);
    }

    /// Look up a sampled value by variable identifier.
    ///
    /// # Arguments
    ///
    /// - `id` - Variable identifier to look up
    ///
    /// # Returns
    ///
    /// `Option<&MonteCarloSampledValue>`: The value if present
    pub fn get(&self, id: &MonteCarloVariableId) -> Option<&MonteCarloSampledValue> {
        self.values.get(id)
    }

    /// Get a scalar value, returning an error if the variable is missing or not a scalar.
    ///
    /// # Arguments
    ///
    /// - `id` - Variable identifier to look up
    ///
    /// # Returns
    ///
    /// `f64`: The scalar value
    ///
    /// # Errors
    ///
    /// Returns [`BraheError::Error`] if the variable is not found or is not a scalar.
    pub fn get_scalar(&self, id: &MonteCarloVariableId) -> Result<f64, BraheError> {
        match self.values.get(id) {
            Some(MonteCarloSampledValue::Scalar(v)) => Ok(*v),
            Some(other) => Err(BraheError::Error(format!(
                "Variable '{}' is {} but expected Scalar",
                id, other
            ))),
            None => Err(BraheError::Error(format!("Variable '{}' not found", id))),
        }
    }

    /// Get a vector value, returning an error if the variable is missing or not a vector.
    ///
    /// # Arguments
    ///
    /// - `id` - Variable identifier to look up
    ///
    /// # Returns
    ///
    /// `&DVector<f64>`: Reference to the vector value
    ///
    /// # Errors
    ///
    /// Returns [`BraheError::Error`] if the variable is not found or is not a vector.
    pub fn get_vector(&self, id: &MonteCarloVariableId) -> Result<&DVector<f64>, BraheError> {
        match self.values.get(id) {
            Some(MonteCarloSampledValue::Vector(v)) => Ok(v),
            Some(other) => Err(BraheError::Error(format!(
                "Variable '{}' is {} but expected Vector",
                id, other
            ))),
            None => Err(BraheError::Error(format!("Variable '{}' not found", id))),
        }
    }

    /// Get a matrix value, returning an error if the variable is missing or not a matrix.
    ///
    /// # Arguments
    ///
    /// - `id` - Variable identifier to look up
    ///
    /// # Returns
    ///
    /// `&DMatrix<f64>`: Reference to the matrix value
    ///
    /// # Errors
    ///
    /// Returns [`BraheError::Error`] if the variable is not found or is not a matrix.
    pub fn get_matrix(&self, id: &MonteCarloVariableId) -> Result<&DMatrix<f64>, BraheError> {
        match self.values.get(id) {
            Some(MonteCarloSampledValue::Matrix(m)) => Ok(m),
            Some(other) => Err(BraheError::Error(format!(
                "Variable '{}' is {} but expected Matrix",
                id, other
            ))),
            None => Err(BraheError::Error(format!("Variable '{}' not found", id))),
        }
    }

    /// Get a table value, returning an error if the variable is missing or not a table.
    ///
    /// # Arguments
    ///
    /// - `id` - Variable identifier to look up
    ///
    /// # Returns
    ///
    /// `&Vec<(f64, Vec<f64>)>`: Reference to the table data
    ///
    /// # Errors
    ///
    /// Returns [`BraheError::Error`] if the variable is not found or is not a table.
    pub fn get_table(
        &self,
        id: &MonteCarloVariableId,
    ) -> Result<&Vec<(f64, Vec<f64>)>, BraheError> {
        match self.values.get(id) {
            Some(MonteCarloSampledValue::Table(t)) => Ok(t),
            Some(other) => Err(BraheError::Error(format!(
                "Variable '{}' is {} but expected Table",
                id, other
            ))),
            None => Err(BraheError::Error(format!("Variable '{}' not found", id))),
        }
    }

    /// Returns an iterator over all variable identifiers in this parameter set.
    ///
    /// # Returns
    ///
    /// `impl Iterator<Item = &MonteCarloVariableId>`: Iterator over variable IDs
    pub fn ids(&self) -> impl Iterator<Item = &MonteCarloVariableId> {
        self.values.keys()
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    // -- MonteCarloVariableId tests --

    #[test]
    fn test_param_index_mass() {
        assert_eq!(MonteCarloVariableId::Mass.param_index(), Some(0));
    }

    #[test]
    fn test_param_index_drag_area() {
        assert_eq!(MonteCarloVariableId::DragArea.param_index(), Some(1));
    }

    #[test]
    fn test_param_index_drag_coefficient() {
        assert_eq!(MonteCarloVariableId::DragCoefficient.param_index(), Some(2));
    }

    #[test]
    fn test_param_index_srp_area() {
        assert_eq!(MonteCarloVariableId::SrpArea.param_index(), Some(3));
    }

    #[test]
    fn test_param_index_reflectivity_coefficient() {
        assert_eq!(
            MonteCarloVariableId::ReflectivityCoefficient.param_index(),
            Some(4)
        );
    }

    #[test]
    fn test_param_index_initial_state_is_none() {
        assert_eq!(MonteCarloVariableId::InitialState.param_index(), None);
    }

    #[test]
    fn test_param_index_eop_table_is_none() {
        assert_eq!(MonteCarloVariableId::EopTable.param_index(), None);
    }

    #[test]
    fn test_param_index_space_weather_table_is_none() {
        assert_eq!(MonteCarloVariableId::SpaceWeatherTable.param_index(), None);
    }

    #[test]
    fn test_param_index_custom_is_none() {
        let id = MonteCarloVariableId::Custom("foo".to_string());
        assert_eq!(id.param_index(), None);
    }

    #[test]
    fn test_variable_id_hash_eq() {
        use std::collections::HashSet;

        let mut set = HashSet::new();
        set.insert(MonteCarloVariableId::Mass);
        set.insert(MonteCarloVariableId::Mass);
        set.insert(MonteCarloVariableId::DragArea);
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn test_variable_id_eq_same() {
        assert_eq!(MonteCarloVariableId::Mass, MonteCarloVariableId::Mass);
    }

    #[test]
    fn test_variable_id_ne_different() {
        assert_ne!(MonteCarloVariableId::Mass, MonteCarloVariableId::DragArea);
    }

    #[test]
    fn test_variable_id_custom_eq() {
        let a = MonteCarloVariableId::Custom("x".to_string());
        let b = MonteCarloVariableId::Custom("x".to_string());
        let c = MonteCarloVariableId::Custom("y".to_string());
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_variable_id_display() {
        assert_eq!(
            format!("{}", MonteCarloVariableId::InitialState),
            "InitialState"
        );
        assert_eq!(format!("{}", MonteCarloVariableId::Mass), "Mass");
        assert_eq!(format!("{}", MonteCarloVariableId::DragArea), "DragArea");
        assert_eq!(
            format!("{}", MonteCarloVariableId::DragCoefficient),
            "DragCoefficient"
        );
        assert_eq!(format!("{}", MonteCarloVariableId::SrpArea), "SrpArea");
        assert_eq!(
            format!("{}", MonteCarloVariableId::ReflectivityCoefficient),
            "ReflectivityCoefficient"
        );
        assert_eq!(format!("{}", MonteCarloVariableId::EopTable), "EopTable");
        assert_eq!(
            format!("{}", MonteCarloVariableId::SpaceWeatherTable),
            "SpaceWeatherTable"
        );
        assert_eq!(
            format!("{}", MonteCarloVariableId::Custom("my_var".to_string())),
            "Custom(my_var)"
        );
    }

    #[test]
    fn test_variable_id_clone() {
        let id = MonteCarloVariableId::Custom("test".to_string());
        let cloned = id.clone();
        assert_eq!(id, cloned);
    }

    #[test]
    fn test_variable_id_debug() {
        let id = MonteCarloVariableId::Mass;
        let debug = format!("{:?}", id);
        assert!(debug.contains("Mass"));
    }

    // -- MonteCarloSampledValue tests --

    #[test]
    fn test_sampled_value_display_scalar() {
        let v = MonteCarloSampledValue::Scalar(3.125);
        assert_eq!(format!("{}", v), "Scalar(3.125)");
    }

    #[test]
    fn test_sampled_value_display_vector() {
        let v = MonteCarloSampledValue::Vector(DVector::from_vec(vec![1.0, 2.0, 3.0]));
        assert_eq!(format!("{}", v), "Vector(3x1)");
    }

    #[test]
    fn test_sampled_value_display_matrix() {
        let m = MonteCarloSampledValue::Matrix(DMatrix::zeros(3, 4));
        assert_eq!(format!("{}", m), "Matrix(3x4)");
    }

    #[test]
    fn test_sampled_value_display_table() {
        let t = MonteCarloSampledValue::Table(vec![(0.0, vec![1.0, 2.0]), (1.0, vec![3.0, 4.0])]);
        assert_eq!(format!("{}", t), "Table(2 rows)");
    }

    // -- MonteCarloSampledParameters tests --

    #[test]
    fn test_new_parameters() {
        let params = MonteCarloSampledParameters::new(5, 999);
        assert_eq!(params.run_index, 5);
        assert_eq!(params.run_seed, 999);
        assert_eq!(params.ids().count(), 0);
    }

    #[test]
    fn test_insert_and_get() {
        let mut params = MonteCarloSampledParameters::new(0, 42);
        params.insert(
            MonteCarloVariableId::Mass,
            MonteCarloSampledValue::Scalar(500.0),
        );
        assert!(params.get(&MonteCarloVariableId::Mass).is_some());
        assert!(params.get(&MonteCarloVariableId::DragArea).is_none());
    }

    #[test]
    fn test_get_scalar_success() {
        let mut params = MonteCarloSampledParameters::new(0, 42);
        params.insert(
            MonteCarloVariableId::Mass,
            MonteCarloSampledValue::Scalar(500.0),
        );
        let mass = params.get_scalar(&MonteCarloVariableId::Mass).unwrap();
        assert_eq!(mass, 500.0);
    }

    #[test]
    fn test_get_scalar_type_mismatch() {
        let mut params = MonteCarloSampledParameters::new(0, 42);
        params.insert(
            MonteCarloVariableId::InitialState,
            MonteCarloSampledValue::Vector(DVector::from_vec(vec![1.0, 2.0])),
        );
        let result = params.get_scalar(&MonteCarloVariableId::InitialState);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("expected Scalar"));
        assert!(err_msg.contains("Vector"));
    }

    #[test]
    fn test_get_scalar_not_found() {
        let params = MonteCarloSampledParameters::new(0, 42);
        let result = params.get_scalar(&MonteCarloVariableId::Mass);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("not found"));
    }

    #[test]
    fn test_get_vector_success() {
        let mut params = MonteCarloSampledParameters::new(0, 42);
        let vec = DVector::from_vec(vec![1.0, 2.0, 3.0]);
        params.insert(
            MonteCarloVariableId::InitialState,
            MonteCarloSampledValue::Vector(vec.clone()),
        );
        let result = params
            .get_vector(&MonteCarloVariableId::InitialState)
            .unwrap();
        assert_eq!(result.nrows(), 3);
        assert_eq!(result[0], 1.0);
    }

    #[test]
    fn test_get_vector_type_mismatch() {
        let mut params = MonteCarloSampledParameters::new(0, 42);
        params.insert(
            MonteCarloVariableId::Mass,
            MonteCarloSampledValue::Scalar(100.0),
        );
        let result = params.get_vector(&MonteCarloVariableId::Mass);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expected Vector"));
    }

    #[test]
    fn test_get_vector_not_found() {
        let params = MonteCarloSampledParameters::new(0, 42);
        let result = params.get_vector(&MonteCarloVariableId::InitialState);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_matrix_success() {
        let mut params = MonteCarloSampledParameters::new(0, 42);
        let mat = DMatrix::from_row_slice(2, 2, &[1.0, 2.0, 3.0, 4.0]);
        params.insert(
            MonteCarloVariableId::Custom("cov".to_string()),
            MonteCarloSampledValue::Matrix(mat),
        );
        let result = params
            .get_matrix(&MonteCarloVariableId::Custom("cov".to_string()))
            .unwrap();
        assert_eq!(result.nrows(), 2);
        assert_eq!(result.ncols(), 2);
    }

    #[test]
    fn test_get_matrix_type_mismatch() {
        let mut params = MonteCarloSampledParameters::new(0, 42);
        params.insert(
            MonteCarloVariableId::Mass,
            MonteCarloSampledValue::Scalar(100.0),
        );
        let result = params.get_matrix(&MonteCarloVariableId::Mass);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expected Matrix"));
    }

    #[test]
    fn test_get_matrix_not_found() {
        let params = MonteCarloSampledParameters::new(0, 42);
        let result = params.get_matrix(&MonteCarloVariableId::Custom("x".to_string()));
        assert!(result.is_err());
    }

    #[test]
    fn test_get_table_success() {
        let mut params = MonteCarloSampledParameters::new(0, 42);
        let table = vec![(0.0, vec![1.0]), (86400.0, vec![2.0])];
        params.insert(
            MonteCarloVariableId::EopTable,
            MonteCarloSampledValue::Table(table),
        );
        let result = params.get_table(&MonteCarloVariableId::EopTable).unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].0, 0.0);
    }

    #[test]
    fn test_get_table_type_mismatch() {
        let mut params = MonteCarloSampledParameters::new(0, 42);
        params.insert(
            MonteCarloVariableId::EopTable,
            MonteCarloSampledValue::Scalar(1.0),
        );
        let result = params.get_table(&MonteCarloVariableId::EopTable);
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("expected Table"));
    }

    #[test]
    fn test_get_table_not_found() {
        let params = MonteCarloSampledParameters::new(0, 42);
        let result = params.get_table(&MonteCarloVariableId::EopTable);
        assert!(result.is_err());
    }

    #[test]
    fn test_ids_iterator() {
        let mut params = MonteCarloSampledParameters::new(0, 42);
        params.insert(
            MonteCarloVariableId::Mass,
            MonteCarloSampledValue::Scalar(100.0),
        );
        params.insert(
            MonteCarloVariableId::DragCoefficient,
            MonteCarloSampledValue::Scalar(2.2),
        );
        let ids: Vec<_> = params.ids().collect();
        assert_eq!(ids.len(), 2);
        assert!(ids.contains(&&MonteCarloVariableId::Mass));
        assert!(ids.contains(&&MonteCarloVariableId::DragCoefficient));
    }

    #[test]
    fn test_insert_overwrites() {
        let mut params = MonteCarloSampledParameters::new(0, 42);
        params.insert(
            MonteCarloVariableId::Mass,
            MonteCarloSampledValue::Scalar(100.0),
        );
        params.insert(
            MonteCarloVariableId::Mass,
            MonteCarloSampledValue::Scalar(200.0),
        );
        assert_eq!(
            params.get_scalar(&MonteCarloVariableId::Mass).unwrap(),
            200.0
        );
        assert_eq!(params.ids().count(), 1);
    }

    #[test]
    fn test_clone() {
        let mut params = MonteCarloSampledParameters::new(3, 77);
        params.insert(
            MonteCarloVariableId::Mass,
            MonteCarloSampledValue::Scalar(100.0),
        );
        let cloned = params.clone();
        assert_eq!(cloned.run_index, 3);
        assert_eq!(cloned.run_seed, 77);
        assert_eq!(
            cloned.get_scalar(&MonteCarloVariableId::Mass).unwrap(),
            100.0
        );
    }
}
