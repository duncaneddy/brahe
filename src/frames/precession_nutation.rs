/*!
 * Selection of the precession-nutation model evaluated by the Earth
 * orientation chains.
 *
 * The CIO-based chain in [`super::gcrf_itrf`] and the equinox-based chain in
 * [`super::equinox`] both evaluate the model returned by
 * [`get_precession_nutation_model`], so the two chains always share a basis.
 */

use std::fmt;
use std::sync::RwLock;

use once_cell::sync::Lazy;
use serde::{Deserialize, Serialize};

/// Precession-nutation model used by the GCRF/ITRF and GCRF/MOD/TOD chains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum PrecessionNutationModel {
    /// IAU 2006 precession with the full IAU 2000A nutation series (SOFA
    /// `iauXys06a`, `iauPn06a`).
    #[default]
    IAU2006A,
    /// IAU 2000 precession with the truncated IAU 2000B nutation series (SOFA
    /// `iauXys00b`, `iauPn00b`); about seven times faster, within about
    /// 1 mas of IAU 2006/2000A.
    IAU2000B,
}

impl fmt::Display for PrecessionNutationModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PrecessionNutationModel::IAU2006A => write!(f, "IAU 2006/2000A"),
            PrecessionNutationModel::IAU2000B => write!(f, "IAU 2000B"),
        }
    }
}

static PN_MODEL: Lazy<RwLock<PrecessionNutationModel>> =
    Lazy::new(|| RwLock::new(PrecessionNutationModel::default()));

/// Set the crate-wide precession-nutation model evaluated by the Earth
/// orientation transformations.
///
/// The setting applies to every subsequent call of the GCRF/ITRF and
/// GCRF/MOD/TOD transformations, including the frame router and the batch
/// forms. Functions that take a model argument explicitly are unaffected.
///
/// # Arguments
/// - `model`: Precession-nutation model to evaluate
///
/// # Returns
/// - Nothing; the crate-wide model is replaced
///
/// # Examples
/// ```
/// use brahe::frames::*;
///
/// set_precession_nutation_model(PrecessionNutationModel::IAU2000B);
/// assert_eq!(get_precession_nutation_model(), PrecessionNutationModel::IAU2000B);
///
/// set_precession_nutation_model(PrecessionNutationModel::IAU2006A);
/// ```
pub fn set_precession_nutation_model(model: PrecessionNutationModel) {
    *PN_MODEL.write().unwrap() = model;
}

/// Get the crate-wide precession-nutation model evaluated by the Earth
/// orientation transformations.
///
/// # Arguments
/// - None
///
/// # Returns
/// - `model`: Precession-nutation model currently selected. IAU 2006/2000A
///   unless [`set_precession_nutation_model`] has been called
///
/// # Examples
/// ```
/// use brahe::frames::*;
///
/// let model = get_precession_nutation_model();
/// println!("{}", model);
/// ```
pub fn get_precession_nutation_model() -> PrecessionNutationModel {
    *PN_MODEL.read().unwrap()
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use serial_test::{parallel, serial};

    use super::*;

    #[test]
    #[serial]
    fn test_get_set_precession_nutation_model() {
        assert_eq!(
            get_precession_nutation_model(),
            PrecessionNutationModel::IAU2006A
        );

        set_precession_nutation_model(PrecessionNutationModel::IAU2000B);
        assert_eq!(
            get_precession_nutation_model(),
            PrecessionNutationModel::IAU2000B
        );

        set_precession_nutation_model(PrecessionNutationModel::IAU2006A);
        assert_eq!(
            get_precession_nutation_model(),
            PrecessionNutationModel::IAU2006A
        );
    }

    #[test]
    #[parallel]
    fn test_precession_nutation_model_display_and_serde() {
        assert_eq!(
            PrecessionNutationModel::IAU2006A.to_string(),
            "IAU 2006/2000A"
        );
        assert_eq!(PrecessionNutationModel::IAU2000B.to_string(), "IAU 2000B");
        assert_eq!(
            PrecessionNutationModel::default(),
            PrecessionNutationModel::IAU2006A
        );

        let json = serde_json::to_string(&PrecessionNutationModel::IAU2000B).unwrap();
        assert_eq!(json, "\"IAU2000B\"");
        let parsed: PrecessionNutationModel = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, PrecessionNutationModel::IAU2000B);
    }
}
