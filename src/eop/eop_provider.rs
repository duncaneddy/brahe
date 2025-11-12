/*!
Defines the EarthOrientationProvider trait
*/

use crate::eop::eop_types::{EOPExtrapolation, EOPType};
use crate::utils::errors::BraheError;

/// `EarthOrientationProvider` is a trait for objects that provide Earth orientation parameters.
///
/// This trait defines a common interface for all Earth orientation providers. An Earth orientation provider
/// is an object that can provide Earth orientation parameters like UT1-UTC, polar motion, and rate of change
/// of the length of day.
///
/// Implementations of this trait are expected to provide specific methods for retrieving these parameters,
/// such as `get_ut1_utc`, `get_pm`, `get_dxdy`, and `get_lod`. These methods should return the requested
/// parameter for a given Modified Julian Date (MJD).
///
/// This trait can be extended to implement custom Earth orientation providers. For example, a custom provider
/// could retrieve Earth orientation parameters from a file type that is not yet supported or a database.
///
/// # Example
///
/// ```ignore
/// use brahe::eop::EarthOrientationProvider;
///
/// struct MyEOPProvider;
///
/// impl EarthOrientationProvider for MyEOPProvider {
///     // Implement the methods here
/// }
/// ```
pub trait EarthOrientationProvider {
    /// Returns the number of EOP data entries loaded in the provider.
    fn len(&self) -> usize;

    /// Returns true if the provider contains no EOP data entries.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the type of EOP data loaded (C04, StandardBulletinA, or Static).
    fn eop_type(&self) -> EOPType;

    /// Returns true if the provider has been initialized with valid EOP data.
    fn is_initialized(&self) -> bool;

    /// Returns the extrapolation behavior (Zero, Hold, or Error) for out-of-range requests.
    fn extrapolation(&self) -> EOPExtrapolation;

    /// Returns true if the provider interpolates between data points, false for step function.
    fn interpolation(&self) -> bool;

    /// Returns the minimum Modified Julian Date covered by the loaded EOP data.
    fn mjd_min(&self) -> f64;

    /// Returns the maximum Modified Julian Date covered by the loaded EOP data.
    fn mjd_max(&self) -> f64;

    /// Returns the last MJD with valid Length of Day (LOD) data available.
    fn mjd_last_lod(&self) -> f64;

    /// Returns the last MJD with valid nutation correction (dX/dY) data available.
    fn mjd_last_dxdy(&self) -> f64;

    /// Get UT1-UTC offset for a given MJD. Units: seconds.
    fn get_ut1_utc(&self, mjd: f64) -> Result<f64, BraheError>;

    /// Get polar motion corrections (x, y) for a given MJD. Units: radians.
    /// Returns: (pm_x, pm_y)
    fn get_pm(&self, mjd: f64) -> Result<(f64, f64), BraheError>;

    /// Get celestial pole offsets (dX, dY) for a given MJD. Units: radians.
    /// Returns: (dX, dY) corrections to IAU 2006/2000A precession-nutation model.
    fn get_dxdy(&self, mjd: f64) -> Result<(f64, f64), BraheError>;

    /// Get Length of Day (LOD) correction for a given MJD. Units: seconds.
    /// LOD represents difference from nominal 86400 second day.
    fn get_lod(&self, mjd: f64) -> Result<f64, BraheError>;

    /// Get all EOP parameters for a given MJD. Units: radians and seconds.
    /// Returns: (pm_x, pm_y, ut1_utc, lod, dX, dY)
    fn get_eop(&self, mjd: f64) -> Result<(f64, f64, f64, f64, f64, f64), BraheError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test implementation of EarthOrientationProvider for testing the default is_empty() method
    struct MockEOPProvider {
        len: usize,
    }

    impl EarthOrientationProvider for MockEOPProvider {
        fn len(&self) -> usize {
            self.len
        }

        fn eop_type(&self) -> EOPType {
            EOPType::C04
        }

        fn is_initialized(&self) -> bool {
            true
        }

        fn extrapolation(&self) -> EOPExtrapolation {
            EOPExtrapolation::Hold
        }

        fn interpolation(&self) -> bool {
            true
        }

        fn mjd_min(&self) -> f64 {
            0.0
        }

        fn mjd_max(&self) -> f64 {
            0.0
        }

        fn mjd_last_lod(&self) -> f64 {
            0.0
        }

        fn mjd_last_dxdy(&self) -> f64 {
            0.0
        }

        fn get_ut1_utc(&self, _mjd: f64) -> Result<f64, BraheError> {
            Ok(0.0)
        }

        fn get_pm(&self, _mjd: f64) -> Result<(f64, f64), BraheError> {
            Ok((0.0, 0.0))
        }

        fn get_dxdy(&self, _mjd: f64) -> Result<(f64, f64), BraheError> {
            Ok((0.0, 0.0))
        }

        fn get_lod(&self, _mjd: f64) -> Result<f64, BraheError> {
            Ok(0.0)
        }

        fn get_eop(&self, _mjd: f64) -> Result<(f64, f64, f64, f64, f64, f64), BraheError> {
            Ok((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
        }
    }

    #[test]
    fn test_earth_orientation_provider_is_empty_default() {
        // Test that default is_empty() returns true when len() == 0
        let empty_provider = MockEOPProvider { len: 0 };
        assert!(empty_provider.is_empty());

        // Test that default is_empty() returns false when len() > 0
        let non_empty_provider = MockEOPProvider { len: 10 };
        assert!(!non_empty_provider.is_empty());

        // Test boundary case with len() == 1
        let single_entry_provider = MockEOPProvider { len: 1 };
        assert!(!single_entry_provider.is_empty());
    }

    #[test]
    fn test_mock_eop_provider_trait_methods() {
        // Test all EarthOrientationProvider trait methods on MockEOPProvider
        let provider = MockEOPProvider { len: 5 };

        // Test metadata methods
        assert_eq!(provider.len(), 5);
        assert!(!provider.is_empty());
        assert_eq!(provider.eop_type(), EOPType::C04);
        assert!(provider.is_initialized());
        assert_eq!(provider.extrapolation(), EOPExtrapolation::Hold);
        assert!(provider.interpolation());

        // Test MJD range methods
        assert_eq!(provider.mjd_min(), 0.0);
        assert_eq!(provider.mjd_max(), 0.0);
        assert_eq!(provider.mjd_last_lod(), 0.0);
        assert_eq!(provider.mjd_last_dxdy(), 0.0);

        // Test EOP data retrieval methods
        let test_mjd = 60000.0;
        assert_eq!(provider.get_ut1_utc(test_mjd).unwrap(), 0.0);
        assert_eq!(provider.get_pm(test_mjd).unwrap(), (0.0, 0.0));
        assert_eq!(provider.get_dxdy(test_mjd).unwrap(), (0.0, 0.0));
        assert_eq!(provider.get_lod(test_mjd).unwrap(), 0.0);
        assert_eq!(
            provider.get_eop(test_mjd).unwrap(),
            (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        );
    }
}
