/*!
 * Shared `StarRecord` trait for star catalog records.
 *
 * Defines a common interface implemented by catalog-specific record types
 * (FK5, Hipparcos, Tycho-2) so that downstream code (reference-frame
 * realization, star-based attitude determination) can work generically
 * across catalogs regardless of source.
 */

use crate::constants::AngleFormat;
use crate::coordinates::{apply_proper_motion, position_radec_to_inertial};
use crate::math::SVector3;
use crate::time::Epoch;

/// Common interface implemented by star catalog record types.
///
/// Provides uniform access to position, proper motion, parallax, radial
/// velocity, and photometric data for a cataloged star, along with derived
/// quantities (`unit_vector`, `radec_at_epoch`) that are computed the same
/// way for every catalog.
pub trait StarRecord {
    /// Catalog identifier string (e.g. `"FK5 1"`).
    fn id(&self) -> String;

    /// Common or cross-catalog name, if available (e.g. `"HD 358"`).
    fn name(&self) -> Option<String>;

    /// Right ascension at the record's reference `epoch()`. Units: *deg*
    fn ra(&self) -> f64;

    /// Declination at the record's reference `epoch()`. Units: *deg*
    fn dec(&self) -> f64;

    /// Proper motion in right ascension (μ_α* = μ_α cos δ), if known. Units: *mas/yr*
    fn pm_ra(&self) -> Option<f64>;

    /// Proper motion in declination, if known. Units: *mas/yr*
    fn pm_dec(&self) -> Option<f64>;

    /// Parallax, if known. Units: *mas*
    fn parallax(&self) -> Option<f64>;

    /// Radial velocity, if known. Units: *km/s*
    fn radial_velocity(&self) -> Option<f64>;

    /// Reference epoch of the catalog position (e.g. J2000.0 for FK5).
    fn epoch(&self) -> Epoch;

    /// Visual magnitude, if known. Units: *mag*
    fn magnitude(&self) -> Option<f64>;

    /// Unit vector toward the star, evaluated at the record's reference `epoch()`.
    ///
    /// Computed from `ra()`/`dec()` via [`position_radec_to_inertial`] with a
    /// unit range, so no proper-motion propagation is applied.
    ///
    /// # Returns
    /// - `u`: Cartesian unit vector `[x, y, z]` toward the star. Units: dimensionless
    fn unit_vector(&self) -> SVector3 {
        position_radec_to_inertial(
            SVector3::new(self.ra(), self.dec(), 1.0),
            AngleFormat::Degrees,
        )
    }

    /// Right ascension and declination propagated from the record's reference
    /// epoch to a target epoch using proper motion (and parallax/radial
    /// velocity, if both are known).
    ///
    /// Missing proper motion components (`pm_ra`/`pm_dec` returning `None`)
    /// are treated as zero.
    ///
    /// # Arguments
    /// - `epoch`: Target epoch to propagate the position to
    /// - `angle_format`: Desired angle format (`Degrees` or `Radians`) for the returned `(ra, dec)`
    ///
    /// # Returns
    /// - `(ra, dec)`: Right ascension and declination at `epoch`. Units: (*angle*, *angle*)
    fn radec_at_epoch(&self, epoch: Epoch, angle_format: AngleFormat) -> (f64, f64) {
        let (ra, dec) = match angle_format {
            AngleFormat::Degrees => (self.ra(), self.dec()),
            AngleFormat::Radians => (self.ra().to_radians(), self.dec().to_radians()),
        };

        apply_proper_motion(
            ra,
            dec,
            self.pm_ra().unwrap_or(0.0),
            self.pm_dec().unwrap_or(0.0),
            self.parallax(),
            self.radial_velocity(),
            self.epoch(),
            epoch,
            angle_format,
        )
    }
}
