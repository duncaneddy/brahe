/*!
 * Defines the TableEOPProvider struct for in-memory EOP data.
 *
 * This provider is constructed from explicit data entries rather than
 * parsed from a file. It is designed for use in Monte Carlo simulations
 * where EOP parameters may be perturbed, and for testing scenarios
 * that need programmatic control over EOP values.
 */

use std::cmp::Ordering;
use std::collections::{BTreeMap, HashMap};
use std::fmt;

use crate::eop::eop_provider::EarthOrientationProvider;
use crate::eop::eop_types::{EOPExtrapolation, EOPType};
use crate::utils::BraheError;

// Type aliases matching FileEOPProvider conventions
type EOPData = (f64, f64, f64, Option<f64>, Option<f64>, Option<f64>);
type EOPDataMap = BTreeMap<TableEOPKey, EOPData>;

/// Entry type for `TableEOPProvider::from_entries`: (mjd, pm_x, pm_y, ut1_utc, lod, dX, dY).
type EOPEntry = (f64, f64, f64, f64, Option<f64>, Option<f64>, Option<f64>);

/// Custom key type for the EOP data BTreeMap, matching the pattern used
/// by FileEOPProvider. Wraps f64 to provide Ord/Eq for BTreeMap keys.
#[derive(Clone, PartialEq)]
struct TableEOPKey(f64);

impl PartialOrd for TableEOPKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for TableEOPKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

impl Eq for TableEOPKey {}

/// Provides Earth Orientation Parameter (EOP) data from in-memory tables.
///
/// `TableEOPProvider` stores EOP data in an internal BTreeMap, supporting
/// interpolation and extrapolation with the same behavior as `FileEOPProvider`.
/// It can be constructed directly from data entries or by copying and
/// optionally perturbing values from an existing provider.
///
/// This provider is useful for:
/// - Monte Carlo simulations with perturbed EOP parameters
/// - Testing with programmatic control over EOP values
/// - Scenarios where EOP data comes from sources other than files
#[derive(Clone)]
pub struct TableEOPProvider {
    initialized: bool,
    eop_type: EOPType,
    data: EOPDataMap,
    extrapolate: EOPExtrapolation,
    interpolate: bool,
    mjd_min: f64,
    mjd_max: f64,
    mjd_last_lod: f64,
    mjd_last_dxdy: f64,
}

impl fmt::Display for TableEOPProvider {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "TableEOPProvider - type: {}, {} entries, mjd_min: {}, mjd_max: {}, mjd_last_lod: \
            {}, mjd_last_dxdy: {}, extrapolation: {}, interpolation: {}",
            self.eop_type(),
            self.len(),
            self.mjd_min(),
            self.mjd_max(),
            self.mjd_last_lod(),
            self.mjd_last_dxdy(),
            self.extrapolation(),
            self.interpolation()
        )
    }
}

impl fmt::Debug for TableEOPProvider {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "TableEOPProvider<Type: {}, Length: {}, mjd_min: {}, mjd_max: {}, mjd_last_lod: \
            {}, mjd_last_dxdy: {}, extrapolation: {}, interpolation: {}>",
            self.eop_type(),
            self.len(),
            self.mjd_min(),
            self.mjd_max(),
            self.mjd_last_lod(),
            self.mjd_last_dxdy(),
            self.extrapolation(),
            self.interpolation()
        )
    }
}

impl TableEOPProvider {
    /// Creates a `TableEOPProvider` from explicit data entries.
    ///
    /// Each entry is a tuple of `(mjd, pm_x, pm_y, ut1_utc, lod, dX, dY)` where
    /// `lod`, `dX`, and `dY` are optional. Units match the `FileEOPProvider`
    /// convention: polar motion in radians, ut1_utc and lod in seconds, dX/dY
    /// in radians.
    ///
    /// # Arguments
    ///
    /// * `entries` - Vector of EOP data entries. Each entry is
    ///   `(mjd, pm_x, pm_y, ut1_utc, lod, dX, dY)`.
    /// * `interpolate` - Whether to linearly interpolate between data points
    /// * `extrapolate` - Behavior for out-of-range date requests
    ///
    /// # Returns
    ///
    /// * `Result<TableEOPProvider, BraheError>` - Initialized provider, or error
    ///   if entries is empty
    ///
    /// # Example
    ///
    /// ```
    /// use brahe::eop::{TableEOPProvider, EarthOrientationProvider, EOPExtrapolation};
    ///
    /// let entries = vec![
    ///     (59000.0, 0.1, 0.2, 0.3, Some(0.001), Some(0.0001), Some(0.0002)),
    ///     (59001.0, 0.11, 0.21, 0.31, Some(0.0011), Some(0.00011), Some(0.00021)),
    /// ];
    ///
    /// let provider = TableEOPProvider::from_entries(entries, true, EOPExtrapolation::Hold).unwrap();
    /// assert!(provider.is_initialized());
    /// assert_eq!(provider.len(), 2);
    /// ```
    pub fn from_entries(
        entries: Vec<EOPEntry>,
        interpolate: bool,
        extrapolate: EOPExtrapolation,
    ) -> Result<Self, BraheError> {
        if entries.is_empty() {
            return Err(BraheError::EOPError(
                "Cannot create TableEOPProvider from empty entries".to_string(),
            ));
        }

        let mut data: EOPDataMap = BTreeMap::new();
        let mut mjd_min = f64::MAX;
        let mut mjd_max = f64::MIN;
        let mut mjd_last_lod = 0.0_f64;
        let mut mjd_last_dxdy = 0.0_f64;

        for (mjd, pm_x, pm_y, ut1_utc, lod, dx, dy) in &entries {
            let mjd = *mjd;

            if mjd < mjd_min {
                mjd_min = mjd;
            }
            if mjd > mjd_max {
                mjd_max = mjd;
            }

            if lod.is_some() {
                mjd_last_lod = mjd_last_lod.max(mjd);
            }
            if dx.is_some() && dy.is_some() {
                mjd_last_dxdy = mjd_last_dxdy.max(mjd);
            }

            // Internal storage order: (pm_x, pm_y, ut1_utc, dX, dY, lod)
            // matching FileEOPProvider convention where .3=dX, .4=dY, .5=lod
            data.insert(TableEOPKey(mjd), (*pm_x, *pm_y, *ut1_utc, *dx, *dy, *lod));
        }

        Ok(Self {
            initialized: true,
            eop_type: EOPType::C04, // Table providers use C04-style (all fields present)
            data,
            extrapolate,
            interpolate,
            mjd_min,
            mjd_max,
            mjd_last_lod,
            mjd_last_dxdy,
        })
    }

    /// Creates a `TableEOPProvider` by copying data from an existing provider,
    /// optionally applying additive perturbations to specific parameters.
    ///
    /// This is the primary constructor for Monte Carlo EOP perturbation. It
    /// samples the base provider at regular intervals and applies constant
    /// offsets from the perturbation map.
    ///
    /// # Arguments
    ///
    /// * `base` - Reference to the base EOP provider to copy from
    /// * `mjd_start` - Start of the MJD range to sample
    /// * `mjd_end` - End of the MJD range to sample (inclusive)
    /// * `step` - Step size in days between samples (typically 1.0)
    /// * `perturbations` - Map of parameter names to additive offsets. Supported
    ///   keys: `"pm_x"`, `"pm_y"`, `"ut1_utc"`, `"lod"`, `"dX"`, `"dY"`. Units
    ///   must match the provider convention (radians for angles, seconds for time).
    /// * `interpolate` - Whether to linearly interpolate between data points
    /// * `extrapolate` - Behavior for out-of-range date requests
    ///
    /// # Returns
    ///
    /// * `Result<TableEOPProvider, BraheError>` - Initialized provider with
    ///   perturbed data, or error if base provider fails
    ///
    /// # Example
    ///
    /// ```
    /// use std::collections::HashMap;
    /// use brahe::eop::{
    ///     TableEOPProvider, StaticEOPProvider, EarthOrientationProvider, EOPExtrapolation,
    /// };
    ///
    /// let base = StaticEOPProvider::from_values((0.1, 0.2, 0.3, 0.4, 0.5, 0.6));
    ///
    /// let mut perturbations = HashMap::new();
    /// perturbations.insert("ut1_utc".to_string(), 0.001);
    ///
    /// let provider = TableEOPProvider::from_perturbed(
    ///     &base, 59000.0, 59002.0, 1.0,
    ///     &perturbations, true, EOPExtrapolation::Hold,
    /// ).unwrap();
    ///
    /// assert_eq!(provider.len(), 3);
    /// // ut1_utc should be base value (0.3) + perturbation (0.001)
    /// assert!((provider.get_ut1_utc(59000.0).unwrap() - 0.301).abs() < 1e-10);
    /// ```
    #[allow(non_snake_case)]
    pub fn from_perturbed(
        base: &dyn EarthOrientationProvider,
        mjd_start: f64,
        mjd_end: f64,
        step: f64,
        perturbations: &HashMap<String, f64>,
        interpolate: bool,
        extrapolate: EOPExtrapolation,
    ) -> Result<Self, BraheError> {
        let d_pm_x = perturbations.get("pm_x").copied().unwrap_or(0.0);
        let d_pm_y = perturbations.get("pm_y").copied().unwrap_or(0.0);
        let d_ut1_utc = perturbations.get("ut1_utc").copied().unwrap_or(0.0);
        let d_lod = perturbations.get("lod").copied().unwrap_or(0.0);
        let d_dX = perturbations.get("dX").copied().unwrap_or(0.0);
        let d_dY = perturbations.get("dY").copied().unwrap_or(0.0);

        let mut entries = Vec::new();
        let mut mjd = mjd_start;

        while mjd <= mjd_end + step * 0.5 * f64::EPSILON {
            let (pm_x, pm_y, ut1_utc, dX, dY, lod) = base.get_eop(mjd)?;

            entries.push((
                mjd,
                pm_x + d_pm_x,
                pm_y + d_pm_y,
                ut1_utc + d_ut1_utc,
                Some(lod + d_lod),
                Some(dX + d_dX),
                Some(dY + d_dY),
            ));

            mjd += step;
        }

        Self::from_entries(entries, interpolate, extrapolate)
    }

    /// Look up EOP data for a given MJD key, returning an error if the key is missing.
    fn get_data(&self, mjd: f64) -> Result<&EOPData, BraheError> {
        self.data.get(&TableEOPKey(mjd)).ok_or_else(|| {
            BraheError::EOPError(format!(
                "EOP data missing for MJD {} in TableEOPProvider",
                mjd
            ))
        })
    }
}

/// Extract a required `Option<f64>` EOP field, returning an error if `None`.
fn require_eop_field(value: Option<f64>, field_name: &str) -> Result<f64, BraheError> {
    value.ok_or_else(|| {
        BraheError::EOPError(format!(
            "Missing {} value in TableEOPProvider data",
            field_name
        ))
    })
}

impl EarthOrientationProvider for TableEOPProvider {
    fn is_initialized(&self) -> bool {
        self.initialized
    }

    fn len(&self) -> usize {
        self.data.len()
    }

    fn eop_type(&self) -> EOPType {
        self.eop_type
    }

    fn extrapolation(&self) -> EOPExtrapolation {
        self.extrapolate
    }

    fn interpolation(&self) -> bool {
        self.interpolate
    }

    fn mjd_min(&self) -> f64 {
        self.mjd_min
    }

    fn mjd_max(&self) -> f64 {
        self.mjd_max
    }

    fn mjd_last_lod(&self) -> f64 {
        self.mjd_last_lod
    }

    fn mjd_last_dxdy(&self) -> f64 {
        self.mjd_last_dxdy
    }

    fn get_ut1_utc(&self, mjd: f64) -> Result<f64, BraheError> {
        if !self.initialized {
            return Err(BraheError::EOPError(
                "TableEOPProvider not initialized".to_string(),
            ));
        }

        if mjd < self.mjd_min {
            match self.extrapolate {
                EOPExtrapolation::Zero => Ok(0.0),
                EOPExtrapolation::Hold => Ok(self.get_data(self.mjd_min)?.2),
                EOPExtrapolation::Error => Err(BraheError::OutOfBoundsError(format!(
                    "Attempted EOP retrieval before start of loaded data. Accessed: {}, Min MJD: {}",
                    mjd, self.mjd_min
                ))),
            }
        } else if mjd <= self.mjd_max {
            if self.interpolate {
                let prev_opt = self.data.range(..=TableEOPKey(mjd)).next_back();
                let next_opt = self.data.range(TableEOPKey(mjd)..).next();

                match (prev_opt, next_opt) {
                    (Some((t1_key, data1)), Some((t2_key, data2))) => {
                        let t1 = t1_key.0;
                        let t2 = t2_key.0;
                        let y1 = data1.2;
                        let y2 = data2.2;

                        if t1 == t2 {
                            Ok(y1)
                        } else {
                            Ok((y2 - y1) / (t2 - t1) * (mjd - t1) + y1)
                        }
                    }
                    (Some((_t1_key, data1)), None) => Ok(data1.2),
                    (None, Some((_t2_key, data2))) => Ok(data2.2),
                    (None, None) => Err(BraheError::EOPError(
                        "No EOP data available for interpolation".to_string(),
                    )),
                }
            } else if let Some(data) = self.data.get(&TableEOPKey(mjd)) {
                Ok(data.2)
            } else {
                match self.data.range(..=TableEOPKey(mjd)).next_back() {
                    Some((_, data)) => Ok(data.2),
                    None => Ok(self.get_data(self.mjd_min)?.2),
                }
            }
        } else {
            match self.extrapolate {
                EOPExtrapolation::Zero => Ok(0.0),
                EOPExtrapolation::Hold => Ok(self.get_data(self.mjd_max)?.2),
                EOPExtrapolation::Error => Err(BraheError::OutOfBoundsError(format!(
                    "Attempted EOP retrieval beyond end of loaded data. Accessed: {}, Max MJD: {}",
                    mjd, self.mjd_max
                ))),
            }
        }
    }

    fn get_pm(&self, mjd: f64) -> Result<(f64, f64), BraheError> {
        if !self.initialized {
            return Err(BraheError::EOPError(
                "TableEOPProvider not initialized".to_string(),
            ));
        }

        if mjd < self.mjd_min {
            match self.extrapolate {
                EOPExtrapolation::Zero => Ok((0.0, 0.0)),
                EOPExtrapolation::Hold => {
                    let first = self.get_data(self.mjd_min)?;
                    Ok((first.0, first.1))
                }
                EOPExtrapolation::Error => Err(BraheError::OutOfBoundsError(format!(
                    "Attempted EOP retrieval before start of loaded data. Accessed: {}, Min MJD: {}",
                    mjd, self.mjd_min
                ))),
            }
        } else if mjd <= self.mjd_max {
            if self.interpolate {
                let prev_opt = self.data.range(..=TableEOPKey(mjd)).next_back();
                let next_opt = self.data.range(TableEOPKey(mjd)..).next();

                match (prev_opt, next_opt) {
                    (Some((t1_key, data1)), Some((t2_key, data2))) => {
                        let t1 = t1_key.0;
                        let t2 = t2_key.0;
                        let pm_x1 = data1.0;
                        let pm_x2 = data2.0;
                        let pm_y1 = data1.1;
                        let pm_y2 = data2.1;

                        if t1 == t2 {
                            Ok((pm_x1, pm_y1))
                        } else {
                            Ok((
                                (pm_x2 - pm_x1) / (t2 - t1) * (mjd - t1) + pm_x1,
                                (pm_y2 - pm_y1) / (t2 - t1) * (mjd - t1) + pm_y1,
                            ))
                        }
                    }
                    (Some((_t1_key, data1)), None) => Ok((data1.0, data1.1)),
                    (None, Some((_t2_key, data2))) => Ok((data2.0, data2.1)),
                    (None, None) => Err(BraheError::EOPError(
                        "No EOP data available for interpolation".to_string(),
                    )),
                }
            } else if let Some(data) = self.data.get(&TableEOPKey(mjd)) {
                Ok((data.0, data.1))
            } else {
                match self.data.range(..=TableEOPKey(mjd)).next_back() {
                    Some((_, data)) => Ok((data.0, data.1)),
                    None => {
                        let first = self.get_data(self.mjd_min)?;
                        Ok((first.0, first.1))
                    }
                }
            }
        } else {
            match self.extrapolate {
                EOPExtrapolation::Zero => Ok((0.0, 0.0)),
                EOPExtrapolation::Hold => {
                    let last = self.get_data(self.mjd_max)?;
                    Ok((last.0, last.1))
                }
                EOPExtrapolation::Error => Err(BraheError::OutOfBoundsError(format!(
                    "Attempted EOP retrieval beyond end of loaded data. Accessed: {}, Max MJD: {}",
                    mjd, self.mjd_max
                ))),
            }
        }
    }

    fn get_dxdy(&self, mjd: f64) -> Result<(f64, f64), BraheError> {
        if !self.initialized {
            return Err(BraheError::EOPError(
                "TableEOPProvider not initialized".to_string(),
            ));
        }

        if mjd < self.mjd_min {
            match self.extrapolate {
                EOPExtrapolation::Zero => Ok((0.0, 0.0)),
                EOPExtrapolation::Hold => {
                    let first = self.get_data(self.mjd_min)?;
                    Ok((
                        require_eop_field(first.3, "dX")?,
                        require_eop_field(first.4, "dY")?,
                    ))
                }
                EOPExtrapolation::Error => Err(BraheError::OutOfBoundsError(format!(
                    "Attempted EOP retrieval before start of loaded data. Accessed: {}, Min MJD: {}",
                    mjd, self.mjd_min
                ))),
            }
        } else if mjd <= self.mjd_last_dxdy {
            if self.interpolate {
                let prev_opt = self.data.range(..=TableEOPKey(mjd)).next_back();
                let next_opt = self.data.range(TableEOPKey(mjd)..).next();

                match (prev_opt, next_opt) {
                    (Some((t1_key, data1)), Some((t2_key, data2))) => {
                        let t1 = t1_key.0;
                        let t2 = t2_key.0;
                        let dx1 = require_eop_field(data1.3, "dX")?;
                        let dx2 = require_eop_field(data2.3, "dX")?;
                        let dy1 = require_eop_field(data1.4, "dY")?;
                        let dy2 = require_eop_field(data2.4, "dY")?;

                        if t1 == t2 {
                            Ok((dx1, dy1))
                        } else {
                            Ok((
                                (dx2 - dx1) / (t2 - t1) * (mjd - t1) + dx1,
                                (dy2 - dy1) / (t2 - t1) * (mjd - t1) + dy1,
                            ))
                        }
                    }
                    (Some((_t1_key, data1)), None) => Ok((
                        require_eop_field(data1.3, "dX")?,
                        require_eop_field(data1.4, "dY")?,
                    )),
                    (None, Some((_t2_key, data2))) => Ok((
                        require_eop_field(data2.3, "dX")?,
                        require_eop_field(data2.4, "dY")?,
                    )),
                    (None, None) => Err(BraheError::EOPError(
                        "No EOP data available for interpolation".to_string(),
                    )),
                }
            } else if let Some(data) = self.data.get(&TableEOPKey(mjd)) {
                Ok((
                    require_eop_field(data.3, "dX")?,
                    require_eop_field(data.4, "dY")?,
                ))
            } else {
                match self.data.range(..=TableEOPKey(mjd)).next_back() {
                    Some((_, data)) => Ok((
                        require_eop_field(data.3, "dX")?,
                        require_eop_field(data.4, "dY")?,
                    )),
                    None => {
                        let first = self.get_data(self.mjd_min)?;
                        Ok((
                            require_eop_field(first.3, "dX")?,
                            require_eop_field(first.4, "dY")?,
                        ))
                    }
                }
            }
        } else {
            match self.extrapolate {
                EOPExtrapolation::Zero => Ok((0.0, 0.0)),
                EOPExtrapolation::Hold => {
                    let last = self.get_data(self.mjd_last_dxdy)?;
                    Ok((
                        require_eop_field(last.3, "dX")?,
                        require_eop_field(last.4, "dY")?,
                    ))
                }
                EOPExtrapolation::Error => Err(BraheError::OutOfBoundsError(format!(
                    "Attempted EOP retrieval beyond end of loaded data. Accessed: {}, Max MJD: {}",
                    mjd, self.mjd_max
                ))),
            }
        }
    }

    fn get_lod(&self, mjd: f64) -> Result<f64, BraheError> {
        if !self.initialized {
            return Err(BraheError::EOPError(
                "TableEOPProvider not initialized".to_string(),
            ));
        }

        if mjd < self.mjd_min {
            match self.extrapolate {
                EOPExtrapolation::Zero => Ok(0.0),
                EOPExtrapolation::Hold => {
                    Ok(require_eop_field(self.get_data(self.mjd_min)?.5, "LOD")?)
                }
                EOPExtrapolation::Error => Err(BraheError::OutOfBoundsError(format!(
                    "Attempted EOP retrieval before start of loaded data. Accessed: {}, Min MJD: {}",
                    mjd, self.mjd_min
                ))),
            }
        } else if mjd <= self.mjd_last_lod {
            if self.interpolate {
                let prev_opt = self.data.range(..=TableEOPKey(mjd)).next_back();
                let next_opt = self.data.range(TableEOPKey(mjd)..).next();

                match (prev_opt, next_opt) {
                    (Some((t1_key, data1)), Some((t2_key, data2))) => {
                        let t1 = t1_key.0;
                        let t2 = t2_key.0;
                        let y1 = require_eop_field(data1.5, "LOD")?;
                        let y2 = require_eop_field(data2.5, "LOD")?;

                        if t1 == t2 {
                            Ok(y1)
                        } else {
                            Ok((y2 - y1) / (t2 - t1) * (mjd - t1) + y1)
                        }
                    }
                    (Some((_t1_key, data1)), None) => Ok(require_eop_field(data1.5, "LOD")?),
                    (None, Some((_t2_key, data2))) => Ok(require_eop_field(data2.5, "LOD")?),
                    (None, None) => Err(BraheError::EOPError(
                        "No EOP data available for interpolation".to_string(),
                    )),
                }
            } else if let Some(data) = self.data.get(&TableEOPKey(mjd)) {
                Ok(require_eop_field(data.5, "LOD")?)
            } else {
                match self.data.range(..=TableEOPKey(mjd)).next_back() {
                    Some((_, data)) => Ok(require_eop_field(data.5, "LOD")?),
                    None => Ok(require_eop_field(self.get_data(self.mjd_min)?.5, "LOD")?),
                }
            }
        } else {
            match self.extrapolate {
                EOPExtrapolation::Zero => Ok(0.0),
                EOPExtrapolation::Hold => Ok(require_eop_field(
                    self.get_data(self.mjd_last_lod)?.5,
                    "LOD",
                )?),
                EOPExtrapolation::Error => Err(BraheError::OutOfBoundsError(format!(
                    "Attempted EOP retrieval beyond end of loaded data. Accessed: {}, Max MJD: {}",
                    mjd, self.mjd_max
                ))),
            }
        }
    }

    #[allow(non_snake_case)]
    fn get_eop(&self, mjd: f64) -> Result<(f64, f64, f64, f64, f64, f64), BraheError> {
        let (pm_x, pm_y) = self.get_pm(mjd)?;
        let ut1_utc = self.get_ut1_utc(mjd)?;
        let (dX, dY) = self.get_dxdy(mjd)?;
        let lod = self.get_lod(mjd)?;
        Ok((pm_x, pm_y, ut1_utc, dX, dY, lod))
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::eop::static_provider::StaticEOPProvider;
    use approx::assert_abs_diff_eq;

    fn make_test_entries() -> Vec<EOPEntry> {
        vec![
            (
                59000.0,
                0.1,
                0.2,
                0.3,
                Some(0.001),
                Some(0.0001),
                Some(0.0002),
            ),
            (
                59001.0,
                0.11,
                0.21,
                0.31,
                Some(0.0011),
                Some(0.00011),
                Some(0.00021),
            ),
            (
                59002.0,
                0.12,
                0.22,
                0.32,
                Some(0.0012),
                Some(0.00012),
                Some(0.00022),
            ),
        ]
    }

    #[test]
    fn test_from_entries_basic() {
        let entries = make_test_entries();
        let provider =
            TableEOPProvider::from_entries(entries, true, EOPExtrapolation::Hold).unwrap();

        assert!(provider.is_initialized());
        assert_eq!(provider.len(), 3);
        assert_eq!(provider.mjd_min(), 59000.0);
        assert_eq!(provider.mjd_max(), 59002.0);
        assert_eq!(provider.eop_type(), EOPType::C04);
        assert_eq!(provider.extrapolation(), EOPExtrapolation::Hold);
        assert!(provider.interpolation());
    }

    #[test]
    fn test_from_entries_empty() {
        let result = TableEOPProvider::from_entries(vec![], true, EOPExtrapolation::Hold);
        assert!(result.is_err());
    }

    #[test]
    fn test_from_entries_single() {
        let entries = vec![(
            59000.0,
            0.1,
            0.2,
            0.3,
            Some(0.001),
            Some(0.0001),
            Some(0.0002),
        )];
        let provider =
            TableEOPProvider::from_entries(entries, false, EOPExtrapolation::Zero).unwrap();

        assert_eq!(provider.len(), 1);
        assert_eq!(provider.mjd_min(), 59000.0);
        assert_eq!(provider.mjd_max(), 59000.0);
    }

    #[test]
    fn test_get_ut1_utc_exact() {
        let entries = make_test_entries();
        let provider =
            TableEOPProvider::from_entries(entries, true, EOPExtrapolation::Hold).unwrap();

        assert_abs_diff_eq!(provider.get_ut1_utc(59000.0).unwrap(), 0.3, epsilon = 1e-12);
        assert_abs_diff_eq!(
            provider.get_ut1_utc(59001.0).unwrap(),
            0.31,
            epsilon = 1e-12
        );
    }

    #[test]
    fn test_get_ut1_utc_interpolated() {
        let entries = make_test_entries();
        let provider =
            TableEOPProvider::from_entries(entries, true, EOPExtrapolation::Hold).unwrap();

        let interp = provider.get_ut1_utc(59000.5).unwrap();
        assert_abs_diff_eq!(interp, (0.3 + 0.31) / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_get_ut1_utc_no_interpolation() {
        let entries = make_test_entries();
        let provider =
            TableEOPProvider::from_entries(entries, false, EOPExtrapolation::Hold).unwrap();

        // Should return previous value when not interpolating
        let val = provider.get_ut1_utc(59000.5).unwrap();
        assert_abs_diff_eq!(val, 0.3, epsilon = 1e-12);
    }

    #[test]
    fn test_get_ut1_utc_extrapolation_hold() {
        let entries = make_test_entries();
        let provider =
            TableEOPProvider::from_entries(entries, true, EOPExtrapolation::Hold).unwrap();

        // Beyond max
        let val = provider.get_ut1_utc(99999.0).unwrap();
        assert_abs_diff_eq!(val, 0.32, epsilon = 1e-12);

        // Before min
        let val = provider.get_ut1_utc(50000.0).unwrap();
        assert_abs_diff_eq!(val, 0.3, epsilon = 1e-12);
    }

    #[test]
    fn test_get_ut1_utc_extrapolation_zero() {
        let entries = make_test_entries();
        let provider =
            TableEOPProvider::from_entries(entries, true, EOPExtrapolation::Zero).unwrap();

        let val = provider.get_ut1_utc(99999.0).unwrap();
        assert_abs_diff_eq!(val, 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_get_ut1_utc_extrapolation_error() {
        let entries = make_test_entries();
        let provider =
            TableEOPProvider::from_entries(entries, true, EOPExtrapolation::Error).unwrap();

        assert!(provider.get_ut1_utc(99999.0).is_err());
        assert!(provider.get_ut1_utc(50000.0).is_err());
    }

    #[test]
    fn test_get_pm_exact() {
        let entries = make_test_entries();
        let provider =
            TableEOPProvider::from_entries(entries, true, EOPExtrapolation::Hold).unwrap();

        let (pm_x, pm_y) = provider.get_pm(59000.0).unwrap();
        assert_abs_diff_eq!(pm_x, 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(pm_y, 0.2, epsilon = 1e-12);
    }

    #[test]
    fn test_get_pm_interpolated() {
        let entries = make_test_entries();
        let provider =
            TableEOPProvider::from_entries(entries, true, EOPExtrapolation::Hold).unwrap();

        let (pm_x, pm_y) = provider.get_pm(59000.5).unwrap();
        assert_abs_diff_eq!(pm_x, (0.1 + 0.11) / 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(pm_y, (0.2 + 0.21) / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_get_dxdy_exact() {
        let entries = make_test_entries();
        let provider =
            TableEOPProvider::from_entries(entries, true, EOPExtrapolation::Hold).unwrap();

        let (dx, dy) = provider.get_dxdy(59000.0).unwrap();
        assert_abs_diff_eq!(dx, 0.0001, epsilon = 1e-12);
        assert_abs_diff_eq!(dy, 0.0002, epsilon = 1e-12);
    }

    #[test]
    fn test_get_dxdy_interpolated() {
        let entries = make_test_entries();
        let provider =
            TableEOPProvider::from_entries(entries, true, EOPExtrapolation::Hold).unwrap();

        let (dx, dy) = provider.get_dxdy(59000.5).unwrap();
        assert_abs_diff_eq!(dx, (0.0001 + 0.00011) / 2.0, epsilon = 1e-12);
        assert_abs_diff_eq!(dy, (0.0002 + 0.00021) / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_get_lod_exact() {
        let entries = make_test_entries();
        let provider =
            TableEOPProvider::from_entries(entries, true, EOPExtrapolation::Hold).unwrap();

        let lod = provider.get_lod(59000.0).unwrap();
        assert_abs_diff_eq!(lod, 0.001, epsilon = 1e-12);
    }

    #[test]
    fn test_get_lod_interpolated() {
        let entries = make_test_entries();
        let provider =
            TableEOPProvider::from_entries(entries, true, EOPExtrapolation::Hold).unwrap();

        let lod = provider.get_lod(59000.5).unwrap();
        assert_abs_diff_eq!(lod, (0.001 + 0.0011) / 2.0, epsilon = 1e-12);
    }

    #[test]
    fn test_get_eop_exact() {
        let entries = make_test_entries();
        let provider =
            TableEOPProvider::from_entries(entries, true, EOPExtrapolation::Hold).unwrap();

        let (pm_x, pm_y, ut1_utc, dx, dy, lod) = provider.get_eop(59000.0).unwrap();
        assert_abs_diff_eq!(pm_x, 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(pm_y, 0.2, epsilon = 1e-12);
        assert_abs_diff_eq!(ut1_utc, 0.3, epsilon = 1e-12);
        assert_abs_diff_eq!(dx, 0.0001, epsilon = 1e-12);
        assert_abs_diff_eq!(dy, 0.0002, epsilon = 1e-12);
        assert_abs_diff_eq!(lod, 0.001, epsilon = 1e-12);
    }

    #[test]
    fn test_from_perturbed_zero_perturbation() {
        let base = StaticEOPProvider::from_values((0.1, 0.2, 0.3, 0.4, 0.5, 0.6));
        let perturbations = HashMap::new();

        let provider = TableEOPProvider::from_perturbed(
            &base,
            59000.0,
            59002.0,
            1.0,
            &perturbations,
            true,
            EOPExtrapolation::Hold,
        )
        .unwrap();

        assert_eq!(provider.len(), 3);
        assert_abs_diff_eq!(provider.get_ut1_utc(59000.0).unwrap(), 0.3, epsilon = 1e-12);
        let (pm_x, pm_y) = provider.get_pm(59000.0).unwrap();
        assert_abs_diff_eq!(pm_x, 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(pm_y, 0.2, epsilon = 1e-12);
    }

    #[test]
    fn test_from_perturbed_with_perturbation() {
        let base = StaticEOPProvider::from_values((0.1, 0.2, 0.3, 0.4, 0.5, 0.6));

        let mut perturbations = HashMap::new();
        perturbations.insert("ut1_utc".to_string(), 0.001);
        perturbations.insert("pm_x".to_string(), 0.01);

        let provider = TableEOPProvider::from_perturbed(
            &base,
            59000.0,
            59002.0,
            1.0,
            &perturbations,
            true,
            EOPExtrapolation::Hold,
        )
        .unwrap();

        assert_eq!(provider.len(), 3);
        assert_abs_diff_eq!(
            provider.get_ut1_utc(59000.0).unwrap(),
            0.301,
            epsilon = 1e-12
        );
        let (pm_x, pm_y) = provider.get_pm(59000.0).unwrap();
        assert_abs_diff_eq!(pm_x, 0.11, epsilon = 1e-12);
        assert_abs_diff_eq!(pm_y, 0.2, epsilon = 1e-12);
    }

    #[test]
    fn test_from_perturbed_all_fields() {
        let base = StaticEOPProvider::from_values((0.1, 0.2, 0.3, 0.4, 0.5, 0.6));

        let mut perturbations = HashMap::new();
        perturbations.insert("pm_x".to_string(), 0.01);
        perturbations.insert("pm_y".to_string(), 0.02);
        perturbations.insert("ut1_utc".to_string(), 0.03);
        perturbations.insert("lod".to_string(), 0.04);
        perturbations.insert("dX".to_string(), 0.05);
        perturbations.insert("dY".to_string(), 0.06);

        let provider = TableEOPProvider::from_perturbed(
            &base,
            59000.0,
            59000.0,
            1.0,
            &perturbations,
            false,
            EOPExtrapolation::Zero,
        )
        .unwrap();

        let (pm_x, pm_y, ut1_utc, dx, dy, lod) = provider.get_eop(59000.0).unwrap();
        assert_abs_diff_eq!(pm_x, 0.11, epsilon = 1e-12);
        assert_abs_diff_eq!(pm_y, 0.22, epsilon = 1e-12);
        assert_abs_diff_eq!(ut1_utc, 0.33, epsilon = 1e-12);
        assert_abs_diff_eq!(dx, 0.45, epsilon = 1e-12);
        assert_abs_diff_eq!(dy, 0.56, epsilon = 1e-12);
        assert_abs_diff_eq!(lod, 0.64, epsilon = 1e-12);
    }

    #[test]
    fn test_display_format() {
        let entries = make_test_entries();
        let provider =
            TableEOPProvider::from_entries(entries, true, EOPExtrapolation::Hold).unwrap();
        let display_string = format!("{}", provider);

        assert!(display_string.contains("TableEOPProvider"));
        assert!(display_string.contains("3"));
        assert!(display_string.contains("59000"));
        assert!(display_string.contains("59002"));
    }

    #[test]
    fn test_debug_format() {
        let entries = make_test_entries();
        let provider =
            TableEOPProvider::from_entries(entries, true, EOPExtrapolation::Hold).unwrap();
        let debug_string = format!("{:?}", provider);

        assert!(debug_string.contains("TableEOPProvider"));
        assert!(debug_string.contains("Length: 3"));
    }

    #[test]
    fn test_mjd_last_lod_tracking() {
        // Test that mjd_last_lod tracks the last entry with LOD present
        let entries = vec![
            (
                59000.0,
                0.1,
                0.2,
                0.3,
                Some(0.001),
                Some(0.0001),
                Some(0.0002),
            ),
            (59001.0, 0.1, 0.2, 0.3, None, Some(0.0001), Some(0.0002)),
            (59002.0, 0.1, 0.2, 0.3, None, None, None),
        ];
        let provider =
            TableEOPProvider::from_entries(entries, true, EOPExtrapolation::Zero).unwrap();

        assert_eq!(provider.mjd_last_lod(), 59000.0);
        assert_eq!(provider.mjd_last_dxdy(), 59001.0);
    }

    #[test]
    fn test_extrapolation_hold_below_min() {
        let entries = make_test_entries();
        let provider =
            TableEOPProvider::from_entries(entries, true, EOPExtrapolation::Hold).unwrap();

        let (pm_x, pm_y) = provider.get_pm(50000.0).unwrap();
        assert_abs_diff_eq!(pm_x, 0.1, epsilon = 1e-12);
        assert_abs_diff_eq!(pm_y, 0.2, epsilon = 1e-12);

        let (dx, dy) = provider.get_dxdy(50000.0).unwrap();
        assert_abs_diff_eq!(dx, 0.0001, epsilon = 1e-12);
        assert_abs_diff_eq!(dy, 0.0002, epsilon = 1e-12);

        let lod = provider.get_lod(50000.0).unwrap();
        assert_abs_diff_eq!(lod, 0.001, epsilon = 1e-12);
    }

    #[test]
    fn test_extrapolation_zero_all_methods() {
        let entries = make_test_entries();
        let provider =
            TableEOPProvider::from_entries(entries, true, EOPExtrapolation::Zero).unwrap();

        assert_eq!(provider.get_ut1_utc(99999.0).unwrap(), 0.0);
        assert_eq!(provider.get_pm(99999.0).unwrap(), (0.0, 0.0));
        assert_eq!(provider.get_dxdy(99999.0).unwrap(), (0.0, 0.0));
        assert_eq!(provider.get_lod(99999.0).unwrap(), 0.0);
    }

    #[test]
    fn test_clone() {
        let entries = make_test_entries();
        let provider =
            TableEOPProvider::from_entries(entries, true, EOPExtrapolation::Hold).unwrap();
        let cloned = provider.clone();

        assert_eq!(cloned.len(), provider.len());
        assert_eq!(cloned.mjd_min(), provider.mjd_min());
        assert_eq!(cloned.mjd_max(), provider.mjd_max());
        assert_abs_diff_eq!(
            cloned.get_ut1_utc(59000.0).unwrap(),
            provider.get_ut1_utc(59000.0).unwrap(),
            epsilon = 1e-12
        );
    }
}
