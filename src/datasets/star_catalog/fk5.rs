/*!
 * FK5 star catalog: record type, container, and fixed-width text parser.
 *
 * The FK5 (Fifth Fundamental Catalogue) is a fixed, J2000.0 catalog of
 * bright stars. Source data is a fixed-width text file (CDS format) with
 * one 190-column line per star.
 */

use polars::prelude::*;

use crate::constants::AngleFormat;
use crate::coordinates::position_radec_to_inertial;
use crate::datasets::star_catalog::traits::StarRecord;
use crate::datasets::star_catalog::{opt_f64, opt_string};
use crate::math::SVector3;
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;

/// A single record from the FK5 star catalog.
///
/// Positions and proper motions are J2000.0 (the FK5 system's native
/// equinox/equator); the legacy B1950 columns and formal-error columns of
/// the source file are not represented.
#[derive(Debug, Clone)]
pub struct FK5Record {
    /// FK5 catalog running number (bytes 1-4).
    pub fk5_id: u32,
    /// Right ascension, J2000.0. Units: *deg*
    pub ra: f64,
    /// Declination, J2000.0. Units: *deg*
    pub dec: f64,
    /// Proper motion in right ascension (μ_α* = μ_α cos δ), J2000.0. Units: *mas/yr*
    pub pm_ra: f64,
    /// Proper motion in declination, J2000.0. Units: *mas/yr*
    pub pm_dec: f64,
    /// Mean epoch of right ascension observations, minus 1900. Units: *yr*
    pub epoch_ra_1900: Option<f64>,
    /// Mean epoch of declination observations, minus 1900. Units: *yr*
    pub epoch_dec_1900: Option<f64>,
    /// Visual magnitude. Units: *mag*
    pub vmag: Option<f64>,
    /// Visual magnitude quality/note flag.
    pub vmag_flag: Option<String>,
    /// Spectral type.
    pub spectral_type: Option<String>,
    /// Trigonometric parallax. Units: *mas*
    pub parallax: Option<f64>,
    /// Radial velocity. Units: *km/s*
    pub radial_velocity: Option<f64>,
    /// Henry Draper (HD) catalog identifier.
    pub hd_id: Option<String>,
    /// Durchmusterung (DM) catalog identifier.
    pub dm_id: Option<String>,
    /// Groombridge Catalogue (GC) identifier.
    pub gc_id: Option<String>,
}

impl StarRecord for FK5Record {
    fn id(&self) -> String {
        format!("FK5 {}", self.fk5_id)
    }

    fn name(&self) -> Option<String> {
        self.hd_id
            .as_ref()
            .map(|hd| format!("HD {hd}"))
            .or_else(|| self.dm_id.clone())
    }

    fn ra(&self) -> f64 {
        self.ra
    }

    fn dec(&self) -> f64 {
        self.dec
    }

    fn pm_ra(&self) -> Option<f64> {
        Some(self.pm_ra)
    }

    fn pm_dec(&self) -> Option<f64> {
        Some(self.pm_dec)
    }

    fn parallax(&self) -> Option<f64> {
        self.parallax
    }

    fn radial_velocity(&self) -> Option<f64> {
        self.radial_velocity
    }

    fn epoch(&self) -> Epoch {
        // FK5 positions/proper motions are referred to J2000.0.
        Epoch::from_jd(2451545.0, TimeSystem::TT)
    }

    fn magnitude(&self) -> Option<f64> {
        self.vmag
    }
}

/// Extract a 1-indexed, inclusive byte range from a fixed-width line and trim it.
///
/// Returns an empty string (rather than panicking) if the line is too short
/// to contain the requested range, so short/truncated lines degrade to
/// missing (`None`) optional fields instead of failing the whole parse.
fn field(line: &str, start_1idx: usize, end_1idx: usize) -> &str {
    line.get(start_1idx - 1..end_1idx).unwrap_or("").trim()
}

/// Parse a single FK5 fixed-width catalog line into a record.
///
/// # Arguments
/// * `line` - One line of the FK5 `FK5_Catalog.txt` fixed-width file
///
/// # Returns
/// * `Result<FK5Record, BraheError>` - Parsed record, or a `ParseError` if a
///   required field (identifier, RA, or Dec) is missing/malformed
fn parse_fk5_line(line: &str) -> Result<FK5Record, BraheError> {
    let fk5_id: u32 = field(line, 1, 4)
        .parse()
        .map_err(|_| BraheError::ParseError(format!("FK5: invalid fk5_id in line: {line:?}")))?;

    let ra_h: f64 = field(line, 6, 7)
        .parse()
        .map_err(|_| BraheError::ParseError(format!("FK5: invalid RA hours in line: {line:?}")))?;
    let ra_m: f64 = field(line, 9, 10).parse().map_err(|_| {
        BraheError::ParseError(format!("FK5: invalid RA minutes in line: {line:?}"))
    })?;
    let ra_s: f64 = field(line, 12, 17).parse().map_err(|_| {
        BraheError::ParseError(format!("FK5: invalid RA seconds in line: {line:?}"))
    })?;
    let ra = 15.0 * (ra_h + ra_m / 60.0 + ra_s / 3600.0);

    // Declination sign is a standalone column, applied to the whole
    // d/m/s magnitude (so a "-0" degrees case is still handled correctly,
    // since the sign never has a chance to be absorbed into `dec_d`).
    let dec_sign = if field(line, 27, 27) == "-" {
        -1.0
    } else {
        1.0
    };
    let dec_d: f64 = field(line, 28, 29).parse().map_err(|_| {
        BraheError::ParseError(format!("FK5: invalid Dec degrees in line: {line:?}"))
    })?;
    let dec_m: f64 = field(line, 31, 32).parse().map_err(|_| {
        BraheError::ParseError(format!("FK5: invalid Dec minutes in line: {line:?}"))
    })?;
    let dec_s: f64 = field(line, 34, 38).parse().map_err(|_| {
        BraheError::ParseError(format!("FK5: invalid Dec seconds in line: {line:?}"))
    })?;
    let dec = dec_sign * (dec_d + dec_m / 60.0 + dec_s / 3600.0);

    // pmRA is tabulated in seconds of time per tropical century; convert to
    // mas/yr: (s/century * 15 deg/s * cos(dec)) -> arcsec/century, then
    // *1000/100 -> mas/yr.
    let pmra_s_per_century: f64 = field(line, 19, 25)
        .parse()
        .map_err(|_| BraheError::ParseError(format!("FK5: invalid pmRA in line: {line:?}")))?;
    let pm_ra = pmra_s_per_century * 15.0 * dec.to_radians().cos() * 1000.0 / 100.0;

    // pmDE is tabulated in arcsec per tropical century; convert to mas/yr.
    let pmde_arcsec_per_century: f64 = field(line, 40, 46)
        .parse()
        .map_err(|_| BraheError::ParseError(format!("FK5: invalid pmDE in line: {line:?}")))?;
    let pm_dec = pmde_arcsec_per_century * 1000.0 / 100.0;

    let epoch_ra_1900 = opt_f64(field(line, 90, 94));
    let epoch_dec_1900 = opt_f64(field(line, 107, 111));
    let vmag = opt_f64(field(line, 124, 128));
    let vmag_flag = opt_string(field(line, 129, 129));
    let spectral_type = opt_string(field(line, 131, 137));
    let parallax = opt_f64(field(line, 139, 144)).map(|plx_arcsec| plx_arcsec * 1000.0);
    let radial_velocity = opt_f64(field(line, 147, 152));
    let hd_id = opt_string(field(line, 167, 172));
    let dm_id = opt_string(field(line, 174, 183));
    let gc_id = opt_string(field(line, 186, 190));

    Ok(FK5Record {
        fk5_id,
        ra,
        dec,
        pm_ra,
        pm_dec,
        epoch_ra_1900,
        epoch_dec_1900,
        vmag,
        vmag_flag,
        spectral_type,
        parallax,
        radial_velocity,
        hd_id,
        dm_id,
        gc_id,
    })
}

/// Parse the full FK5 fixed-width catalog text into records.
///
/// Blank lines are skipped; every other line is parsed with [`parse_fk5_line`].
///
/// # Arguments
/// * `data` - Raw contents of `FK5_Catalog.txt`
///
/// # Returns
/// * `Result<Vec<FK5Record>, BraheError>` - Parsed records, in file order
pub(crate) fn parse_fk5_text(data: &str) -> Result<Vec<FK5Record>, BraheError> {
    data.lines()
        .filter(|line| !line.trim().is_empty())
        .map(parse_fk5_line)
        .collect()
}

/// Container for FK5 star catalog records with lookup and filter methods.
///
/// Wraps a `Vec<FK5Record>` and provides lookup by FK5 identifier,
/// magnitude filtering, and cone-search filtering. Filter methods return a
/// new `FK5Catalog` instance (immutable pattern), enabling method chaining.
pub struct FK5Catalog {
    records: Vec<FK5Record>,
}

impl FK5Catalog {
    /// Create a new FK5Catalog from a vector of records.
    pub fn new(records: Vec<FK5Record>) -> Self {
        Self { records }
    }

    /// Number of records in the catalog.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// Check if the catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Get a reference to all records.
    pub fn records(&self) -> &[FK5Record] {
        &self.records
    }

    /// Consume the container and return the underlying records.
    pub fn into_records(self) -> Vec<FK5Record> {
        self.records
    }

    /// Look up a record by FK5 catalog identifier.
    pub fn get_by_id(&self, fk5_id: u32) -> Option<&FK5Record> {
        self.records.iter().find(|r| r.fk5_id == fk5_id)
    }

    /// Filter records to those with visual magnitude at or brighter than `max_mag`.
    ///
    /// Records with unknown magnitude are excluded. Recall that smaller
    /// (more negative) magnitudes are brighter, so this keeps
    /// `vmag <= max_mag`.
    ///
    /// # Arguments
    /// * `max_mag` - Faintest visual magnitude to include. Units: *mag*
    pub fn filter_by_magnitude(&self, max_mag: f64) -> Self {
        Self::new(
            self.records
                .iter()
                .filter(|r| r.vmag.is_some_and(|v| v <= max_mag))
                .cloned()
                .collect(),
        )
    }

    /// Filter records to those within an angular radius of a cone center.
    ///
    /// Angular separation is computed from the dot product of each record's
    /// [`StarRecord::unit_vector`] with the cone center's unit vector.
    ///
    /// # Arguments
    /// * `ra` - Cone center right ascension. Units: (*angle*)
    /// * `dec` - Cone center declination. Units: (*angle*)
    /// * `radius` - Cone half-angle. Units: (*angle*)
    /// * `angle_format` - Format for `ra`, `dec`, and `radius` (Radians or Degrees)
    pub fn filter_by_cone(
        &self,
        ra: f64,
        dec: f64,
        radius: f64,
        angle_format: AngleFormat,
    ) -> Self {
        let center = position_radec_to_inertial(SVector3::new(ra, dec, 1.0), angle_format);
        let radius_rad = match angle_format {
            AngleFormat::Degrees => radius.to_radians(),
            AngleFormat::Radians => radius,
        };

        Self::new(
            self.records
                .iter()
                .filter(|r| {
                    let cos_sep = center.dot(&r.unit_vector()).clamp(-1.0, 1.0);
                    cos_sep.acos() <= radius_rad
                })
                .cloned()
                .collect(),
        )
    }

    /// Convert the catalog to a Polars DataFrame.
    ///
    /// One row per record, one column per `FK5Record` field, using the same
    /// field names. Missing optional values become nulls.
    ///
    /// # Returns
    /// * `Result<DataFrame, BraheError>` - Catalog data as a DataFrame
    pub fn to_dataframe(&self) -> Result<DataFrame, BraheError> {
        let u32_col: Column = Series::new(
            "fk5_id".into(),
            self.records.iter().map(|r| r.fk5_id).collect::<Vec<_>>(),
        )
        .into();

        let f64_col = |name: &str, f: fn(&FK5Record) -> f64| -> Column {
            Series::new(name.into(), self.records.iter().map(f).collect::<Vec<_>>()).into()
        };

        let opt_f64_col = |name: &str, f: fn(&FK5Record) -> Option<f64>| -> Column {
            Series::new(name.into(), self.records.iter().map(f).collect::<Vec<_>>()).into()
        };

        let opt_str_col = |name: &str, f: fn(&FK5Record) -> Option<&str>| -> Column {
            Series::new(name.into(), self.records.iter().map(f).collect::<Vec<_>>()).into()
        };

        let columns = vec![
            u32_col,
            f64_col("ra", |r| r.ra),
            f64_col("dec", |r| r.dec),
            f64_col("pm_ra", |r| r.pm_ra),
            f64_col("pm_dec", |r| r.pm_dec),
            opt_f64_col("epoch_ra_1900", |r| r.epoch_ra_1900),
            opt_f64_col("epoch_dec_1900", |r| r.epoch_dec_1900),
            opt_f64_col("vmag", |r| r.vmag),
            opt_str_col("vmag_flag", |r| r.vmag_flag.as_deref()),
            opt_str_col("spectral_type", |r| r.spectral_type.as_deref()),
            opt_f64_col("parallax", |r| r.parallax),
            opt_f64_col("radial_velocity", |r| r.radial_velocity),
            opt_str_col("hd_id", |r| r.hd_id.as_deref()),
            opt_str_col("dm_id", |r| r.dm_id.as_deref()),
            opt_str_col("gc_id", |r| r.gc_id.as_deref()),
        ];

        DataFrame::new(self.records.len(), columns)
            .map_err(|e| BraheError::Error(format!("Failed to create FK5 DataFrame: {}", e)))
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use serial_test::parallel;

    const SAMPLE: &str = include_str!("../../../test_assets/star_catalog/FK5_Catalog_sample.txt");

    #[test]
    #[parallel]
    fn test_fk5_parse_sample() {
        let records = parse_fk5_text(SAMPLE).unwrap();
        assert_eq!(records.len(), 10);

        let r1 = &records[0];
        assert_eq!(r1.fk5_id, 1);
        assert_abs_diff_eq!(r1.ra, 2.096937500, epsilon = 1e-9);
        assert_abs_diff_eq!(r1.dec, 29.090438888888887, epsilon = 1e-9);
        assert_abs_diff_eq!(r1.pm_ra, 136.19004725467414, epsilon = 1e-6);
        assert_abs_diff_eq!(r1.pm_dec, -163.3, epsilon = 1e-9);
        assert_abs_diff_eq!(r1.epoch_ra_1900.unwrap(), 43.31, epsilon = 1e-9);
        assert_abs_diff_eq!(r1.epoch_dec_1900.unwrap(), 33.00, epsilon = 1e-9);
        assert_abs_diff_eq!(r1.vmag.unwrap(), 2.06, epsilon = 1e-9);
        assert_eq!(r1.spectral_type.as_deref(), Some("A0p"));
        assert_abs_diff_eq!(r1.parallax.unwrap(), 24.0, epsilon = 1e-9);
        assert_abs_diff_eq!(r1.radial_velocity.unwrap(), -11.7, epsilon = 1e-9);
        assert_eq!(r1.hd_id.as_deref(), Some("358"));
        assert_eq!(r1.dm_id.as_deref(), Some("BD+28    4"));
        assert_eq!(r1.gc_id.as_deref(), Some("127"));
        assert_eq!(r1.vmag_flag, None);
    }

    #[test]
    #[parallel]
    fn test_fk5_catalog_filters() {
        let records = parse_fk5_text(SAMPLE).unwrap();
        let catalog = FK5Catalog::new(records);

        let bright = catalog.filter_by_magnitude(3.0);
        let expected = catalog
            .records()
            .iter()
            .filter(|r| r.vmag.is_some_and(|v| v <= 3.0))
            .count();
        assert_eq!(bright.len(), expected);
        assert!(bright.len() < catalog.len());
        assert!(bright.records().iter().all(|r| r.vmag.unwrap() <= 3.0));

        let r1 = catalog.get_by_id(1).unwrap();
        let cone = catalog.filter_by_cone(r1.ra, r1.dec, 0.1, AngleFormat::Degrees);
        assert!(cone.get_by_id(1).is_some());
    }

    #[test]
    #[parallel]
    fn test_fk5_star_record_trait() {
        let records = parse_fk5_text(SAMPLE).unwrap();
        let r1 = &records[0];

        assert_eq!(r1.id(), "FK5 1");
        assert_eq!(r1.name(), Some("HD 358".to_string()));

        let u = r1.unit_vector();
        assert_abs_diff_eq!(u.norm(), 1.0, epsilon = 1e-12);

        let (ra, dec) = r1.radec_at_epoch(r1.epoch(), AngleFormat::Degrees);
        assert_abs_diff_eq!(ra, r1.ra, epsilon = 1e-9);
        assert_abs_diff_eq!(dec, r1.dec, epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    fn test_fk5_to_dataframe() {
        let records = parse_fk5_text(SAMPLE).unwrap();
        let catalog = FK5Catalog::new(records);
        let df = catalog.to_dataframe().unwrap();

        assert_eq!(df.height(), 10);
        assert_eq!(df.width(), 15);

        let vmag = df.column("vmag").unwrap();
        assert_abs_diff_eq!(vmag.f64().unwrap().get(0).unwrap(), 2.06, epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    #[cfg_attr(not(feature = "integration"), ignore)]
    fn test_fk5_full_download() {
        let catalog = crate::datasets::star_catalog::get_fk5_catalog(None).unwrap();
        assert_eq!(catalog.len(), 1535);
    }
}
