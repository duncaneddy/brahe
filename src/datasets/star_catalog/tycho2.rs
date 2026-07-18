/*!
 * Tycho-2 star catalog: record type, container, and pipe-delimited text parser.
 *
 * The Tycho-2 Catalogue is a fixed, astrometric catalog of ~2.54 million
 * stars derived from the ESA Hipparcos satellite's star mapper data,
 * referred to the ICRS. Source data is a pipe-delimited text file with one
 * 32-field line per star.
 */

use polars::prelude::*;

use crate::constants::AngleFormat;
use crate::coordinates::position_radec_to_inertial;
use crate::datasets::star_catalog::traits::StarRecord;
use crate::datasets::star_catalog::{opt_f64, opt_string};
use crate::math::SVector3;
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;

/// A single record from the Tycho-2 star catalog.
///
/// A small fraction of entries (`pflag == "X"`, see [`Tycho2Record::pflag`])
/// have no mean astrometric solution: their `ra`/`dec`/`pm_ra`/`pm_dec`/
/// `epoch_ra`/`epoch_dec` fields are all `None`. The [`StarRecord`]
/// implementation falls back to the always-present observed position
/// (`ra_observed`/`dec_observed`, epoch ~1991.5) for such records.
///
/// Tycho-2 does not carry a parallax or radial velocity column, so
/// [`StarRecord::parallax`] and [`StarRecord::radial_velocity`] always
/// return `None` for Tycho-2 records.
#[derive(Debug, Clone)]
pub struct Tycho2Record {
    /// Tycho-2 identifier, first component (GSC region number).
    pub tyc1: u16,
    /// Tycho-2 identifier, second component (running number within region).
    pub tyc2: u16,
    /// Tycho-2 identifier, third component (component number, for double/multiple entries).
    pub tyc3: u8,
    /// Mean position flag: `None`/blank for a normal entry, `"P"` if the
    /// mean position was obtained from the photocenter, `"X"` if no mean
    /// position could be computed (see struct docs).
    pub pflag: Option<String>,
    /// Mean right ascension, ICRS. `None` when `pflag == "X"`. Units: *deg*
    pub ra: Option<f64>,
    /// Mean declination, ICRS. `None` when `pflag == "X"`. Units: *deg*
    pub dec: Option<f64>,
    /// Proper motion in right ascension (μ_α* = μ_α cos δ). Units: *mas/yr*
    pub pm_ra: Option<f64>,
    /// Proper motion in declination. Units: *mas/yr*
    pub pm_dec: Option<f64>,
    /// Mean epoch of the right ascension. Units: *yr*
    pub epoch_ra: Option<f64>,
    /// Mean epoch of the declination. Units: *yr*
    pub epoch_dec: Option<f64>,
    /// Tycho-2 BT (blue) magnitude. Units: *mag*
    pub bt_mag: Option<f64>,
    /// Tycho-2 VT (visual) magnitude. Units: *mag*
    pub vt_mag: Option<f64>,
    /// Johnson V-band approximation, computed as `VT - 0.090*(BT-VT)` when
    /// both `bt_mag` and `vt_mag` are present, else `vt_mag`, else `None`.
    /// Formula from the Tycho-2 catalog documentation. Units: *mag*
    pub vmag: Option<f64>,
    /// Set (`"T"`) if this entry also has a Tycho-1 record.
    pub tycho1_flag: Option<String>,
    /// Hipparcos catalog identifier, if this star is also in Hipparcos.
    pub hip_id: Option<u32>,
    /// Observed right ascension, epoch ~1991.5. Always present, even when
    /// `ra` is `None`. Units: *deg*
    pub ra_observed: f64,
    /// Observed declination, epoch ~1991.5. Always present, even when `dec`
    /// is `None`. Units: *deg*
    pub dec_observed: f64,
}

impl StarRecord for Tycho2Record {
    fn id(&self) -> String {
        format!("TYC {}-{}-{}", self.tyc1, self.tyc2, self.tyc3)
    }

    fn name(&self) -> Option<String> {
        self.hip_id.map(|h| format!("HIP {h}"))
    }

    fn ra(&self) -> f64 {
        self.ra.unwrap_or(self.ra_observed)
    }

    fn dec(&self) -> f64 {
        self.dec.unwrap_or(self.dec_observed)
    }

    fn pm_ra(&self) -> Option<f64> {
        self.pm_ra
    }

    fn pm_dec(&self) -> Option<f64> {
        self.pm_dec
    }

    fn parallax(&self) -> Option<f64> {
        // Tycho-2 has no parallax column.
        None
    }

    fn radial_velocity(&self) -> Option<f64> {
        // Tycho-2 has no radial velocity column.
        None
    }

    fn epoch(&self) -> Epoch {
        // J2000.0 mean epoch for propagated positions; per-star epoch_ra/
        // epoch_dec are retained as fields for users needing exact epochs.
        //
        // For pflag == "X" records, ra()/dec() actually return the observed
        // position (epoch ~1991.5, see struct docs) while this still reports
        // J2000.0. This is harmless because such records carry no mean
        // proper motion, so radec_at_epoch's propagation over the
        // (incorrect) epoch span is a no-op.
        Epoch::from_jd(2451545.0, TimeSystem::TT)
    }

    fn magnitude(&self) -> Option<f64> {
        self.vmag
    }
}

/// Extract and trim a 0-indexed pipe-delimited field.
fn field<'a>(fields: &[&'a str], idx: usize) -> &'a str {
    fields.get(idx).copied().unwrap_or("").trim()
}

/// Parse a single Tycho-2 pipe-delimited catalog line into a record.
///
/// # Arguments
/// * `line` - One line of the Tycho-2 `Tycho2_Catalog.txt` pipe-delimited file
///
/// # Returns
/// * `Result<Tycho2Record, BraheError>` - Parsed record, or a `ParseError` if
///   a required field (TYC identifier triple, or observed RA/Dec) is
///   missing/malformed
fn parse_tycho2_line(line: &str) -> Result<Tycho2Record, BraheError> {
    let fields: Vec<&str> = line.split('|').collect();

    let mut tyc_parts = field(&fields, 0).split_whitespace();
    let tyc1: u16 = tyc_parts
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| {
            BraheError::ParseError(format!("Tycho-2: invalid TYC1 in line: {line:?}"))
        })?;
    let tyc2: u16 = tyc_parts
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| {
            BraheError::ParseError(format!("Tycho-2: invalid TYC2 in line: {line:?}"))
        })?;
    let tyc3: u8 = tyc_parts
        .next()
        .and_then(|s| s.parse().ok())
        .ok_or_else(|| {
            BraheError::ParseError(format!("Tycho-2: invalid TYC3 in line: {line:?}"))
        })?;

    let pflag = opt_string(field(&fields, 1));
    let ra = opt_f64(field(&fields, 2));
    let dec = opt_f64(field(&fields, 3));
    let pm_ra = opt_f64(field(&fields, 4));
    let pm_dec = opt_f64(field(&fields, 5));
    let epoch_ra = opt_f64(field(&fields, 10));
    let epoch_dec = opt_f64(field(&fields, 11));
    let bt_mag = opt_f64(field(&fields, 17));
    let vt_mag = opt_f64(field(&fields, 19));
    let tycho1_flag = opt_string(field(&fields, 22));
    let hip_id = field(&fields, 23).parse::<u32>().ok();

    let ra_observed: f64 = field(&fields, 24).parse().map_err(|_| {
        BraheError::ParseError(format!("Tycho-2: invalid observed RA in line: {line:?}"))
    })?;
    let dec_observed: f64 = field(&fields, 25).parse().map_err(|_| {
        BraheError::ParseError(format!("Tycho-2: invalid observed Dec in line: {line:?}"))
    })?;

    // Johnson V-band approximation from the Tycho-2 catalog documentation.
    let vmag = match (bt_mag, vt_mag) {
        (Some(bt), Some(vt)) => Some(vt - 0.090 * (bt - vt)),
        _ => vt_mag,
    };

    Ok(Tycho2Record {
        tyc1,
        tyc2,
        tyc3,
        pflag,
        ra,
        dec,
        pm_ra,
        pm_dec,
        epoch_ra,
        epoch_dec,
        bt_mag,
        vt_mag,
        vmag,
        tycho1_flag,
        hip_id,
        ra_observed,
        dec_observed,
    })
}

/// Parse the full Tycho-2 pipe-delimited catalog text into records.
///
/// Blank lines are skipped; every other line is parsed with
/// [`parse_tycho2_line`]. Records with `pflag == "X"` (no mean position) are
/// retained, not skipped; see [`Tycho2Record`] for how they are handled.
///
/// # Arguments
/// * `data` - Raw contents of `Tycho2_Catalog.txt`
///
/// # Returns
/// * `Result<Vec<Tycho2Record>, BraheError>` - Parsed records, in file order
pub(crate) fn parse_tycho2_text(data: &str) -> Result<Vec<Tycho2Record>, BraheError> {
    data.lines()
        .filter(|line| !line.trim().is_empty())
        .map(parse_tycho2_line)
        .collect()
}

/// Container for Tycho-2 star catalog records with lookup and filter methods.
///
/// Wraps a `Vec<Tycho2Record>` and provides lookup by TYC identifier triple,
/// magnitude filtering, and cone-search filtering. Filter methods return a
/// new `Tycho2Catalog` instance (immutable pattern), enabling method
/// chaining.
pub struct Tycho2Catalog {
    records: Vec<Tycho2Record>,
}

impl Tycho2Catalog {
    /// Create a new Tycho2Catalog from a vector of records.
    pub fn new(records: Vec<Tycho2Record>) -> Self {
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
    pub fn records(&self) -> &[Tycho2Record] {
        &self.records
    }

    /// Consume the container and return the underlying records.
    pub fn into_records(self) -> Vec<Tycho2Record> {
        self.records
    }

    /// Look up a record by TYC identifier triple.
    pub fn get_by_id(&self, tyc1: u16, tyc2: u16, tyc3: u8) -> Option<&Tycho2Record> {
        self.records
            .iter()
            .find(|r| r.tyc1 == tyc1 && r.tyc2 == tyc2 && r.tyc3 == tyc3)
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
    /// One row per record, one column per `Tycho2Record` field, using the
    /// same field names. Missing optional values become nulls.
    ///
    /// # Returns
    /// * `Result<DataFrame, BraheError>` - Catalog data as a DataFrame
    pub fn to_dataframe(&self) -> Result<DataFrame, BraheError> {
        let u16_col = |name: &str, f: fn(&Tycho2Record) -> u16| -> Column {
            Series::new(
                name.into(),
                self.records.iter().map(|r| f(r) as u32).collect::<Vec<_>>(),
            )
            .into()
        };

        let u8_col: Column = Series::new(
            "tyc3".into(),
            self.records
                .iter()
                .map(|r| r.tyc3 as u32)
                .collect::<Vec<_>>(),
        )
        .into();

        let f64_col = |name: &str, f: fn(&Tycho2Record) -> f64| -> Column {
            Series::new(name.into(), self.records.iter().map(f).collect::<Vec<_>>()).into()
        };

        let opt_f64_col = |name: &str, f: fn(&Tycho2Record) -> Option<f64>| -> Column {
            Series::new(name.into(), self.records.iter().map(f).collect::<Vec<_>>()).into()
        };

        let opt_u32_col = |name: &str, f: fn(&Tycho2Record) -> Option<u32>| -> Column {
            Series::new(name.into(), self.records.iter().map(f).collect::<Vec<_>>()).into()
        };

        let opt_str_col = |name: &str, f: fn(&Tycho2Record) -> Option<&str>| -> Column {
            Series::new(name.into(), self.records.iter().map(f).collect::<Vec<_>>()).into()
        };

        let columns = vec![
            u16_col("tyc1", |r| r.tyc1),
            u16_col("tyc2", |r| r.tyc2),
            u8_col,
            opt_str_col("pflag", |r| r.pflag.as_deref()),
            opt_f64_col("ra", |r| r.ra),
            opt_f64_col("dec", |r| r.dec),
            opt_f64_col("pm_ra", |r| r.pm_ra),
            opt_f64_col("pm_dec", |r| r.pm_dec),
            opt_f64_col("epoch_ra", |r| r.epoch_ra),
            opt_f64_col("epoch_dec", |r| r.epoch_dec),
            opt_f64_col("bt_mag", |r| r.bt_mag),
            opt_f64_col("vt_mag", |r| r.vt_mag),
            opt_f64_col("vmag", |r| r.vmag),
            opt_str_col("tycho1_flag", |r| r.tycho1_flag.as_deref()),
            opt_u32_col("hip_id", |r| r.hip_id),
            f64_col("ra_observed", |r| r.ra_observed),
            f64_col("dec_observed", |r| r.dec_observed),
        ];

        DataFrame::new(self.records.len(), columns)
            .map_err(|e| BraheError::Error(format!("Failed to create Tycho-2 DataFrame: {}", e)))
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use serial_test::parallel;

    const SAMPLE: &str =
        include_str!("../../../test_assets/star_catalog/Tycho2_Catalog_sample.txt");

    #[test]
    #[parallel]
    fn test_tycho2_parse_sample() {
        let records = parse_tycho2_text(SAMPLE).unwrap();
        assert_eq!(records.len(), 20);

        let r1 = &records[0];
        assert_eq!(r1.tyc1, 1);
        assert_eq!(r1.tyc2, 8);
        assert_eq!(r1.tyc3, 1);
        assert_eq!(r1.pflag, None);
        assert_abs_diff_eq!(r1.ra.unwrap(), 2.31750494, epsilon = 1e-9);
        assert_abs_diff_eq!(r1.dec.unwrap(), 2.23184345, epsilon = 1e-9);
        assert_abs_diff_eq!(r1.pm_ra.unwrap(), -16.3, epsilon = 1e-9);
        assert_abs_diff_eq!(r1.pm_dec.unwrap(), -9.0, epsilon = 1e-9);
        assert_abs_diff_eq!(r1.bt_mag.unwrap(), 12.146, epsilon = 1e-9);
        assert_abs_diff_eq!(r1.vt_mag.unwrap(), 12.146, epsilon = 1e-9);
        assert_abs_diff_eq!(r1.vmag.unwrap(), 12.146, epsilon = 1e-9);
        assert_eq!(r1.hip_id, None);
        assert_eq!(r1.tycho1_flag, None);

        // TYC 1-13-1 (real sample data) has BT != VT, so it actually
        // exercises the VT - 0.090*(BT-VT) subtraction term (TYC 1-8-1
        // above has BT == VT, which is degenerate for that formula).
        let r2 = records
            .iter()
            .find(|r| r.tyc1 == 1 && r.tyc2 == 13 && r.tyc3 == 1)
            .unwrap();
        assert_abs_diff_eq!(r2.bt_mag.unwrap(), 10.488, epsilon = 1e-9);
        assert_abs_diff_eq!(r2.vt_mag.unwrap(), 8.670, epsilon = 1e-9);
        assert_abs_diff_eq!(r2.vmag.unwrap(), 8.50638, epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    fn test_tycho2_x_flag_record() {
        let records = parse_tycho2_text(SAMPLE).unwrap();

        // TYC 1-41-1 has pflag == "X": no mean position was computed.
        let rx = records
            .iter()
            .find(|r| r.tyc1 == 1 && r.tyc2 == 41 && r.tyc3 == 1)
            .unwrap();
        assert_eq!(rx.pflag.as_deref(), Some("X"));
        assert_eq!(rx.ra, None);
        assert_eq!(rx.dec, None);
        assert_eq!(rx.pm_ra, None);
        assert_eq!(rx.pm_dec, None);

        // The observed position is always present.
        assert_abs_diff_eq!(rx.ra_observed, 1.80874194, epsilon = 1e-9);
        assert_abs_diff_eq!(rx.dec_observed, 2.31573611, epsilon = 1e-9);

        // StarRecord::ra/dec fall back to the observed position.
        assert_abs_diff_eq!(rx.ra(), rx.ra_observed, epsilon = 1e-12);
        assert_abs_diff_eq!(rx.dec(), rx.dec_observed, epsilon = 1e-12);
    }

    #[test]
    #[parallel]
    fn test_tycho2_catalog_filters() {
        let records = parse_tycho2_text(SAMPLE).unwrap();
        let catalog = Tycho2Catalog::new(records);

        let bright = catalog.filter_by_magnitude(10.0);
        let expected = catalog
            .records()
            .iter()
            .filter(|r| r.vmag.is_some_and(|v| v <= 10.0))
            .count();
        assert_eq!(bright.len(), expected);
        assert!(bright.len() < catalog.len());
        assert!(bright.records().iter().all(|r| r.vmag.unwrap() <= 10.0));

        let r1 = catalog.get_by_id(1, 8, 1).unwrap();
        let cone = catalog.filter_by_cone(r1.ra(), r1.dec(), 0.1, AngleFormat::Degrees);
        assert!(cone.get_by_id(1, 8, 1).is_some());

        assert!(catalog.get_by_id(9999, 9999, 9).is_none());
    }

    #[test]
    #[parallel]
    fn test_tycho2_star_record_trait() {
        let records = parse_tycho2_text(SAMPLE).unwrap();
        let r1 = &records[0];

        assert_eq!(r1.id(), "TYC 1-8-1");
        assert_eq!(r1.name(), None);

        let u = r1.unit_vector();
        assert_abs_diff_eq!(u.norm(), 1.0, epsilon = 1e-12);

        let (ra, dec) = r1.radec_at_epoch(r1.epoch(), AngleFormat::Degrees);
        assert_abs_diff_eq!(ra, r1.ra(), epsilon = 1e-9);
        assert_abs_diff_eq!(dec, r1.dec(), epsilon = 1e-9);

        assert_eq!(r1.radial_velocity(), None);
        assert_eq!(r1.parallax(), None);

        // TYC 1-58-1 (real sample data) has a HIP cross-identification.
        let r_hip = records
            .iter()
            .find(|r| r.tyc1 == 1 && r.tyc2 == 58 && r.tyc3 == 1)
            .unwrap();
        assert_eq!(r_hip.hip_id, Some(416));
        assert_eq!(r_hip.name(), Some("HIP 416".to_string()));
    }

    #[test]
    #[parallel]
    fn test_tycho2_to_dataframe() {
        let records = parse_tycho2_text(SAMPLE).unwrap();
        let catalog = Tycho2Catalog::new(records);
        let df = catalog.to_dataframe().unwrap();

        assert_eq!(df.height(), 20);
        assert_eq!(df.width(), 17);

        let vmag = df.column("vmag").unwrap();
        assert_abs_diff_eq!(vmag.f64().unwrap().get(0).unwrap(), 12.146, epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    #[ignore = "downloads the full 526 MB Tycho-2 catalog; run explicitly with -- --ignored"]
    /// Downloads the full Tycho-2 catalog (~526 MB). Run explicitly with
    /// `cargo test -- --ignored`; never part of CI or feature runs.
    fn test_tycho2_full_download() {
        let catalog = crate::datasets::star_catalog::get_tycho2_catalog(None).unwrap();
        assert!(catalog.len() > 2_500_000);

        let r1 = catalog.get_by_id(1, 8, 1).unwrap();
        assert_abs_diff_eq!(r1.vmag.unwrap(), 12.146, epsilon = 1e-9);
    }
}
