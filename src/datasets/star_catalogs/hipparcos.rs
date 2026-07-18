/*!
 * Hipparcos star catalog: record type, container, and pipe-delimited text parser.
 *
 * The Hipparcos Catalogue is a fixed, astrometric catalog of ~118,000 stars
 * derived from the ESA Hipparcos satellite mission, referred to the ICRS at
 * epoch J1991.25. Source data is a pipe-delimited text file (CDS `hip_main`
 * format) with one 78-field line per star.
 */

use polars::prelude::*;

use crate::constants::AngleFormat;
use crate::coordinates::position_radec_to_inertial;
use crate::datasets::star_catalogs::traits::StarRecord;
use crate::datasets::star_catalogs::{opt_f64, opt_string};
use crate::math::SVector3;
use crate::time::{Epoch, TimeSystem};
use crate::utils::BraheError;

/// A single record from the Hipparcos star catalog.
///
/// Positions and proper motions are ICRS at epoch J1991.25. The Hipparcos
/// main catalog (`hip_main`) does not carry a radial velocity column, so
/// [`StarRecord::radial_velocity`] always returns `None` for Hipparcos
/// records.
#[derive(Debug, Clone)]
pub struct HipparcosRecord {
    /// Hipparcos catalog identifier.
    pub hip_id: u32,
    /// Visual magnitude. Units: *mag*
    pub vmag: Option<f64>,
    /// Magnitude uncertainty/variability flag.
    pub var_flag: Option<String>,
    /// Right ascension, ICRS, epoch J1991.25. Units: *deg*
    pub ra: f64,
    /// Declination, ICRS, epoch J1991.25. Units: *deg*
    pub dec: f64,
    /// Trigonometric parallax. Units: *mas*
    pub parallax: Option<f64>,
    /// Proper motion in right ascension (μ_α* = μ_α cos δ), ICRS. Units: *mas/yr*
    pub pm_ra: Option<f64>,
    /// Proper motion in declination, ICRS. Units: *mas/yr*
    pub pm_dec: Option<f64>,
    /// Standard error in right ascension. Units: *mas*
    pub e_ra: Option<f64>,
    /// Standard error in declination. Units: *mas*
    pub e_dec: Option<f64>,
    /// Standard error in parallax. Units: *mas*
    pub e_parallax: Option<f64>,
    /// Standard error in right ascension proper motion. Units: *mas/yr*
    pub e_pm_ra: Option<f64>,
    /// Standard error in declination proper motion. Units: *mas/yr*
    pub e_pm_dec: Option<f64>,
    /// Mean Tycho BT magnitude. Units: *mag*
    pub bt_mag: Option<f64>,
    /// Mean Tycho VT magnitude. Units: *mag*
    pub vt_mag: Option<f64>,
    /// Johnson B-V colour. Units: *mag*
    pub b_v: Option<f64>,
    /// Hipparcos-system magnitude. Units: *mag*
    pub hp_mag: Option<f64>,
    /// Variability type flag.
    pub hvar_type: Option<String>,
    /// Double/multiple system flag.
    pub mult_flag: Option<String>,
    /// Henry Draper (HD) catalog identifier.
    pub hd_id: Option<u32>,
    /// Bonner Durchmusterung (BD) identifier, as stored in the source file:
    /// a leading `'B'` (fit into a 10-character column) followed by the
    /// zone and number, e.g. `"B+00 5077"` for BD+00 5077. See
    /// [`StarRecord::name`] for the fully expanded form.
    pub bd_id: Option<String>,
    /// Cordoba Durchmusterung (CoD) identifier, as stored in the source
    /// file: a leading `'C'` followed by the zone and number, e.g.
    /// `"C-41 15372"` for CoD-41 15372. See [`StarRecord::name`] for the
    /// fully expanded form.
    pub cod_id: Option<String>,
    /// Cape Photographic Durchmusterung (CPD) identifier, as stored in the
    /// source file: a leading `'P'` followed by the zone and number, e.g.
    /// `"P-52 12237"` for CPD-52 12237. See [`StarRecord::name`] for the
    /// fully expanded form.
    pub cpd_id: Option<String>,
    /// Spectral type.
    pub spectral_type: Option<String>,
}

impl StarRecord for HipparcosRecord {
    fn id(&self) -> String {
        format!("HIP {}", self.hip_id)
    }

    fn name(&self) -> Option<String> {
        self.hd_id
            .map(|hd| format!("HD {hd}"))
            .or_else(|| self.bd_id.as_deref().map(|s| expand_dm_id("BD", s)))
            .or_else(|| self.cod_id.as_deref().map(|s| expand_dm_id("CoD", s)))
            .or_else(|| self.cpd_id.as_deref().map(|s| expand_dm_id("CPD", s)))
    }

    fn ra(&self) -> f64 {
        self.ra
    }

    fn dec(&self) -> f64 {
        self.dec
    }

    fn pm_ra(&self) -> Option<f64> {
        self.pm_ra
    }

    fn pm_dec(&self) -> Option<f64> {
        self.pm_dec
    }

    fn parallax(&self) -> Option<f64> {
        self.parallax
    }

    fn radial_velocity(&self) -> Option<f64> {
        // hip_main has no radial velocity column.
        None
    }

    fn epoch(&self) -> Epoch {
        // Hipparcos positions/proper motions are referred to ICRS, J1991.25.
        Epoch::from_jd(2448349.0625, TimeSystem::TT)
    }

    fn magnitude(&self) -> Option<f64> {
        self.vmag
    }
}

/// Extract and trim a 0-indexed pipe-delimited field.
fn field<'a>(fields: &[&'a str], idx: usize) -> &'a str {
    fields.get(idx).copied().unwrap_or("").trim()
}

/// Expand a raw Durchmusterung identifier into an unambiguous name.
///
/// The source file abbreviates each DM catalog to a single leading letter
/// to fit a 10-character column (`'B'` for BD, `'C'` for CoD, `'P'` for
/// CPD), e.g. `"B+00 5077"`. That abbreviation is ambiguous through the
/// public [`StarRecord::name`] API, so this strips the leading letter and
/// prepends the full catalog code instead, e.g. `expand_dm_id("BD", "B+00
/// 5077")` -> `"BD +00 5077"`.
///
/// # Arguments
/// * `catalog_code` - Full catalog code to prepend (`"BD"`, `"CoD"`, or `"CPD"`)
/// * `raw` - Raw field value, with its single-letter catalog abbreviation still attached
///
/// # Returns
/// * `String` - `catalog_code` followed by a space and the zone/number, with
///   the source's leading abbreviation letter removed
fn expand_dm_id(catalog_code: &str, raw: &str) -> String {
    format!("{catalog_code} {}", raw.get(1..).unwrap_or(""))
}

/// Parse a single Hipparcos pipe-delimited catalog line into a record.
///
/// A handful of `hip_main` entries lack astrometric solutions (empty RA/Dec
/// fields); such lines are skipped and return `Ok(None)` rather than an
/// error, since they are a normal, documented feature of the source file
/// rather than a malformed line.
///
/// # Arguments
/// * `line` - One line of the Hipparcos `Hipparcos_Catalog.txt` pipe-delimited file
///
/// # Returns
/// * `Result<Option<HipparcosRecord>, BraheError>` - Parsed record, `None` if
///   the line has no astrometric solution, or a `ParseError` if a required
///   field (identifier, or malformed RA/Dec) is missing/malformed
fn parse_hipparcos_line(line: &str) -> Result<Option<HipparcosRecord>, BraheError> {
    let fields: Vec<&str> = line.split('|').collect();

    let hip_id: u32 = field(&fields, 1).parse().map_err(|_| {
        BraheError::ParseError(format!("Hipparcos: invalid hip_id in line: {line:?}"))
    })?;

    let ra_str = field(&fields, 8);
    let dec_str = field(&fields, 9);
    if ra_str.is_empty() || dec_str.is_empty() {
        // No astrometric solution for this entry; skip it.
        return Ok(None);
    }
    let ra: f64 = ra_str
        .parse()
        .map_err(|_| BraheError::ParseError(format!("Hipparcos: invalid RA in line: {line:?}")))?;
    let dec: f64 = dec_str
        .parse()
        .map_err(|_| BraheError::ParseError(format!("Hipparcos: invalid Dec in line: {line:?}")))?;

    let vmag = opt_f64(field(&fields, 5));
    let var_flag = opt_string(field(&fields, 6));
    let parallax = opt_f64(field(&fields, 11));
    let pm_ra = opt_f64(field(&fields, 12));
    let pm_dec = opt_f64(field(&fields, 13));
    let e_ra = opt_f64(field(&fields, 14));
    let e_dec = opt_f64(field(&fields, 15));
    let e_parallax = opt_f64(field(&fields, 16));
    let e_pm_ra = opt_f64(field(&fields, 17));
    let e_pm_dec = opt_f64(field(&fields, 18));
    let bt_mag = opt_f64(field(&fields, 32));
    let vt_mag = opt_f64(field(&fields, 34));
    let b_v = opt_f64(field(&fields, 37));
    let hp_mag = opt_f64(field(&fields, 44));
    let hvar_type = opt_string(field(&fields, 52));
    let mult_flag = opt_string(field(&fields, 59));
    let hd_id = field(&fields, 71).parse::<u32>().ok();
    let bd_id = opt_string(field(&fields, 72));
    let cod_id = opt_string(field(&fields, 73));
    let cpd_id = opt_string(field(&fields, 74));
    let spectral_type = opt_string(field(&fields, 76));

    Ok(Some(HipparcosRecord {
        hip_id,
        vmag,
        var_flag,
        ra,
        dec,
        parallax,
        pm_ra,
        pm_dec,
        e_ra,
        e_dec,
        e_parallax,
        e_pm_ra,
        e_pm_dec,
        bt_mag,
        vt_mag,
        b_v,
        hp_mag,
        hvar_type,
        mult_flag,
        hd_id,
        bd_id,
        cod_id,
        cpd_id,
        spectral_type,
    }))
}

/// Parse the full Hipparcos pipe-delimited catalog text into records.
///
/// Blank lines are skipped; every other line is parsed with
/// [`parse_hipparcos_line`]. Lines with no astrometric solution are also
/// skipped (see [`parse_hipparcos_line`]).
///
/// # Arguments
/// * `data` - Raw contents of `Hipparcos_Catalog.txt`
///
/// # Returns
/// * `Result<Vec<HipparcosRecord>, BraheError>` - Parsed records, in file order
pub(crate) fn parse_hipparcos_text(data: &str) -> Result<Vec<HipparcosRecord>, BraheError> {
    data.lines()
        .filter(|line| !line.trim().is_empty())
        .filter_map(|line| parse_hipparcos_line(line).transpose())
        .collect()
}

/// Container for Hipparcos star catalog records with lookup and filter methods.
///
/// Wraps a `Vec<HipparcosRecord>` and provides lookup by Hipparcos
/// identifier, magnitude filtering, and cone-search filtering. Filter
/// methods return a new `HipparcosCatalog` instance (immutable pattern),
/// enabling method chaining.
pub struct HipparcosCatalog {
    records: Vec<HipparcosRecord>,
}

impl HipparcosCatalog {
    /// Create a new HipparcosCatalog from a vector of records.
    pub fn new(records: Vec<HipparcosRecord>) -> Self {
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
    pub fn records(&self) -> &[HipparcosRecord] {
        &self.records
    }

    /// Consume the container and return the underlying records.
    pub fn into_records(self) -> Vec<HipparcosRecord> {
        self.records
    }

    /// Look up a record by Hipparcos catalog identifier.
    pub fn get_by_id(&self, hip_id: u32) -> Option<&HipparcosRecord> {
        self.records.iter().find(|r| r.hip_id == hip_id)
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
    /// One row per record, one column per `HipparcosRecord` field, using the
    /// same field names. Missing optional values become nulls.
    ///
    /// # Returns
    /// * `Result<DataFrame, BraheError>` - Catalog data as a DataFrame
    pub fn to_dataframe(&self) -> Result<DataFrame, BraheError> {
        let u32_col: Column = Series::new(
            "hip_id".into(),
            self.records.iter().map(|r| r.hip_id).collect::<Vec<_>>(),
        )
        .into();

        let f64_col = |name: &str, f: fn(&HipparcosRecord) -> f64| -> Column {
            Series::new(name.into(), self.records.iter().map(f).collect::<Vec<_>>()).into()
        };

        let opt_f64_col = |name: &str, f: fn(&HipparcosRecord) -> Option<f64>| -> Column {
            Series::new(name.into(), self.records.iter().map(f).collect::<Vec<_>>()).into()
        };

        let opt_u32_col = |name: &str, f: fn(&HipparcosRecord) -> Option<u32>| -> Column {
            Series::new(name.into(), self.records.iter().map(f).collect::<Vec<_>>()).into()
        };

        let opt_str_col = |name: &str, f: fn(&HipparcosRecord) -> Option<&str>| -> Column {
            Series::new(name.into(), self.records.iter().map(f).collect::<Vec<_>>()).into()
        };

        let columns = vec![
            u32_col,
            opt_f64_col("vmag", |r| r.vmag),
            opt_str_col("var_flag", |r| r.var_flag.as_deref()),
            f64_col("ra", |r| r.ra),
            f64_col("dec", |r| r.dec),
            opt_f64_col("parallax", |r| r.parallax),
            opt_f64_col("pm_ra", |r| r.pm_ra),
            opt_f64_col("pm_dec", |r| r.pm_dec),
            opt_f64_col("e_ra", |r| r.e_ra),
            opt_f64_col("e_dec", |r| r.e_dec),
            opt_f64_col("e_parallax", |r| r.e_parallax),
            opt_f64_col("e_pm_ra", |r| r.e_pm_ra),
            opt_f64_col("e_pm_dec", |r| r.e_pm_dec),
            opt_f64_col("bt_mag", |r| r.bt_mag),
            opt_f64_col("vt_mag", |r| r.vt_mag),
            opt_f64_col("b_v", |r| r.b_v),
            opt_f64_col("hp_mag", |r| r.hp_mag),
            opt_str_col("hvar_type", |r| r.hvar_type.as_deref()),
            opt_str_col("mult_flag", |r| r.mult_flag.as_deref()),
            opt_u32_col("hd_id", |r| r.hd_id),
            opt_str_col("bd_id", |r| r.bd_id.as_deref()),
            opt_str_col("cod_id", |r| r.cod_id.as_deref()),
            opt_str_col("cpd_id", |r| r.cpd_id.as_deref()),
            opt_str_col("spectral_type", |r| r.spectral_type.as_deref()),
        ];

        DataFrame::new(self.records.len(), columns)
            .map_err(|e| BraheError::Error(format!("Failed to create Hipparcos DataFrame: {}", e)))
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use serial_test::parallel;

    const SAMPLE: &str =
        include_str!("../../../test_assets/star_catalogs/Hipparcos_Catalog_sample.txt");

    #[test]
    #[parallel]
    fn test_hipparcos_parse_sample() {
        let records = parse_hipparcos_text(SAMPLE).unwrap();
        assert_eq!(records.len(), 20);

        let r1 = &records[0];
        assert_eq!(r1.hip_id, 1);
        assert_abs_diff_eq!(r1.vmag.unwrap(), 9.10, epsilon = 1e-9);
        assert_abs_diff_eq!(r1.ra, 000.00091185, epsilon = 1e-9);
        assert_abs_diff_eq!(r1.dec, 01.08901332, epsilon = 1e-9);
        assert_abs_diff_eq!(r1.parallax.unwrap(), 3.54, epsilon = 1e-9);
        assert_abs_diff_eq!(r1.pm_ra.unwrap(), -5.20, epsilon = 1e-9);
        assert_abs_diff_eq!(r1.pm_dec.unwrap(), -1.88, epsilon = 1e-9);
        assert_eq!(r1.hd_id, Some(224700));
        assert_eq!(r1.bd_id.as_deref(), Some("B+00 5077"));
        assert_eq!(r1.cod_id, None);
        assert_eq!(r1.cpd_id, None);
        assert_eq!(r1.spectral_type.as_deref(), Some("F5"));
        assert_eq!(r1.var_flag, None);
    }

    #[test]
    #[parallel]
    fn test_hipparcos_catalog_filters() {
        let records = parse_hipparcos_text(SAMPLE).unwrap();
        let catalog = HipparcosCatalog::new(records);

        let bright = catalog.filter_by_magnitude(7.0);
        let expected = catalog
            .records()
            .iter()
            .filter(|r| r.vmag.is_some_and(|v| v <= 7.0))
            .count();
        assert_eq!(bright.len(), expected);
        assert!(bright.len() < catalog.len());
        assert!(bright.records().iter().all(|r| r.vmag.unwrap() <= 7.0));

        let r1 = catalog.get_by_id(1).unwrap();
        let cone = catalog.filter_by_cone(r1.ra, r1.dec, 0.1, AngleFormat::Degrees);
        assert!(cone.get_by_id(1).is_some());
    }

    #[test]
    #[parallel]
    fn test_hipparcos_star_record_trait() {
        let records = parse_hipparcos_text(SAMPLE).unwrap();
        let r1 = &records[0];

        assert_eq!(r1.id(), "HIP 1");
        assert_eq!(r1.name(), Some("HD 224700".to_string()));

        let u = r1.unit_vector();
        assert_abs_diff_eq!(u.norm(), 1.0, epsilon = 1e-12);

        let (ra, dec) = r1.radec_at_epoch(r1.epoch(), AngleFormat::Degrees);
        assert_abs_diff_eq!(ra, r1.ra, epsilon = 1e-9);
        assert_abs_diff_eq!(dec, r1.dec, epsilon = 1e-9);

        assert_eq!(r1.radial_velocity(), None);
        assert_abs_diff_eq!(r1.magnitude().unwrap(), r1.vmag.unwrap(), epsilon = 1e-9);
    }

    /// Overwrite the pipe-delimited field at `idx` (0-indexed) with
    /// `replacement`, for constructing deliberately malformed lines from a
    /// known-good sample line.
    fn set_pipe_field(line: &str, idx: usize, replacement: &str) -> String {
        let mut fields: Vec<&str> = line.split('|').collect();
        fields[idx] = replacement;
        fields.join("|")
    }

    #[test]
    #[parallel]
    fn test_hipparcos_parse_line_invalid_hip_id() {
        let line1 = SAMPLE.lines().next().unwrap();
        let bad_line = set_pipe_field(line1, 1, "not-a-number");
        assert!(parse_hipparcos_line(&bad_line).is_err());
    }

    #[test]
    #[parallel]
    fn test_hipparcos_parse_line_no_astrometric_solution() {
        // A blank RA field (index 8) means no astrometric solution was
        // computed for this entry: the line is skipped (`Ok(None)`), not an
        // error.
        let line1 = SAMPLE.lines().next().unwrap();
        let no_solution_line = set_pipe_field(line1, 8, "");
        assert!(parse_hipparcos_line(&no_solution_line).unwrap().is_none());
    }

    #[test]
    #[parallel]
    fn test_hipparcos_catalog_container_methods() {
        let records = parse_hipparcos_text(SAMPLE).unwrap();
        let catalog = HipparcosCatalog::new(records);

        assert!(!catalog.is_empty());
        assert_eq!(catalog.records().len(), catalog.len());

        let count = catalog.len();
        let records = catalog.into_records();
        assert_eq!(records.len(), count);

        let empty = HipparcosCatalog::new(Vec::new());
        assert!(empty.is_empty());
    }

    #[test]
    #[parallel]
    fn test_hipparcos_filter_by_cone_radians() {
        let records = parse_hipparcos_text(SAMPLE).unwrap();
        let catalog = HipparcosCatalog::new(records);

        let r1 = catalog.get_by_id(1).unwrap();
        let cone = catalog.filter_by_cone(
            r1.ra.to_radians(),
            r1.dec.to_radians(),
            0.1f64.to_radians(),
            AngleFormat::Radians,
        );
        assert!(cone.get_by_id(1).is_some());
    }

    /// Build a `HipparcosRecord` with only the DM-identifier fields set, for
    /// isolating [`HipparcosRecord::name`]'s fallback-chain behavior from
    /// what specific identifier combinations happen to occur in the 20-line
    /// sample asset.
    fn dm_only_record(
        hd_id: Option<u32>,
        bd_id: Option<&str>,
        cod_id: Option<&str>,
        cpd_id: Option<&str>,
    ) -> HipparcosRecord {
        HipparcosRecord {
            hip_id: 0,
            vmag: None,
            var_flag: None,
            ra: 0.0,
            dec: 0.0,
            parallax: None,
            pm_ra: None,
            pm_dec: None,
            e_ra: None,
            e_dec: None,
            e_parallax: None,
            e_pm_ra: None,
            e_pm_dec: None,
            bt_mag: None,
            vt_mag: None,
            b_v: None,
            hp_mag: None,
            hvar_type: None,
            mult_flag: None,
            hd_id,
            bd_id: bd_id.map(str::to_string),
            cod_id: cod_id.map(str::to_string),
            cpd_id: cpd_id.map(str::to_string),
            spectral_type: None,
        }
    }

    #[test]
    #[parallel]
    fn test_hipparcos_name_fallback_chain() {
        let records = parse_hipparcos_text(SAMPLE).unwrap();

        // HIP 7 (real sample data): no HD, has BD ("B+19 5185" -> "BD +19 5185").
        let r7 = records.iter().find(|r| r.hip_id == 7).unwrap();
        assert_eq!(r7.hd_id, None);
        assert_eq!(r7.bd_id.as_deref(), Some("B+19 5185"));
        assert_eq!(r7.name(), Some("BD +19 5185".to_string()));

        // HIP 16 (real sample data): no HD, no BD, no CoD, has CPD
        // ("P-55 10131" -> "CPD -55 10131").
        let r16 = records.iter().find(|r| r.hip_id == 16).unwrap();
        assert_eq!(r16.hd_id, None);
        assert_eq!(r16.bd_id, None);
        assert_eq!(r16.cod_id, None);
        assert_eq!(r16.cpd_id.as_deref(), Some("P-55 10131"));
        assert_eq!(r16.name(), Some("CPD -55 10131".to_string()));

        // No sample record has a CoD without also having an HD, so the CoD
        // branch (and CoD-over-CPD priority) is exercised with a
        // constructed record instead.
        let cod_only = dm_only_record(None, None, Some("C-36 16128"), Some("P-36  9818"));
        assert_eq!(cod_only.name(), Some("CoD -36 16128".to_string()));

        // BD and CoD both present, no HD: BD wins (BD > CoD priority).
        let bd_and_cod = dm_only_record(None, Some("B+19 5185"), Some("C-36 16128"), None);
        assert_eq!(bd_and_cod.name(), Some("BD +19 5185".to_string()));

        // Full priority order: HD > BD > CoD > CPD.
        let all_present = dm_only_record(
            Some(224700),
            Some("B+00 5077"),
            Some("C-36 16128"),
            Some("P-36  9818"),
        );
        assert_eq!(all_present.name(), Some("HD 224700".to_string()));

        let none_present = dm_only_record(None, None, None, None);
        assert_eq!(none_present.name(), None);
    }

    #[test]
    #[parallel]
    fn test_hipparcos_to_dataframe() {
        let records = parse_hipparcos_text(SAMPLE).unwrap();
        let catalog = HipparcosCatalog::new(records);
        let df = catalog.to_dataframe().unwrap();

        assert_eq!(df.height(), 20);
        assert_eq!(df.width(), 24);

        let vmag = df.column("vmag").unwrap();
        assert_abs_diff_eq!(vmag.f64().unwrap().get(0).unwrap(), 9.10, epsilon = 1e-9);
    }

    #[test]
    #[parallel]
    #[cfg_attr(not(feature = "integration"), ignore)]
    fn test_hipparcos_full_download() {
        let catalog = crate::datasets::star_catalogs::get_hipparcos_catalog(None).unwrap();
        assert!(catalog.len() > 117_000);

        let sirius = catalog.get_by_id(32349).unwrap();
        assert_abs_diff_eq!(sirius.vmag.unwrap(), -1.44, epsilon = 1e-9);
    }
}
