/*!
 * Container types for GCAT catalog data with search and filter methods.
 *
 * `GCATSatcat` and `GCATPsatcat` wrap vectors of records and provide
 * convenient lookup, search, and filter operations. Filter methods return
 * new containers (immutable pattern), enabling method chaining.
 */

use polars::prelude::*;

use crate::datasets::gcat::records::{GCATPsatcatRecord, GCATSatcatRecord};
use crate::utils::BraheError;

/// Container for GCAT SATCAT records with search and filter methods.
///
/// Wraps a `Vec<GCATSatcatRecord>` and provides lookup by JCAT/SATCAT number,
/// name search, and various field-based filters. All filter methods return a
/// new `GCATSatcat` instance (immutable pattern).
pub struct GCATSatcat {
    records: Vec<GCATSatcatRecord>,
}

impl GCATSatcat {
    /// Create a new GCATSatcat from a vector of records.
    pub fn new(records: Vec<GCATSatcatRecord>) -> Self {
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
    pub fn records(&self) -> &[GCATSatcatRecord] {
        &self.records
    }

    /// Consume the container and return the underlying records.
    pub fn into_records(self) -> Vec<GCATSatcatRecord> {
        self.records
    }

    /// Look up a record by JCAT identifier.
    pub fn get_by_jcat(&self, jcat: &str) -> Option<&GCATSatcatRecord> {
        self.records.iter().find(|r| r.jcat == jcat)
    }

    /// Look up a record by NORAD SATCAT number.
    pub fn get_by_satcat(&self, satcat_num: &str) -> Option<&GCATSatcatRecord> {
        self.records
            .iter()
            .find(|r| r.satcat.as_deref() == Some(satcat_num))
    }

    /// Search records by name (case-insensitive substring match).
    ///
    /// Searches both the `name` and `pl_name` fields.
    pub fn search_by_name(&self, pattern: &str) -> Self {
        let pattern_lower = pattern.to_lowercase();
        Self::new(
            self.records
                .iter()
                .filter(|r| {
                    r.name
                        .as_ref()
                        .is_some_and(|n| n.to_lowercase().contains(&pattern_lower))
                        || r.pl_name
                            .as_ref()
                            .is_some_and(|n| n.to_lowercase().contains(&pattern_lower))
                })
                .cloned()
                .collect(),
        )
    }

    /// Filter records by object type (exact match).
    pub fn filter_by_type(&self, object_type: &str) -> Self {
        Self::new(
            self.records
                .iter()
                .filter(|r| r.object_type.as_deref() == Some(object_type))
                .cloned()
                .collect(),
        )
    }

    /// Filter records by owner (exact match).
    pub fn filter_by_owner(&self, owner: &str) -> Self {
        Self::new(
            self.records
                .iter()
                .filter(|r| r.owner.as_deref() == Some(owner))
                .cloned()
                .collect(),
        )
    }

    /// Filter records by responsible state (exact match).
    pub fn filter_by_state(&self, state: &str) -> Self {
        Self::new(
            self.records
                .iter()
                .filter(|r| r.state.as_deref() == Some(state))
                .cloned()
                .collect(),
        )
    }

    /// Filter records by status code (exact match).
    pub fn filter_by_status(&self, status: &str) -> Self {
        Self::new(
            self.records
                .iter()
                .filter(|r| r.status.as_deref() == Some(status))
                .cloned()
                .collect(),
        )
    }

    /// Filter records by perigee altitude range in km.
    pub fn filter_by_perigee_range(&self, min_km: f64, max_km: f64) -> Self {
        Self::new(
            self.records
                .iter()
                .filter(|r| r.perigee.is_some_and(|p| p >= min_km && p <= max_km))
                .cloned()
                .collect(),
        )
    }

    /// Filter records by apogee altitude range in km.
    pub fn filter_by_apogee_range(&self, min_km: f64, max_km: f64) -> Self {
        Self::new(
            self.records
                .iter()
                .filter(|r| r.apogee.is_some_and(|a| a >= min_km && a <= max_km))
                .cloned()
                .collect(),
        )
    }

    /// Filter records by inclination range in degrees.
    pub fn filter_by_inc_range(&self, min_deg: f64, max_deg: f64) -> Self {
        Self::new(
            self.records
                .iter()
                .filter(|r| r.inc.is_some_and(|i| i >= min_deg && i <= max_deg))
                .cloned()
                .collect(),
        )
    }

    /// Convert the catalog to a Polars DataFrame.
    ///
    /// All record fields become columns. String fields are `Utf8` type,
    /// numeric fields are `Float64` type. Missing values are represented as null.
    ///
    /// # Returns
    ///
    /// * `Result<DataFrame, BraheError>` - Polars DataFrame
    pub fn to_dataframe(&self) -> Result<DataFrame, BraheError> {
        // Helper closures to extract column vectors
        let str_col = |name: &str, f: fn(&GCATSatcatRecord) -> Option<&str>| -> Column {
            Series::new(name.into(), self.records.iter().map(f).collect::<Vec<_>>()).into()
        };

        let f64_col = |name: &str, f: fn(&GCATSatcatRecord) -> Option<f64>| -> Column {
            Series::new(name.into(), self.records.iter().map(f).collect::<Vec<_>>()).into()
        };

        let jcat_col: Column = Series::new(
            "jcat".into(),
            self.records
                .iter()
                .map(|r| r.jcat.as_str())
                .collect::<Vec<_>>(),
        )
        .into();

        let columns = vec![
            jcat_col,
            str_col("satcat", |r| r.satcat.as_deref()),
            str_col("launch_tag", |r| r.launch_tag.as_deref()),
            str_col("piece", |r| r.piece.as_deref()),
            str_col("object_type", |r| r.object_type.as_deref()),
            str_col("name", |r| r.name.as_deref()),
            str_col("pl_name", |r| r.pl_name.as_deref()),
            str_col("ldate", |r| r.ldate.as_deref()),
            str_col("parent", |r| r.parent.as_deref()),
            str_col("sdate", |r| r.sdate.as_deref()),
            str_col("primary", |r| r.primary.as_deref()),
            str_col("ddate", |r| r.ddate.as_deref()),
            str_col("status", |r| r.status.as_deref()),
            str_col("dest", |r| r.dest.as_deref()),
            str_col("owner", |r| r.owner.as_deref()),
            str_col("state", |r| r.state.as_deref()),
            str_col("manufacturer", |r| r.manufacturer.as_deref()),
            str_col("bus", |r| r.bus.as_deref()),
            str_col("motor", |r| r.motor.as_deref()),
            f64_col("mass", |r| r.mass),
            str_col("mass_flag", |r| r.mass_flag.as_deref()),
            f64_col("dry_mass", |r| r.dry_mass),
            str_col("dry_flag", |r| r.dry_flag.as_deref()),
            f64_col("tot_mass", |r| r.tot_mass),
            str_col("tot_flag", |r| r.tot_flag.as_deref()),
            f64_col("length", |r| r.length),
            str_col("length_flag", |r| r.length_flag.as_deref()),
            f64_col("diameter", |r| r.diameter),
            str_col("diameter_flag", |r| r.diameter_flag.as_deref()),
            f64_col("span", |r| r.span),
            str_col("span_flag", |r| r.span_flag.as_deref()),
            str_col("shape", |r| r.shape.as_deref()),
            str_col("odate", |r| r.odate.as_deref()),
            f64_col("perigee", |r| r.perigee),
            str_col("perigee_flag", |r| r.perigee_flag.as_deref()),
            f64_col("apogee", |r| r.apogee),
            str_col("apogee_flag", |r| r.apogee_flag.as_deref()),
            f64_col("inc", |r| r.inc),
            str_col("inc_flag", |r| r.inc_flag.as_deref()),
            str_col("op_orbit", |r| r.op_orbit.as_deref()),
            str_col("oqual", |r| r.oqual.as_deref()),
            str_col("alt_names", |r| r.alt_names.as_deref()),
        ];

        DataFrame::new(columns)
            .map_err(|e| BraheError::Error(format!("Failed to create SATCAT DataFrame: {}", e)))
    }
}

/// Container for GCAT PSATCAT records with search and filter methods.
///
/// Wraps a `Vec<GCATPsatcatRecord>` and provides lookup by JCAT, name search,
/// and various field-based filters.
pub struct GCATPsatcat {
    records: Vec<GCATPsatcatRecord>,
}

impl GCATPsatcat {
    /// Create a new GCATPsatcat from a vector of records.
    pub fn new(records: Vec<GCATPsatcatRecord>) -> Self {
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
    pub fn records(&self) -> &[GCATPsatcatRecord] {
        &self.records
    }

    /// Consume the container and return the underlying records.
    pub fn into_records(self) -> Vec<GCATPsatcatRecord> {
        self.records
    }

    /// Look up a record by JCAT identifier.
    pub fn get_by_jcat(&self, jcat: &str) -> Option<&GCATPsatcatRecord> {
        self.records.iter().find(|r| r.jcat == jcat)
    }

    /// Search records by name (case-insensitive substring match).
    pub fn search_by_name(&self, pattern: &str) -> Self {
        let pattern_lower = pattern.to_lowercase();
        Self::new(
            self.records
                .iter()
                .filter(|r| {
                    r.name
                        .as_ref()
                        .is_some_and(|n| n.to_lowercase().contains(&pattern_lower))
                })
                .cloned()
                .collect(),
        )
    }

    /// Filter records by mission category (exact match).
    pub fn filter_by_category(&self, category: &str) -> Self {
        Self::new(
            self.records
                .iter()
                .filter(|r| r.category.as_deref() == Some(category))
                .cloned()
                .collect(),
        )
    }

    /// Filter records by mission class (exact match).
    pub fn filter_by_class(&self, class: &str) -> Self {
        Self::new(
            self.records
                .iter()
                .filter(|r| r.class.as_deref() == Some(class))
                .cloned()
                .collect(),
        )
    }

    /// Filter records by mission result (exact match).
    pub fn filter_by_result(&self, result: &str) -> Self {
        Self::new(
            self.records
                .iter()
                .filter(|r| r.result.as_deref() == Some(result))
                .cloned()
                .collect(),
        )
    }

    /// Filter for active payloads (result is "S" for success and no end date).
    ///
    /// A payload is considered active if the result is "S" (success) and
    /// `tdate` is either absent or `"*"` (GCAT's marker for "still active").
    pub fn filter_active(&self) -> Self {
        Self::new(
            self.records
                .iter()
                .filter(|r| {
                    r.result.as_deref() == Some("S")
                        && (r.tdate.is_none() || r.tdate.as_deref() == Some("*"))
                })
                .cloned()
                .collect(),
        )
    }

    /// Convert the catalog to a Polars DataFrame.
    ///
    /// All record fields become columns. String fields are `Utf8` type,
    /// numeric fields are `Float64` type. Missing values are represented as null.
    ///
    /// # Returns
    ///
    /// * `Result<DataFrame, BraheError>` - Polars DataFrame
    pub fn to_dataframe(&self) -> Result<DataFrame, BraheError> {
        let str_col = |name: &str, f: fn(&GCATPsatcatRecord) -> Option<&str>| -> Column {
            Series::new(name.into(), self.records.iter().map(f).collect::<Vec<_>>()).into()
        };

        let f64_col = |name: &str, f: fn(&GCATPsatcatRecord) -> Option<f64>| -> Column {
            Series::new(name.into(), self.records.iter().map(f).collect::<Vec<_>>()).into()
        };

        let jcat_col: Column = Series::new(
            "jcat".into(),
            self.records
                .iter()
                .map(|r| r.jcat.as_str())
                .collect::<Vec<_>>(),
        )
        .into();

        let columns = vec![
            jcat_col,
            str_col("piece", |r| r.piece.as_deref()),
            str_col("name", |r| r.name.as_deref()),
            str_col("ldate", |r| r.ldate.as_deref()),
            str_col("tlast", |r| r.tlast.as_deref()),
            str_col("top", |r| r.top.as_deref()),
            str_col("tdate", |r| r.tdate.as_deref()),
            str_col("tf", |r| r.tf.as_deref()),
            str_col("program", |r| r.program.as_deref()),
            str_col("plane", |r| r.plane.as_deref()),
            str_col("att", |r| r.att.as_deref()),
            str_col("mvr", |r| r.mvr.as_deref()),
            str_col("class", |r| r.class.as_deref()),
            str_col("category", |r| r.category.as_deref()),
            str_col("result", |r| r.result.as_deref()),
            str_col("control", |r| r.control.as_deref()),
            str_col("discipline", |r| r.discipline.as_deref()),
            str_col("un_state", |r| r.un_state.as_deref()),
            str_col("un_reg", |r| r.un_reg.as_deref()),
            f64_col("un_period", |r| r.un_period),
            f64_col("un_perigee", |r| r.un_perigee),
            f64_col("un_apogee", |r| r.un_apogee),
            f64_col("un_inc", |r| r.un_inc),
            str_col("disp_epoch", |r| r.disp_epoch.as_deref()),
            f64_col("disp_peri", |r| r.disp_peri),
            f64_col("disp_apo", |r| r.disp_apo),
            f64_col("disp_inc", |r| r.disp_inc),
            str_col("comment", |r| r.comment.as_deref()),
        ];

        DataFrame::new(columns)
            .map_err(|e| BraheError::Error(format!("Failed to create PSATCAT DataFrame: {}", e)))
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    fn sample_satcat_records() -> Vec<GCATSatcatRecord> {
        vec![
            GCATSatcatRecord {
                jcat: "S049652".to_string(),
                satcat: Some("25544".to_string()),
                launch_tag: Some("1998-067".to_string()),
                piece: Some("A".to_string()),
                object_type: Some("P".to_string()),
                name: Some("ISS (Zarya)".to_string()),
                pl_name: Some("Zarya".to_string()),
                ldate: Some("1998 Nov 20".to_string()),
                parent: None,
                sdate: None,
                primary: Some("E".to_string()),
                ddate: None,
                status: Some("O".to_string()),
                dest: Some("LEO".to_string()),
                owner: Some("NASA".to_string()),
                state: Some("US".to_string()),
                manufacturer: None,
                bus: None,
                motor: None,
                mass: Some(19323.0),
                mass_flag: None,
                dry_mass: None,
                dry_flag: None,
                tot_mass: None,
                tot_flag: None,
                length: Some(12.6),
                length_flag: None,
                diameter: Some(4.1),
                diameter_flag: None,
                span: Some(73.2),
                span_flag: None,
                shape: Some("Cyl".to_string()),
                odate: None,
                perigee: Some(408.0),
                perigee_flag: None,
                apogee: Some(418.0),
                apogee_flag: None,
                inc: Some(51.6),
                inc_flag: None,
                op_orbit: Some("LEO/I".to_string()),
                oqual: None,
                alt_names: Some("Zarya".to_string()),
            },
            GCATSatcatRecord {
                jcat: "S049653".to_string(),
                satcat: Some("25545".to_string()),
                launch_tag: Some("1998-067".to_string()),
                piece: Some("B".to_string()),
                object_type: Some("R".to_string()),
                name: Some("Proton-K Blk DM".to_string()),
                pl_name: None,
                ldate: Some("1998 Nov 20".to_string()),
                parent: None,
                sdate: None,
                primary: Some("E".to_string()),
                ddate: Some("1998 Dec  9".to_string()),
                status: Some("D".to_string()),
                dest: Some("LEO".to_string()),
                owner: None,
                state: Some("RU".to_string()),
                manufacturer: None,
                bus: None,
                motor: None,
                mass: None,
                mass_flag: None,
                dry_mass: None,
                dry_flag: None,
                tot_mass: None,
                tot_flag: None,
                length: None,
                length_flag: None,
                diameter: None,
                diameter_flag: None,
                span: None,
                span_flag: None,
                shape: None,
                odate: None,
                perigee: Some(185.0),
                perigee_flag: None,
                apogee: Some(280.0),
                apogee_flag: None,
                inc: Some(51.6),
                inc_flag: None,
                op_orbit: Some("LEO/I".to_string()),
                oqual: None,
                alt_names: None,
            },
            GCATSatcatRecord {
                jcat: "S055000".to_string(),
                satcat: Some("43205".to_string()),
                launch_tag: Some("2018-017".to_string()),
                piece: Some("A".to_string()),
                object_type: Some("P".to_string()),
                name: Some("Starlink-0".to_string()),
                pl_name: Some("Starlink-0".to_string()),
                ldate: Some("2018 Feb 22".to_string()),
                parent: None,
                sdate: None,
                primary: Some("E".to_string()),
                ddate: None,
                status: Some("O".to_string()),
                dest: Some("LEO".to_string()),
                owner: Some("SpaceX".to_string()),
                state: Some("US".to_string()),
                manufacturer: None,
                bus: None,
                motor: None,
                mass: Some(260.0),
                mass_flag: None,
                dry_mass: None,
                dry_flag: None,
                tot_mass: None,
                tot_flag: None,
                length: None,
                length_flag: None,
                diameter: None,
                diameter_flag: None,
                span: None,
                span_flag: None,
                shape: None,
                odate: None,
                perigee: Some(540.0),
                perigee_flag: None,
                apogee: Some(550.0),
                apogee_flag: None,
                inc: Some(53.0),
                inc_flag: None,
                op_orbit: Some("LEO/I".to_string()),
                oqual: None,
                alt_names: None,
            },
        ]
    }

    fn sample_psatcat_records() -> Vec<GCATPsatcatRecord> {
        vec![
            GCATPsatcatRecord {
                jcat: "S049652".to_string(),
                piece: Some("A".to_string()),
                name: Some("ISS (Zarya)".to_string()),
                ldate: Some("1998 Nov 20".to_string()),
                tlast: Some("2025 Jan  1".to_string()),
                top: Some("1998 Nov 20".to_string()),
                tdate: Some("*".to_string()),
                tf: None,
                program: Some("ISS".to_string()),
                plane: None,
                att: Some("3AX".to_string()),
                mvr: Some("Y".to_string()),
                class: Some("Station".to_string()),
                category: Some("Human spaceflight".to_string()),
                result: Some("S".to_string()),
                control: Some("NASA/RSA".to_string()),
                discipline: Some("Life sci".to_string()),
                un_state: Some("US".to_string()),
                un_reg: Some("1998-067A".to_string()),
                un_period: Some(92.9),
                un_perigee: Some(408.0),
                un_apogee: Some(418.0),
                un_inc: Some(51.6),
                disp_epoch: None,
                disp_peri: None,
                disp_apo: None,
                disp_inc: None,
                comment: Some("International Space Station".to_string()),
            },
            GCATPsatcatRecord {
                jcat: "S052103".to_string(),
                piece: Some("A".to_string()),
                name: Some("Starlink-1".to_string()),
                ldate: Some("2019 May 24".to_string()),
                tlast: None,
                top: Some("2019 Jun  1".to_string()),
                tdate: Some("2020 Jun  1".to_string()),
                tf: None,
                program: Some("Starlink".to_string()),
                plane: None,
                att: Some("3AX".to_string()),
                mvr: Some("Y".to_string()),
                class: Some("Com".to_string()),
                category: Some("Communications".to_string()),
                result: Some("S".to_string()),
                control: Some("SpaceX".to_string()),
                discipline: Some("Comm".to_string()),
                un_state: Some("US".to_string()),
                un_reg: None,
                un_period: None,
                un_perigee: None,
                un_apogee: None,
                un_inc: None,
                disp_epoch: None,
                disp_peri: None,
                disp_apo: None,
                disp_inc: None,
                comment: None,
            },
        ]
    }

    // === GCATSatcat tests ===

    #[test]
    fn test_satcat_new_and_len() {
        let catalog = GCATSatcat::new(sample_satcat_records());
        assert_eq!(catalog.len(), 3);
        assert!(!catalog.is_empty());
    }

    #[test]
    fn test_satcat_empty() {
        let catalog = GCATSatcat::new(vec![]);
        assert_eq!(catalog.len(), 0);
        assert!(catalog.is_empty());
    }

    #[test]
    fn test_satcat_records() {
        let catalog = GCATSatcat::new(sample_satcat_records());
        assert_eq!(catalog.records().len(), 3);
    }

    #[test]
    fn test_satcat_into_records() {
        let catalog = GCATSatcat::new(sample_satcat_records());
        let records = catalog.into_records();
        assert_eq!(records.len(), 3);
    }

    #[test]
    fn test_satcat_get_by_jcat() {
        let catalog = GCATSatcat::new(sample_satcat_records());
        let iss = catalog.get_by_jcat("S049652");
        assert!(iss.is_some());
        assert_eq!(iss.unwrap().name.as_deref(), Some("ISS (Zarya)"));

        assert!(catalog.get_by_jcat("NONEXISTENT").is_none());
    }

    #[test]
    fn test_satcat_get_by_satcat() {
        let catalog = GCATSatcat::new(sample_satcat_records());
        let iss = catalog.get_by_satcat("25544");
        assert!(iss.is_some());
        assert_eq!(iss.unwrap().jcat, "S049652");

        assert!(catalog.get_by_satcat("99999").is_none());
    }

    #[test]
    fn test_satcat_search_by_name() {
        let catalog = GCATSatcat::new(sample_satcat_records());

        let iss_results = catalog.search_by_name("iss");
        assert_eq!(iss_results.len(), 1);
        assert_eq!(iss_results.records()[0].jcat, "S049652");

        // Search by pl_name
        let zarya_results = catalog.search_by_name("zarya");
        assert_eq!(zarya_results.len(), 1);

        let starlink_results = catalog.search_by_name("starlink");
        assert_eq!(starlink_results.len(), 1);

        let no_results = catalog.search_by_name("nonexistent");
        assert_eq!(no_results.len(), 0);
    }

    #[test]
    fn test_satcat_filter_by_type() {
        let catalog = GCATSatcat::new(sample_satcat_records());

        let payloads = catalog.filter_by_type("P");
        assert_eq!(payloads.len(), 2);

        let rockets = catalog.filter_by_type("R");
        assert_eq!(rockets.len(), 1);
    }

    #[test]
    fn test_satcat_filter_by_owner() {
        let catalog = GCATSatcat::new(sample_satcat_records());

        let nasa = catalog.filter_by_owner("NASA");
        assert_eq!(nasa.len(), 1);

        let spacex = catalog.filter_by_owner("SpaceX");
        assert_eq!(spacex.len(), 1);
    }

    #[test]
    fn test_satcat_filter_by_state() {
        let catalog = GCATSatcat::new(sample_satcat_records());

        let us = catalog.filter_by_state("US");
        assert_eq!(us.len(), 2);

        let ru = catalog.filter_by_state("RU");
        assert_eq!(ru.len(), 1);
    }

    #[test]
    fn test_satcat_filter_by_status() {
        let catalog = GCATSatcat::new(sample_satcat_records());

        let operational = catalog.filter_by_status("O");
        assert_eq!(operational.len(), 2);

        let decayed = catalog.filter_by_status("D");
        assert_eq!(decayed.len(), 1);
    }

    #[test]
    fn test_satcat_filter_by_perigee_range() {
        let catalog = GCATSatcat::new(sample_satcat_records());

        let low = catalog.filter_by_perigee_range(100.0, 300.0);
        assert_eq!(low.len(), 1); // Proton-K at 185 km

        let mid = catalog.filter_by_perigee_range(400.0, 600.0);
        assert_eq!(mid.len(), 2); // ISS at 408 km, Starlink at 540 km
    }

    #[test]
    fn test_satcat_filter_by_apogee_range() {
        let catalog = GCATSatcat::new(sample_satcat_records());

        let low = catalog.filter_by_apogee_range(200.0, 300.0);
        assert_eq!(low.len(), 1); // Proton-K at 280 km
    }

    #[test]
    fn test_satcat_filter_by_inc_range() {
        let catalog = GCATSatcat::new(sample_satcat_records());

        let all_matching = catalog.filter_by_inc_range(50.0, 55.0);
        assert_eq!(all_matching.len(), 3);

        let none_matching = catalog.filter_by_inc_range(90.0, 100.0);
        assert_eq!(none_matching.len(), 0);
    }

    #[test]
    fn test_satcat_immutable_filters() {
        // Verify that filtering does not modify the original catalog
        let catalog = GCATSatcat::new(sample_satcat_records());
        let _payloads = catalog.filter_by_type("P");
        assert_eq!(catalog.len(), 3); // Original unchanged
    }

    // === GCATPsatcat tests ===

    #[test]
    fn test_psatcat_new_and_len() {
        let catalog = GCATPsatcat::new(sample_psatcat_records());
        assert_eq!(catalog.len(), 2);
        assert!(!catalog.is_empty());
    }

    #[test]
    fn test_psatcat_get_by_jcat() {
        let catalog = GCATPsatcat::new(sample_psatcat_records());
        let iss = catalog.get_by_jcat("S049652");
        assert!(iss.is_some());
        assert_eq!(iss.unwrap().name.as_deref(), Some("ISS (Zarya)"));
    }

    #[test]
    fn test_psatcat_search_by_name() {
        let catalog = GCATPsatcat::new(sample_psatcat_records());

        let results = catalog.search_by_name("starlink");
        assert_eq!(results.len(), 1);
        assert_eq!(results.records()[0].jcat, "S052103");
    }

    #[test]
    fn test_psatcat_filter_by_category() {
        let catalog = GCATPsatcat::new(sample_psatcat_records());

        let comms = catalog.filter_by_category("Communications");
        assert_eq!(comms.len(), 1);
    }

    #[test]
    fn test_psatcat_filter_by_class() {
        let catalog = GCATPsatcat::new(sample_psatcat_records());

        let stations = catalog.filter_by_class("Station");
        assert_eq!(stations.len(), 1);
    }

    #[test]
    fn test_psatcat_filter_by_result() {
        let catalog = GCATPsatcat::new(sample_psatcat_records());

        let success = catalog.filter_by_result("S");
        assert_eq!(success.len(), 2);
    }

    #[test]
    fn test_psatcat_filter_active() {
        let catalog = GCATPsatcat::new(sample_psatcat_records());

        // ISS: result="S", tdate=None → active
        // Starlink-1: result="S", tdate=Some → not active
        let active = catalog.filter_active();
        assert_eq!(active.len(), 1);
        assert_eq!(active.records()[0].jcat, "S049652");
    }

    #[test]
    fn test_psatcat_into_records() {
        let catalog = GCATPsatcat::new(sample_psatcat_records());
        let records = catalog.into_records();
        assert_eq!(records.len(), 2);
    }

    // === DataFrame tests ===

    #[test]
    fn test_satcat_to_dataframe() {
        let catalog = GCATSatcat::new(sample_satcat_records());
        let df = catalog.to_dataframe().unwrap();

        assert_eq!(df.height(), 3);
        assert_eq!(df.width(), 42); // 41 required columns + alt_names

        // Verify column names
        let col_names = df.get_column_names();
        assert_eq!(col_names[0].as_str(), "jcat");
        assert_eq!(col_names[1].as_str(), "satcat");
        assert_eq!(col_names[5].as_str(), "name");

        // Verify jcat values
        let jcat = df.column("jcat").unwrap();
        assert_eq!(jcat.str().unwrap().get(0), Some("S049652"));

        // Verify numeric column
        let perigee = df.column("perigee").unwrap();
        assert_eq!(perigee.f64().unwrap().get(0), Some(408.0));
    }

    #[test]
    fn test_satcat_to_dataframe_empty() {
        let catalog = GCATSatcat::new(vec![]);
        let df = catalog.to_dataframe().unwrap();
        assert_eq!(df.height(), 0);
        assert_eq!(df.width(), 42);
    }

    #[test]
    fn test_psatcat_to_dataframe() {
        let catalog = GCATPsatcat::new(sample_psatcat_records());
        let df = catalog.to_dataframe().unwrap();

        assert_eq!(df.height(), 2);
        assert_eq!(df.width(), 28);

        let jcat = df.column("jcat").unwrap();
        assert_eq!(jcat.str().unwrap().get(0), Some("S049652"));

        let un_period = df.column("un_period").unwrap();
        assert_eq!(un_period.f64().unwrap().get(0), Some(92.9));
    }

    #[test]
    fn test_psatcat_to_dataframe_empty() {
        let catalog = GCATPsatcat::new(vec![]);
        let df = catalog.to_dataframe().unwrap();
        assert_eq!(df.height(), 0);
        assert_eq!(df.width(), 28);
    }
}
