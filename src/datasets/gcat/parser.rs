/*!
 * TSV parsing logic for GCAT satellite catalog files.
 *
 * Handles the GCAT-specific TSV format where:
 * - The first header column is prefixed with `#`
 * - Comment lines after the header start with `#`
 * - Missing/unknown values are represented as `-` or empty strings
 * - Numeric fields may contain whitespace
 */

use crate::datasets::gcat::records::{GCATPsatcatRecord, GCATSatcatRecord};
use crate::utils::BraheError;

/// Parse a string value from a TSV field.
///
/// Returns `None` for empty strings, whitespace-only strings, or the
/// placeholder `-` that GCAT uses for missing values.
fn parse_optional_string(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed == "-" {
        None
    } else {
        Some(trimmed.to_string())
    }
}

/// Parse a numeric (f64) value from a TSV field.
///
/// Returns `None` for empty strings, `-`, or values that cannot be parsed
/// as floating-point numbers.
fn parse_optional_f64(value: &str) -> Option<f64> {
    let trimmed = value.trim();
    if trimmed.is_empty() || trimmed == "-" {
        None
    } else {
        trimmed.parse::<f64>().ok()
    }
}

/// Parse GCAT SATCAT TSV data into a vector of records.
///
/// Expects the standard `satcat.tsv` format with 42 tab-separated columns
/// (41 standard plus an optional `alt_names` column).
/// The first line must be a header row (first column prefixed with `#`).
/// Lines starting with `#` after the header are treated as comments and skipped.
///
/// # Arguments
///
/// * `data` - Raw TSV string content
///
/// # Returns
///
/// * `Result<Vec<GCATSatcatRecord>, BraheError>` - Parsed records
///
/// # Examples
/// ```no_run
/// use brahe::datasets::gcat::parser::parse_satcat_tsv;
/// let data = std::fs::read_to_string("satcat.tsv").unwrap();
/// let records = parse_satcat_tsv(&data).unwrap();
/// ```
pub fn parse_satcat_tsv(data: &str) -> Result<Vec<GCATSatcatRecord>, BraheError> {
    let mut lines = data.lines();

    // Skip header line (starts with #)
    let header = lines
        .next()
        .ok_or_else(|| BraheError::ParseError("GCAT SATCAT TSV data is empty".to_string()))?;

    if !header.starts_with('#') {
        return Err(BraheError::ParseError(
            "GCAT SATCAT TSV header must start with '#'".to_string(),
        ));
    }

    let mut records = Vec::new();

    for (line_num, line) in lines.enumerate() {
        // Skip comment lines and empty lines
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();

        if fields.len() < 41 {
            return Err(BraheError::ParseError(format!(
                "GCAT SATCAT line {} has {} columns, expected 41",
                line_num + 2,
                fields.len()
            )));
        }

        let jcat = fields[0].trim().to_string();
        if jcat.is_empty() || jcat == "-" {
            return Err(BraheError::ParseError(format!(
                "GCAT SATCAT line {} has empty JCAT identifier",
                line_num + 2
            )));
        }

        records.push(GCATSatcatRecord {
            jcat,
            satcat: parse_optional_string(fields[1]),
            launch_tag: parse_optional_string(fields[2]),
            piece: parse_optional_string(fields[3]),
            object_type: parse_optional_string(fields[4]),
            name: parse_optional_string(fields[5]),
            pl_name: parse_optional_string(fields[6]),
            ldate: parse_optional_string(fields[7]),
            parent: parse_optional_string(fields[8]),
            sdate: parse_optional_string(fields[9]),
            primary: parse_optional_string(fields[10]),
            ddate: parse_optional_string(fields[11]),
            status: parse_optional_string(fields[12]),
            dest: parse_optional_string(fields[13]),
            owner: parse_optional_string(fields[14]),
            state: parse_optional_string(fields[15]),
            manufacturer: parse_optional_string(fields[16]),
            bus: parse_optional_string(fields[17]),
            motor: parse_optional_string(fields[18]),
            mass: parse_optional_f64(fields[19]),
            mass_flag: parse_optional_string(fields[20]),
            dry_mass: parse_optional_f64(fields[21]),
            dry_flag: parse_optional_string(fields[22]),
            tot_mass: parse_optional_f64(fields[23]),
            tot_flag: parse_optional_string(fields[24]),
            length: parse_optional_f64(fields[25]),
            length_flag: parse_optional_string(fields[26]),
            diameter: parse_optional_f64(fields[27]),
            diameter_flag: parse_optional_string(fields[28]),
            span: parse_optional_f64(fields[29]),
            span_flag: parse_optional_string(fields[30]),
            shape: parse_optional_string(fields[31]),
            odate: parse_optional_string(fields[32]),
            perigee: parse_optional_f64(fields[33]),
            perigee_flag: parse_optional_string(fields[34]),
            apogee: parse_optional_f64(fields[35]),
            apogee_flag: parse_optional_string(fields[36]),
            inc: parse_optional_f64(fields[37]),
            inc_flag: parse_optional_string(fields[38]),
            op_orbit: parse_optional_string(fields[39]),
            oqual: parse_optional_string(fields[40]),
            // Column 41 (alt_names) may be absent in some rows
            alt_names: fields.get(41).and_then(|f| parse_optional_string(f)),
        });
    }

    Ok(records)
}

/// Parse GCAT PSATCAT TSV data into a vector of records.
///
/// Expects the standard `psatcat.tsv` format with 28 tab-separated columns.
/// The first line must be a header row (first column prefixed with `#`).
/// Lines starting with `#` after the header are treated as comments and skipped.
///
/// # Arguments
///
/// * `data` - Raw TSV string content
///
/// # Returns
///
/// * `Result<Vec<GCATPsatcatRecord>, BraheError>` - Parsed records
///
/// # Examples
/// ```no_run
/// use brahe::datasets::gcat::parser::parse_psatcat_tsv;
/// let data = std::fs::read_to_string("psatcat.tsv").unwrap();
/// let records = parse_psatcat_tsv(&data).unwrap();
/// ```
pub fn parse_psatcat_tsv(data: &str) -> Result<Vec<GCATPsatcatRecord>, BraheError> {
    let mut lines = data.lines();

    // Skip header line (starts with #)
    let header = lines
        .next()
        .ok_or_else(|| BraheError::ParseError("GCAT PSATCAT TSV data is empty".to_string()))?;

    if !header.starts_with('#') {
        return Err(BraheError::ParseError(
            "GCAT PSATCAT TSV header must start with '#'".to_string(),
        ));
    }

    let mut records = Vec::new();

    for (line_num, line) in lines.enumerate() {
        // Skip comment lines and empty lines
        if line.starts_with('#') || line.trim().is_empty() {
            continue;
        }

        let fields: Vec<&str> = line.split('\t').collect();

        if fields.len() < 28 {
            return Err(BraheError::ParseError(format!(
                "GCAT PSATCAT line {} has {} columns, expected 28",
                line_num + 2,
                fields.len()
            )));
        }

        let jcat = fields[0].trim().to_string();
        if jcat.is_empty() || jcat == "-" {
            return Err(BraheError::ParseError(format!(
                "GCAT PSATCAT line {} has empty JCAT identifier",
                line_num + 2
            )));
        }

        records.push(GCATPsatcatRecord {
            jcat,
            piece: parse_optional_string(fields[1]),
            name: parse_optional_string(fields[2]),
            ldate: parse_optional_string(fields[3]),
            tlast: parse_optional_string(fields[4]),
            top: parse_optional_string(fields[5]),
            tdate: parse_optional_string(fields[6]),
            tf: parse_optional_string(fields[7]),
            program: parse_optional_string(fields[8]),
            plane: parse_optional_string(fields[9]),
            att: parse_optional_string(fields[10]),
            mvr: parse_optional_string(fields[11]),
            class: parse_optional_string(fields[12]),
            category: parse_optional_string(fields[13]),
            result: parse_optional_string(fields[14]),
            control: parse_optional_string(fields[15]),
            discipline: parse_optional_string(fields[16]),
            un_state: parse_optional_string(fields[17]),
            un_reg: parse_optional_string(fields[18]),
            un_period: parse_optional_f64(fields[19]),
            un_perigee: parse_optional_f64(fields[20]),
            un_apogee: parse_optional_f64(fields[21]),
            un_inc: parse_optional_f64(fields[22]),
            disp_epoch: parse_optional_string(fields[23]),
            disp_peri: parse_optional_f64(fields[24]),
            disp_apo: parse_optional_f64(fields[25]),
            disp_inc: parse_optional_f64(fields[26]),
            comment: parse_optional_string(fields[27]),
        });
    }

    Ok(records)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    fn sample_satcat_tsv() -> String {
        // 42 columns: jcat through alt_names (41 required + 1 optional alt_names)
        let header = "#JCAT\tSatcat\tLTag\tPiece\tType\tName\tPLName\tLDate\tParent\tSDate\tPrimary\tDDate\tStatus\tDest\tOwner\tState\tManufacturer\tBus\tMotor\tMass\tMassFlag\tDryMass\tDryFlag\tTotMass\tTotFlag\tLength\tLengthFlag\tDiameter\tDiameterFlag\tSpan\tSpanFlag\tShape\tODate\tPerigee\tPerigeeFlag\tApogee\tApogeeFlag\tInc\tIncFlag\tOpOrbit\tOQual\tAltNames";
        let row1 = "S049652\t25544\t1998-067\tA\tP\tISS (Zarya)\tZarya\t1998 Nov 20\t-\t1998 Nov 20\tE\t-\tO\tLEO\tNASA\tUS\tKhrunichev\tFGB\t-\t19323\t-\t-\t-\t-\t-\t12.6\t-\t4.1\t-\t73.2\t-\tCyl\t2020 Jan  1\t408\t-\t418\t-\t51.6\t-\tLEO/I\tQ\tZarya";
        let row2 = "S049653\t25545\t1998-067\tB\tR\tProton-K Block-DM Blk DM\t-\t1998 Nov 20\t-\t1998 Nov 20\tE\t1998 Dec  9\tD\tLEO\t-\tRU\tKhrunichev\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t185\t-\t280\t-\t51.6\t-\tLEO/I\tE\t-";
        format!("{}\n{}\n{}\n", header, row1, row2)
    }

    fn sample_psatcat_tsv() -> String {
        let header = "#JCAT\tPiece\tName\tLDate\tTLast\tTop\tTDate\tTF\tProgram\tPlane\tAtt\tMvr\tClass\tCategory\tResult\tControl\tDiscipline\tUNState\tUNReg\tUNPeriod\tUNPerigee\tUNApogee\tUNInc\tDispEpoch\tDispPeri\tDispApo\tDispInc\tComment";
        let row1 = "S049652\tA\tISS (Zarya)\t1998 Nov 20\t2025 Jan  1\t1998 Nov 20\t-\t-\tISS\t-\t3AX\tY\tStation\tHuman spaceflight\tS\tNASA/RSA\tLife sci\tUS\t1998-067A\t92.9\t408\t418\t51.6\t-\t-\t-\t-\tInternational Space Station";
        let row2 = "S052103\tA\tStarlink-1\t2019 May 24\t2020 Jun  1\t2019 Jun  1\t2020 Jun  1\t-\tStarlink\t-\t3AX\tY\tCom\tCommunications\tS\tSpaceX\tComm\tUS\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-";
        format!("{}\n{}\n{}\n", header, row1, row2)
    }

    #[test]
    fn test_parse_satcat_basic() {
        let data = sample_satcat_tsv();
        let records = parse_satcat_tsv(&data).unwrap();
        assert_eq!(records.len(), 2);

        let iss = &records[0];
        assert_eq!(iss.jcat, "S049652");
        assert_eq!(iss.satcat.as_deref(), Some("25544"));
        assert_eq!(iss.name.as_deref(), Some("ISS (Zarya)"));
        assert_eq!(iss.object_type.as_deref(), Some("P"));
        assert_eq!(iss.status.as_deref(), Some("O"));
        assert_eq!(iss.mass, Some(19323.0));
        assert_eq!(iss.perigee, Some(408.0));
        assert_eq!(iss.apogee, Some(418.0));
        assert_eq!(iss.inc, Some(51.6));
        assert_eq!(iss.alt_names.as_deref(), Some("Zarya"));

        let rb = &records[1];
        assert_eq!(rb.jcat, "S049653");
        assert_eq!(rb.object_type.as_deref(), Some("R"));
        assert!(rb.mass.is_none()); // "-" → None
        assert_eq!(rb.ddate.as_deref(), Some("1998 Dec  9"));
        assert!(rb.alt_names.is_none()); // "-" → None
    }

    #[test]
    fn test_parse_psatcat_basic() {
        let data = sample_psatcat_tsv();
        let records = parse_psatcat_tsv(&data).unwrap();
        assert_eq!(records.len(), 2);

        let iss = &records[0];
        assert_eq!(iss.jcat, "S049652");
        assert_eq!(iss.name.as_deref(), Some("ISS (Zarya)"));
        assert_eq!(iss.program.as_deref(), Some("ISS"));
        assert_eq!(iss.category.as_deref(), Some("Human spaceflight"));
        assert_eq!(iss.un_period, Some(92.9));
        assert_eq!(iss.un_perigee, Some(408.0));
        assert_eq!(iss.un_inc, Some(51.6));
        assert_eq!(iss.comment.as_deref(), Some("International Space Station"));

        let starlink = &records[1];
        assert_eq!(starlink.jcat, "S052103");
        assert_eq!(starlink.program.as_deref(), Some("Starlink"));
        assert!(starlink.un_period.is_none());
        assert!(starlink.comment.is_none()); // "-" → None
    }

    #[test]
    fn test_parse_satcat_empty_data() {
        let result = parse_satcat_tsv("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_satcat_no_hash_header() {
        let data = "JCAT\tSatcat\n";
        let result = parse_satcat_tsv(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_satcat_insufficient_columns() {
        let data = "#JCAT\tSatcat\nS001\tonly_two_cols\n";
        let result = parse_satcat_tsv(data);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_satcat_skips_comments() {
        let header = "#JCAT\tSatcat\tLTag\tPiece\tType\tName\tPLName\tLDate\tParent\tSDate\tPrimary\tDDate\tStatus\tDest\tOwner\tState\tManufacturer\tBus\tMotor\tMass\tMassFlag\tDryMass\tDryFlag\tTotMass\tTotFlag\tLength\tLengthFlag\tDiameter\tDiameterFlag\tSpan\tSpanFlag\tShape\tODate\tPerigee\tPerigeeFlag\tApogee\tApogeeFlag\tInc\tIncFlag\tOpOrbit\tOQual\tAltNames";
        let comment = "# This is a comment";
        let row = "S049652\t25544\t1998-067\tA\tP\tISS\t-\t1998 Nov 20\t-\t1998 Nov 20\tE\t-\tO\tLEO\tNASA\tUS\t-\t-\t-\t19323\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t408\t-\t418\t-\t51.6\t-\tLEO/I\tQ\t-";
        let data = format!("{}\n{}\n{}\n", header, comment, row);
        let records = parse_satcat_tsv(&data).unwrap();
        assert_eq!(records.len(), 1);
    }

    #[test]
    fn test_parse_satcat_skips_empty_lines() {
        let header = "#JCAT\tSatcat\tLTag\tPiece\tType\tName\tPLName\tLDate\tParent\tSDate\tPrimary\tDDate\tStatus\tDest\tOwner\tState\tManufacturer\tBus\tMotor\tMass\tMassFlag\tDryMass\tDryFlag\tTotMass\tTotFlag\tLength\tLengthFlag\tDiameter\tDiameterFlag\tSpan\tSpanFlag\tShape\tODate\tPerigee\tPerigeeFlag\tApogee\tApogeeFlag\tInc\tIncFlag\tOpOrbit\tOQual\tAltNames";
        let row = "S049652\t25544\t1998-067\tA\tP\tISS\t-\t1998 Nov 20\t-\t1998 Nov 20\tE\t-\tO\tLEO\tNASA\tUS\t-\t-\t-\t19323\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t-\t408\t-\t418\t-\t51.6\t-\tLEO/I\tQ\t-";
        let data = format!("{}\n\n{}\n\n", header, row);
        let records = parse_satcat_tsv(&data).unwrap();
        assert_eq!(records.len(), 1);
    }

    #[test]
    fn test_parse_optional_string() {
        assert_eq!(parse_optional_string("hello"), Some("hello".to_string()));
        assert_eq!(parse_optional_string(" hello "), Some("hello".to_string()));
        assert_eq!(parse_optional_string("-"), None);
        assert_eq!(parse_optional_string(""), None);
        assert_eq!(parse_optional_string("  "), None);
    }

    #[test]
    fn test_parse_optional_f64() {
        assert_eq!(parse_optional_f64("42.5"), Some(42.5));
        assert_eq!(parse_optional_f64(" 42.5 "), Some(42.5));
        assert_eq!(parse_optional_f64("-"), None);
        assert_eq!(parse_optional_f64(""), None);
        assert_eq!(parse_optional_f64("not_a_number"), None);
    }

    #[test]
    fn test_parse_psatcat_empty_data() {
        let result = parse_psatcat_tsv("");
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_psatcat_no_hash_header() {
        let data = "JCAT\tPiece\n";
        let result = parse_psatcat_tsv(data);
        assert!(result.is_err());
    }
}
