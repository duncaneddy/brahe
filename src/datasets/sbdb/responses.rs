//! SBDB Lookup API response parsing.

use serde::Deserialize;

use crate::utils::BraheError;

/// A resolved JPL Small-Body Database object.
///
/// Physical parameters (`gm`, `radius`) are populated only when the database
/// provides them, converted to SI units on parse.
#[derive(Debug, Clone)]
pub struct SBDBObject {
    /// Primary SPK-ID (NAIF ID), e.g. `2000001` for Ceres.
    pub spkid: i32,
    /// Full designation and name, e.g. `"1 Ceres"`.
    pub full_name: String,
    /// Primary designation, e.g. `"1"`.
    pub des: String,
    /// Object name without alternate designation, when distinct.
    pub shortname: Option<String>,
    /// Object kind code, e.g. `"an"` (numbered asteroid).
    pub kind: String,
    /// Whether the object is a near-Earth object.
    pub neo: bool,
    /// Gravitational parameter [m^3/s^2], if catalogued.
    pub gm: Option<f64>,
    /// Mean radius [m], derived from the catalogued diameter, if present.
    pub radius: Option<f64>,
}

#[derive(Deserialize)]
struct SbdbRawResponse {
    object: Option<SbdbRawObject>,
    #[serde(default)]
    phys_par: Vec<SbdbRawPhysPar>,
    message: Option<String>,
    #[serde(default)]
    list: Vec<SbdbRawListEntry>,
}

#[derive(Deserialize)]
struct SbdbRawObject {
    spkid: String,
    fullname: String,
    des: String,
    shortname: Option<String>,
    kind: String,
    #[serde(default)]
    neo: bool,
}

#[derive(Deserialize)]
struct SbdbRawPhysPar {
    name: String,
    value: Option<String>,
}

#[derive(Deserialize)]
struct SbdbRawListEntry {
    pdes: Option<String>,
    name: Option<String>,
}

impl SBDBObject {
    /// The object's NAIF ID (identical to its SPK-ID).
    pub fn naif_id(&self) -> i32 {
        self.spkid
    }

    /// Parse an SBDB Lookup API JSON response body into an [`SBDBObject`].
    ///
    /// Returns an error for ambiguous matches (enumerating the candidates),
    /// no-match responses, or malformed JSON.
    pub(crate) fn from_json(body: &str) -> Result<SBDBObject, BraheError> {
        let raw: SbdbRawResponse = serde_json::from_str(body)
            .map_err(|e| BraheError::ParseError(format!("Failed to parse SBDB response: {}", e)))?;

        // Ambiguous match: SBDB returns code "300" with a candidate list.
        if !raw.list.is_empty() {
            let candidates = raw
                .list
                .iter()
                .map(|c| {
                    format!(
                        "{} ({})",
                        c.name.as_deref().unwrap_or("?"),
                        c.pdes.as_deref().unwrap_or("?")
                    )
                })
                .collect::<Vec<_>>()
                .join(", ");
            return Err(BraheError::Error(format!(
                "SBDB lookup matched multiple objects; specify a designation. Candidates: {}",
                candidates
            )));
        }

        let object = raw.object.ok_or_else(|| {
            let msg = raw
                .message
                .unwrap_or_else(|| "specified object was not found".to_string());
            BraheError::Error(format!("SBDB lookup: {}", msg))
        })?;

        let spkid = object.spkid.parse::<i32>().map_err(|e| {
            BraheError::ParseError(format!("Invalid SBDB spkid '{}': {}", object.spkid, e))
        })?;

        let mut gm = None;
        let mut radius = None;
        for pp in &raw.phys_par {
            let value = match pp.value.as_deref().and_then(|v| v.parse::<f64>().ok()) {
                Some(v) => v,
                None => continue,
            };
            match pp.name.as_str() {
                // SBDB GM is km^3/s^2 -> m^3/s^2.
                "GM" => gm = Some(value * 1.0e9),
                // SBDB diameter is km; radius = diameter/2 in meters.
                "diameter" => radius = Some(value * 1000.0 / 2.0),
                _ => {}
            }
        }

        Ok(SBDBObject {
            spkid,
            full_name: object.fullname,
            des: object.des,
            shortname: object.shortname,
            kind: object.kind,
            neo: object.neo,
            gm,
            radius,
        })
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    const CERES_JSON: &str = r#"{
        "object": {"spkid":"2000001","fullname":"1 Ceres","des":"1",
                   "shortname":"Ceres","neo":false,"kind":"an"},
        "phys_par": [
            {"name":"GM","value":"62.6284","units":"km^3/s^2"},
            {"name":"diameter","value":"939.4","units":"km"}
        ]
    }"#;

    #[test]
    fn test_parse_ceres_success() {
        let obj = SBDBObject::from_json(CERES_JSON).unwrap();
        assert_eq!(obj.spkid, 2000001);
        assert_eq!(obj.naif_id(), 2000001);
        assert_eq!(obj.full_name, "1 Ceres");
        assert_eq!(obj.des, "1");
        assert_eq!(obj.shortname.as_deref(), Some("Ceres"));
        assert_eq!(obj.kind, "an");
        assert!(!obj.neo);
        // GM: 62.6284 km^3/s^2 -> 6.26284e10 m^3/s^2
        assert!((obj.gm.unwrap() - 6.26284e10).abs() < 1.0e6);
        // radius: 939.4 km diameter / 2 -> 469.7 km -> 469_700 m
        assert!((obj.radius.unwrap() - 469_700.0).abs() < 1.0);
    }

    #[test]
    fn test_parse_missing_phys_par_is_none() {
        let json = r#"{"object":{"spkid":"2000004","fullname":"4 Vesta",
                       "des":"4","neo":false,"kind":"an"}}"#;
        let obj = SBDBObject::from_json(json).unwrap();
        assert_eq!(obj.spkid, 2000004);
        assert_eq!(obj.shortname, None);
        assert_eq!(obj.gm, None);
        assert_eq!(obj.radius, None);
    }

    #[test]
    fn test_parse_ambiguous_lists_candidates() {
        let json = r#"{"code":"300","list":[
            {"pdes":"1","name":"Ceres"},{"pdes":"1P","name":"Halley"}]}"#;
        let err = SBDBObject::from_json(json).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("multiple"));
        assert!(msg.contains("Ceres"));
        assert!(msg.contains("Halley"));
    }

    #[test]
    fn test_parse_not_found() {
        let json = r#"{"code":"200","message":"specified object was not found"}"#;
        let err = SBDBObject::from_json(json).unwrap_err();
        assert!(err.to_string().contains("not found"));
    }

    #[test]
    fn test_parse_invalid_json() {
        let err = SBDBObject::from_json("not json").unwrap_err();
        assert!(matches!(err, BraheError::ParseError(_)));
    }
}
