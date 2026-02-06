/*!
 * Type definitions for the CelestrakClient API client.
 *
 * Defines the core enums used to construct CelestrakClient queries:
 * query types, output formats, and supplemental GP data sources.
 */

use std::fmt;

/// CelestrakClient query type.
///
/// Determines which CelestrakClient API endpoint is queried.
///
/// # Examples
///
/// ```
/// use brahe::celestrak::CelestrakQueryType;
///
/// let qt = CelestrakQueryType::GP;
/// assert_eq!(qt.as_str(), "gp");
/// assert_eq!(qt.endpoint_path(), "/NORAD/elements/gp.php");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[allow(clippy::upper_case_acronyms)]
pub enum CelestrakQueryType {
    /// General Perturbations (OMM) data from the main GP endpoint.
    GP,
    /// Supplemental General Perturbations data from alternate sources.
    SupGP,
    /// Satellite Catalog records.
    SATCAT,
}

impl CelestrakQueryType {
    /// Returns a short string identifier for this query type.
    pub fn as_str(&self) -> &'static str {
        match self {
            CelestrakQueryType::GP => "gp",
            CelestrakQueryType::SupGP => "sup_gp",
            CelestrakQueryType::SATCAT => "satcat",
        }
    }

    /// Returns the API endpoint path for this query type.
    pub fn endpoint_path(&self) -> &'static str {
        match self {
            CelestrakQueryType::GP => "/NORAD/elements/gp.php",
            CelestrakQueryType::SupGP => "/NORAD/elements/supplemental/sup-gp.php",
            CelestrakQueryType::SATCAT => "/satcat/records.php",
        }
    }
}

impl fmt::Display for CelestrakQueryType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Output format for CelestrakClient query results.
///
/// Controls the format of the response data from CelestrakClient.
/// JSON is the default and is required for typed deserialization.
///
/// # Examples
///
/// ```
/// use brahe::celestrak::CelestrakOutputFormat;
///
/// let format = CelestrakOutputFormat::Json;
/// assert!(format.is_json());
/// assert_eq!(format.as_str(), "JSON");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CelestrakOutputFormat {
    /// TLE (Two-Line Element) format.
    Tle,
    /// 2LE format (TLE without the name line).
    TwoLe,
    /// 3LE format (TLE with name line).
    ThreeLe,
    /// XML format.
    Xml,
    /// KVN (Keyword-Value Notation) format.
    Kvn,
    /// JSON format.
    Json,
    /// JSON with pretty-printing.
    JsonPretty,
    /// CSV format.
    Csv,
}

impl CelestrakOutputFormat {
    /// Returns the query parameter value for this output format.
    pub fn as_str(&self) -> &'static str {
        match self {
            CelestrakOutputFormat::Tle => "TLE",
            CelestrakOutputFormat::TwoLe => "2LE",
            CelestrakOutputFormat::ThreeLe => "3LE",
            CelestrakOutputFormat::Xml => "XML",
            CelestrakOutputFormat::Kvn => "KVN",
            CelestrakOutputFormat::Json => "JSON",
            CelestrakOutputFormat::JsonPretty => "JSON-PRETTY",
            CelestrakOutputFormat::Csv => "CSV",
        }
    }

    /// Returns true if this format produces JSON output.
    pub fn is_json(&self) -> bool {
        matches!(
            self,
            CelestrakOutputFormat::Json | CelestrakOutputFormat::JsonPretty
        )
    }
}

impl fmt::Display for CelestrakOutputFormat {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Supplemental GP data source.
///
/// Identifies the source of supplemental GP data available from
/// CelestrakClient's sup-gp.php endpoint. These are operator-provided
/// ephemeris datasets not derived from the standard 18th SDS catalog.
///
/// # Examples
///
/// ```
/// use brahe::celestrak::SupGPSource;
///
/// let source = SupGPSource::SpaceX;
/// assert_eq!(source.as_str(), "spacex");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SupGPSource {
    /// SpaceX operator-provided ephemerides.
    SpaceX,
    /// SpaceX extended ephemerides.
    SpaceXSup,
    /// Planet Labs operator-provided ephemerides.
    Planet,
    /// OneWeb operator-provided ephemerides.
    OneWeb,
    /// Starlink ephemerides.
    Starlink,
    /// Starlink extended ephemerides.
    StarlinkSup,
    /// GEO protected zone supplemental data.
    Geo,
    /// GPS operational constellation.
    Gps,
    /// GLONASS operational constellation.
    Glonass,
    /// Meteosat supplemental data.
    Meteosat,
    /// Intelsat supplemental data.
    Intelsat,
    /// SES supplemental data.
    Ses,
    /// Iridium operator-provided ephemerides.
    Iridium,
    /// Iridium NEXT extended ephemerides.
    IridiumNext,
    /// Orbcomm supplemental data.
    Orbcomm,
    /// Globalstar supplemental data.
    Globalstar,
    /// Swarm supplemental data.
    SwarmTechnologies,
    /// Amateur radio satellites.
    Amateur,
    /// CelestrakClient special supplemental data.
    CelesTrak,
}

impl SupGPSource {
    /// Returns the query parameter value for this supplemental source.
    pub fn as_str(&self) -> &'static str {
        match self {
            SupGPSource::SpaceX => "spacex",
            SupGPSource::SpaceXSup => "spacex-sup",
            SupGPSource::Planet => "planet",
            SupGPSource::OneWeb => "oneweb",
            SupGPSource::Starlink => "starlink",
            SupGPSource::StarlinkSup => "starlink-sup",
            SupGPSource::Geo => "geo",
            SupGPSource::Gps => "gps",
            SupGPSource::Glonass => "glonass",
            SupGPSource::Meteosat => "meteosat",
            SupGPSource::Intelsat => "intelsat",
            SupGPSource::Ses => "ses",
            SupGPSource::Iridium => "iridium",
            SupGPSource::IridiumNext => "iridium-next",
            SupGPSource::Orbcomm => "orbcomm",
            SupGPSource::Globalstar => "globalstar",
            SupGPSource::SwarmTechnologies => "swarm",
            SupGPSource::Amateur => "amateur",
            SupGPSource::CelesTrak => "celestrak",
        }
    }
}

impl fmt::Display for SupGPSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    // -- CelestrakQueryType tests --

    #[test]
    fn test_query_type_as_str() {
        assert_eq!(CelestrakQueryType::GP.as_str(), "gp");
        assert_eq!(CelestrakQueryType::SupGP.as_str(), "sup_gp");
        assert_eq!(CelestrakQueryType::SATCAT.as_str(), "satcat");
    }

    #[test]
    fn test_query_type_endpoint_path() {
        assert_eq!(
            CelestrakQueryType::GP.endpoint_path(),
            "/NORAD/elements/gp.php"
        );
        assert_eq!(
            CelestrakQueryType::SupGP.endpoint_path(),
            "/NORAD/elements/supplemental/sup-gp.php"
        );
        assert_eq!(
            CelestrakQueryType::SATCAT.endpoint_path(),
            "/satcat/records.php"
        );
    }

    #[test]
    fn test_query_type_display() {
        assert_eq!(format!("{}", CelestrakQueryType::GP), "gp");
        assert_eq!(format!("{}", CelestrakQueryType::SupGP), "sup_gp");
        assert_eq!(format!("{}", CelestrakQueryType::SATCAT), "satcat");
    }

    // -- CelestrakOutputFormat tests --

    #[test]
    fn test_output_format_as_str() {
        assert_eq!(CelestrakOutputFormat::Tle.as_str(), "TLE");
        assert_eq!(CelestrakOutputFormat::TwoLe.as_str(), "2LE");
        assert_eq!(CelestrakOutputFormat::ThreeLe.as_str(), "3LE");
        assert_eq!(CelestrakOutputFormat::Xml.as_str(), "XML");
        assert_eq!(CelestrakOutputFormat::Kvn.as_str(), "KVN");
        assert_eq!(CelestrakOutputFormat::Json.as_str(), "JSON");
        assert_eq!(CelestrakOutputFormat::JsonPretty.as_str(), "JSON-PRETTY");
        assert_eq!(CelestrakOutputFormat::Csv.as_str(), "CSV");
    }

    #[test]
    fn test_output_format_display() {
        assert_eq!(format!("{}", CelestrakOutputFormat::Tle), "TLE");
        assert_eq!(format!("{}", CelestrakOutputFormat::TwoLe), "2LE");
        assert_eq!(format!("{}", CelestrakOutputFormat::ThreeLe), "3LE");
        assert_eq!(format!("{}", CelestrakOutputFormat::Xml), "XML");
        assert_eq!(format!("{}", CelestrakOutputFormat::Kvn), "KVN");
        assert_eq!(format!("{}", CelestrakOutputFormat::Json), "JSON");
        assert_eq!(
            format!("{}", CelestrakOutputFormat::JsonPretty),
            "JSON-PRETTY"
        );
        assert_eq!(format!("{}", CelestrakOutputFormat::Csv), "CSV");
    }

    #[test]
    fn test_output_format_is_json() {
        assert!(CelestrakOutputFormat::Json.is_json());
        assert!(CelestrakOutputFormat::JsonPretty.is_json());
        assert!(!CelestrakOutputFormat::Tle.is_json());
        assert!(!CelestrakOutputFormat::TwoLe.is_json());
        assert!(!CelestrakOutputFormat::ThreeLe.is_json());
        assert!(!CelestrakOutputFormat::Xml.is_json());
        assert!(!CelestrakOutputFormat::Kvn.is_json());
        assert!(!CelestrakOutputFormat::Csv.is_json());
    }

    // -- SupGPSource tests --

    #[test]
    fn test_sup_gp_source_as_str() {
        assert_eq!(SupGPSource::SpaceX.as_str(), "spacex");
        assert_eq!(SupGPSource::SpaceXSup.as_str(), "spacex-sup");
        assert_eq!(SupGPSource::Planet.as_str(), "planet");
        assert_eq!(SupGPSource::OneWeb.as_str(), "oneweb");
        assert_eq!(SupGPSource::Starlink.as_str(), "starlink");
        assert_eq!(SupGPSource::StarlinkSup.as_str(), "starlink-sup");
        assert_eq!(SupGPSource::Geo.as_str(), "geo");
        assert_eq!(SupGPSource::Gps.as_str(), "gps");
        assert_eq!(SupGPSource::Glonass.as_str(), "glonass");
        assert_eq!(SupGPSource::Meteosat.as_str(), "meteosat");
        assert_eq!(SupGPSource::Intelsat.as_str(), "intelsat");
        assert_eq!(SupGPSource::Ses.as_str(), "ses");
        assert_eq!(SupGPSource::Iridium.as_str(), "iridium");
        assert_eq!(SupGPSource::IridiumNext.as_str(), "iridium-next");
        assert_eq!(SupGPSource::Orbcomm.as_str(), "orbcomm");
        assert_eq!(SupGPSource::Globalstar.as_str(), "globalstar");
        assert_eq!(SupGPSource::SwarmTechnologies.as_str(), "swarm");
        assert_eq!(SupGPSource::Amateur.as_str(), "amateur");
        assert_eq!(SupGPSource::CelesTrak.as_str(), "celestrak");
    }

    #[test]
    fn test_sup_gp_source_display() {
        assert_eq!(format!("{}", SupGPSource::SpaceX), "spacex");
        assert_eq!(format!("{}", SupGPSource::Planet), "planet");
        assert_eq!(format!("{}", SupGPSource::Starlink), "starlink");
        assert_eq!(format!("{}", SupGPSource::Iridium), "iridium");
        assert_eq!(format!("{}", SupGPSource::CelesTrak), "celestrak");
    }

    // -- Shared trait tests --

    #[test]
    fn test_enum_equality() {
        assert_eq!(CelestrakQueryType::GP, CelestrakQueryType::GP);
        assert_ne!(CelestrakQueryType::GP, CelestrakQueryType::SATCAT);
        assert_eq!(CelestrakOutputFormat::Json, CelestrakOutputFormat::Json);
        assert_ne!(CelestrakOutputFormat::Json, CelestrakOutputFormat::Csv);
        assert_eq!(SupGPSource::SpaceX, SupGPSource::SpaceX);
        assert_ne!(SupGPSource::SpaceX, SupGPSource::Planet);
    }

    #[test]
    fn test_enum_clone() {
        let qt = CelestrakQueryType::GP;
        let cloned = qt;
        assert_eq!(qt, cloned);

        let fmt = CelestrakOutputFormat::Json;
        let cloned = fmt;
        assert_eq!(fmt, cloned);

        let src = SupGPSource::SpaceX;
        let cloned = src;
        assert_eq!(src, cloned);
    }

    #[test]
    fn test_enum_debug() {
        assert_eq!(format!("{:?}", CelestrakQueryType::GP), "GP");
        assert_eq!(format!("{:?}", CelestrakQueryType::SupGP), "SupGP");
        assert_eq!(format!("{:?}", CelestrakQueryType::SATCAT), "SATCAT");
        assert_eq!(format!("{:?}", CelestrakOutputFormat::Json), "Json");
        assert_eq!(format!("{:?}", CelestrakOutputFormat::ThreeLe), "ThreeLe");
        assert_eq!(format!("{:?}", SupGPSource::SpaceX), "SpaceX");
        assert_eq!(format!("{:?}", SupGPSource::IridiumNext), "IridiumNext");
    }
}
