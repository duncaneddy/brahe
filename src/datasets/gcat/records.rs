/*!
 * Record types for GCAT (General Catalog of Artificial Space Objects) data.
 *
 * Defines `GCATSatcatRecord` (41-column satellite catalog) and
 * `GCATPsatcatRecord` (28-column payload catalog) structs that mirror
 * the TSV schemas from Jonathan McDowell's GCAT.
 */

/// A single record from the GCAT SATCAT (satellite catalog).
///
/// Contains physical, orbital, and administrative metadata for an artificial
/// space object. All 41 columns from the `satcat.tsv` file are represented.
/// Orbital values (perigee, apogee) are in km and inclination in degrees,
/// matching the source data.
#[derive(Debug, Clone)]
pub struct GCATSatcatRecord {
    /// JCAT catalog identifier (primary key)
    pub jcat: String,
    /// NORAD/USSPACECOM SATCAT number
    pub satcat: Option<String>,
    /// Launch tag identifier
    pub launch_tag: Option<String>,
    /// Piece designation
    pub piece: Option<String>,
    /// Object type (e.g. "P" for payload, "R" for rocket body)
    pub object_type: Option<String>,
    /// Current object name
    pub name: Option<String>,
    /// Payload name (may differ from current name)
    pub pl_name: Option<String>,
    /// Launch date
    pub ldate: Option<String>,
    /// Parent object identifier
    pub parent: Option<String>,
    /// Separation/deployment date
    pub sdate: Option<String>,
    /// Primary body orbited
    pub primary: Option<String>,
    /// Decay/deorbit date
    pub ddate: Option<String>,
    /// Current status code
    pub status: Option<String>,
    /// Destination/orbit description
    pub dest: Option<String>,
    /// Owner/operator organization
    pub owner: Option<String>,
    /// Responsible state/country
    pub state: Option<String>,
    /// Manufacturer organization
    pub manufacturer: Option<String>,
    /// Spacecraft bus type
    pub bus: Option<String>,
    /// Motor/propulsion type
    pub motor: Option<String>,
    /// Launch mass in kg
    pub mass: Option<f64>,
    /// Mass quality flag
    pub mass_flag: Option<String>,
    /// Dry mass in kg
    pub dry_mass: Option<f64>,
    /// Dry mass quality flag
    pub dry_flag: Option<String>,
    /// Total mass in kg
    pub tot_mass: Option<f64>,
    /// Total mass quality flag
    pub tot_flag: Option<String>,
    /// Length in meters
    pub length: Option<f64>,
    /// Length quality flag
    pub length_flag: Option<String>,
    /// Diameter in meters
    pub diameter: Option<f64>,
    /// Diameter quality flag
    pub diameter_flag: Option<String>,
    /// Span (solar array span) in meters
    pub span: Option<f64>,
    /// Span quality flag
    pub span_flag: Option<String>,
    /// Shape description
    pub shape: Option<String>,
    /// Operational orbit epoch date
    pub odate: Option<String>,
    /// Perigee altitude in km
    pub perigee: Option<f64>,
    /// Perigee quality flag
    pub perigee_flag: Option<String>,
    /// Apogee altitude in km
    pub apogee: Option<f64>,
    /// Apogee quality flag
    pub apogee_flag: Option<String>,
    /// Orbital inclination in degrees
    pub inc: Option<f64>,
    /// Inclination quality flag
    pub inc_flag: Option<String>,
    /// Operational orbit class
    pub op_orbit: Option<String>,
    /// Orbit quality code
    pub oqual: Option<String>,
    /// Alternative names
    pub alt_names: Option<String>,
}

/// A single record from the GCAT PSATCAT (payload satellite catalog).
///
/// Contains payload-specific metadata including mission details, UN registry
/// information, and disposal orbit parameters. All 28 columns from the
/// `psatcat.tsv` file are represented.
#[derive(Debug, Clone)]
pub struct GCATPsatcatRecord {
    /// JCAT catalog identifier (primary key)
    pub jcat: String,
    /// Piece designation
    pub piece: Option<String>,
    /// Payload name
    pub name: Option<String>,
    /// Launch date
    pub ldate: Option<String>,
    /// Last contact date
    pub tlast: Option<String>,
    /// Operational start date
    pub top: Option<String>,
    /// End of operations date
    pub tdate: Option<String>,
    /// Date quality flag
    pub tf: Option<String>,
    /// Program/constellation name
    pub program: Option<String>,
    /// Orbital plane identifier
    pub plane: Option<String>,
    /// Attitude control type
    pub att: Option<String>,
    /// Maneuver capability
    pub mvr: Option<String>,
    /// Mission class
    pub class: Option<String>,
    /// Mission category
    pub category: Option<String>,
    /// Mission result/outcome
    pub result: Option<String>,
    /// Control authority
    pub control: Option<String>,
    /// Mission discipline
    pub discipline: Option<String>,
    /// UN registry state
    pub un_state: Option<String>,
    /// UN registry number
    pub un_reg: Option<String>,
    /// UN registered orbital period in minutes
    pub un_period: Option<f64>,
    /// UN registered perigee in km
    pub un_perigee: Option<f64>,
    /// UN registered apogee in km
    pub un_apogee: Option<f64>,
    /// UN registered inclination in degrees
    pub un_inc: Option<f64>,
    /// Disposal orbit epoch
    pub disp_epoch: Option<String>,
    /// Disposal perigee in km
    pub disp_peri: Option<f64>,
    /// Disposal apogee in km
    pub disp_apo: Option<f64>,
    /// Disposal inclination in degrees
    pub disp_inc: Option<f64>,
    /// Comments
    pub comment: Option<String>,
}
