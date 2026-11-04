/*!
 * IGS product archive sources.
 */

use std::fmt;

/// Public archive that serves IGS products.
///
/// CDDIS holds the full archive and requires NASA Earthdata credentials.
/// Garner (the Scripps Institution of Oceanography mirror) holds the full
/// archive from GPS week 576 and is served over unencrypted HTTP. BKG holds
/// GPS weeks 2238 and later, and only the IGS combination and CAS products.
///
/// # Examples
///
/// ```
/// use brahe::igs::IGSProductSource;
///
/// let source = IGSProductSource::BKG;
/// assert_eq!(source.default_base_url(), "https://igs.bkg.bund.de/root_ftp/IGS/products/");
/// assert_eq!(source.to_string(), "BKG");
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(clippy::upper_case_acronyms)]
pub enum IGSProductSource {
    /// NASA Crustal Dynamics Data Information System; requires Earthdata credentials.
    CDDIS,
    /// Scripps Institution of Oceanography Garner archive, served over plain HTTP.
    Garner,
    /// German Federal Agency for Cartography and Geodesy archive.
    BKG,
}

impl IGSProductSource {
    /// Base URL of the source's product directory, with a trailing slash.
    ///
    /// # Returns
    /// * `&'static str`: Directory URL that contains one subdirectory per GPS week
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSProductSource;
    ///
    /// assert_eq!(
    ///     IGSProductSource::CDDIS.default_base_url(),
    ///     "https://cddis.nasa.gov/archive/gnss/products/"
    /// );
    /// assert_eq!(
    ///     IGSProductSource::Garner.default_base_url(),
    ///     "http://garner.ucsd.edu/pub/products/"
    /// );
    /// ```
    pub fn default_base_url(&self) -> &'static str {
        match self {
            IGSProductSource::CDDIS => "https://cddis.nasa.gov/archive/gnss/products/",
            IGSProductSource::Garner => "http://garner.ucsd.edu/pub/products/",
            IGSProductSource::BKG => "https://igs.bkg.bund.de/root_ftp/IGS/products/",
        }
    }
}

impl fmt::Display for IGSProductSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            IGSProductSource::CDDIS => write!(f, "CDDIS"),
            IGSProductSource::Garner => write!(f, "Garner"),
            IGSProductSource::BKG => write!(f, "BKG"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serial_test::parallel;

    #[test]
    #[parallel]
    fn test_igs_product_source_default_base_url() {
        assert_eq!(
            IGSProductSource::CDDIS.default_base_url(),
            "https://cddis.nasa.gov/archive/gnss/products/"
        );
        assert_eq!(
            IGSProductSource::Garner.default_base_url(),
            "http://garner.ucsd.edu/pub/products/"
        );
        assert_eq!(
            IGSProductSource::BKG.default_base_url(),
            "https://igs.bkg.bund.de/root_ftp/IGS/products/"
        );
    }

    #[test]
    #[parallel]
    fn test_igs_product_source_display() {
        assert_eq!(IGSProductSource::CDDIS.to_string(), "CDDIS");
        assert_eq!(IGSProductSource::Garner.to_string(), "Garner");
        assert_eq!(IGSProductSource::BKG.to_string(), "BKG");
    }
}
