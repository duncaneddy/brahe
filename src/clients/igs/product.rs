/*!
 * IGS products found in an archive listing, and filters over them.
 */

use crate::clients::igs::file_name::{
    IGSContentType, IGSFileFormat, IGSProductFileName, IGSSolutionType,
};

/// A product file listed in an IGS archive's GPS week directory.
///
/// # Examples
///
/// ```
/// use brahe::igs::{IGSProduct, IGSProductFileName};
///
/// let name = "COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz";
/// let product = IGSProduct {
///     file_name: IGSProductFileName::parse(name).unwrap(),
///     name: name.to_string(),
///     url: format!("http://garner.ucsd.edu/pub/products/2340/{name}"),
///     size: None,
///     gps_week: 2340,
/// };
/// assert_eq!(product.file_name.center(), "COD");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct IGSProduct {
    /// Parsed fields of the file name.
    pub file_name: IGSProductFileName,
    /// File name as published.
    pub name: String,
    /// URL the file is served from.
    pub url: String,
    /// File size in bytes, when the listing gives it.
    pub size: Option<u64>,
    /// GPS week of the archive directory the file is listed in.
    pub gps_week: u32,
}

/// Selects IGS products by analysis center, solution, content, and format.
///
/// Each criterion left as `None` matches every product. The center is
/// compared without regard to case, so `"cod"` and `"COD"` select the same
/// products.
///
/// # Examples
///
/// ```
/// use brahe::igs::{IGSContentType, IGSProductFilter, IGSSolutionType};
///
/// let filter = IGSProductFilter::new()
///     .with_center("cod")
///     .with_solution(IGSSolutionType::Final)
///     .with_content(IGSContentType::Orbit);
/// assert_eq!(filter.center.as_deref(), Some("cod"));
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IGSProductFilter {
    /// Analysis center code, such as `COD`, compared without regard to case.
    pub center: Option<String>,
    /// Solution type, such as Final or Rapid.
    pub solution: Option<IGSSolutionType>,
    /// Content type, such as Orbit or Clock.
    pub content: Option<IGSContentType>,
    /// File format, such as SP3.
    pub format: Option<IGSFileFormat>,
}

impl IGSProductFilter {
    /// Creates a filter that matches every product.
    ///
    /// # Returns
    /// * `IGSProductFilter`: Filter with no criteria set
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSProductFilter;
    ///
    /// assert_eq!(IGSProductFilter::new(), IGSProductFilter::default());
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Restricts the filter to one analysis center.
    ///
    /// # Arguments
    /// * `center` - Three-character analysis center code, in any case
    ///
    /// # Returns
    /// * `IGSProductFilter`: The filter, for chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSProductFilter;
    ///
    /// let filter = IGSProductFilter::new().with_center("COD");
    /// assert_eq!(filter.center.as_deref(), Some("COD"));
    /// ```
    pub fn with_center(mut self, center: &str) -> Self {
        self.center = Some(center.to_string());
        self
    }

    /// Restricts the filter to one solution type.
    ///
    /// # Arguments
    /// * `solution` - Solution type to keep
    ///
    /// # Returns
    /// * `IGSProductFilter`: The filter, for chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::{IGSProductFilter, IGSSolutionType};
    ///
    /// let filter = IGSProductFilter::new().with_solution(IGSSolutionType::Rapid);
    /// assert_eq!(filter.solution, Some(IGSSolutionType::Rapid));
    /// ```
    pub fn with_solution(mut self, solution: IGSSolutionType) -> Self {
        self.solution = Some(solution);
        self
    }

    /// Restricts the filter to one content type.
    ///
    /// # Arguments
    /// * `content` - Content type to keep
    ///
    /// # Returns
    /// * `IGSProductFilter`: The filter, for chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::{IGSContentType, IGSProductFilter};
    ///
    /// let filter = IGSProductFilter::new().with_content(IGSContentType::Clock);
    /// assert_eq!(filter.content, Some(IGSContentType::Clock));
    /// ```
    pub fn with_content(mut self, content: IGSContentType) -> Self {
        self.content = Some(content);
        self
    }

    /// Restricts the filter to one file format.
    ///
    /// # Arguments
    /// * `format` - File format to keep
    ///
    /// # Returns
    /// * `IGSProductFilter`: The filter, for chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::{IGSFileFormat, IGSProductFilter};
    ///
    /// let filter = IGSProductFilter::new().with_format(IGSFileFormat::SP3);
    /// assert_eq!(filter.format, Some(IGSFileFormat::SP3));
    /// ```
    pub fn with_format(mut self, format: IGSFileFormat) -> Self {
        self.format = Some(format);
        self
    }

    /// Whether a product meets every criterion of the filter.
    ///
    /// # Arguments
    /// * `product` - Product to test
    ///
    /// # Returns
    /// * `bool`: `true` when every criterion that is set matches the product's file name
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::{IGSProduct, IGSProductFileName, IGSProductFilter, IGSSolutionType};
    ///
    /// let name = "COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz";
    /// let product = IGSProduct {
    ///     file_name: IGSProductFileName::parse(name).unwrap(),
    ///     name: name.to_string(),
    ///     url: format!("http://garner.ucsd.edu/pub/products/2340/{name}"),
    ///     size: None,
    ///     gps_week: 2340,
    /// };
    /// assert!(IGSProductFilter::new().with_center("cod").matches(&product));
    /// assert!(!IGSProductFilter::new().with_solution(IGSSolutionType::Rapid).matches(&product));
    /// ```
    pub fn matches(&self, product: &IGSProduct) -> bool {
        let name = &product.file_name;
        self.center
            .as_deref()
            .is_none_or(|center| center.eq_ignore_ascii_case(name.center()))
            && self
                .solution
                .as_ref()
                .is_none_or(|solution| solution == name.solution())
            && self
                .content
                .as_ref()
                .is_none_or(|content| content == name.content())
            && self
                .format
                .as_ref()
                .is_none_or(|format| format == name.format())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use serial_test::parallel;

    fn product(name: &str) -> IGSProduct {
        IGSProduct {
            file_name: IGSProductFileName::parse(name).unwrap(),
            name: name.to_string(),
            url: format!("http://localhost/2340/{name}"),
            size: None,
            gps_week: 2340,
        }
    }

    #[test]
    #[parallel]
    fn test_igs_product_filter_new_matches_everything() {
        let filter = IGSProductFilter::new();
        assert_eq!(filter, IGSProductFilter::default());
        assert!(filter.matches(&product("COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz")));
        assert!(filter.matches(&product("igr23400.sp3.Z")));
    }

    #[test]
    #[parallel]
    fn test_igs_product_filter_center_is_case_insensitive() {
        let cod = product("COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz");
        let esa = product("ESA0OPSFIN_20243150000_01D_05M_ORB.SP3.gz");
        for center in ["cod", "COD", "Cod"] {
            let filter = IGSProductFilter::new().with_center(center);
            assert!(filter.matches(&cod), "{center}");
            assert!(!filter.matches(&esa), "{center}");
        }
    }

    #[test]
    #[parallel]
    fn test_igs_product_filter_solution_content_format() {
        let orbit = product("COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz");
        let clock = product("COD0OPSFIN_20243150000_01D_30S_CLK.CLK.gz");
        let rapid = product("COD0OPSRAP_20243150000_01D_05M_ORB.SP3.gz");

        let filter = IGSProductFilter::new()
            .with_solution(IGSSolutionType::Final)
            .with_content(IGSContentType::Orbit);
        assert!(filter.matches(&orbit));
        assert!(!filter.matches(&clock));
        assert!(!filter.matches(&rapid));

        let filter = IGSProductFilter::new().with_format(IGSFileFormat::CLK);
        assert!(filter.matches(&clock));
        assert!(!filter.matches(&orbit));
    }

    #[test]
    #[parallel]
    fn test_igs_product_filter_builders_set_fields() {
        let filter = IGSProductFilter::new()
            .with_center("esa")
            .with_solution(IGSSolutionType::Rapid)
            .with_content(IGSContentType::Clock)
            .with_format(IGSFileFormat::CLK);
        assert_eq!(
            filter,
            IGSProductFilter {
                center: Some("esa".to_string()),
                solution: Some(IGSSolutionType::Rapid),
                content: Some(IGSContentType::Clock),
                format: Some(IGSFileFormat::CLK),
            }
        );
    }
}
