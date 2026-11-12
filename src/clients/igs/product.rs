/*!
 * IGS products found in an archive listing, filters over them, and options
 * for SP3 range downloads.
 */

use crate::clients::igs::centers::IGSAnalysisCenter;
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
///     gps_week: 2340,
/// };
/// assert_eq!(product.file_name.center().code(), "COD");
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct IGSProduct {
    /// Parsed fields of the file name.
    pub file_name: IGSProductFileName,
    /// File name as published.
    pub name: String,
    /// URL the file is served from.
    pub url: String,
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
/// use brahe::igs::{IGSAnalysisCenter, IGSContentType, IGSProductFilter, IGSSolutionType};
///
/// let filter = IGSProductFilter::new()
///     .with_center(IGSAnalysisCenter::COD)
///     .with_solution(IGSSolutionType::Final)
///     .with_content(IGSContentType::Orbit);
/// assert_eq!(filter.center, Some(IGSAnalysisCenter::COD));
/// ```
#[derive(Debug, Clone, PartialEq, Default)]
pub struct IGSProductFilter {
    /// Analysis center, such as COD or IGS.
    pub center: Option<IGSAnalysisCenter>,
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
    /// * `center` - Analysis center the products must come from
    ///
    /// # Returns
    /// * `IGSProductFilter`: The filter, for chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::{IGSAnalysisCenter, IGSProductFilter};
    ///
    /// let filter = IGSProductFilter::new().with_center(IGSAnalysisCenter::COD);
    /// assert_eq!(filter.center, Some(IGSAnalysisCenter::COD));
    /// ```
    pub fn with_center(mut self, center: IGSAnalysisCenter) -> Self {
        self.center = Some(center);
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
    /// use brahe::igs::{
    ///     IGSAnalysisCenter, IGSProduct, IGSProductFileName, IGSProductFilter, IGSSolutionType,
    /// };
    ///
    /// let name = "COD0OPSFIN_20243150000_01D_05M_ORB.SP3.gz";
    /// let product = IGSProduct {
    ///     file_name: IGSProductFileName::parse(name).unwrap(),
    ///     name: name.to_string(),
    ///     url: format!("http://garner.ucsd.edu/pub/products/2340/{name}"),
    ///     gps_week: 2340,
    /// };
    /// assert!(IGSProductFilter::new().with_center(IGSAnalysisCenter::COD).matches(&product));
    /// assert!(!IGSProductFilter::new().with_solution(IGSSolutionType::Rapid).matches(&product));
    /// ```
    pub fn matches(&self, product: &IGSProduct) -> bool {
        let name = &product.file_name;
        self.center
            .as_ref()
            .is_none_or(|center| center == name.center())
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

/// Options for [`IGSProductsClient::download_sp3_range`](crate::clients::igs::IGSProductsClient::download_sp3_range).
///
/// With `pad` set, one daily file on each side of the requested span is
/// included when the source has it, so a trajectory chained across the files
/// uses centered fits at the span edges. With `sampling` unset, each day uses
/// the coarsest sampling interval the center publishes for it; with
/// `sampling` set, only files at that interval are used. Short names, used
/// before GPS week 2238, do not encode a sampling interval, so `sampling`
/// applies only to long-name files.
///
/// # Examples
///
/// ```
/// use brahe::igs::IGSRangeOptions;
///
/// let options = IGSRangeOptions::new();
/// assert!(options.pad);
/// assert_eq!(options.sampling, None);
///
/// let options = IGSRangeOptions::new().with_pad(false).with_sampling(300.0);
/// assert!(!options.pad);
/// assert_eq!(options.sampling, Some(300.0));
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct IGSRangeOptions {
    /// Include one daily file on each side of the span when it exists.
    pub pad: bool,
    /// Sampling interval of the files to use, in seconds, or `None` for the coarsest published.
    pub sampling: Option<f64>,
}

impl Default for IGSRangeOptions {
    fn default() -> Self {
        Self {
            pad: true,
            sampling: None,
        }
    }
}

impl IGSRangeOptions {
    /// Creates options with padding on and the coarsest published sampling.
    ///
    /// # Returns
    /// * `IGSRangeOptions`: Options with `pad` true and `sampling` `None`
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSRangeOptions;
    ///
    /// assert_eq!(IGSRangeOptions::new(), IGSRangeOptions::default());
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets whether one daily file is added on each side of the span.
    ///
    /// # Arguments
    /// * `pad` - Include the neighboring daily files when they exist
    ///
    /// # Returns
    /// * `IGSRangeOptions`: The options, for chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSRangeOptions;
    ///
    /// assert!(!IGSRangeOptions::new().with_pad(false).pad);
    /// ```
    pub fn with_pad(mut self, pad: bool) -> Self {
        self.pad = pad;
        self
    }

    /// Selects the files published at one sampling interval.
    ///
    /// # Arguments
    /// * `sampling` - Sampling interval in seconds, for example `300.0` for `05M` files
    ///
    /// # Returns
    /// * `IGSRangeOptions`: The options, for chaining
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::igs::IGSRangeOptions;
    ///
    /// assert_eq!(IGSRangeOptions::new().with_sampling(900.0).sampling, Some(900.0));
    /// ```
    pub fn with_sampling(mut self, sampling: f64) -> Self {
        self.sampling = Some(sampling);
        self
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
            let filter = IGSProductFilter::new().with_center(IGSAnalysisCenter::parse(center));
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
            .with_center(IGSAnalysisCenter::ESA)
            .with_solution(IGSSolutionType::Rapid)
            .with_content(IGSContentType::Clock)
            .with_format(IGSFileFormat::CLK);
        assert_eq!(
            filter,
            IGSProductFilter {
                center: Some(IGSAnalysisCenter::ESA),
                solution: Some(IGSSolutionType::Rapid),
                content: Some(IGSContentType::Clock),
                format: Some(IGSFileFormat::CLK),
            }
        );
    }

    #[test]
    #[parallel]
    fn test_igs_range_options_defaults_and_builders() {
        let options = IGSRangeOptions::new();
        assert_eq!(options, IGSRangeOptions::default());
        assert!(options.pad);
        assert_eq!(options.sampling, None);

        let options = IGSRangeOptions::new().with_pad(false).with_sampling(300.0);
        assert_eq!(
            options,
            IGSRangeOptions {
                pad: false,
                sampling: Some(300.0),
            }
        );
    }
}
