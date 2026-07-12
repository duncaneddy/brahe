/*!
 * Enumeration of the NAIF kernel files brahe can download, plus a
 * kernel-or-path source type used to load them.
 */

/// Base URL for NAIF generic planetary DE SPK kernels.
const NAIF_PLANETS_BASE_URL: &str =
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/";

/// Base URL for NAIF generic satellite ephemeris kernels.
const NAIF_SATELLITES_BASE_URL: &str =
    "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/";

/// Base URL for NAIF generic binary PCK kernels.
const NAIF_PCK_BASE_URL: &str = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/";

/// A NAIF generic kernel that brahe knows how to download and cache.
///
/// Every variant maps to a concrete file in NAIF's public archive. The
/// planetary DE variants and satellite ephemeris variants are SPK
/// (ephemeris) kernels; [`SPICEKernel::MoonPaDe440`] is a binary PCK
/// (orientation) kernel. Use [`SPICEKernel::name`] for the short name used
/// as a registry key, [`SPICEKernel::filename`] for the on-disk/cache file
/// name, and [`SPICEKernel::url`] for the full download URL.
///
/// This is distinct from `EphemerisSource`, which is a force-model concept.
/// `SPICEKernel` is purely "which kernel file to open".
#[allow(clippy::upper_case_acronyms)]
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum SPICEKernel {
    // Planetary DE SPK kernels (naif .../spk/planets/)
    /// JPL DE430 planetary ephemeris.
    DE430,
    /// JPL DE432s planetary ephemeris (small).
    DE432s,
    /// JPL DE435 planetary ephemeris.
    DE435,
    /// JPL DE438 planetary ephemeris.
    DE438,
    /// JPL DE440 planetary ephemeris (1550-2650 CE).
    DE440,
    /// JPL DE440s planetary ephemeris (small; 1849-2150 CE).
    DE440s,
    /// JPL DE442 planetary ephemeris.
    DE442,
    /// JPL DE442s planetary ephemeris (small).
    DE442s,

    // Satellite ephemeris kernels (naif .../spk/satellites/)
    /// Mars satellite ephemeris (mar099).
    Mar099,
    /// Mars satellite ephemeris (mar099s, small).
    Mar099s,
    /// Jupiter satellite ephemeris (jup365).
    Jup365,
    /// Saturn satellite ephemeris (sat441).
    Sat441,
    /// Uranus satellite ephemeris (ura184).
    Ura184,
    /// Neptune satellite ephemeris (nep097).
    Nep097,
    /// Pluto-system ephemeris (plu060).
    Plu060,

    // Binary PCK kernels (naif .../pck/)
    /// Lunar principal-axes binary PCK (moon_pa_de440).
    MoonPaDe440,
}

impl SPICEKernel {
    /// Short kernel name, used as the registry key and NAIF download-cache
    /// identifier.
    ///
    /// # Returns
    /// - `&'static str`: e.g. `"de440s"`, `"mar099s"`, `"moon_pa_de440"`
    pub fn name(&self) -> &'static str {
        match self {
            SPICEKernel::DE430 => "de430",
            SPICEKernel::DE432s => "de432s",
            SPICEKernel::DE435 => "de435",
            SPICEKernel::DE438 => "de438",
            SPICEKernel::DE440 => "de440",
            SPICEKernel::DE440s => "de440s",
            SPICEKernel::DE442 => "de442",
            SPICEKernel::DE442s => "de442s",
            SPICEKernel::Mar099 => "mar099",
            SPICEKernel::Mar099s => "mar099s",
            SPICEKernel::Jup365 => "jup365",
            SPICEKernel::Sat441 => "sat441",
            SPICEKernel::Ura184 => "ura184",
            SPICEKernel::Nep097 => "nep097",
            SPICEKernel::Plu060 => "plu060",
            SPICEKernel::MoonPaDe440 => "moon_pa_de440",
        }
    }

    /// On-disk file name for this kernel in NAIF's archive and the local
    /// cache.
    ///
    /// Most kernels are `<name>.bsp`, but the file name is not always
    /// derivable from the name: [`SPICEKernel::Ura184`] is published as
    /// `ura184_part-3.bsp` and [`SPICEKernel::MoonPaDe440`] as
    /// `moon_pa_de440_200625.bpc`.
    ///
    /// # Returns
    /// - `&'static str`: e.g. `"de440s.bsp"`, `"ura184_part-3.bsp"`,
    ///   `"moon_pa_de440_200625.bpc"`
    pub fn filename(&self) -> &'static str {
        match self {
            SPICEKernel::DE430 => "de430.bsp",
            SPICEKernel::DE432s => "de432s.bsp",
            SPICEKernel::DE435 => "de435.bsp",
            SPICEKernel::DE438 => "de438.bsp",
            SPICEKernel::DE440 => "de440.bsp",
            SPICEKernel::DE440s => "de440s.bsp",
            SPICEKernel::DE442 => "de442.bsp",
            SPICEKernel::DE442s => "de442s.bsp",
            SPICEKernel::Mar099 => "mar099.bsp",
            SPICEKernel::Mar099s => "mar099s.bsp",
            SPICEKernel::Jup365 => "jup365.bsp",
            SPICEKernel::Sat441 => "sat441.bsp",
            SPICEKernel::Ura184 => "ura184_part-3.bsp",
            SPICEKernel::Nep097 => "nep097.bsp",
            SPICEKernel::Plu060 => "plu060.bsp",
            SPICEKernel::MoonPaDe440 => "moon_pa_de440_200625.bpc",
        }
    }

    /// Whether this kernel is a binary PCK (orientation) kernel rather than
    /// an SPK (ephemeris) kernel.
    ///
    /// # Returns
    /// - `bool`: `true` only for [`SPICEKernel::MoonPaDe440`]
    pub fn is_pck(&self) -> bool {
        matches!(self, SPICEKernel::MoonPaDe440)
    }

    /// Full download URL for this kernel in NAIF's public archive.
    ///
    /// # Returns
    /// - `String`: the base URL for the kernel's category joined with its
    ///   [`filename`](SPICEKernel::filename)
    pub(crate) fn url(&self) -> String {
        let base = match self {
            SPICEKernel::DE430
            | SPICEKernel::DE432s
            | SPICEKernel::DE435
            | SPICEKernel::DE438
            | SPICEKernel::DE440
            | SPICEKernel::DE440s
            | SPICEKernel::DE442
            | SPICEKernel::DE442s => NAIF_PLANETS_BASE_URL,
            SPICEKernel::Mar099
            | SPICEKernel::Mar099s
            | SPICEKernel::Jup365
            | SPICEKernel::Sat441
            | SPICEKernel::Ura184
            | SPICEKernel::Nep097
            | SPICEKernel::Plu060 => NAIF_SATELLITES_BASE_URL,
            SPICEKernel::MoonPaDe440 => NAIF_PCK_BASE_URL,
        };
        format!("{}{}", base, self.filename())
    }

    /// Resolve a short kernel name (as returned by [`SPICEKernel::name`])
    /// back to its variant.
    ///
    /// # Arguments
    /// - `name`: A short kernel name, e.g. `"de440s"` or `"moon_pa_de440"`
    ///
    /// # Returns
    /// - `Some(SPICEKernel)` if `name` matches a known kernel, else `None`
    pub fn from_name(name: &str) -> Option<SPICEKernel> {
        SPICEKernel::all()
            .iter()
            .copied()
            .find(|k| k.name() == name)
    }

    /// Every known kernel variant, for iterating over all downloadable
    /// kernels or documentation.
    ///
    /// Not used by `load_all_kernels`, which loads a curated subset
    /// (one DE ephemeris, not every DE version) rather than every variant
    /// here.
    ///
    /// # Returns
    /// - `&'static [SPICEKernel]`: all variants
    pub fn all() -> &'static [SPICEKernel] {
        &[
            SPICEKernel::DE430,
            SPICEKernel::DE432s,
            SPICEKernel::DE435,
            SPICEKernel::DE438,
            SPICEKernel::DE440,
            SPICEKernel::DE440s,
            SPICEKernel::DE442,
            SPICEKernel::DE442s,
            SPICEKernel::Mar099,
            SPICEKernel::Mar099s,
            SPICEKernel::Jup365,
            SPICEKernel::Sat441,
            SPICEKernel::Ura184,
            SPICEKernel::Nep097,
            SPICEKernel::Plu060,
            SPICEKernel::MoonPaDe440,
        ]
    }
}

/// A kernel to load: a known [`SPICEKernel`] (downloaded and cached on
/// demand) or a local file path (bring-your-own kernel).
///
/// [`From<&str>`](KernelSource::from) resolves the string to a
/// [`SPICEKernel`] if it matches a known kernel name and otherwise treats it
/// as a file path, so `load_kernel("de440s")` and
/// `load_kernel("/path/to/custom.bsp")` both work.
#[derive(Debug, Clone)]
pub enum KernelSource {
    /// A known NAIF kernel, downloaded and cached on demand.
    Kernel(SPICEKernel),
    /// A local file path to a `.bsp`/`.bpc` kernel.
    Path(String),
}

impl From<SPICEKernel> for KernelSource {
    fn from(kernel: SPICEKernel) -> Self {
        KernelSource::Kernel(kernel)
    }
}

impl From<&str> for KernelSource {
    fn from(name_or_path: &str) -> Self {
        match SPICEKernel::from_name(name_or_path) {
            Some(kernel) => KernelSource::Kernel(kernel),
            None => KernelSource::Path(name_or_path.to_string()),
        }
    }
}

impl From<String> for KernelSource {
    fn from(name_or_path: String) -> Self {
        match SPICEKernel::from_name(&name_or_path) {
            Some(kernel) => KernelSource::Kernel(kernel),
            None => KernelSource::Path(name_or_path),
        }
    }
}

impl KernelSource {
    /// Registry key for this source: the kernel name for a known kernel, or
    /// the path string for a bring-your-own kernel.
    ///
    /// # Returns
    /// - `&str`: the kernel name or the path string
    pub fn key(&self) -> &str {
        match self {
            KernelSource::Kernel(kernel) => kernel.name(),
            KernelSource::Path(path) => path.as_str(),
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn test_naif_kernel_names_roundtrip() {
        for k in SPICEKernel::all() {
            assert_eq!(SPICEKernel::from_name(k.name()), Some(*k));
        }
        assert_eq!(SPICEKernel::from_name("nonexistent"), None);
    }

    #[test]
    fn test_naif_kernel_filenames() {
        assert_eq!(SPICEKernel::DE440s.filename(), "de440s.bsp");
        assert_eq!(SPICEKernel::Ura184.filename(), "ura184_part-3.bsp");
        assert_eq!(
            SPICEKernel::MoonPaDe440.filename(),
            "moon_pa_de440_200625.bpc"
        );
        assert!(SPICEKernel::MoonPaDe440.is_pck());
        assert!(!SPICEKernel::Sat441.is_pck());
    }

    #[test]
    fn test_kernel_source_from_str() {
        assert!(matches!(
            KernelSource::from("de440s"),
            KernelSource::Kernel(SPICEKernel::DE440s)
        ));
        assert!(matches!(
            KernelSource::from("/tmp/custom.bsp"),
            KernelSource::Path(_)
        ));
        assert_eq!(KernelSource::from("jup365").key(), "jup365");
    }

    #[test]
    fn test_naif_kernel_filename_exhaustive() {
        // Every variant's on-disk file name, covering all arms of `filename`.
        let cases = [
            (SPICEKernel::DE430, "de430.bsp"),
            (SPICEKernel::DE432s, "de432s.bsp"),
            (SPICEKernel::DE435, "de435.bsp"),
            (SPICEKernel::DE438, "de438.bsp"),
            (SPICEKernel::DE440, "de440.bsp"),
            (SPICEKernel::DE440s, "de440s.bsp"),
            (SPICEKernel::DE442, "de442.bsp"),
            (SPICEKernel::DE442s, "de442s.bsp"),
            (SPICEKernel::Mar099, "mar099.bsp"),
            (SPICEKernel::Mar099s, "mar099s.bsp"),
            (SPICEKernel::Jup365, "jup365.bsp"),
            (SPICEKernel::Sat441, "sat441.bsp"),
            (SPICEKernel::Ura184, "ura184_part-3.bsp"),
            (SPICEKernel::Nep097, "nep097.bsp"),
            (SPICEKernel::Plu060, "plu060.bsp"),
            (SPICEKernel::MoonPaDe440, "moon_pa_de440_200625.bpc"),
        ];
        for (kernel, filename) in cases {
            assert_eq!(kernel.filename(), filename);
        }
    }

    #[test]
    fn test_naif_kernel_url_by_category() {
        // Planetary DE kernels use the planets base URL.
        assert_eq!(
            SPICEKernel::DE440s.url(),
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440s.bsp"
        );
        for k in [
            SPICEKernel::DE430,
            SPICEKernel::DE432s,
            SPICEKernel::DE435,
            SPICEKernel::DE438,
            SPICEKernel::DE440,
            SPICEKernel::DE442,
            SPICEKernel::DE442s,
        ] {
            assert!(k.url().contains("/spk/planets/"), "{}", k.url());
        }

        // Satellite ephemeris kernels use the satellites base URL (and the
        // filename override for Ura184).
        assert_eq!(
            SPICEKernel::Ura184.url(),
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/ura184_part-3.bsp"
        );
        for k in [
            SPICEKernel::Mar099,
            SPICEKernel::Mar099s,
            SPICEKernel::Jup365,
            SPICEKernel::Sat441,
            SPICEKernel::Nep097,
            SPICEKernel::Plu060,
        ] {
            assert!(k.url().contains("/spk/satellites/"), "{}", k.url());
        }

        // The binary PCK kernel uses the pck base URL.
        assert_eq!(
            SPICEKernel::MoonPaDe440.url(),
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/pck/moon_pa_de440_200625.bpc"
        );
    }

    #[test]
    fn test_kernel_source_from_string_and_kernel() {
        // From<String>: known name -> Kernel, unknown -> Path.
        assert!(matches!(
            KernelSource::from("de440s".to_string()),
            KernelSource::Kernel(SPICEKernel::DE440s)
        ));
        assert!(matches!(
            KernelSource::from("/tmp/custom.bsp".to_string()),
            KernelSource::Path(_)
        ));
        // From<SPICEKernel>.
        assert!(matches!(
            KernelSource::from(SPICEKernel::Jup365),
            KernelSource::Kernel(SPICEKernel::Jup365)
        ));
        // key() returns the path string for a bring-your-own source.
        assert_eq!(
            KernelSource::from("/tmp/custom.bsp".to_string()).key(),
            "/tmp/custom.bsp"
        );
    }
}
