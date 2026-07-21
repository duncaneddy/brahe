/*!
 * Handle to a generated/cached Horizons SPK kernel.
 */

use std::path::{Path, PathBuf};

use crate::spice::load_spice_kernel;
use crate::utils::BraheError;

/// A handle to a generated (or cached) Horizons SPK kernel on disk.
///
/// Provides the cached file path, the raw kernel bytes, or a direct load into
/// the SPICE registry.
#[derive(Debug, Clone)]
pub struct HorizonsSPKResponse {
    path: PathBuf,
    spk_file_id: Option<String>,
}

impl HorizonsSPKResponse {
    /// Construct a response wrapping a cached SPK path.
    #[cfg_attr(not(test), allow(dead_code))]
    pub(crate) fn new(path: PathBuf, spk_file_id: Option<String>) -> Self {
        HorizonsSPKResponse { path, spk_file_id }
    }

    /// Path to the cached `.bsp` kernel.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// The Horizons `spk_file_id`, if the server returned one.
    pub fn spk_file_id(&self) -> Option<&str> {
        self.spk_file_id.as_deref()
    }

    /// Read the raw SPK bytes from the cached file.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<u8>)` - The kernel file contents.
    /// * `Err(BraheError)` - If the file cannot be read.
    pub fn bytes(&self) -> Result<Vec<u8>, BraheError> {
        std::fs::read(&self.path).map_err(|e| {
            BraheError::IoError(format!("Failed to read SPK {}: {}", self.path.display(), e))
        })
    }

    /// Load the cached SPK into the global SPICE registry.
    ///
    /// # Returns
    ///
    /// * `Ok(())` - On successful load.
    /// * `Err(BraheError)` - If the kernel cannot be loaded.
    pub fn load(&self) -> Result<(), BraheError> {
        let path = self.path.to_str().ok_or_else(|| {
            BraheError::IoError(format!(
                "SPK path is not valid UTF-8: {}",
                self.path.display()
            ))
        })?;
        load_spice_kernel(path)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use serial_test::serial;

    #[test]
    #[serial]
    fn test_path_and_id_getters() {
        let resp = HorizonsSPKResponse::new(PathBuf::from("/tmp/x.bsp"), Some("2000001".into()));
        assert_eq!(resp.path(), Path::new("/tmp/x.bsp"));
        assert_eq!(resp.spk_file_id(), Some("2000001"));
    }

    #[test]
    #[serial]
    fn test_bytes_reads_file() {
        // test_assets/de440s.bsp ships with the repo as a real SPK.
        let resp = HorizonsSPKResponse::new(PathBuf::from("test_assets/de440s.bsp"), None);
        let bytes = resp.bytes().unwrap();
        assert!(bytes.len() > 100);
        assert_eq!(&bytes[0..7], b"DAF/SPK");
    }

    #[test]
    #[serial]
    fn test_load_into_registry() {
        use crate::spice::clear_spice_kernels;
        let resp = HorizonsSPKResponse::new(PathBuf::from("test_assets/de440s.bsp"), None);
        resp.load().unwrap();
        clear_spice_kernels(); // returns (), not Result — do not .unwrap()
    }

    #[test]
    #[serial]
    fn test_bytes_missing_file_errors() {
        let resp = HorizonsSPKResponse::new(PathBuf::from("does/not/exist.bsp"), None);
        assert!(resp.bytes().is_err());
    }
}
