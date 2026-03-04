/*!
 * Filesystem utilities for safe file operations.
 */

use std::fs;
use std::io;
use std::path::Path;

/// Write data to a file atomically using write-to-temp-then-rename.
///
/// Writes `data` to a temporary file in the same directory as `filepath`, calls `sync_all()`
/// to flush to disk, then renames the temp file to the target path. On POSIX systems,
/// `fs::rename` within the same filesystem is atomic, so readers will never see a
/// truncated or partially-written file.
///
/// # Arguments
///
/// * `filepath` - Target file path
/// * `data` - Data to write
///
/// # Returns
///
/// * `Ok(())` if the write succeeded
/// * `Err(io::Error)` on failure (temp file is cleaned up on error)
pub fn atomic_write(filepath: &Path, data: impl AsRef<[u8]>) -> Result<(), io::Error> {
    let parent = filepath.parent().unwrap_or_else(|| Path::new("."));

    // Ensure parent directory exists
    fs::create_dir_all(parent)?;

    // Build temp filename: .{filename}.{pid}.tmp
    let filename = filepath.file_name().unwrap_or_default().to_string_lossy();
    let tmp_name = format!(".{}.{}.tmp", filename, std::process::id());
    let tmp_path = parent.join(&tmp_name);

    // Write to temp file, sync, then rename
    let result = (|| -> Result<(), io::Error> {
        let file = fs::File::create(&tmp_path)?;
        let mut writer = io::BufWriter::new(file);
        io::Write::write_all(&mut writer, data.as_ref())?;
        let file = io::Write::flush(&mut writer)
            .and_then(|_| writer.into_inner().map_err(|e| e.into_error()))?;
        file.sync_all()?;
        drop(file);

        fs::rename(&tmp_path, filepath)?;
        Ok(())
    })();

    // Clean up temp file on error
    if result.is_err() {
        let _ = fs::remove_file(&tmp_path);
    }

    result
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn test_atomic_write_basic() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("test.txt");

        atomic_write(&filepath, b"hello world").unwrap();

        let contents = fs::read_to_string(&filepath).unwrap();
        assert_eq!(contents, "hello world");
    }

    #[test]
    fn test_atomic_write_creates_parent_dirs() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("a").join("b").join("test.txt");

        atomic_write(&filepath, b"nested").unwrap();

        let contents = fs::read_to_string(&filepath).unwrap();
        assert_eq!(contents, "nested");
    }

    #[test]
    fn test_atomic_write_overwrites_existing() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("test.txt");

        atomic_write(&filepath, b"first").unwrap();
        atomic_write(&filepath, b"second").unwrap();

        let contents = fs::read_to_string(&filepath).unwrap();
        assert_eq!(contents, "second");
    }

    #[test]
    fn test_atomic_write_no_temp_file_on_success() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("test.txt");

        atomic_write(&filepath, b"data").unwrap();

        // No .tmp files should remain
        let entries: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(entries.is_empty());
    }
}
