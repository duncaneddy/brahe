/*!
 * Filesystem utilities for safe file operations.
 */

use std::fs;
use std::io;
use std::path::Path;
use std::time::SystemTime;

use crate::time::Epoch;
use crate::utils::BraheError;

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

/// Sets a file's modification time to now, so a `304 Not Modified` answer
/// restarts a cache's freshness window without rewriting its content.
///
/// # Arguments
/// * `path` - The file to touch
///
/// # Returns
/// * `Ok(())`: The modification time was updated
/// * `Err(BraheError)`: If the file cannot be opened or its time set
pub(crate) fn touch(path: &Path) -> Result<(), BraheError> {
    fs::OpenOptions::new()
        .write(true)
        .open(path)
        .and_then(|f| f.set_modified(SystemTime::now()))
        .map_err(|e| BraheError::IoError(format!("Failed to update {}: {e}", path.display())))
}

/// A file's modification time as an [`Epoch`], for use as a cache's
/// `retrieved` timestamp when no sidecar value is available.
///
/// # Arguments
/// * `path` - The file whose modification time is read
///
/// # Returns
/// * `Ok(Epoch)`: The modification time, in UTC
/// * `Err(BraheError)`: If the file's modification time cannot be read
pub(crate) fn modified_epoch(path: &Path) -> Result<Epoch, BraheError> {
    let modified = fs::metadata(path)
        .and_then(|m| m.modified())
        .map_err(|e| BraheError::IoError(format!("Failed to read file modification time: {e}")))?;
    let secs = modified
        .duration_since(SystemTime::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64();
    Ok(Epoch::from_unix_timestamp(secs))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use serial_test::parallel;
    use tempfile::tempdir;

    #[test]
    #[parallel]
    fn test_atomic_write_basic() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("test.txt");

        atomic_write(&filepath, b"hello world").unwrap();

        let contents = fs::read_to_string(&filepath).unwrap();
        assert_eq!(contents, "hello world");
    }

    #[test]
    #[parallel]
    fn test_atomic_write_creates_parent_dirs() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("a").join("b").join("test.txt");

        atomic_write(&filepath, b"nested").unwrap();

        let contents = fs::read_to_string(&filepath).unwrap();
        assert_eq!(contents, "nested");
    }

    #[test]
    #[parallel]
    fn test_atomic_write_overwrites_existing() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("test.txt");

        atomic_write(&filepath, b"first").unwrap();
        atomic_write(&filepath, b"second").unwrap();

        let contents = fs::read_to_string(&filepath).unwrap();
        assert_eq!(contents, "second");
    }

    #[test]
    #[parallel]
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

    #[test]
    #[parallel]
    fn test_touch_updates_modification_time() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("test.txt");
        fs::write(&filepath, b"data").unwrap();

        let file = fs::OpenOptions::new().write(true).open(&filepath).unwrap();
        file.set_modified(SystemTime::now() - std::time::Duration::from_secs(3600))
            .unwrap();

        touch(&filepath).unwrap();

        let modified = fs::metadata(&filepath).unwrap().modified().unwrap();
        assert!(
            SystemTime::now()
                .duration_since(modified)
                .unwrap()
                .as_secs()
                < 5
        );
    }

    #[test]
    #[parallel]
    fn test_touch_missing_file_errors() {
        let dir = tempdir().unwrap();
        assert!(touch(&dir.path().join("missing.txt")).is_err());
    }

    #[test]
    #[parallel]
    fn test_modified_epoch_reads_modification_time() {
        let dir = tempdir().unwrap();
        let filepath = dir.path().join("test.txt");
        fs::write(&filepath, b"data").unwrap();

        let target = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000);
        let file = fs::OpenOptions::new().write(true).open(&filepath).unwrap();
        file.set_modified(target).unwrap();

        let epoch = modified_epoch(&filepath).unwrap();
        assert_eq!(epoch, Epoch::from_unix_timestamp(1_700_000_000.0));
    }

    #[test]
    #[parallel]
    fn test_modified_epoch_missing_file_errors() {
        let dir = tempdir().unwrap();
        assert!(modified_epoch(&dir.path().join("missing.txt")).is_err());
    }
}
