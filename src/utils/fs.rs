/*!
 * Filesystem utilities for safe file operations.
 */

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
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

/// Whether two paths refer to the same location on disk.
///
/// Compares canonical paths when both exist, and falls back to a literal
/// comparison otherwise (for example, before a destination directory has
/// been created).
///
/// # Arguments
/// * `a` - First path
/// * `b` - Second path
///
/// # Returns
/// * `bool`: `true` if the paths resolve to the same location
///
/// # Examples
///
/// ```
/// use std::path::Path;
/// use brahe::utils::fs::same_path;
///
/// assert!(same_path(Path::new("./data/a.txt"), Path::new("./data/a.txt")));
/// assert!(!same_path(Path::new("./data/a.txt"), Path::new("./data/b.txt")));
/// ```
pub fn same_path(a: &Path, b: &Path) -> bool {
    match (fs::canonicalize(a), fs::canonicalize(b)) {
        (Ok(a), Ok(b)) => a == b,
        _ => a == b,
    }
}

/// Staging path used while a file is written beside `target`, so a partial
/// write never appears at `target` itself.
///
/// # Arguments
/// * `target` - Final destination path
///
/// # Returns
/// * `PathBuf`: A `.{name}.{pid}.tmp` sibling of `target`
fn staging_path(target: &Path) -> PathBuf {
    target.with_file_name(format!(
        ".{}.{}.tmp",
        target
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file"),
        std::process::id()
    ))
}

/// Moves a file to a destination path, so `source` no longer exists
/// afterwards.
///
/// Tries [`fs::rename`] first, which is atomic on a single filesystem. When
/// `source` and `target` are on different filesystems, `fs::rename` fails
/// with [`io::ErrorKind::CrossesDevices`]; in that case the file is copied
/// to a staging sibling of `target`, renamed into place, and the source is
/// then removed.
///
/// # Arguments
/// * `source` - File to move; removed on success
/// * `target` - Destination path
///
/// # Returns
/// * `Ok(())`: The file was moved
/// * `Err(BraheError)`: If the rename fails for a reason other than a cross-device move, or the fallback (copy to a sibling staging file, rename into place, remove the source) fails
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use brahe::utils::fs::move_file;
///
/// move_file(Path::new("./cache/data.txt"), Path::new("./out/data.txt")).unwrap();
/// ```
pub fn move_file(source: &Path, target: &Path) -> Result<(), BraheError> {
    match fs::rename(source, target) {
        Ok(()) => Ok(()),
        Err(e) if e.kind() == io::ErrorKind::CrossesDevices => {
            copy_file(source, target)?;
            fs::remove_file(source).map_err(|e| {
                BraheError::IoError(format!(
                    "Failed to remove {} after moving it to {}: {}",
                    source.display(),
                    target.display(),
                    e
                ))
            })
        }
        Err(e) => Err(BraheError::IoError(format!(
            "Failed to move {} to {}: {}",
            source.display(),
            target.display(),
            e
        ))),
    }
}

/// Copies a file to a destination path, leaving `source` in place.
///
/// The copy is written to a staging sibling of `target` and renamed into
/// place, so a partial copy never appears at `target`.
///
/// # Arguments
/// * `source` - File to copy; left in place
/// * `target` - Destination path
///
/// # Returns
/// * `Ok(())`: The file was copied
/// * `Err(BraheError)`: If the copy or the rename into place fails; the staging file is removed
///
/// # Examples
///
/// ```no_run
/// use std::path::Path;
/// use brahe::utils::fs::copy_file;
///
/// copy_file(Path::new("./cache/data.txt"), Path::new("./out/data.txt")).unwrap();
/// ```
pub fn copy_file(source: &Path, target: &Path) -> Result<(), BraheError> {
    let staging = staging_path(target);
    if let Err(e) = fs::copy(source, &staging).and_then(|_| fs::rename(&staging, target)) {
        let _ = fs::remove_file(&staging);
        return Err(BraheError::IoError(format!(
            "Failed to copy {} to {}: {}",
            source.display(),
            target.display(),
            e
        )));
    }
    Ok(())
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
    fn test_same_path_compares_canonical_locations() {
        let dir = tempdir().unwrap();
        let file = dir.path().join("a.txt");
        fs::write(&file, b"data").unwrap();

        assert!(same_path(&file, &file));
        assert!(same_path(&file, &dir.path().join(".").join("a.txt")));
        assert!(!same_path(&file, &dir.path().join("b.txt")));

        // Neither path exists, so the comparison is literal.
        let missing = dir.path().join("missing");
        assert!(same_path(&missing, &missing));
        assert!(!same_path(&missing, &dir.path().join("other")));
    }

    #[test]
    #[parallel]
    fn test_move_file_moves_and_removes_source() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source.txt");
        let target = dir.path().join("nested").join("target.txt");
        fs::create_dir_all(target.parent().unwrap()).unwrap();
        fs::write(&source, b"payload").unwrap();

        move_file(&source, &target).unwrap();

        assert!(!source.exists());
        assert_eq!(fs::read(&target).unwrap(), b"payload");
    }

    #[test]
    #[parallel]
    fn test_move_file_missing_source_errors() {
        let dir = tempdir().unwrap();
        let err = move_file(&dir.path().join("missing.txt"), &dir.path().join("out.txt"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("Failed to move"), "{err}");
    }

    #[test]
    #[parallel]
    fn test_copy_file_keeps_source_and_leaves_no_staging_file() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source.txt");
        let target = dir.path().join("target.txt");
        fs::write(&source, b"payload").unwrap();

        copy_file(&source, &target).unwrap();

        assert_eq!(fs::read(&source).unwrap(), b"payload");
        assert_eq!(fs::read(&target).unwrap(), b"payload");
        let staging: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.file_name().to_string_lossy().ends_with(".tmp"))
            .collect();
        assert!(staging.is_empty());
    }

    #[test]
    #[parallel]
    fn test_copy_file_missing_source_errors() {
        let dir = tempdir().unwrap();
        let target = dir.path().join("target.txt");
        let err = copy_file(&dir.path().join("missing.txt"), &target)
            .unwrap_err()
            .to_string();
        assert!(err.contains("Failed to copy"), "{err}");
        assert!(!target.exists());
    }

    #[test]
    #[parallel]
    fn test_staging_path_is_a_hidden_sibling() {
        let staging = staging_path(Path::new("/tmp/out/data.txt"));
        assert_eq!(staging.parent().unwrap(), Path::new("/tmp/out"));
        let name = staging.file_name().unwrap().to_str().unwrap();
        assert!(name.starts_with(".data.txt."), "{name}");
        assert!(name.ends_with(".tmp"), "{name}");

        // A path with no file name component falls back to the `file` stem.
        let fallback = staging_path(Path::new("/tmp/out/"));
        let name = fallback.file_name().unwrap().to_str().unwrap();
        assert!(name.starts_with(".out."), "{name}");
        let rooted = staging_path(Path::new("/"));
        let name = rooted.file_name().unwrap().to_str().unwrap();
        assert!(name.starts_with(".file."), "{name}");
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
