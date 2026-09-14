/*!
 * Filesystem utilities for safe file operations.
 */

use std::ffi::OsString;
use std::fs;
use std::io;
use std::path::{Component, Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
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

    let tmp_path = staging_path(filepath);

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

/// Characters that turn a value into a path rather than a single component
/// of one.
const FORBIDDEN_PATH_COMPONENT_CHARACTERS: [char; 3] = ['/', '\\', '\0'];

/// Validates that a value names no path of its own, so that a path assembled
/// from it always addresses an entry inside a single directory.
///
/// # Arguments
/// * `field` - Name of the value being validated, used in the error message
/// * `value` - Value to validate, which may be empty
///
/// # Returns
/// * `Ok(())`: The value is safe to use as a single path component
/// * `Err(BraheError)`: If the value contains `/`, `\` or NUL, or is `.` or `..`
///
/// # Examples
///
/// ```
/// use brahe::utils::fs::validate_path_component;
///
/// assert!(validate_path_component("object_name", "STARLINK-38128").is_ok());
/// assert!(validate_path_component("object_name", "../evil").is_err());
/// assert!(validate_path_component("object_name", "..").is_err());
/// ```
pub fn validate_path_component(field: &str, value: &str) -> Result<(), BraheError> {
    if value.contains(FORBIDDEN_PATH_COMPONENT_CHARACTERS) {
        return Err(BraheError::Error(format!(
            "invalid {field}: '{value}' must not contain '/', '\\' or NUL"
        )));
    }
    if value == "." || value == ".." {
        return Err(BraheError::Error(format!(
            "invalid {field}: '{value}' must not be '.' or '..'"
        )));
    }
    Ok(())
}

/// Counter that makes each staging path distinct within a process, so two
/// threads writing to the same target do not share, truncate or delete each
/// other's staging file.
static STAGING_COUNTER: AtomicU64 = AtomicU64::new(0);

/// Staging path used while a file is written beside `target`, so a partial
/// write never appears at `target` itself.
///
/// Every call returns a different name: the process id separates processes
/// and a monotonic counter separates calls within one process.
///
/// # Arguments
/// * `target` - Final destination path
///
/// # Returns
/// * `PathBuf`: A `.{name}.{pid}.{n}.tmp` sibling of `target`, unique to this call
fn staging_path(target: &Path) -> PathBuf {
    target.with_file_name(format!(
        ".{}.{}.{}.tmp",
        target
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("file"),
        std::process::id(),
        STAGING_COUNTER.fetch_add(1, Ordering::Relaxed)
    ))
}

/// Normalises `path` to an absolute path without `.` or `..` components,
/// without consulting the filesystem.
///
/// A relative path is taken against the current working directory first, so
/// the result is always absolute. `.` components are dropped and each `..`
/// removes the preceding component when that component is a normal name; a
/// `..` directly under the root is dropped, since the root is its own parent.
///
/// # Arguments
/// * `path` - Path to normalise, which need not exist
///
/// # Returns
/// * `Some(PathBuf)`: The normalised absolute path
/// * `None`: If `path` is relative and the current working directory cannot be read
fn normalize_lexically(path: &Path) -> Option<PathBuf> {
    let absolute = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir().ok()?.join(path)
    };

    let mut normalized = PathBuf::new();
    for component in absolute.components() {
        match component {
            Component::Prefix(_) | Component::RootDir | Component::Normal(_) => {
                normalized.push(component.as_os_str())
            }
            Component::CurDir => {}
            Component::ParentDir => {
                if matches!(
                    normalized.components().next_back(),
                    Some(Component::Normal(_))
                ) {
                    normalized.pop();
                }
            }
        }
    }
    Some(normalized)
}

/// Resolves `path` as far as the filesystem allows.
///
/// The path is first normalised lexically, which makes it absolute and
/// removes every `.` and `..`. The longest prefix of the result that exists
/// is then canonicalized, so symlinks in the part that exists are followed,
/// and the components that do not exist yet are appended to it.
///
/// # Arguments
/// * `path` - Path to resolve, which need not exist
///
/// # Returns
/// * `Some(PathBuf)`: The resolved path
/// * `None`: If `path` cannot be normalised, or if no prefix of it can be canonicalized
fn resolve_existing_ancestor(path: &Path) -> Option<PathBuf> {
    let mut existing = normalize_lexically(path)?;
    let mut remainder: Vec<OsString> = Vec::new();
    loop {
        if let Ok(canonical) = fs::canonicalize(&existing) {
            let mut resolved = canonical;
            for part in remainder.iter().rev() {
                resolved.push(part);
            }
            return Some(resolved);
        }
        remainder.push(existing.file_name()?.to_os_string());
        if !existing.pop() {
            return None;
        }
    }
}

/// Whether `path` is `root` itself or lies underneath it.
///
/// `path` need not exist. It is normalised lexically first — made absolute
/// against the current working directory, with `.` dropped and `..` applied
/// to the preceding component — and the longest existing prefix of the result
/// is then canonicalized, so a symlink in the part that exists cannot hide a
/// path that ends up inside `root`. Because `..` is applied before the
/// filesystem is consulted, a `..` that crosses a symlinked directory is
/// judged on the written path rather than on where that symlink points.
///
/// # Arguments
/// * `path` - Path to test, existing or not
/// * `root` - Directory that must contain `path`
///
/// # Returns
/// * `bool`: `true` if `path` resolves to `root` or to an entry inside it; `false` if `root` does not exist or `path` cannot be resolved
///
/// # Examples
///
/// ```
/// use std::path::Path;
/// use brahe::utils::fs::is_within;
///
/// let root = std::env::temp_dir();
/// assert!(is_within(&root.join("a").join("b"), &root));
/// assert!(is_within(&root, &root));
/// assert!(is_within(&root.join("a").join("..").join("b"), &root));
/// assert!(!is_within(Path::new("/"), &root));
/// ```
pub fn is_within(path: &Path, root: &Path) -> bool {
    let Ok(root) = fs::canonicalize(root) else {
        return false;
    };
    resolve_existing_ancestor(path).is_some_and(|resolved| resolved.starts_with(&root))
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
/// The copy is written to a staging sibling of `target`, synced, and renamed
/// into place, so a partial copy never appears at `target`. The staging file
/// is created exclusively, so an unexpected file at that path is an error
/// rather than something to truncate.
///
/// # Arguments
/// * `source` - File to copy; left in place
/// * `target` - Destination path
///
/// # Returns
/// * `Ok(())`: The file was copied
/// * `Err(BraheError)`: If the source cannot be read, the staging file cannot be created, or the copy or the rename into place fails; a staging file this call created is removed
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
    let failed = |e: io::Error| {
        BraheError::IoError(format!(
            "Failed to copy {} to {}: {}",
            source.display(),
            target.display(),
            e
        ))
    };

    let staging = staging_path(target);
    let mut reader = fs::File::open(source).map_err(failed)?;
    let mut writer = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&staging)
        .map_err(failed)?;

    // Only now does this call own `staging` and may remove it on failure.
    let result = io::copy(&mut reader, &mut writer).and_then(|_| writer.sync_all());
    drop(writer);
    if let Err(e) = result.and_then(|()| fs::rename(&staging, target)) {
        let _ = fs::remove_file(&staging);
        return Err(failed(e));
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
    use serial_test::{parallel, serial};
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
    fn test_validate_path_component_accepts_plain_values() {
        for value in ["STARLINK-38128", "", "a_b", "...", ".hidden", "a.b.c"] {
            assert!(
                validate_path_component("object_name", value).is_ok(),
                "expected '{value}' to be accepted"
            );
        }
    }

    #[test]
    #[parallel]
    fn test_validate_path_component_rejects_separators_and_traversal() {
        for value in ["a/b", "a\\b", "a\0b", "../evil", "/tmp/evil"] {
            let err = validate_path_component("object_name", value)
                .unwrap_err()
                .to_string();
            assert!(
                err.contains("must not contain '/', '\\' or NUL"),
                "{value}: {err}"
            );
        }

        for value in [".", ".."] {
            let err = validate_path_component("metadata", value)
                .unwrap_err()
                .to_string();
            assert!(err.contains("must not be '.' or '..'"), "{value}: {err}");
            assert!(err.contains("invalid metadata"), "{err}");
        }
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
    fn test_copy_file_leaves_nothing_behind_when_the_rename_fails() {
        let dir = tempdir().unwrap();
        let source = dir.path().join("source.txt");
        fs::write(&source, b"payload").unwrap();

        // Renaming a file onto an existing directory fails after the staging
        // file has already been written.
        let target = dir.path().join("target_dir");
        fs::create_dir(&target).unwrap();

        let err = copy_file(&source, &target).unwrap_err().to_string();
        assert!(err.contains("Failed to copy"), "{err}");

        assert_eq!(fs::read(&source).unwrap(), b"payload");
        assert!(target.is_dir());
        assert_eq!(fs::read_dir(&target).unwrap().count(), 0);
        assert!(
            fs::read_dir(dir.path())
                .unwrap()
                .filter_map(|e| e.ok())
                .all(|e| !e.file_name().to_string_lossy().ends_with(".tmp"))
        );
    }

    #[test]
    #[parallel]
    fn test_staging_path_is_a_unique_hidden_sibling() {
        let staging = staging_path(Path::new("/tmp/out/data.txt"));
        assert_eq!(staging.parent().unwrap(), Path::new("/tmp/out"));
        let name = staging.file_name().unwrap().to_str().unwrap();
        assert!(name.starts_with(".data.txt."), "{name}");
        assert!(name.ends_with(".tmp"), "{name}");

        // Two calls for the same target never name the same staging file.
        assert_ne!(
            staging_path(Path::new("/tmp/out/data.txt")),
            staging_path(Path::new("/tmp/out/data.txt"))
        );

        let trailing_slash = staging_path(Path::new("/tmp/out/"));
        let name = trailing_slash.file_name().unwrap().to_str().unwrap();
        assert!(name.starts_with(".out."), "{name}");

        // A path with no file name component falls back to the `file` stem.
        let rooted = staging_path(Path::new("/"));
        let name = rooted.file_name().unwrap().to_str().unwrap();
        assert!(name.starts_with(".file."), "{name}");
    }

    #[test]
    #[parallel]
    fn test_is_within_covers_missing_paths_and_escapes() {
        let dir = tempdir().unwrap();
        let root = dir.path().join("cache");
        fs::create_dir_all(root.join("nested")).unwrap();

        assert!(is_within(&root, &root));
        assert!(is_within(&root.join("nested"), &root));

        // The destination need not exist yet.
        assert!(is_within(&root.join("exports"), &root));
        assert!(is_within(
            &root.join("exports").join("a").join("b.txt"),
            &root
        ));

        // `.` and `..` are applied before the comparison.
        assert!(is_within(&root.join(".").join("exports"), &root));
        assert!(!is_within(
            &root.join("nested").join("..").join(".."),
            &root
        ));

        // `..` after a component that does not exist still lands inside.
        assert!(is_within(&root.join("missing").join(".."), &root));
        assert!(
            is_within(&root.join("missing").join("..").join("exports"), &root),
            "a missing intermediate must not defeat the containment check"
        );

        // `..` that walks back out of the root is rejected.
        assert!(!is_within(
            &root.join("missing").join("..").join(".."),
            &root
        ));
        assert!(!is_within(
            &root.join("nested").join("..").join("..").join("escape"),
            &root
        ));

        let outside = dir.path().join("outside");
        assert!(!is_within(&outside, &root));
        assert!(!is_within(dir.path(), &root));

        // A root that does not exist contains nothing.
        assert!(!is_within(&root, &dir.path().join("missing")));
    }

    #[test]
    #[serial]
    fn test_is_within_resolves_relative_paths_against_the_working_directory() {
        let dir = tempdir().unwrap();
        let root = fs::canonicalize(dir.path()).unwrap();
        fs::create_dir_all(root.join("cache")).unwrap();

        let original = std::env::current_dir().unwrap();
        std::env::set_current_dir(&root).unwrap();

        let relative_inside = is_within(Path::new("cache/exports"), &root.join("cache"));
        let bare_inside = is_within(Path::new("out.txt"), &root);
        let relative_outside = is_within(Path::new(".."), &root);

        std::env::set_current_dir(original).unwrap();

        assert!(relative_inside);
        assert!(bare_inside);
        assert!(!relative_outside);
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
