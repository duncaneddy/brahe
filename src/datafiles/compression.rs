//! Detection and decoding of compressed data file payloads.
//!
//! Data products are distributed gzip-compressed (`.gz`) or, for older
//! archives, Unix-compressed (`.Z`). Gzip data is decoded and encoded here;
//! Unix compress data is recognized so that callers can report a clear error.

use std::borrow::Cow;
use std::io::{Read, Write};

use flate2::Compression as GzipLevel;
use flate2::read::MultiGzDecoder;
use flate2::write::GzEncoder;

use crate::utils::BraheError;

/// Magic bytes at the start of a gzip stream.
const GZIP_MAGIC: [u8; 2] = [0x1f, 0x8b];
/// Magic bytes at the start of a Unix compress (`.Z`) stream.
const UNIX_COMPRESS_MAGIC: [u8; 2] = [0x1f, 0x9d];

/// Maximum number of bytes a gzip payload may decompress to (1 GiB).
pub const MAX_DECOMPRESSED_BYTES: u64 = 1 << 30;

/// Error message for Unix compress (`.Z`) data, which cannot be decoded.
pub const UNIX_COMPRESS_UNSUPPORTED: &str = "Unix compress (.Z) data is not supported; decompress it externally (gzip -d) and read the plain file";

/// Compression format of a byte payload, identified by its leading magic bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Compression {
    /// No recognized compression; the bytes are used as they are.
    None,
    /// A gzip stream (magic `1f 8b`).
    Gzip,
    /// A Unix compress stream (magic `1f 9d`).
    UnixCompress,
}

/// Identify the compression format of a payload from its first two bytes.
///
/// # Arguments
///
/// * `bytes` - The payload
///
/// # Returns
///
/// * `Compression` - `Gzip` for magic `1f 8b`, `UnixCompress` for `1f 9d`, and
///   `None` for anything else, including payloads shorter than two bytes
pub(crate) fn compression_of(bytes: &[u8]) -> Compression {
    match bytes.get(..2) {
        Some(magic) if magic == GZIP_MAGIC => Compression::Gzip,
        Some(magic) if magic == UNIX_COMPRESS_MAGIC => Compression::UnixCompress,
        _ => Compression::None,
    }
}

/// Decode a payload that may be gzip-compressed.
///
/// Concatenated gzip members are decoded as a single stream. The decoded size
/// is limited to [`MAX_DECOMPRESSED_BYTES`] (1 GiB) across all members, which
/// bounds the memory used for untrusted input.
///
/// # Arguments
///
/// * `bytes` - The payload, either plain or gzip-compressed
///
/// # Returns
///
/// * `Result<Cow<[u8]>, BraheError>` - The payload borrowed unchanged when it is not
///   compressed, or the decoded bytes. An `Error` holding
///   [`UNIX_COMPRESS_UNSUPPORTED`] for Unix compress data, an `Error` describing the
///   failure for a corrupt gzip stream, or an `Error` when the decoded size exceeds 1 GiB.
pub(crate) fn decompress(bytes: &[u8]) -> Result<Cow<'_, [u8]>, BraheError> {
    decompress_limited(bytes, MAX_DECOMPRESSED_BYTES)
}

/// Decode a payload that may be gzip-compressed, limiting the decoded size.
///
/// # Arguments
///
/// * `bytes` - The payload, either plain or gzip-compressed
/// * `limit` - Maximum number of decoded bytes, summed over all gzip members
///
/// # Returns
///
/// * `Result<Cow<[u8]>, BraheError>` - As for [`decompress`], with an `Error` when the
///   decoded size exceeds `limit`
fn decompress_limited(bytes: &[u8], limit: u64) -> Result<Cow<'_, [u8]>, BraheError> {
    match compression_of(bytes) {
        Compression::None => Ok(Cow::Borrowed(bytes)),
        Compression::Gzip => {
            let mut decoded = Vec::new();
            MultiGzDecoder::new(bytes)
                .take(limit.saturating_add(1))
                .read_to_end(&mut decoded)
                .map_err(|e| BraheError::Error(format!("invalid gzip data: {}", e)))?;
            if decoded.len() as u64 > limit {
                return Err(BraheError::Error(format!(
                    "decompressed gzip data exceeds the limit of {} bytes",
                    limit
                )));
            }
            Ok(Cow::Owned(decoded))
        }
        Compression::UnixCompress => Err(BraheError::Error(UNIX_COMPRESS_UNSUPPORTED.to_string())),
    }
}

/// Compress a payload as a gzip stream at the default compression level.
///
/// # Arguments
///
/// * `bytes` - The payload to compress
///
/// # Returns
///
/// * `Result<Vec<u8>, BraheError>` - The gzip stream, or an `Error` if encoding fails
pub(crate) fn gzip(bytes: &[u8]) -> Result<Vec<u8>, BraheError> {
    let mut encoder = GzEncoder::new(Vec::new(), GzipLevel::default());
    encoder
        .write_all(bytes)
        .and_then(|()| encoder.finish())
        .map_err(|e| BraheError::Error(format!("gzip compression failed: {}", e)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::rstest;
    use serial_test::parallel;

    #[rstest]
    #[case(&[0x1f, 0x8b, 0x08], Compression::Gzip)]
    #[case(&[0x1f, 0x9d, 0x90], Compression::UnixCompress)]
    #[case(b"#dP2024", Compression::None)]
    #[case(&[0x1f], Compression::None)]
    #[case(&[], Compression::None)]
    #[parallel]
    fn test_compression_of(#[case] bytes: &[u8], #[case] expected: Compression) {
        assert_eq!(compression_of(bytes), expected);
    }

    #[test]
    #[parallel]
    fn test_decompress_plain_is_borrowed() {
        let plain = b"#dP2024 plain text";
        match decompress(plain).unwrap() {
            Cow::Borrowed(out) => assert_eq!(out, plain),
            Cow::Owned(_) => panic!("plain bytes must not be copied"),
        }
    }

    #[test]
    #[parallel]
    fn test_gzip_roundtrip() {
        let plain: Vec<u8> = (0..10_000u32).flat_map(|i| i.to_le_bytes()).collect();
        let packed = gzip(&plain).unwrap();
        assert_eq!(compression_of(&packed), Compression::Gzip);
        assert!(packed.len() < plain.len());
        match decompress(&packed).unwrap() {
            Cow::Owned(out) => assert_eq!(out, plain),
            Cow::Borrowed(_) => panic!("gzip data must be decoded"),
        }
        assert_eq!(&*decompress(&gzip(b"").unwrap()).unwrap(), b"");
    }

    #[test]
    #[parallel]
    fn test_decompress_concatenated_members() {
        let mut packed = gzip(b"first ").unwrap();
        packed.extend(gzip(b"second").unwrap());
        assert_eq!(&*decompress(&packed).unwrap(), b"first second");
    }

    #[test]
    #[parallel]
    fn test_decompress_corrupt_gzip() {
        let mut packed = gzip(b"some text to compress, long enough to matter").unwrap();
        packed.truncate(packed.len() / 2);
        match decompress(&packed) {
            Err(BraheError::Error(msg)) => assert!(msg.starts_with("invalid gzip data"), "{msg}"),
            other => panic!("expected a gzip error, got {other:?}"),
        }
        match decompress(&[0x1f, 0x8b, 0x00, 0x00]) {
            Err(BraheError::Error(msg)) => assert!(msg.starts_with("invalid gzip data"), "{msg}"),
            other => panic!("expected a gzip error, got {other:?}"),
        }
    }

    #[test]
    #[parallel]
    fn test_decompress_limit_single_member() {
        const MIB: usize = 1 << 20;
        let packed = gzip(&vec![0u8; 2 * MIB]).unwrap();
        assert!(packed.len() < MIB);
        match decompress_limited(&packed, MIB as u64) {
            Err(BraheError::Error(msg)) => {
                assert!(msg.contains("exceeds the limit of 1048576 bytes"), "{msg}")
            }
            other => panic!("expected a size error, got {other:?}"),
        }
        assert_eq!(
            decompress_limited(&packed, 2 * MIB as u64).unwrap().len(),
            2 * MIB
        );
    }

    #[test]
    #[parallel]
    fn test_decompress_limit_exactly_at_limit() {
        let packed = gzip(&[7u8; 1000]).unwrap();
        assert_eq!(decompress_limited(&packed, 1000).unwrap().len(), 1000);
        assert!(decompress_limited(&packed, 999).is_err());
    }

    #[test]
    #[parallel]
    fn test_decompress_limit_across_members() {
        let mut packed = gzip(&[1u8; 600]).unwrap();
        packed.extend(gzip(&[2u8; 600]).unwrap());
        assert!(matches!(
            decompress_limited(&packed, 1000),
            Err(BraheError::Error(_))
        ));
        assert_eq!(decompress_limited(&packed, 1200).unwrap().len(), 1200);
    }

    #[test]
    #[parallel]
    fn test_decompress_limit_value() {
        assert_eq!(MAX_DECOMPRESSED_BYTES, 1_073_741_824);
    }

    #[test]
    #[parallel]
    fn test_decompress_unix_compress_unsupported() {
        match decompress(&[0x1f, 0x9d, 0x90, 0x00]) {
            Err(BraheError::Error(msg)) => assert_eq!(msg, UNIX_COMPRESS_UNSUPPORTED),
            other => panic!("expected an unsupported error, got {other:?}"),
        }
    }
}
