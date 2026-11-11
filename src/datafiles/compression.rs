//! Detection and decoding of compressed data file payloads.
//!
//! Data products are distributed gzip-compressed (`.gz`) or, for older
//! archives, Unix-compressed (`.Z`). Both are decoded here, and gzip data is
//! also encoded.

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

/// Maximum number of bytes a compressed payload may decompress to (1 GiB).
pub const MAX_DECOMPRESSED_BYTES: u64 = 1 << 30;

/// Code that resets the string table of a Unix compress stream in block mode.
const LZW_CLEAR: u32 = 256;
/// Width of the first codes of a Unix compress stream, in bits.
const LZW_INITIAL_BITS: u32 = 9;
/// Largest code width a Unix compress stream can declare, in bits.
const LZW_MAX_BITS: u32 = 16;

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
///
/// # Examples
///
/// ```ignore
/// assert_eq!(compression_of(&[0x1f, 0x8b, 0x08]), Compression::Gzip);
/// assert_eq!(compression_of(b"#dP2024"), Compression::None);
/// ```
pub(crate) fn compression_of(bytes: &[u8]) -> Compression {
    match bytes.get(..2) {
        Some(magic) if magic == GZIP_MAGIC => Compression::Gzip,
        Some(magic) if magic == UNIX_COMPRESS_MAGIC => Compression::UnixCompress,
        _ => Compression::None,
    }
}

/// Decode a payload that may be gzip-compressed or Unix-compressed.
///
/// Concatenated gzip members are decoded as a single stream. The decoded size
/// is limited to [`MAX_DECOMPRESSED_BYTES`] (1 GiB), across all gzip members,
/// which bounds the memory used for untrusted input.
///
/// # Arguments
///
/// * `bytes` - The payload: plain, gzip-compressed, or Unix-compressed
///
/// # Returns
///
/// * `Result<Cow<[u8]>, BraheError>` - The payload borrowed unchanged when it is not
///   compressed, or the decoded bytes. An `Error` describing the failure for a corrupt
///   gzip or Unix compress stream, or an `Error` when the decoded size exceeds 1 GiB.
///
/// # Examples
///
/// ```ignore
/// let packed = gzip(b"#dP2024").unwrap();
/// assert_eq!(&*decompress(&packed).unwrap(), b"#dP2024");
/// assert_eq!(&*decompress(b"#dP2024").unwrap(), b"#dP2024");
/// assert_eq!(&*decompress(&[0x1f, 0x9d, 0x90, 0x23, 0x64, 0x50]).unwrap(), b"#2");
/// ```
pub(crate) fn decompress(bytes: &[u8]) -> Result<Cow<'_, [u8]>, BraheError> {
    decompress_limited(bytes, MAX_DECOMPRESSED_BYTES)
}

/// Decode a payload that may be compressed, limiting the decoded size.
///
/// # Arguments
///
/// * `bytes` - The payload: plain, gzip-compressed, or Unix-compressed
/// * `limit` - Maximum number of decoded bytes, summed over all gzip members
///
/// # Returns
///
/// * `Result<Cow<[u8]>, BraheError>` - As for [`decompress`], with an `Error` when the
///   decoded size exceeds `limit`
///
/// # Examples
///
/// ```ignore
/// let packed = gzip(&[0u8; 1024]).unwrap();
/// assert_eq!(decompress_limited(&packed, 1024).unwrap().len(), 1024);
/// assert!(decompress_limited(&packed, 1023).is_err());
/// ```
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
        Compression::UnixCompress => unix_decompress(bytes, limit).map(Cow::Owned),
    }
}

/// Decode a Unix compress (`.Z`) stream.
///
/// The stream is a three-byte header followed by LZW codes packed least
/// significant bit first. The third header byte gives the largest code width
/// (9 to 16 bits) and whether code 256 clears the string table. Codes start
/// 9 bits wide and widen by one bit each time the table fills the current
/// width. The `compress` utility writes codes in groups of eight, so the
/// codes left in a group are skipped whenever the width changes or the table
/// is cleared.
///
/// # Arguments
///
/// * `bytes` - The stream, starting with the magic bytes `1f 9d`
/// * `limit` - Maximum number of decoded bytes
///
/// # Returns
///
/// * `Result<Vec<u8>, BraheError>` - The decoded bytes, or an `Error` when the header is
///   malformed, a code refers to a table entry that does not exist, the stream ends
///   inside a code, or the decoded size exceeds `limit`
///
/// # Examples
///
/// ```ignore
/// // Header `1f 9d 90`, then the 9-bit codes for `#` and `2`.
/// assert_eq!(unix_decompress(&[0x1f, 0x9d, 0x90, 0x23, 0x64, 0x50], 16).unwrap(), b"#2");
/// ```
fn unix_decompress(bytes: &[u8], limit: u64) -> Result<Vec<u8>, BraheError> {
    let invalid =
        |detail: &str| BraheError::Error(format!("invalid Unix compress data: {}", detail));
    let flags = *bytes.get(2).ok_or_else(|| invalid("missing header"))?;
    if flags & 0x60 != 0 {
        return Err(invalid("unknown header flags"));
    }
    let block_mode = flags & 0x80 != 0;
    let max_bits = match u32::from(flags & 0x1f) {
        // `compress` cannot write 9-bit streams; a declared 9 behaves as 10.
        9 => 10,
        bits @ 10..=LZW_MAX_BITS => bits,
        _ => return Err(invalid("code width outside 9 to 16 bits")),
    };

    // Entry `code` of the string table is the string of `prefix[code]`
    // followed by the byte `suffix[code]`. Codes below 256 are single bytes.
    let mut prefix = vec![0u16; 1 << max_bits];
    let mut suffix = vec![0u8; 1 << max_bits];
    // Last code in the table. In block mode code 256 is reserved for clearing.
    let mut last = if block_mode { LZW_CLEAR } else { LZW_CLEAR - 1 };
    let mut bits = LZW_INITIAL_BITS;
    let mut mask = (1u32 << bits) - 1;

    let mut next = 3;
    // Offset at which the current run of equal-width codes started.
    let mut run_start = next;
    let mut buffer = 0u32;
    let mut buffered = 0u32;
    let mut previous: Option<u32> = None;
    let mut first_byte = 0u8;
    let mut string = Vec::new();
    let mut decoded = Vec::new();

    loop {
        if last >= mask && bits < max_bits {
            next = lzw_next_group(next, run_start, bits);
            run_start = next;
            (buffer, buffered) = (0, 0);
            bits += 1;
            mask = (1 << bits) - 1;
        }

        let Some(&byte) = bytes.get(next) else {
            break;
        };
        buffer |= u32::from(byte) << buffered;
        buffered += 8;
        next += 1;
        if buffered < bits {
            let byte = *bytes
                .get(next)
                .ok_or_else(|| invalid("stream ends inside a code"))?;
            buffer |= u32::from(byte) << buffered;
            buffered += 8;
            next += 1;
        }
        let code = buffer & mask;
        buffer >>= bits;
        buffered -= bits;

        if code == LZW_CLEAR && block_mode {
            next = lzw_next_group(next, run_start, bits);
            run_start = next;
            (buffer, buffered) = (0, 0);
            bits = LZW_INITIAL_BITS;
            mask = (1 << bits) - 1;
            last = LZW_CLEAR - 1;
            continue;
        }

        let mut entry = code;
        if code > last {
            // The only code that may be one past the table is the string of
            // the previous code followed by its own first byte.
            match previous {
                Some(previous) if code == last + 1 && previous <= last => {
                    string.push(first_byte);
                    entry = previous;
                }
                _ => return Err(invalid("code refers to a missing table entry")),
            }
        }
        while entry >= 256 {
            string.push(suffix[entry as usize]);
            entry = u32::from(prefix[entry as usize]);
        }
        first_byte = entry as u8;
        string.push(first_byte);

        if let Some(previous) = previous
            && last < mask
        {
            last += 1;
            prefix[last as usize] = previous as u16;
            suffix[last as usize] = first_byte;
        }
        previous = Some(code);

        if (decoded.len() + string.len()) as u64 > limit {
            return Err(BraheError::Error(format!(
                "decompressed Unix compress data exceeds the limit of {} bytes",
                limit
            )));
        }
        decoded.extend(string.drain(..).rev());
    }
    Ok(decoded)
}

/// Offset of the next group of eight codes in a Unix compress stream.
///
/// Eight codes of `bits` bits fill `bits` bytes, so groups start every `bits`
/// bytes from the start of the run.
///
/// # Arguments
///
/// * `next` - Offset of the next unread byte
/// * `run_start` - Offset at which the current run of equal-width codes started
/// * `bits` - Code width of the run, in bits
///
/// # Returns
///
/// * `usize` - `next` rounded up to a whole number of groups from `run_start`
///
/// # Examples
///
/// ```ignore
/// assert_eq!(lzw_next_group(5, 3, 9), 12);
/// assert_eq!(lzw_next_group(12, 3, 9), 12);
/// ```
fn lzw_next_group(next: usize, run_start: usize, bits: u32) -> usize {
    let group = bits as usize;
    run_start + (next - run_start).div_ceil(group) * group
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
///
/// # Examples
///
/// ```ignore
/// let packed = gzip(b"#dP2024").unwrap();
/// assert_eq!(compression_of(&packed), Compression::Gzip);
/// ```
pub(crate) fn gzip(bytes: &[u8]) -> Result<Vec<u8>, BraheError> {
    let mut encoder = GzEncoder::new(Vec::new(), GzipLevel::default());
    encoder
        .write_all(bytes)
        .and_then(|()| encoder.finish())
        .map_err(|e| BraheError::Error(format!("gzip compression failed: {}", e)))
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
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

    #[rstest]
    #[case(&[0x1f, 0x9d, 0x90], b"")]
    #[case(&[0x1f, 0x9d, 0x90, 0x23, 0x64, 0x50], b"#2")]
    // A code one past the table: the previous string plus its first byte.
    #[case(&[0x1f, 0x9d, 0x90, 97, 2, 10, 28, 8], b"aaaaaaaaaa")]
    // `a`, a clear code, padding to the end of the group of eight codes, `b`, `c`.
    #[case(&[0x1f, 0x9d, 0x90, 97, 0, 2, 0, 0, 0, 0, 0, 0, 98, 198, 0], b"abc")]
    // Without block mode, code 256 is the first table entry.
    #[case(&[0x1f, 0x9d, 0x10, 97, 196, 0, 4, 8], b"ababab")]
    #[parallel]
    fn test_decompress_unix_compress(#[case] packed: &[u8], #[case] expected: &[u8]) {
        match decompress(packed).unwrap() {
            Cow::Owned(out) => assert_eq!(out, expected),
            Cow::Borrowed(_) => panic!("Unix compress data must be decoded"),
        }
    }

    #[test]
    #[parallel]
    fn test_decompress_unix_compress_file() {
        // A 16-bit stream that widens its codes from 9 to 16 bits. The length
        // and FNV-1a hash are those of the output of `gzip -d`.
        let packed = std::fs::read("test_assets/sp3/gbm18432.sp3.Z").unwrap();
        let decoded = decompress(&packed).unwrap();
        assert_eq!(decoded.len(), 1_681_439);
        let hash = decoded.iter().fold(0xcbf2_9ce4_8422_2325u64, |hash, &b| {
            (hash ^ u64::from(b)).wrapping_mul(0x0000_0100_0000_01b3)
        });
        assert_eq!(hash, 0x288b_89ef_80b5_bbeb);
        assert!(decoded.starts_with(b"#cP2015  5  5  0  0  0.00000000"));
    }

    #[rstest]
    #[case(&[0x1f, 0x9d], "missing header")]
    #[case(&[0x1f, 0x9d, 0xb0], "unknown header flags")]
    #[case(&[0x1f, 0x9d, 0x88], "code width outside 9 to 16 bits")]
    #[case(&[0x1f, 0x9d, 0x90, 0x23], "stream ends inside a code")]
    #[case(&[0x1f, 0x9d, 0x90, 0x2c, 0x01], "code refers to a missing table entry")]
    #[case(&[0x1f, 0x9d, 0x90, 97, 4, 2], "code refers to a missing table entry")]
    #[parallel]
    fn test_decompress_unix_compress_errors(#[case] packed: &[u8], #[case] detail: &str) {
        match decompress(packed) {
            Err(BraheError::Error(msg)) => {
                assert_eq!(msg, format!("invalid Unix compress data: {detail}"))
            }
            other => panic!("expected a Unix compress error, got {other:?}"),
        }
    }

    #[test]
    #[parallel]
    fn test_decompress_unix_compress_limit() {
        let packed = [0x1f, 0x9d, 0x90, 97, 2, 10, 28, 8];
        assert_eq!(decompress_limited(&packed, 10).unwrap().len(), 10);
        match decompress_limited(&packed, 9) {
            Err(BraheError::Error(msg)) => {
                assert!(msg.contains("exceeds the limit of 9 bytes"), "{msg}")
            }
            other => panic!("expected a size error, got {other:?}"),
        }
    }
}
