/*!
 * DAF (Double precision Array File) container parser.
 *
 * DAF is the binary container format for SPICE kernels (SPK ephemerides and
 * binary PCK orientation files). This module parses the container structure
 * only; segment data interpretation lives in `segments.rs`.
 *
 * # References
 * - NAIF DAF Required Reading:
 *   <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/daf.html>
 */

use std::fs;
use std::path::Path;

use crate::utils::BraheError;

/// Size of a DAF physical record in bytes.
const RECORD_BYTES: usize = 1024;
/// Words (8-byte doubles) per record.
const RECORD_WORDS: usize = 128;

/// A single segment descriptor (summary) plus its name.
#[derive(Debug, Clone)]
pub(crate) struct DafSummary {
    /// Segment name from the name record (trimmed).
    pub name: String,
    /// The ND double-precision summary components.
    pub doubles: Vec<f64>,
    /// The NI integer summary components.
    pub ints: Vec<i32>,
}

/// A parsed DAF container: file-record metadata, all segment summaries, and
/// the file contents as endian-normalized 8-byte words.
#[derive(Debug)]
pub(crate) struct DafFile {
    /// ID word, e.g. "DAF/SPK" or "DAF/PCK" (trimmed).
    pub id_word: String,
    /// Number of double components per summary.
    #[allow(dead_code)]
    pub nd: usize,
    /// Number of integer components per summary.
    #[allow(dead_code)]
    pub ni: usize,
    /// All segment summaries in file order.
    pub summaries: Vec<DafSummary>,
    /// Entire file as f64 words (native-endian after normalization).
    words: Vec<f64>,
}

fn read_i32(bytes: &[u8], offset: usize, big_endian: bool) -> i32 {
    let b: [u8; 4] = bytes[offset..offset + 4].try_into().unwrap();
    if big_endian {
        i32::from_be_bytes(b)
    } else {
        i32::from_le_bytes(b)
    }
}

fn read_f64(bytes: &[u8], offset: usize, big_endian: bool) -> f64 {
    let b: [u8; 8] = bytes[offset..offset + 8].try_into().unwrap();
    if big_endian {
        f64::from_be_bytes(b)
    } else {
        f64::from_le_bytes(b)
    }
}

/// Read a summary-record control field (`NEXT`/`NSUM`), stored as an
/// integer-valued `f64`, and validate it is a finite, nonnegative integer
/// before converting to `usize`. Rejects `NaN`, negative, and fractional
/// values that would otherwise silently truncate on cast.
fn read_daf_count(
    bytes: &[u8],
    offset: usize,
    big_endian: bool,
    field: &str,
) -> Result<usize, BraheError> {
    let v = read_f64(bytes, offset, big_endian);
    if !v.is_finite() || v < 0.0 || v.fract() != 0.0 {
        return Err(BraheError::IoError(format!(
            "Invalid DAF: {} control value {} is not a nonnegative integer",
            field, v
        )));
    }
    Ok(v as usize)
}

impl DafFile {
    /// Parse a DAF from a file on disk.
    ///
    /// # Arguments
    /// - `path`: Path to a binary SPICE kernel (`.bsp`, `.bpc`)
    ///
    /// # Returns
    /// - Parsed `DafFile`, or `BraheError::IoError` on read/parse failure
    pub fn from_file(path: &Path) -> Result<Self, BraheError> {
        let bytes = fs::read(path).map_err(|e| {
            BraheError::IoError(format!("Failed to read kernel {}: {}", path.display(), e))
        })?;
        Self::from_bytes(&bytes)
    }

    /// Parse a DAF from an in-memory byte buffer.
    ///
    /// # Arguments
    /// - `bytes`: Raw bytes of a binary SPICE kernel
    ///
    /// # Returns
    /// - Parsed `DafFile`, or `BraheError::IoError` on parse failure
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BraheError> {
        if bytes.len() < RECORD_BYTES || !bytes.len().is_multiple_of(RECORD_BYTES) {
            return Err(BraheError::IoError(format!(
                "Invalid DAF: file size {} is not a positive multiple of {}",
                bytes.len(),
                RECORD_BYTES
            )));
        }

        let id_word = String::from_utf8_lossy(&bytes[0..8]).trim().to_string();
        if !id_word.starts_with("DAF/") {
            return Err(BraheError::IoError(format!(
                "Invalid DAF: ID word '{}' does not start with 'DAF/'",
                id_word
            )));
        }

        // Endianness from LOCFMT (bytes 88..96); fall back to an ND sanity
        // check for pre-LOCFMT files.
        let locfmt = String::from_utf8_lossy(&bytes[88..96]).trim().to_string();
        let big_endian = match locfmt.as_str() {
            "LTL-IEEE" => false,
            "BIG-IEEE" => true,
            _ => {
                let nd_le = read_i32(bytes, 8, false);
                if (1..=124).contains(&nd_le) {
                    false
                } else {
                    let nd_be = read_i32(bytes, 8, true);
                    if (1..=124).contains(&nd_be) {
                        true
                    } else {
                        return Err(BraheError::IoError(format!(
                            "Invalid DAF: unrecognized binary format tag '{}'",
                            locfmt
                        )));
                    }
                }
            }
        };

        // Validate as signed values first: a corrupt/adversarial header could
        // otherwise encode a negative count that wraps to a huge `usize` on
        // cast, causing an overflow panic instead of a clean parse error.
        let nd_raw = read_i32(bytes, 8, big_endian);
        let ni_raw = read_i32(bytes, 12, big_endian);
        let fward_raw = read_i32(bytes, 76, big_endian);
        let bward_raw = read_i32(bytes, 80, big_endian);
        let n_records = bytes.len() / RECORD_BYTES;
        if nd_raw <= 0
            || ni_raw < 2
            || fward_raw <= 0
            || bward_raw <= 0
            || fward_raw as usize > n_records
            || bward_raw as usize > n_records
        {
            return Err(BraheError::IoError(format!(
                "Invalid DAF: implausible file record (ND={}, NI={}, FWARD={}, BWARD={}, records={})",
                nd_raw, ni_raw, fward_raw, bward_raw, n_records
            )));
        }
        let nd = nd_raw as usize;
        let ni = ni_raw as usize;
        let fward = fward_raw as usize;
        // Summary size in words; NI ints pack two per word (rounded up).
        let ss = nd + ni.div_ceil(2);

        // Walk the summary-record linked list.
        let mut summaries = Vec::new();
        let mut record = fward;
        let mut visited = 0usize;
        while record != 0 {
            if record > n_records || record + 1 > n_records {
                return Err(BraheError::IoError(format!(
                    "Invalid DAF: summary record {} out of range ({} records)",
                    record, n_records
                )));
            }
            visited += 1;
            if visited > n_records {
                return Err(BraheError::IoError(
                    "Invalid DAF: summary record list does not terminate".to_string(),
                ));
            }

            let rec_off = (record - 1) * RECORD_BYTES;
            let next = read_daf_count(bytes, rec_off, big_endian, "NEXT")?;
            let nsum = read_daf_count(bytes, rec_off + 16, big_endian, "NSUM")?;
            let max_nsum = (RECORD_WORDS - 3) / ss;
            if nsum > max_nsum {
                return Err(BraheError::IoError(format!(
                    "Invalid DAF: summary record {} claims {} summaries (max {})",
                    record, nsum, max_nsum
                )));
            }

            // Name record immediately follows the summary record.
            let name_off = record * RECORD_BYTES;
            let nc = 8 * ss;

            for i in 0..nsum {
                let s_off = rec_off + (3 + i * ss) * 8;
                let mut doubles = Vec::with_capacity(nd);
                for d in 0..nd {
                    doubles.push(read_f64(bytes, s_off + d * 8, big_endian));
                }
                let mut ints = Vec::with_capacity(ni);
                for k in 0..ni {
                    ints.push(read_i32(bytes, s_off + nd * 8 + k * 4, big_endian));
                }
                let name =
                    String::from_utf8_lossy(&bytes[name_off + i * nc..name_off + (i + 1) * nc])
                        .trim()
                        .to_string();
                summaries.push(DafSummary {
                    name,
                    doubles,
                    ints,
                });
            }

            record = next;
        }

        // Convert the whole file to native-endian f64 words once.
        let mut words = Vec::with_capacity(bytes.len() / 8);
        for chunk in bytes.chunks_exact(8) {
            let b: [u8; 8] = chunk.try_into().unwrap();
            words.push(if big_endian {
                f64::from_be_bytes(b)
            } else {
                f64::from_le_bytes(b)
            });
        }

        Ok(DafFile {
            id_word,
            nd,
            ni,
            summaries,
            words,
        })
    }

    /// Return the words in the 1-based inclusive address range `[start_addr, end_addr]`.
    ///
    /// # Arguments
    /// - `start_addr`: First word address (1-based)
    /// - `end_addr`: Last word address (1-based, inclusive)
    ///
    /// # Returns
    /// - Slice of `f64` words in the requested range, or `BraheError::IoError` if
    ///   the range is invalid or out of bounds
    pub fn words(&self, start_addr: usize, end_addr: usize) -> Result<&[f64], BraheError> {
        if start_addr == 0 || end_addr < start_addr || end_addr > self.words.len() {
            return Err(BraheError::IoError(format!(
                "DAF address range [{}, {}] invalid (file has {} words)",
                start_addr,
                end_addr,
                self.words.len()
            )));
        }
        Ok(&self.words[start_addr - 1..end_addr])
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn de440s_path() -> Option<PathBuf> {
        let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_assets")
            .join("de440s.bsp");
        p.exists().then_some(p)
    }

    #[test]
    fn test_daf_from_file_de440s() {
        let Some(path) = de440s_path() else { return };
        let daf = DafFile::from_file(&path).unwrap();

        assert_eq!(daf.id_word, "DAF/SPK");
        assert_eq!(daf.nd, 2);
        assert_eq!(daf.ni, 6);
        assert_eq!(daf.summaries.len(), 14);

        // Every segment: 6 ints [target, center, frame, type, start, end];
        // DE440s is all Type 2, frame 1 (J2000/ICRF)
        let mut pairs: Vec<(i32, i32)> = Vec::new();
        for s in &daf.summaries {
            assert_eq!(s.doubles.len(), 2);
            assert_eq!(s.ints.len(), 6);
            assert_eq!(s.ints[2], 1, "frame must be 1 (J2000)");
            assert_eq!(s.ints[3], 2, "type must be 2");
            assert!(s.doubles[0] < s.doubles[1], "start_et < end_et");
            assert!(s.ints[4] > 0 && s.ints[5] > s.ints[4]);
            pairs.push((s.ints[0], s.ints[1]));
        }
        for expect in [
            (1, 0),
            (2, 0),
            (3, 0),
            (4, 0),
            (5, 0),
            (6, 0),
            (7, 0),
            (8, 0),
            (9, 0),
            (10, 0),
            (301, 3),
            (399, 3),
            (199, 1),
            (299, 2),
        ] {
            assert!(pairs.contains(&expect), "missing segment {:?}", expect);
        }
    }

    #[test]
    fn test_daf_words_addressing() {
        let Some(path) = de440s_path() else { return };
        let daf = DafFile::from_file(&path).unwrap();
        let s = &daf.summaries[0];
        let (start, end) = (s.ints[4] as usize, s.ints[5] as usize);
        let w = daf.words(start, end).unwrap();
        assert_eq!(w.len(), end - start + 1);
        // Trailer N (last word) must be a positive integer-valued double
        let n = w[w.len() - 1];
        assert!(n > 0.0 && n.fract() == 0.0);
        // Out-of-range address errors
        assert!(daf.words(0, 10).is_err());
        assert!(daf.words(1, usize::MAX / 8).is_err());
    }

    #[test]
    fn test_daf_rejects_garbage() {
        assert!(DafFile::from_bytes(&[0u8; 100]).is_err()); // too short
        let mut junk = vec![0u8; 2048];
        junk[..8].copy_from_slice(b"NOTADAF ");
        assert!(DafFile::from_bytes(&junk).is_err()); // bad ID word
    }

    #[test]
    fn test_daf_rejects_negative_header_fields() {
        // A negative ND must not wrap to a huge usize on cast; it should be
        // rejected cleanly instead of panicking downstream.
        let mut file = vec![0u8; 1024];
        file[..8].copy_from_slice(b"DAF/SPK ");
        file[8..12].copy_from_slice(&(-1i32).to_le_bytes()); // ND (invalid)
        file[12..16].copy_from_slice(&6i32.to_le_bytes()); // NI
        file[76..80].copy_from_slice(&1i32.to_le_bytes()); // FWARD
        file[80..84].copy_from_slice(&1i32.to_le_bytes()); // BWARD
        file[88..96].copy_from_slice(b"LTL-IEEE");
        assert!(DafFile::from_bytes(&file).is_err());
    }

    #[test]
    fn test_daf_rejects_invalid_control_fields() {
        // NEXT = NaN in the summary record must be rejected rather than
        // silently cast to an incorrect usize.
        let mut file = vec![0u8; 2 * 1024];
        file[..8].copy_from_slice(b"DAF/SPK ");
        file[8..12].copy_from_slice(&2i32.to_le_bytes()); // ND
        file[12..16].copy_from_slice(&6i32.to_le_bytes()); // NI
        file[76..80].copy_from_slice(&2i32.to_le_bytes()); // FWARD -> record 2
        file[80..84].copy_from_slice(&2i32.to_le_bytes()); // BWARD
        file[88..96].copy_from_slice(b"LTL-IEEE");

        let rec = 1024;
        file[rec..rec + 8].copy_from_slice(&f64::NAN.to_le_bytes()); // NEXT
        file[rec + 8..rec + 16].copy_from_slice(&0f64.to_le_bytes()); // PREV
        file[rec + 16..rec + 24].copy_from_slice(&0f64.to_le_bytes()); // NSUM
        assert!(DafFile::from_bytes(&file).is_err());
    }

    #[test]
    fn test_daf_big_endian_synthetic() {
        // Build a minimal valid big-endian DAF with one summary and no data,
        // to exercise the byte-swap path. Layout mirrors from_bytes' parsing.
        let mut file = vec![0u8; 3 * 1024];
        file[..8].copy_from_slice(b"DAF/SPK ");
        file[8..12].copy_from_slice(&2i32.to_be_bytes()); // ND
        file[12..16].copy_from_slice(&6i32.to_be_bytes()); // NI
        file[76..80].copy_from_slice(&2i32.to_be_bytes()); // FWARD -> record 2
        file[80..84].copy_from_slice(&2i32.to_be_bytes()); // BWARD
        file[84..88].copy_from_slice(&300i32.to_be_bytes()); // FREE
        file[88..96].copy_from_slice(b"BIG-IEEE");

        // Summary record = record 2 (bytes 1024..2048)
        let rec = 1024;
        file[rec..rec + 8].copy_from_slice(&0f64.to_be_bytes()); // NEXT
        file[rec + 8..rec + 16].copy_from_slice(&0f64.to_be_bytes()); // PREV
        file[rec + 16..rec + 24].copy_from_slice(&1f64.to_be_bytes()); // NSUM
        // Summary: doubles [start_et, end_et] then ints [10,0,1,2,257,260]
        file[rec + 24..rec + 32].copy_from_slice(&(-1000f64).to_be_bytes());
        file[rec + 32..rec + 40].copy_from_slice(&1000f64.to_be_bytes());
        let ints: [i32; 6] = [10, 0, 1, 2, 257, 260];
        for (i, v) in ints.iter().enumerate() {
            let off = rec + 40 + i * 4;
            file[off..off + 4].copy_from_slice(&v.to_be_bytes());
        }
        // Name record = record 3, ASCII spaces
        for b in &mut file[2048..2048 + 40] {
            *b = b' ';
        }

        let daf = DafFile::from_bytes(&file).unwrap();
        assert_eq!(daf.nd, 2);
        assert_eq!(daf.ni, 6);
        assert_eq!(daf.summaries.len(), 1);
        assert_eq!(daf.summaries[0].doubles, vec![-1000.0, 1000.0]);
        assert_eq!(daf.summaries[0].ints, vec![10, 0, 1, 2, 257, 260]);
    }
}
