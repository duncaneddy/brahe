/*!
 * Native binary PCK (Planetary Constants Kernel) reader for high-precision
 * body orientation, e.g. the lunar principal-axes frames (MOON_PA_DE440).
 *
 * Binary PCK type 2 segments store Chebyshev coefficients for the three
 * 3-1-3 Euler angles (phi, delta, w) rotating the reference frame (ICRF)
 * to the body-fixed frame.
 *
 * # References
 * - NAIF PCK Required Reading:
 *   <https://naif.jpl.nasa.gov/pub/naif/toolkit_docs/C/req/pck.html>
 */

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

use nalgebra::{Matrix3, Vector3};

use crate::attitude::RotationMatrix;
use crate::constants::AngleFormat;
use crate::utils::BraheError;

use super::daf::DafFile;
use super::segments::ChebyshevSegment;

/// A loaded binary PCK kernel with in-memory Chebyshev coefficients.
#[derive(Debug)]
pub struct BPCK {
    /// Segments grouped by body-frame class ID, in file order.
    segments_by_frame: HashMap<i32, Vec<Arc<ChebyshevSegment>>>,
}

impl BPCK {
    /// Load a binary PCK kernel from a file.
    ///
    /// # Arguments
    /// - `path`: Path to a `.bpc` file (PCK type 2 supported)
    ///
    /// # Returns
    /// - Loaded kernel, or an error naming any unsupported segment type
    pub fn from_file(path: &Path) -> Result<Self, BraheError> {
        let daf = DafFile::from_file(path)?;
        Self::from_daf(daf)
    }

    /// Load a binary PCK kernel from an in-memory byte buffer.
    ///
    /// # Arguments
    /// - `bytes`: Raw bytes of a binary SPICE kernel
    ///
    /// # Returns
    /// - Loaded kernel, or an error naming any unsupported segment type
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, BraheError> {
        Self::from_daf(DafFile::from_bytes(bytes)?)
    }

    pub(crate) fn from_daf(daf: DafFile) -> Result<Self, BraheError> {
        if daf.id_word != "DAF/PCK" {
            return Err(BraheError::IoError(format!(
                "Not a binary PCK kernel: ID word is '{}', expected 'DAF/PCK'",
                daf.id_word
            )));
        }
        let mut segments_by_frame: HashMap<i32, Vec<Arc<ChebyshevSegment>>> = HashMap::new();
        for summary in &daf.summaries {
            let seg = Arc::new(ChebyshevSegment::from_pck_summary(&daf, summary)?);
            segments_by_frame.entry(seg.target).or_default().push(seg);
        }
        Ok(BPCK { segments_by_frame })
    }

    /// Body-frame class IDs present in this kernel.
    pub(crate) fn frame_ids(&self) -> Vec<i32> {
        self.segments_by_frame.keys().copied().collect()
    }

    /// Select the last segment for `frame_id` covering `et` (SPICE
    /// convention: the most recently loaded/last-listed segment takes
    /// precedence).
    fn segment(&self, frame_id: i32, et: f64) -> Result<&Arc<ChebyshevSegment>, BraheError> {
        let segs = self.segments_by_frame.get(&frame_id).ok_or_else(|| {
            BraheError::Error(format!(
                "Frame class ID {} not found in loaded binary PCK data",
                frame_id
            ))
        })?;
        segs.iter().rev().find(|s| s.covers(et)).ok_or_else(|| {
            BraheError::Error(format!(
                "Epoch ET {} outside PCK coverage for frame class ID {}",
                et, frame_id
            ))
        })
    }

    /// 3-1-3 Euler angles and rates of the body-fixed frame relative to the
    /// segment reference frame (ICRF for DE440-era kernels).
    ///
    /// # Arguments
    /// - `frame_id`: Body-frame class ID (e.g. 31008 for MOON_PA_DE440)
    /// - `et`: TDB seconds past J2000
    ///
    /// # Returns
    /// - `(angles, rates)`: `angles = [phi, delta, w]` in [rad];
    ///   `rates` are their time derivatives in [rad/s]
    pub fn euler_angles(
        &self,
        frame_id: i32,
        et: f64,
    ) -> Result<(Vector3<f64>, Vector3<f64>), BraheError> {
        let seg = self.segment(frame_id, et)?;
        let (angles, rates) = seg.state(et)?;
        Ok((angles, rates))
    }

    /// Rotation matrix from the segment reference frame (ICRF) to the
    /// body-fixed frame: `R = Rz(w) · Rx(delta) · Rz(phi)`.
    ///
    /// # Arguments
    /// - `frame_id`: Body-frame class ID (e.g. 31008 for MOON_PA_DE440)
    /// - `et`: TDB seconds past J2000
    ///
    /// # Returns
    /// - 3x3 rotation matrix (ICRF to body-fixed). Dimensionless.
    pub fn rotation_matrix(&self, frame_id: i32, et: f64) -> Result<Matrix3<f64>, BraheError> {
        let (a, _) = self.euler_angles(frame_id, et)?;
        Ok(RotationMatrix::Rz(a[2], AngleFormat::Radians).to_matrix()
            * RotationMatrix::Rx(a[1], AngleFormat::Radians).to_matrix()
            * RotationMatrix::Rz(a[0], AngleFormat::Radians).to_matrix())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use approx::assert_abs_diff_eq;

    use super::*;

    /// Build a minimal little-endian binary PCK with one type-2 segment for
    /// frame 31006 rel frame 1, one record covering et in [0, 1000], with
    /// linear angles: phi = 0.1 + 0.2s, delta = 0.3, w = 0.4 + 0.5s
    /// (s = (et-500)/500).
    fn synthetic_bpck_bytes() -> Vec<u8> {
        let degree = 1usize;
        let rsize = 2 + 3 * (degree + 1); // 8
        // Data: record [MID, RADIUS, phi0, phi1, d0, d1, w0, w1] + trailer
        let data: Vec<f64> = vec![
            500.0,
            500.0, // MID, RADIUS
            0.1,
            0.2, // phi: a0 + a1*T1(s)
            0.3,
            0.0, // delta
            0.4,
            0.5, // w
            0.0,
            1000.0,
            rsize as f64,
            1.0, // INIT, INTLEN, RSIZE, N
        ];

        let mut file = vec![0u8; 4 * 1024];
        file[..8].copy_from_slice(b"DAF/PCK ");
        file[8..12].copy_from_slice(&2i32.to_le_bytes()); // ND
        file[12..16].copy_from_slice(&5i32.to_le_bytes()); // NI
        file[76..80].copy_from_slice(&2i32.to_le_bytes()); // FWARD
        file[80..84].copy_from_slice(&2i32.to_le_bytes()); // BWARD
        file[84..88].copy_from_slice(&500i32.to_le_bytes()); // FREE
        file[88..96].copy_from_slice(b"LTL-IEEE");

        // Summary record (record 2): NEXT=0, PREV=0, NSUM=1
        let rec = 1024;
        file[rec + 16..rec + 24].copy_from_slice(&1f64.to_le_bytes());
        // doubles: start_et, end_et
        file[rec + 24..rec + 32].copy_from_slice(&0f64.to_le_bytes());
        file[rec + 32..rec + 40].copy_from_slice(&1000f64.to_le_bytes());
        // ints: [frame_class_id, reference_frame, type, start_addr, end_addr]
        // Data record = record 4 => word address 385 ((4-1)*128 + 1)
        let start_addr = 385i32;
        let end_addr = start_addr + data.len() as i32 - 1;
        for (i, v) in [31006i32, 1, 2, start_addr, end_addr].iter().enumerate() {
            let off = rec + 40 + i * 4;
            file[off..off + 4].copy_from_slice(&v.to_le_bytes());
        }
        // Name record (record 3): spaces
        for b in &mut file[2048..2048 + 40] {
            *b = b' ';
        }
        // Data (record 4)
        for (i, v) in data.iter().enumerate() {
            let off = 3 * 1024 + i * 8;
            file[off..off + 8].copy_from_slice(&v.to_le_bytes());
        }
        file
    }

    #[test]
    fn test_bpck_euler_angles_exact() {
        let bpck = BPCK::from_bytes(&synthetic_bpck_bytes()).unwrap();
        // et=750 -> s=0.5: phi=0.2, delta=0.3, w=0.65
        let (angles, rates) = bpck.euler_angles(31006, 750.0).unwrap();
        assert_abs_diff_eq!(angles[0], 0.1 + 0.2 * 0.5, epsilon = 1e-12);
        assert_abs_diff_eq!(angles[1], 0.3, epsilon = 1e-12);
        assert_abs_diff_eq!(angles[2], 0.4 + 0.5 * 0.5, epsilon = 1e-12);
        // Rates: d(angle)/det = a1 / RADIUS
        assert_abs_diff_eq!(rates[0], 0.2 / 500.0, epsilon = 1e-15);
        assert_abs_diff_eq!(rates[1], 0.0, epsilon = 1e-15);
        assert_abs_diff_eq!(rates[2], 0.5 / 500.0, epsilon = 1e-15);
    }

    #[test]
    fn test_bpck_rotation_matrix_is_313() {
        let bpck = BPCK::from_bytes(&synthetic_bpck_bytes()).unwrap();
        let (angles, _) = bpck.euler_angles(31006, 750.0).unwrap();
        let r = bpck.rotation_matrix(31006, 750.0).unwrap();
        // Orthonormal, det = +1
        let rtr = r.transpose() * r;
        assert_abs_diff_eq!(
            (rtr - nalgebra::Matrix3::identity()).norm(),
            0.0,
            epsilon = 1e-12
        );
        assert_abs_diff_eq!(r.determinant(), 1.0, epsilon = 1e-12);
        // Matches explicit Rz(w)*Rx(delta)*Rz(phi)
        use crate::attitude::RotationMatrix;
        use crate::constants::AngleFormat;
        let expected = RotationMatrix::Rz(angles[2], AngleFormat::Radians).to_matrix()
            * RotationMatrix::Rx(angles[1], AngleFormat::Radians).to_matrix()
            * RotationMatrix::Rz(angles[0], AngleFormat::Radians).to_matrix();
        assert_abs_diff_eq!((r - expected).norm(), 0.0, epsilon = 1e-12);
    }

    #[test]
    fn test_bpck_unknown_frame_error() {
        let bpck = BPCK::from_bytes(&synthetic_bpck_bytes()).unwrap();
        let err = bpck.euler_angles(99999, 750.0).unwrap_err();
        assert!(format!("{}", err).contains("99999"));
    }

    #[test]
    fn test_bpck_out_of_coverage_error() {
        let bpck = BPCK::from_bytes(&synthetic_bpck_bytes()).unwrap();
        // Frame exists but et=2000 is outside the [0, 1000] coverage
        let err = bpck.euler_angles(31006, 2000.0).unwrap_err();
        let msg = format!("{}", err);
        assert!(msg.contains("coverage") && msg.contains("31006"));
    }

    #[test]
    fn test_bpck_frame_ids() {
        let bpck = BPCK::from_bytes(&synthetic_bpck_bytes()).unwrap();
        assert_eq!(bpck.frame_ids(), vec![31006]);
    }

    #[test]
    fn test_bpck_rejects_garbage_bytes() {
        assert!(BPCK::from_bytes(&[0u8; 100]).is_err()); // too short
        let mut junk = vec![0u8; 2048];
        junk[..8].copy_from_slice(b"NOTADAF ");
        assert!(BPCK::from_bytes(&junk).is_err()); // bad ID word
    }

    #[test]
    fn test_bpck_rejects_spk_file() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("test_assets")
            .join("de440s.bsp");
        if !path.exists() {
            return;
        }
        let err = BPCK::from_file(&path).unwrap_err();
        assert!(format!("{}", err).contains("DAF/SPK"));
    }
}
