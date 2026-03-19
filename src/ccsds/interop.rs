/*!
 * Brahe interop for CCSDS types.
 *
 * Provides conversion between CCSDS message types and brahe's native
 * trajectory, propagator, and state vector types.
 */

use nalgebra::SVector;

use crate::ccsds::common::CCSDSRefFrame;
use crate::ccsds::oem::OEM;
use crate::trajectories::sorbit_trajectory::SOrbitTrajectory;
use crate::trajectories::traits::{OrbitFrame, OrbitRepresentation, Trajectory};
use crate::utils::errors::BraheError;

/// Map a CCSDS reference frame to a brahe `OrbitFrame`.
///
/// Only inertial and terrestrial frames supported by brahe are mapped.
/// Orbit-relative frames (RTN, TNW, RSW) and exotic frames return an error.
pub fn ccsds_ref_frame_to_orbit_frame(frame: &CCSDSRefFrame) -> Result<OrbitFrame, BraheError> {
    match frame {
        CCSDSRefFrame::EME2000 => Ok(OrbitFrame::EME2000),
        CCSDSRefFrame::J2000 => Ok(OrbitFrame::EME2000),
        CCSDSRefFrame::GCRF => Ok(OrbitFrame::GCRF),
        CCSDSRefFrame::ITRF2000
        | CCSDSRefFrame::ITRF93
        | CCSDSRefFrame::ITRF97
        | CCSDSRefFrame::ITRF2005
        | CCSDSRefFrame::ITRF2008
        | CCSDSRefFrame::ITRF2014 => Ok(OrbitFrame::ITRF),
        CCSDSRefFrame::TEME | CCSDSRefFrame::TOD => Ok(OrbitFrame::ECI),
        _ => Err(BraheError::Error(format!(
            "Cannot map CCSDS frame '{}' to brahe OrbitFrame",
            frame
        ))),
    }
}

impl OEM {
    /// Convert a single OEM segment to a brahe `SOrbitTrajectory`.
    ///
    /// The trajectory contains Cartesian state vectors (position/velocity)
    /// in the reference frame specified by the segment metadata.
    ///
    /// # Arguments
    ///
    /// * `segment_idx` - Index of the segment to convert (0-based)
    ///
    /// # Returns
    ///
    /// * `Result<SOrbitTrajectory, BraheError>` - Trajectory or error
    pub fn segment_to_orbit_trajectory(
        &self,
        segment_idx: usize,
    ) -> Result<SOrbitTrajectory, BraheError> {
        let segment = self.segments.get(segment_idx).ok_or_else(|| {
            BraheError::OutOfBoundsError(format!(
                "OEM segment index {} out of range (have {})",
                segment_idx,
                self.segments.len()
            ))
        })?;

        let orbit_frame = ccsds_ref_frame_to_orbit_frame(&segment.metadata.ref_frame)?;

        let mut traj = SOrbitTrajectory::new(orbit_frame, OrbitRepresentation::Cartesian, None);

        traj.name = Some(segment.metadata.object_name.clone());

        for sv in &segment.states {
            let state = SVector::<f64, 6>::new(
                sv.position[0],
                sv.position[1],
                sv.position[2],
                sv.velocity[0],
                sv.velocity[1],
                sv.velocity[2],
            );
            traj.add(sv.epoch, state);
        }

        Ok(traj)
    }

    /// Convert all segments to brahe `SOrbitTrajectory` objects.
    ///
    /// # Returns
    ///
    /// * `Result<Vec<SOrbitTrajectory>, BraheError>` - One trajectory per segment
    pub fn to_orbit_trajectories(&self) -> Result<Vec<SOrbitTrajectory>, BraheError> {
        (0..self.segments.len())
            .map(|i| self.segment_to_orbit_trajectory(i))
            .collect()
    }
}

impl TryFrom<&OEM> for SOrbitTrajectory {
    type Error = BraheError;

    /// Convert a single-segment OEM to a trajectory.
    ///
    /// Returns an error if the OEM has zero or more than one segment.
    fn try_from(oem: &OEM) -> Result<Self, Self::Error> {
        if oem.segments.len() != 1 {
            return Err(BraheError::Error(format!(
                "TryFrom<&OEM> requires exactly 1 segment, but OEM has {}",
                oem.segments.len()
            )));
        }
        oem.segment_to_orbit_trajectory(0)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::ccsds::oem::OEM;
    use crate::trajectories::traits::Trajectory;

    fn setup_eop() {
        use crate::eop::*;
        let eop = StaticEOPProvider::new();
        set_global_eop_provider(eop);
    }

    #[test]
    fn test_oem_to_trajectory_example4() {
        setup_eop();

        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample4.txt").unwrap();
        let oem = OEM::from_str(&content).unwrap();

        let traj = oem.segment_to_orbit_trajectory(0).unwrap();
        assert_eq!(traj.len(), 3);
        assert_eq!(traj.name.as_deref(), Some("MARS GLOBAL SURVEYOR"));
        assert_eq!(traj.frame, OrbitFrame::EME2000);

        // Verify first state
        let (_epoch, state) = traj.first().unwrap();
        assert!((state[0] - 2789.619 * 1000.0).abs() < 1.0);
        assert!((state[3] - 4.73372 * 1000.0).abs() < 1.0);
    }

    #[test]
    fn test_oem_to_trajectory_example5() {
        setup_eop();

        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample5.txt").unwrap();
        let oem = OEM::from_str(&content).unwrap();

        let traj = oem.segment_to_orbit_trajectory(0).unwrap();
        assert_eq!(traj.len(), 49);
        assert_eq!(traj.frame, OrbitFrame::GCRF);
    }

    #[test]
    fn test_oem_to_trajectories_multi_segment() {
        setup_eop();

        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let oem = OEM::from_str(&content).unwrap();

        let trajs = oem.to_orbit_trajectories().unwrap();
        assert_eq!(trajs.len(), 3);
    }

    #[test]
    fn test_oem_try_from_single_segment() {
        setup_eop();

        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample4.txt").unwrap();
        let oem = OEM::from_str(&content).unwrap();

        let traj = SOrbitTrajectory::try_from(&oem).unwrap();
        assert_eq!(traj.len(), 3);
    }

    #[test]
    fn test_oem_try_from_multi_segment_fails() {
        setup_eop();

        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample1.txt").unwrap();
        let oem = OEM::from_str(&content).unwrap();

        assert!(SOrbitTrajectory::try_from(&oem).is_err());
    }

    #[test]
    fn test_oem_segment_out_of_bounds() {
        setup_eop();

        let content = std::fs::read_to_string("test_assets/ccsds/oem/OEMExample4.txt").unwrap();
        let oem = OEM::from_str(&content).unwrap();

        assert!(oem.segment_to_orbit_trajectory(5).is_err());
    }

    #[test]
    fn test_ccsds_ref_frame_mapping() {
        assert_eq!(
            ccsds_ref_frame_to_orbit_frame(&CCSDSRefFrame::EME2000).unwrap(),
            OrbitFrame::EME2000
        );
        assert_eq!(
            ccsds_ref_frame_to_orbit_frame(&CCSDSRefFrame::GCRF).unwrap(),
            OrbitFrame::GCRF
        );
        assert_eq!(
            ccsds_ref_frame_to_orbit_frame(&CCSDSRefFrame::ITRF2000).unwrap(),
            OrbitFrame::ITRF
        );
        assert_eq!(
            ccsds_ref_frame_to_orbit_frame(&CCSDSRefFrame::J2000).unwrap(),
            OrbitFrame::EME2000
        );
        // Orbit-relative frames should fail
        assert!(ccsds_ref_frame_to_orbit_frame(&CCSDSRefFrame::RTN).is_err());
    }
}
