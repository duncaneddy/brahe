/*!
 * Per-satellite samples joined across SP3 files.
 *
 * Every [`SP3Satellite`] is built here, from one file or from a set of files
 * in time order.
 */

use std::collections::BTreeSet;

use crate::datafiles::sp3::parse::sp3_error;
use crate::datafiles::sp3::satellite::{SP3Sample, SP3Satellite};
use crate::datafiles::sp3::types::{SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
use crate::time::Epoch;
use crate::utils::{BraheError, DOrbitStateProvider};

impl SP3 {
    /// Build an SP3 file by sampling orbit state providers on a regular grid.
    ///
    /// Each provider is sampled with `state_itrf` at `start + k * step` for
    /// `k = 0..=floor((end - start) / step)`. Velocities are stored when
    /// `options.include_velocity` is set. Clocks and clock rates are absent
    /// and are written as the format's bad-clock value unless set with
    /// [`SP3::set_clocks`] and [`SP3::set_clock_rates`]. The file type follows
    /// the satellite systems.
    ///
    /// # Arguments
    ///
    /// * `propagators` - Satellite identifiers paired with the providers that supply their
    ///   states: numerical, Keplerian, and SGP propagators, orbit trajectories, and
    ///   [`SP3Satellite`]
    /// * `start` - First epoch of the file
    /// * `end` - Last epoch to include; the grid stops at the last point not after `end`
    /// * `step` - Epoch spacing. Units: s
    /// * `options` - Header options
    ///
    /// # Returns
    ///
    /// * `Result<SP3, BraheError>` - The file, or an error if `end` precedes `start`, the
    ///   header options or satellites are invalid, or a provider cannot supply a state at a
    ///   grid epoch. A provider error names the satellite and the epoch.
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{SP3, SP3SatelliteId, SP3WriteOptions};
    /// use brahe::eop::{StaticEOPProvider, set_global_eop_provider};
    /// use brahe::propagators::KeplerianPropagator;
    /// use brahe::constants::{AngleFormat, R_EARTH};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::utils::DOrbitStateProvider;
    /// use nalgebra::Vector6;
    ///
    /// set_global_eop_provider(StaticEOPProvider::from_zero());
    /// let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let prop = KeplerianPropagator::from_keplerian(
    ///     t0,
    ///     Vector6::new(R_EARTH + 800e3, 0.001, 98.0, 0.0, 0.0, 0.0),
    ///     AngleFormat::Degrees,
    ///     60.0,
    /// )
    /// .unwrap();
    /// let leo: SP3SatelliteId = "L01".parse().unwrap();
    ///
    /// let sp3 = SP3::from_propagators(
    ///     &[(leo, &prop as &dyn DOrbitStateProvider)],
    ///     t0,
    ///     t0 + 3600.0,
    ///     60.0,
    ///     SP3WriteOptions::default(),
    /// )
    /// .unwrap();
    /// assert_eq!(sp3.num_epochs(), 61);
    /// ```
    pub fn from_propagators(
        propagators: &[(SP3SatelliteId, &dyn DOrbitStateProvider)],
        start: Epoch,
        end: Epoch,
        step: f64,
        options: SP3WriteOptions,
    ) -> Result<Self, BraheError> {
        let include_velocity = options.include_velocity;
        let ids: Vec<SP3SatelliteId> = propagators.iter().map(|(id, _)| *id).collect();
        let mut sp3 = SP3::new(ids, step, options)?;

        let span = end - start;
        if span.is_nan() || span < 0.0 {
            return Err(sp3_error(&format!(
                "end epoch {} precedes start epoch {}",
                end, start
            )));
        }
        let n_steps = (span / step + 1e-9).floor() as usize;
        for k in 0..=n_steps {
            let epoch = start + step * k as f64;
            let index = sp3.push_epoch(epoch)?;
            for (id, provider) in propagators {
                let state = provider.state_itrf(epoch).map_err(|e| {
                    sp3_error(&format!("state of {} at {} unavailable: {}", id, epoch, e))
                })?;
                let mut record = SP3Record::new([state[0], state[1], state[2]]);
                if include_velocity {
                    record = record.with_velocity([state[3], state[4], state[5]]);
                }
                sp3.set_record(*id, index, record)?;
            }
        }
        Ok(sp3)
    }
}

/// Chain every satellite across a set of SP3 files.
///
/// Files without epochs are ignored. The others are sorted by start epoch,
/// then by end epoch, so a one-epoch file precedes a longer file starting at
/// the same epoch whatever the input order.
/// Consecutive files may share their boundary epoch, where the later file's
/// sample is kept when it has one; any further overlap is an error.
///
/// # Arguments
///
/// * `files` - SP3 files in any order
///
/// # Returns
///
/// * `Result<Vec<SP3Satellite>, BraheError>` - One satellite per identifier in the union
///   of the files' satellite lists that has at least one valid sample, in ascending
///   identifier order, or an error naming two overlapping files
pub(crate) fn chain_files(files: &[SP3]) -> Result<Vec<SP3Satellite>, BraheError> {
    let mut ordered: Vec<&SP3> = files.iter().filter(|f| f.num_epochs() > 0).collect();
    ordered.sort_by_key(|f| (f.start_epoch(), f.end_epoch()));
    for pair in ordered.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        if b.start_epoch() < a.end_epoch() && b.start_epoch() != a.end_epoch() {
            return Err(sp3_error(&format!(
                "files overlap: {} file from {} to {} and {} file from {} to {}; consecutive \
                 files may share only their boundary epoch",
                a.header().agency,
                a.start_epoch(),
                a.end_epoch(),
                b.header().agency,
                b.start_epoch(),
                b.end_epoch()
            )));
        }
    }
    let ids: BTreeSet<SP3SatelliteId> = ordered
        .iter()
        .flat_map(|f| f.satellite_ids().iter().copied())
        .collect();
    satellites_from_files(&ordered, ids)
}

/// Build the satellites `ids`, in the given order, skipping those without a
/// valid sample.
///
/// # Arguments
///
/// * `files` - Files sorted by start epoch, overlapping at most at a shared boundary epoch
/// * `ids` - Satellites to build
///
/// # Returns
///
/// * `Result<Vec<SP3Satellite>, BraheError>` - The satellites that have at least one
///   valid sample
pub(crate) fn satellites_from_files(
    files: &[&SP3],
    ids: impl IntoIterator<Item = SP3SatelliteId>,
) -> Result<Vec<SP3Satellite>, BraheError> {
    let mut satellites = Vec::new();
    for id in ids {
        let samples = joined_samples(files, id);
        if !samples.is_empty() {
            satellites.push(SP3Satellite::from_samples(id, samples)?);
        }
    }
    Ok(satellites)
}

/// Build one satellite.
///
/// # Arguments
///
/// * `files` - Files sorted by start epoch, overlapping at most at a shared boundary epoch
/// * `id` - Satellite to build
///
/// # Returns
///
/// * `Result<SP3Satellite, BraheError>` - The satellite, or an error if it has no valid
///   sample in any file
pub(crate) fn satellite_from_files(
    files: &[&SP3],
    id: SP3SatelliteId,
) -> Result<SP3Satellite, BraheError> {
    SP3Satellite::from_samples(id, joined_samples(files, id))
}

/// Valid samples of `id` from every file, in epoch order.
///
/// When a file's first sample falls on the epoch of the previous file's last
/// sample, the later file's sample replaces the earlier one.
fn joined_samples(files: &[&SP3], id: SP3SatelliteId) -> Vec<SP3Sample> {
    let mut samples: Vec<SP3Sample> = Vec::new();
    for file in files {
        let mut next = file.samples(id);
        if let (Some(last), Some(first)) = (samples.last(), next.first())
            && last.epoch == first.epoch
        {
            samples.pop();
        }
        samples.append(&mut next);
    }
    samples
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::constants::AngleFormat;
    use crate::constants::R_EARTH;
    use crate::datafiles::sp3::types::{SP3FileType, SP3Record, SP3TimeSystem, SP3WriteOptions};
    use crate::frames::CelestialFrame;
    use crate::propagators::traits::DStatePropagator;
    use crate::propagators::{
        DNumericalOrbitPropagator, ForceModelConfig, KeplerianPropagator,
        NumericalPropagationConfig,
    };
    use crate::time::{Epoch, TimeSystem};
    use crate::trajectories::dorbit_trajectory::DOrbitTrajectory;
    use crate::trajectories::traits::{OrbitRepresentation, Trajectory};
    use crate::utils::DOrbitStateProvider;
    use crate::utils::testing::{setup_global_test_eop, setup_global_test_gravity_model};
    use approx::assert_abs_diff_eq;
    use nalgebra::DVector;
    use nalgebra::Vector6;
    use serial_test::{parallel, serial};

    const TRUNC: &str = "test_assets/sp3/gbm19500_truncated.sp3";
    const AFTER_DROP: &str = "test_assets/sp3/gbm19500_after_drop.sp3";
    const AFTER_NO_DROP: &str = "test_assets/sp3/gbm19500_after_no_drop.sp3";
    const LARGE_GAP: &str = "test_assets/sp3/gbm19500_large_gap.sp3";

    fn sid(s: &str) -> SP3SatelliteId {
        s.parse().unwrap()
    }

    fn find<'a>(sats: &'a [SP3Satellite], id: &str) -> &'a SP3Satellite {
        sats.iter().find(|s| s.id().to_string() == id).unwrap()
    }

    /// A position-only file at 300 s holding `ids` at `epochs`, with each
    /// satellite moving linearly along x.
    fn linear_file(ids: &[SP3SatelliteId], epochs: &[Epoch], options: SP3WriteOptions) -> SP3 {
        let mut sp3 = SP3::new(ids.to_vec(), 300.0, options.with_velocity(false)).unwrap();
        let origin = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
        for epoch in epochs {
            let index = sp3.push_epoch(*epoch).unwrap();
            for (n, id) in ids.iter().enumerate() {
                let x = 2.0e7 + 1.0e6 * n as f64 + 1000.0 * (*epoch - origin);
                sp3.set_record(*id, index, SP3Record::new([x, 1.0e7, 1.0e7]))
                    .unwrap();
            }
        }
        sp3
    }

    fn gps(minute: u8, second: f64) -> Epoch {
        Epoch::from_datetime(2024, 11, 10, 0, minute, second, 0.0, TimeSystem::GPS)
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_files_shared_boundary_keeps_later() {
        let a = SP3::from_file(TRUNC).unwrap();
        let b = SP3::from_file(AFTER_DROP).unwrap();
        let sats = SP3Satellite::from_files(&[a.clone(), b.clone()]).unwrap();
        let c01 = sats.iter().find(|s| s.id().to_string() == "C01").unwrap();
        assert_eq!(c01.len(), 3);
        let p_b = b
            .record("C01".parse().unwrap(), 0)
            .unwrap()
            .unwrap()
            .position;
        assert_eq!(c01.positions()[1], p_b);
        assert!(!c01.has_gaps());

        let p_a = a.record(sid("C01"), 1).unwrap().unwrap().position;
        assert_ne!(p_a, p_b);
        assert_eq!(c01.epochs()[1], b.epochs()[0]);
        assert_eq!(sats.len(), 87);
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_files_order_and_duplicates() {
        let a = SP3::from_file(TRUNC).unwrap();
        let b = SP3::from_file(AFTER_DROP).unwrap();
        let c = SP3::from_file(AFTER_NO_DROP).unwrap();
        let fwd = SP3Satellite::from_files(&[a.clone(), b.clone(), c.clone()]).unwrap();
        let rev = SP3Satellite::from_files(&[c.clone(), b.clone(), a.clone()]).unwrap();
        assert_eq!(fwd[0].epochs(), rev[0].epochs());
        assert!(
            SP3Satellite::from_files(&[a.clone(), a.clone()])
                .unwrap_err()
                .to_string()
                .contains("overlap")
        );

        assert_eq!(fwd.len(), rev.len());
        for (f, r) in fwd.iter().zip(&rev) {
            assert_eq!(f.id(), r.id());
            assert_eq!(f.epochs(), r.epochs());
            assert_eq!(f.positions(), r.positions());
        }
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_files_overlap_names_both_files() {
        let a = SP3::from_file(TRUNC).unwrap();
        let err = SP3Satellite::from_files(&[a.clone(), a.clone()])
            .unwrap_err()
            .to_string();
        assert!(err.contains("overlap"), "{err}");
        assert_eq!(err.matches("GFZ").count(), 2, "{err}");
        assert!(err.contains(&a.start_epoch().to_string()), "{err}");
        assert!(err.contains(&a.end_epoch().to_string()), "{err}");

        let mut shifted = SP3::new(
            vec![sid("G01")],
            300.0,
            SP3WriteOptions::default().with_agency("ABC"),
        )
        .unwrap();
        shifted.push_epoch(a.start_epoch() + 150.0).unwrap();
        let err = SP3Satellite::from_files(&[shifted, a.clone()])
            .unwrap_err()
            .to_string();
        assert!(err.contains("overlap"), "{err}");
        assert!(err.contains("GFZ") && err.contains("ABC"), "{err}");
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_files_large_gap() {
        let a = SP3::from_file(TRUNC).unwrap();
        let d = SP3::from_file(LARGE_GAP).unwrap();
        let sats = SP3Satellite::from_files(&[a, d]).unwrap();
        let g01 = find(&sats, "G01");
        assert_eq!(g01.len(), 4);
        assert!(g01.has_gaps());
        assert_eq!(g01.segments().len(), 2);

        let err = g01.to_trajectory().unwrap_err().to_string();
        assert!(err.contains("G01") && err.contains("gap"), "{err}");
        assert_eq!(g01.to_trajectory_segments().unwrap().len(), 2);

        let inside_gap = Epoch::from_datetime(2017, 5, 21, 0, 12, 0.0, 0.0, TimeSystem::GPS);
        let err = g01.state_itrf(inside_gap).unwrap_err().to_string();
        assert!(err.contains("G01") && err.contains("gap"), "{err}");
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_files_contiguous() {
        let a = SP3::from_file(TRUNC).unwrap();
        let b = SP3::from_file(AFTER_DROP).unwrap();
        let c = SP3::from_file(AFTER_NO_DROP).unwrap();
        let sats = SP3Satellite::from_files(&[a, b, c]).unwrap();
        assert_eq!(sats.len(), 87);
        let start = Epoch::from_datetime(2017, 5, 21, 0, 0, 0.0, 0.0, TimeSystem::GPS);
        let expected: Vec<Epoch> = (0..4).map(|k| start + 300.0 * k as f64).collect();
        for sat in &sats {
            assert_eq!(sat.len(), 4, "{}", sat.id());
            assert_eq!(sat.epochs(), expected.as_slice(), "{}", sat.id());
            assert!(!sat.has_gaps(), "{}", sat.id());
        }
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_files_missing_satellite_gives_gap() {
        let ids = [sid("G01"), sid("G02")];
        let first = linear_file(
            &ids,
            &[gps(0, 0.0), gps(5, 0.0)],
            SP3WriteOptions::default(),
        );
        let middle = linear_file(
            &ids[..1],
            &[gps(10, 0.0), gps(15, 0.0)],
            SP3WriteOptions::default(),
        );
        let last = linear_file(
            &ids,
            &[gps(20, 0.0), gps(25, 0.0)],
            SP3WriteOptions::default(),
        );
        let sats = SP3Satellite::from_files(&[first, middle, last]).unwrap();
        assert_eq!(sats.len(), 2);

        let g01 = find(&sats, "G01");
        assert_eq!(g01.len(), 6);
        assert!(!g01.has_gaps());

        let g02 = find(&sats, "G02");
        assert_eq!(g02.len(), 4);
        assert!(g02.has_gaps());
        assert_eq!(
            g02.segments(),
            vec![(gps(0, 0.0), gps(5, 0.0)), (gps(20, 0.0), gps(25, 0.0))]
        );
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_files_invalid_later_boundary_keeps_earlier() {
        let id = sid("G01");
        let first = linear_file(
            &[id],
            &[gps(0, 0.0), gps(5, 0.0)],
            SP3WriteOptions::default(),
        );
        let mut second = SP3::new(
            vec![id],
            300.0,
            SP3WriteOptions::default().with_velocity(false),
        )
        .unwrap();
        second.push_epoch(gps(5, 0.0)).unwrap();
        second.push_epoch(gps(10, 0.0)).unwrap();
        second
            .set_record(id, 1, SP3Record::new([3.0e7, 1.0e7, 1.0e7]))
            .unwrap();
        let expected = first.record(id, 1).unwrap().unwrap().position;

        let sats = SP3Satellite::from_files(&[first, second]).unwrap();
        let g01 = find(&sats, "G01");
        assert_eq!(g01.len(), 3);
        assert_eq!(g01.positions()[1], expected);
        assert!(!g01.has_gaps());
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_files_ascending_id_order() {
        let header_order = [sid("C05"), sid("R01"), sid("G02"), sid("E01"), sid("G01")];
        let first = linear_file(
            &header_order,
            &[gps(0, 0.0), gps(5, 0.0)],
            SP3WriteOptions::default(),
        );
        let second = linear_file(
            &[sid("J01"), sid("G03")],
            &[gps(10, 0.0), gps(15, 0.0)],
            SP3WriteOptions::default(),
        );
        let ids: Vec<String> = SP3Satellite::from_files(&[first.clone(), second])
            .unwrap()
            .iter()
            .map(|s| s.id().to_string())
            .collect();
        assert_eq!(ids, ["G01", "G02", "G03", "R01", "E01", "C05", "J01"]);

        let header: Vec<String> = first
            .satellites()
            .unwrap()
            .iter()
            .map(|s| s.id().to_string())
            .collect();
        assert_eq!(header, ["C05", "R01", "G02", "E01", "G01"]);
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_files_empty_inputs() {
        assert!(SP3Satellite::from_files(&[]).unwrap().is_empty());

        let a = SP3::from_file(TRUNC).unwrap();
        let no_epochs = SP3::new(vec![sid("G01")], 300.0, SP3WriteOptions::default()).unwrap();
        let sats = SP3Satellite::from_files(&[no_epochs.clone(), a.clone(), no_epochs]).unwrap();
        assert_eq!(sats.len(), 87);
        assert_eq!(find(&sats, "G01").len(), 2);
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_files_mixed_time_systems_sort_by_instant() {
        let id = sid("G01");
        let gps_file = linear_file(
            &[id],
            &[gps(0, 10.0), gps(5, 10.0), gps(10, 10.0)],
            SP3WriteOptions::default(),
        );
        let utc =
            |minute: u8| Epoch::from_datetime(2024, 11, 10, 0, minute, 0.0, 0.0, TimeSystem::UTC);
        let utc_file = linear_file(
            &[id],
            &[utc(10), utc(15)],
            SP3WriteOptions::default().with_time_system(SP3TimeSystem::UTC),
        );
        assert!(utc_file.start_epoch() > gps_file.end_epoch());

        for files in [
            [gps_file.clone(), utc_file.clone()],
            [utc_file.clone(), gps_file.clone()],
        ] {
            let sats = SP3Satellite::from_files(&files).unwrap();
            let g01 = find(&sats, "G01");
            assert_eq!(g01.len(), 5);
            let epochs = g01.epochs();
            assert_eq!(epochs[2], gps(10, 10.0));
            assert_eq!(epochs[3], utc(10));
            assert!(epochs.windows(2).all(|w| w[1] - w[0] > 0.0));
            assert!(!g01.has_gaps());
        }
    }

    #[test]
    #[parallel]
    fn test_SP3_satellites_matches_from_files_single() {
        let a = SP3::from_file(TRUNC).unwrap();
        let single = a.satellites().unwrap();
        let chained = SP3Satellite::from_files(std::slice::from_ref(&a)).unwrap();
        assert_eq!(single.len(), chained.len());
        for sat in &single {
            let other = chained.iter().find(|s| s.id() == sat.id()).unwrap();
            assert_eq!(sat.epochs(), other.epochs());
            assert_eq!(sat.records(), other.records());
            assert_eq!(sat.velocities(), other.velocities());
            let one = a.satellite(sat.id()).unwrap();
            assert_eq!(one.records(), sat.records());
        }
    }

    /// A file at `interval` holding `id` at `epochs`, moving at 1000 m/s
    /// along x, with velocity records when `velocity` is true.
    fn uniform_file(id: SP3SatelliteId, interval: f64, epochs: &[Epoch], velocity: bool) -> SP3 {
        let options = SP3WriteOptions::default().with_velocity(velocity);
        let mut sp3 = SP3::new(vec![id], interval, options).unwrap();
        for epoch in epochs {
            let index = sp3.push_epoch(*epoch).unwrap();
            let x = 2.0e7 + 1000.0 * (*epoch - gps(0, 0.0));
            let mut record = SP3Record::new([x, 1.0e7, 1.0e7]);
            if velocity {
                record = record.with_velocity([1000.0, 0.0, 0.0]);
            }
            sp3.set_record(id, index, record).unwrap();
        }
        sp3
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_files_equal_starts_independent_of_order() {
        let id = sid("G01");
        let long = linear_file(
            &[id],
            &[gps(0, 0.0), gps(5, 0.0)],
            SP3WriteOptions::default(),
        );
        let mut single = SP3::new(
            vec![id],
            300.0,
            SP3WriteOptions::default().with_velocity(false),
        )
        .unwrap();
        single.push_epoch(gps(0, 0.0)).unwrap();
        single
            .set_record(id, 0, SP3Record::new([3.0e7, 1.0e7, 1.0e7]))
            .unwrap();
        let expected = long.record(id, 0).unwrap().unwrap().position;

        let forward = SP3Satellite::from_files(&[long.clone(), single.clone()]).unwrap();
        let reverse = SP3Satellite::from_files(&[single, long]).unwrap();
        for sats in [&forward, &reverse] {
            let g01 = find(sats, "G01");
            assert_eq!(g01.epochs(), [gps(0, 0.0), gps(5, 0.0)]);
            assert_eq!(g01.positions()[0], expected);
        }
        assert_eq!(
            find(&forward, "G01").records(),
            find(&reverse, "G01").records()
        );
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_files_one_epoch_file_twice() {
        let id = sid("G01");
        let single = linear_file(&[id], &[gps(0, 0.0)], SP3WriteOptions::default());
        let sats = SP3Satellite::from_files(&[single.clone(), single]).unwrap();
        let g01 = find(&sats, "G01");
        assert_eq!(g01.len(), 1);
        assert_eq!(g01.epochs(), [gps(0, 0.0)]);
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_files_velocity_source() {
        let id = sid("G01");
        let first = [gps(0, 0.0), gps(5, 0.0), gps(10, 0.0)];
        let second = [gps(15, 0.0), gps(20, 0.0), gps(25, 0.0)];

        let both = SP3Satellite::from_files(&[
            uniform_file(id, 300.0, &first, true),
            uniform_file(id, 300.0, &second, true),
        ])
        .unwrap();
        let g01 = find(&both, "G01");
        assert!(g01.has_file_velocities());
        assert!(g01.velocities().iter().all(|v| *v == [1000.0, 0.0, 0.0]));

        let mixed = SP3Satellite::from_files(&[
            uniform_file(id, 300.0, &first, true),
            uniform_file(id, 300.0, &second, false),
        ])
        .unwrap();
        let g01 = find(&mixed, "G01");
        assert_eq!(g01.len(), 6);
        assert!(!g01.has_file_velocities());
        assert_eq!(g01.records()[0].velocity, Some([1000.0, 0.0, 0.0]));
        for v in g01.velocities() {
            assert_abs_diff_eq!(v[0], 1000.0, epsilon = 1e-6);
            assert_abs_diff_eq!(v[1], 0.0, epsilon = 1e-6);
            assert_abs_diff_eq!(v[2], 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_files_mixed_intervals_gap_threshold() {
        let id = sid("G01");
        let fine = uniform_file(id, 300.0, &[gps(0, 0.0), gps(5, 0.0), gps(10, 0.0)], false);

        // 900 s after the last 300 s sample: within 1.5 x 900 s.
        let coarse = uniform_file(
            id,
            900.0,
            &[gps(25, 0.0), gps(40, 0.0), gps(55, 0.0)],
            false,
        );
        let sats = SP3Satellite::from_files(&[fine.clone(), coarse]).unwrap();
        let g01 = find(&sats, "G01");
        assert_eq!(g01.len(), 6);
        assert!(!g01.has_gaps());

        // 1500 s after the last 300 s sample: beyond 1.5 x 900 s.
        let late = uniform_file(id, 900.0, &[gps(35, 0.0), gps(50, 0.0)], false);
        let sats = SP3Satellite::from_files(&[fine, late]).unwrap();
        let g01 = find(&sats, "G01");
        assert_eq!(
            g01.segments(),
            vec![(gps(0, 0.0), gps(10, 0.0)), (gps(35, 0.0), gps(50, 0.0))]
        );
    }

    /// A 24 h position-only file at 900 s starting at `start`, sampled from
    /// `propagator`.
    fn keplerian_day(start: Epoch, id: SP3SatelliteId, propagator: &KeplerianPropagator) -> SP3 {
        let mut sp3 = SP3::new(
            vec![id],
            900.0,
            SP3WriteOptions::default().with_velocity(false),
        )
        .unwrap();
        for k in 0..=96 {
            let epoch = start + 900.0 * k as f64;
            let index = sp3.push_epoch(epoch).unwrap();
            let x = propagator.state_itrf(epoch).unwrap();
            sp3.set_record(id, index, SP3Record::new([x[0], x[1], x[2]]))
                .unwrap();
        }
        sp3
    }

    /// The 15 000 km orbit is sampled coarsely enough at 900 s that a
    /// one-sided fit at the end of a file is far less accurate than a
    /// centered one.
    #[test]
    #[serial]
    fn test_SP3Satellite_from_files_boundary_velocity_uses_centered_fit() {
        setup_global_test_eop();
        let id = sid("G01");
        let day1 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
        let day2 = day1 + 86400.0;
        let propagator = KeplerianPropagator::from_keplerian(
            day1,
            Vector6::new(15_000e3, 0.01, 55.0, 30.0, 0.0, 45.0),
            AngleFormat::Degrees,
            60.0,
        )
        .unwrap();
        let first = keplerian_day(day1, id, &propagator);
        let second = keplerian_day(day2, id, &propagator);

        let truth = propagator.state_itrf(day2).unwrap();
        let error = |v: [f64; 3]| {
            ((v[0] - truth[3]).powi(2) + (v[1] - truth[4]).powi(2) + (v[2] - truth[5]).powi(2))
                .sqrt()
        };

        let single = first.satellite(id).unwrap();
        assert_eq!(*single.epochs().last().unwrap(), day2);
        let err_single = error(*single.velocities().last().unwrap());

        let sats = SP3Satellite::from_files(&[first, second]).unwrap();
        let chained = find(&sats, "G01");
        assert_eq!(chained.len(), 193);
        assert!(!chained.has_gaps());
        let boundary = chained.epochs().iter().position(|e| *e == day2).unwrap();
        let err_chained = error(chained.velocities()[boundary]);

        assert!(
            err_chained < err_single,
            "chained {err_chained} m/s, single {err_single} m/s"
        );
        assert!(
            err_chained < 0.1 * err_single,
            "chained {err_chained} m/s, single {err_single} m/s"
        );
    }

    fn leo_propagator(epoch: Epoch, raan: f64) -> KeplerianPropagator {
        KeplerianPropagator::from_keplerian(
            epoch,
            Vector6::new(R_EARTH + 800e3, 0.001, 98.0, raan, 0.0, 0.0),
            AngleFormat::Degrees,
            60.0,
        )
        .unwrap()
    }

    fn day_start() -> Epoch {
        Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS)
    }

    fn two_leo_sp3(options: SP3WriteOptions) -> (SP3, KeplerianPropagator, KeplerianPropagator) {
        setup_global_test_eop();
        let t0 = day_start();
        let a = leo_propagator(t0, 0.0);
        let b = leo_propagator(t0, 40.0);
        let sp3 = SP3::from_propagators(
            &[
                (sid("L01"), &a as &dyn DOrbitStateProvider),
                (sid("L02"), &b as &dyn DOrbitStateProvider),
            ],
            t0,
            t0 + 86400.0,
            30.0,
            options,
        )
        .unwrap();
        (sp3, a, b)
    }

    #[test]
    #[serial]
    fn test_SP3_from_propagators_keplerian() {
        let (sp3, a, b) = two_leo_sp3(SP3WriteOptions::default());
        assert_eq!(sp3.header().file_type, SP3FileType::LEO);
        assert_eq!(sp3.num_epochs(), 2881);
        assert_eq!(sp3.header().epoch_interval, 30.0);
        assert!(sp3.header().has_velocity);
        assert_eq!(sp3.epochs()[0], day_start());
        assert_eq!(sp3.start_epoch(), day_start());

        for (id, prop) in [("L01", &a), ("L02", &b)] {
            for k in [0, 1, 1000, 2880] {
                let epoch = day_start() + 30.0 * k as f64;
                let truth = prop.state_itrf(epoch).unwrap();
                let record = sp3.record(sid(id), k).unwrap().unwrap();
                for i in 0..3 {
                    assert_abs_diff_eq!(record.position[i], truth[i], epsilon = 1e-9);
                    assert_abs_diff_eq!(record.velocity.unwrap()[i], truth[3 + i], epsilon = 1e-9);
                }
                assert_eq!(record.clock, None);
                assert_eq!(record.clock_rate, None);
            }
        }
    }

    #[test]
    #[serial]
    fn test_SP3_from_propagators_mixed_file_type() {
        setup_global_test_eop();
        let t0 = day_start();
        let a = leo_propagator(t0, 0.0);
        let b = leo_propagator(t0, 40.0);
        let sp3 = SP3::from_propagators(
            &[
                (sid("G01"), &a as &dyn DOrbitStateProvider),
                (sid("L01"), &b as &dyn DOrbitStateProvider),
            ],
            t0,
            t0 + 600.0,
            30.0,
            SP3WriteOptions::default(),
        )
        .unwrap();
        assert_eq!(sp3.header().file_type, SP3FileType::Mixed);
    }

    #[test]
    #[serial]
    fn test_SP3_from_propagators_clocks_absent_and_round_trip() {
        let (sp3, a, _) = two_leo_sp3(SP3WriteOptions::default());
        for k in 0..sp3.num_epochs() {
            for id in ["L01", "L02"] {
                assert_eq!(sp3.record(sid(id), k).unwrap().unwrap().clock, None);
            }
        }
        let text = sp3.to_string().unwrap();
        assert!(text.contains("999999.999999"));

        let parsed = SP3::from_str(&text).unwrap();
        let sat = parsed.satellite(sid("L01")).unwrap();
        assert_eq!(sat.len(), 2881);
        for k in [0, 7, 1440, 2880] {
            let epoch = day_start() + 30.0 * k as f64;
            let truth = a.state_itrf(epoch).unwrap();
            let state = sat.state_itrf(epoch).unwrap();
            for i in 0..3 {
                // The file stores positions in km with six decimals, a 1 mm grid.
                assert_abs_diff_eq!(state[i], truth[i], epsilon = 5.1e-4);
            }
        }
    }

    #[test]
    #[serial]
    fn test_SP3_from_propagators_grid_includes_end_within_tolerance() {
        setup_global_test_eop();
        let t0 = day_start();
        let a = leo_propagator(t0, 0.0);
        let providers = [(sid("L01"), &a as &dyn DOrbitStateProvider)];
        let build = |end: Epoch| {
            SP3::from_propagators(&providers, t0, end, 30.0, SP3WriteOptions::default()).unwrap()
        };
        assert_eq!(build(t0 + 100.0).num_epochs(), 4);
        assert_eq!(build(t0 + 90.0).num_epochs(), 4);
        assert_eq!(build(t0).num_epochs(), 1);

        let err =
            SP3::from_propagators(&providers, t0, t0 - 30.0, 30.0, SP3WriteOptions::default())
                .unwrap_err()
                .to_string();
        assert!(err.contains("precedes"), "{err}");
        assert!(
            SP3::from_propagators(&providers, t0, t0 + 60.0, 0.0, SP3WriteOptions::default())
                .is_err()
        );
    }

    #[test]
    #[serial]
    fn test_SP3_from_propagators_position_only_rejects_clock_rates() {
        let (mut sp3, _, _) = two_leo_sp3(SP3WriteOptions::default().with_velocity(false));
        assert!(!sp3.header().has_velocity);
        assert!(
            sp3.record(sid("L01"), 3)
                .unwrap()
                .unwrap()
                .velocity
                .is_none()
        );
        assert!(!sp3.to_string().unwrap().contains("\nV"));

        let rates = vec![Some(1e-12); sp3.num_epochs()];
        let err = sp3
            .set_clock_rates(sid("L01"), &rates)
            .unwrap_err()
            .to_string();
        assert!(err.contains("velocity"), "{err}");
    }

    #[test]
    #[serial]
    fn test_SP3_set_clocks_and_rates() {
        let (mut sp3, _, _) = two_leo_sp3(SP3WriteOptions::default());
        let n = sp3.num_epochs();
        let clocks: Vec<Option<f64>> = (0..n)
            .map(|k| {
                if k == 5 {
                    None
                } else {
                    Some(1e-5 + 1e-9 * k as f64)
                }
            })
            .collect();
        let rates: Vec<Option<f64>> = (0..n).map(|k| Some(1e-12 * k as f64)).collect();
        sp3.set_clocks(sid("L01"), &clocks).unwrap();
        sp3.set_clock_rates(sid("L01"), &rates).unwrap();

        for k in [0, 5, 6, n - 1] {
            let record = sp3.record(sid("L01"), k).unwrap().unwrap();
            assert_eq!(record.clock, clocks[k]);
            assert_eq!(record.clock_rate, rates[k]);
        }
        assert_eq!(sp3.record(sid("L02"), 6).unwrap().unwrap().clock, None);

        let parsed = SP3::from_str(&sp3.to_string().unwrap()).unwrap();
        let record = parsed.record(sid("L01"), 6).unwrap().unwrap();
        assert_abs_diff_eq!(record.clock.unwrap(), clocks[6].unwrap(), epsilon = 1e-12);
        assert_abs_diff_eq!(
            record.clock_rate.unwrap(),
            rates[6].unwrap(),
            epsilon = 1e-16
        );
        assert_eq!(parsed.record(sid("L01"), 5).unwrap().unwrap().clock, None);
    }

    #[test]
    #[serial]
    fn test_SP3_set_clocks_errors() {
        let (mut sp3, _, _) = two_leo_sp3(SP3WriteOptions::default());
        let n = sp3.num_epochs();

        let err = sp3
            .set_clocks(sid("L01"), &vec![Some(1e-5); n - 1])
            .unwrap_err()
            .to_string();
        assert!(err.contains("epochs"), "{err}");
        assert!(
            sp3.set_clock_rates(sid("L01"), &vec![Some(1e-12); n + 1])
                .is_err()
        );
        assert!(sp3.set_clocks(sid("G05"), &vec![None; n]).is_err());

        let mut sparse = SP3::new(vec![sid("G01")], 300.0, SP3WriteOptions::default()).unwrap();
        sparse.push_epoch(gps(0, 0.0)).unwrap();
        sparse.push_epoch(gps(5, 0.0)).unwrap();
        sparse
            .set_record(sid("G01"), 0, SP3Record::new([2.0e7, 1.0e7, 1.0e7]))
            .unwrap();
        let err = sparse
            .set_clocks(sid("G01"), &[Some(1e-5), Some(1e-5)])
            .unwrap_err()
            .to_string();
        assert!(err.contains("G01") && err.contains("no record"), "{err}");
        assert_eq!(sparse.record(sid("G01"), 0).unwrap().unwrap().clock, None);

        sparse.set_clocks(sid("G01"), &[Some(1e-5), None]).unwrap();
        assert_eq!(
            sparse.record(sid("G01"), 0).unwrap().unwrap().clock,
            Some(1e-5)
        );
        assert!(
            sparse
                .set_clock_rates(sid("G01"), &[None, Some(1e-12)])
                .is_err()
        );
    }

    #[test]
    #[serial]
    fn test_SP3_from_propagators_trajectory_outside_span_errors() {
        setup_global_test_eop();
        let t0 = day_start();
        let prop = leo_propagator(t0, 0.0);
        let mut traj = DOrbitTrajectory::new(
            6,
            CelestialFrame::ITRF,
            OrbitRepresentation::Cartesian,
            None,
        )
        .unwrap();
        for k in 0..=720 {
            let epoch = t0 + 60.0 * k as f64;
            let s = prop.state_itrf(epoch).unwrap();
            traj.add(epoch, DVector::from_column_slice(s.as_slice()))
                .unwrap();
        }

        let covered = SP3::from_propagators(
            &[(sid("L01"), &traj as &dyn DOrbitStateProvider)],
            t0,
            t0 + 43200.0,
            60.0,
            SP3WriteOptions::default(),
        )
        .unwrap();
        assert_eq!(covered.num_epochs(), 721);

        let err = SP3::from_propagators(
            &[(sid("L01"), &traj as &dyn DOrbitStateProvider)],
            t0,
            t0 + 86400.0,
            60.0,
            SP3WriteOptions::default(),
        )
        .unwrap_err()
        .to_string();
        assert!(err.contains("L01"), "{err}");
        let first_missing: Epoch = t0 + 43260.0;
        assert!(err.contains(&first_missing.to_string()), "{err}");
    }

    #[test]
    #[serial]
    fn test_SP3_from_propagators_resamples_sp3_satellite() {
        let (sp3, _, _) = two_leo_sp3(SP3WriteOptions::default());
        let sat = sp3.satellite(sid("L02")).unwrap();
        let t0 = day_start();
        let resampled = SP3::from_propagators(
            &[(sid("L02"), &sat as &dyn DOrbitStateProvider)],
            t0 + 3600.0,
            t0 + 7200.0,
            45.0,
            SP3WriteOptions::default().with_agency("DEMO"),
        )
        .unwrap();
        assert_eq!(resampled.num_epochs(), 81);
        assert_eq!(resampled.header().agency, "DEMO");
        assert_eq!(resampled.header().epoch_interval, 45.0);
        for k in [0, 40, 80] {
            let epoch = t0 + 3600.0 + 45.0 * k as f64;
            let truth = sat.state_itrf(epoch).unwrap();
            let record = resampled.record(sid("L02"), k).unwrap().unwrap();
            for i in 0..3 {
                assert_abs_diff_eq!(record.position[i], truth[i], epsilon = 1e-9);
                assert_abs_diff_eq!(record.velocity.unwrap()[i], truth[3 + i], epsilon = 1e-9);
            }
        }
    }

    #[test]
    #[serial]
    fn test_SP3_from_propagators_numerical_propagator() {
        setup_global_test_eop();
        setup_global_test_gravity_model();
        let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::UTC);
        let state = DVector::from_vec(vec![R_EARTH + 700e3, 0.0, 0.0, 0.0, 6000.0, 4500.0]);
        let mut prop = DNumericalOrbitPropagator::new(
            t0,
            state,
            NumericalPropagationConfig::default(),
            ForceModelConfig::earth_gravity(),
            None,
            None,
            None,
            None,
        )
        .unwrap();
        prop.propagate_to(t0 + 7200.0).unwrap();

        let sp3 = SP3::from_propagators(
            &[(sid("L05"), &prop as &dyn DOrbitStateProvider)],
            t0,
            t0 + 7200.0,
            60.0,
            SP3WriteOptions::default(),
        )
        .unwrap();
        assert_eq!(sp3.num_epochs(), 121);
        for k in [0, 1, 60, 120] {
            let epoch = t0 + 60.0 * k as f64;
            let truth = prop.state_itrf(epoch).unwrap();
            let record = sp3.record(sid("L05"), k).unwrap().unwrap();
            for i in 0..3 {
                assert_abs_diff_eq!(record.position[i], truth[i], epsilon = 1e-9);
                assert_abs_diff_eq!(record.velocity.unwrap()[i], truth[3 + i], epsilon = 1e-9);
            }
        }
    }
}
