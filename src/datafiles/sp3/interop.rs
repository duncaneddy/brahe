/*!
 * Per-satellite samples joined across SP3 files.
 *
 * Every [`SP3Satellite`] is built here, from one file or from a set of files
 * in time order.
 */

use std::collections::BTreeSet;

use crate::datafiles::sp3::parse::sp3_error;
use crate::datafiles::sp3::satellite::{SP3Sample, SP3Satellite};
use crate::datafiles::sp3::types::{SP3, SP3SatelliteId};
use crate::utils::BraheError;

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
    use crate::datafiles::sp3::types::{SP3Record, SP3TimeSystem, SP3WriteOptions};
    use crate::propagators::KeplerianPropagator;
    use crate::time::{Epoch, TimeSystem};
    use crate::utils::DOrbitStateProvider;
    use crate::utils::testing::setup_global_test_eop;
    use approx::assert_abs_diff_eq;
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
}
