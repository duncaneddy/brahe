/*!
 * SP3 files built from orbit trajectories.
 */

use crate::datafiles::sp3::parse::sp3_error;
use crate::datafiles::sp3::types::{GRID_TOLERANCE, SP3, SP3Header, SP3Record, SP3SatelliteId};
use crate::time::Epoch;
use crate::trajectories::dorbit_trajectory::DOrbitTrajectory;
use crate::utils::{BraheError, DOrbitStateProvider};

impl SP3 {
    /// Build an SP3 file from orbit trajectories.
    ///
    /// Without `step`, the file holds the epochs the trajectories store: the
    /// trajectories must store the same epochs, to within 1 ms, on a regular
    /// grid whose interval is the spacing of the first two. With `step`, each
    /// trajectory is interpolated at `start + k * step` over the span all
    /// trajectories cover, from the latest first epoch to the earliest last
    /// epoch. States are converted to ITRF with `state_itrf`. Velocities are
    /// stored when `header.has_velocity` is set. Clocks and clock rates are
    /// absent and are written as the format's bad-clock value unless set with
    /// [`SP3::set_clocks`] and [`SP3::set_clock_rates`]. The file type follows
    /// the satellite systems.
    ///
    /// A propagator supplies the trajectory it has propagated, and an
    /// [`SP3Satellite`](crate::datafiles::sp3::SP3Satellite) supplies one with
    /// `to_trajectory`.
    ///
    /// # Arguments
    ///
    /// * `trajectories` - Satellite identifiers paired with their trajectories
    /// * `step` - Epoch spacing to resample at, or `None` to keep the stored epochs. Units: s
    /// * `header` - Header of the file; its epoch interval is set from the epochs written
    ///
    /// # Returns
    ///
    /// * `Result<SP3, BraheError>` - The file, or an error if no trajectory is given, a
    ///   trajectory is empty, the satellites are invalid, the trajectories share no span,
    ///   the stored epochs differ between trajectories, number fewer than two, or are off
    ///   a regular grid, or a state is unavailable or not finite. A state error names the satellite and
    ///   the epoch.
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{SP3, SP3SatelliteId, SP3Header};
    /// use brahe::eop::{StaticEOPProvider, set_global_eop_provider};
    /// use brahe::propagators::KeplerianPropagator;
    /// use brahe::propagators::traits::SStatePropagator;
    /// use brahe::constants::{AngleFormat, R_EARTH};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use nalgebra::Vector6;
    ///
    /// set_global_eop_provider(StaticEOPProvider::from_zero());
    /// let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let mut prop = KeplerianPropagator::from_keplerian(
    ///     t0,
    ///     Vector6::new(R_EARTH + 800e3, 0.001, 98.0, 0.0, 0.0, 0.0),
    ///     AngleFormat::Degrees,
    ///     60.0,
    /// )
    /// .unwrap();
    /// prop.propagate_to(t0 + 3600.0).unwrap();
    /// let leo: SP3SatelliteId = "L01".parse().unwrap();
    ///
    /// // The 61 states the propagator stored, one per 60 s step.
    /// let sp3 = SP3::from_trajectories(&[(leo, &prop.trajectory)], None, SP3Header::default())
    ///     .unwrap();
    /// assert_eq!(sp3.num_epochs(), 61);
    ///
    /// // The same span resampled at 300 s.
    /// let sp3 = SP3::from_trajectories(&[(leo, &prop.trajectory)], Some(300.0), SP3Header::default())
    ///     .unwrap();
    /// assert_eq!(sp3.num_epochs(), 13);
    /// ```
    pub fn from_trajectories(
        trajectories: &[(SP3SatelliteId, &DOrbitTrajectory)],
        step: Option<f64>,
        header: SP3Header,
    ) -> Result<Self, BraheError> {
        let Some((first_id, first)) = trajectories.first() else {
            return Err(sp3_error("no trajectory given"));
        };
        if let Some((id, _)) = trajectories.iter().find(|(_, t)| t.epochs.is_empty()) {
            return Err(sp3_error(&format!("trajectory of {} has no states", id)));
        }

        let (epochs, interval) = match step {
            Some(step) => (common_grid(trajectories, step)?, step),
            None => {
                if first.epochs.len() < 2 {
                    return Err(sp3_error(&format!(
                        "trajectory of {} stores one epoch, which gives no epoch interval; \
                         pass a step to resample",
                        first_id
                    )));
                }
                for (id, trajectory) in &trajectories[1..] {
                    let same = trajectory.epochs.len() == first.epochs.len()
                        && trajectory
                            .epochs
                            .iter()
                            .zip(&first.epochs)
                            .all(|(a, b)| (*a - *b).abs() <= GRID_TOLERANCE);
                    if !same {
                        return Err(sp3_error(&format!(
                            "trajectories of {} and {} store different epochs; pass a step to \
                             resample",
                            first_id, id
                        )));
                    }
                }
                let interval = first.epochs[1] - first.epochs[0];
                if let Some(off) = first.epochs.iter().find(|e| {
                    let offset = **e - first.epochs[0];
                    ((offset / interval).round() * interval - offset).abs() > GRID_TOLERANCE
                }) {
                    return Err(sp3_error(&format!(
                        "stored epoch {} is off the {} s grid of the first two epochs; pass a \
                         step to resample",
                        off, interval
                    )));
                }
                (first.epochs.clone(), interval)
            }
        };

        let include_velocity = header.has_velocity;
        let ids: Vec<SP3SatelliteId> = trajectories.iter().map(|(id, _)| *id).collect();
        let mut sp3 = SP3::new(ids, interval, header)?;
        let mut records = Vec::with_capacity(trajectories.len());
        for epoch in epochs {
            records.clear();
            for (id, trajectory) in trajectories {
                let state = trajectory.state_itrf(epoch).map_err(|e| {
                    sp3_error(&format!("state of {} at {} unavailable: {}", id, epoch, e))
                })?;
                let mut record = SP3Record::new([state[0], state[1], state[2]]);
                if include_velocity {
                    record = record.with_velocity([state[3], state[4], state[5]]);
                }
                records.push((*id, record));
            }
            sp3.set_epoch(epoch, &records)?;
        }
        Ok(sp3)
    }
}

/// Epochs `start + k * step` over the span every trajectory covers.
///
/// # Arguments
///
/// * `trajectories` - Satellite identifiers paired with non-empty trajectories
/// * `step` - Epoch spacing. Units: s
///
/// # Returns
///
/// * `Result<Vec<Epoch>, BraheError>` - The epochs from the latest first epoch up to the
///   last grid point not after the earliest last epoch, or an error if the step is not
///   positive and finite or the trajectories share no span
fn common_grid(
    trajectories: &[(SP3SatelliteId, &DOrbitTrajectory)],
    step: f64,
) -> Result<Vec<Epoch>, BraheError> {
    if !(step.is_finite() && step > 0.0) {
        return Err(sp3_error(&format!(
            "step must be positive and finite, got {}",
            step
        )));
    }
    let mut start = trajectories[0].1.epochs[0];
    let mut end = trajectories[0].1.epochs[trajectories[0].1.epochs.len() - 1];
    for (_, trajectory) in &trajectories[1..] {
        start = start.max(trajectory.epochs[0]);
        end = end.min(trajectory.epochs[trajectory.epochs.len() - 1]);
    }
    let span = end - start;
    if span < 0.0 {
        return Err(sp3_error(&format!(
            "trajectories share no span: the latest starts at {} and the earliest ends at {}",
            start, end
        )));
    }
    let n_steps = (span / step + 1e-9).floor() as usize;
    Ok((0..=n_steps).map(|k| start + step * k as f64).collect())
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::constants::AngleFormat;
    use crate::constants::R_EARTH;
    use crate::datafiles::sp3::satellite::SP3Satellite;
    use crate::datafiles::sp3::types::{SP3FileType, SP3Header, SP3Record, SP3TimeSystem};
    use crate::frames::CelestialFrame;
    use crate::propagators::traits::{DStatePropagator, SStatePropagator};
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
    fn linear_file(ids: &[SP3SatelliteId], epochs: &[Epoch], options: SP3Header) -> SP3 {
        linear_file_at(300.0, ids, epochs, options)
    }

    /// Start and end epoch of each stretch of a satellite between its gaps.
    fn spans(sat: &SP3Satellite) -> Vec<(Epoch, Epoch)> {
        let mut spans = Vec::new();
        let mut start = sat.start_epoch();
        for (before, after) in sat.gaps() {
            spans.push((start, before));
            start = after;
        }
        spans.push((start, sat.end_epoch()));
        spans
    }

    /// ITRF velocity a satellite provides at each of its samples.
    fn velocities(sat: &SP3Satellite) -> Vec<[f64; 3]> {
        sat.epochs()
            .iter()
            .map(|&epoch| {
                let state = sat.state_itrf(epoch).unwrap();
                [state[3], state[4], state[5]]
            })
            .collect()
    }

    /// A position-only file at `interval` seconds holding `ids` at `epochs`,
    /// with each satellite moving linearly along x.
    fn linear_file_at(
        interval: f64,
        ids: &[SP3SatelliteId],
        epochs: &[Epoch],
        options: SP3Header,
    ) -> SP3 {
        let header = SP3Header {
            has_velocity: false,
            ..options
        };
        let mut sp3 = SP3::new(ids.to_vec(), interval, header).unwrap();
        let origin = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
        for epoch in epochs {
            sp3.set_epoch(*epoch, &[]).unwrap();
            for (n, id) in ids.iter().enumerate() {
                let x = 2.0e7 + 1.0e6 * n as f64 + 1000.0 * (*epoch - origin);
                sp3.set_record(*id, *epoch, SP3Record::new([x, 1.0e7, 1.0e7]))
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
            .record("C01".parse().unwrap(), b.epochs()[0])
            .unwrap()
            .unwrap()
            .position;
        assert_eq!(c01.positions()[1], p_b);
        assert_eq!(spans(c01).len(), 1);

        let p_a = a
            .record(sid("C01"), a.epochs()[1])
            .unwrap()
            .unwrap()
            .position;
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
        assert!(err.contains(&a.start_epoch().unwrap().to_string()), "{err}");
        assert!(err.contains(&a.end_epoch().unwrap().to_string()), "{err}");

        let mut shifted = SP3::new(
            vec![sid("G01")],
            300.0,
            SP3Header {
                agency: "ABC".to_string(),
                ..Default::default()
            },
        )
        .unwrap();
        shifted
            .set_epoch(a.start_epoch().unwrap() + 150.0, &[])
            .unwrap();
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
        assert_eq!(spans(g01).len(), 2);

        let inside_gap = Epoch::from_datetime(2017, 5, 21, 0, 12, 0.0, 0.0, TimeSystem::GPS);
        assert!(g01.state_itrf(inside_gap).is_ok());
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
            assert_eq!(spans(sat).len(), 1, "{}", sat.id());
        }
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_files_missing_satellite_gives_gap() {
        let ids = [sid("G01"), sid("G02")];
        let first = linear_file(&ids, &[gps(0, 0.0), gps(5, 0.0)], SP3Header::default());
        let middle = linear_file(
            &ids[..1],
            &[gps(10, 0.0), gps(15, 0.0)],
            SP3Header::default(),
        );
        let last = linear_file(&ids, &[gps(20, 0.0), gps(25, 0.0)], SP3Header::default());
        let sats = SP3Satellite::from_files(&[first, middle, last]).unwrap();
        assert_eq!(sats.len(), 2);

        let g01 = find(&sats, "G01");
        assert_eq!(g01.len(), 6);
        assert_eq!(spans(g01).len(), 1);

        let g02 = find(&sats, "G02");
        assert_eq!(g02.len(), 4);
        assert_eq!(spans(g02).len(), 2);
        assert_eq!(
            spans(g02),
            vec![(gps(0, 0.0), gps(5, 0.0)), (gps(20, 0.0), gps(25, 0.0))]
        );
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_files_invalid_later_boundary_keeps_earlier() {
        let id = sid("G01");
        let first = linear_file(&[id], &[gps(0, 0.0), gps(5, 0.0)], SP3Header::default());
        let mut second = SP3::new(
            vec![id],
            300.0,
            SP3Header {
                has_velocity: false,
                ..Default::default()
            },
        )
        .unwrap();
        second.set_epoch(gps(5, 0.0), &[]).unwrap();
        second.set_epoch(gps(10, 0.0), &[]).unwrap();
        second
            .set_record(
                id,
                second.epochs()[1],
                SP3Record::new([3.0e7, 1.0e7, 1.0e7]),
            )
            .unwrap();
        let expected = first
            .record(id, first.epochs()[1])
            .unwrap()
            .unwrap()
            .position;

        let sats = SP3Satellite::from_files(&[first, second]).unwrap();
        let g01 = find(&sats, "G01");
        assert_eq!(g01.len(), 3);
        assert_eq!(g01.positions()[1], expected);
        assert_eq!(spans(g01).len(), 1);
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_files_ascending_id_order() {
        let header_order = [sid("C05"), sid("R01"), sid("G02"), sid("E01"), sid("G01")];
        let first = linear_file(
            &header_order,
            &[gps(0, 0.0), gps(5, 0.0)],
            SP3Header::default(),
        );
        let second = linear_file(
            &[sid("J01"), sid("G03")],
            &[gps(10, 0.0), gps(15, 0.0)],
            SP3Header::default(),
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
        let no_epochs = SP3::new(vec![sid("G01")], 300.0, SP3Header::default()).unwrap();
        let sats = SP3Satellite::from_files(&[no_epochs.clone(), a.clone(), no_epochs]).unwrap();
        assert_eq!(sats.len(), 87);
        assert_eq!(find(&sats, "G01").len(), 2);
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_files_mixed_time_systems_sort_by_instant() {
        let id = sid("G01");
        let gps_file = linear_file_at(
            30.0,
            &[id],
            &[gps(9, 10.0), gps(9, 40.0), gps(10, 10.0)],
            SP3Header::default(),
        );
        let utc = |minute: u8, second: f64| {
            Epoch::from_datetime(2024, 11, 10, 0, minute, second, 0.0, TimeSystem::UTC)
        };
        let utc_file = linear_file_at(
            30.0,
            &[id],
            &[utc(10, 8.0), utc(10, 38.0)],
            SP3Header {
                time_system: SP3TimeSystem::UTC,
                ..Default::default()
            },
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
            assert_eq!(epochs[3], utc(10, 8.0));
            assert!(epochs.windows(2).all(|w| w[1] - w[0] > 0.0));
            assert_eq!(spans(g01).len(), 1);
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
            assert_eq!(velocities(sat), velocities(other));
            let one = a.satellite(sat.id()).unwrap();
            assert_eq!(one.records(), sat.records());
        }
    }

    /// A file at `interval` holding `id` at `epochs`, moving at 1000 m/s
    /// along x, with velocity records when `velocity` is true.
    fn uniform_file(id: SP3SatelliteId, interval: f64, epochs: &[Epoch], velocity: bool) -> SP3 {
        let options = SP3Header {
            has_velocity: velocity,
            ..Default::default()
        };
        let mut sp3 = SP3::new(vec![id], interval, options).unwrap();
        for epoch in epochs {
            sp3.set_epoch(*epoch, &[]).unwrap();
            let x = 2.0e7 + 1000.0 * (*epoch - gps(0, 0.0));
            let mut record = SP3Record::new([x, 1.0e7, 1.0e7]);
            if velocity {
                record = record.with_velocity([1000.0, 0.0, 0.0]);
            }
            sp3.set_record(id, *epoch, record).unwrap();
        }
        sp3
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_files_equal_starts_independent_of_order() {
        let id = sid("G01");
        let long = linear_file(&[id], &[gps(0, 0.0), gps(5, 0.0)], SP3Header::default());
        let mut single = SP3::new(
            vec![id],
            300.0,
            SP3Header {
                has_velocity: false,
                ..Default::default()
            },
        )
        .unwrap();
        single.set_epoch(gps(0, 0.0), &[]).unwrap();
        single
            .set_record(
                id,
                single.epochs()[0],
                SP3Record::new([3.0e7, 1.0e7, 1.0e7]),
            )
            .unwrap();
        let expected = long.record(id, long.epochs()[0]).unwrap().unwrap().position;

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
        let single = linear_file(&[id], &[gps(0, 0.0)], SP3Header::default());
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
        assert!(velocities(g01).iter().all(|v| *v == [1000.0, 0.0, 0.0]));

        let mixed = SP3Satellite::from_files(&[
            uniform_file(id, 300.0, &first, true),
            uniform_file(id, 300.0, &second, false),
        ])
        .unwrap();
        let g01 = find(&mixed, "G01");
        assert_eq!(g01.len(), 6);
        assert_eq!(g01.records()[0].velocity, Some([1000.0, 0.0, 0.0]));
        for v in velocities(g01) {
            assert_abs_diff_eq!(v[0], 1000.0, epsilon = 1e-6);
            assert_abs_diff_eq!(v[1], 0.0, epsilon = 1e-6);
            assert_abs_diff_eq!(v[2], 0.0, epsilon = 1e-6);
        }
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_files_rejects_different_intervals() {
        let id = sid("G01");
        let fine = uniform_file(id, 300.0, &[gps(0, 0.0), gps(5, 0.0), gps(10, 0.0)], false);
        let coarse = uniform_file(
            id,
            900.0,
            &[gps(25, 0.0), gps(40, 0.0), gps(55, 0.0)],
            false,
        );
        let err = SP3Satellite::from_files(&[fine, coarse])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("different epoch intervals: 300 s and 900 s"),
            "{err}"
        );
    }

    /// A 24 h position-only file at 900 s starting at `start`, sampled from
    /// `propagator`.
    fn keplerian_day(start: Epoch, id: SP3SatelliteId, propagator: &KeplerianPropagator) -> SP3 {
        let mut sp3 = SP3::new(
            vec![id],
            900.0,
            SP3Header {
                has_velocity: false,
                ..Default::default()
            },
        )
        .unwrap();
        for k in 0..=96 {
            let epoch = start + 900.0 * k as f64;
            sp3.set_epoch(epoch, &[]).unwrap();
            let x = propagator.state_itrf(epoch).unwrap();
            sp3.set_record(id, epoch, SP3Record::new([x[0], x[1], x[2]]))
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
        let err_single = error(*velocities(&single).last().unwrap());

        let sats = SP3Satellite::from_files(&[first, second]).unwrap();
        let chained = find(&sats, "G01");
        assert_eq!(chained.len(), 193);
        assert_eq!(spans(chained).len(), 1);
        let boundary = chained.epochs().iter().position(|e| *e == day2).unwrap();
        let err_chained = error(velocities(chained)[boundary]);

        assert!(
            err_chained < err_single,
            "chained {err_chained} m/s, single {err_single} m/s"
        );
        assert!(
            err_chained < 0.1 * err_single,
            "chained {err_chained} m/s, single {err_single} m/s"
        );
    }

    /// A position-only 900 s file of `id` sampled from `propagator` at
    /// `start + 900 k + shift(k)` for `k = 0..=96`.
    fn shifted_keplerian_day(
        start: Epoch,
        id: SP3SatelliteId,
        propagator: &KeplerianPropagator,
        shift: impl Fn(usize) -> f64,
    ) -> SP3 {
        let mut sp3 = SP3::new(
            vec![id],
            900.0,
            SP3Header {
                has_velocity: false,
                ..Default::default()
            },
        )
        .unwrap();
        for k in 0..=96 {
            let epoch = start + 900.0 * k as f64 + shift(k);
            sp3.set_epoch(epoch, &[]).unwrap();
            let x = propagator.state_itrf(epoch).unwrap();
            sp3.set_record(id, epoch, SP3Record::new([x[0], x[1], x[2]]))
                .unwrap();
        }
        sp3
    }

    /// Position errors of the chained G01 450 s before and after `boundary`.
    fn boundary_errors(
        files: &[SP3],
        propagator: &KeplerianPropagator,
        boundary: Epoch,
    ) -> [f64; 2] {
        let sats = SP3Satellite::from_files(files).unwrap();
        let chained = find(&sats, "G01");
        assert_eq!(chained.len(), 193);
        assert_eq!(spans(chained).len(), 1);
        [-450.0, 450.0].map(|dt| {
            let q = boundary + dt;
            let got = chained.state_itrf(q).unwrap();
            let truth = propagator.state_itrf(q).unwrap();
            (0..3)
                .map(|i| (got[i] - truth[i]).powi(2))
                .sum::<f64>()
                .sqrt()
        })
    }

    #[test]
    #[serial]
    fn test_SP3Satellite_from_files_boundary_within_grid_tolerance() {
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
        let exact = boundary_errors(
            &[
                keplerian_day(day1, id, &propagator),
                keplerian_day(day2, id, &propagator),
            ],
            &propagator,
            day2,
        );
        let assert_close = |errors: [f64; 2]| {
            for (got, reference) in errors.iter().zip(exact) {
                assert!(
                    (got - reference).abs() < 1e-3,
                    "error {got} m, exact boundary {reference} m"
                );
            }
        };

        let text = keplerian_day(day1, id, &propagator)
            .to_string()
            .unwrap()
            .replacen(
                "*  2024 11 11  0  0  0.00000000",
                "*  2024 11 10 23 59 59.99999999",
                1,
            );
        let day1_early_end = SP3::from_str(&text).unwrap();
        assert_abs_diff_eq!(
            day2 - day1_early_end.end_epoch().unwrap(),
            1e-8,
            epsilon = 1e-9
        );
        assert_close(boundary_errors(
            &[day1_early_end, keplerian_day(day2, id, &propagator)],
            &propagator,
            day2,
        ));

        let day2_late = shifted_keplerian_day(day2, id, &propagator, |_| 1e-5);
        assert_close(boundary_errors(
            &[keplerian_day(day1, id, &propagator), day2_late],
            &propagator,
            day2,
        ));

        let day2_early = shifted_keplerian_day(day2, id, &propagator, |_| -1e-8);
        assert_close(boundary_errors(
            &[day2_early, keplerian_day(day1, id, &propagator)],
            &propagator,
            day2,
        ));
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_files_overlap_beyond_grid_tolerance() {
        let id = sid("G01");
        let t0 = gps(0, 0.0);
        let mut a = SP3::new(vec![id], 300.0, SP3Header::default()).unwrap();
        let mut b = a.clone();
        for k in 0..3 {
            a.set_epoch(t0 + 300.0 * k as f64, &[]).unwrap();
            b.set_epoch(t0 + 600.0 - 2e-3 + 300.0 * k as f64, &[])
                .unwrap();
        }
        let err = SP3Satellite::from_files(&[a, b]).unwrap_err().to_string();
        assert!(err.contains("overlap"), "{err}");
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_files_rejects_near_duplicate_boundary() {
        let id = sid("G01");
        let a = linear_file(
            &[id],
            &[gps(0, 0.0), gps(5, 0.0), gps(10, 0.0)],
            SP3Header::default(),
        );
        let late: Vec<Epoch> = [10, 15, 20].iter().map(|m| gps(*m, 0.0) + 2e-3).collect();
        let b = linear_file(&[id], &late, SP3Header::default());
        let err = SP3Satellite::from_files(&[a, b]).unwrap_err().to_string();
        assert!(err.contains("G01"), "{err}");
        assert!(err.contains(&gps(10, 0.0).to_string()), "{err}");
        assert!(err.contains(&late[0].to_string()), "{err}");
        assert!(err.contains("half the 300 s epoch interval"), "{err}");
    }

    fn leo_propagator(epoch: Epoch, raan: f64) -> KeplerianPropagator {
        KeplerianPropagator::from_keplerian(
            epoch,
            Vector6::new(R_EARTH + 800e3, 0.001, 98.0, raan, 0.0, 0.0),
            AngleFormat::Degrees,
            30.0,
        )
        .unwrap()
    }

    /// ITRF trajectory holding the states of a LEO propagator at `offsets`
    /// seconds after the start of the day.
    fn leo_trajectory(raan: f64, offsets: &[f64]) -> DOrbitTrajectory {
        setup_global_test_eop();
        let t0 = day_start();
        let prop = leo_propagator(t0, raan);
        let mut trajectory = DOrbitTrajectory::new(
            6,
            CelestialFrame::ITRF,
            OrbitRepresentation::Cartesian,
            None,
        )
        .unwrap();
        for offset in offsets {
            let state = prop.state_itrf(t0 + *offset).unwrap();
            trajectory
                .add(t0 + *offset, DVector::from_column_slice(state.as_slice()))
                .unwrap();
        }
        trajectory
    }

    fn day_start() -> Epoch {
        Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS)
    }

    fn two_leo_sp3(options: SP3Header) -> (SP3, KeplerianPropagator, KeplerianPropagator) {
        setup_global_test_eop();
        let t0 = day_start();
        let mut a = leo_propagator(t0, 0.0);
        let mut b = leo_propagator(t0, 40.0);
        a.propagate_to(t0 + 86400.0).unwrap();
        b.propagate_to(t0 + 86400.0).unwrap();
        let sp3 = SP3::from_trajectories(
            &[(sid("L01"), &a.trajectory), (sid("L02"), &b.trajectory)],
            None,
            options,
        )
        .unwrap();
        (sp3, a, b)
    }

    #[test]
    #[serial]
    fn test_SP3_from_trajectories_stored_epochs() {
        let (sp3, a, b) = two_leo_sp3(SP3Header::default());
        assert_eq!(sp3.header.file_type, SP3FileType::LEO);
        assert_eq!(sp3.num_epochs(), 2881);
        assert_eq!(sp3.header.epoch_interval, 30.0);
        assert!(sp3.header.has_velocity);
        assert_eq!(sp3.epochs()[0], day_start());
        assert_eq!(sp3.start_epoch(), Some(day_start()));

        for (id, prop) in [("L01", &a), ("L02", &b)] {
            for k in [0, 1, 1000, 2880] {
                let epoch = day_start() + 30.0 * k as f64;
                let truth = prop.state_itrf(epoch).unwrap();
                let record = sp3.record(sid(id), sp3.epochs()[k]).unwrap().unwrap();
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
    fn test_SP3_from_trajectories_mixed_file_type() {
        let offsets = [0.0, 30.0, 60.0];
        let (a, b) = (
            leo_trajectory(0.0, &offsets),
            leo_trajectory(40.0, &offsets),
        );
        let sp3 = SP3::from_trajectories(
            &[(sid("G01"), &a), (sid("L01"), &b)],
            None,
            SP3Header::default(),
        )
        .unwrap();
        assert_eq!(sp3.header.file_type, SP3FileType::Mixed);
        assert_eq!(sp3.num_epochs(), 3);
    }

    #[test]
    #[serial]
    fn test_SP3_from_trajectories_clocks_absent_and_round_trip() {
        let (sp3, a, _) = two_leo_sp3(SP3Header::default());
        for k in 0..sp3.num_epochs() {
            for id in ["L01", "L02"] {
                assert_eq!(
                    sp3.record(sid(id), sp3.epochs()[k]).unwrap().unwrap().clock,
                    None
                );
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
    fn test_SP3_from_trajectories_resampled_grid() {
        let t0 = day_start();
        let header = SP3Header::default;
        let uneven = leo_trajectory(0.0, &[0.0, 30.0, 60.0, 90.0, 100.0]);
        let build = |trajectory: &DOrbitTrajectory, step: f64| {
            SP3::from_trajectories(&[(sid("L01"), trajectory)], Some(step), header())
        };
        let sp3 = build(&uneven, 30.0).unwrap();
        assert_eq!(sp3.epochs(), &[t0, t0 + 30.0, t0 + 60.0, t0 + 90.0]);
        assert_eq!(sp3.header.epoch_interval, 30.0);
        assert_eq!(build(&uneven, 50.0).unwrap().num_epochs(), 3);
        assert_eq!(
            build(&leo_trajectory(0.0, &[0.0]), 30.0)
                .unwrap()
                .num_epochs(),
            1
        );
        for step in [0.0, -30.0, f64::NAN] {
            let err = build(&uneven, step).unwrap_err().to_string();
            assert!(err.contains("step must be positive and finite"), "{err}");
        }

        // Resampling covers the span every trajectory covers.
        let early = leo_trajectory(0.0, &[0.0, 30.0, 60.0, 90.0, 120.0]);
        let late = leo_trajectory(40.0, &[60.0, 90.0, 120.0, 150.0, 180.0]);
        let both = [(sid("L01"), &early), (sid("L02"), &late)];
        let sp3 = SP3::from_trajectories(&both, Some(30.0), header()).unwrap();
        assert_eq!(sp3.epochs(), &[t0 + 60.0, t0 + 90.0, t0 + 120.0]);

        let apart = leo_trajectory(40.0, &[150.0, 180.0]);
        let disjoint = [(sid("L01"), &early), (sid("L02"), &apart)];
        let err = SP3::from_trajectories(&disjoint, Some(30.0), header())
            .unwrap_err()
            .to_string();
        assert!(err.contains("share no span"), "{err}");
    }

    #[test]
    #[serial]
    fn test_SP3_from_trajectories_stored_epoch_errors() {
        let header = SP3Header::default;
        let stored = |trajectories: &[(SP3SatelliteId, &DOrbitTrajectory)]| {
            SP3::from_trajectories(trajectories, None, header())
                .unwrap_err()
                .to_string()
        };
        assert!(stored(&[]).contains("no trajectory given"));

        let empty = DOrbitTrajectory::new(
            6,
            CelestialFrame::ITRF,
            OrbitRepresentation::Cartesian,
            None,
        )
        .unwrap();
        let err = stored(&[(sid("L01"), &empty)]);
        assert!(err.contains("L01 has no states"), "{err}");
        assert!(SP3::from_trajectories(&[(sid("L01"), &empty)], Some(30.0), header()).is_err());

        let err = stored(&[(sid("L01"), &leo_trajectory(0.0, &[0.0]))]);
        assert!(
            err.contains("one epoch") && err.contains("pass a step"),
            "{err}"
        );

        let uneven = leo_trajectory(0.0, &[0.0, 30.0, 60.0, 100.0]);
        let err = stored(&[(sid("L01"), &uneven)]);
        let off: Epoch = day_start() + 100.0;
        assert!(err.contains("off the 30 s grid"), "{err}");
        assert!(err.contains(&off.to_string()), "{err}");

        let a = leo_trajectory(0.0, &[0.0, 30.0, 60.0]);
        let shifted = leo_trajectory(40.0, &[0.0, 30.0, 90.0]);
        let shorter = leo_trajectory(40.0, &[0.0, 30.0]);
        for other in [&shifted, &shorter] {
            let err = stored(&[(sid("L01"), &a), (sid("L02"), other)]);
            assert!(err.contains("L01 and L02 store different epochs"), "{err}");
        }

        // A skipped grid point is kept as a skipped epoch.
        let skipped = leo_trajectory(0.0, &[0.0, 30.0, 90.0]);
        let sp3 = SP3::from_trajectories(&[(sid("L01"), &skipped)], None, header()).unwrap();
        assert_eq!(sp3.num_epochs(), 3);
        assert_eq!(sp3.header.epoch_interval, 30.0);
    }

    #[test]
    #[serial]
    fn test_SP3_from_trajectories_position_only_rejects_clock_rates() {
        let (mut sp3, _, _) = two_leo_sp3(SP3Header {
            has_velocity: false,
            ..Default::default()
        });
        assert!(!sp3.header.has_velocity);
        assert!(
            sp3.record(sid("L01"), sp3.epochs()[3])
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
        let (mut sp3, _, _) = two_leo_sp3(SP3Header::default());
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
            let record = sp3.record(sid("L01"), sp3.epochs()[k]).unwrap().unwrap();
            assert_eq!(record.clock, clocks[k]);
            assert_eq!(record.clock_rate, rates[k]);
        }
        assert_eq!(
            sp3.record(sid("L02"), sp3.epochs()[6])
                .unwrap()
                .unwrap()
                .clock,
            None
        );

        let parsed = SP3::from_str(&sp3.to_string().unwrap()).unwrap();
        let record = parsed
            .record(sid("L01"), parsed.epochs()[6])
            .unwrap()
            .unwrap();
        assert_abs_diff_eq!(record.clock.unwrap(), clocks[6].unwrap(), epsilon = 1e-12);
        assert_abs_diff_eq!(
            record.clock_rate.unwrap(),
            rates[6].unwrap(),
            epsilon = 1e-16
        );
        assert_eq!(
            parsed
                .record(sid("L01"), parsed.epochs()[5])
                .unwrap()
                .unwrap()
                .clock,
            None
        );
    }

    #[test]
    #[serial]
    fn test_SP3_set_clocks_errors() {
        let (mut sp3, _, _) = two_leo_sp3(SP3Header::default());
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

        let mut sparse = SP3::new(vec![sid("G01")], 300.0, SP3Header::default()).unwrap();
        sparse.set_epoch(gps(0, 0.0), &[]).unwrap();
        sparse.set_epoch(gps(5, 0.0), &[]).unwrap();
        sparse
            .set_record(
                sid("G01"),
                sparse.epochs()[0],
                SP3Record::new([2.0e7, 1.0e7, 1.0e7]),
            )
            .unwrap();
        let err = sparse
            .set_clocks(sid("G01"), &[Some(1e-5), Some(1e-5)])
            .unwrap_err()
            .to_string();
        assert!(err.contains("G01") && err.contains("no record"), "{err}");
        assert_eq!(
            sparse
                .record(sid("G01"), sparse.epochs()[0])
                .unwrap()
                .unwrap()
                .clock,
            None
        );

        sparse.set_clocks(sid("G01"), &[Some(1e-5), None]).unwrap();
        assert_eq!(
            sparse
                .record(sid("G01"), sparse.epochs()[0])
                .unwrap()
                .unwrap()
                .clock,
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
    fn test_SP3_from_trajectories_resamples_sp3_satellite() {
        let (sp3, _, _) = two_leo_sp3(SP3Header::default());
        let sat = sp3.satellite(sid("L02")).unwrap();
        let t0 = day_start();
        let resampled = SP3::from_trajectories(
            &[(sid("L02"), &sat.to_trajectory().unwrap())],
            Some(45.0),
            SP3Header {
                agency: "DEMO".to_string(),
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!(resampled.num_epochs(), 1921);
        assert_eq!(resampled.header.agency, "DEMO");
        assert_eq!(resampled.header.epoch_interval, 45.0);
        for k in [100, 960, 1800] {
            let epoch = t0 + 45.0 * k as f64;
            let truth = sat.state_itrf(epoch).unwrap();
            let record = resampled.record(sid("L02"), epoch).unwrap().unwrap();
            for i in 0..3 {
                assert_abs_diff_eq!(record.position[i], truth[i], epsilon = 1e-6);
                assert_abs_diff_eq!(record.velocity.unwrap()[i], truth[3 + i], epsilon = 1e-6);
            }
        }
    }

    #[test]
    #[serial]
    fn test_SP3_from_trajectories_numerical_propagator() {
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

        let sp3 = SP3::from_trajectories(
            &[(sid("L05"), prop.trajectory())],
            Some(60.0),
            SP3Header::default(),
        )
        .unwrap();
        assert_eq!(sp3.num_epochs(), 121);
        for k in [0, 1, 60, 120] {
            let epoch = t0 + 60.0 * k as f64;
            let truth = prop.state_itrf(epoch).unwrap();
            let record = sp3.record(sid("L05"), sp3.epochs()[k]).unwrap().unwrap();
            for i in 0..3 {
                assert_abs_diff_eq!(record.position[i], truth[i], epsilon = 1e-9);
                assert_abs_diff_eq!(record.velocity.unwrap()[i], truth[3 + i], epsilon = 1e-9);
            }
        }
    }

    #[test]
    #[serial]
    fn test_SP3_from_trajectories_non_finite_state_errors() {
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
        let good = prop.state_itrf(t0).unwrap();
        traj.add(t0, DVector::from_column_slice(good.as_slice()))
            .unwrap();
        traj.add(t0 + 60.0, DVector::from_element(6, f64::NAN))
            .unwrap();

        let err = SP3::from_trajectories(&[(sid("L01"), &traj)], None, SP3Header::default())
            .unwrap_err()
            .to_string();
        let second: Epoch = t0 + 60.0;
        assert!(err.contains("L01") && err.contains("non-finite"), "{err}");
        assert!(err.contains(&second.to_string()), "{err}");
    }
}
