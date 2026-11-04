/*!
 * Per-satellite view of SP3 precise orbit data.
 *
 * [`SP3Satellite`] holds the valid samples of one satellite, the velocity at
 * each sample, and one ITRF trajectory per contiguous run of samples.
 */

use std::ops::Range;

use nalgebra::DVector;

use crate::datafiles::sp3::parse::sp3_error;
use crate::datafiles::sp3::types::{SP3Record, SP3SatelliteId};
use crate::frames::CelestialFrame;
use crate::math::interpolation::{InterpolationConfig, InterpolationMethod};
use crate::time::Epoch;
use crate::trajectories::dorbit_trajectory::DOrbitTrajectory;
use crate::trajectories::traits::{OrbitRepresentation, Trajectory};
use crate::utils::{BraheError, Identifiable};

/// Default degree of the Lagrange polynomials used by [`SP3Satellite`].
///
/// It sets the default interpolation method of an [`SP3Satellite`] and of the
/// trajectories it returns, and the fixed degree of the polynomial fit that
/// derives velocities for files without velocity records. A degree of 9 fits
/// 10 samples.
///
/// # Examples
///
/// ```
/// use brahe::datafiles::sp3::SP3_DEFAULT_LAGRANGE_DEGREE;
///
/// assert_eq!(SP3_DEFAULT_LAGRANGE_DEGREE, 9);
/// ```
pub const SP3_DEFAULT_LAGRANGE_DEGREE: usize = 9;

/// Spacing, as a multiple of the epoch interval, above which two consecutive
/// samples belong to different runs.
const GAP_FACTOR: f64 = 1.5;

/// One valid sample of a satellite, as collected from one or more SP3 files.
#[derive(Debug, Clone)]
pub(crate) struct SP3Sample {
    /// Epoch of the sample
    pub(crate) epoch: Epoch,
    /// Record of the satellite at the epoch; its position is never the zero position
    pub(crate) record: SP3Record,
    /// Epoch interval of the file the sample comes from. Units: s
    pub(crate) interval: f64,
    /// Whether the file the sample comes from carries velocity records
    pub(crate) has_file_velocity: bool,
}

/// Orbit and clock samples of one satellite from SP3 data.
///
/// Only epochs where the satellite has a valid position are kept; every data
/// accessor returns one entry per sample, aligned with [`SP3Satellite::epochs`].
/// Positions and velocities are in the file's Earth-fixed frame (ITRF), in m
/// and m/s.
///
/// Velocities come from the file when every sample has one. Otherwise each
/// sample's velocity is the derivative of a Lagrange polynomial of degree
/// [`SP3_DEFAULT_LAGRANGE_DEGREE`] fitted through neighboring samples of the
/// same contiguous run, as centered on the sample as the run allows. A run
/// shorter than the fit uses all its samples. A sample that forms a run on its
/// own has no derived velocity and reports `NaN`.
///
/// Two consecutive samples spaced more than 1.5 times the larger of their
/// epoch intervals apart start a new run, so the satellite has a gap between
/// them. Each run of at least two samples becomes one ITRF
/// [`DOrbitTrajectory`] named after the satellite.
///
/// # Examples
///
/// ```
/// use brahe::datafiles::sp3::{SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
/// use brahe::time::{Epoch, TimeSystem};
///
/// let id: SP3SatelliteId = "G01".parse().unwrap();
/// let options = SP3WriteOptions::default().with_velocity(false);
/// let mut sp3 = SP3::new(vec![id], 900.0, options).unwrap();
/// let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
/// for k in 0..4 {
///     let index = sp3.push_epoch(t0 + 900.0 * k as f64).unwrap();
///     let x = 2.0e7 + 900.0 * k as f64;
///     sp3.set_record(id, index, SP3Record::new([x, 1.0e7, 1.0e7])).unwrap();
/// }
///
/// let sat = sp3.satellite(id).unwrap();
/// assert_eq!(sat.len(), 4);
/// assert!(!sat.has_file_velocities());
/// assert!((sat.velocities()[0][0] - 1.0).abs() < 1e-9);
/// assert!(!sat.has_gaps());
/// ```
#[derive(Debug, Clone)]
pub struct SP3Satellite {
    id: SP3SatelliteId,
    epochs: Vec<Epoch>,
    records: Vec<SP3Record>,
    velocities: Vec<[f64; 3]>,
    has_file_velocities: bool,
    runs: Vec<Range<usize>>,
    trajectories: Vec<DOrbitTrajectory>,
    interpolation_method: InterpolationMethod,
}

impl SP3Satellite {
    /// Build a satellite from its valid samples.
    ///
    /// # Arguments
    ///
    /// * `id` - Satellite identifier
    /// * `samples` - Valid samples in strictly increasing epoch order
    ///
    /// # Returns
    ///
    /// * `Result<SP3Satellite, BraheError>` - The satellite, or an error if there are no
    ///   samples or the epochs are not strictly increasing
    pub(crate) fn from_samples(
        id: SP3SatelliteId,
        samples: Vec<SP3Sample>,
    ) -> Result<SP3Satellite, BraheError> {
        if samples.is_empty() {
            return Err(sp3_error(&format!("satellite {} has no valid sample", id)));
        }
        for pair in samples.windows(2) {
            if pair[1].epoch - pair[0].epoch <= 0.0 {
                return Err(sp3_error(&format!(
                    "samples of satellite {} are not in increasing epoch order: {} follows {}",
                    id, pair[1].epoch, pair[0].epoch
                )));
            }
        }

        let runs = contiguous_runs(&samples);
        let has_file_velocities = samples
            .iter()
            .all(|s| s.has_file_velocity && s.record.velocity.is_some());
        let epochs: Vec<Epoch> = samples.iter().map(|s| s.epoch).collect();
        let velocities = if has_file_velocities {
            samples
                .iter()
                .map(|s| s.record.velocity.unwrap_or([f64::NAN; 3]))
                .collect()
        } else {
            let positions: Vec<[f64; 3]> = samples.iter().map(|s| s.record.position).collect();
            derive_velocities(&epochs, &positions, &runs)
        };

        let mut satellite = SP3Satellite {
            id,
            epochs,
            records: samples.into_iter().map(|s| s.record).collect(),
            velocities,
            has_file_velocities,
            runs,
            trajectories: Vec::new(),
            interpolation_method: InterpolationMethod::Lagrange {
                degree: SP3_DEFAULT_LAGRANGE_DEGREE,
            },
        };
        satellite.trajectories = satellite.build_trajectories()?;
        Ok(satellite)
    }

    /// Build one ITRF trajectory per run of at least two samples.
    fn build_trajectories(&self) -> Result<Vec<DOrbitTrajectory>, BraheError> {
        let name = self.id.to_string();
        let mut trajectories = Vec::new();
        for run in self.runs.iter().filter(|run| run.len() >= 2) {
            let mut traj = DOrbitTrajectory::new(
                6,
                CelestialFrame::ITRF,
                OrbitRepresentation::Cartesian,
                None,
            )?
            .with_name(&name)
            .with_interpolation_method(run_method(self.interpolation_method, run.len()));
            for i in run.clone() {
                let p = self.records[i].position;
                let v = self.velocities[i];
                traj.add(
                    self.epochs[i],
                    DVector::from_column_slice(&[p[0], p[1], p[2], v[0], v[1], v[2]]),
                )?;
            }
            trajectories.push(traj);
        }
        Ok(trajectories)
    }

    /// Satellite identifier.
    ///
    /// # Returns
    ///
    /// * `SP3SatelliteId` - The identifier
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3WriteOptions::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # sp3.push_epoch(t0).unwrap();
    /// # sp3.set_record(id, 0, SP3Record::new([2.0e7, 1.0e7, 1.0e7])).unwrap();
    /// let sat = sp3.satellite(id).unwrap();
    /// assert_eq!(sat.id().to_string(), "G01");
    /// ```
    pub fn id(&self) -> SP3SatelliteId {
        self.id
    }

    /// Number of valid samples.
    ///
    /// # Returns
    ///
    /// * `usize` - Number of samples
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3WriteOptions::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # for k in 0..3 {
    /// #     sp3.push_epoch(t0 + 900.0 * k as f64).unwrap();
    /// # }
    /// sp3.set_record(id, 0, SP3Record::new([2.0e7, 1.0e7, 1.0e7])).unwrap();
    /// sp3.set_record(id, 2, SP3Record::new([2.0e7, 1.1e7, 1.0e7])).unwrap();
    /// assert_eq!(sp3.satellite(id).unwrap().len(), 2);
    /// ```
    pub fn len(&self) -> usize {
        self.epochs.len()
    }

    /// Whether the satellite has no samples.
    ///
    /// A satellite built from an SP3 file always has at least one sample.
    ///
    /// # Returns
    ///
    /// * `bool` - `true` if there are no samples
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3WriteOptions::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # sp3.push_epoch(t0).unwrap();
    /// # sp3.set_record(id, 0, SP3Record::new([2.0e7, 1.0e7, 1.0e7])).unwrap();
    /// assert!(!sp3.satellite(id).unwrap().is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.epochs.is_empty()
    }

    /// Epoch of the first sample.
    ///
    /// # Returns
    ///
    /// * `Epoch` - First sample epoch
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3WriteOptions::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # for k in 0..3 {
    /// #     sp3.push_epoch(t0 + 900.0 * k as f64).unwrap();
    /// # }
    /// sp3.set_record(id, 1, SP3Record::new([2.0e7, 1.0e7, 1.0e7])).unwrap();
    /// sp3.set_record(id, 2, SP3Record::new([2.0e7, 1.1e7, 1.0e7])).unwrap();
    /// assert_eq!(sp3.satellite(id).unwrap().start_epoch(), t0 + 900.0);
    /// ```
    pub fn start_epoch(&self) -> Epoch {
        self.epochs[0]
    }

    /// Epoch of the last sample.
    ///
    /// # Returns
    ///
    /// * `Epoch` - Last sample epoch
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3WriteOptions::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # for k in 0..3 {
    /// #     sp3.push_epoch(t0 + 900.0 * k as f64).unwrap();
    /// # }
    /// sp3.set_record(id, 0, SP3Record::new([2.0e7, 1.0e7, 1.0e7])).unwrap();
    /// sp3.set_record(id, 1, SP3Record::new([2.0e7, 1.1e7, 1.0e7])).unwrap();
    /// assert_eq!(sp3.satellite(id).unwrap().end_epoch(), t0 + 900.0);
    /// ```
    pub fn end_epoch(&self) -> Epoch {
        self.epochs[self.epochs.len() - 1]
    }

    /// Epochs of the valid samples, in increasing order.
    ///
    /// # Returns
    ///
    /// * `&[Epoch]` - Sample epochs
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3WriteOptions::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # for k in 0..3 {
    /// #     sp3.push_epoch(t0 + 900.0 * k as f64).unwrap();
    /// # }
    /// sp3.set_record(id, 0, SP3Record::new([2.0e7, 1.0e7, 1.0e7])).unwrap();
    /// sp3.set_record(id, 2, SP3Record::new([2.0e7, 1.1e7, 1.0e7])).unwrap();
    /// assert_eq!(sp3.satellite(id).unwrap().epochs(), &[t0, t0 + 1800.0]);
    /// ```
    pub fn epochs(&self) -> &[Epoch] {
        &self.epochs
    }

    /// Positions of the valid samples.
    ///
    /// # Returns
    ///
    /// * `Vec<[f64; 3]>` - Position `[x, y, z]` of each sample in ITRF. Units: m
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3WriteOptions::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # sp3.push_epoch(t0).unwrap();
    /// sp3.set_record(id, 0, SP3Record::new([2.0e7, 1.0e7, 1.0e7])).unwrap();
    /// assert_eq!(sp3.satellite(id).unwrap().positions(), vec![[2.0e7, 1.0e7, 1.0e7]]);
    /// ```
    pub fn positions(&self) -> Vec<[f64; 3]> {
        self.records.iter().map(|r| r.position).collect()
    }

    /// Velocities of the valid samples.
    ///
    /// The values come from the file when every sample has a file velocity,
    /// and are derived from the positions otherwise. A derived velocity is
    /// `NaN` for a sample that forms a run on its own.
    ///
    /// # Returns
    ///
    /// * `&[[f64; 3]]` - Velocity `[vx, vy, vz]` of each sample in ITRF. Units: m/s
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let options = SP3WriteOptions::default().with_velocity(false);
    /// # let mut sp3 = SP3::new(vec![id], 60.0, options).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # for k in 0..2 {
    /// #     sp3.push_epoch(t0 + 60.0 * k as f64).unwrap();
    /// # }
    /// sp3.set_record(id, 0, SP3Record::new([2.0e7, 1.0e7, 1.0e7])).unwrap();
    /// sp3.set_record(id, 1, SP3Record::new([2.0e7 + 600.0, 1.0e7, 1.0e7])).unwrap();
    /// let sat = sp3.satellite(id).unwrap();
    /// assert!((sat.velocities()[0][0] - 10.0).abs() < 1e-9);
    /// ```
    pub fn velocities(&self) -> &[[f64; 3]] {
        &self.velocities
    }

    /// Whether the velocities come from the file.
    ///
    /// # Returns
    ///
    /// * `bool` - `true` if every sample carries a file velocity, `false` if the
    ///   velocities are derived from the positions
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3WriteOptions::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # sp3.push_epoch(t0).unwrap();
    /// let record = SP3Record::new([2.0e7, 1.0e7, 1.0e7]).with_velocity([0.0, 3000.0, 0.0]);
    /// sp3.set_record(id, 0, record).unwrap();
    /// assert!(sp3.satellite(id).unwrap().has_file_velocities());
    /// ```
    pub fn has_file_velocities(&self) -> bool {
        self.has_file_velocities
    }

    /// Clock corrections of the valid samples.
    ///
    /// # Returns
    ///
    /// * `Vec<Option<f64>>` - Clock correction of each sample, `None` where the file
    ///   has a bad or missing clock. Units: s
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3WriteOptions::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # for k in 0..2 {
    /// #     sp3.push_epoch(t0 + 900.0 * k as f64).unwrap();
    /// # }
    /// sp3.set_record(id, 0, SP3Record::new([2.0e7, 1.0e7, 1.0e7]).with_clock(1e-5)).unwrap();
    /// sp3.set_record(id, 1, SP3Record::new([2.0e7, 1.1e7, 1.0e7])).unwrap();
    /// assert_eq!(sp3.satellite(id).unwrap().clocks(), vec![Some(1e-5), None]);
    /// ```
    pub fn clocks(&self) -> Vec<Option<f64>> {
        self.records.iter().map(|r| r.clock).collect()
    }

    /// Clock rates of the valid samples.
    ///
    /// # Returns
    ///
    /// * `Vec<Option<f64>>` - Clock rate of each sample, `None` where the file has a
    ///   bad or missing rate. Units: s/s
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3WriteOptions::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # sp3.push_epoch(t0).unwrap();
    /// let record = SP3Record::new([2.0e7, 1.0e7, 1.0e7])
    ///     .with_velocity([0.0, 3000.0, 0.0])
    ///     .with_clock_rate(1e-12);
    /// sp3.set_record(id, 0, record).unwrap();
    /// assert_eq!(sp3.satellite(id).unwrap().clock_rates(), vec![Some(1e-12)]);
    /// ```
    pub fn clock_rates(&self) -> Vec<Option<f64>> {
        self.records.iter().map(|r| r.clock_rate).collect()
    }

    /// Full file records of the valid samples.
    ///
    /// The records hold the file values unchanged, including standard
    /// deviations, flags, and correlations; derived velocities are not
    /// written into them.
    ///
    /// # Returns
    ///
    /// * `&[SP3Record]` - Record of each sample
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3WriteOptions::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # sp3.push_epoch(t0).unwrap();
    /// let mut record = SP3Record::new([2.0e7, 1.0e7, 1.0e7]);
    /// record.maneuver = true;
    /// sp3.set_record(id, 0, record).unwrap();
    /// assert!(sp3.satellite(id).unwrap().records()[0].maneuver);
    /// ```
    pub fn records(&self) -> &[SP3Record] {
        &self.records
    }

    /// Spans of the contiguous runs of samples.
    ///
    /// A run that holds a single sample has equal start and end epochs.
    ///
    /// # Returns
    ///
    /// * `Vec<(Epoch, Epoch)>` - First and last sample epoch of each run, in time order
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3WriteOptions::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// for k in [0, 1, 2, 5, 6] {
    ///     let index = sp3.push_epoch(t0 + 900.0 * k as f64).unwrap();
    ///     let y = 1.0e7 + 1000.0 * k as f64;
    ///     sp3.set_record(id, index, SP3Record::new([2.0e7, y, 1.0e7])).unwrap();
    /// }
    /// let sat = sp3.satellite(id).unwrap();
    /// assert_eq!(sat.segments(), vec![(t0, t0 + 1800.0), (t0 + 4500.0, t0 + 5400.0)]);
    /// assert!(sat.has_gaps());
    /// ```
    pub fn segments(&self) -> Vec<(Epoch, Epoch)> {
        self.runs
            .iter()
            .map(|run| (self.epochs[run.start], self.epochs[run.end - 1]))
            .collect()
    }

    /// Whether the samples form more than one contiguous run.
    ///
    /// # Returns
    ///
    /// * `bool` - `true` if the satellite has at least one gap
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3WriteOptions::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// for k in 0..3 {
    ///     let index = sp3.push_epoch(t0 + 900.0 * k as f64).unwrap();
    ///     let y = 1.0e7 + 1000.0 * k as f64;
    ///     sp3.set_record(id, index, SP3Record::new([2.0e7, y, 1.0e7])).unwrap();
    /// }
    /// assert!(!sp3.satellite(id).unwrap().has_gaps());
    /// ```
    pub fn has_gaps(&self) -> bool {
        self.runs.len() > 1
    }

    /// Trajectory over all samples.
    ///
    /// The trajectory is a Cartesian ITRF [`DOrbitTrajectory`] named after the
    /// satellite, using the satellite's interpolation method. A Lagrange degree
    /// larger than the number of samples allows is reduced to the largest
    /// degree they support.
    ///
    /// # Returns
    ///
    /// * `Result<DOrbitTrajectory, BraheError>` - The trajectory, or an error naming the
    ///   satellite and the gap spans if the satellite has a gap, or naming the satellite and
    ///   epoch if it has a single sample
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// use brahe::trajectories::traits::Trajectory;
    /// use brahe::utils::Identifiable;
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3WriteOptions::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # for k in 0..3 {
    /// #     let index = sp3.push_epoch(t0 + 900.0 * k as f64).unwrap();
    /// #     let y = 1.0e7 + 1000.0 * k as f64;
    /// #     sp3.set_record(id, index, SP3Record::new([2.0e7, y, 1.0e7])).unwrap();
    /// # }
    /// let traj = sp3.satellite(id).unwrap().to_trajectory().unwrap();
    /// assert_eq!(traj.len(), 3);
    /// assert_eq!(traj.get_name(), Some("G01"));
    /// ```
    pub fn to_trajectory(&self) -> Result<DOrbitTrajectory, BraheError> {
        if self.has_gaps() {
            let gaps: Vec<String> = self
                .runs
                .windows(2)
                .map(|pair| {
                    format!(
                        "from {} to {}",
                        self.epochs[pair[0].end - 1],
                        self.epochs[pair[1].start]
                    )
                })
                .collect();
            return Err(sp3_error(&format!(
                "satellite {} has {} {} ({}); use to_trajectory_segments for one trajectory \
                 per contiguous run",
                self.id,
                gaps.len(),
                if gaps.len() == 1 { "gap" } else { "gaps" },
                gaps.join(", ")
            )));
        }
        self.trajectories.first().cloned().ok_or_else(|| {
            sp3_error(&format!(
                "satellite {} has a single sample at {}; a trajectory needs at least two \
                 consecutive samples",
                self.id, self.epochs[0]
            ))
        })
    }

    /// One trajectory per contiguous run of at least two samples.
    ///
    /// Each trajectory is a Cartesian ITRF [`DOrbitTrajectory`] named after
    /// the satellite, using the satellite's interpolation method with a
    /// Lagrange degree reduced to what the run supports. Runs of a single
    /// sample are omitted.
    ///
    /// # Returns
    ///
    /// * `Result<Vec<DOrbitTrajectory>, BraheError>` - Trajectories in time order
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3WriteOptions::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// for k in [0, 1, 2, 5, 6] {
    ///     let index = sp3.push_epoch(t0 + 900.0 * k as f64).unwrap();
    ///     let y = 1.0e7 + 1000.0 * k as f64;
    ///     sp3.set_record(id, index, SP3Record::new([2.0e7, y, 1.0e7])).unwrap();
    /// }
    /// let sat = sp3.satellite(id).unwrap();
    /// assert!(sat.to_trajectory().is_err());
    /// assert_eq!(sat.to_trajectory_segments().unwrap().len(), 2);
    /// ```
    pub fn to_trajectory_segments(&self) -> Result<Vec<DOrbitTrajectory>, BraheError> {
        Ok(self.trajectories.clone())
    }
}

impl InterpolationConfig for SP3Satellite {
    /// Set the interpolation method using the builder pattern.
    ///
    /// The method applies to the trajectories returned afterwards. A Lagrange
    /// degree larger than a run supports is reduced for that run's trajectory.
    /// The fit that derives velocities is not affected.
    ///
    /// # Arguments
    ///
    /// * `method` - Interpolation method
    ///
    /// # Returns
    ///
    /// * `SP3Satellite` - The satellite with the method set
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// use brahe::math::interpolation::{InterpolationConfig, InterpolationMethod};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3WriteOptions::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # for k in 0..3 {
    /// #     let index = sp3.push_epoch(t0 + 900.0 * k as f64).unwrap();
    /// #     let y = 1.0e7 + 1000.0 * k as f64;
    /// #     sp3.set_record(id, index, SP3Record::new([2.0e7, y, 1.0e7])).unwrap();
    /// # }
    /// let sat = sp3
    ///     .satellite(id)
    ///     .unwrap()
    ///     .with_interpolation_method(InterpolationMethod::Linear);
    /// let traj = sat.to_trajectory().unwrap();
    /// assert_eq!(traj.get_interpolation_method(), InterpolationMethod::Linear);
    /// ```
    fn with_interpolation_method(mut self, method: InterpolationMethod) -> Self {
        self.set_interpolation_method(method);
        self
    }

    /// Set the interpolation method.
    ///
    /// The method applies to the trajectories returned afterwards. A Lagrange
    /// degree larger than a run supports is reduced for that run's trajectory.
    /// The fit that derives velocities is not affected.
    ///
    /// # Arguments
    ///
    /// * `method` - Interpolation method
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// use brahe::math::interpolation::{InterpolationConfig, InterpolationMethod};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3WriteOptions::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # for k in 0..3 {
    /// #     let index = sp3.push_epoch(t0 + 900.0 * k as f64).unwrap();
    /// #     let y = 1.0e7 + 1000.0 * k as f64;
    /// #     sp3.set_record(id, index, SP3Record::new([2.0e7, y, 1.0e7])).unwrap();
    /// # }
    /// let mut sat = sp3.satellite(id).unwrap();
    /// sat.set_interpolation_method(InterpolationMethod::Lagrange { degree: 5 });
    /// let traj = sat.to_trajectory().unwrap();
    /// assert_eq!(traj.get_interpolation_method(), InterpolationMethod::Lagrange { degree: 2 });
    /// ```
    fn set_interpolation_method(&mut self, method: InterpolationMethod) {
        self.interpolation_method = method;
        for traj in &mut self.trajectories {
            let run_len = traj.len();
            traj.set_interpolation_method(run_method(method, run_len));
        }
    }

    /// Get the interpolation method.
    ///
    /// # Returns
    ///
    /// * `InterpolationMethod` - The method as set, before any per-run Lagrange degree
    ///   reduction; defaults to `Lagrange { degree: SP3_DEFAULT_LAGRANGE_DEGREE }`
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// use brahe::datafiles::sp3::SP3_DEFAULT_LAGRANGE_DEGREE;
    /// use brahe::math::interpolation::{InterpolationConfig, InterpolationMethod};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3WriteOptions::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # sp3.push_epoch(t0).unwrap();
    /// # sp3.set_record(id, 0, SP3Record::new([2.0e7, 1.0e7, 1.0e7])).unwrap();
    /// let sat = sp3.satellite(id).unwrap();
    /// assert_eq!(
    ///     sat.get_interpolation_method(),
    ///     InterpolationMethod::Lagrange { degree: SP3_DEFAULT_LAGRANGE_DEGREE }
    /// );
    /// ```
    fn get_interpolation_method(&self) -> InterpolationMethod {
        self.interpolation_method
    }
}

/// Interpolation method for a run of `run_len` samples, with a Lagrange
/// degree reduced to at most `run_len - 1`.
fn run_method(method: InterpolationMethod, run_len: usize) -> InterpolationMethod {
    match method {
        InterpolationMethod::Lagrange { degree } => InterpolationMethod::Lagrange {
            degree: degree.min(run_len.saturating_sub(1)),
        },
        other => other,
    }
}

/// Split samples into index ranges of contiguous runs.
///
/// A new run starts where two consecutive samples are spaced more than
/// [`GAP_FACTOR`] times the larger of their epoch intervals apart.
fn contiguous_runs(samples: &[SP3Sample]) -> Vec<Range<usize>> {
    let mut runs = Vec::new();
    let mut start = 0;
    for i in 1..samples.len() {
        let spacing = samples[i].epoch - samples[i - 1].epoch;
        if spacing > GAP_FACTOR * samples[i - 1].interval.max(samples[i].interval) {
            runs.push(start..i);
            start = i;
        }
    }
    runs.push(start..samples.len());
    runs
}

/// Velocity at each sample from a Lagrange fit within its run.
///
/// Each sample uses the `min(SP3_DEFAULT_LAGRANGE_DEGREE + 1, n)` samples of
/// its run of length `n` that are most nearly centered on it. Samples in runs
/// of one sample get `NaN`.
fn derive_velocities(
    epochs: &[Epoch],
    positions: &[[f64; 3]],
    runs: &[Range<usize>],
) -> Vec<[f64; 3]> {
    let mut velocities = vec![[f64::NAN; 3]; epochs.len()];
    for run in runs {
        let n = run.len();
        if n < 2 {
            continue;
        }
        let m = (SP3_DEFAULT_LAGRANGE_DEGREE + 1).min(n);
        for i in 0..n {
            let offset = i.saturating_sub(m / 2).min(n - m);
            let window = (run.start + offset)..(run.start + offset + m);
            let origin = epochs[window.start];
            let times: Vec<f64> = window.clone().map(|k| epochs[k] - origin).collect();
            velocities[run.start + i] =
                lagrange_derivative_at_node(&times, &positions[window], i - offset);
        }
    }
    velocities
}

/// Derivative of the Lagrange polynomial through `(times, values)` at one of
/// its nodes.
///
/// With `x_i = times[node]` the derivative is
/// `Σ_{j≠i} p_j·w_j + p_i·Σ_{k≠i} 1/(x_i − x_k)` where
/// `w_j = 1/(x_j − x_i)·Π_{k≠i,j} (x_i − x_k)/(x_j − x_k)`.
///
/// # Arguments
///
/// * `times` - Distinct node abscissae. Units: s
/// * `values` - Value at each node. Units: m
/// * `node` - Index of the node to differentiate at
///
/// # Returns
///
/// * `[f64; 3]` - Derivative at the node. Units: m/s
fn lagrange_derivative_at_node(times: &[f64], values: &[[f64; 3]], node: usize) -> [f64; 3] {
    let xi = times[node];
    let mut rate = [0.0; 3];
    let mut node_weight = 0.0;
    for (j, (&xj, pj)) in times.iter().zip(values).enumerate() {
        if j == node {
            continue;
        }
        node_weight += 1.0 / (xi - xj);
        let mut weight = 1.0 / (xj - xi);
        for (k, &xk) in times.iter().enumerate() {
            if k != node && k != j {
                weight *= (xi - xk) / (xj - xk);
            }
        }
        for axis in 0..3 {
            rate[axis] += weight * pj[axis];
        }
    }
    for axis in 0..3 {
        rate[axis] += node_weight * values[node][axis];
    }
    rate
}

#[cfg(test)]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::constants::AngleFormat;
    use crate::datafiles::sp3::types::{SP3, SP3WriteOptions};
    use crate::frames::{CelestialFrame, ReferenceFrame};
    use crate::propagators::KeplerianPropagator;
    use crate::time::TimeSystem;
    use crate::utils::testing::setup_global_test_eop;
    use crate::utils::{DOrbitStateProvider, Identifiable};
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector6;
    use serial_test::{parallel, serial};

    const STEP: f64 = 900.0;
    const N_EPOCHS: usize = 97;

    fn sid(s: &str) -> SP3SatelliteId {
        s.parse().unwrap()
    }

    fn t0() -> Epoch {
        Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS)
    }

    fn gps_propagator(raan: f64, mean_anomaly: f64) -> KeplerianPropagator {
        KeplerianPropagator::from_keplerian(
            t0(),
            Vector6::new(26_560e3, 0.01, 55.0, raan, 0.0, mean_anomaly),
            AngleFormat::Degrees,
            60.0,
        )
        .unwrap()
    }

    /// A 24 h position-only file at 900 s with four satellites:
    /// G01 absent at epochs 40..=43, G02 with no clock at epoch 5 and absent
    /// at epoch 7, G03 present only at epochs 0..=10 and 20, and G04 holding
    /// only a zero-position record.
    fn build_sp3() -> (SP3, Vec<KeplerianPropagator>) {
        setup_global_test_eop();
        let ids = vec![sid("G01"), sid("G02"), sid("G03"), sid("G04")];
        let propagators = vec![
            gps_propagator(0.0, 0.0),
            gps_propagator(60.0, 30.0),
            gps_propagator(120.0, 60.0),
            gps_propagator(180.0, 90.0),
        ];
        let mut sp3 = SP3::new(
            ids.clone(),
            STEP,
            SP3WriteOptions::default().with_velocity(false),
        )
        .unwrap();
        for k in 0..N_EPOCHS {
            let epoch = t0() + k as f64 * STEP;
            let index = sp3.push_epoch(epoch).unwrap();
            for (n, id) in ids.iter().enumerate() {
                let present = match n {
                    0 => !(40..=43).contains(&k),
                    1 => k != 7,
                    2 => k <= 10 || k == 20,
                    _ => false,
                };
                if !present {
                    continue;
                }
                let x = propagators[n].state_itrf(epoch).unwrap();
                let mut record = SP3Record::new([x[0], x[1], x[2]]);
                if !(n == 1 && k == 5) {
                    record = record.with_clock(1.0e-5 + 1.0e-9 * k as f64);
                }
                sp3.set_record(*id, index, record).unwrap();
            }
        }
        sp3.set_record(sid("G04"), 3, SP3Record::new([0.0; 3]))
            .unwrap();
        (sp3, propagators)
    }

    fn sample(epoch: Epoch, position: [f64; 3], interval: f64) -> SP3Sample {
        SP3Sample {
            epoch,
            record: SP3Record::new(position),
            interval,
            has_file_velocity: false,
        }
    }

    /// Position of a body in uniformly accelerated motion at `t` seconds.
    fn quadratic_position(t: f64) -> [f64; 3] {
        [
            1.0e7 + 100.0 * t + 0.01 * t * t,
            2.0e7 - 50.0 * t,
            3.0e6 + 0.002 * t * t,
        ]
    }

    fn quadratic_velocity(t: f64) -> [f64; 3] {
        [100.0 + 0.02 * t, -50.0, 0.004 * t]
    }

    #[test]
    #[serial]
    fn test_SP3Satellite_gap_split() {
        let (sp3, _) = build_sp3();
        let t0 = t0();
        let g01 = sp3.satellite("G01".parse().unwrap()).unwrap();
        assert_eq!(g01.len(), 93);
        assert!(g01.has_gaps());
        assert_eq!(
            g01.segments(),
            vec![
                (t0, t0 + 39.0 * 900.0),
                (t0 + 44.0 * 900.0, t0 + 96.0 * 900.0)
            ]
        );
        let err = g01.to_trajectory().unwrap_err().to_string();
        assert!(err.contains("G01") && err.contains("to_trajectory_segments"));
        let gap_start: Epoch = t0 + 39.0 * 900.0;
        let gap_end: Epoch = t0 + 44.0 * 900.0;
        assert!(err.contains(&gap_start.to_string()) && err.contains(&gap_end.to_string()));
        let segs = g01.to_trajectory_segments().unwrap();
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].get_name(), Some("G01"));
        assert_eq!(segs[0].frame, ReferenceFrame::from(CelestialFrame::ITRF));
        assert_eq!(segs[0].len(), 40);
        assert_eq!(segs[1].len(), 53);
        assert_eq!(segs[1].epochs[0], t0 + 44.0 * 900.0);
    }

    #[test]
    #[serial]
    fn test_SP3Satellite_accessors_skip_invalid_and_keep_bad_clocks() {
        let (sp3, _) = build_sp3();
        let t0 = t0();
        let g02 = sp3.satellite("G02".parse().unwrap()).unwrap();
        assert_eq!(g02.id(), sid("G02"));
        assert_eq!(g02.len(), N_EPOCHS - 1);
        assert!(!g02.is_empty());
        assert_eq!(g02.epochs().len(), g02.positions().len());
        assert_eq!(g02.epochs().len(), g02.velocities().len());
        assert_eq!(g02.epochs().len(), g02.clocks().len());
        assert_eq!(g02.epochs().len(), g02.clock_rates().len());
        assert_eq!(g02.epochs().len(), g02.records().len());
        assert_eq!(g02.clocks()[5], None);
        assert_eq!(g02.clocks()[4], Some(1.0e-5 + 4.0e-9));
        assert!(g02.clock_rates().iter().all(|r| r.is_none()));
        assert!(!g02.epochs().contains(&(t0 + 7.0 * 900.0)));
        assert!(!g02.has_file_velocities());
        assert_eq!(g02.start_epoch(), t0);
        assert_eq!(g02.end_epoch(), t0 + 96.0 * 900.0);
        let expected = sp3.record(sid("G02"), 8).unwrap().unwrap();
        assert_eq!(g02.positions()[7], expected.position);
        assert_eq!(&g02.records()[7], expected);
    }

    #[test]
    #[serial]
    fn test_SP3Satellite_single_sample_run() {
        let (sp3, _) = build_sp3();
        let t0 = t0();
        let g03 = sp3.satellite("G03".parse().unwrap()).unwrap();
        assert_eq!(g03.len(), 12);
        assert_eq!(
            g03.segments(),
            vec![
                (t0, t0 + 10.0 * 900.0),
                (t0 + 20.0 * 900.0, t0 + 20.0 * 900.0)
            ]
        );
        assert_eq!(g03.to_trajectory_segments().unwrap().len(), 1);
        assert!(g03.to_trajectory().unwrap_err().to_string().contains("G03"));
        assert!(g03.velocities()[11].iter().all(|v| v.is_nan()));
        assert!(g03.velocities()[10].iter().all(|v| v.is_finite()));
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_to_trajectory_single_sample() {
        let t0 = t0();
        let sat =
            SP3Satellite::from_samples(sid("G07"), vec![sample(t0, [1.0e7, 2.0e7, 3.0e6], 900.0)])
                .unwrap();
        assert!(!sat.has_gaps());
        assert_eq!(sat.segments(), vec![(t0, t0)]);
        assert!(sat.to_trajectory_segments().unwrap().is_empty());
        let err = sat.to_trajectory().unwrap_err().to_string();
        assert!(err.contains("G07") && err.contains(&t0.to_string()));
    }

    #[test]
    #[serial]
    fn test_SP3_satellite_without_valid_sample() {
        let (sp3, _) = build_sp3();
        let err = sp3.satellite(sid("G04")).unwrap_err().to_string();
        assert!(err.contains("G04"));
        assert!(sp3.satellite(sid("G05")).is_err());
        let ids: Vec<SP3SatelliteId> = sp3.satellites().unwrap().iter().map(|s| s.id()).collect();
        assert_eq!(ids, vec![sid("G01"), sid("G02"), sid("G03")]);
    }

    #[test]
    #[serial]
    fn test_SP3Satellite_InterpolationConfig_propagates_to_trajectories() {
        let (sp3, _) = build_sp3();
        let g01 = sp3.satellite(sid("G01")).unwrap();
        let default_method = InterpolationMethod::Lagrange {
            degree: SP3_DEFAULT_LAGRANGE_DEGREE,
        };
        assert_eq!(g01.get_interpolation_method(), default_method);
        for traj in g01.to_trajectory_segments().unwrap() {
            assert_eq!(traj.get_interpolation_method(), default_method);
        }

        let method = InterpolationMethod::Lagrange { degree: 5 };
        let g01 = g01.with_interpolation_method(method);
        assert_eq!(g01.get_interpolation_method(), method);
        for traj in g01.to_trajectory_segments().unwrap() {
            assert_eq!(traj.get_interpolation_method(), method);
        }

        let mut g02 = sp3.satellite(sid("G02")).unwrap();
        g02.set_interpolation_method(InterpolationMethod::HermiteCubic);
        assert_eq!(
            g02.get_interpolation_method(),
            InterpolationMethod::HermiteCubic
        );
        for traj in g02.to_trajectory_segments().unwrap() {
            assert_eq!(
                traj.get_interpolation_method(),
                InterpolationMethod::HermiteCubic
            );
        }
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_short_run_limits_lagrange_degree() {
        let t0 = t0();
        let samples = (0..3)
            .map(|k| {
                sample(
                    t0 + 900.0 * k as f64,
                    quadratic_position(900.0 * k as f64),
                    900.0,
                )
            })
            .collect();
        let sat = SP3Satellite::from_samples(sid("G08"), samples)
            .unwrap()
            .with_interpolation_method(InterpolationMethod::Lagrange { degree: 7 });
        assert_eq!(
            sat.get_interpolation_method(),
            InterpolationMethod::Lagrange { degree: 7 }
        );
        assert_eq!(
            sat.to_trajectory().unwrap().get_interpolation_method(),
            InterpolationMethod::Lagrange { degree: 2 }
        );
    }

    #[test]
    #[serial]
    fn test_SP3Satellite_derived_velocity_matches_propagator() {
        let (sp3, propagators) = build_sp3();
        let g01 = sp3.satellite(sid("G01")).unwrap();
        let half = SP3_DEFAULT_LAGRANGE_DEGREE.div_ceil(2);
        let runs = [(0usize, 40usize), (40, 93)];
        for (start, end) in runs {
            for i in (start + half)..(end - half) {
                let epoch = g01.epochs()[i];
                let truth = propagators[0].state_itrf(epoch).unwrap();
                for axis in 0..3 {
                    assert_abs_diff_eq!(g01.velocities()[i][axis], truth[3 + axis], epsilon = 1e-3);
                }
            }
        }
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_derived_velocity_exact_for_polynomials() {
        let t0 = t0();
        let offsets = [
            0.0, 300.0, 600.0, 900.0, 1200.0, 1500.0, 1800.0, 2100.0, 2400.0, 2700.0, 3000.0,
            3300.0, 3600.0,
        ];
        let samples = offsets
            .iter()
            .map(|&t| sample(t0 + t, quadratic_position(t), 300.0))
            .collect();
        let sat = SP3Satellite::from_samples(sid("G09"), samples).unwrap();
        assert!(!sat.has_file_velocities());
        for (i, &t) in offsets.iter().enumerate() {
            let truth = quadratic_velocity(t);
            for (derived, expected) in sat.velocities()[i].iter().zip(truth) {
                assert_abs_diff_eq!(*derived, expected, epsilon = 1e-6);
            }
        }

        let pair = vec![
            sample(t0, [0.0, 1.0e7, 2.0e7], 60.0),
            sample(t0 + 60.0, [600.0, 1.0e7 - 120.0, 2.0e7], 60.0),
        ];
        let sat = SP3Satellite::from_samples(sid("G10"), pair).unwrap();
        for velocity in sat.velocities() {
            assert_abs_diff_eq!(velocity[0], 10.0, epsilon = 1e-9);
            assert_abs_diff_eq!(velocity[1], -2.0, epsilon = 1e-9);
            assert_abs_diff_eq!(velocity[2], 0.0, epsilon = 1e-9);
        }
    }

    #[test]
    #[parallel]
    fn test_lagrange_derivative_at_node() {
        let times = [0.0, 1.0, 2.5, 4.0, 4.5, 7.0];
        let cubic = |t: f64| 2.0 - 3.0 * t + 0.5 * t * t * t;
        let cubic_rate = |t: f64| -3.0 + 1.5 * t * t;
        let values: Vec<[f64; 3]> = times
            .iter()
            .map(|&t| [cubic(t), 4.0 * t, -cubic(t)])
            .collect();
        for (node, &t) in times.iter().enumerate() {
            let rate = lagrange_derivative_at_node(&times, &values, node);
            assert_abs_diff_eq!(rate[0], cubic_rate(t), epsilon = 1e-10);
            assert_abs_diff_eq!(rate[1], 4.0, epsilon = 1e-10);
            assert_abs_diff_eq!(rate[2], -cubic_rate(t), epsilon = 1e-10);
        }
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_file_velocities() {
        let t0 = t0();
        let with_velocity = |k: usize, has_file_velocity: bool| {
            let t = 900.0 * k as f64;
            SP3Sample {
                epoch: t0 + t,
                record: SP3Record::new(quadratic_position(t)).with_velocity([1.0, 2.0, 3.0 + t]),
                interval: 900.0,
                has_file_velocity,
            }
        };
        let samples: Vec<SP3Sample> = (0..5).map(|k| with_velocity(k, true)).collect();
        let sat = SP3Satellite::from_samples(sid("G11"), samples.clone()).unwrap();
        assert!(sat.has_file_velocities());
        assert_eq!(sat.velocities()[2], [1.0, 2.0, 1803.0]);

        let mut mixed = samples;
        mixed[3] = with_velocity(3, false);
        let sat = SP3Satellite::from_samples(sid("G11"), mixed).unwrap();
        assert!(!sat.has_file_velocities());
        for (i, epoch) in sat.epochs().iter().enumerate() {
            let truth = quadratic_velocity(*epoch - t0);
            for (derived, expected) in sat.velocities()[i].iter().zip(truth) {
                assert_abs_diff_eq!(*derived, expected, epsilon = 1e-6);
            }
        }
        assert_eq!(sat.records()[2].velocity, Some([1.0, 2.0, 1803.0]));
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_gap_threshold_uses_larger_interval() {
        let t0 = t0();
        let p = [1.0e7, 2.0e7, 3.0e6];
        let samples = vec![
            sample(t0, p, 300.0),
            sample(t0 + 300.0, p, 300.0),
            sample(t0 + 1200.0, p, 900.0),
            sample(t0 + 2100.0, p, 900.0),
            sample(t0 + 3500.0, p, 900.0),
            sample(t0 + 4400.0, p, 900.0),
        ];
        let sat = SP3Satellite::from_samples(sid("G12"), samples).unwrap();
        assert_eq!(
            sat.segments(),
            vec![(t0, t0 + 2100.0), (t0 + 3500.0, t0 + 4400.0)]
        );
        assert!(sat.has_gaps());
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_samples_errors() {
        let t0 = t0();
        let p = [1.0e7, 2.0e7, 3.0e6];
        let err = SP3Satellite::from_samples(sid("G13"), Vec::new())
            .unwrap_err()
            .to_string();
        assert!(err.contains("G13"));
        let unordered = vec![sample(t0 + 900.0, p, 900.0), sample(t0, p, 900.0)];
        assert!(SP3Satellite::from_samples(sid("G13"), unordered).is_err());
        let repeated = vec![sample(t0, p, 900.0), sample(t0, p, 900.0)];
        assert!(SP3Satellite::from_samples(sid("G13"), repeated).is_err());
    }
}
