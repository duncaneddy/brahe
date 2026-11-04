/*!
 * Per-satellite view of SP3 precise orbit data.
 *
 * [`SP3Satellite`] holds the valid samples of one satellite, the velocity at
 * each sample, and one ITRF trajectory per contiguous run of samples.
 */

use std::ops::Range;

use nalgebra::{DVector, Vector6};
use uuid::Uuid;

use crate::constants::AngleFormat;
use crate::datafiles::sp3::interop::chain_files;
use crate::datafiles::sp3::parse::sp3_error;
use crate::datafiles::sp3::types::{SP3, SP3Record, SP3SatelliteId};
use crate::frames::CelestialFrame;
use crate::math::interpolation::{InterpolationConfig, InterpolationMethod};
use crate::time::Epoch;
use crate::trajectories::dorbit_trajectory::DOrbitTrajectory;
use crate::trajectories::traits::{OrbitRepresentation, Trajectory};
use crate::utils::{BraheError, DOrbitStateProvider, DStateProvider, Identifiable};

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
/// The satellite is an orbit state provider ([`DStateProvider`] and
/// [`DOrbitStateProvider`]) and is [`Identifiable`], with the satellite
/// identifier string (such as `"G01"`) as its default name, so it can be passed
/// to [`location_accesses`](crate::access::location_accesses) directly. States
/// are interpolated in ITRF from the run containing the requested epoch, both
/// ends of a run included, and converted to other frames on request. A request
/// before the first sample, after the last sample, inside a gap, or at a sample
/// that forms a run on its own is an error naming the satellite. Changing the
/// name, ID, or UUID also changes those of the trajectories it returns.
///
/// # Examples
///
/// ```
/// use brahe::datafiles::sp3::{SP3, SP3Record, SP3SatelliteId, SP3WriteOptions};
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::utils::{DOrbitStateProvider, Identifiable};
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
///
/// assert_eq!(sat.get_name(), Some("G01"));
/// let state = sat.state_itrf(t0 + 900.0).unwrap();
/// assert!((state[0] - 2.0e7 - 900.0).abs() < 1e-6);
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
    name: Option<String>,
    uuid: Option<Uuid>,
    numeric_id: Option<u64>,
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
        let epochs: Vec<Epoch> = samples.iter().map(|s| s.epoch).collect();
        let file_velocities: Option<Vec<[f64; 3]>> = samples
            .iter()
            .map(|s| s.record.velocity.filter(|_| s.has_file_velocity))
            .collect();
        let has_file_velocities = file_velocities.is_some();
        let velocities = file_velocities.unwrap_or_else(|| {
            let positions: Vec<[f64; 3]> = samples.iter().map(|s| s.record.position).collect();
            derive_velocities(&epochs, &positions, &runs)
        });

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
            name: Some(id.to_string()),
            uuid: None,
            numeric_id: None,
        };
        satellite.trajectories = satellite.build_trajectories()?;
        Ok(satellite)
    }

    /// Chain every satellite across several SP3 files.
    ///
    /// The files are sorted by start epoch, so they may be passed in any
    /// order; files without epochs are ignored. For each satellite in the
    /// union of the files' satellite lists, the valid samples of every file
    /// are joined in time order before velocities are derived, so samples
    /// next to a file boundary get centered fits. Epochs compare as absolute
    /// instants, so files in different time systems chain directly.
    ///
    /// Two consecutive files may share their boundary epoch. The later file's
    /// sample is kept there when it is valid, and the earlier file's otherwise.
    /// A file that lacks a satellite contributes no samples, which leaves a
    /// gap in that satellite. Velocities come from the files only when every
    /// sample comes from a file with velocity records. Agencies, coordinate
    /// systems, orbit types, and epoch intervals need not match; gaps use the
    /// larger epoch interval of the two samples around them.
    ///
    /// # Arguments
    ///
    /// * `files` - SP3 files covering consecutive, non-overlapping spans
    ///
    /// # Returns
    ///
    /// * `Result<Vec<SP3Satellite>, BraheError>` - One satellite for each identifier with
    ///   at least one valid sample, in ascending identifier order (GPS, GLONASS, Galileo,
    ///   BeiDou, QZSS, IRNSS, SBAS, LEO, then by number), or an error naming the agency
    ///   and span of two files that overlap by more than a shared boundary epoch
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{SP3, SP3Record, SP3Satellite, SP3SatelliteId, SP3WriteOptions};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let id: SP3SatelliteId = "G01".parse().unwrap();
    /// let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let file = |start: usize| {
    ///     let options = SP3WriteOptions::default().with_velocity(false);
    ///     let mut sp3 = SP3::new(vec![id], 900.0, options).unwrap();
    ///     for k in start..start + 3 {
    ///         let index = sp3.push_epoch(t0 + 900.0 * k as f64).unwrap();
    ///         let x = 2.0e7 + 900.0 * k as f64;
    ///         sp3.set_record(id, index, SP3Record::new([x, 1.0e7, 1.0e7])).unwrap();
    ///     }
    ///     sp3
    /// };
    ///
    /// // Epochs 0-2 and 2-4 share epoch 2.
    /// let sats = SP3Satellite::from_files(&[file(2), file(0)]).unwrap();
    /// assert_eq!(sats.len(), 1);
    /// assert_eq!(sats[0].len(), 5);
    /// assert!(!sats[0].has_gaps());
    ///
    /// // A file passed twice overlaps itself.
    /// assert!(SP3Satellite::from_files(&[file(0), file(0)]).is_err());
    /// ```
    pub fn from_files(files: &[SP3]) -> Result<Vec<SP3Satellite>, BraheError> {
        chain_files(files)
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

    /// Trajectory of the run that contains `epoch`, both ends included.
    ///
    /// # Arguments
    ///
    /// * `epoch` - Epoch to look up
    ///
    /// # Returns
    ///
    /// * `Result<&DOrbitTrajectory, BraheError>` - The segment, or an error naming the
    ///   satellite and the span of samples when `epoch` is outside the samples, inside
    ///   a gap, or at a sample that forms a run on its own
    fn segment_for(&self, epoch: Epoch) -> Result<&DOrbitTrajectory, BraheError> {
        let index = self
            .trajectories
            .partition_point(|traj| traj.end_epoch().is_some_and(|end| end < epoch));
        if let Some(traj) = self.trajectories.get(index)
            && traj.start_epoch().is_some_and(|start| start <= epoch)
        {
            return Ok(traj);
        }
        Err(self.unavailable_error(epoch))
    }

    /// Error for an epoch that no segment contains.
    fn unavailable_error(&self, epoch: Epoch) -> BraheError {
        let (first, last) = (self.start_epoch(), self.end_epoch());
        let detail = if epoch < first || epoch > last {
            format!(
                "satellite {} has samples from {} to {}; epoch {} is outside that span",
                self.id, first, last, epoch
            )
        } else {
            let next = self.epochs.partition_point(|&e| e <= epoch);
            let previous = next - 1;
            if self.epochs[previous] == epoch {
                format!(
                    "satellite {} has a single sample at {} that is not part of a contiguous run \
                     of at least two samples",
                    self.id, epoch
                )
            } else {
                format!(
                    "satellite {} has no samples from {} to {}; epoch {} is inside that gap",
                    self.id, self.epochs[previous], self.epochs[next], epoch
                )
            }
        };
        BraheError::OutOfBoundsError(format!("SP3: {}", detail))
    }

    /// Give every trajectory the satellite's name, numeric ID, and UUID.
    fn sync_trajectory_identity(&mut self) {
        for traj in &mut self.trajectories {
            traj.set_identity(self.name.as_deref(), self.uuid, self.numeric_id);
        }
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
    /// is raised to at least 1 and reduced to at most the number of samples
    /// minus one.
    ///
    /// # Returns
    ///
    /// * `Result<DOrbitTrajectory, BraheError>` - The trajectory, or an error if the
    ///   satellite has a gap, naming the satellite and, for each gap, the epoch of the last
    ///   sample before it and of the first sample after it; or an error naming the
    ///   satellite and epoch if it has a single sample
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
    /// Lagrange degree raised to at least 1 and reduced to at most the run
    /// length minus one. Runs of a single sample are omitted.
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
    /// The method applies to the trajectories returned afterwards. For each
    /// run's trajectory a Lagrange degree is raised to at least 1 and reduced
    /// to at most the run length minus one. The fit that derives velocities is
    /// not affected.
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
    /// The method applies to the trajectories returned afterwards. For each
    /// run's trajectory a Lagrange degree is raised to at least 1 and reduced
    /// to at most the run length minus one. The fit that derives velocities is
    /// not affected.
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
    ///   adjustment; defaults to `Lagrange { degree: SP3_DEFAULT_LAGRANGE_DEGREE }`
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

impl Identifiable for SP3Satellite {
    fn with_name(mut self, name: &str) -> Self {
        self.set_name(Some(name));
        self
    }

    fn with_uuid(mut self, uuid: Uuid) -> Self {
        self.uuid = Some(uuid);
        self.sync_trajectory_identity();
        self
    }

    fn with_new_uuid(mut self) -> Self {
        self.generate_uuid();
        self
    }

    fn with_id(mut self, id: u64) -> Self {
        self.set_id(Some(id));
        self
    }

    fn with_identity(mut self, name: Option<&str>, uuid: Option<Uuid>, id: Option<u64>) -> Self {
        self.set_identity(name, uuid, id);
        self
    }

    fn set_identity(&mut self, name: Option<&str>, uuid: Option<Uuid>, id: Option<u64>) {
        self.name = name.map(str::to_string);
        self.uuid = uuid;
        self.numeric_id = id;
        self.sync_trajectory_identity();
    }

    fn set_id(&mut self, id: Option<u64>) {
        self.numeric_id = id;
        self.sync_trajectory_identity();
    }

    fn set_name(&mut self, name: Option<&str>) {
        self.name = name.map(str::to_string);
        self.sync_trajectory_identity();
    }

    fn generate_uuid(&mut self) {
        self.uuid = Some(Uuid::now_v7());
        self.sync_trajectory_identity();
    }

    fn get_id(&self) -> Option<u64> {
        self.numeric_id
    }

    fn get_name(&self) -> Option<&str> {
        self.name.as_deref()
    }

    fn get_uuid(&self) -> Option<Uuid> {
        self.uuid
    }
}

impl DStateProvider for SP3Satellite {
    /// Returns the state in ITRF at `epoch`: position (m) then velocity (m/s).
    fn state(&self, epoch: Epoch) -> Result<DVector<f64>, BraheError> {
        self.segment_for(epoch)?.state(epoch)
    }

    fn state_dim(&self) -> usize {
        6
    }
}

impl DOrbitStateProvider for SP3Satellite {
    fn state_eci(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        self.segment_for(epoch)?.state_eci(epoch)
    }

    fn state_ecef(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        self.segment_for(epoch)?.state_ecef(epoch)
    }

    fn state_gcrf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        self.segment_for(epoch)?.state_gcrf(epoch)
    }

    fn state_itrf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        self.segment_for(epoch)?.state_itrf(epoch)
    }

    fn state_eme2000(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        self.segment_for(epoch)?.state_eme2000(epoch)
    }

    fn state_bci(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        self.segment_for(epoch)?.state_bci(epoch)
    }

    fn state_bcbf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        self.segment_for(epoch)?.state_bcbf(epoch)
    }

    fn state_in_frame(
        &self,
        frame: CelestialFrame,
        epoch: Epoch,
    ) -> Result<Vector6<f64>, BraheError> {
        self.segment_for(epoch)?.state_in_frame(frame, epoch)
    }

    fn state_koe_osc(
        &self,
        epoch: Epoch,
        angle_format: AngleFormat,
    ) -> Result<Vector6<f64>, BraheError> {
        self.segment_for(epoch)?.state_koe_osc(epoch, angle_format)
    }

    fn state_koe_mean(
        &self,
        epoch: Epoch,
        angle_format: AngleFormat,
    ) -> Result<Vector6<f64>, BraheError> {
        self.segment_for(epoch)?.state_koe_mean(epoch, angle_format)
    }
}

/// Interpolation method for a run of `run_len >= 2` samples, with a
/// Lagrange degree limited to `1..=run_len - 1`.
fn run_method(method: InterpolationMethod, run_len: usize) -> InterpolationMethod {
    match method {
        InterpolationMethod::Lagrange { degree } => InterpolationMethod::Lagrange {
            degree: degree.max(1).min(run_len - 1),
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
    use crate::access::{ElevationConstraint, PointLocation, location_accesses};
    use crate::constants::AngleFormat;
    use crate::datafiles::sp3::types::{SP3, SP3WriteOptions};
    use crate::frames::{CelestialFrame, ReferenceFrame};
    use crate::propagators::KeplerianPropagator;
    use crate::time::TimeSystem;
    use crate::trajectories::traits::InterpolatableTrajectory;
    use crate::utils::testing::setup_global_test_eop;
    use crate::utils::{DOrbitStateProvider, DStateProvider, Identifiable};
    use approx::assert_abs_diff_eq;
    use nalgebra::Vector6;
    use serial_test::{parallel, serial};
    use uuid::Uuid;

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
    #[parallel]
    fn test_SP3Satellite_lagrange_degree_zero_raised_to_one() {
        let t0 = t0();
        let samples = (0..4)
            .map(|k| {
                sample(
                    t0 + 900.0 * k as f64,
                    quadratic_position(900.0 * k as f64),
                    900.0,
                )
            })
            .collect();
        let sat = SP3Satellite::from_samples(sid("G14"), samples)
            .unwrap()
            .with_interpolation_method(InterpolationMethod::Lagrange { degree: 0 });
        assert_eq!(
            sat.get_interpolation_method(),
            InterpolationMethod::Lagrange { degree: 0 }
        );
        let segments = sat.to_trajectory_segments().unwrap();
        assert_eq!(segments.len(), 1);
        assert_eq!(
            segments[0].get_interpolation_method(),
            InterpolationMethod::Lagrange { degree: 1 }
        );
        let state = segments[0].interpolate(&(t0 + 450.0)).unwrap();
        let p0 = quadratic_position(0.0);
        let p1 = quadratic_position(900.0);
        for axis in 0..3 {
            assert_abs_diff_eq!(state[axis], 0.5 * (p0[axis] + p1[axis]), epsilon = 1e-6);
        }
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

    /// A 24 h position-only file at 900 s with three gap-free satellites,
    /// and the propagators it was sampled from, named after the satellites.
    fn build_clean_sp3() -> (SP3, Vec<KeplerianPropagator>) {
        setup_global_test_eop();
        let names = ["G01", "G02", "G03"];
        let ids: Vec<SP3SatelliteId> = names.iter().map(|n| sid(n)).collect();
        let propagators: Vec<KeplerianPropagator> = [(0.0, 0.0), (60.0, 30.0), (120.0, 60.0)]
            .iter()
            .zip(names)
            .map(|(&(raan, anomaly), name)| gps_propagator(raan, anomaly).with_name(name))
            .collect();
        let mut sp3 = SP3::new(
            ids.clone(),
            STEP,
            SP3WriteOptions::default().with_velocity(false),
        )
        .unwrap();
        for k in 0..N_EPOCHS {
            let epoch = t0() + k as f64 * STEP;
            let index = sp3.push_epoch(epoch).unwrap();
            for (id, propagator) in ids.iter().zip(&propagators) {
                let x = propagator.state_itrf(epoch).unwrap();
                sp3.set_record(*id, index, SP3Record::new([x[0], x[1], x[2]]))
                    .unwrap();
            }
        }
        (sp3, propagators)
    }

    #[test]
    #[serial]
    fn test_SP3Satellite_DOrbitStateProvider_returns_file_positions_at_run_edges() {
        let (sp3, _) = build_sp3();
        let g01 = sp3.satellite(sid("G01")).unwrap();
        let positions = g01.positions();
        let epochs = g01.epochs();
        // First sample, both samples bordering the gap, and last sample.
        for i in [0, 39, 40, g01.len() - 1] {
            let state = g01.state_itrf(epochs[i]).unwrap();
            for axis in 0..3 {
                assert_abs_diff_eq!(state[axis], positions[i][axis], epsilon = 1e-6);
            }
            let velocity = g01.velocities()[i];
            for axis in 0..3 {
                assert_abs_diff_eq!(state[3 + axis], velocity[axis], epsilon = 1e-6);
            }
        }
        assert_eq!(g01.state_dim(), 6);
        assert_eq!(g01.state(epochs[0]).unwrap().len(), 6);
    }

    #[test]
    #[serial]
    fn test_SP3Satellite_DOrbitStateProvider_errors_outside_runs() {
        let (sp3, _) = build_sp3();
        let t0 = t0();
        let g01 = sp3.satellite(sid("G01")).unwrap();

        let gap_midpoint = t0 + 41.5 * STEP;
        let err = g01.state_itrf(gap_midpoint).unwrap_err().to_string();
        assert!(err.contains("G01"), "{err}");
        let gap_start: Epoch = t0 + 39.0 * STEP;
        let gap_end: Epoch = t0 + 44.0 * STEP;
        assert!(
            err.contains(&gap_start.to_string()) && err.contains(&gap_end.to_string()),
            "{err}"
        );

        let before = g01.state_itrf(t0 - 1.0).unwrap_err().to_string();
        assert!(
            before.contains("G01") && before.contains(&t0.to_string()),
            "{before}"
        );
        let end: Epoch = t0 + 96.0 * STEP;
        let after = g01.state_itrf(end + 1.0).unwrap_err().to_string();
        assert!(
            after.contains("G01") && after.contains(&end.to_string()),
            "{after}"
        );

        // Every provider method and batch form reports the same error.
        assert!(g01.state(gap_midpoint).is_err());
        assert!(g01.state_eci(gap_midpoint).is_err());
        assert!(g01.state_gcrf(gap_midpoint).is_err());
        assert!(g01.state_ecef(gap_midpoint).is_err());
        assert!(g01.state_eme2000(gap_midpoint).is_err());
        assert!(g01.state_bci(gap_midpoint).is_err());
        assert!(g01.state_bcbf(gap_midpoint).is_err());
        assert!(
            g01.state_in_frame(CelestialFrame::GCRF, gap_midpoint)
                .is_err()
        );
        assert!(
            g01.state_koe_osc(gap_midpoint, AngleFormat::Degrees)
                .is_err()
        );
        assert!(
            g01.state_koe_mean(gap_midpoint, AngleFormat::Degrees)
                .is_err()
        );
        assert!(g01.states_eci(&[t0, gap_midpoint]).is_err());
    }

    #[test]
    #[serial]
    fn test_SP3Satellite_DOrbitStateProvider_isolated_sample_errors() {
        let (sp3, _) = build_sp3();
        let t0 = t0();
        let g03 = sp3.satellite(sid("G03")).unwrap();
        let isolated: Epoch = t0 + 20.0 * STEP;
        let err = g03.state_itrf(isolated).unwrap_err().to_string();
        assert!(
            err.contains("G03") && err.contains(&isolated.to_string()),
            "{err}"
        );
        assert!(g03.state_itrf(t0 + 15.0 * STEP).is_err());
        assert!(g03.state_itrf(t0 + 5.0 * STEP).is_ok());
    }

    #[test]
    #[serial]
    fn test_SP3Satellite_DOrbitStateProvider_matches_segment_trajectory() {
        let (sp3, _) = build_sp3();
        let g01 = sp3.satellite(sid("G01")).unwrap();
        let segments = g01.to_trajectory_segments().unwrap();
        let t = t0() + 10.0 * STEP + 137.0;
        let from_satellite = g01.state_eci(t).unwrap();
        let from_segment = segments[0].state_eci(t).unwrap();
        for axis in 0..6 {
            assert_abs_diff_eq!(from_satellite[axis], from_segment[axis], epsilon = 1e-9);
        }

        // The second run dispatches to the second segment.
        let t = t0() + 60.0 * STEP + 11.0;
        let from_satellite = g01.state_itrf(t).unwrap();
        let from_segment = segments[1].state_itrf(t).unwrap();
        for axis in 0..6 {
            assert_abs_diff_eq!(from_satellite[axis], from_segment[axis], epsilon = 1e-9);
        }

        let epochs = [t0() + 100.0, t0() + 60.0 * STEP + 11.0];
        let batch = g01.states_gcrf(&epochs).unwrap();
        for (state, epoch) in batch.iter().zip(epochs) {
            assert_eq!(*state, g01.state_gcrf(epoch).unwrap());
        }
    }

    #[test]
    #[serial]
    fn test_SP3Satellite_DOrbitStateProvider_follows_interpolation_method() {
        let (sp3, _) = build_sp3();
        let t = t0() + 10.0 * STEP + 450.0;
        let default = sp3.satellite(sid("G01")).unwrap();
        let low = default
            .clone()
            .with_interpolation_method(InterpolationMethod::Lagrange { degree: 1 });
        let default_state = default.state_itrf(t).unwrap();
        let low_state = low.state_itrf(t).unwrap();
        assert!((default_state - low_state).norm() > 1.0);
    }

    #[test]
    #[serial]
    fn test_SP3Satellite_Identifiable() {
        let (sp3, _) = build_sp3();
        let mut sat = sp3.satellite(sid("G01")).unwrap();
        assert_eq!(sat.get_name(), Some("G01"));
        assert_eq!(sat.get_id(), None);
        assert_eq!(sat.get_uuid(), None);

        sat.set_name(Some("Block IIF"));
        assert_eq!(sat.get_name(), Some("Block IIF"));
        sat.set_id(Some(7));
        assert_eq!(sat.get_id(), Some(7));
        let uuid = Uuid::now_v7();
        let sat = sat.with_uuid(uuid);
        assert_eq!(sat.get_uuid(), Some(uuid));

        let renamed = sp3.satellite(sid("G02")).unwrap().with_name("spare");
        assert_eq!(renamed.get_name(), Some("spare"));
        for traj in renamed.to_trajectory_segments().unwrap() {
            assert_eq!(traj.get_name(), Some("spare"));
        }

        let mut sat = sp3
            .satellite(sid("G02"))
            .unwrap()
            .with_new_uuid()
            .with_id(3);
        assert!(sat.get_uuid().is_some());
        sat.generate_uuid();
        sat.set_identity(Some("a"), None, Some(9));
        assert_eq!(
            (sat.get_name(), sat.get_uuid(), sat.get_id()),
            (Some("a"), None, Some(9))
        );
        let sat = sat.with_identity(None, Some(uuid), None);
        assert_eq!(
            (sat.get_name(), sat.get_uuid(), sat.get_id()),
            (None, Some(uuid), None)
        );
    }

    fn assert_windows_match(
        sp3_windows: &[crate::access::AccessWindow],
        reference: &[crate::access::AccessWindow],
    ) {
        assert!(!reference.is_empty());
        assert_eq!(sp3_windows.len(), reference.len());
        for (a, b) in sp3_windows.iter().zip(reference) {
            assert!((a.window_open - b.window_open).abs() < 1.0);
            assert!((a.window_close - b.window_close).abs() < 1.0);
            assert_eq!(a.satellite_name, b.satellite_name);
        }
    }

    #[test]
    #[serial]
    fn test_SP3Satellite_location_accesses_matches_propagator() {
        let (sp3, propagators) = build_clean_sp3();
        let station = PointLocation::new(10.0, 45.0, 0.0).unwrap();
        let constraint = ElevationConstraint::new(Some(10.0), None).unwrap();
        let (start, end) = (t0(), t0() + 86400.0);

        let g01 = sp3.satellite(sid("G01")).unwrap();
        let windows =
            location_accesses(&station, &g01, start, end, &constraint, None, None).unwrap();
        let reference = location_accesses(
            &station,
            &propagators[0],
            start,
            end,
            &constraint,
            None,
            None,
        )
        .unwrap();
        assert_windows_match(&windows, &reference);
        assert_eq!(windows[0].satellite_name.as_deref(), Some("G01"));
    }

    #[test]
    #[serial]
    fn test_SP3Satellite_location_accesses_constellation() {
        let (sp3, propagators) = build_clean_sp3();
        let station = PointLocation::new(10.0, 45.0, 0.0).unwrap();
        let constraint = ElevationConstraint::new(Some(10.0), None).unwrap();
        let (start, end) = (t0(), t0() + 86400.0);

        let satellites = sp3.satellites().unwrap();
        assert_eq!(satellites.len(), 3);
        let windows =
            location_accesses(&station, &satellites, start, end, &constraint, None, None).unwrap();
        let reference =
            location_accesses(&station, &propagators, start, end, &constraint, None, None).unwrap();
        assert_windows_match(&windows, &reference);
        for name in ["G01", "G02", "G03"] {
            assert!(
                windows
                    .iter()
                    .any(|w| w.satellite_name.as_deref() == Some(name)),
                "no window for {name}"
            );
        }
    }
}
