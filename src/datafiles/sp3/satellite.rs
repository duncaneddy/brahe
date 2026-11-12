/*!
 * Per-satellite view of SP3 precise orbit data.
 *
 * [`SP3Satellite`] holds the valid samples of one satellite, interpolates
 * orbit states from them, and converts them to ITRF trajectories.
 */

use std::collections::BTreeSet;

use nalgebra::{DVector, Vector3, Vector6};
use uuid::Uuid;

use crate::constants::AngleFormat;
use crate::coordinates::state_eci_to_koe;
use crate::datafiles::sp3::parse::sp3_error;
use crate::datafiles::sp3::types::{GRID_TOLERANCE, SP3, SP3Record, SP3SatelliteId};
use crate::frames::{CelestialFrame, state_frame_to_frame};
use crate::math::interpolation::{InterpolationMethod, interpolate_lagrange_svector};
use crate::time::Epoch;
use crate::trajectories::dorbit_trajectory::DOrbitTrajectory;
use crate::trajectories::traits::{OrbitRepresentation, Trajectory};
use crate::utils::{BraheError, DOrbitStateProvider, DStateProvider, Identifiable};

/// Degree of the Lagrange polynomials an [`SP3`] gives its satellites unless
/// its `interpolation_degree` is changed. A degree of 9 fits 10 samples.
pub(crate) const DEFAULT_INTERPOLATION_DEGREE: usize = 9;

/// Spacing, as a multiple of the epoch interval, above which two consecutive
/// samples have a gap between them.
const GAP_FACTOR: f64 = 1.5;

/// Orbit and clock samples of one satellite from SP3 data.
///
/// Only epochs where the satellite has a valid position are kept, and
/// [`SP3Satellite::records`] holds one [`SP3Record`] per epoch of
/// [`SP3Satellite::epochs`]. Positions and velocities are in the file's
/// Earth-fixed frame (ITRF), in m and m/s.
///
/// The satellite is an orbit state provider ([`DStateProvider`] and
/// [`DOrbitStateProvider`]) and is [`Identifiable`], named after its
/// identifier (such as `"G01"`), so it can be passed to
/// [`location_accesses`](crate::access::location_accesses). A state is
/// interpolated in ITRF with a Lagrange polynomial of degree
/// [`SP3Satellite::interpolation_degree`] through the samples around the
/// requested epoch, as centered as the samples allow; a satellite with fewer
/// samples uses all of them. The velocity is interpolated from the records
/// when every sample of the fit has one, and is the derivative of the
/// position polynomial otherwise. A request before the first sample or after
/// the last one is an error, as is any request to a satellite with a single
/// sample.
///
/// Missing samples do not stop interpolation: a state between two samples
/// that are further apart than the epoch interval is fitted through the
/// samples around the hole and loses accuracy as the hole grows.
/// [`SP3Satellite::gaps`] lists those holes.
///
/// # Examples
///
/// ```
/// use brahe::datafiles::sp3::{SP3, SP3Header, SP3Record, SP3SatelliteId};
/// use brahe::time::{Epoch, TimeSystem};
/// use brahe::trajectories::traits::Trajectory;
/// use brahe::utils::{DOrbitStateProvider, Identifiable};
///
/// let id: SP3SatelliteId = "G01".parse().unwrap();
/// let header = SP3Header { has_velocity: false, ..Default::default() };
/// let mut sp3 = SP3::new(vec![id], 900.0, header).unwrap();
/// let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
/// for k in 0..4 {
///     let x = 2.0e7 + 900.0 * k as f64;
///     sp3.set_record(id, t0 + 900.0 * k as f64, SP3Record::new([x, 1.0e7, 1.0e7])).unwrap();
/// }
///
/// let sat = sp3.satellite(id).unwrap();
/// assert_eq!(sat.len(), 4);
/// assert_eq!(sat.get_name(), Some("G01"));
/// assert!(sat.gaps().is_empty());
///
/// let state = sat.state_itrf(t0 + 450.0).unwrap();
/// assert!((state[0] - 2.0e7 - 450.0).abs() < 1e-6);
/// assert!((state[3] - 1.0).abs() < 1e-9);
///
/// assert_eq!(sat.to_trajectory().unwrap().len(), 4);
/// ```
#[derive(Debug, Clone)]
pub struct SP3Satellite {
    id: SP3SatelliteId,
    epochs: Vec<Epoch>,
    records: Vec<SP3Record>,
    interval: f64,
    interpolation_degree: usize,
    name: Option<String>,
    uuid: Option<Uuid>,
    numeric_id: Option<u64>,
}

impl SP3Satellite {
    /// Build a satellite from its valid samples in epoch order.
    ///
    /// # Arguments
    ///
    /// * `id` - Satellite identifier
    /// * `samples` - Epoch and record of each valid sample
    /// * `interval` - Epoch interval of the samples. Units: s
    /// * `interpolation_degree` - Degree of the Lagrange polynomials fitted through the samples
    ///
    /// # Returns
    ///
    /// * `Result<SP3Satellite, BraheError>` - The satellite, or an error if there are no
    ///   samples, the degree is zero, or two consecutive samples are out of order or
    ///   closer than half the epoch interval
    pub(crate) fn from_samples(
        id: SP3SatelliteId,
        samples: Vec<(Epoch, SP3Record)>,
        interval: f64,
        interpolation_degree: usize,
    ) -> Result<SP3Satellite, BraheError> {
        if samples.is_empty() {
            return Err(sp3_error(&format!("satellite {} has no valid sample", id)));
        }
        check_interpolation_degree(interpolation_degree)?;
        let (epochs, records): (Vec<Epoch>, Vec<SP3Record>) = samples.into_iter().unzip();

        for pair in epochs.windows(2) {
            let spacing = pair[1] - pair[0];
            if spacing <= 0.0 {
                return Err(sp3_error(&format!(
                    "samples of satellite {} are not in increasing epoch order: {} follows {}",
                    id, pair[1], pair[0]
                )));
            }
            if spacing < 0.5 * interval {
                return Err(sp3_error(&format!(
                    "samples of satellite {} at {} and {} are {} s apart, less than half \
                     the {} s epoch interval",
                    id, pair[0], pair[1], spacing, interval
                )));
            }
        }

        Ok(SP3Satellite {
            id,
            epochs,
            records,
            interval,
            interpolation_degree,
            name: Some(id.to_string()),
            uuid: None,
            numeric_id: None,
        })
    }

    /// Build the satellites `ids` from files sorted in time order.
    ///
    /// The valid samples of each file are joined in order. Where two
    /// consecutive files share a boundary epoch, to within the tolerance of
    /// the epoch grid, the later file's sample replaces the earlier one. An
    /// identifier without a valid sample in any file is skipped. The
    /// satellites take the interpolation degree of the first file.
    ///
    /// # Arguments
    ///
    /// * `files` - SP3 files in time order, with at least one file
    /// * `ids` - Identifiers of the satellites to build
    ///
    /// # Returns
    ///
    /// * `Result<Vec<SP3Satellite>, BraheError>` - The satellites in the order of `ids`,
    ///   or an error if the files have different epoch intervals or the samples of a
    ///   satellite cannot be joined
    pub(crate) fn from_ordered_files(
        files: &[&SP3],
        ids: impl IntoIterator<Item = SP3SatelliteId>,
    ) -> Result<Vec<SP3Satellite>, BraheError> {
        let Some(first) = files.first() else {
            return Ok(Vec::new());
        };
        let interval = first.header.epoch_interval;
        if let Some(other) = files
            .iter()
            .find(|f| (f.header.epoch_interval - interval).abs() > GRID_TOLERANCE)
        {
            return Err(sp3_error(&format!(
                "files have different epoch intervals: {} s and {} s",
                interval, other.header.epoch_interval
            )));
        }

        let mut satellites = Vec::new();
        for id in ids {
            let mut samples: Vec<(Epoch, SP3Record)> = Vec::new();
            for file in files {
                let mut next = file.samples(id);
                if let (Some((last, _)), Some((first, _))) = (samples.last(), next.first())
                    && (*first - *last).abs() <= GRID_TOLERANCE
                {
                    samples.pop();
                }
                samples.append(&mut next);
            }
            if !samples.is_empty() {
                satellites.push(SP3Satellite::from_samples(
                    id,
                    samples,
                    interval,
                    first.interpolation_degree,
                )?);
            }
        }
        Ok(satellites)
    }

    /// Chain every satellite across several SP3 files.
    ///
    /// The files are sorted by start epoch, so they may be passed in any
    /// order, and files without epochs are ignored. Each satellite in the
    /// union of the files' satellite lists collects its valid samples from
    /// every file that contains it; a file that lacks a satellite leaves a
    /// gap in it. Two consecutive files may share their boundary epoch, where
    /// the later file's sample is kept. The files must have the same epoch
    /// interval. Time systems, agencies, coordinate system labels, and orbit
    /// types need not match. The satellites take the interpolation degree of
    /// the earliest file.
    ///
    /// # Arguments
    ///
    /// * `files` - SP3 files covering consecutive, non-overlapping spans
    ///
    /// # Returns
    ///
    /// * `Result<Vec<SP3Satellite>, BraheError>` - One satellite for each identifier with
    ///   at least one valid sample, in ascending identifier order, or an error if the
    ///   files have different epoch intervals, two files overlap by more than a shared
    ///   boundary epoch, or two consecutive samples are closer than half the epoch
    ///   interval
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{SP3, SP3Header, SP3Record, SP3Satellite, SP3SatelliteId};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let id: SP3SatelliteId = "G01".parse().unwrap();
    /// let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// let file = |start: usize| {
    ///     let header = SP3Header { has_velocity: false, ..Default::default() };
    ///     let mut sp3 = SP3::new(vec![id], 900.0, header).unwrap();
    ///     for k in start..start + 3 {
    ///         sp3.set_epoch(t0 + 900.0 * k as f64, &[]).unwrap();
    ///         let x = 2.0e7 + 900.0 * k as f64;
    ///         sp3.set_record(id, t0 + 900.0 * k as f64, SP3Record::new([x, 1.0e7, 1.0e7])).unwrap();
    ///     }
    ///     sp3
    /// };
    ///
    /// // Epochs 0-2 and 2-4 share epoch 2.
    /// let sats = SP3Satellite::from_files(&[file(2), file(0)]).unwrap();
    /// assert_eq!(sats.len(), 1);
    /// assert_eq!(sats[0].len(), 5);
    ///
    /// // A file with two or more epochs passed twice overlaps itself.
    /// assert!(SP3Satellite::from_files(&[file(0), file(0)]).is_err());
    /// ```
    pub fn from_files(files: &[SP3]) -> Result<Vec<SP3Satellite>, BraheError> {
        let mut spans: Vec<(Epoch, Epoch, &SP3)> = files
            .iter()
            .filter_map(|f| Some((f.start_epoch()?, f.end_epoch()?, f)))
            .collect();
        spans.sort_by_key(|(start, end, _)| (*start, *end));
        for pair in spans.windows(2) {
            let ((a_start, a_end, a), (b_start, b_end, b)) = (pair[0], pair[1]);
            if b_start - a_end < -GRID_TOLERANCE {
                return Err(sp3_error(&format!(
                    "files overlap: {} file from {} to {} and {} file from {} to {}; consecutive \
                     files may share only their boundary epoch",
                    a.header.agency, a_start, a_end, b.header.agency, b_start, b_end
                )));
            }
        }
        let ordered: Vec<&SP3> = spans.iter().map(|(_, _, f)| *f).collect();
        let ids: BTreeSet<SP3SatelliteId> = ordered
            .iter()
            .flat_map(|f| f.satellite_ids().iter().copied())
            .collect();
        SP3Satellite::from_ordered_files(&ordered, ids)
    }

    /// ITRF state at an epoch, interpolated from the samples around it.
    fn interpolate(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        self.interpolate_at(epoch, self.interpolation_degree)
    }

    /// ITRF state at an epoch from a Lagrange fit of at most `degree`.
    fn interpolate_at(&self, epoch: Epoch, degree: usize) -> Result<Vector6<f64>, BraheError> {
        let n = self.epochs.len();
        if n < 2 || epoch < self.epochs[0] || epoch > self.epochs[n - 1] {
            return Err(self.unavailable_error(epoch));
        }

        let m = degree.saturating_add(1).min(n);
        let before = self.epochs.partition_point(|&e| e <= epoch) - 1;
        let offset = before.saturating_sub((m - 1) / 2).min(n - m);
        let window = offset..(offset + m);

        let origin = self.epochs[window.start];
        let times: Vec<f64> = self.epochs[window.clone()]
            .iter()
            .map(|&e| e - origin)
            .collect();
        let t = epoch - origin;
        let records = &self.records[window];
        let positions: Vec<Vector3<f64>> =
            records.iter().map(|r| Vector3::from(r.position)).collect();
        let file_velocities: Option<Vec<Vector3<f64>>> = records
            .iter()
            .map(|r| r.velocity.map(Vector3::from))
            .collect();

        let position = interpolate_lagrange_svector(&times, &positions, t);
        let velocity = match file_velocities {
            Some(velocities) => interpolate_lagrange_svector(&times, &velocities, t),
            None => lagrange_derivative(&times, &positions, t),
        };
        Ok(Vector6::new(
            position[0],
            position[1],
            position[2],
            velocity[0],
            velocity[1],
            velocity[2],
        ))
    }

    /// Error for a state request the samples cannot answer.
    fn unavailable_error(&self, epoch: Epoch) -> BraheError {
        let (first, last) = (self.start_epoch(), self.end_epoch());
        let detail = if self.epochs.len() < 2 {
            format!(
                "satellite {} has a single sample at {}; a state needs at least two samples",
                self.id, first
            )
        } else {
            format!(
                "satellite {} has samples from {} to {}; epoch {} is outside that span",
                self.id, first, last, epoch
            )
        };
        BraheError::OutOfBoundsError(format!("SP3: {}", detail))
    }

    /// Degree of the Lagrange polynomials fitted through the samples.
    ///
    /// # Returns
    ///
    /// * `usize` - The degree; a degree of 9 fits 10 samples
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Header, SP3Record, SP3SatelliteId};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3Header::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # sp3.set_record(id, t0, SP3Record::new([2.0e7, 1.0e7, 1.0e7])).unwrap();
    /// assert_eq!(sp3.satellite(id).unwrap().interpolation_degree(), 9);
    ///
    /// sp3.interpolation_degree = 7;
    /// assert_eq!(sp3.satellite(id).unwrap().interpolation_degree(), 7);
    /// ```
    pub fn interpolation_degree(&self) -> usize {
        self.interpolation_degree
    }

    /// Set the degree of the Lagrange polynomials fitted through the samples.
    ///
    /// # Arguments
    ///
    /// * `degree` - Degree of at least 1; a fit uses `degree + 1` samples, or every
    ///   sample when the satellite has fewer
    ///
    /// # Returns
    ///
    /// * `Result<(), BraheError>` - An error if `degree` is zero
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Header, SP3Record, SP3SatelliteId};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3Header::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # sp3.set_record(id, t0, SP3Record::new([2.0e7, 1.0e7, 1.0e7])).unwrap();
    /// let mut sat = sp3.satellite(id).unwrap();
    /// sat.set_interpolation_degree(5).unwrap();
    /// assert_eq!(sat.interpolation_degree(), 5);
    /// assert!(sat.set_interpolation_degree(0).is_err());
    /// ```
    pub fn set_interpolation_degree(&mut self, degree: usize) -> Result<(), BraheError> {
        check_interpolation_degree(degree)?;
        self.interpolation_degree = degree;
        Ok(())
    }

    /// Return the satellite with another interpolation degree.
    ///
    /// # Arguments
    ///
    /// * `degree` - Degree of at least 1; a fit uses `degree + 1` samples, or every
    ///   sample when the satellite has fewer
    ///
    /// # Returns
    ///
    /// * `Result<SP3Satellite, BraheError>` - The satellite, or an error if `degree` is zero
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Header, SP3Record, SP3SatelliteId};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3Header::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # sp3.set_record(id, t0, SP3Record::new([2.0e7, 1.0e7, 1.0e7])).unwrap();
    /// let sat = sp3.satellite(id).unwrap().with_interpolation_degree(5).unwrap();
    /// assert_eq!(sat.interpolation_degree(), 5);
    /// ```
    pub fn with_interpolation_degree(mut self, degree: usize) -> Result<Self, BraheError> {
        self.set_interpolation_degree(degree)?;
        Ok(self)
    }

    /// Spans where samples are missing.
    ///
    /// Two consecutive samples more than 1.5 epoch intervals apart have a gap
    /// between them. A satellite with zero positions at 10:15 and 10:30 in a
    /// 15-minute file has one gap, from 10:00 to 10:45.
    ///
    /// # Returns
    ///
    /// * `Vec<(Epoch, Epoch)>` - The last sample before and the first sample after each
    ///   gap, in time order
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{SP3, SP3Header, SP3Record, SP3SatelliteId};
    /// use brahe::time::{Epoch, TimeSystem};
    ///
    /// let id: SP3SatelliteId = "G01".parse().unwrap();
    /// let mut sp3 = SP3::new(vec![id], 900.0, SP3Header::default()).unwrap();
    /// let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// for k in [0, 1, 2, 5, 6] {
    ///     let y = 1.0e7 + 1000.0 * k as f64;
    ///     sp3.set_record(id, t0 + 900.0 * k as f64, SP3Record::new([2.0e7, y, 1.0e7])).unwrap();
    /// }
    ///
    /// let gaps = sp3.satellite(id).unwrap().gaps();
    /// assert_eq!(gaps, vec![(t0 + 1800.0, t0 + 4500.0)]);
    /// ```
    pub fn gaps(&self) -> Vec<(Epoch, Epoch)> {
        self.epochs
            .windows(2)
            .filter(|pair| pair[1] - pair[0] > GAP_FACTOR * self.interval)
            .map(|pair| (pair[0], pair[1]))
            .collect()
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
    /// # use brahe::datafiles::sp3::{SP3, SP3Header, SP3Record, SP3SatelliteId};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3Header::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # sp3.set_epoch(t0, &[]).unwrap();
    /// # sp3.set_record(id, sp3.epochs()[0], SP3Record::new([2.0e7, 1.0e7, 1.0e7])).unwrap();
    /// assert_eq!(sp3.satellite(id).unwrap().id(), id);
    /// ```
    pub fn id(&self) -> SP3SatelliteId {
        self.id
    }

    /// Number of valid samples.
    ///
    /// # Returns
    ///
    /// * `usize` - The sample count
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Header, SP3Record, SP3SatelliteId};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3Header::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # sp3.set_epoch(t0, &[]).unwrap();
    /// # sp3.set_record(id, sp3.epochs()[0], SP3Record::new([2.0e7, 1.0e7, 1.0e7])).unwrap();
    /// assert_eq!(sp3.satellite(id).unwrap().len(), 1);
    /// ```
    pub fn len(&self) -> usize {
        self.epochs.len()
    }

    /// Whether the satellite has no samples, which never holds for a
    /// satellite built from SP3 data.
    ///
    /// # Returns
    ///
    /// * `bool` - `false`
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Header, SP3Record, SP3SatelliteId};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3Header::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # sp3.set_epoch(t0, &[]).unwrap();
    /// # sp3.set_record(id, sp3.epochs()[0], SP3Record::new([2.0e7, 1.0e7, 1.0e7])).unwrap();
    /// assert!(!sp3.satellite(id).unwrap().is_empty());
    /// ```
    pub fn is_empty(&self) -> bool {
        self.epochs.is_empty()
    }

    /// Epoch of the first valid sample.
    ///
    /// # Returns
    ///
    /// * `Epoch` - The first epoch
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Header, SP3Record, SP3SatelliteId};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3Header::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # sp3.set_epoch(t0, &[]).unwrap();
    /// # sp3.set_record(id, sp3.epochs()[0], SP3Record::new([2.0e7, 1.0e7, 1.0e7])).unwrap();
    /// assert_eq!(sp3.satellite(id).unwrap().start_epoch(), t0);
    /// ```
    pub fn start_epoch(&self) -> Epoch {
        self.epochs[0]
    }

    /// Epoch of the last valid sample.
    ///
    /// # Returns
    ///
    /// * `Epoch` - The last epoch
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Header, SP3Record, SP3SatelliteId};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3Header::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # sp3.set_epoch(t0, &[]).unwrap();
    /// # sp3.set_record(id, sp3.epochs()[0], SP3Record::new([2.0e7, 1.0e7, 1.0e7])).unwrap();
    /// assert_eq!(sp3.satellite(id).unwrap().end_epoch(), t0);
    /// ```
    pub fn end_epoch(&self) -> Epoch {
        self.epochs[self.epochs.len() - 1]
    }

    /// Epochs of the valid samples, in ascending order.
    ///
    /// # Returns
    ///
    /// * `&[Epoch]` - One epoch per sample
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Header, SP3Record, SP3SatelliteId};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3Header::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # sp3.set_epoch(t0, &[]).unwrap();
    /// # sp3.set_record(id, sp3.epochs()[0], SP3Record::new([2.0e7, 1.0e7, 1.0e7])).unwrap();
    /// assert_eq!(sp3.satellite(id).unwrap().epochs(), &[t0]);
    /// ```
    pub fn epochs(&self) -> &[Epoch] {
        &self.epochs
    }

    /// ITRF positions of the valid samples.
    ///
    /// # Returns
    ///
    /// * `Vec<[f64; 3]>` - One position per sample. Units: m
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Header, SP3Record, SP3SatelliteId};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3Header::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # sp3.set_epoch(t0, &[]).unwrap();
    /// sp3.set_record(id, sp3.epochs()[0], SP3Record::new([2.0e7, 1.0e7, 1.0e7])).unwrap();
    /// assert_eq!(sp3.satellite(id).unwrap().positions(), vec![[2.0e7, 1.0e7, 1.0e7]]);
    /// ```
    pub fn positions(&self) -> Vec<[f64; 3]> {
        self.records.iter().map(|r| r.position).collect()
    }

    /// ITRF velocities the file carries for the valid samples.
    ///
    /// Only velocity records read from the file are returned; nothing is
    /// derived from the positions. Use the state methods for a velocity at any
    /// epoch.
    ///
    /// # Returns
    ///
    /// * `Vec<Option<[f64; 3]>>` - One velocity per sample, `None` where the file has
    ///   none. Units: m/s
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Header, SP3Record, SP3SatelliteId};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3Header::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # sp3.set_epoch(t0, &[]).unwrap();
    /// let record = SP3Record::new([2.0e7, 1.0e7, 1.0e7]).with_velocity([0.0, 3.0e3, 0.0]);
    /// sp3.set_record(id, sp3.epochs()[0], record).unwrap();
    /// assert_eq!(sp3.satellite(id).unwrap().velocities(), vec![Some([0.0, 3.0e3, 0.0])]);
    /// ```
    pub fn velocities(&self) -> Vec<Option<[f64; 3]>> {
        self.records.iter().map(|r| r.velocity).collect()
    }

    /// Clock corrections of the valid samples.
    ///
    /// # Returns
    ///
    /// * `Vec<Option<f64>>` - One clock correction per sample, `None` where the file has
    ///   none. Units: s
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Header, SP3Record, SP3SatelliteId};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3Header::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # sp3.set_epoch(t0, &[]).unwrap();
    /// sp3.set_record(id, sp3.epochs()[0], SP3Record::new([2.0e7, 1.0e7, 1.0e7]).with_clock(1.5e-5)).unwrap();
    /// assert_eq!(sp3.satellite(id).unwrap().clocks(), vec![Some(1.5e-5)]);
    /// ```
    pub fn clocks(&self) -> Vec<Option<f64>> {
        self.records.iter().map(|r| r.clock).collect()
    }

    /// Clock rates of the valid samples.
    ///
    /// # Returns
    ///
    /// * `Vec<Option<f64>>` - One clock rate per sample, `None` where the file has none.
    ///   Units: s/s
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Header, SP3Record, SP3SatelliteId};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3Header::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # sp3.set_epoch(t0, &[]).unwrap();
    /// let record = SP3Record::new([2.0e7, 1.0e7, 1.0e7])
    ///     .with_velocity([0.0, 3.0e3, 0.0])
    ///     .with_clock_rate(1.0e-12);
    /// sp3.set_record(id, sp3.epochs()[0], record).unwrap();
    /// assert_eq!(sp3.satellite(id).unwrap().clock_rates(), vec![Some(1.0e-12)]);
    /// ```
    pub fn clock_rates(&self) -> Vec<Option<f64>> {
        self.records.iter().map(|r| r.clock_rate).collect()
    }

    /// Records of the valid samples, aligned with [`SP3Satellite::epochs`].
    ///
    /// Each record holds the position, the velocity and clock values the file
    /// carries, their standard deviations, the flags, and the correlations.
    ///
    /// # Returns
    ///
    /// * `&[SP3Record]` - One record per sample
    ///
    /// # Examples
    ///
    /// ```
    /// # use brahe::datafiles::sp3::{SP3, SP3Header, SP3Record, SP3SatelliteId};
    /// # use brahe::time::{Epoch, TimeSystem};
    /// # let id: SP3SatelliteId = "G01".parse().unwrap();
    /// # let mut sp3 = SP3::new(vec![id], 900.0, SP3Header::default()).unwrap();
    /// # let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// # sp3.set_epoch(t0, &[]).unwrap();
    /// sp3.set_record(id, sp3.epochs()[0], SP3Record::new([2.0e7, 1.0e7, 1.0e7]).with_clock(1.5e-5)).unwrap();
    ///
    /// let sat = sp3.satellite(id).unwrap();
    /// assert_eq!(sat.records()[0].position, [2.0e7, 1.0e7, 1.0e7]);
    /// assert_eq!(sat.records()[0].clock, Some(1.5e-5));
    /// ```
    pub fn records(&self) -> &[SP3Record] {
        &self.records
    }

    /// Convert the samples to an ITRF trajectory.
    ///
    /// The trajectory is a six-element Cartesian [`DOrbitTrajectory`] in the
    /// ITRF frame with the name, ID, and UUID of the satellite and Lagrange
    /// interpolation of degree [`SP3Satellite::interpolation_degree`], or the
    /// number of samples less one for a satellite with fewer samples. The
    /// state at each sample is the one the satellite provides at that epoch,
    /// so a velocity comes from the records only when every sample of the
    /// fit around it has one, and from the position fit otherwise. Gaps are
    /// not marked in the trajectory; see [`SP3Satellite::gaps`].
    ///
    /// # Returns
    ///
    /// * `Result<DOrbitTrajectory, BraheError>` - The trajectory, or an error for a
    ///   satellite with a single sample
    ///
    /// # Examples
    ///
    /// ```
    /// use brahe::datafiles::sp3::{SP3, SP3Header, SP3Record, SP3SatelliteId};
    /// use brahe::time::{Epoch, TimeSystem};
    /// use brahe::trajectories::traits::Trajectory;
    ///
    /// let id: SP3SatelliteId = "G01".parse().unwrap();
    /// let header = SP3Header { has_velocity: false, ..Default::default() };
    /// let mut sp3 = SP3::new(vec![id], 900.0, header).unwrap();
    /// let t0 = Epoch::from_datetime(2024, 11, 10, 0, 0, 0.0, 0.0, TimeSystem::GPS);
    /// for k in 0..5 {
    ///     let y = 1.0e7 + 1000.0 * k as f64;
    ///     sp3.set_record(id, t0 + 900.0 * k as f64, SP3Record::new([2.0e7, y, 1.0e7])).unwrap();
    /// }
    ///
    /// let trajectory = sp3.satellite(id).unwrap().to_trajectory().unwrap();
    /// assert_eq!(trajectory.len(), 5);
    /// assert_eq!(trajectory.start_epoch(), Some(t0));
    /// ```
    pub fn to_trajectory(&self) -> Result<DOrbitTrajectory, BraheError> {
        if self.epochs.len() < 2 {
            return Err(self.unavailable_error(self.epochs[0]));
        }
        let mut trajectory = DOrbitTrajectory::new(
            6,
            CelestialFrame::ITRF,
            OrbitRepresentation::Cartesian,
            None,
        )?
        .with_identity(self.name.as_deref(), self.uuid, self.numeric_id)
        .with_interpolation_method(InterpolationMethod::Lagrange {
            degree: self.interpolation_degree.min(self.epochs.len() - 1),
        });
        for &epoch in &self.epochs {
            let state = self.interpolate(epoch)?;
            trajectory.add(epoch, DVector::from_column_slice(state.as_slice()))?;
        }
        Ok(trajectory)
    }
}

/// Error for an interpolation degree of zero.
fn check_interpolation_degree(degree: usize) -> Result<(), BraheError> {
    if degree == 0 {
        return Err(sp3_error("interpolation degree must be at least 1"));
    }
    Ok(())
}

impl Identifiable for SP3Satellite {
    fn with_name(mut self, name: &str) -> Self {
        self.name = Some(name.to_string());
        self
    }

    fn with_uuid(mut self, uuid: Uuid) -> Self {
        self.uuid = Some(uuid);
        self
    }

    fn with_new_uuid(mut self) -> Self {
        self.uuid = Some(Uuid::now_v7());
        self
    }

    fn with_id(mut self, id: u64) -> Self {
        self.numeric_id = Some(id);
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
    }

    fn set_id(&mut self, id: Option<u64>) {
        self.numeric_id = id;
    }

    fn set_name(&mut self, name: Option<&str>) {
        self.name = name.map(str::to_string);
    }

    fn generate_uuid(&mut self) {
        self.uuid = Some(Uuid::now_v7());
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
    /// ITRF state of the satellite at an epoch.
    fn state(&self, epoch: Epoch) -> Result<DVector<f64>, BraheError> {
        let state = self.interpolate(epoch)?;
        Ok(DVector::from_column_slice(state.as_slice()))
    }

    fn state_dim(&self) -> usize {
        6
    }
}

impl DOrbitStateProvider for SP3Satellite {
    fn state_eci(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        self.state_in_frame(CelestialFrame::GCRF, epoch)
    }

    fn state_ecef(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        self.interpolate(epoch)
    }

    fn state_gcrf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        self.state_in_frame(CelestialFrame::GCRF, epoch)
    }

    fn state_itrf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        self.interpolate(epoch)
    }

    fn state_eme2000(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        self.state_in_frame(CelestialFrame::EME2000, epoch)
    }

    fn state_bci(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        self.state_in_frame(CelestialFrame::GCRF, epoch)
    }

    fn state_bcbf(&self, epoch: Epoch) -> Result<Vector6<f64>, BraheError> {
        self.interpolate(epoch)
    }

    fn state_in_frame(
        &self,
        frame: CelestialFrame,
        epoch: Epoch,
    ) -> Result<Vector6<f64>, BraheError> {
        let state = self.interpolate(epoch)?;
        if frame == CelestialFrame::ITRF {
            return Ok(state);
        }
        state_frame_to_frame(CelestialFrame::ITRF, frame, epoch, state)
    }

    fn state_koe_osc(
        &self,
        epoch: Epoch,
        angle_format: AngleFormat,
    ) -> Result<Vector6<f64>, BraheError> {
        Ok(state_eci_to_koe(self.state_gcrf(epoch)?, angle_format))
    }
}

/// Derivative at `t` of the Lagrange polynomial through `values` at `times`.
///
/// # Arguments
///
/// * `times` - Distinct sample times. Units: s
/// * `values` - Sample values, one per time
/// * `t` - Time at which to evaluate the derivative. Units: s
///
/// # Returns
///
/// * `Vector3<f64>` - Time derivative of the interpolating polynomial. Units: value units per s
fn lagrange_derivative(times: &[f64], values: &[Vector3<f64>], t: f64) -> Vector3<f64> {
    let mut rate = Vector3::zeros();
    for (j, (&xj, value)) in times.iter().zip(values).enumerate() {
        let mut basis_rate = 0.0;
        for (m, &xm) in times.iter().enumerate() {
            if m == j {
                continue;
            }
            let mut term = 1.0 / (xj - xm);
            for (k, &xk) in times.iter().enumerate() {
                if k != j && k != m {
                    term *= (t - xk) / (xj - xk);
                }
            }
            basis_rate += term;
        }
        rate += basis_rate * value;
    }
    rate
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
#[allow(non_snake_case)]
mod tests {
    use super::*;
    use crate::access::{ElevationConstraint, PointLocation, location_accesses};
    use crate::constants::{AngleFormat, MJD_ZERO};
    use crate::datafiles::sp3::types::{SP3, SP3Header};
    use crate::frames::{CelestialFrame, ReferenceFrame, bias_precession_nutation, polar_motion};
    use crate::math::interpolation::InterpolationConfig;
    use crate::propagators::KeplerianPropagator;
    use crate::time::TimeSystem;
    use crate::trajectories::traits::InterpolatableTrajectory;
    use crate::utils::testing::setup_global_test_eop;
    use crate::utils::{DOrbitStateProvider, DStateProvider, Identifiable};
    use approx::assert_abs_diff_eq;
    use nalgebra::{Matrix3, Vector3, Vector6};
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
            SP3Header {
                has_velocity: false,
                ..Default::default()
            },
        )
        .unwrap();
        for k in 0..N_EPOCHS {
            let epoch = t0() + k as f64 * STEP;
            sp3.set_epoch(epoch, &[]).unwrap();
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
                sp3.set_record(*id, epoch, record).unwrap();
            }
        }
        sp3.set_record(sid("G04"), sp3.epochs()[3], SP3Record::new([0.0; 3]))
            .unwrap();
        (sp3, propagators)
    }

    fn sample(epoch: Epoch, position: [f64; 3]) -> (Epoch, SP3Record) {
        (epoch, SP3Record::new(position))
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
    fn test_SP3Satellite_gaps() {
        let (sp3, _) = build_sp3();
        let t0 = t0();
        let g01 = sp3.satellite("G01".parse().unwrap()).unwrap();
        assert_eq!(g01.len(), 93);
        assert_eq!(g01.gaps(), vec![(t0 + 39.0 * 900.0, t0 + 44.0 * 900.0)]);
        assert!(sp3.satellite(sid("G02")).unwrap().gaps().len() == 1);

        let trajectory = g01.to_trajectory().unwrap();
        assert_eq!(trajectory.get_name(), Some("G01"));
        assert_eq!(trajectory.frame, ReferenceFrame::from(CelestialFrame::ITRF));
        assert_eq!(trajectory.len(), 93);
        assert_eq!(trajectory.epochs[39], t0 + 39.0 * 900.0);
        assert_eq!(trajectory.epochs[40], t0 + 44.0 * 900.0);
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
        assert_eq!(g02.epochs().len(), g02.clocks().len());
        assert_eq!(g02.epochs().len(), g02.clock_rates().len());
        assert_eq!(g02.epochs().len(), g02.records().len());
        assert_eq!(g02.clocks()[5], None);
        assert_eq!(g02.clocks()[4], Some(1.0e-5 + 4.0e-9));
        assert!(g02.clock_rates().iter().all(|r| r.is_none()));
        assert!(!g02.epochs().contains(&(t0 + 7.0 * 900.0)));
        assert_eq!(g02.start_epoch(), t0);
        assert_eq!(g02.end_epoch(), t0 + 96.0 * 900.0);
        let expected = sp3.record(sid("G02"), sp3.epochs()[8]).unwrap().unwrap();
        assert_eq!(g02.positions()[7], expected.position);
        assert_eq!(&g02.records()[7], expected);
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_velocities_from_file() {
        let with = SP3::from_file("test_assets/sp3/example-a-2.sp3").unwrap();
        let mut present = 0;
        for sat in with.satellites().unwrap() {
            let velocities = sat.velocities();
            assert_eq!(velocities.len(), sat.len());
            for (velocity, record) in velocities.iter().zip(sat.records()) {
                assert_eq!(*velocity, record.velocity);
                present += usize::from(velocity.is_some());
            }
        }
        assert!(present > 0);

        let without = SP3::from_file("test_assets/sp3/example-a-1.sp3").unwrap();
        for sat in without.satellites().unwrap() {
            assert_eq!(sat.velocities().len(), sat.len());
            assert!(sat.velocities().iter().all(Option::is_none));
        }
    }

    #[test]
    #[serial]
    fn test_SP3Satellite_isolated_sample_is_kept() {
        let (sp3, _) = build_sp3();
        let t0 = t0();
        let g03 = sp3.satellite("G03".parse().unwrap()).unwrap();
        assert_eq!(g03.len(), 12);
        let isolated: Epoch = t0 + 20.0 * STEP;
        assert_eq!(g03.gaps(), vec![(t0 + 10.0 * STEP, isolated)]);
        assert_eq!(g03.to_trajectory().unwrap().len(), 12);

        let state = g03.state_itrf(isolated).unwrap();
        assert_eq!([state[0], state[1], state[2]], g03.positions()[11]);
        assert!(g03.state_itrf(t0 + 15.0 * STEP).is_ok());
        assert!(g03.state_itrf(t0 + 5.0 * STEP).is_ok());
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_to_trajectory_single_sample() {
        let t0 = t0();
        let sat = SP3Satellite::from_samples(
            sid("G07"),
            vec![sample(t0, [1.0e7, 2.0e7, 3.0e6])],
            900.0,
            DEFAULT_INTERPOLATION_DEGREE,
        )
        .unwrap();
        assert!(sat.gaps().is_empty());
        let err = sat.to_trajectory().unwrap_err().to_string();
        assert!(err.contains("single sample"), "{err}");
        let err = sat.state_itrf(t0).unwrap_err().to_string();
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
    fn test_SP3Satellite_interpolation_degree() {
        let (mut sp3, _) = build_sp3();
        let g01 = sp3.satellite(sid("G01")).unwrap();
        assert_eq!(g01.interpolation_degree(), 9);
        assert_eq!(
            g01.to_trajectory().unwrap().get_interpolation_method(),
            InterpolationMethod::Lagrange { degree: 9 }
        );

        // A satellite takes the degree of its file and can replace it.
        sp3.interpolation_degree = 5;
        let mut g01 = sp3.satellite(sid("G01")).unwrap();
        assert_eq!(g01.interpolation_degree(), 5);
        assert!(
            sp3.satellites()
                .unwrap()
                .iter()
                .all(|s| s.interpolation_degree() == 5)
        );
        assert_eq!(
            g01.to_trajectory().unwrap().get_interpolation_method(),
            InterpolationMethod::Lagrange { degree: 5 }
        );
        let t = t0() + 10.0 * STEP + 137.0;
        let fifth = g01.state_itrf(t).unwrap();
        assert_eq!(fifth, g01.interpolate_at(t, 5).unwrap());
        g01.set_interpolation_degree(7).unwrap();
        assert_eq!(
            g01.state_itrf(t).unwrap(),
            g01.interpolate_at(t, 7).unwrap()
        );
        assert_ne!(g01.state_itrf(t).unwrap(), fifth);
        let g01 = g01.with_interpolation_degree(3).unwrap();
        assert_eq!(g01.interpolation_degree(), 3);

        let err = g01.clone().with_interpolation_degree(0).unwrap_err();
        assert!(err.to_string().contains("at least 1"), "{err}");
        sp3.interpolation_degree = 0;
        assert!(sp3.satellite(sid("G01")).is_err());
        assert!(sp3.satellites().is_err());

        let t0 = t0();
        let samples = (0..3)
            .map(|k| sample(t0 + 900.0 * k as f64, quadratic_position(900.0 * k as f64)))
            .collect();
        let sat =
            SP3Satellite::from_samples(sid("G08"), samples, 900.0, DEFAULT_INTERPOLATION_DEGREE)
                .unwrap();
        assert_eq!(
            sat.to_trajectory().unwrap().get_interpolation_method(),
            InterpolationMethod::Lagrange { degree: 2 }
        );
        let state = sat.state_itrf(t0 + 450.0).unwrap();
        for (value, expected) in state.iter().zip(quadratic_position(450.0)) {
            assert_abs_diff_eq!(*value, expected, epsilon = 1e-6);
        }
        // A degree above the number of samples fits all of them.
        let all = sat.clone().with_interpolation_degree(usize::MAX).unwrap();
        assert_eq!(all.state_itrf(t0 + 450.0).unwrap(), state);
    }

    #[test]
    #[serial]
    fn test_SP3Satellite_derived_velocity_matches_propagator() {
        let (sp3, propagators) = build_sp3();
        let g01 = sp3.satellite(sid("G01")).unwrap();
        let half = DEFAULT_INTERPOLATION_DEGREE.div_ceil(2);
        // Samples before and after the gap.
        let stretches = [(0usize, 40usize), (40, 93)];
        for (start, end) in stretches {
            for i in (start + half)..(end - half) {
                let epoch = g01.epochs()[i];
                let truth = propagators[0].state_itrf(epoch).unwrap();
                let state = g01.state_itrf(epoch).unwrap();
                for axis in 3..6 {
                    assert_abs_diff_eq!(state[axis], truth[axis], epsilon = 1e-3);
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
            .map(|&t| sample(t0 + t, quadratic_position(t)))
            .collect();
        let sat =
            SP3Satellite::from_samples(sid("G09"), samples, 300.0, DEFAULT_INTERPOLATION_DEGREE)
                .unwrap();
        for t in offsets.iter().copied().chain([150.0, 1777.0, 3599.0]) {
            let state = sat.state_itrf(t0 + t).unwrap();
            for (derived, expected) in state.iter().skip(3).zip(quadratic_velocity(t)) {
                assert_abs_diff_eq!(*derived, expected, epsilon = 1e-6);
            }
        }

        let pair = vec![
            sample(t0, [0.0, 1.0e7, 2.0e7]),
            sample(t0 + 60.0, [600.0, 1.0e7 - 120.0, 2.0e7]),
        ];
        let sat = SP3Satellite::from_samples(sid("G10"), pair, 60.0, DEFAULT_INTERPOLATION_DEGREE)
            .unwrap();
        for t in [0.0, 30.0, 60.0] {
            let state = sat.state_itrf(t0 + t).unwrap();
            assert_abs_diff_eq!(state[3], 10.0, epsilon = 1e-9);
            assert_abs_diff_eq!(state[4], -2.0, epsilon = 1e-9);
            assert_abs_diff_eq!(state[5], 0.0, epsilon = 1e-9);
        }
    }

    #[test]
    #[parallel]
    fn test_lagrange_derivative() {
        let times = [0.0, 1.0, 2.5, 4.0, 4.5, 7.0];
        let cubic = |t: f64| 2.0 - 3.0 * t + 0.5 * t * t * t;
        let cubic_rate = |t: f64| -3.0 + 1.5 * t * t;
        let values: Vec<Vector3<f64>> = times
            .iter()
            .map(|&t| Vector3::new(cubic(t), 4.0 * t, -cubic(t)))
            .collect();
        for t in times.iter().copied().chain([0.3, 3.2, 6.9]) {
            let rate = lagrange_derivative(&times, &values, t);
            assert_abs_diff_eq!(rate[0], cubic_rate(t), epsilon = 1e-10);
            assert_abs_diff_eq!(rate[1], 4.0, epsilon = 1e-10);
            assert_abs_diff_eq!(rate[2], -cubic_rate(t), epsilon = 1e-10);
        }
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_file_velocities() {
        let t0 = t0();
        let with_velocity = |k: usize| {
            let t = 900.0 * k as f64;
            (
                t0 + t,
                SP3Record::new(quadratic_position(t)).with_velocity([1.0, 2.0, 3.0 + t]),
            )
        };
        let samples: Vec<(Epoch, SP3Record)> = (0..5).map(with_velocity).collect();
        let sat = SP3Satellite::from_samples(
            sid("G11"),
            samples.clone(),
            900.0,
            DEFAULT_INTERPOLATION_DEGREE,
        )
        .unwrap();
        let state = sat.state_itrf(t0 + 1800.0).unwrap();
        assert_eq!([state[3], state[4], state[5]], [1.0, 2.0, 1803.0]);
        let trajectory = sat.to_trajectory().unwrap();
        assert_eq!(trajectory.states[2][5], 1803.0);

        let mut mixed = samples;
        mixed[3].1.velocity = None;
        let sat =
            SP3Satellite::from_samples(sid("G11"), mixed, 900.0, DEFAULT_INTERPOLATION_DEGREE)
                .unwrap();
        for epoch in sat.epochs() {
            let state = sat.state_itrf(*epoch).unwrap();
            for (derived, expected) in state.iter().skip(3).zip(quadratic_velocity(*epoch - t0)) {
                assert_abs_diff_eq!(*derived, expected, epsilon = 1e-6);
            }
        }
        assert_eq!(sat.records()[2].velocity, Some([1.0, 2.0, 1803.0]));
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_gap_threshold() {
        let t0 = t0();
        let p = [1.0e7, 2.0e7, 3.0e6];
        let samples = vec![
            sample(t0, p),
            sample(t0 + 900.0, p),
            sample(t0 + 2250.0, p),
            sample(t0 + 3150.0, p),
            sample(t0 + 4501.0, p),
            sample(t0 + 5401.0, p),
        ];
        let sat =
            SP3Satellite::from_samples(sid("G12"), samples, 900.0, DEFAULT_INTERPOLATION_DEGREE)
                .unwrap();
        assert_eq!(sat.gaps(), vec![(t0 + 3150.0, t0 + 4501.0)]);
        assert_eq!(sat.to_trajectory().unwrap().len(), 6);
    }

    #[test]
    #[parallel]
    fn test_SP3Satellite_from_samples_errors() {
        let t0 = t0();
        let p = [1.0e7, 2.0e7, 3.0e6];
        let err =
            SP3Satellite::from_samples(sid("G13"), Vec::new(), 900.0, DEFAULT_INTERPOLATION_DEGREE)
                .unwrap_err()
                .to_string();
        assert!(err.contains("G13"));
        let unordered = vec![sample(t0 + 900.0, p), sample(t0, p)];
        assert!(
            SP3Satellite::from_samples(sid("G13"), unordered, 900.0, DEFAULT_INTERPOLATION_DEGREE)
                .is_err()
        );
        let repeated = vec![sample(t0, p), sample(t0, p)];
        assert!(
            SP3Satellite::from_samples(sid("G13"), repeated, 900.0, DEFAULT_INTERPOLATION_DEGREE)
                .is_err()
        );

        let (e1, e2): (Epoch, Epoch) = (t0 + 900.0, t0 + 1300.0);
        let close = vec![sample(t0, p), sample(e1, p), sample(e2, p)];
        let err =
            SP3Satellite::from_samples(sid("G13"), close, 900.0, DEFAULT_INTERPOLATION_DEGREE)
                .unwrap_err()
                .to_string();
        assert!(err.contains("G13"), "{err}");
        assert!(err.contains(&e1.to_string()), "{err}");
        assert!(err.contains(&e2.to_string()), "{err}");
        assert!(err.contains("half the 900 s epoch interval"), "{err}");

        let pair = vec![sample(t0, p), sample(t0 + 900.0, p)];
        assert!(SP3Satellite::from_samples(sid("G13"), pair, 900.0, 0).is_err());
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
            SP3Header {
                has_velocity: false,
                ..Default::default()
            },
        )
        .unwrap();
        for k in 0..N_EPOCHS {
            let epoch = t0() + k as f64 * STEP;
            sp3.set_epoch(epoch, &[]).unwrap();
            for (id, propagator) in ids.iter().zip(&propagators) {
                let x = propagator.state_itrf(epoch).unwrap();
                sp3.set_record(*id, epoch, SP3Record::new([x[0], x[1], x[2]]))
                    .unwrap();
            }
        }
        (sp3, propagators)
    }

    #[test]
    #[serial]
    fn test_SP3Satellite_DOrbitStateProvider_returns_file_positions_at_samples() {
        let (sp3, _) = build_sp3();
        let g01 = sp3.satellite(sid("G01")).unwrap();
        let positions = g01.positions();
        let epochs = g01.epochs();
        // First sample, both samples bordering the gap, and last sample.
        for i in [0, 39, 40, g01.len() - 1] {
            let state = g01.state_itrf(epochs[i]).unwrap();
            for axis in 0..3 {
                assert_eq!(state[axis], positions[i][axis]);
            }
            assert!(state.iter().all(|v| v.is_finite()));
        }
        assert_eq!(g01.state_dim(), 6);
        assert_eq!(g01.state(epochs[0]).unwrap().len(), 6);
    }

    #[test]
    #[serial]
    fn test_SP3Satellite_DOrbitStateProvider_errors_outside_samples() {
        let (sp3, _) = build_sp3();
        let t0 = t0();
        let g01 = sp3.satellite(sid("G01")).unwrap();

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
        let outside = t0 - 1.0;
        assert!(g01.state(outside).is_err());
        assert!(g01.state_eci(outside).is_err());
        assert!(g01.state_gcrf(outside).is_err());
        assert!(g01.state_ecef(outside).is_err());
        assert!(g01.state_eme2000(outside).is_err());
        assert!(g01.state_bci(outside).is_err());
        assert!(g01.state_bcbf(outside).is_err());
        assert!(g01.state_in_frame(CelestialFrame::GCRF, outside).is_err());
        assert!(g01.state_koe_osc(outside, AngleFormat::Degrees).is_err());
        assert!(g01.state_koe_mean(outside, AngleFormat::Degrees).is_err());
        assert!(g01.states_eci(&[t0, outside]).is_err());
    }

    #[test]
    #[serial]
    fn test_SP3Satellite_DOrbitStateProvider_interpolates_inside_gap() {
        let (sp3, propagators) = build_sp3();
        let g01 = sp3.satellite(sid("G01")).unwrap();
        // Epochs 40..=43 are missing, so the fit spans a 75 min hole and is
        // far less accurate than between adjacent samples.
        let inside = t0() + 41.5 * STEP;
        let error = |epoch: Epoch| {
            let state = g01.state_itrf(epoch).unwrap();
            let truth = propagators[0].state_itrf(epoch).unwrap();
            (0..3)
                .map(|i| (state[i] - truth[i]).powi(2))
                .sum::<f64>()
                .sqrt()
        };
        let clean = error(t0() + 20.5 * STEP);
        let gap = error(inside);
        assert!(clean < 1e-2, "{clean}");
        assert!(gap.is_finite() && gap > 100.0 * clean, "{gap} vs {clean}");
    }

    #[test]
    #[serial]
    fn test_SP3Satellite_DOrbitStateProvider_matches_trajectory() {
        let (sp3, _) = build_sp3();
        let g01 = sp3.satellite(sid("G01")).unwrap();
        let trajectory = g01.to_trajectory().unwrap();
        // Between samples the satellite differentiates the position fit, while
        // a trajectory interpolates the sample velocities.
        let tolerance = |axis: usize| if axis < 3 { 1e-9 } else { 1e-6 };
        let t = t0() + 10.0 * STEP + 137.0;
        let from_satellite = g01.state_eci(t).unwrap();
        let from_segment = trajectory.state_eci(t).unwrap();
        for axis in 0..6 {
            assert_abs_diff_eq!(
                from_satellite[axis],
                from_segment[axis],
                epsilon = tolerance(axis)
            );
        }

        // An epoch after the gap.
        let t = t0() + 60.0 * STEP + 11.0;
        let from_satellite = g01.state_itrf(t).unwrap();
        let from_segment = trajectory.state_itrf(t).unwrap();
        for axis in 0..6 {
            assert_abs_diff_eq!(
                from_satellite[axis],
                from_segment[axis],
                epsilon = tolerance(axis)
            );
        }

        let epochs = [t0() + 100.0, t0() + 60.0 * STEP + 11.0];
        let batch = g01.states_gcrf(&epochs).unwrap();
        for (state, epoch) in batch.iter().zip(epochs) {
            assert_eq!(*state, g01.state_gcrf(epoch).unwrap());
        }
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
        assert_eq!(renamed.to_trajectory().unwrap().get_name(), Some("spare"));

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

    /// ITRF position of `propagator` at `epoch`, computed as
    /// `state_gcrf_to_itrf` does but with the Earth rotation angle evaluated
    /// from a two-part UT1 Julian date.
    ///
    /// `earth_rotation` evaluates the angle from a single `f64` MJD, which
    /// resolves the instant only in steps of about 0.6 µs. At GPS radius that
    /// moves `state_itrf` positions by up to 6.0e-4 m, differently at each
    /// epoch, so they are not smooth in time below that level. This position
    /// is smooth, and differs from `state_itrf` by at most 6.0e-4 m over the
    /// test orbits.
    fn smooth_itrf_position(propagator: &KeplerianPropagator, epoch: Epoch) -> Vector3<f64> {
        let (year, month, day, hour, minute, second, nanosecond) =
            epoch.to_datetime_as_time_system(TimeSystem::UT1);
        let mjd_day = Epoch::from_date(year, month, day, TimeSystem::UT1)
            .mjd_as_time_system(TimeSystem::UT1)
            .round();
        let day_fraction =
            (3600.0 * hour as f64 + 60.0 * minute as f64 + second + 1e-9 * nanosecond) / 86400.0;
        // SAFETY: iauEra00 is a pure function of its two scalar arguments.
        let era = unsafe { rsofa::iauEra00(MJD_ZERO + mjd_day, day_fraction) };
        let (c, s) = (era.cos(), era.sin());
        let rotation = Matrix3::new(c, s, 0.0, -s, c, 0.0, 0.0, 0.0, 1.0);
        let gcrf = propagator.state_gcrf(epoch).unwrap();
        polar_motion(epoch)
            * rotation
            * bias_precession_nutation(epoch)
            * Vector3::new(gcrf[0], gcrf[1], gcrf[2])
    }

    /// ITRF velocity of `propagator` at `epoch`: the fourth-order central
    /// difference, with step `h` (s), of [`smooth_itrf_position`].
    ///
    /// The truncation error scales as `h^4` and the roundoff as `1/h`. Over
    /// the test orbits the steps 10 s and 20 s agree to 1.6e-7 m/s, while
    /// 20 s and 40 s differ by 2.1e-6 m/s, so away from the daily Earth
    /// orientation nodes the reference is accurate to about 2e-7 m/s. The
    /// Earth orientation parameters are interpolated linearly between daily
    /// nodes, so at each node the UT1 rate changes and the ITRF velocity jumps
    /// by up to 8e-6 m/s over the test orbits. Step doubling cannot see that
    /// jump: within `2 h` of a node the reference falls between the two
    /// one-sided derivatives, up to 4e-6 m/s from either. The samples at
    /// 00:00 GPS lie 18 s from 00:00 UTC and are the ends of the samples, so the interior
    /// error figures are unaffected. This is the time derivative of the ITRF
    /// position. The `state_itrf` velocity is not: `state_gcrf_to_itrf` uses
    /// the constant `OMEGA_EARTH` and omits the precession-nutation and polar
    /// motion rates, and differs from this reference by up to 8.8e-5 m/s.
    fn reference_itrf_velocity(
        propagator: &KeplerianPropagator,
        epoch: Epoch,
        h: f64,
    ) -> Vector3<f64> {
        let p = |dt: f64| smooth_itrf_position(propagator, epoch + dt);
        (p(-2.0 * h) - 8.0 * p(-h) + 8.0 * p(h) - p(2.0 * h)) / (12.0 * h)
    }

    /// Maximum interpolation and derived-velocity errors of one Lagrange
    /// degree, split into the interior of the samples and their edges.
    #[derive(Debug, Default)]
    struct AccuracyErrors {
        itrf_position_interior: f64,
        itrf_position_edge: f64,
        itrf_velocity_interior: f64,
        itrf_velocity_edge: f64,
        gcrf_position_interior: f64,
        gcrf_position_edge: f64,
        gcrf_velocity_interior: f64,
        gcrf_velocity_edge: f64,
    }

    fn update_max(interior: bool, error: f64, interior_max: &mut f64, edge_max: &mut f64) {
        let target = if interior { interior_max } else { edge_max };
        *target = target.max(error);
    }

    // Measured over the three GPS-like orbits of `build_clean_sp3` (a = 26560
    // km, e = 0.01, i = 55 deg), 97 samples at 900 s, positions from
    // `state_itrf`. Positions are compared at mid-sample epochs and derived
    // velocities at the samples. The interior holds the samples and
    // mid-samples at least `degree / 2` samples from both run edges; the edge
    // holds the rest. ITRF errors are against `smooth_itrf_position` and
    // `reference_itrf_velocity`; GCRF errors compare `to_gcrf()` trajectories
    // against the exact `state_gcrf` of the propagators. Units: m and m/s.
    //
    // degree | ITRF position     | ITRF velocity     | GCRF position     | GCRF velocity
    //        | interior  edge    | interior  edge    | interior  edge    | interior  edge
    //      5 | 1.8e0     7.6e0   | 6.9e-3    6.9e-2  | 8.9e-1    3.7e0   | 6.9e-3    6.9e-2
    //      7 | 1.8e-2    2.2e-1  | 6.8e-5    2.4e-3  | 6.3e-3    7.7e-2  | 1.3e-4    2.4e-3
    //      9 | 8.1e-4    1.0e-2  | 1.5e-6    1.2e-4  | 8.2e-5    3.2e-3  | 8.8e-5    1.2e-4
    //     11 | 6.8e-4    4.8e-3  | 8.5e-7    5.2e-5  | 1.8e-6    2.2e-4  | 8.7e-5    8.9e-5
    //
    // Degree 9 is the smallest degree with interior errors within 1e-3 m and
    // 1e-4 m/s; degree 7 misses the position bound by a factor of 18.
    //
    // ITRF position errors level off near 7e-4 m because the samples carry
    // the up to 6.0e-4 m Earth-rotation-angle noise of `state_itrf` (see
    // `smooth_itrf_position`), which dominates from degree 9 up. `to_gcrf()` rotates each sample
    // back with the same angle, which cancels that noise, so GCRF positions
    // keep improving with degree. Interpolating in GCRF is 10 times more
    // accurate in the interior at degree 9 (8.2e-5 m against 8.1e-4 m). GCRF
    // velocities carry the up to 8.8e-5 m/s error of the ITRF-to-GCRF velocity
    // transform (see `reference_itrf_velocity`), which dominates their
    // interior error from degree 9 up.
    #[test]
    #[serial]
    fn test_SP3Satellite_interpolation_accuracy_keplerian() {
        let (sp3, propagators) = build_clean_sp3();
        let degrees = [5, 7, 9, 11];
        assert!(degrees.contains(&DEFAULT_INTERPOLATION_DEGREE));
        let mut errors: Vec<AccuracyErrors> =
            degrees.iter().map(|_| AccuracyErrors::default()).collect();
        let mut reference_step_difference = 0.0_f64;

        for (id, propagator) in sp3.satellite_ids().iter().zip(&propagators) {
            let sat = sp3.satellite(*id).unwrap();
            let epochs = sat.epochs();
            let positions = sat.positions();
            let n = epochs.len();
            let mid_epochs: Vec<Epoch> = epochs[..n - 1].iter().map(|&t| t + 0.5 * STEP).collect();

            let reference_velocities: Vec<Vector3<f64>> = epochs
                .iter()
                .map(|&t| {
                    let v = reference_itrf_velocity(propagator, t, 10.0);
                    let check = reference_itrf_velocity(propagator, t, 20.0);
                    reference_step_difference = reference_step_difference.max((v - check).norm());
                    v
                })
                .collect();
            let gcrf_states: Vec<Vector6<f64>> = epochs
                .iter()
                .map(|&t| propagator.state_gcrf(t).unwrap())
                .collect();
            let mid_itrf_positions: Vec<Vector3<f64>> = mid_epochs
                .iter()
                .map(|&t| smooth_itrf_position(propagator, t))
                .collect();
            let mid_gcrf_states: Vec<Vector6<f64>> = mid_epochs
                .iter()
                .map(|&t| propagator.state_gcrf(t).unwrap())
                .collect();

            for (&degree, e) in degrees.iter().zip(errors.iter_mut()) {
                let velocities: Vec<[f64; 3]> = epochs
                    .iter()
                    .map(|&t| {
                        let state = sat.interpolate_at(t, degree).unwrap();
                        [state[3], state[4], state[5]]
                    })
                    .collect();
                let mut itrf = DOrbitTrajectory::new(
                    6,
                    CelestialFrame::ITRF,
                    OrbitRepresentation::Cartesian,
                    None,
                )
                .unwrap()
                .with_interpolation_method(InterpolationMethod::Lagrange { degree });
                for ((&epoch, p), v) in epochs.iter().zip(&positions).zip(&velocities) {
                    itrf.add(
                        epoch,
                        DVector::from_column_slice(&[p[0], p[1], p[2], v[0], v[1], v[2]]),
                    )
                    .unwrap();
                }
                let gcrf = itrf.to_gcrf().unwrap();

                let half = degree / 2;
                for i in 0..n {
                    let interior = i >= half && i + half < n;
                    let velocity_error =
                        (Vector3::from(velocities[i]) - reference_velocities[i]).norm();
                    update_max(
                        interior,
                        velocity_error,
                        &mut e.itrf_velocity_interior,
                        &mut e.itrf_velocity_edge,
                    );
                    let state = gcrf.interpolate(&epochs[i]).unwrap();
                    let gcrf_velocity_error = (state.rows(3, 3) - gcrf_states[i].rows(3, 3)).norm();
                    update_max(
                        interior,
                        gcrf_velocity_error,
                        &mut e.gcrf_velocity_interior,
                        &mut e.gcrf_velocity_edge,
                    );
                }
                for (i, &t) in mid_epochs.iter().enumerate() {
                    let interior = i >= half && i + 1 + half < n;
                    let state = itrf.interpolate(&t).unwrap();
                    let position_error = (state.rows(0, 3) - mid_itrf_positions[i]).norm();
                    update_max(
                        interior,
                        position_error,
                        &mut e.itrf_position_interior,
                        &mut e.itrf_position_edge,
                    );
                    let state = gcrf.interpolate(&t).unwrap();
                    let gcrf_position_error =
                        (state.rows(0, 3) - mid_gcrf_states[i].rows(0, 3)).norm();
                    update_max(
                        interior,
                        gcrf_position_error,
                        &mut e.gcrf_position_interior,
                        &mut e.gcrf_position_edge,
                    );
                }
            }
        }

        assert!(
            reference_step_difference < 1e-6,
            "{reference_step_difference}"
        );
        let meets_bounds = |e: &AccuracyErrors| {
            e.itrf_position_interior <= 1e-3 && e.itrf_velocity_interior <= 1e-4
        };
        for (&degree, e) in degrees.iter().zip(&errors) {
            if degree < DEFAULT_INTERPOLATION_DEGREE {
                assert!(
                    !meets_bounds(e),
                    "degree {degree} also meets the bounds: {e:?}"
                );
            }
            if degree == DEFAULT_INTERPOLATION_DEGREE {
                assert!(meets_bounds(e), "{e:?}");
                assert!(e.gcrf_position_interior <= 1e-3, "{e:?}");
                assert!(e.gcrf_velocity_interior <= 1e-4, "{e:?}");
                assert!(
                    e.itrf_position_edge <= 2e-2 && e.gcrf_position_edge <= 2e-2,
                    "{e:?}"
                );
                assert!(
                    e.itrf_velocity_edge <= 5e-4 && e.gcrf_velocity_edge <= 5e-4,
                    "{e:?}"
                );
            }
        }
    }

    /// Interior and edge errors of `sat` at its default interpolation
    /// method: positions at mid-sample epochs against `smooth_itrf_position`
    /// and velocities at the samples against `reference_itrf_velocity`.
    ///
    /// # Returns
    ///
    /// * `[f64; 4]` - Position interior, position edge (m), velocity interior,
    ///   velocity edge (m/s)
    fn itrf_errors(sat: &SP3Satellite, propagator: &KeplerianPropagator) -> [f64; 4] {
        let epochs = sat.epochs();
        let n = epochs.len();
        let half = DEFAULT_INTERPOLATION_DEGREE / 2;
        let mut errors = [0.0_f64; 4];
        let [
            position_interior,
            position_edge,
            velocity_interior,
            velocity_edge,
        ] = &mut errors;
        for (i, &epoch) in epochs.iter().enumerate() {
            let reference = reference_itrf_velocity(propagator, epoch, 10.0);
            let velocity = sat
                .state_itrf(epoch)
                .unwrap()
                .fixed_rows::<3>(3)
                .into_owned();
            let error = (velocity - reference).norm();
            update_max(
                i >= half && i + half < n,
                error,
                velocity_interior,
                velocity_edge,
            );
            if i + 1 < n {
                let t = epoch + 0.5 * STEP;
                let state = sat.state(t).unwrap();
                let error = (state.rows(0, 3) - smooth_itrf_position(propagator, t)).norm();
                update_max(
                    i >= half && i + 1 + half < n,
                    error,
                    position_interior,
                    position_edge,
                );
            }
        }
        errors
    }

    // The orbits of `test_SP3Satellite_interpolation_accuracy_keplerian` at
    // SP3 file precision: written with `to_string` and parsed back, which
    // rounds each position component to the 1 mm grid of the format. Maximum
    // errors at degree 9 (m and m/s):
    //
    // samples      | position interior  edge   | velocity interior  edge
    // exact        | 8.1e-4             1.0e-2 | 1.5e-6             1.2e-4
    // file (1 mm)  | 1.4e-3             1.2e-2 | 2.2e-6             1.4e-4
    //
    // Interpolation and the velocity fit are linear in the samples, so the
    // rounding adds at most its own interpolant to the exact-sample error.
    // Each rounded component is within 0.5 mm, so each rounding vector is
    // within sqrt(3) * 0.5 mm = 8.7e-4 m. Every interior mid-sample sits at
    // the center of a 10-sample equispaced window, where the Lagrange basis
    // has Lebesgue constant sum_j |l_j| = 1.563, so the rounding adds at most
    // 1.563 * 8.7e-4 = 1.35e-3 m to the position error. Every interior sample
    // is node 4 or 5 of its 10-sample window, where the derivative weights
    // sum to sum_j |l_j'| = 2.48 / 900 s = 2.76e-3 1/s, so the rounding adds
    // at most 2.4e-6 m/s to the velocity error. The bounds asserted, per
    // satellite, are the exact-sample error plus these terms: 2.1e-3 m to
    // 2.2e-3 m and 3.8e-6 m/s to 3.9e-6 m/s.
    #[test]
    #[serial]
    fn test_SP3Satellite_interpolation_accuracy_file_precision() {
        let (sp3, propagators) = build_clean_sp3();
        let parsed = SP3::from_str(&sp3.to_string().unwrap()).unwrap();

        let m = DEFAULT_INTERPOLATION_DEGREE + 1;
        let times: Vec<f64> = (0..m).map(|k| STEP * k as f64).collect();
        let basis = |j: usize, t: f64| {
            (0..m)
                .filter(|&k| k != j)
                .map(|k| (t - times[k]) / (times[j] - times[k]))
                .product::<f64>()
        };
        let center = 0.5 * (times[m / 2 - 1] + times[m / 2]);
        let lebesgue: f64 = (0..m).map(|j| basis(j, center).abs()).sum();
        let derivative_weights = |node: usize| -> f64 {
            (0..m)
                .map(|j| {
                    let mut unit = vec![Vector3::zeros(); m];
                    unit[j] = Vector3::x();
                    lagrange_derivative(&times, &unit, times[node])[0].abs()
                })
                .sum()
        };
        let derivative_lebesgue = derivative_weights(m / 2 - 1).max(derivative_weights(m / 2));
        let rounding = 3.0_f64.sqrt() * 0.5e-3;

        for (id, propagator) in sp3.satellite_ids().iter().zip(&propagators) {
            let exact = itrf_errors(&sp3.satellite(*id).unwrap(), propagator);
            let rounded_sat = parsed.satellite(*id).unwrap();
            let rounded = itrf_errors(&rounded_sat, propagator);
            for (p, q) in rounded_sat
                .positions()
                .iter()
                .zip(sp3.satellite(*id).unwrap().positions())
            {
                for axis in 0..3 {
                    assert!((p[axis] - q[axis]).abs() <= 0.5e-3 + 1e-9);
                }
            }
            let position_bound = exact[0] + lebesgue * rounding;
            let velocity_bound = exact[2] + derivative_lebesgue * rounding;
            assert!(
                rounded[0] <= position_bound,
                "{id}: {rounded:?} {position_bound}"
            );
            assert!(
                rounded[2] <= velocity_bound,
                "{id}: {rounded:?} {velocity_bound}"
            );
        }
    }

    // CODE final orbits at 300 s for G02, G05, and E11, subsampled to 900 s
    // and interpolated at the withheld epochs. The file positions are rounded
    // to 1 mm, which sets a floor near 1e-3 m. Maximum position errors (m):
    //
    // degree | G02 interior  edge   | G05 interior  edge   | E11 interior  edge
    //      5 | 1.5e0         8.0e0  | 1.4e0         7.9e0  | 9.1e-1        5.0e0
    //      7 | 1.2e-2        2.1e-1 | 1.2e-2        2.0e-1 | 6.9e-3        1.0e-1
    //      9 | 1.4e-3        4.7e-3 | 1.1e-3        6.0e-3 | 1.1e-3        4.0e-3
    //     11 | 1.3e-3        2.0e-2 | 1.1e-3        5.0e-3 | 1.2e-3        6.2e-3
    #[test]
    #[parallel]
    fn test_SP3Satellite_interpolation_accuracy_cod() {
        let file = SP3::from_file("test_assets/sp3/COD0OPSFIN_20243150000_01D_05M_ORB_subset.SP3")
            .unwrap();
        let ids = file.satellite_ids().to_vec();
        assert_eq!(ids, vec![sid("G02"), sid("G05"), sid("E11")]);
        assert_eq!(file.num_epochs(), 72);

        let mut subsampled = SP3::new(
            ids.clone(),
            900.0,
            SP3Header {
                has_velocity: false,
                ..Default::default()
            },
        )
        .unwrap();
        for k in (0..file.num_epochs()).step_by(3) {
            subsampled.set_epoch(file.epochs()[k], &[]).unwrap();
            for id in &ids {
                let record = file.record(*id, file.epochs()[k]).unwrap().unwrap().clone();
                subsampled
                    .set_record(*id, file.epochs()[k], record)
                    .unwrap();
            }
        }

        let n = subsampled.num_epochs();
        let half = DEFAULT_INTERPOLATION_DEGREE / 2;
        let mut max_interior_error = 0.0_f64;
        for id in &ids {
            let sat = subsampled.satellite(*id).unwrap();
            for k in (0..file.num_epochs()).filter(|k| k % 3 != 0) {
                let before = k / 3;
                if before < half || before + 1 + half >= n {
                    continue;
                }
                let state = sat.state(file.epochs()[k]).unwrap();
                let truth = file
                    .record(*id, file.epochs()[k])
                    .unwrap()
                    .unwrap()
                    .position;
                let error = (0..3)
                    .map(|axis| (state[axis] - truth[axis]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                max_interior_error = max_interior_error.max(error);
            }
        }
        assert!(max_interior_error <= 1e-2, "{max_interior_error}");
    }
}
