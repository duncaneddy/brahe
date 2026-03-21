// CCSDS Python bindings for OEM, OMM, and OPM message types.
//
// Design: Sub-objects (segments, states, maneuvers) use a dual-mode enum.
// In Proxy mode they hold a Py<PyAny> reference to the parent and an index,
// delegating all get/set operations through internal methods on the parent.
// In Owned mode they hold a standalone copy of the Rust data.
// This ensures mutations on proxy sub-objects reflect back to the owning message,
// while standalone objects can be constructed independently and appended.

use crate::ccsds::cdm::{
    CDM as RustCDM, CDMObject, CDMObjectMetadata, CDMRTNCovariance, CDMStateVector,
};
use crate::ccsds::common::{
    CCSDSFormat, CCSDSRefFrame, CCSDSTimeSystem, ODMHeader,
};
use crate::ccsds::interop::ccsds_ref_frame_to_orbit_frame;
use crate::ccsds::oem::{OEM as RustOEM, OEMMetadata, OEMSegment, OEMStateVector};
use crate::ccsds::omm::OMM as RustOMM;
use crate::ccsds::opm::{OPM as RustOPM, OPMManeuver};
use crate::trajectories::DOrbitTrajectory;
use crate::trajectories::traits::OrbitFrame;

/// Push all states from a trajectory into an OEM segment, converting to the
/// segment's declared reference frame using the trajectory's frame-aware methods.
fn push_trajectory_states(seg: &mut OEMSegment, traj: &DOrbitTrajectory) -> Result<(), crate::utils::BraheError> {
    let orbit_frame = ccsds_ref_frame_to_orbit_frame(&seg.metadata.ref_frame)?;
    for epoch in traj.epochs.iter() {
        let state = match orbit_frame {
            OrbitFrame::EME2000 => traj.state_eme2000(*epoch)?,
            OrbitFrame::GCRF => traj.state_gcrf(*epoch)?,
            OrbitFrame::ECI => traj.state_eci(*epoch)?,
            OrbitFrame::ECEF | OrbitFrame::ITRF => traj.state_itrf(*epoch)?,
        };
        seg.states.push(OEMStateVector {
            epoch: *epoch,
            position: [state[0], state[1], state[2]],
            velocity: [state[3], state[4], state[5]],
            acceleration: None,
        });
    }
    Ok(())
}

// ─────────────────────────────────────────────
// PyOEMStateVector — typed proxy/owned for a state vector
// ─────────────────────────────────────────────

enum StateVectorMode {
    Proxy { parent: Py<PyAny>, seg_idx: usize, sv_idx: usize },
    Owned { data: OEMStateVector },
}

/// A single OEM state vector with typed property access.
///
/// State vectors can be created standalone or accessed via segment states.
/// Proxy instances (from an OEM) propagate mutations back to the parent.
///
/// Args:
///     epoch (Epoch): Epoch of the state vector
///     position (list[float]): Position [x, y, z] in meters
///     velocity (list[float]): Velocity [vx, vy, vz] in m/s
///     acceleration (list[float] | None): Optional acceleration [ax, ay, az] in m/s²
///
/// Example:
///     ```python
///     from brahe import Epoch
///     from brahe.ccsds import OEMStateVector
///     sv = OEMStateVector(
///         epoch=Epoch.from_datetime(2024, 1, 1, 0, 0, 0),
///         position=[7000e3, 0.0, 0.0],
///         velocity=[0.0, 7500.0, 0.0],
///     )
///     print(sv.position)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OEMStateVector")]
pub struct PyOEMStateVector {
    mode: StateVectorMode,
}

impl PyOEMStateVector {
    fn to_rust_data(&self, py: Python) -> PyResult<OEMStateVector> {
        match &self.mode {
            StateVectorMode::Owned { data } => Ok(data.clone()),
            StateVectorMode::Proxy { parent, seg_idx, sv_idx } => {
                let epoch_obj = parent.bind(py).call_method1("_sv_get_epoch", (*seg_idx, *sv_idx))?;
                let epoch: PyEpoch = extract_epoch(&epoch_obj)?;
                let pos: Vec<f64> = parent.bind(py).call_method1("_sv_get_position", (*seg_idx, *sv_idx))?.extract()?;
                let vel: Vec<f64> = parent.bind(py).call_method1("_sv_get_velocity", (*seg_idx, *sv_idx))?.extract()?;
                let acc: Option<Vec<f64>> = parent.bind(py).call_method1("_sv_get_acceleration", (*seg_idx, *sv_idx))?.extract()?;
                Ok(OEMStateVector {
                    epoch: epoch.obj,
                    position: vec_to_array3(pos, "position")?,
                    velocity: vec_to_array3(vel, "velocity")?,
                    acceleration: match acc { Some(v) => Some(vec_to_array3(v, "acceleration")?), None => None },
                })
            }
        }
    }
}

#[pymethods]
impl PyOEMStateVector {
    #[new]
    #[pyo3(signature = (epoch, position, velocity, acceleration=None))]
    fn new(
        epoch: PyEpoch,
        position: &pyo3::Bound<'_, pyo3::types::PyAny>,
        velocity: &pyo3::Bound<'_, pyo3::types::PyAny>,
        acceleration: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<Self> {
        let pos = pyany_to_array3(position, "position")?;
        let vel = pyany_to_array3(velocity, "velocity")?;
        let acc = pyany_to_optional_array3(acceleration, "acceleration")?;
        Ok(Self {
            mode: StateVectorMode::Owned {
                data: OEMStateVector { epoch: epoch.obj, position: pos, velocity: vel, acceleration: acc },
            },
        })
    }

    /// Epoch of this state vector.
    ///
    /// Returns:
    ///     Epoch: Epoch object
    #[getter]
    fn epoch(&self, py: Python) -> PyResult<PyEpoch> {
        match &self.mode {
            StateVectorMode::Owned { data } => Ok(PyEpoch { obj: data.epoch }),
            StateVectorMode::Proxy { parent, seg_idx, sv_idx } => {
                let obj = parent.bind(py).call_method1("_sv_get_epoch", (*seg_idx, *sv_idx))?;
                extract_epoch(&obj)
            }
        }
    }

    /// Set the epoch of this state vector.
    ///
    /// Args:
    ///     val (Epoch): New epoch
    #[setter]
    fn set_epoch(&mut self, py: Python, val: PyEpoch) -> PyResult<()> {
        match &mut self.mode {
            StateVectorMode::Owned { data } => { data.epoch = val.obj; Ok(()) }
            StateVectorMode::Proxy { parent, seg_idx, sv_idx } => {
                parent.bind(py).call_method1("_sv_set_epoch", (*seg_idx, *sv_idx, val))?;
                Ok(())
            }
        }
    }

    /// Position vector [x, y, z] in meters.
    ///
    /// Returns:
    ///     numpy.ndarray: Position [x, y, z] in meters
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
        match &self.mode {
            StateVectorMode::Owned { data } => Ok(data.position.to_vec().into_pyarray(py)),
            StateVectorMode::Proxy { parent, seg_idx, sv_idx } => {
                let pos: Vec<f64> = parent.bind(py).call_method1("_sv_get_position", (*seg_idx, *sv_idx))?.extract()?;
                Ok(pos.into_pyarray(py))
            }
        }
    }

    /// Set the position vector.
    ///
    /// Args:
    ///     val (list[float]): Position [x, y, z] in meters
    #[setter]
    fn set_position(&mut self, py: Python, val: &pyo3::Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
        let arr = pyany_to_array3(val, "position")?;
        match &mut self.mode {
            StateVectorMode::Owned { data } => { data.position = arr; Ok(()) }
            StateVectorMode::Proxy { parent, seg_idx, sv_idx } => {
                parent.bind(py).call_method1("_sv_set_position", (*seg_idx, *sv_idx, arr.to_vec()))?;
                Ok(())
            }
        }
    }

    /// Velocity vector [vx, vy, vz] in m/s.
    ///
    /// Returns:
    ///     numpy.ndarray: Velocity [vx, vy, vz] in m/s
    #[getter]
    fn velocity<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
        match &self.mode {
            StateVectorMode::Owned { data } => Ok(data.velocity.to_vec().into_pyarray(py)),
            StateVectorMode::Proxy { parent, seg_idx, sv_idx } => {
                let vel: Vec<f64> = parent.bind(py).call_method1("_sv_get_velocity", (*seg_idx, *sv_idx))?.extract()?;
                Ok(vel.into_pyarray(py))
            }
        }
    }

    /// Set the velocity vector.
    ///
    /// Args:
    ///     val (list[float]): Velocity [vx, vy, vz] in m/s
    #[setter]
    fn set_velocity(&mut self, py: Python, val: &pyo3::Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
        let arr = pyany_to_array3(val, "velocity")?;
        match &mut self.mode {
            StateVectorMode::Owned { data } => { data.velocity = arr; Ok(()) }
            StateVectorMode::Proxy { parent, seg_idx, sv_idx } => {
                parent.bind(py).call_method1("_sv_set_velocity", (*seg_idx, *sv_idx, arr.to_vec()))?;
                Ok(())
            }
        }
    }

    /// Optional acceleration vector [ax, ay, az] in m/s².
    ///
    /// Returns:
    ///     numpy.ndarray: Acceleration [ax, ay, az] in m/s², or None
    #[getter]
    fn acceleration<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyArray<f64, Ix1>>>> {
        match &self.mode {
            StateVectorMode::Owned { data } => Ok(data.acceleration.map(|a| a.to_vec().into_pyarray(py))),
            StateVectorMode::Proxy { parent, seg_idx, sv_idx } => {
                let acc: Option<Vec<f64>> = parent.bind(py).call_method1("_sv_get_acceleration", (*seg_idx, *sv_idx))?.extract()?;
                Ok(acc.map(|a| a.into_pyarray(py)))
            }
        }
    }

    /// Set the acceleration vector.
    ///
    /// Args:
    ///     val (list[float] | None): Acceleration [ax, ay, az] in m/s², or None
    #[setter]
    fn set_acceleration(&mut self, py: Python, val: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>) -> PyResult<()> {
        let arr = pyany_to_optional_array3(val, "acceleration")?;
        match &mut self.mode {
            StateVectorMode::Owned { data } => {
                data.acceleration = arr;
                Ok(())
            }
            StateVectorMode::Proxy { parent, seg_idx, sv_idx } => {
                let py_val: Option<Vec<f64>> = arr.map(|a| a.to_vec());
                parent.bind(py).call_method1("_sv_set_acceleration", (*seg_idx, *sv_idx, py_val))?;
                Ok(())
            }
        }
    }

    /// Combined state vector [x, y, z, vx, vy, vz] as a numpy array.
    ///
    /// Returns:
    ///     numpy.ndarray: 6-element state vector (position in meters, velocity in m/s)
    ///
    /// Example:
    ///     ```python
    ///     sv = oem.segments[0].states[0]
    ///     state = sv.state  # numpy array [x, y, z, vx, vy, vz]
    ///     ```
    #[getter]
    fn state<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyArray<f64, Ix1>>> {
        match &self.mode {
            StateVectorMode::Owned { data } => {
                let p = &data.position;
                let v = &data.velocity;
                Ok(vec![p[0], p[1], p[2], v[0], v[1], v[2]].into_pyarray(py))
            }
            StateVectorMode::Proxy { parent, seg_idx, sv_idx } => {
                let pos: Vec<f64> = parent.bind(py).call_method1("_sv_get_position", (*seg_idx, *sv_idx))?.extract()?;
                let vel: Vec<f64> = parent.bind(py).call_method1("_sv_get_velocity", (*seg_idx, *sv_idx))?.extract()?;
                Ok(vec![pos[0], pos[1], pos[2], vel[0], vel[1], vel[2]].into_pyarray(py))
            }
        }
    }

    /// Set combined state vector [x, y, z, vx, vy, vz].
    ///
    /// Args:
    ///     val (list[float] | numpy.ndarray): 6-element state vector
    ///
    /// Example:
    ///     ```python
    ///     import numpy as np
    ///     sv.state = np.array([7000e3, 0, 0, 0, 7500, 0])
    ///     ```
    #[setter]
    fn set_state(&mut self, py: Python, val: &pyo3::Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
        let sv = pyany_to_svector::<6>(val).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("state: {}", e))
        })?;
        let pos = [sv[0], sv[1], sv[2]];
        let vel = [sv[3], sv[4], sv[5]];
        match &mut self.mode {
            StateVectorMode::Owned { data } => {
                data.position = pos;
                data.velocity = vel;
                Ok(())
            }
            StateVectorMode::Proxy { parent, seg_idx, sv_idx } => {
                parent.bind(py).call_method1("_sv_set_position", (*seg_idx, *sv_idx, pos.to_vec()))?;
                parent.bind(py).call_method1("_sv_set_velocity", (*seg_idx, *sv_idx, vel.to_vec()))?;
                Ok(())
            }
        }
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let epoch: PyEpoch = self.epoch(py)?;
        match &self.mode {
            StateVectorMode::Owned { .. } => {
                Ok(format!("OEMStateVector(epoch={})", epoch.obj))
            }
            StateVectorMode::Proxy { seg_idx, sv_idx, .. } => {
                Ok(format!(
                    "OEMStateVector(epoch={}, seg={}, idx={})",
                    epoch.obj, seg_idx, sv_idx
                ))
            }
        }
    }
}

// ─────────────────────────────────────────────
// OEM collection proxies
// ─────────────────────────────────────────────

/// Collection wrapper for OEM states within a segment.
///
/// Example:
///     ```python
///     states = seg.states
///     print(len(states))
///     sv = states[0]
///     print(sv.position)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OEMStates")]
pub struct PyOEMStates {
    parent: Py<PyAny>,
    seg_idx: usize,
}

#[pymethods]
impl PyOEMStates {
    fn __len__(&self, py: Python) -> PyResult<usize> {
        self.parent
            .bind(py)
            .call_method1("_num_states", (self.seg_idx,))?
            .extract()
    }

    fn __getitem__(&self, py: Python, index: isize) -> PyResult<PyOEMStateVector> {
        let len: usize = self.__len__(py)?;
        let actual = normalize_index(index, len, "state")?;
        Ok(PyOEMStateVector {
            mode: StateVectorMode::Proxy {
                parent: self.parent.clone_ref(py),
                seg_idx: self.seg_idx,
                sv_idx: actual,
            },
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyOEMStateIterator>> {
        let py = slf.py();
        Py::new(
            py,
            PyOEMStateIterator {
                parent: slf.parent.clone_ref(py),
                seg_idx: slf.seg_idx,
                index: 0,
            },
        )
    }

    /// Append a state vector to this segment.
    ///
    /// Args:
    ///     sv (OEMStateVector): State vector to append
    fn append(&self, py: Python, sv: &Bound<'_, PyOEMStateVector>) -> PyResult<()> {
        let data = sv.borrow().to_rust_data(py)?;
        let parent_oem: &Bound<PyOEM> = self.parent.bind(py).cast()?;
        let mut oem = parent_oem.borrow_mut();
        let seg = oem.inner.segments.get_mut(self.seg_idx).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err("segment index out of range")
        })?;
        seg.states.push(data);
        Ok(())
    }

    /// Extend with state vectors from an iterable.
    ///
    /// Args:
    ///     iterable: Iterable of OEMStateVector objects
    fn extend(&self, py: Python, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let parent_oem: &Bound<PyOEM> = self.parent.bind(py).cast()?;
        let iterator = iterable.try_iter()?;
        for item in iterator {
            let bound: Bound<'_, PyAny> = item?;
            let sv_ref: &Bound<PyOEMStateVector> = bound.cast()?;
            let data = sv_ref.borrow().to_rust_data(py)?;
            let mut oem = parent_oem.borrow_mut();
            let seg = oem.inner.segments.get_mut(self.seg_idx).ok_or_else(|| {
                pyo3::exceptions::PyIndexError::new_err("segment index out of range")
            })?;
            seg.states.push(data);
        }
        Ok(())
    }

    fn __delitem__(&self, py: Python, index: isize) -> PyResult<()> {
        let len = self.__len__(py)?;
        let actual = normalize_index(index, len, "state")?;
        let parent_oem: &Bound<PyOEM> = self.parent.bind(py).cast()?;
        let mut oem = parent_oem.borrow_mut();
        let seg = oem.inner.segments.get_mut(self.seg_idx).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err("segment index out of range")
        })?;
        seg.states.remove(actual);
        Ok(())
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let len = self.__len__(py)?;
        Ok(format!("OEMStates(len={})", len))
    }
}

/// Iterator over OEM state vectors.
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OEMStateIterator")]
pub struct PyOEMStateIterator {
    parent: Py<PyAny>,
    seg_idx: usize,
    index: usize,
}

#[pymethods]
impl PyOEMStateIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<PyOEMStateVector>> {
        let len: usize = self
            .parent
            .bind(py)
            .call_method1("_num_states", (self.seg_idx,))?
            .extract()?;
        if self.index >= len {
            return Ok(None);
        }
        let sv = PyOEMStateVector {
            mode: StateVectorMode::Proxy {
                parent: self.parent.clone_ref(py),
                seg_idx: self.seg_idx,
                sv_idx: self.index,
            },
        };
        self.index += 1;
        Ok(Some(sv))
    }
}

// ─────────────────────────────────────────────
// PyOEMSegment — typed proxy/owned for a segment
// ─────────────────────────────────────────────

enum SegmentMode {
    Proxy { parent: Py<PyAny>, seg_idx: usize },
    Owned { data: Box<OEMSegment> },
}

/// Proxy for a single OEM segment with property access to metadata and states.
///
/// Segments can be created standalone or accessed via OEM segments collection.
/// Proxy instances propagate mutations back to the parent OEM.
///
/// Args:
///     object_name (str): Spacecraft name
///     object_id (str): International designator
///     center_name (str): Central body name
///     ref_frame (str): Reference frame name
///     time_system (str): Time system name
///     start_time (Epoch): Start time of ephemeris data
///     stop_time (Epoch): Stop time of ephemeris data
///     interpolation (str | None): Interpolation method
///     interpolation_degree (int | None): Interpolation degree
///
/// Example:
///     ```python
///     from brahe import Epoch
///     from brahe.ccsds import OEMSegment
///     start = Epoch.from_datetime(2024, 1, 1, 0, 0, 0)
///     stop = Epoch.from_datetime(2024, 1, 1, 1, 0, 0)
///     seg = OEMSegment(
///         object_name="SAT1", object_id="2024-001A",
///         center_name="EARTH", ref_frame="GCRF", time_system="UTC",
///         start_time=start, stop_time=stop,
///     )
///     print(seg.object_name)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OEMSegment")]
pub struct PyOEMSegment {
    mode: SegmentMode,
}

impl PyOEMSegment {
    fn to_rust_data(&self, py: Python) -> PyResult<OEMSegment> {
        match &self.mode {
            SegmentMode::Owned { data } => Ok(data.as_ref().clone()),
            SegmentMode::Proxy { parent, seg_idx } => {
                let obj_name: String = parent.bind(py).call_method1("_seg_get_object_name", (*seg_idx,))?.extract()?;
                let obj_id: String = parent.bind(py).call_method1("_seg_get_object_id", (*seg_idx,))?.extract()?;
                let ctr_name: String = parent.bind(py).call_method1("_seg_get_center_name", (*seg_idx,))?.extract()?;
                let rf_str: String = parent.bind(py).call_method1("_seg_get_ref_frame", (*seg_idx,))?.extract()?;
                let ts_str: String = parent.bind(py).call_method1("_seg_get_time_system", (*seg_idx,))?.extract()?;
                let start_obj = parent.bind(py).call_method1("_seg_get_start_time", (*seg_idx,))?;
                let start: PyEpoch = extract_epoch(&start_obj)?;
                let stop_obj = parent.bind(py).call_method1("_seg_get_stop_time", (*seg_idx,))?;
                let stop: PyEpoch = extract_epoch(&stop_obj)?;
                let interp: Option<String> = parent.bind(py).call_method1("_seg_get_interpolation", (*seg_idx,))?.extract()?;
                let interp_deg: Option<u32> = parent.bind(py).call_method1("_seg_get_interpolation_degree", (*seg_idx,))?.extract()?;
                let comments: Vec<String> = parent.bind(py).call_method1("_seg_get_comments", (*seg_idx,))?.extract()?;
                let num_states: usize = parent.bind(py).call_method1("_num_states", (*seg_idx,))?.extract()?;

                let mut states = Vec::with_capacity(num_states);
                for i in 0..num_states {
                    let epoch_obj = parent.bind(py).call_method1("_sv_get_epoch", (*seg_idx, i))?;
                    let sv_epoch: PyEpoch = extract_epoch(&epoch_obj)?;
                    let pos: Vec<f64> = parent.bind(py).call_method1("_sv_get_position", (*seg_idx, i))?.extract()?;
                    let vel: Vec<f64> = parent.bind(py).call_method1("_sv_get_velocity", (*seg_idx, i))?.extract()?;
                    let acc: Option<Vec<f64>> = parent.bind(py).call_method1("_sv_get_acceleration", (*seg_idx, i))?.extract()?;
                    states.push(OEMStateVector {
                        epoch: sv_epoch.obj,
                        position: vec_to_array3(pos, "position")?,
                        velocity: vec_to_array3(vel, "velocity")?,
                        acceleration: match acc { Some(v) => Some(vec_to_array3(v, "acceleration")?), None => None },
                    });
                }

                let rf = CCSDSRefFrame::parse(&rf_str);
                let ts = CCSDSTimeSystem::parse(&ts_str).map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

                Ok(OEMSegment {
                    metadata: OEMMetadata {
                        object_name: obj_name, object_id: obj_id, center_name: ctr_name,
                        ref_frame: rf, ref_frame_epoch: None, time_system: ts,
                        start_time: start.obj, useable_start_time: None, useable_stop_time: None, stop_time: stop.obj,
                        interpolation: interp, interpolation_degree: interp_deg,
                        comments: Vec::new(),
                    },
                    comments,
                    states,
                    covariances: Vec::new(),
                })
            }
        }
    }
}

#[pymethods]
impl PyOEMSegment {
    #[new]
    #[pyo3(signature = (object_name, object_id, center_name, ref_frame, time_system, start_time, stop_time, interpolation=None, interpolation_degree=None))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        object_name: String,
        object_id: String,
        center_name: String,
        ref_frame: String,
        time_system: String,
        start_time: PyEpoch,
        stop_time: PyEpoch,
        interpolation: Option<String>,
        interpolation_degree: Option<u32>,
    ) -> PyResult<Self> {
        let rf = CCSDSRefFrame::parse(&ref_frame);
        let ts = CCSDSTimeSystem::parse(&time_system).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid time_system: {}", e))
        })?;
        Ok(Self {
            mode: SegmentMode::Owned {
                data: Box::new(OEMSegment {
                    metadata: OEMMetadata {
                        object_name,
                        object_id,
                        center_name,
                        ref_frame: rf,
                        ref_frame_epoch: None,
                        time_system: ts,
                        start_time: start_time.obj,
                        useable_start_time: None,
                        useable_stop_time: None,
                        stop_time: stop_time.obj,
                        interpolation,
                        interpolation_degree,
                        comments: Vec::new(),
                    },
                    comments: Vec::new(),
                    states: Vec::new(),
                    covariances: Vec::new(),
                }),
            },
        })
    }

    /// Spacecraft name.
    ///
    /// Returns:
    ///     str: Object name
    #[getter]
    fn object_name(&self, py: Python) -> PyResult<String> {
        match &self.mode {
            SegmentMode::Owned { data } => Ok(data.metadata.object_name.clone()),
            SegmentMode::Proxy { parent, seg_idx } => {
                parent.bind(py).call_method1("_seg_get_object_name", (*seg_idx,))?.extract()
            }
        }
    }

    /// Set spacecraft name.
    ///
    /// Args:
    ///     val (str): Object name
    #[setter]
    fn set_object_name(&mut self, py: Python, val: String) -> PyResult<()> {
        match &mut self.mode {
            SegmentMode::Owned { data } => { data.metadata.object_name = val; Ok(()) }
            SegmentMode::Proxy { parent, seg_idx } => {
                parent.bind(py).call_method1("_seg_set_object_name", (*seg_idx, val))?; Ok(())
            }
        }
    }

    /// International designator (e.g., "1996-062A").
    ///
    /// Returns:
    ///     str: Object ID
    #[getter]
    fn object_id(&self, py: Python) -> PyResult<String> {
        match &self.mode {
            SegmentMode::Owned { data } => Ok(data.metadata.object_id.clone()),
            SegmentMode::Proxy { parent, seg_idx } => {
                parent.bind(py).call_method1("_seg_get_object_id", (*seg_idx,))?.extract()
            }
        }
    }

    /// Set international designator.
    ///
    /// Args:
    ///     val (str): Object ID
    #[setter]
    fn set_object_id(&mut self, py: Python, val: String) -> PyResult<()> {
        match &mut self.mode {
            SegmentMode::Owned { data } => { data.metadata.object_id = val; Ok(()) }
            SegmentMode::Proxy { parent, seg_idx } => {
                parent.bind(py).call_method1("_seg_set_object_id", (*seg_idx, val))?; Ok(())
            }
        }
    }

    /// Central body name (e.g., "EARTH", "MARS BARYCENTER").
    ///
    /// Returns:
    ///     str: Center name
    #[getter]
    fn center_name(&self, py: Python) -> PyResult<String> {
        match &self.mode {
            SegmentMode::Owned { data } => Ok(data.metadata.center_name.clone()),
            SegmentMode::Proxy { parent, seg_idx } => {
                parent.bind(py).call_method1("_seg_get_center_name", (*seg_idx,))?.extract()
            }
        }
    }

    /// Set central body name.
    ///
    /// Args:
    ///     val (str): Center name
    #[setter]
    fn set_center_name(&mut self, py: Python, val: String) -> PyResult<()> {
        match &mut self.mode {
            SegmentMode::Owned { data } => { data.metadata.center_name = val; Ok(()) }
            SegmentMode::Proxy { parent, seg_idx } => {
                parent.bind(py).call_method1("_seg_set_center_name", (*seg_idx, val))?; Ok(())
            }
        }
    }

    /// Reference frame name (e.g., "J2000", "GCRF").
    ///
    /// Returns:
    ///     str: Reference frame
    #[getter]
    fn ref_frame(&self, py: Python) -> PyResult<String> {
        match &self.mode {
            SegmentMode::Owned { data } => Ok(format!("{}", data.metadata.ref_frame)),
            SegmentMode::Proxy { parent, seg_idx } => {
                parent.bind(py).call_method1("_seg_get_ref_frame", (*seg_idx,))?.extract()
            }
        }
    }

    /// Set reference frame.
    ///
    /// Args:
    ///     val (str): Reference frame name
    #[setter]
    fn set_ref_frame(&mut self, py: Python, val: String) -> PyResult<()> {
        match &mut self.mode {
            SegmentMode::Owned { data } => { data.metadata.ref_frame = CCSDSRefFrame::parse(&val); Ok(()) }
            SegmentMode::Proxy { parent, seg_idx } => {
                parent.bind(py).call_method1("_seg_set_ref_frame", (*seg_idx, val))?; Ok(())
            }
        }
    }

    /// Time system name (e.g., "UTC", "TDB").
    ///
    /// Returns:
    ///     str: Time system
    #[getter]
    fn time_system(&self, py: Python) -> PyResult<String> {
        match &self.mode {
            SegmentMode::Owned { data } => Ok(format!("{}", data.metadata.time_system)),
            SegmentMode::Proxy { parent, seg_idx } => {
                parent.bind(py).call_method1("_seg_get_time_system", (*seg_idx,))?.extract()
            }
        }
    }

    /// Set time system.
    ///
    /// Args:
    ///     val (str): Time system name
    #[setter]
    fn set_time_system(&mut self, py: Python, val: String) -> PyResult<()> {
        match &mut self.mode {
            SegmentMode::Owned { data } => {
                let ts = CCSDSTimeSystem::parse(&val).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!("Invalid time_system: {}", e))
                })?;
                data.metadata.time_system = ts;
                Ok(())
            }
            SegmentMode::Proxy { parent, seg_idx } => {
                parent.bind(py).call_method1("_seg_set_time_system", (*seg_idx, val))?; Ok(())
            }
        }
    }

    /// Start time of the ephemeris data.
    ///
    /// Returns:
    ///     Epoch: Start time
    #[getter]
    fn start_time(&self, py: Python) -> PyResult<PyEpoch> {
        match &self.mode {
            SegmentMode::Owned { data } => Ok(PyEpoch { obj: data.metadata.start_time }),
            SegmentMode::Proxy { parent, seg_idx } => {
                let obj = parent.bind(py).call_method1("_seg_get_start_time", (*seg_idx,))?;
                extract_epoch(&obj)
            }
        }
    }

    /// Set start time.
    ///
    /// Args:
    ///     val (Epoch): Start time
    #[setter]
    fn set_start_time(&mut self, py: Python, val: PyEpoch) -> PyResult<()> {
        match &mut self.mode {
            SegmentMode::Owned { data } => { data.metadata.start_time = val.obj; Ok(()) }
            SegmentMode::Proxy { parent, seg_idx } => {
                parent.bind(py).call_method1("_seg_set_start_time", (*seg_idx, val))?; Ok(())
            }
        }
    }

    /// Stop time of the ephemeris data.
    ///
    /// Returns:
    ///     Epoch: Stop time
    #[getter]
    fn stop_time(&self, py: Python) -> PyResult<PyEpoch> {
        match &self.mode {
            SegmentMode::Owned { data } => Ok(PyEpoch { obj: data.metadata.stop_time }),
            SegmentMode::Proxy { parent, seg_idx } => {
                let obj = parent.bind(py).call_method1("_seg_get_stop_time", (*seg_idx,))?;
                extract_epoch(&obj)
            }
        }
    }

    /// Set stop time.
    ///
    /// Args:
    ///     val (Epoch): Stop time
    #[setter]
    fn set_stop_time(&mut self, py: Python, val: PyEpoch) -> PyResult<()> {
        match &mut self.mode {
            SegmentMode::Owned { data } => { data.metadata.stop_time = val.obj; Ok(()) }
            SegmentMode::Proxy { parent, seg_idx } => {
                parent.bind(py).call_method1("_seg_set_stop_time", (*seg_idx, val))?; Ok(())
            }
        }
    }

    /// Interpolation method (e.g., "HERMITE", "LAGRANGE"), or None.
    ///
    /// Returns:
    ///     str: Interpolation method, or None
    #[getter]
    fn interpolation(&self, py: Python) -> PyResult<Option<String>> {
        match &self.mode {
            SegmentMode::Owned { data } => Ok(data.metadata.interpolation.clone()),
            SegmentMode::Proxy { parent, seg_idx } => {
                parent.bind(py).call_method1("_seg_get_interpolation", (*seg_idx,))?.extract()
            }
        }
    }

    /// Set interpolation method.
    ///
    /// Args:
    ///     val (str | None): Interpolation method
    #[setter]
    fn set_interpolation(&mut self, py: Python, val: Option<String>) -> PyResult<()> {
        match &mut self.mode {
            SegmentMode::Owned { data } => { data.metadata.interpolation = val; Ok(()) }
            SegmentMode::Proxy { parent, seg_idx } => {
                parent.bind(py).call_method1("_seg_set_interpolation", (*seg_idx, val))?; Ok(())
            }
        }
    }

    /// Interpolation degree, or None.
    ///
    /// Returns:
    ///     int: Interpolation degree, or None
    #[getter]
    fn interpolation_degree(&self, py: Python) -> PyResult<Option<u32>> {
        match &self.mode {
            SegmentMode::Owned { data } => Ok(data.metadata.interpolation_degree),
            SegmentMode::Proxy { parent, seg_idx } => {
                parent.bind(py).call_method1("_seg_get_interpolation_degree", (*seg_idx,))?.extract()
            }
        }
    }

    /// Set interpolation degree.
    ///
    /// Args:
    ///     val (int | None): Interpolation degree
    #[setter]
    fn set_interpolation_degree(&mut self, py: Python, val: Option<u32>) -> PyResult<()> {
        match &mut self.mode {
            SegmentMode::Owned { data } => { data.metadata.interpolation_degree = val; Ok(()) }
            SegmentMode::Proxy { parent, seg_idx } => {
                parent.bind(py).call_method1("_seg_set_interpolation_degree", (*seg_idx, val))?; Ok(())
            }
        }
    }

    /// Comments from the data block.
    ///
    /// Returns:
    ///     list[str]: Comments
    #[getter]
    fn comments(&self, py: Python) -> PyResult<Vec<String>> {
        match &self.mode {
            SegmentMode::Owned { data } => Ok(data.comments.clone()),
            SegmentMode::Proxy { parent, seg_idx } => {
                parent.bind(py).call_method1("_seg_get_comments", (*seg_idx,))?.extract()
            }
        }
    }

    /// Set comments.
    ///
    /// Args:
    ///     val (list[str]): Comments
    #[setter]
    fn set_comments(&mut self, py: Python, val: Vec<String>) -> PyResult<()> {
        match &mut self.mode {
            SegmentMode::Owned { data } => { data.comments = val; Ok(()) }
            SegmentMode::Proxy { parent, seg_idx } => {
                parent.bind(py).call_method1("_seg_set_comments", (*seg_idx, val))?; Ok(())
            }
        }
    }

    /// Number of state vectors in this segment.
    ///
    /// Returns:
    ///     int: Number of states
    #[getter]
    fn num_states(&self, py: Python) -> PyResult<usize> {
        match &self.mode {
            SegmentMode::Owned { data } => Ok(data.states.len()),
            SegmentMode::Proxy { parent, seg_idx } => {
                parent.bind(py).call_method1("_num_states", (*seg_idx,))?.extract()
            }
        }
    }

    /// Number of covariance matrices in this segment.
    ///
    /// Returns:
    ///     int: Number of covariances
    #[getter]
    fn num_covariances(&self, py: Python) -> PyResult<usize> {
        match &self.mode {
            SegmentMode::Owned { data } => Ok(data.covariances.len()),
            SegmentMode::Proxy { parent, seg_idx } => {
                parent.bind(py).call_method1("_seg_get_num_covariances", (*seg_idx,))?.extract()
            }
        }
    }

    /// Collection of state vectors, supporting len/indexing/iteration.
    ///
    /// For proxy segments, returns an OEMStates proxy collection.
    /// For owned segments, returns a Python list of standalone OEMStateVector objects.
    ///
    /// Returns:
    ///     OEMStates | list[OEMStateVector]: State vector collection
    #[getter]
    fn states(&self, py: Python) -> PyResult<Py<PyAny>> {
        match &self.mode {
            SegmentMode::Proxy { parent, seg_idx } => {
                let obj = Py::new(py, PyOEMStates {
                    parent: parent.clone_ref(py),
                    seg_idx: *seg_idx,
                })?;
                Ok(obj.into_any())
            }
            SegmentMode::Owned { data } => {
                let list = pyo3::types::PyList::empty(py);
                for sv in &data.states {
                    let py_sv = Py::new(py, PyOEMStateVector {
                        mode: StateVectorMode::Owned { data: sv.clone() },
                    })?;
                    list.append(py_sv)?;
                }
                Ok(list.into_any().unbind())
            }
        }
    }

    /// Add a state vector to this segment.
    ///
    /// Args:
    ///     epoch (Epoch): Epoch of the state vector
    ///     position (list[float]): Position [x, y, z] in meters
    ///     velocity (list[float]): Velocity [vx, vy, vz] in m/s
    ///     acceleration (list[float] | None): Optional acceleration [ax, ay, az] in m/s²
    ///
    /// Returns:
    ///     int: Index of the new state vector
    #[pyo3(signature = (epoch, position, velocity, acceleration=None))]
    fn add_state(
        &mut self,
        py: Python,
        epoch: PyEpoch,
        position: &pyo3::Bound<'_, pyo3::types::PyAny>,
        velocity: &pyo3::Bound<'_, pyo3::types::PyAny>,
        acceleration: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>,
    ) -> PyResult<usize> {
        let pos = pyany_to_array3(position, "position")?;
        let vel = pyany_to_array3(velocity, "velocity")?;
        let acc = pyany_to_optional_array3(acceleration, "acceleration")?;
        match &mut self.mode {
            SegmentMode::Owned { data } => {
                data.states.push(OEMStateVector { epoch: epoch.obj, position: pos, velocity: vel, acceleration: acc });
                Ok(data.states.len() - 1)
            }
            SegmentMode::Proxy { parent, seg_idx } => {
                let acc_vec: Option<Vec<f64>> = acc.map(|a| a.to_vec());
                parent.bind(py).call_method1("_add_state", (*seg_idx, epoch, pos.to_vec(), vel.to_vec(), acc_vec))?.extract()
            }
        }
    }

    /// Bulk-add states from an orbital trajectory to this segment.
    ///
    /// Iterates the trajectory's epochs and states, extracting position and
    /// velocity components to create OEM state vectors.
    ///
    /// Args:
    ///     trajectory (OrbitTrajectory): Orbital trajectory to import states from
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     from brahe.ccsds import OEMSegment
    ///     seg = OEMSegment(
    ///         object_name="SAT", object_id="2024-001A",
    ///         center_name="EARTH", ref_frame="EME2000", time_system="UTC",
    ///         start_time=epoch, stop_time=stop_epoch,
    ///     )
    ///     seg.add_trajectory(prop.trajectory)
    ///     ```
    fn add_trajectory(&mut self, py: Python, trajectory: PyRef<PyOrbitalTrajectory>) -> PyResult<()> {
        let traj = &trajectory.trajectory;
        match &mut self.mode {
            SegmentMode::Owned { data } => {
                push_trajectory_states(data, traj).map_err(|e| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Failed to convert trajectory states: {}", e
                    ))
                })?;
                Ok(())
            }
            SegmentMode::Proxy { parent, seg_idx } => {
                // Get the ref_frame from the parent OEM's segment metadata
                let parent_ref = parent.bind(py);
                let oem_bound = parent_ref.cast::<PyOEM>()
                    .map_err(|_| pyo3::exceptions::PyRuntimeError::new_err(
                        "Parent is not an OEM object"
                    ))?;
                let ref_frame = oem_bound.borrow().inner.segments[*seg_idx].metadata.ref_frame.clone();
                let orbit_frame = ccsds_ref_frame_to_orbit_frame(&ref_frame).map_err(|e| {
                    pyo3::exceptions::PyValueError::new_err(format!(
                        "Unsupported ref_frame for trajectory conversion: {}", e
                    ))
                })?;

                for epoch in traj.epochs.iter() {
                    let state = match orbit_frame {
                        OrbitFrame::EME2000 => traj.state_eme2000(*epoch),
                        OrbitFrame::GCRF => traj.state_gcrf(*epoch),
                        OrbitFrame::ECI => traj.state_eci(*epoch),
                        OrbitFrame::ECEF | OrbitFrame::ITRF => traj.state_itrf(*epoch),
                    }.map_err(|e| {
                        pyo3::exceptions::PyRuntimeError::new_err(format!(
                            "Failed to convert trajectory state at {}: {}", epoch, e
                        ))
                    })?;

                    let pos = vec![state[0], state[1], state[2]];
                    let vel = vec![state[3], state[4], state[5]];
                    let acc: Option<Vec<f64>> = None;
                    parent.bind(py).call_method1(
                        "_add_state",
                        (*seg_idx, PyEpoch { obj: *epoch }, pos, vel, acc),
                    )?;
                }
                Ok(())
            }
        }
    }

    /// Remove a state vector from this segment by index.
    ///
    /// Args:
    ///     idx (int): Index of the state vector to remove
    fn remove_state(&mut self, py: Python, idx: usize) -> PyResult<()> {
        match &mut self.mode {
            SegmentMode::Owned { data } => {
                if idx >= data.states.len() {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "state index {} out of range (have {})", idx, data.states.len()
                    )));
                }
                data.states.remove(idx);
                Ok(())
            }
            SegmentMode::Proxy { parent, seg_idx } => {
                parent.bind(py).call_method1("_remove_state", (*seg_idx, idx))?; Ok(())
            }
        }
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let name: String = self.object_name(py)?;
        let n: usize = self.num_states(py)?;
        let frame: String = self.ref_frame(py)?;
        Ok(format!(
            "OEMSegment(object='{}', states={}, frame='{}')",
            name, n, frame
        ))
    }
}

/// Collection wrapper for OEM segments, supporting len/indexing/iteration.
///
/// Example:
///     ```python
///     segments = oem.segments
///     print(len(segments))
///     seg = segments[0]
///     for seg in segments:
///         print(seg.object_name)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OEMSegments")]
pub struct PyOEMSegments {
    parent: Py<PyAny>,
}

#[pymethods]
impl PyOEMSegments {
    fn __len__(&self, py: Python) -> PyResult<usize> {
        self.parent
            .bind(py)
            .call_method0("_num_segments")?
            .extract()
    }

    fn __getitem__(&self, py: Python, index: isize) -> PyResult<PyOEMSegment> {
        let len: usize = self.__len__(py)?;
        let actual = normalize_index(index, len, "segment")?;
        Ok(PyOEMSegment {
            mode: SegmentMode::Proxy {
                parent: self.parent.clone_ref(py),
                seg_idx: actual,
            },
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyOEMSegmentIterator>> {
        let py = slf.py();
        Py::new(
            py,
            PyOEMSegmentIterator {
                parent: slf.parent.clone_ref(py),
                index: 0,
            },
        )
    }

    /// Append a segment to the OEM.
    ///
    /// Args:
    ///     seg (OEMSegment): Segment to append
    fn append(&self, py: Python, seg: &Bound<'_, PyOEMSegment>) -> PyResult<()> {
        let data = seg.borrow().to_rust_data(py)?;
        let parent_oem: &Bound<PyOEM> = self.parent.bind(py).cast()?;
        parent_oem.borrow_mut().inner.segments.push(data);
        Ok(())
    }

    /// Extend with segments from an iterable.
    ///
    /// Args:
    ///     iterable: Iterable of OEMSegment objects
    fn extend(&self, py: Python, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let parent_oem: &Bound<PyOEM> = self.parent.bind(py).cast()?;
        let iterator = iterable.try_iter()?;
        for item in iterator {
            let bound: Bound<'_, PyAny> = item?;
            let seg_ref: &Bound<PyOEMSegment> = bound.cast()?;
            let data = seg_ref.borrow().to_rust_data(py)?;
            parent_oem.borrow_mut().inner.segments.push(data);
        }
        Ok(())
    }

    fn __delitem__(&self, py: Python, index: isize) -> PyResult<()> {
        let len = self.__len__(py)?;
        let actual = normalize_index(index, len, "segment")?;
        let parent_oem: &Bound<PyOEM> = self.parent.bind(py).cast()?;
        parent_oem.borrow_mut().inner.segments.remove(actual);
        Ok(())
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let len = self.__len__(py)?;
        Ok(format!("OEMSegments(len={})", len))
    }
}

/// Iterator over OEM segments.
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OEMSegmentIterator")]
pub struct PyOEMSegmentIterator {
    parent: Py<PyAny>,
    index: usize,
}

#[pymethods]
impl PyOEMSegmentIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<PyOEMSegment>> {
        let len: usize = self
            .parent
            .bind(py)
            .call_method0("_num_segments")?
            .extract()?;
        if self.index >= len {
            return Ok(None);
        }
        let seg = PyOEMSegment {
            mode: SegmentMode::Proxy {
                parent: self.parent.clone_ref(py),
                seg_idx: self.index,
            },
        };
        self.index += 1;
        Ok(Some(seg))
    }
}

// ─────────────────────────────────────────────
// PyOEM — Orbit Ephemeris Message
// ─────────────────────────────────────────────

/// Python wrapper for CCSDS Orbit Ephemeris Message (OEM).
///
/// OEM messages contain time-ordered sequences of state vectors (position
/// and velocity), optionally with accelerations and covariance matrices.
///
/// Header-level fields are properties with getters and setters. Segment data
/// is accessed via the ``segments`` property which supports indexing,
/// iteration, and mutation.
///
/// Example:
///     ```python
///     from brahe.ccsds import OEM
///     from brahe import Epoch
///
///     # Parse existing
///     oem = OEM.from_file("ephemeris.oem")
///     print(oem.originator)
///     seg = oem.segments[0]
///     sv = seg.states[0]
///     print(sv.position)
///
///     # Construct from scratch
///     start = Epoch.from_datetime(2024, 1, 1, 0, 0, 0)
///     stop = Epoch.from_datetime(2024, 1, 1, 1, 0, 0)
///     oem = OEM(originator="MY_ORG")
///     oem.add_segment(
///         object_name="SAT1", object_id="2024-001A",
///         center_name="EARTH", ref_frame="GCRF", time_system="UTC",
///         start_time=start, stop_time=stop,
///     )
///     oem.segments[0].add_state(
///         epoch=start, position=[7000e3, 0, 0], velocity=[0, 7500, 0],
///     )
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OEM")]
pub struct PyOEM {
    inner: RustOEM,
}

#[pymethods]
impl PyOEM {
    // --- constructor ---

    /// Create a new empty OEM message.
    ///
    /// Args:
    ///     originator (str): Originator of the message
    ///     format_version (float): CCSDS format version (default 3.0)
    ///     classification (str | None): Optional classification string
    ///
    /// Returns:
    ///     OEM: New empty OEM message
    ///
    /// Example:
    ///     ```python
    ///     from brahe.ccsds import OEM
    ///     oem = OEM(originator="MY_ORG")
    ///     ```
    #[new]
    #[pyo3(signature = (originator, format_version=3.0, classification=None))]
    fn new(originator: String, format_version: f64, classification: Option<String>) -> Self {
        PyOEM {
            inner: RustOEM {
                header: ODMHeader {
                    format_version,
                    classification,
                    creation_date: crate::time::Epoch::now(),
                    originator,
                    message_id: None,
                    comments: Vec::new(),
                },
                segments: Vec::new(),
            },
        }
    }

    // --- static constructors ---

    /// Parse an OEM from a string, auto-detecting the format (KVN, XML, or JSON).
    ///
    /// Args:
    ///     content (str): String content of the OEM message
    ///
    /// Returns:
    ///     OEM: Parsed OEM message
    #[staticmethod]
    #[allow(clippy::should_implement_trait)]
    fn from_str(content: &str) -> PyResult<Self> {
        let inner = RustOEM::from_str(content)?;
        Ok(PyOEM { inner })
    }

    /// Parse an OEM from a file, auto-detecting the format.
    ///
    /// Args:
    ///     path (str): Path to the OEM file
    ///
    /// Returns:
    ///     OEM: Parsed OEM message
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let inner = RustOEM::from_file(path)?;
        Ok(PyOEM { inner })
    }

    // --- serialization methods ---

    /// Write the OEM to a string in the specified format.
    ///
    /// Args:
    ///     format (str): Output format - "KVN", "XML", or "JSON"
    ///
    /// Returns:
    ///     str: Serialized OEM string
    fn to_string(&self, format: &str) -> PyResult<String> {
        let fmt = parse_format(format)?;
        let result = self.inner.to_string(fmt)?;
        Ok(result)
    }

    /// Write the OEM to JSON with explicit key case control.
    ///
    /// Args:
    ///     uppercase_keys (bool): If True, use uppercase CCSDS keywords. Default: False.
    ///
    /// Returns:
    ///     str: Serialized JSON string
    #[pyo3(signature = (uppercase_keys=false))]
    fn to_json_string(&self, uppercase_keys: bool) -> PyResult<String> {
        let key_case = if uppercase_keys {
            crate::ccsds::common::CCSDSJsonKeyCase::Upper
        } else {
            crate::ccsds::common::CCSDSJsonKeyCase::Lower
        };
        let result = self.inner.to_json_string(key_case)?;
        Ok(result)
    }

    /// Write the OEM to a file in the specified format.
    ///
    /// Args:
    ///     path (str): Output file path
    ///     format (str): Output format - "KVN", "XML", or "JSON"
    fn to_file(&self, path: &str, format: &str) -> PyResult<()> {
        let fmt = parse_format(format)?;
        self.inner.to_file(path, fmt)?;
        Ok(())
    }

    /// Convert the OEM to a Python dictionary.
    ///
    /// Epochs are serialized as CCSDS datetime strings for JSON/dict compatibility.
    ///
    /// Returns:
    ///     dict: Dictionary representation of the OEM
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new(py);

        // Header
        let header = pyo3::types::PyDict::new(py);
        header.set_item("format_version", self.inner.header.format_version)?;
        header.set_item(
            "classification",
            self.inner.header.classification.as_deref(),
        )?;
        header.set_item(
            "creation_date",
            crate::ccsds::common::format_ccsds_datetime(&self.inner.header.creation_date),
        )?;
        header.set_item("originator", &self.inner.header.originator)?;
        header.set_item("message_id", self.inner.header.message_id.as_deref())?;
        header.set_item("comments", &self.inner.header.comments)?;
        dict.set_item("header", header)?;

        // Segments
        let segments = pyo3::types::PyList::empty(py);
        for seg in &self.inner.segments {
            let seg_dict = pyo3::types::PyDict::new(py);

            // Metadata
            let meta = pyo3::types::PyDict::new(py);
            meta.set_item("object_name", &seg.metadata.object_name)?;
            meta.set_item("object_id", &seg.metadata.object_id)?;
            meta.set_item("center_name", &seg.metadata.center_name)?;
            meta.set_item("ref_frame", format!("{}", seg.metadata.ref_frame))?;
            meta.set_item("time_system", format!("{}", seg.metadata.time_system))?;
            meta.set_item(
                "start_time",
                crate::ccsds::common::format_ccsds_datetime(&seg.metadata.start_time),
            )?;
            meta.set_item(
                "stop_time",
                crate::ccsds::common::format_ccsds_datetime(&seg.metadata.stop_time),
            )?;
            meta.set_item("interpolation", seg.metadata.interpolation.as_deref())?;
            meta.set_item("interpolation_degree", seg.metadata.interpolation_degree)?;
            meta.set_item("comments", &seg.metadata.comments)?;
            seg_dict.set_item("metadata", meta)?;

            // States
            let states = pyo3::types::PyList::empty(py);
            for sv in &seg.states {
                let sv_dict = pyo3::types::PyDict::new(py);
                sv_dict.set_item(
                    "epoch",
                    crate::ccsds::common::format_ccsds_datetime(&sv.epoch),
                )?;
                sv_dict.set_item("position", sv.position.to_vec())?;
                sv_dict.set_item("velocity", sv.velocity.to_vec())?;
                sv_dict.set_item("acceleration", sv.acceleration.map(|a| a.to_vec()))?;
                states.append(sv_dict)?;
            }
            seg_dict.set_item("states", states)?;
            seg_dict.set_item("comments", &seg.comments)?;
            seg_dict.set_item("num_covariances", seg.covariances.len())?;

            segments.append(seg_dict)?;
        }
        dict.set_item("segments", segments)?;

        Ok(dict)
    }

    // --- shortcut method ---

    /// Get a state vector from a segment by index (shortcut).
    ///
    /// This is a convenience method equivalent to ``oem.segments[segment_idx].states[state_idx]``.
    ///
    /// Args:
    ///     segment_idx (int): Segment index (0-based)
    ///     state_idx (int): State vector index (0-based)
    ///
    /// Returns:
    ///     OEMStateVector: State vector proxy
    fn state(
        slf: Bound<'_, Self>,
        segment_idx: usize,
        state_idx: usize,
    ) -> PyResult<PyOEMStateVector> {
        let inner = slf.borrow();
        let num_segs = inner.inner.segments.len();
        if segment_idx >= num_segs {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "segment index {} out of range (have {})",
                segment_idx, num_segs
            )));
        }
        let num_states = inner.inner.segments[segment_idx].states.len();
        if state_idx >= num_states {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "state index {} out of range (have {})",
                state_idx, num_states
            )));
        }
        let parent: Py<PyAny> = slf.clone().into_any().unbind();
        Ok(PyOEMStateVector {
            mode: StateVectorMode::Proxy {
                parent,
                seg_idx: segment_idx,
                sv_idx: state_idx,
            },
        })
    }

    // --- header-level properties ---

    /// CCSDS format version.
    ///
    /// Returns:
    ///     float: Format version
    #[getter]
    fn format_version(&self) -> f64 {
        self.inner.header.format_version
    }

    /// Set CCSDS format version.
    ///
    /// Args:
    ///     val (float): Format version
    #[setter]
    fn set_format_version(&mut self, val: f64) {
        self.inner.header.format_version = val;
    }

    /// Originator of the message.
    ///
    /// Returns:
    ///     str: Originator string
    #[getter]
    fn originator(&self) -> String {
        self.inner.header.originator.clone()
    }

    /// Set originator.
    ///
    /// Args:
    ///     val (str): Originator string
    #[setter]
    fn set_originator(&mut self, val: String) {
        self.inner.header.originator = val;
    }

    /// Classification of the message, or None.
    ///
    /// Returns:
    ///     str: Classification string, or None
    #[getter]
    fn classification(&self) -> Option<String> {
        self.inner.header.classification.clone()
    }

    /// Set classification.
    ///
    /// Args:
    ///     val (str | None): Classification string
    #[setter]
    fn set_classification(&mut self, val: Option<String>) {
        self.inner.header.classification = val;
    }

    /// Creation date of the message.
    ///
    /// Returns:
    ///     Epoch: Creation date
    #[getter]
    fn creation_date(&self) -> PyEpoch {
        PyEpoch {
            obj: self.inner.header.creation_date,
        }
    }

    /// Set creation date.
    ///
    /// Args:
    ///     val (Epoch): Creation date
    #[setter]
    fn set_creation_date(&mut self, val: PyEpoch) {
        self.inner.header.creation_date = val.obj;
    }

    /// Message ID, or None.
    ///
    /// Returns:
    ///     str: Message ID, or None
    #[getter]
    fn message_id(&self) -> Option<String> {
        self.inner.header.message_id.clone()
    }

    /// Set message ID.
    ///
    /// Args:
    ///     val (str | None): Message ID
    #[setter]
    fn set_message_id(&mut self, val: Option<String>) {
        self.inner.header.message_id = val;
    }

    /// Segment collection, supporting len/indexing/iteration.
    ///
    /// Returns:
    ///     OEMSegments: Collection of OEM segments
    #[getter]
    fn segments(slf: Bound<'_, Self>) -> PyOEMSegments {
        let parent: Py<PyAny> = slf.clone().into_any().unbind();
        PyOEMSegments { parent }
    }

    // --- trajectory interop methods ---

    /// Convert a single OEM segment to an OrbitTrajectory.
    ///
    /// The trajectory contains Cartesian state vectors (position/velocity)
    /// in the reference frame specified by the segment metadata.
    ///
    /// Args:
    ///     segment_idx (int): Index of the segment to convert (0-based)
    ///
    /// Returns:
    ///     OrbitalTrajectory: Trajectory containing the segment's state vectors
    ///
    /// Raises:
    ///     BraheError: If segment index is out of range or frame is unsupported
    ///
    /// Example:
    ///     ```python
    ///     from brahe.ccsds import OEM
    ///     oem = OEM.from_file("ephemeris.oem")
    ///     traj = oem.segment_to_trajectory(0)
    ///     print(f"States: {len(traj)}")
    ///     ```
    fn segment_to_trajectory(&self, segment_idx: usize) -> PyResult<PyOrbitalTrajectory> {
        let dtraj = self.inner.segment_to_trajectory(segment_idx)?;
        Ok(PyOrbitalTrajectory { trajectory: dtraj })
    }

    /// Convert all OEM segments to OrbitTrajectory objects.
    ///
    /// Returns one trajectory per segment, each containing the segment's
    /// state vectors in its reference frame.
    ///
    /// Returns:
    ///     list[OrbitalTrajectory]: One trajectory per segment
    ///
    /// Raises:
    ///     BraheError: If any segment's frame is unsupported
    ///
    /// Example:
    ///     ```python
    ///     from brahe.ccsds import OEM
    ///     oem = OEM.from_file("multi_segment.oem")
    ///     trajs = oem.to_trajectories()
    ///     for t in trajs:
    ///         print(f"{t.name}: {len(t)} states")
    ///     ```
    fn to_trajectories(&self) -> PyResult<Vec<PyOrbitalTrajectory>> {
        let n = self.inner.segments.len();
        (0..n)
            .map(|i| self.segment_to_trajectory(i))
            .collect()
    }

    // --- builder methods ---

    /// Add a new segment to the OEM.
    ///
    /// Args:
    ///     object_name (str): Spacecraft name
    ///     object_id (str): International designator
    ///     center_name (str): Central body name
    ///     ref_frame (str): Reference frame name
    ///     time_system (str): Time system name
    ///     start_time (Epoch): Start time of ephemeris data
    ///     stop_time (Epoch): Stop time of ephemeris data
    ///     interpolation (str | None): Interpolation method
    ///     interpolation_degree (int | None): Interpolation degree
    ///     trajectory (OrbitTrajectory | None): Optional trajectory to populate states from
    ///
    /// Returns:
    ///     int: Index of the new segment
    ///
    /// Example:
    ///     ```python
    ///     from brahe import Epoch
    ///     from brahe.ccsds import OEM
    ///     oem = OEM(originator="MY_ORG")
    ///     start = Epoch.from_datetime(2024, 1, 1, 0, 0, 0)
    ///     stop = Epoch.from_datetime(2024, 1, 1, 1, 0, 0)
    ///     idx = oem.add_segment(
    ///         object_name="SAT1", object_id="2024-001A",
    ///         center_name="EARTH", ref_frame="GCRF", time_system="UTC",
    ///         start_time=start, stop_time=stop,
    ///         trajectory=prop.trajectory,
    ///     )
    ///     ```
    #[pyo3(signature = (object_name, object_id, center_name, ref_frame, time_system, start_time, stop_time, interpolation=None, interpolation_degree=None, trajectory=None))]
    #[allow(clippy::too_many_arguments)]
    fn add_segment(
        &mut self,
        object_name: String,
        object_id: String,
        center_name: String,
        ref_frame: String,
        time_system: String,
        start_time: PyEpoch,
        stop_time: PyEpoch,
        interpolation: Option<String>,
        interpolation_degree: Option<u32>,
        trajectory: Option<PyRef<PyOrbitalTrajectory>>,
    ) -> PyResult<usize> {
        let rf = CCSDSRefFrame::parse(&ref_frame);
        let ts = CCSDSTimeSystem::parse(&time_system).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid time_system: {}", e))
        })?;
        let mut seg = OEMSegment {
            metadata: OEMMetadata {
                object_name,
                object_id,
                center_name,
                ref_frame: rf,
                ref_frame_epoch: None,
                time_system: ts,
                start_time: start_time.obj,
                useable_start_time: None,
                useable_stop_time: None,
                stop_time: stop_time.obj,
                interpolation,
                interpolation_degree,
                comments: Vec::new(),
            },
            comments: Vec::new(),
            states: Vec::new(),
            covariances: Vec::new(),
        };
        if let Some(traj) = trajectory {
            push_trajectory_states(&mut seg, &traj.trajectory).map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Failed to convert trajectory states: {}", e
                ))
            })?;
        }
        self.inner.segments.push(seg);
        Ok(self.inner.segments.len() - 1)
    }

    /// Remove a segment by index.
    ///
    /// Args:
    ///     idx (int): Index of the segment to remove
    fn remove_segment(&mut self, idx: usize) -> PyResult<()> {
        if idx >= self.inner.segments.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "segment index {} out of range (have {})",
                idx,
                self.inner.segments.len()
            )));
        }
        self.inner.segments.remove(idx);
        Ok(())
    }

    // --- internal delegate methods (prefixed with _) ---

    fn _num_segments(&self) -> usize {
        self.inner.segments.len()
    }

    fn _num_states(&self, seg_idx: usize) -> PyResult<usize> {
        let seg = self.get_seg(seg_idx)?;
        Ok(seg.states.len())
    }

    // -- segment metadata getters/setters --

    fn _seg_get_object_name(&self, idx: usize) -> PyResult<String> {
        Ok(self.get_seg(idx)?.metadata.object_name.clone())
    }

    fn _seg_set_object_name(&mut self, idx: usize, val: String) -> PyResult<()> {
        self.get_seg_mut(idx)?.metadata.object_name = val;
        Ok(())
    }

    fn _seg_get_object_id(&self, idx: usize) -> PyResult<String> {
        Ok(self.get_seg(idx)?.metadata.object_id.clone())
    }

    fn _seg_set_object_id(&mut self, idx: usize, val: String) -> PyResult<()> {
        self.get_seg_mut(idx)?.metadata.object_id = val;
        Ok(())
    }

    fn _seg_get_center_name(&self, idx: usize) -> PyResult<String> {
        Ok(self.get_seg(idx)?.metadata.center_name.clone())
    }

    fn _seg_set_center_name(&mut self, idx: usize, val: String) -> PyResult<()> {
        self.get_seg_mut(idx)?.metadata.center_name = val;
        Ok(())
    }

    fn _seg_get_ref_frame(&self, idx: usize) -> PyResult<String> {
        Ok(format!("{}", self.get_seg(idx)?.metadata.ref_frame))
    }

    fn _seg_set_ref_frame(&mut self, idx: usize, val: String) -> PyResult<()> {
        self.get_seg_mut(idx)?.metadata.ref_frame = CCSDSRefFrame::parse(&val);
        Ok(())
    }

    fn _seg_get_time_system(&self, idx: usize) -> PyResult<String> {
        Ok(format!("{}", self.get_seg(idx)?.metadata.time_system))
    }

    fn _seg_set_time_system(&mut self, idx: usize, val: String) -> PyResult<()> {
        let ts = CCSDSTimeSystem::parse(&val).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid time_system: {}", e))
        })?;
        self.get_seg_mut(idx)?.metadata.time_system = ts;
        Ok(())
    }

    fn _seg_get_start_time(&self, idx: usize) -> PyResult<PyEpoch> {
        Ok(PyEpoch {
            obj: self.get_seg(idx)?.metadata.start_time,
        })
    }

    fn _seg_set_start_time(&mut self, idx: usize, val: PyEpoch) -> PyResult<()> {
        self.get_seg_mut(idx)?.metadata.start_time = val.obj;
        Ok(())
    }

    fn _seg_get_stop_time(&self, idx: usize) -> PyResult<PyEpoch> {
        Ok(PyEpoch {
            obj: self.get_seg(idx)?.metadata.stop_time,
        })
    }

    fn _seg_set_stop_time(&mut self, idx: usize, val: PyEpoch) -> PyResult<()> {
        self.get_seg_mut(idx)?.metadata.stop_time = val.obj;
        Ok(())
    }

    fn _seg_get_interpolation(&self, idx: usize) -> PyResult<Option<String>> {
        Ok(self.get_seg(idx)?.metadata.interpolation.clone())
    }

    fn _seg_set_interpolation(&mut self, idx: usize, val: Option<String>) -> PyResult<()> {
        self.get_seg_mut(idx)?.metadata.interpolation = val;
        Ok(())
    }

    fn _seg_get_interpolation_degree(&self, idx: usize) -> PyResult<Option<u32>> {
        Ok(self.get_seg(idx)?.metadata.interpolation_degree)
    }

    fn _seg_set_interpolation_degree(&mut self, idx: usize, val: Option<u32>) -> PyResult<()> {
        self.get_seg_mut(idx)?.metadata.interpolation_degree = val;
        Ok(())
    }

    fn _seg_get_comments(&self, idx: usize) -> PyResult<Vec<String>> {
        Ok(self.get_seg(idx)?.comments.clone())
    }

    fn _seg_set_comments(&mut self, idx: usize, val: Vec<String>) -> PyResult<()> {
        self.get_seg_mut(idx)?.comments = val;
        Ok(())
    }

    fn _seg_get_num_covariances(&self, idx: usize) -> PyResult<usize> {
        Ok(self.get_seg(idx)?.covariances.len())
    }

    // -- state vector getters/setters --

    fn _sv_get_epoch(&self, seg_idx: usize, sv_idx: usize) -> PyResult<PyEpoch> {
        Ok(PyEpoch {
            obj: self.get_sv(seg_idx, sv_idx)?.epoch,
        })
    }

    fn _sv_set_epoch(&mut self, seg_idx: usize, sv_idx: usize, val: PyEpoch) -> PyResult<()> {
        self.get_sv_mut(seg_idx, sv_idx)?.epoch = val.obj;
        Ok(())
    }

    fn _sv_get_position(&self, seg_idx: usize, sv_idx: usize) -> PyResult<Vec<f64>> {
        Ok(self.get_sv(seg_idx, sv_idx)?.position.to_vec())
    }

    fn _sv_set_position(
        &mut self,
        seg_idx: usize,
        sv_idx: usize,
        val: Vec<f64>,
    ) -> PyResult<()> {
        let arr = vec_to_array3(val, "position")?;
        self.get_sv_mut(seg_idx, sv_idx)?.position = arr;
        Ok(())
    }

    fn _sv_get_velocity(&self, seg_idx: usize, sv_idx: usize) -> PyResult<Vec<f64>> {
        Ok(self.get_sv(seg_idx, sv_idx)?.velocity.to_vec())
    }

    fn _sv_set_velocity(
        &mut self,
        seg_idx: usize,
        sv_idx: usize,
        val: Vec<f64>,
    ) -> PyResult<()> {
        let arr = vec_to_array3(val, "velocity")?;
        self.get_sv_mut(seg_idx, sv_idx)?.velocity = arr;
        Ok(())
    }

    fn _sv_get_acceleration(
        &self,
        seg_idx: usize,
        sv_idx: usize,
    ) -> PyResult<Option<Vec<f64>>> {
        Ok(self.get_sv(seg_idx, sv_idx)?.acceleration.map(|a| a.to_vec()))
    }

    fn _sv_set_acceleration(
        &mut self,
        seg_idx: usize,
        sv_idx: usize,
        val: Option<Vec<f64>>,
    ) -> PyResult<()> {
        let arr = match val {
            Some(v) => Some(vec_to_array3(v, "acceleration")?),
            None => None,
        };
        self.get_sv_mut(seg_idx, sv_idx)?.acceleration = arr;
        Ok(())
    }

    // -- state add/remove --

    fn _add_state(
        &mut self,
        seg_idx: usize,
        epoch: PyEpoch,
        position: Vec<f64>,
        velocity: Vec<f64>,
        acceleration: Option<Vec<f64>>,
    ) -> PyResult<usize> {
        let pos = vec_to_array3(position, "position")?;
        let vel = vec_to_array3(velocity, "velocity")?;
        let acc = match acceleration {
            Some(v) => Some(vec_to_array3(v, "acceleration")?),
            None => None,
        };
        let seg = self.get_seg_mut(seg_idx)?;
        seg.states.push(OEMStateVector {
            epoch: epoch.obj,
            position: pos,
            velocity: vel,
            acceleration: acc,
        });
        Ok(seg.states.len() - 1)
    }

    fn _remove_state(&mut self, seg_idx: usize, sv_idx: usize) -> PyResult<()> {
        let seg = self.get_seg_mut(seg_idx)?;
        if sv_idx >= seg.states.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "state index {} out of range (have {})",
                sv_idx,
                seg.states.len()
            )));
        }
        seg.states.remove(sv_idx);
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "OEM(segments={}, originator='{}')",
            self.inner.segments.len(),
            self.inner.header.originator
        )
    }
}

impl PyOEM {
    fn get_seg(&self, idx: usize) -> PyResult<&OEMSegment> {
        self.inner.segments.get(idx).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "segment index {} out of range (have {})",
                idx,
                self.inner.segments.len()
            ))
        })
    }

    fn get_seg_mut(&mut self, idx: usize) -> PyResult<&mut OEMSegment> {
        let len = self.inner.segments.len();
        self.inner.segments.get_mut(idx).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "segment index {} out of range (have {})",
                idx, len
            ))
        })
    }

    fn get_sv(&self, seg_idx: usize, sv_idx: usize) -> PyResult<&OEMStateVector> {
        let seg = self.get_seg(seg_idx)?;
        seg.states.get(sv_idx).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "state index {} out of range (have {})",
                sv_idx,
                seg.states.len()
            ))
        })
    }

    fn get_sv_mut(&mut self, seg_idx: usize, sv_idx: usize) -> PyResult<&mut OEMStateVector> {
        let seg = self.get_seg_mut(seg_idx)?;
        let len = seg.states.len();
        seg.states.get_mut(sv_idx).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "state index {} out of range (have {})",
                sv_idx, len
            ))
        })
    }
}

// ─────────────────────────────────────────────
// PyOPMManeuver — typed proxy/owned for a maneuver
// ─────────────────────────────────────────────

enum ManeuverMode {
    Proxy { parent: Py<PyAny>, man_idx: usize },
    Owned { data: OPMManeuver },
}

/// A single OPM maneuver with typed property access.
///
/// Maneuvers can be created standalone or accessed via OPM maneuvers collection.
/// Proxy instances propagate mutations back to the parent OPM.
///
/// Args:
///     epoch_ignition (Epoch): Epoch of ignition
///     duration (float): Maneuver duration in seconds
///     ref_frame (str): Reference frame for delta-V
///     dv (list[float]): Delta-V [dv1, dv2, dv3] in m/s
///     delta_mass (float | None): Mass change in kg
///
/// Example:
///     ```python
///     from brahe import Epoch
///     from brahe.ccsds import OPMManeuver
///     m = OPMManeuver(
///         epoch_ignition=Epoch.from_datetime(2024, 1, 1, 0, 0, 0),
///         duration=300.0, ref_frame="RTN", dv=[1.0, 0.0, 0.0],
///     )
///     print(m.ref_frame)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OPMManeuver")]
pub struct PyOPMManeuver {
    mode: ManeuverMode,
}

impl PyOPMManeuver {
    fn to_rust_data(&self, py: Python) -> PyResult<OPMManeuver> {
        match &self.mode {
            ManeuverMode::Owned { data } => Ok(data.clone()),
            ManeuverMode::Proxy { parent, man_idx } => {
                let epoch_obj = parent.bind(py).call_method1("_man_get_epoch_ignition", (*man_idx,))?;
                let epoch: PyEpoch = extract_epoch(&epoch_obj)?;
                let duration: f64 = parent.bind(py).call_method1("_man_get_duration", (*man_idx,))?.extract()?;
                let delta_mass: Option<f64> = parent.bind(py).call_method1("_man_get_delta_mass", (*man_idx,))?.extract()?;
                let rf_str: String = parent.bind(py).call_method1("_man_get_ref_frame", (*man_idx,))?.extract()?;
                let dv: Vec<f64> = parent.bind(py).call_method1("_man_get_dv", (*man_idx,))?.extract()?;
                let comments: Vec<String> = parent.bind(py).call_method1("_man_get_comments", (*man_idx,))?.extract()?;
                Ok(OPMManeuver {
                    epoch_ignition: epoch.obj,
                    duration,
                    delta_mass,
                    ref_frame: CCSDSRefFrame::parse(&rf_str),
                    dv: vec_to_array3(dv, "dv")?,
                    comments,
                })
            }
        }
    }
}

#[pymethods]
impl PyOPMManeuver {
    #[new]
    #[pyo3(signature = (epoch_ignition, duration, ref_frame, dv, delta_mass=None))]
    fn new(
        epoch_ignition: PyEpoch,
        duration: f64,
        ref_frame: String,
        dv: &pyo3::Bound<'_, pyo3::types::PyAny>,
        delta_mass: Option<f64>,
    ) -> PyResult<Self> {
        let rf = CCSDSRefFrame::parse(&ref_frame);
        let dv_arr = pyany_to_array3(dv, "dv")?;
        Ok(Self {
            mode: ManeuverMode::Owned {
                data: OPMManeuver { epoch_ignition: epoch_ignition.obj, duration, delta_mass, ref_frame: rf, dv: dv_arr, comments: Vec::new() },
            },
        })
    }

    /// Epoch of ignition.
    ///
    /// Returns:
    ///     Epoch: Epoch of ignition
    #[getter]
    fn epoch_ignition(&self, py: Python) -> PyResult<PyEpoch> {
        match &self.mode {
            ManeuverMode::Owned { data } => Ok(PyEpoch { obj: data.epoch_ignition }),
            ManeuverMode::Proxy { parent, man_idx } => {
                let obj = parent.bind(py).call_method1("_man_get_epoch_ignition", (*man_idx,))?;
                extract_epoch(&obj)
            }
        }
    }

    /// Set epoch of ignition.
    ///
    /// Args:
    ///     val (Epoch): Epoch of ignition
    #[setter]
    fn set_epoch_ignition(&mut self, py: Python, val: PyEpoch) -> PyResult<()> {
        match &mut self.mode {
            ManeuverMode::Owned { data } => { data.epoch_ignition = val.obj; Ok(()) }
            ManeuverMode::Proxy { parent, man_idx } => {
                parent.bind(py).call_method1("_man_set_epoch_ignition", (*man_idx, val))?;
                Ok(())
            }
        }
    }

    /// Maneuver duration in seconds.
    ///
    /// Returns:
    ///     float: Duration in seconds
    #[getter]
    fn duration(&self, py: Python) -> PyResult<f64> {
        match &self.mode {
            ManeuverMode::Owned { data } => Ok(data.duration),
            ManeuverMode::Proxy { parent, man_idx } => {
                parent.bind(py).call_method1("_man_get_duration", (*man_idx,))?.extract()
            }
        }
    }

    /// Set maneuver duration.
    ///
    /// Args:
    ///     val (float): Duration in seconds
    #[setter]
    fn set_duration(&mut self, py: Python, val: f64) -> PyResult<()> {
        match &mut self.mode {
            ManeuverMode::Owned { data } => { data.duration = val; Ok(()) }
            ManeuverMode::Proxy { parent, man_idx } => {
                parent.bind(py).call_method1("_man_set_duration", (*man_idx, val))?;
                Ok(())
            }
        }
    }

    /// Mass change in kg (negative for mass decrease), or None.
    ///
    /// Returns:
    ///     float: Delta mass in kg, or None
    #[getter]
    fn delta_mass(&self, py: Python) -> PyResult<Option<f64>> {
        match &self.mode {
            ManeuverMode::Owned { data } => Ok(data.delta_mass),
            ManeuverMode::Proxy { parent, man_idx } => {
                parent.bind(py).call_method1("_man_get_delta_mass", (*man_idx,))?.extract()
            }
        }
    }

    /// Set delta mass.
    ///
    /// Args:
    ///     val (float | None): Delta mass in kg
    #[setter]
    fn set_delta_mass(&mut self, py: Python, val: Option<f64>) -> PyResult<()> {
        match &mut self.mode {
            ManeuverMode::Owned { data } => { data.delta_mass = val; Ok(()) }
            ManeuverMode::Proxy { parent, man_idx } => {
                parent.bind(py).call_method1("_man_set_delta_mass", (*man_idx, val))?;
                Ok(())
            }
        }
    }

    /// Reference frame for the delta-V.
    ///
    /// Returns:
    ///     str: Reference frame name
    #[getter]
    fn ref_frame(&self, py: Python) -> PyResult<String> {
        match &self.mode {
            ManeuverMode::Owned { data } => Ok(format!("{}", data.ref_frame)),
            ManeuverMode::Proxy { parent, man_idx } => {
                parent.bind(py).call_method1("_man_get_ref_frame", (*man_idx,))?.extract()
            }
        }
    }

    /// Set reference frame for the delta-V.
    ///
    /// Args:
    ///     val (str): Reference frame name
    #[setter]
    fn set_ref_frame(&mut self, py: Python, val: String) -> PyResult<()> {
        match &mut self.mode {
            ManeuverMode::Owned { data } => { data.ref_frame = CCSDSRefFrame::parse(&val); Ok(()) }
            ManeuverMode::Proxy { parent, man_idx } => {
                parent.bind(py).call_method1("_man_set_ref_frame", (*man_idx, val))?;
                Ok(())
            }
        }
    }

    /// Delta-V vector [dv1, dv2, dv3] in m/s.
    ///
    /// Returns:
    ///     list[float]: Delta-V [dv1, dv2, dv3] in m/s
    #[getter]
    fn dv(&self, py: Python) -> PyResult<Vec<f64>> {
        match &self.mode {
            ManeuverMode::Owned { data } => Ok(data.dv.to_vec()),
            ManeuverMode::Proxy { parent, man_idx } => {
                parent.bind(py).call_method1("_man_get_dv", (*man_idx,))?.extract()
            }
        }
    }

    /// Set delta-V vector.
    ///
    /// Args:
    ///     val (list[float]): Delta-V [dv1, dv2, dv3] in m/s
    #[setter]
    fn set_dv(&mut self, py: Python, val: &pyo3::Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
        let arr = pyany_to_array3(val, "dv")?;
        match &mut self.mode {
            ManeuverMode::Owned { data } => { data.dv = arr; Ok(()) }
            ManeuverMode::Proxy { parent, man_idx } => {
                parent.bind(py).call_method1("_man_set_dv", (*man_idx, arr.to_vec()))?;
                Ok(())
            }
        }
    }

    /// Comments for this maneuver.
    ///
    /// Returns:
    ///     list[str]: Comments
    #[getter]
    fn comments(&self, py: Python) -> PyResult<Vec<String>> {
        match &self.mode {
            ManeuverMode::Owned { data } => Ok(data.comments.clone()),
            ManeuverMode::Proxy { parent, man_idx } => {
                parent.bind(py).call_method1("_man_get_comments", (*man_idx,))?.extract()
            }
        }
    }

    /// Set comments.
    ///
    /// Args:
    ///     val (list[str]): Comments
    #[setter]
    fn set_comments(&mut self, py: Python, val: Vec<String>) -> PyResult<()> {
        match &mut self.mode {
            ManeuverMode::Owned { data } => { data.comments = val; Ok(()) }
            ManeuverMode::Proxy { parent, man_idx } => {
                parent.bind(py).call_method1("_man_set_comments", (*man_idx, val))?;
                Ok(())
            }
        }
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let frame: String = self.ref_frame(py)?;
        match &self.mode {
            ManeuverMode::Owned { .. } => {
                Ok(format!("OPMManeuver(ref_frame='{}')", frame))
            }
            ManeuverMode::Proxy { man_idx, .. } => {
                Ok(format!(
                    "OPMManeuver(ref_frame='{}', idx={})",
                    frame, man_idx
                ))
            }
        }
    }
}

// ─────────────────────────────────────────────
// OPM maneuver collection proxy
// ─────────────────────────────────────────────

/// Collection wrapper for OPM maneuvers, supporting len/indexing/iteration.
///
/// Example:
///     ```python
///     maneuvers = opm.maneuvers
///     print(len(maneuvers))
///     m = maneuvers[0]
///     print(m.ref_frame)
///     for m in maneuvers:
///         print(m.dv)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OPMManeuvers")]
pub struct PyOPMManeuvers {
    parent: Py<PyAny>,
}

#[pymethods]
impl PyOPMManeuvers {
    fn __len__(&self, py: Python) -> PyResult<usize> {
        self.parent
            .bind(py)
            .call_method0("_num_maneuvers")?
            .extract()
    }

    fn __getitem__(&self, py: Python, index: isize) -> PyResult<PyOPMManeuver> {
        let len: usize = self.__len__(py)?;
        let actual = normalize_index(index, len, "maneuver")?;
        Ok(PyOPMManeuver {
            mode: ManeuverMode::Proxy {
                parent: self.parent.clone_ref(py),
                man_idx: actual,
            },
        })
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyResult<Py<PyOPMManeuverIterator>> {
        let py = slf.py();
        Py::new(
            py,
            PyOPMManeuverIterator {
                parent: slf.parent.clone_ref(py),
                index: 0,
            },
        )
    }

    /// Append a maneuver to the OPM.
    ///
    /// Args:
    ///     man (OPMManeuver): Maneuver to append
    fn append(&self, py: Python, man: &Bound<'_, PyOPMManeuver>) -> PyResult<()> {
        let data = man.borrow().to_rust_data(py)?;
        let parent_opm: &Bound<PyOPM> = self.parent.bind(py).cast()?;
        parent_opm.borrow_mut().inner.maneuvers.push(data);
        Ok(())
    }

    /// Extend with maneuvers from an iterable.
    ///
    /// Args:
    ///     iterable: Iterable of OPMManeuver objects
    fn extend(&self, py: Python, iterable: &Bound<'_, PyAny>) -> PyResult<()> {
        let parent_opm: &Bound<PyOPM> = self.parent.bind(py).cast()?;
        let iterator = iterable.try_iter()?;
        for item in iterator {
            let bound: Bound<'_, PyAny> = item?;
            let man_ref: &Bound<PyOPMManeuver> = bound.cast()?;
            let data = man_ref.borrow().to_rust_data(py)?;
            parent_opm.borrow_mut().inner.maneuvers.push(data);
        }
        Ok(())
    }

    fn __delitem__(&self, py: Python, index: isize) -> PyResult<()> {
        let len = self.__len__(py)?;
        let actual = normalize_index(index, len, "maneuver")?;
        let parent_opm: &Bound<PyOPM> = self.parent.bind(py).cast()?;
        parent_opm.borrow_mut().inner.maneuvers.remove(actual);
        Ok(())
    }

    fn __repr__(&self, py: Python) -> PyResult<String> {
        let len = self.__len__(py)?;
        Ok(format!("OPMManeuvers(len={})", len))
    }
}

/// Iterator over OPM maneuvers.
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OPMManeuverIterator")]
pub struct PyOPMManeuverIterator {
    parent: Py<PyAny>,
    index: usize,
}

#[pymethods]
impl PyOPMManeuverIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&mut self, py: Python) -> PyResult<Option<PyOPMManeuver>> {
        let len: usize = self
            .parent
            .bind(py)
            .call_method0("_num_maneuvers")?
            .extract()?;
        if self.index >= len {
            return Ok(None);
        }
        let m = PyOPMManeuver {
            mode: ManeuverMode::Proxy {
                parent: self.parent.clone_ref(py),
                man_idx: self.index,
            },
        };
        self.index += 1;
        Ok(Some(m))
    }
}

// ─────────────────────────────────────────────
// PyOMM — Orbit Mean-elements Message
// ─────────────────────────────────────────────

/// Python wrapper for CCSDS Orbit Mean-elements Message (OMM).
///
/// OMM is flat (no segments), so all fields are exposed as properties
/// with getters and setters.
///
/// Example:
///     ```python
///     from brahe.ccsds import OMM
///
///     omm = OMM.from_file("gp_data.omm")
///     print(omm.object_name)
///     omm.object_name = "ISS"
///     print(omm.eccentricity)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OMM")]
pub struct PyOMM {
    inner: RustOMM,
}

#[pymethods]
impl PyOMM {
    // --- constructors ---

    /// Parse an OMM from a string, auto-detecting the format.
    ///
    /// Args:
    ///     content (str): String content of the OMM message
    ///
    /// Returns:
    ///     OMM: Parsed OMM message
    #[staticmethod]
    #[allow(clippy::should_implement_trait)]
    fn from_str(content: &str) -> PyResult<Self> {
        let inner = RustOMM::from_str(content)?;
        Ok(PyOMM { inner })
    }

    /// Parse an OMM from a file, auto-detecting the format.
    ///
    /// Args:
    ///     path (str): Path to the OMM file
    ///
    /// Returns:
    ///     OMM: Parsed OMM message
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let inner = RustOMM::from_file(path)?;
        Ok(PyOMM { inner })
    }

    /// Create an OMM from a GPRecord.
    ///
    /// Validates that required orbital element fields are present (epoch,
    /// eccentricity, inclination, ra_of_asc_node, arg_of_pericenter,
    /// mean_anomaly) and builds an OMM with defaults for missing metadata.
    ///
    /// Args:
    ///     gp (GPRecord): GP record to convert.
    ///
    /// Returns:
    ///     OMM: CCSDS OMM message constructed from the GP record.
    ///
    /// Raises:
    ///     BraheError: If required orbital element fields are missing.
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     from brahe.ccsds import OMM
    ///
    ///     record = bh.GPRecord.from_json('{"OBJECT_NAME": "ISS", "EPOCH": "2024-01-15T12:00:00.000", "ECCENTRICITY": 0.0001, "INCLINATION": 51.64, "RA_OF_ASC_NODE": 200.0, "ARG_OF_PERICENTER": 100.0, "MEAN_ANOMALY": 260.0}')
    ///     omm = OMM.from_gp_record(record)
    ///     print(omm.object_name)
    ///     ```
    #[staticmethod]
    fn from_gp_record(gp: &PyGPRecord) -> PyResult<Self> {
        let omm = RustOMM::from_gp_record(&gp.inner)
            .map_err(|e| BraheError::new_err(e.to_string()))?;
        Ok(PyOMM { inner: omm })
    }

    /// Convert this OMM to a GPRecord.
    ///
    /// Maps all OMM fields back to the GPRecord format. This conversion
    /// is infallible since all GPRecord fields are optional.
    ///
    /// Returns:
    ///     GPRecord: GP record with fields populated from this OMM.
    ///
    /// Example:
    ///     ```python
    ///     from brahe.ccsds import OMM
    ///
    ///     omm = OMM.from_file("test_assets/ccsds/omm/OMMExample1.txt")
    ///     gp = omm.to_gp_record()
    ///     print(gp.object_name)
    ///     ```
    fn to_gp_record(&self) -> PyGPRecord {
        PyGPRecord {
            inner: self.inner.to_gp_record(),
        }
    }

    // --- serialization ---

    /// Convert the OMM to a Python dictionary.
    ///
    /// Returns:
    ///     dict: Dictionary representation of the OMM
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new(py);

        // Header
        let header = pyo3::types::PyDict::new(py);
        header.set_item("format_version", self.inner.header.format_version)?;
        header.set_item(
            "creation_date",
            crate::ccsds::common::format_ccsds_datetime(&self.inner.header.creation_date),
        )?;
        header.set_item("originator", &self.inner.header.originator)?;
        dict.set_item("header", header)?;

        // Metadata
        let meta = pyo3::types::PyDict::new(py);
        meta.set_item("object_name", &self.inner.metadata.object_name)?;
        meta.set_item("object_id", &self.inner.metadata.object_id)?;
        meta.set_item("center_name", &self.inner.metadata.center_name)?;
        meta.set_item("ref_frame", format!("{}", self.inner.metadata.ref_frame))?;
        meta.set_item(
            "time_system",
            format!("{}", self.inner.metadata.time_system),
        )?;
        meta.set_item(
            "mean_element_theory",
            &self.inner.metadata.mean_element_theory,
        )?;
        dict.set_item("metadata", meta)?;

        // Mean elements
        let me = pyo3::types::PyDict::new(py);
        me.set_item(
            "epoch",
            crate::ccsds::common::format_ccsds_datetime(&self.inner.mean_elements.epoch),
        )?;
        me.set_item("mean_motion", self.inner.mean_elements.mean_motion)?;
        me.set_item(
            "semi_major_axis",
            self.inner.mean_elements.semi_major_axis,
        )?;
        me.set_item("eccentricity", self.inner.mean_elements.eccentricity)?;
        me.set_item("inclination", self.inner.mean_elements.inclination)?;
        me.set_item("ra_of_asc_node", self.inner.mean_elements.ra_of_asc_node)?;
        me.set_item(
            "arg_of_pericenter",
            self.inner.mean_elements.arg_of_pericenter,
        )?;
        me.set_item("mean_anomaly", self.inner.mean_elements.mean_anomaly)?;
        me.set_item("gm", self.inner.mean_elements.gm)?;
        dict.set_item("mean_elements", me)?;

        // TLE parameters
        if let Some(ref tle) = self.inner.tle_parameters {
            let tle_dict = pyo3::types::PyDict::new(py);
            tle_dict.set_item("ephemeris_type", tle.ephemeris_type)?;
            tle_dict.set_item(
                "classification_type",
                tle.classification_type.map(|c| c.to_string()),
            )?;
            tle_dict.set_item("norad_cat_id", tle.norad_cat_id)?;
            tle_dict.set_item("element_set_no", tle.element_set_no)?;
            tle_dict.set_item("rev_at_epoch", tle.rev_at_epoch)?;
            tle_dict.set_item("bstar", tle.bstar)?;
            tle_dict.set_item("bterm", tle.bterm)?;
            tle_dict.set_item("mean_motion_dot", tle.mean_motion_dot)?;
            tle_dict.set_item("mean_motion_ddot", tle.mean_motion_ddot)?;
            tle_dict.set_item("agom", tle.agom)?;
            dict.set_item("tle_parameters", tle_dict)?;
        }

        // Spacecraft parameters
        if let Some(ref sc) = self.inner.spacecraft_parameters {
            let sc_dict = pyo3::types::PyDict::new(py);
            sc_dict.set_item("mass", sc.mass)?;
            sc_dict.set_item("solar_rad_area", sc.solar_rad_area)?;
            sc_dict.set_item("solar_rad_coeff", sc.solar_rad_coeff)?;
            sc_dict.set_item("drag_area", sc.drag_area)?;
            sc_dict.set_item("drag_coeff", sc.drag_coeff)?;
            dict.set_item("spacecraft_parameters", sc_dict)?;
        }

        // User-defined
        if let Some(ref ud) = self.inner.user_defined {
            let ud_dict = pyo3::types::PyDict::new(py);
            for (k, v) in &ud.parameters {
                ud_dict.set_item(k, v)?;
            }
            dict.set_item("user_defined", ud_dict)?;
        }

        Ok(dict)
    }

    // --- serialization ---

    /// Write the OMM to a string in the specified format.
    ///
    /// Args:
    ///     format (str): Output format - "KVN", "XML", or "JSON"
    ///
    /// Returns:
    ///     str: Serialized OMM string
    fn to_string(&self, format: &str) -> PyResult<String> {
        let fmt = parse_format(format)?;
        let result = self.inner.to_string(fmt)?;
        Ok(result)
    }

    /// Write the OMM to JSON with explicit key case control.
    ///
    /// Args:
    ///     uppercase_keys (bool): If True, use uppercase CCSDS keywords. Default: False.
    ///
    /// Returns:
    ///     str: Serialized JSON string
    #[pyo3(signature = (uppercase_keys=false))]
    fn to_json_string(&self, uppercase_keys: bool) -> PyResult<String> {
        let key_case = if uppercase_keys {
            crate::ccsds::common::CCSDSJsonKeyCase::Upper
        } else {
            crate::ccsds::common::CCSDSJsonKeyCase::Lower
        };
        let result = self.inner.to_json_string(key_case)?;
        Ok(result)
    }

    /// Write the OMM to a file in the specified format.
    ///
    /// Args:
    ///     path (str): Output file path
    ///     format (str): Output format - "KVN", "XML", or "JSON"
    fn to_file(&self, path: &str, format: &str) -> PyResult<()> {
        let fmt = parse_format(format)?;
        self.inner.to_file(path, fmt)?;
        Ok(())
    }

    // --- header properties ---

    /// CCSDS format version.
    ///
    /// Returns:
    ///     float: Format version
    #[getter]
    fn format_version(&self) -> f64 {
        self.inner.header.format_version
    }

    /// Set CCSDS format version.
    ///
    /// Args:
    ///     val (float): Format version
    #[setter]
    fn set_format_version(&mut self, val: f64) {
        self.inner.header.format_version = val;
    }

    /// Originator of the message.
    ///
    /// Returns:
    ///     str: Originator string
    #[getter]
    fn originator(&self) -> String {
        self.inner.header.originator.clone()
    }

    /// Set originator.
    ///
    /// Args:
    ///     val (str): Originator string
    #[setter]
    fn set_originator(&mut self, val: String) {
        self.inner.header.originator = val;
    }

    /// Creation date of the message.
    ///
    /// Returns:
    ///     Epoch: Creation date
    #[getter]
    fn creation_date(&self) -> PyEpoch {
        PyEpoch {
            obj: self.inner.header.creation_date,
        }
    }

    /// Set creation date.
    ///
    /// Args:
    ///     val (Epoch): Creation date
    #[setter]
    fn set_creation_date(&mut self, val: PyEpoch) {
        self.inner.header.creation_date = val.obj;
    }

    // --- metadata properties ---

    /// Object name.
    ///
    /// Returns:
    ///     str: Object name
    #[getter]
    fn object_name(&self) -> String {
        self.inner.metadata.object_name.clone()
    }

    /// Set object name.
    ///
    /// Args:
    ///     val (str): Object name
    #[setter]
    fn set_object_name(&mut self, val: String) {
        self.inner.metadata.object_name = val;
    }

    /// Object ID (international designator).
    ///
    /// Returns:
    ///     str: Object ID
    #[getter]
    fn object_id(&self) -> String {
        self.inner.metadata.object_id.clone()
    }

    /// Set object ID.
    ///
    /// Args:
    ///     val (str): Object ID
    #[setter]
    fn set_object_id(&mut self, val: String) {
        self.inner.metadata.object_id = val;
    }

    /// Center body name.
    ///
    /// Returns:
    ///     str: Center name
    #[getter]
    fn center_name(&self) -> String {
        self.inner.metadata.center_name.clone()
    }

    /// Set center body name.
    ///
    /// Args:
    ///     val (str): Center name
    #[setter]
    fn set_center_name(&mut self, val: String) {
        self.inner.metadata.center_name = val;
    }

    /// Reference frame name.
    ///
    /// Returns:
    ///     str: Reference frame
    #[getter]
    fn ref_frame(&self) -> String {
        format!("{}", self.inner.metadata.ref_frame)
    }

    /// Set reference frame.
    ///
    /// Args:
    ///     val (str): Reference frame name
    #[setter]
    fn set_ref_frame(&mut self, val: String) -> PyResult<()> {
        self.inner.metadata.ref_frame = CCSDSRefFrame::parse(&val);
        Ok(())
    }

    /// Time system name.
    ///
    /// Returns:
    ///     str: Time system
    #[getter]
    fn time_system(&self) -> String {
        format!("{}", self.inner.metadata.time_system)
    }

    /// Set time system.
    ///
    /// Args:
    ///     val (str): Time system name
    #[setter]
    fn set_time_system(&mut self, val: String) -> PyResult<()> {
        self.inner.metadata.time_system = CCSDSTimeSystem::parse(&val).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid time_system: {}", e))
        })?;
        Ok(())
    }

    /// Mean element theory (e.g., "SGP/SGP4").
    ///
    /// Returns:
    ///     str: Mean element theory
    #[getter]
    fn mean_element_theory(&self) -> String {
        self.inner.metadata.mean_element_theory.clone()
    }

    /// Set mean element theory.
    ///
    /// Args:
    ///     val (str): Mean element theory
    #[setter]
    fn set_mean_element_theory(&mut self, val: String) {
        self.inner.metadata.mean_element_theory = val;
    }

    // --- mean element properties ---

    /// Epoch of the mean elements.
    ///
    /// Returns:
    ///     Epoch: Epoch of mean elements
    #[getter]
    fn epoch(&self) -> PyEpoch {
        PyEpoch {
            obj: self.inner.mean_elements.epoch,
        }
    }

    /// Set epoch of the mean elements.
    ///
    /// Args:
    ///     val (Epoch): Epoch
    #[setter]
    fn set_epoch(&mut self, val: PyEpoch) {
        self.inner.mean_elements.epoch = val.obj;
    }

    /// Mean motion in rev/day, or None if not set.
    ///
    /// Returns:
    ///     float: Mean motion, or None
    #[getter]
    fn mean_motion(&self) -> Option<f64> {
        self.inner.mean_elements.mean_motion
    }

    /// Set mean motion.
    ///
    /// Args:
    ///     val (float | None): Mean motion in rev/day
    #[setter]
    fn set_mean_motion(&mut self, val: Option<f64>) {
        self.inner.mean_elements.mean_motion = val;
    }

    /// Eccentricity.
    ///
    /// Returns:
    ///     float: Eccentricity
    #[getter]
    fn eccentricity(&self) -> f64 {
        self.inner.mean_elements.eccentricity
    }

    /// Set eccentricity.
    ///
    /// Args:
    ///     val (float): Eccentricity
    #[setter]
    fn set_eccentricity(&mut self, val: f64) {
        self.inner.mean_elements.eccentricity = val;
    }

    /// Inclination in degrees.
    ///
    /// Returns:
    ///     float: Inclination (degrees)
    #[getter]
    fn inclination(&self) -> f64 {
        self.inner.mean_elements.inclination
    }

    /// Set inclination.
    ///
    /// Args:
    ///     val (float): Inclination in degrees
    #[setter]
    fn set_inclination(&mut self, val: f64) {
        self.inner.mean_elements.inclination = val;
    }

    /// Right ascension of ascending node in degrees.
    ///
    /// Returns:
    ///     float: RAAN (degrees)
    #[getter]
    fn ra_of_asc_node(&self) -> f64 {
        self.inner.mean_elements.ra_of_asc_node
    }

    /// Set RAAN.
    ///
    /// Args:
    ///     val (float): RAAN in degrees
    #[setter]
    fn set_ra_of_asc_node(&mut self, val: f64) {
        self.inner.mean_elements.ra_of_asc_node = val;
    }

    /// Argument of pericenter in degrees.
    ///
    /// Returns:
    ///     float: Argument of pericenter (degrees)
    #[getter]
    fn arg_of_pericenter(&self) -> f64 {
        self.inner.mean_elements.arg_of_pericenter
    }

    /// Set argument of pericenter.
    ///
    /// Args:
    ///     val (float): Argument of pericenter in degrees
    #[setter]
    fn set_arg_of_pericenter(&mut self, val: f64) {
        self.inner.mean_elements.arg_of_pericenter = val;
    }

    /// Mean anomaly in degrees.
    ///
    /// Returns:
    ///     float: Mean anomaly (degrees)
    #[getter]
    fn mean_anomaly(&self) -> f64 {
        self.inner.mean_elements.mean_anomaly
    }

    /// Set mean anomaly.
    ///
    /// Args:
    ///     val (float): Mean anomaly in degrees
    #[setter]
    fn set_mean_anomaly(&mut self, val: f64) {
        self.inner.mean_elements.mean_anomaly = val;
    }

    /// Gravitational parameter in m^3/s^2, or None.
    ///
    /// Returns:
    ///     float: GM, or None
    #[getter]
    fn gm(&self) -> Option<f64> {
        self.inner.mean_elements.gm
    }

    /// Set GM.
    ///
    /// Args:
    ///     val (float | None): GM in m^3/s^2
    #[setter]
    fn set_gm(&mut self, val: Option<f64>) {
        self.inner.mean_elements.gm = val;
    }

    // --- TLE parameter properties ---

    /// NORAD catalog ID, or None.
    ///
    /// Returns:
    ///     int: NORAD catalog ID, or None
    #[getter]
    fn norad_cat_id(&self) -> Option<u32> {
        self.inner
            .tle_parameters
            .as_ref()
            .and_then(|t| t.norad_cat_id)
    }

    /// Set NORAD catalog ID.
    ///
    /// Args:
    ///     val (int | None): NORAD catalog ID
    #[setter]
    fn set_norad_cat_id(&mut self, val: Option<u32>) {
        if let Some(ref mut tle) = self.inner.tle_parameters {
            tle.norad_cat_id = val;
        }
    }

    /// BSTAR drag term, or None.
    ///
    /// Returns:
    ///     float: BSTAR, or None
    #[getter]
    fn bstar(&self) -> Option<f64> {
        self.inner.tle_parameters.as_ref().and_then(|t| t.bstar)
    }

    /// Set BSTAR drag term.
    ///
    /// Args:
    ///     val (float | None): BSTAR
    #[setter]
    fn set_bstar(&mut self, val: Option<f64>) {
        if let Some(ref mut tle) = self.inner.tle_parameters {
            tle.bstar = val;
        }
    }

    /// First derivative of mean motion (rev/day^2), or None.
    ///
    /// Returns:
    ///     float: Mean motion dot, or None
    #[getter]
    fn mean_motion_dot(&self) -> Option<f64> {
        self.inner
            .tle_parameters
            .as_ref()
            .and_then(|t| t.mean_motion_dot)
    }

    /// Set mean motion dot.
    ///
    /// Args:
    ///     val (float | None): Mean motion dot in rev/day^2
    #[setter]
    fn set_mean_motion_dot(&mut self, val: Option<f64>) {
        if let Some(ref mut tle) = self.inner.tle_parameters {
            tle.mean_motion_dot = val;
        }
    }

    /// Second derivative of mean motion (rev/day^3), or None.
    ///
    /// Returns:
    ///     float: Mean motion double-dot, or None
    #[getter]
    fn mean_motion_ddot(&self) -> Option<f64> {
        self.inner
            .tle_parameters
            .as_ref()
            .and_then(|t| t.mean_motion_ddot)
    }

    /// Set mean motion double-dot.
    ///
    /// Args:
    ///     val (float | None): Mean motion double-dot in rev/day^3
    #[setter]
    fn set_mean_motion_ddot(&mut self, val: Option<f64>) {
        if let Some(ref mut tle) = self.inner.tle_parameters {
            tle.mean_motion_ddot = val;
        }
    }

    /// Element set number, or None.
    ///
    /// Returns:
    ///     int: Element set number, or None
    #[getter]
    fn element_set_no(&self) -> Option<u32> {
        self.inner
            .tle_parameters
            .as_ref()
            .and_then(|t| t.element_set_no)
    }

    /// Set element set number.
    ///
    /// Args:
    ///     val (int | None): Element set number
    #[setter]
    fn set_element_set_no(&mut self, val: Option<u32>) {
        if let Some(ref mut tle) = self.inner.tle_parameters {
            tle.element_set_no = val;
        }
    }

    /// Revolution number at epoch, or None.
    ///
    /// Returns:
    ///     int: Rev at epoch, or None
    #[getter]
    fn rev_at_epoch(&self) -> Option<u32> {
        self.inner
            .tle_parameters
            .as_ref()
            .and_then(|t| t.rev_at_epoch)
    }

    /// Set revolution number at epoch.
    ///
    /// Args:
    ///     val (int | None): Rev at epoch
    #[setter]
    fn set_rev_at_epoch(&mut self, val: Option<u32>) {
        if let Some(ref mut tle) = self.inner.tle_parameters {
            tle.rev_at_epoch = val;
        }
    }

    /// Classification type character, or None.
    ///
    /// Returns:
    ///     str: Classification type, or None
    #[getter]
    fn classification_type(&self) -> Option<String> {
        self.inner
            .tle_parameters
            .as_ref()
            .and_then(|t| t.classification_type.map(|c| c.to_string()))
    }

    /// Set classification type.
    ///
    /// Args:
    ///     val (str | None): Classification type character
    #[setter]
    fn set_classification_type(&mut self, val: Option<String>) -> PyResult<()> {
        if let Some(ref mut tle) = self.inner.tle_parameters {
            tle.classification_type = match val {
                Some(s) => {
                    let c = s.chars().next().ok_or_else(|| {
                        pyo3::exceptions::PyValueError::new_err(
                            "classification_type must be a single character",
                        )
                    })?;
                    Some(c)
                }
                None => None,
            };
        }
        Ok(())
    }

    /// Ephemeris type, or None.
    ///
    /// Returns:
    ///     int: Ephemeris type, or None
    #[getter]
    fn ephemeris_type(&self) -> Option<u32> {
        self.inner
            .tle_parameters
            .as_ref()
            .and_then(|t| t.ephemeris_type)
    }

    /// Set ephemeris type.
    ///
    /// Args:
    ///     val (int | None): Ephemeris type
    #[setter]
    fn set_ephemeris_type(&mut self, val: Option<u32>) {
        if let Some(ref mut tle) = self.inner.tle_parameters {
            tle.ephemeris_type = val;
        }
    }

    fn __repr__(&self) -> String {
        format!(
            "OMM(object='{}', id='{}')",
            self.inner.metadata.object_name, self.inner.metadata.object_id
        )
    }
}

// ─────────────────────────────────────────────
// PyOPM — Orbit Parameter Message
// ─────────────────────────────────────────────

/// Python wrapper for CCSDS Orbit Parameter Message (OPM).
///
/// Scalar metadata is exposed as properties with getters and setters.
/// Maneuvers are accessed via the ``maneuvers`` property which supports
/// indexing, iteration, and mutation.
///
/// Example:
///     ```python
///     from brahe.ccsds import OPM
///
///     opm = OPM.from_file("state.opm")
///     print(opm.position)
///     opm.object_name = "NEW_SAT"
///     print(len(opm.maneuvers))
///     m = opm.maneuvers[0]
///     print(m.ref_frame)
///     m.ref_frame = "RTN"
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "OPM")]
pub struct PyOPM {
    inner: RustOPM,
}

#[pymethods]
impl PyOPM {
    // --- constructors ---

    /// Parse an OPM from a string, auto-detecting the format.
    ///
    /// Args:
    ///     content (str): String content of the OPM message
    ///
    /// Returns:
    ///     OPM: Parsed OPM message
    #[staticmethod]
    #[allow(clippy::should_implement_trait)]
    fn from_str(content: &str) -> PyResult<Self> {
        let inner = RustOPM::from_str(content)?;
        Ok(PyOPM { inner })
    }

    /// Parse an OPM from a file, auto-detecting the format.
    ///
    /// Args:
    ///     path (str): Path to the OPM file
    ///
    /// Returns:
    ///     OPM: Parsed OPM message
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let inner = RustOPM::from_file(path)?;
        Ok(PyOPM { inner })
    }

    // --- serialization ---

    /// Write the OPM to a string in the specified format.
    ///
    /// Args:
    ///     format (str): Output format - "KVN", "XML", or "JSON"
    ///
    /// Returns:
    ///     str: Serialized OPM string
    fn to_string(&self, format: &str) -> PyResult<String> {
        let fmt = parse_format(format)?;
        let result = self.inner.to_string(fmt)?;
        Ok(result)
    }

    /// Write the OPM to JSON with explicit key case control.
    ///
    /// Args:
    ///     uppercase_keys (bool): If True, use uppercase CCSDS keywords. Default: False.
    ///
    /// Returns:
    ///     str: Serialized JSON string
    #[pyo3(signature = (uppercase_keys=false))]
    fn to_json_string(&self, uppercase_keys: bool) -> PyResult<String> {
        let key_case = if uppercase_keys {
            crate::ccsds::common::CCSDSJsonKeyCase::Upper
        } else {
            crate::ccsds::common::CCSDSJsonKeyCase::Lower
        };
        let result = self.inner.to_json_string(key_case)?;
        Ok(result)
    }

    /// Write the OPM to a file in the specified format.
    ///
    /// Args:
    ///     path (str): Output file path
    ///     format (str): Output format - "KVN", "XML", or "JSON"
    fn to_file(&self, path: &str, format: &str) -> PyResult<()> {
        let fmt = parse_format(format)?;
        self.inner.to_file(path, fmt)?;
        Ok(())
    }

    /// Convert the OPM to a Python dictionary.
    ///
    /// Epochs are serialized as CCSDS datetime strings for JSON/dict compatibility.
    ///
    /// Returns:
    ///     dict: Dictionary representation of the OPM
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, pyo3::types::PyDict>> {
        let dict = pyo3::types::PyDict::new(py);

        // Header
        let header = pyo3::types::PyDict::new(py);
        header.set_item("format_version", self.inner.header.format_version)?;
        header.set_item(
            "creation_date",
            crate::ccsds::common::format_ccsds_datetime(&self.inner.header.creation_date),
        )?;
        header.set_item("originator", &self.inner.header.originator)?;
        header.set_item("comments", &self.inner.header.comments)?;
        dict.set_item("header", header)?;

        // Metadata
        let meta = pyo3::types::PyDict::new(py);
        meta.set_item("object_name", &self.inner.metadata.object_name)?;
        meta.set_item("object_id", &self.inner.metadata.object_id)?;
        meta.set_item("center_name", &self.inner.metadata.center_name)?;
        meta.set_item("ref_frame", format!("{}", self.inner.metadata.ref_frame))?;
        meta.set_item(
            "time_system",
            format!("{}", self.inner.metadata.time_system),
        )?;
        dict.set_item("metadata", meta)?;

        // State vector
        let sv = pyo3::types::PyDict::new(py);
        sv.set_item(
            "epoch",
            crate::ccsds::common::format_ccsds_datetime(&self.inner.state_vector.epoch),
        )?;
        sv.set_item("position", self.inner.state_vector.position.to_vec())?;
        sv.set_item("velocity", self.inner.state_vector.velocity.to_vec())?;
        dict.set_item("state_vector", sv)?;

        // Keplerian elements
        if let Some(ref kep) = self.inner.keplerian_elements {
            let kep_dict = pyo3::types::PyDict::new(py);
            kep_dict.set_item("semi_major_axis", kep.semi_major_axis)?;
            kep_dict.set_item("eccentricity", kep.eccentricity)?;
            kep_dict.set_item("inclination", kep.inclination)?;
            kep_dict.set_item("ra_of_asc_node", kep.ra_of_asc_node)?;
            kep_dict.set_item("arg_of_pericenter", kep.arg_of_pericenter)?;
            kep_dict.set_item("true_anomaly", kep.true_anomaly)?;
            kep_dict.set_item("mean_anomaly", kep.mean_anomaly)?;
            kep_dict.set_item("gm", kep.gm)?;
            dict.set_item("keplerian_elements", kep_dict)?;
        }

        // Spacecraft parameters
        if let Some(ref sc) = self.inner.spacecraft_parameters {
            let sc_dict = pyo3::types::PyDict::new(py);
            sc_dict.set_item("mass", sc.mass)?;
            sc_dict.set_item("solar_rad_area", sc.solar_rad_area)?;
            sc_dict.set_item("solar_rad_coeff", sc.solar_rad_coeff)?;
            sc_dict.set_item("drag_area", sc.drag_area)?;
            sc_dict.set_item("drag_coeff", sc.drag_coeff)?;
            dict.set_item("spacecraft_parameters", sc_dict)?;
        }

        // Maneuvers
        if !self.inner.maneuvers.is_empty() {
            let mans = pyo3::types::PyList::empty(py);
            for m in &self.inner.maneuvers {
                let m_dict = pyo3::types::PyDict::new(py);
                m_dict.set_item(
                    "epoch_ignition",
                    crate::ccsds::common::format_ccsds_datetime(&m.epoch_ignition),
                )?;
                m_dict.set_item("duration", m.duration)?;
                m_dict.set_item("delta_mass", m.delta_mass)?;
                m_dict.set_item("ref_frame", format!("{}", m.ref_frame))?;
                m_dict.set_item("dv", m.dv.to_vec())?;
                mans.append(m_dict)?;
            }
            dict.set_item("maneuvers", mans)?;
        }

        // User-defined
        if let Some(ref ud) = self.inner.user_defined {
            let ud_dict = pyo3::types::PyDict::new(py);
            for (k, v) in &ud.parameters {
                ud_dict.set_item(k, v)?;
            }
            dict.set_item("user_defined", ud_dict)?;
        }

        Ok(dict)
    }

    // --- header properties ---

    /// CCSDS format version.
    ///
    /// Returns:
    ///     float: Format version
    #[getter]
    fn format_version(&self) -> f64 {
        self.inner.header.format_version
    }

    /// Set CCSDS format version.
    ///
    /// Args:
    ///     val (float): Format version
    #[setter]
    fn set_format_version(&mut self, val: f64) {
        self.inner.header.format_version = val;
    }

    /// Originator of the message.
    ///
    /// Returns:
    ///     str: Originator string
    #[getter]
    fn originator(&self) -> String {
        self.inner.header.originator.clone()
    }

    /// Set originator.
    ///
    /// Args:
    ///     val (str): Originator string
    #[setter]
    fn set_originator(&mut self, val: String) {
        self.inner.header.originator = val;
    }

    /// Creation date of the message.
    ///
    /// Returns:
    ///     Epoch: Creation date
    #[getter]
    fn creation_date(&self) -> PyEpoch {
        PyEpoch {
            obj: self.inner.header.creation_date,
        }
    }

    /// Set creation date.
    ///
    /// Args:
    ///     val (Epoch): Creation date
    #[setter]
    fn set_creation_date(&mut self, val: PyEpoch) {
        self.inner.header.creation_date = val.obj;
    }

    // --- metadata properties ---

    /// Object name.
    ///
    /// Returns:
    ///     str: Object name
    #[getter]
    fn object_name(&self) -> String {
        self.inner.metadata.object_name.clone()
    }

    /// Set object name.
    ///
    /// Args:
    ///     val (str): Object name
    #[setter]
    fn set_object_name(&mut self, val: String) {
        self.inner.metadata.object_name = val;
    }

    /// Object ID (international designator).
    ///
    /// Returns:
    ///     str: Object ID
    #[getter]
    fn object_id(&self) -> String {
        self.inner.metadata.object_id.clone()
    }

    /// Set object ID.
    ///
    /// Args:
    ///     val (str): Object ID
    #[setter]
    fn set_object_id(&mut self, val: String) {
        self.inner.metadata.object_id = val;
    }

    /// Center body name.
    ///
    /// Returns:
    ///     str: Center name
    #[getter]
    fn center_name(&self) -> String {
        self.inner.metadata.center_name.clone()
    }

    /// Set center body name.
    ///
    /// Args:
    ///     val (str): Center name
    #[setter]
    fn set_center_name(&mut self, val: String) {
        self.inner.metadata.center_name = val;
    }

    /// Reference frame name.
    ///
    /// Returns:
    ///     str: Reference frame
    #[getter]
    fn ref_frame(&self) -> String {
        format!("{}", self.inner.metadata.ref_frame)
    }

    /// Set reference frame.
    ///
    /// Args:
    ///     val (str): Reference frame name
    #[setter]
    fn set_ref_frame(&mut self, val: String) -> PyResult<()> {
        self.inner.metadata.ref_frame = CCSDSRefFrame::parse(&val);
        Ok(())
    }

    /// Time system name.
    ///
    /// Returns:
    ///     str: Time system
    #[getter]
    fn time_system(&self) -> String {
        format!("{}", self.inner.metadata.time_system)
    }

    /// Set time system.
    ///
    /// Args:
    ///     val (str): Time system name
    #[setter]
    fn set_time_system(&mut self, val: String) -> PyResult<()> {
        self.inner.metadata.time_system = CCSDSTimeSystem::parse(&val).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("Invalid time_system: {}", e))
        })?;
        Ok(())
    }

    // --- state vector properties ---

    /// State vector epoch.
    ///
    /// Returns:
    ///     Epoch: Epoch of the state vector
    #[getter]
    fn epoch(&self) -> PyEpoch {
        PyEpoch {
            obj: self.inner.state_vector.epoch,
        }
    }

    /// Set state vector epoch.
    ///
    /// Args:
    ///     val (Epoch): Epoch
    #[setter]
    fn set_epoch(&mut self, val: PyEpoch) {
        self.inner.state_vector.epoch = val.obj;
    }

    /// Position vector [x, y, z] in meters.
    ///
    /// Returns:
    ///     numpy.ndarray: Position [x, y, z] in meters
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        self.inner.state_vector.position.to_vec().into_pyarray(py)
    }

    /// Set position vector.
    ///
    /// Args:
    ///     val (list[float]): Position [x, y, z] in meters
    #[setter]
    fn set_position(&mut self, val: &pyo3::Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
        self.inner.state_vector.position = pyany_to_array3(val, "position")?;
        Ok(())
    }

    /// Velocity vector [vx, vy, vz] in m/s.
    ///
    /// Returns:
    ///     numpy.ndarray: Velocity [vx, vy, vz] in m/s
    #[getter]
    fn velocity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        self.inner.state_vector.velocity.to_vec().into_pyarray(py)
    }

    /// Set velocity vector.
    ///
    /// Args:
    ///     val (list[float] | numpy.ndarray): Velocity [vx, vy, vz] in m/s
    #[setter]
    fn set_velocity(&mut self, val: &pyo3::Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
        self.inner.state_vector.velocity = pyany_to_array3(val, "velocity")?;
        Ok(())
    }

    /// Combined state vector [x, y, z, vx, vy, vz] as a numpy array.
    ///
    /// Returns:
    ///     numpy.ndarray: 6-element state vector (position in meters, velocity in m/s)
    ///
    /// Example:
    ///     ```python
    ///     opm = OPM.from_file("state.opm")
    ///     state = opm.state  # numpy array [x, y, z, vx, vy, vz]
    ///     ```
    #[getter]
    fn state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        let p = &self.inner.state_vector.position;
        let v = &self.inner.state_vector.velocity;
        vec![p[0], p[1], p[2], v[0], v[1], v[2]].into_pyarray(py)
    }

    /// Set combined state vector [x, y, z, vx, vy, vz].
    ///
    /// Args:
    ///     val (list[float] | numpy.ndarray): 6-element state vector
    ///
    /// Example:
    ///     ```python
    ///     import numpy as np
    ///     opm.state = np.array([7000e3, 0, 0, 0, 7500, 0])
    ///     ```
    #[setter]
    fn set_state(&mut self, val: &pyo3::Bound<'_, pyo3::types::PyAny>) -> PyResult<()> {
        let sv = pyany_to_svector::<6>(val).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("state: {}", e))
        })?;
        self.inner.state_vector.position = [sv[0], sv[1], sv[2]];
        self.inner.state_vector.velocity = [sv[3], sv[4], sv[5]];
        Ok(())
    }

    // --- keplerian properties ---

    /// Whether Keplerian elements are present.
    ///
    /// Returns:
    ///     bool: True if Keplerian elements exist
    #[getter]
    fn has_keplerian_elements(&self) -> bool {
        self.inner.keplerian_elements.is_some()
    }

    /// Semi-major axis in meters, or None.
    ///
    /// Returns:
    ///     float: Semi-major axis (meters), or None
    #[getter]
    fn semi_major_axis(&self) -> Option<f64> {
        self.inner
            .keplerian_elements
            .as_ref()
            .map(|k| k.semi_major_axis)
    }

    /// Eccentricity, or None if no Keplerian elements.
    ///
    /// Returns:
    ///     float: Eccentricity, or None
    #[getter]
    fn eccentricity(&self) -> Option<f64> {
        self.inner
            .keplerian_elements
            .as_ref()
            .map(|k| k.eccentricity)
    }

    /// Inclination in degrees, or None if no Keplerian elements.
    ///
    /// Returns:
    ///     float: Inclination (degrees), or None
    #[getter]
    fn inclination(&self) -> Option<f64> {
        self.inner
            .keplerian_elements
            .as_ref()
            .map(|k| k.inclination)
    }

    /// Right ascension of ascending node in degrees, or None.
    ///
    /// Returns:
    ///     float: RAAN (degrees), or None
    #[getter]
    fn ra_of_asc_node(&self) -> Option<f64> {
        self.inner
            .keplerian_elements
            .as_ref()
            .map(|k| k.ra_of_asc_node)
    }

    /// Argument of pericenter in degrees, or None.
    ///
    /// Returns:
    ///     float: Argument of pericenter (degrees), or None
    #[getter]
    fn arg_of_pericenter(&self) -> Option<f64> {
        self.inner
            .keplerian_elements
            .as_ref()
            .map(|k| k.arg_of_pericenter)
    }

    /// True anomaly in degrees, or None.
    ///
    /// Returns:
    ///     float: True anomaly (degrees), or None
    #[getter]
    fn true_anomaly(&self) -> Option<f64> {
        self.inner
            .keplerian_elements
            .as_ref()
            .and_then(|k| k.true_anomaly)
    }

    /// Mean anomaly in degrees, or None.
    ///
    /// Returns:
    ///     float: Mean anomaly (degrees), or None
    #[getter]
    fn mean_anomaly(&self) -> Option<f64> {
        self.inner
            .keplerian_elements
            .as_ref()
            .and_then(|k| k.mean_anomaly)
    }

    /// Gravitational parameter in m^3/s^2, or None.
    ///
    /// Returns:
    ///     float: GM (m^3/s^2), or None
    #[getter]
    fn gm(&self) -> Option<f64> {
        self.inner
            .keplerian_elements
            .as_ref()
            .and_then(|k| k.gm)
    }

    // --- spacecraft properties ---

    /// Spacecraft mass in kg, or None.
    ///
    /// Returns:
    ///     float: Mass (kg), or None
    #[getter]
    fn mass(&self) -> Option<f64> {
        self.inner
            .spacecraft_parameters
            .as_ref()
            .and_then(|s| s.mass)
    }

    /// Solar radiation pressure area in m^2, or None.
    ///
    /// Returns:
    ///     float: Solar radiation pressure area (m^2), or None
    #[getter]
    fn solar_rad_area(&self) -> Option<f64> {
        self.inner
            .spacecraft_parameters
            .as_ref()
            .and_then(|s| s.solar_rad_area)
    }

    /// Solar radiation pressure coefficient, or None.
    ///
    /// Returns:
    ///     float: Solar radiation pressure coefficient, or None
    #[getter]
    fn solar_rad_coeff(&self) -> Option<f64> {
        self.inner
            .spacecraft_parameters
            .as_ref()
            .and_then(|s| s.solar_rad_coeff)
    }

    /// Drag area in m^2, or None.
    ///
    /// Returns:
    ///     float: Drag area (m^2), or None
    #[getter]
    fn drag_area(&self) -> Option<f64> {
        self.inner
            .spacecraft_parameters
            .as_ref()
            .and_then(|s| s.drag_area)
    }

    /// Drag coefficient, or None.
    ///
    /// Returns:
    ///     float: Drag coefficient, or None
    #[getter]
    fn drag_coeff(&self) -> Option<f64> {
        self.inner
            .spacecraft_parameters
            .as_ref()
            .and_then(|s| s.drag_coeff)
    }

    // --- maneuver access ---

    /// Maneuver collection, supporting len/indexing/iteration.
    ///
    /// Returns:
    ///     OPMManeuvers: Collection of maneuvers
    #[getter]
    fn maneuvers(slf: Bound<'_, Self>) -> PyOPMManeuvers {
        let parent: Py<PyAny> = slf.clone().into_any().unbind();
        PyOPMManeuvers { parent }
    }

    // --- builder methods ---

    /// Add a maneuver to the OPM.
    ///
    /// Args:
    ///     epoch_ignition (Epoch): Epoch of ignition
    ///     duration (float): Duration in seconds
    ///     ref_frame (str): Reference frame for delta-V
    ///     dv (list[float]): Delta-V [dv1, dv2, dv3] in m/s
    ///     delta_mass (float | None): Mass change in kg
    ///
    /// Returns:
    ///     int: Index of the new maneuver
    #[pyo3(signature = (epoch_ignition, duration, ref_frame, dv, delta_mass=None))]
    fn add_maneuver(
        &mut self,
        epoch_ignition: PyEpoch,
        duration: f64,
        ref_frame: String,
        dv: &pyo3::Bound<'_, pyo3::types::PyAny>,
        delta_mass: Option<f64>,
    ) -> PyResult<usize> {
        let rf = CCSDSRefFrame::parse(&ref_frame);
        let dv_arr = pyany_to_array3(dv, "dv")?;
        self.inner.maneuvers.push(OPMManeuver {
            epoch_ignition: epoch_ignition.obj,
            duration,
            delta_mass,
            ref_frame: rf,
            dv: dv_arr,
            comments: Vec::new(),
        });
        Ok(self.inner.maneuvers.len() - 1)
    }

    /// Remove a maneuver by index.
    ///
    /// Args:
    ///     idx (int): Index of the maneuver to remove
    fn remove_maneuver(&mut self, idx: usize) -> PyResult<()> {
        if idx >= self.inner.maneuvers.len() {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "maneuver index {} out of range (have {})",
                idx,
                self.inner.maneuvers.len()
            )));
        }
        self.inner.maneuvers.remove(idx);
        Ok(())
    }

    // --- internal delegate methods for maneuver proxies ---

    fn _num_maneuvers(&self) -> usize {
        self.inner.maneuvers.len()
    }

    fn _man_get_epoch_ignition(&self, idx: usize) -> PyResult<PyEpoch> {
        Ok(PyEpoch {
            obj: self.get_man(idx)?.epoch_ignition,
        })
    }

    fn _man_set_epoch_ignition(&mut self, idx: usize, val: PyEpoch) -> PyResult<()> {
        self.get_man_mut(idx)?.epoch_ignition = val.obj;
        Ok(())
    }

    fn _man_get_duration(&self, idx: usize) -> PyResult<f64> {
        Ok(self.get_man(idx)?.duration)
    }

    fn _man_set_duration(&mut self, idx: usize, val: f64) -> PyResult<()> {
        self.get_man_mut(idx)?.duration = val;
        Ok(())
    }

    fn _man_get_delta_mass(&self, idx: usize) -> PyResult<Option<f64>> {
        Ok(self.get_man(idx)?.delta_mass)
    }

    fn _man_set_delta_mass(&mut self, idx: usize, val: Option<f64>) -> PyResult<()> {
        self.get_man_mut(idx)?.delta_mass = val;
        Ok(())
    }

    fn _man_get_ref_frame(&self, idx: usize) -> PyResult<String> {
        Ok(format!("{}", self.get_man(idx)?.ref_frame))
    }

    fn _man_set_ref_frame(&mut self, idx: usize, val: String) -> PyResult<()> {
        let rf = CCSDSRefFrame::parse(&val);
        self.get_man_mut(idx)?.ref_frame = rf;
        Ok(())
    }

    fn _man_get_dv(&self, idx: usize) -> PyResult<Vec<f64>> {
        Ok(self.get_man(idx)?.dv.to_vec())
    }

    fn _man_set_dv(&mut self, idx: usize, val: Vec<f64>) -> PyResult<()> {
        let arr = vec_to_array3(val, "dv")?;
        self.get_man_mut(idx)?.dv = arr;
        Ok(())
    }

    fn _man_get_comments(&self, idx: usize) -> PyResult<Vec<String>> {
        Ok(self.get_man(idx)?.comments.clone())
    }

    fn _man_set_comments(&mut self, idx: usize, val: Vec<String>) -> PyResult<()> {
        self.get_man_mut(idx)?.comments = val;
        Ok(())
    }

    fn __repr__(&self) -> String {
        format!(
            "OPM(object='{}', frame='{}')",
            self.inner.metadata.object_name, self.inner.metadata.ref_frame
        )
    }
}

impl PyOPM {
    fn get_man(&self, idx: usize) -> PyResult<&OPMManeuver> {
        self.inner.maneuvers.get(idx).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "maneuver index {} out of range (have {})",
                idx,
                self.inner.maneuvers.len()
            ))
        })
    }

    fn get_man_mut(&mut self, idx: usize) -> PyResult<&mut OPMManeuver> {
        let len = self.inner.maneuvers.len();
        self.inner.maneuvers.get_mut(idx).ok_or_else(|| {
            pyo3::exceptions::PyIndexError::new_err(format!(
                "maneuver index {} out of range (have {})",
                idx, len
            ))
        })
    }
}

// ─────────────────────────────────────────────
// CDM — Sub-objects for programmatic creation
// ─────────────────────────────────────────────

/// A CDM state vector with position and velocity at TCA.
///
/// Args:
///     position (list[float]): Position [x, y, z] in meters
///     velocity (list[float]): Velocity [vx, vy, vz] in m/s
///
/// Example:
///     ```python
///     from brahe.ccsds import CDMStateVector
///     sv = CDMStateVector(
///         position=[7000e3, 0.0, 0.0],
///         velocity=[0.0, 7500.0, 0.0],
///     )
///     print(sv.position)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "CDMStateVector")]
pub struct PyCDMStateVector {
    inner: CDMStateVector,
}

#[pymethods]
impl PyCDMStateVector {
    #[new]
    #[pyo3(signature = (position, velocity))]
    fn new(
        position: &pyo3::Bound<'_, pyo3::types::PyAny>,
        velocity: &pyo3::Bound<'_, pyo3::types::PyAny>,
    ) -> PyResult<Self> {
        let pos = pyany_to_array3(position, "position")?;
        let vel = pyany_to_array3(velocity, "velocity")?;
        Ok(Self {
            inner: CDMStateVector::new(pos, vel),
        })
    }

    /// numpy.ndarray: Position [x, y, z] in meters
    #[getter]
    fn position<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        self.inner.position.to_vec().into_pyarray(py)
    }

    /// numpy.ndarray: Velocity [vx, vy, vz] in m/s
    #[getter]
    fn velocity<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        self.inner.velocity.to_vec().into_pyarray(py)
    }

    fn __repr__(&self) -> String {
        format!(
            "CDMStateVector(pos=[{:.1}, {:.1}, {:.1}] m, vel=[{:.1}, {:.1}, {:.1}] m/s)",
            self.inner.position[0], self.inner.position[1], self.inner.position[2],
            self.inner.velocity[0], self.inner.velocity[1], self.inner.velocity[2],
        )
    }
}

/// A CDM RTN covariance matrix (6x6 position/velocity).
///
/// Args:
///     matrix (list[list[float]]): 6x6 symmetric covariance matrix in RTN frame.
///         Units: position-position m², position-velocity m²/s, velocity-velocity m²/s²
///
/// Example:
///     ```python
///     import numpy as np
///     from brahe.ccsds import CDMRTNCovariance
///     cov = CDMRTNCovariance(matrix=np.eye(6).tolist())
///     print(cov.matrix[0][0])
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "CDMRTNCovariance")]
pub struct PyCDMRTNCovariance {
    inner: CDMRTNCovariance,
}

#[pymethods]
impl PyCDMRTNCovariance {
    #[new]
    #[pyo3(signature = (matrix))]
    fn new(matrix: &pyo3::Bound<'_, pyo3::types::PyAny>) -> PyResult<Self> {
        let mat = pyany_to_smatrix::<6, 6>(matrix)?;
        Ok(Self {
            inner: CDMRTNCovariance::from_6x6(mat),
        })
    }

    /// numpy.ndarray: 6x6 RTN covariance matrix
    #[getter]
    fn matrix<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix2>> {
        let m = self.inner.to_6x6();
        matrix_to_numpy!(py, m, 6, 6, f64)
    }

    fn __repr__(&self) -> String {
        let m = self.inner.to_6x6();
        format!(
            "CDMRTNCovariance(diag=[{:.3e}, {:.3e}, {:.3e}, {:.3e}, {:.3e}, {:.3e}])",
            m[(0, 0)], m[(1, 1)], m[(2, 2)], m[(3, 3)], m[(4, 4)], m[(5, 5)],
        )
    }
}

/// One object in a CDM (metadata + state vector + covariance).
///
/// Combines object identity, reference frame, state at TCA, and RTN covariance.
///
/// Args:
///     designator (str): Catalog ID (e.g. "12345")
///     catalog_name (str): Catalog source (e.g. "SATCAT")
///     name (str): Object name (e.g. "SATELLITE A")
///     international_designator (str): COSPAR ID (e.g. "2020-001A")
///     ephemeris_name (str): Ephemeris source (e.g. "NONE")
///     covariance_method (str): "CALCULATED" or "DEFAULT"
///     maneuverable (str): "YES", "NO", "N/A", or "UNKNOWN"
///     ref_frame (str): Reference frame (e.g. "EME2000")
///     state_vector (CDMStateVector): State at TCA
///     rtn_covariance (CDMRTNCovariance): RTN covariance matrix
///
/// Example:
///     ```python
///     from brahe.ccsds import CDMObject, CDMStateVector, CDMRTNCovariance
///     import numpy as np
///
///     sv = CDMStateVector([7000e3, 0.0, 0.0], [0.0, 7500.0, 0.0])
///     cov = CDMRTNCovariance(np.eye(6).tolist())
///     obj = CDMObject(
///         designator="12345",
///         catalog_name="SATCAT",
///         name="SAT A",
///         international_designator="2020-001A",
///         ephemeris_name="NONE",
///         covariance_method="CALCULATED",
///         maneuverable="YES",
///         ref_frame="EME2000",
///         state_vector=sv,
///         rtn_covariance=cov,
///     )
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "CDMObject")]
pub struct PyCDMObject {
    inner: CDMObject,
}

#[pymethods]
impl PyCDMObject {
    #[new]
    #[pyo3(signature = (designator, catalog_name, name, international_designator, ephemeris_name, covariance_method, maneuverable, ref_frame, state_vector, rtn_covariance))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        designator: &str,
        catalog_name: &str,
        name: &str,
        international_designator: &str,
        ephemeris_name: &str,
        covariance_method: &str,
        maneuverable: &str,
        ref_frame: &str,
        state_vector: &PyCDMStateVector,
        rtn_covariance: &PyCDMRTNCovariance,
    ) -> Self {
        let metadata = CDMObjectMetadata::new(
            String::new(), // object label set by CDM constructor
            designator.to_string(),
            catalog_name.to_string(),
            name.to_string(),
            international_designator.to_string(),
            ephemeris_name.to_string(),
            covariance_method.to_string(),
            maneuverable.to_string(),
            CCSDSRefFrame::parse(ref_frame),
        );
        Self {
            inner: CDMObject::new(
                metadata,
                state_vector.inner.clone(),
                rtn_covariance.inner.clone(),
            ),
        }
    }

    /// str: Catalog ID
    #[getter]
    fn designator(&self) -> &str {
        &self.inner.metadata.object_designator
    }

    /// str: Catalog source
    #[getter]
    fn catalog_name(&self) -> &str {
        &self.inner.metadata.catalog_name
    }

    /// str: Object name
    #[getter]
    fn name(&self) -> &str {
        &self.inner.metadata.object_name
    }

    /// str: COSPAR international designator
    #[getter]
    fn international_designator(&self) -> &str {
        &self.inner.metadata.international_designator
    }

    /// str: Ephemeris name
    #[getter]
    fn ephemeris_name(&self) -> &str {
        &self.inner.metadata.ephemeris_name
    }

    /// str: Covariance method (CALCULATED or DEFAULT)
    #[getter]
    fn covariance_method(&self) -> &str {
        &self.inner.metadata.covariance_method
    }

    /// str: Maneuverable flag (YES, NO, N/A, UNKNOWN)
    #[getter]
    fn maneuverable(&self) -> &str {
        &self.inner.metadata.maneuverable
    }

    /// str: Reference frame
    #[getter]
    fn ref_frame(&self) -> String {
        format!("{}", self.inner.metadata.ref_frame)
    }

    /// numpy.ndarray: State vector [x, y, z, vx, vy, vz] in m and m/s
    #[getter]
    fn state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        let sv = &self.inner.data.state_vector;
        vec![
            sv.position[0], sv.position[1], sv.position[2],
            sv.velocity[0], sv.velocity[1], sv.velocity[2],
        ].into_pyarray(py)
    }

    /// numpy.ndarray: RTN covariance matrix (6x6)
    #[getter]
    fn covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix2>> {
        let m = self.inner.data.rtn_covariance.to_6x6();
        matrix_to_numpy!(py, m, 6, 6, f64)
    }

    fn __repr__(&self) -> String {
        format!(
            "CDMObject(name='{}', designator='{}')",
            self.inner.metadata.object_name,
            self.inner.metadata.object_designator,
        )
    }
}

// ─────────────────────────────────────────────
// CDM — Top-level message
// ─────────────────────────────────────────────

/// A CCSDS Conjunction Data Message (CDM).
///
/// CDM messages describe a conjunction between two space objects, containing
/// state vectors, covariance matrices, and collision probability data.
///
/// Can be created programmatically or parsed from KVN/XML/JSON.
///
/// Args:
///     originator (str): Originator of the message
///     message_id (str): Unique message identifier
///     tca (Epoch): Time of Closest Approach
///     miss_distance (float): Miss distance in meters
///     object1 (CDMObject): First conjunction object
///     object2 (CDMObject): Second conjunction object
///
/// Example:
///     ```python
///     from brahe.ccsds import CDM, CDMObject, CDMStateVector, CDMRTNCovariance
///     import numpy as np
///
///     sv1 = CDMStateVector([7000e3, 0.0, 0.0], [0.0, 7500.0, 0.0])
///     cov1 = CDMRTNCovariance(np.eye(6).tolist())
///     obj1 = CDMObject("12345", "SATCAT", "SAT A", "2020-001A",
///                       "NONE", "CALCULATED", "YES", "EME2000", sv1, cov1)
///
///     sv2 = CDMStateVector([7001e3, 0.0, 0.0], [0.0, -7500.0, 0.0])
///     cov2 = CDMRTNCovariance(np.eye(6).tolist())
///     obj2 = CDMObject("67890", "SATCAT", "SAT B", "2021-002B",
///                       "NONE", "CALCULATED", "NO", "EME2000", sv2, cov2)
///
///     tca = Epoch.from_datetime(2024, 1, 15, 12, 0, 0.0, 0.0, TimeSystem.UTC)
///     cdm = CDM(originator="TEST_ORG", message_id="MSG001",
///               tca=tca, miss_distance=715.0, object1=obj1, object2=obj2)
///     print(cdm.miss_distance)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "CDM")]
pub struct PyCDM {
    inner: RustCDM,
}

#[pymethods]
impl PyCDM {
    /// Create a new CDM message programmatically.
    ///
    /// Args:
    ///     originator (str): Originator of the message
    ///     message_id (str): Unique message identifier
    ///     tca (Epoch): Time of Closest Approach
    ///     miss_distance (float): Miss distance in meters
    ///     object1 (CDMObject): First conjunction object
    ///     object2 (CDMObject): Second conjunction object
    ///
    /// Returns:
    ///     CDM: New CDM message
    #[new]
    #[pyo3(signature = (originator, message_id, tca, miss_distance, object1, object2))]
    fn new(
        originator: &str,
        message_id: &str,
        tca: &PyEpoch,
        miss_distance: f64,
        object1: &PyCDMObject,
        object2: &PyCDMObject,
    ) -> Self {
        let mut obj1 = object1.inner.clone();
        let mut obj2 = object2.inner.clone();
        obj1.metadata.object = "OBJECT1".to_string();
        obj2.metadata.object = "OBJECT2".to_string();
        let inner = RustCDM::new(
            originator.to_string(),
            message_id.to_string(),
            tca.obj,
            miss_distance,
            obj1,
            obj2,
        );
        PyCDM { inner }
    }

    /// Parse a CDM from a string, auto-detecting the format.
    ///
    /// Args:
    ///     content (str): String content of the CDM message
    ///
    /// Returns:
    ///     CDM: Parsed CDM message
    #[staticmethod]
    #[allow(clippy::should_implement_trait)]
    fn from_str(content: &str) -> PyResult<Self> {
        let inner = RustCDM::from_str(content)?;
        Ok(PyCDM { inner })
    }

    /// Parse a CDM from a file, auto-detecting the format.
    ///
    /// Args:
    ///     path (str): Path to the CDM file
    ///
    /// Returns:
    ///     CDM: Parsed CDM message
    #[staticmethod]
    fn from_file(path: &str) -> PyResult<Self> {
        let inner = RustCDM::from_file(path)?;
        Ok(PyCDM { inner })
    }

    /// Write the CDM to a string in the specified format.
    ///
    /// Args:
    ///     format (str): Output format - "KVN", "XML", or "JSON"
    ///
    /// Returns:
    ///     str: Serialized CDM string
    fn to_string(&self, format: &str) -> PyResult<String> {
        let fmt = parse_format(format)?;
        let result = self.inner.to_string(fmt)?;
        Ok(result)
    }

    /// Write the CDM to JSON with explicit key case control.
    ///
    /// Args:
    ///     uppercase_keys (bool): If True, use uppercase CCSDS keywords. Default: False.
    ///
    /// Returns:
    ///     str: Serialized JSON string
    #[pyo3(signature = (uppercase_keys=false))]
    fn to_json_string(&self, uppercase_keys: bool) -> PyResult<String> {
        let key_case = if uppercase_keys {
            crate::ccsds::common::CCSDSJsonKeyCase::Upper
        } else {
            crate::ccsds::common::CCSDSJsonKeyCase::Lower
        };
        let result = self.inner.to_json_string(key_case)?;
        Ok(result)
    }

    /// Write the CDM to a file in the specified format.
    ///
    /// Args:
    ///     path (str): Output file path
    ///     format (str): Output format - "KVN", "XML", or "JSON"
    fn to_file(&self, path: &str, format: &str) -> PyResult<()> {
        let fmt = parse_format(format)?;
        self.inner.to_file(path, fmt)?;
        Ok(())
    }

    // --- Header properties ---

    /// float: CDM format version (e.g. 1.0, 2.0)
    #[getter]
    fn format_version(&self) -> f64 {
        self.inner.header.format_version
    }

    /// str: Originator of the message
    #[getter]
    fn originator(&self) -> &str {
        &self.inner.header.originator
    }

    /// str: Unique message identifier
    #[getter]
    fn message_id(&self) -> &str {
        &self.inner.header.message_id
    }

    /// Optional[str]: Spacecraft name(s) CDM applies to
    #[getter]
    fn message_for(&self) -> Option<&str> {
        self.inner.header.message_for.as_deref()
    }

    /// Epoch: Creation date of the message
    #[getter]
    fn creation_date(&self) -> PyEpoch {
        PyEpoch { obj: self.inner.header.creation_date }
    }

    // --- Relative metadata properties ---

    /// Epoch: Time of Closest Approach
    #[getter]
    fn tca(&self) -> PyEpoch {
        PyEpoch { obj: *self.inner.tca() }
    }

    /// float: Miss distance in meters
    #[getter]
    fn miss_distance(&self) -> f64 {
        self.inner.miss_distance()
    }

    /// Optional[float]: Collision probability
    #[getter]
    fn collision_probability(&self) -> Option<f64> {
        self.inner.collision_probability()
    }

    /// Optional[str]: Collision probability method
    #[getter]
    fn collision_probability_method(&self) -> Option<&str> {
        self.inner.relative_metadata.collision_probability_method.as_deref()
    }

    /// Set the collision probability.
    ///
    /// Args:
    ///     value (float | None): Collision probability value
    #[setter]
    fn set_collision_probability(&mut self, value: Option<f64>) {
        self.inner.relative_metadata.collision_probability = value;
    }

    /// Set the collision probability method.
    ///
    /// Args:
    ///     value (str | None): Collision probability method name (e.g. "FOSTER-1992")
    #[setter]
    fn set_collision_probability_method(&mut self, value: Option<String>) {
        self.inner.relative_metadata.collision_probability_method = value;
    }

    /// Optional[float]: Relative speed in m/s
    #[getter]
    fn relative_speed(&self) -> Option<f64> {
        self.inner.relative_metadata.relative_speed
    }

    // --- Object 1 properties ---

    /// str: Object 1 name
    #[getter]
    fn object1_name(&self) -> &str {
        &self.inner.object1.metadata.object_name
    }

    /// str: Object 1 designator (catalog ID)
    #[getter]
    fn object1_designator(&self) -> &str {
        &self.inner.object1.metadata.object_designator
    }

    /// str: Object 1 international designator
    #[getter]
    fn object1_international_designator(&self) -> &str {
        &self.inner.object1.metadata.international_designator
    }

    /// numpy.ndarray: Object 1 state vector [x, y, z, vx, vy, vz] in m and m/s
    #[getter]
    fn object1_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        let s = self.inner.object1_state();
        (0..6).map(|i| s[i]).collect::<Vec<f64>>().into_pyarray(py)
    }

    /// numpy.ndarray: Object 1 RTN covariance matrix (6x6) in m², m²/s, m²/s²
    #[getter]
    fn object1_covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix2>> {
        let m = self.inner.object1_rtn_covariance_6x6();
        matrix_to_numpy!(py, m, 6, 6, f64)
    }

    /// str: Object 1 reference frame
    #[getter]
    fn object1_ref_frame(&self) -> String {
        format!("{}", self.inner.object1.metadata.ref_frame)
    }

    // --- Object 2 properties ---

    /// str: Object 2 name
    #[getter]
    fn object2_name(&self) -> &str {
        &self.inner.object2.metadata.object_name
    }

    /// str: Object 2 designator (catalog ID)
    #[getter]
    fn object2_designator(&self) -> &str {
        &self.inner.object2.metadata.object_designator
    }

    /// str: Object 2 international designator
    #[getter]
    fn object2_international_designator(&self) -> &str {
        &self.inner.object2.metadata.international_designator
    }

    /// numpy.ndarray: Object 2 state vector [x, y, z, vx, vy, vz] in m and m/s
    #[getter]
    fn object2_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        let s = self.inner.object2_state();
        (0..6).map(|i| s[i]).collect::<Vec<f64>>().into_pyarray(py)
    }

    /// numpy.ndarray: Object 2 RTN covariance matrix (6x6) in m², m²/s, m²/s²
    #[getter]
    fn object2_covariance<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix2>> {
        let m = self.inner.object2_rtn_covariance_6x6();
        matrix_to_numpy!(py, m, 6, 6, f64)
    }

    /// str: Object 2 reference frame
    #[getter]
    fn object2_ref_frame(&self) -> String {
        format!("{}", self.inner.object2.metadata.ref_frame)
    }

    fn __repr__(&self) -> String {
        format!(
            "CDM(tca={}, miss_distance={:.1}m, obj1='{}', obj2='{}')",
            crate::ccsds::common::format_ccsds_datetime(self.inner.tca()),
            self.inner.miss_distance(),
            self.inner.object1.metadata.object_name,
            self.inner.object2.metadata.object_name,
        )
    }
}

// ─────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────

/// Extract a PyEpoch from a Bound<PyAny> (works around from_py_object error type).
fn extract_epoch(obj: &pyo3::Bound<'_, pyo3::types::PyAny>) -> PyResult<PyEpoch> {
    let bound: &pyo3::Bound<'_, PyEpoch> = obj.cast()?;
    Ok(bound.borrow().clone())
}

/// Parse a format string into CCSDSFormat.
fn parse_format(format: &str) -> PyResult<CCSDSFormat> {
    match format.to_uppercase().as_str() {
        "KVN" => Ok(CCSDSFormat::KVN),
        "XML" => Ok(CCSDSFormat::XML),
        "JSON" => Ok(CCSDSFormat::JSON),
        _ => Err(pyo3::exceptions::PyValueError::new_err(format!(
            "Unknown format '{}'. Expected 'KVN', 'XML', or 'JSON'",
            format
        ))),
    }
}

/// Normalize a Python-style index (supporting negative indexing).
fn normalize_index(index: isize, len: usize, kind: &str) -> PyResult<usize> {
    let len_i = len as isize;
    let actual = if index < 0 { len_i + index } else { index };
    if actual < 0 || actual >= len_i {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "{} index {} out of range (have {})",
            kind, index, len
        )));
    }
    Ok(actual as usize)
}

/// Convert a Vec<f64> to [f64; 3], validating length.
fn vec_to_array3(v: Vec<f64>, name: &str) -> PyResult<[f64; 3]> {
    if v.len() != 3 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{} must have exactly 3 elements, got {}",
            name,
            v.len()
        )));
    }
    Ok([v[0], v[1], v[2]])
}

/// Convert a Python array-like (list, numpy array, slice) to [f64; 3].
fn pyany_to_array3(val: &pyo3::Bound<'_, pyo3::types::PyAny>, name: &str) -> PyResult<[f64; 3]> {
    let sv = pyany_to_svector::<3>(val).map_err(|e| {
        pyo3::exceptions::PyValueError::new_err(format!("{}: {}", name, e))
    })?;
    Ok([sv[0], sv[1], sv[2]])
}

/// Convert an optional Python array-like to Option<[f64; 3]>.
/// Handles both Rust `None` and Python `None`.
fn pyany_to_optional_array3(
    val: Option<&pyo3::Bound<'_, pyo3::types::PyAny>>,
    name: &str,
) -> PyResult<Option<[f64; 3]>> {
    match val {
        Some(v) if !v.is_none() => Ok(Some(pyany_to_array3(v, name)?)),
        _ => Ok(None),
    }
}
