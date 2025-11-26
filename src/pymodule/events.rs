// ================================
// Event Detection Python Bindings
// ================================
//
// Python bindings for event detection during numerical orbit propagation.
// All imports must be in mod.rs due to PyO3 limitations.

// ================================
// Enums
// ================================

/// Event detection direction.
///
/// Specifies which type of zero-crossing to detect: increasing (negative to positive),
/// decreasing (positive to negative), or any crossing.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Detect decreasing crossings only
///     direction = bh.EventDirection.DECREASING
///
///     # Compare directions
///     if direction == bh.EventDirection.DECREASING:
///         print("Decreasing")
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "EventDirection")]
pub struct PyEventDirection {
    pub(crate) direction: events::EventDirection,
}

#[pymethods]
impl PyEventDirection {
    /// Detect increasing zero-crossings (negative to positive).
    #[classattr]
    #[pyo3(name = "INCREASING")]
    fn increasing() -> Self {
        PyEventDirection {
            direction: events::EventDirection::Increasing,
        }
    }

    /// Detect decreasing zero-crossings (positive to negative).
    #[classattr]
    #[pyo3(name = "DECREASING")]
    fn decreasing() -> Self {
        PyEventDirection {
            direction: events::EventDirection::Decreasing,
        }
    }

    /// Detect any zero-crossing.
    #[classattr]
    #[pyo3(name = "ANY")]
    fn any() -> Self {
        PyEventDirection {
            direction: events::EventDirection::Any,
        }
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.direction == other.direction),
            CompareOp::Ne => Ok(self.direction != other.direction),
            _ => Err(exceptions::PyNotImplementedError::new_err(
                "Comparison not supported",
            )),
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.direction)
    }

    fn __repr__(&self) -> String {
        format!("EventDirection.{:?}", self.direction)
    }
}

/// Edge type for binary event detection.
///
/// Specifies which boolean transition to detect: rising edge (false → true),
/// falling edge (true → false), or any edge.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Detect falling edge (true → false, e.g., eclipse entry)
///     edge = bh.EdgeType.FALLING_EDGE
///
///     def is_sunlit(epoch, state):
///         return state[0] > 0  # Simple check
///
///     event = bh.BinaryEvent("Eclipse Entry", is_sunlit, edge)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "EdgeType")]
pub struct PyEdgeType {
    pub(crate) edge: events::EdgeType,
}

#[pymethods]
impl PyEdgeType {
    /// Detect rising edge (false → true transition).
    #[classattr]
    #[pyo3(name = "RISING_EDGE")]
    fn rising_edge() -> Self {
        PyEdgeType {
            edge: events::EdgeType::RisingEdge,
        }
    }

    /// Detect falling edge (true → false transition).
    #[classattr]
    #[pyo3(name = "FALLING_EDGE")]
    fn falling_edge() -> Self {
        PyEdgeType {
            edge: events::EdgeType::FallingEdge,
        }
    }

    /// Detect any edge (any boolean transition).
    #[classattr]
    #[pyo3(name = "ANY_EDGE")]
    fn any_edge() -> Self {
        PyEdgeType {
            edge: events::EdgeType::AnyEdge,
        }
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.edge == other.edge),
            CompareOp::Ne => Ok(self.edge != other.edge),
            _ => Err(exceptions::PyNotImplementedError::new_err(
                "Comparison not supported",
            )),
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.edge)
    }

    fn __repr__(&self) -> String {
        format!("EdgeType.{:?}", self.edge)
    }
}

/// Action to take when an event is detected.
///
/// Determines whether propagation should stop or continue after an event is detected.
/// Can be set as the default action via `.is_terminal()` or returned from a callback
/// to override the default.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Terminal event (stops propagation)
///     event = bh.TimeEvent(target_epoch, "Maneuver").is_terminal()
///
///     # Callback can override the action
///     def callback(epoch, state):
///         return (None, bh.EventAction.STOP)  # Override to stop
///
///     event = bh.TimeEvent(epoch, "Check").with_callback(callback)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "EventAction")]
pub struct PyEventAction {
    pub(crate) action: events::EventAction,
}

#[pymethods]
impl PyEventAction {
    /// Stop propagation after this event.
    #[classattr]
    #[pyo3(name = "STOP")]
    fn stop() -> Self {
        PyEventAction {
            action: events::EventAction::Stop,
        }
    }

    /// Continue propagation after this event.
    #[classattr]
    #[pyo3(name = "CONTINUE")]
    fn continue_() -> Self {
        PyEventAction {
            action: events::EventAction::Continue,
        }
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.action == other.action),
            CompareOp::Ne => Ok(self.action != other.action),
            _ => Err(exceptions::PyNotImplementedError::new_err(
                "Comparison not supported",
            )),
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.action)
    }

    fn __repr__(&self) -> String {
        format!("EventAction.{:?}", self.action)
    }
}

/// Type of event: instantaneous or period.
///
/// Instantaneous events occur at a single point in time (e.g., apoapsis crossing).
/// Period events maintain a condition over an interval (e.g., time within eclipse).
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Check event type from a detected event
///     if detected_event.event_type == bh.EventType.INSTANTANEOUS:
///         print(f"Event at {detected_event.window_open}")
///     else:
///         print(f"Period from {detected_event.window_open} to {detected_event.window_close}")
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "EventType")]
pub struct PyEventType {
    pub(crate) event_type: events::EventType,
}

#[pymethods]
impl PyEventType {
    /// Instantaneous event (occurs at a point in time).
    #[classattr]
    #[pyo3(name = "INSTANTANEOUS")]
    fn instantaneous() -> Self {
        PyEventType {
            event_type: events::EventType::Instantaneous,
        }
    }

    /// Period event (maintains condition over interval).
    #[classattr]
    #[pyo3(name = "PERIOD")]
    fn period() -> Self {
        PyEventType {
            event_type: events::EventType::Period,
        }
    }

    fn __richcmp__(&self, other: &Self, op: CompareOp) -> PyResult<bool> {
        match op {
            CompareOp::Eq => Ok(self.event_type == other.event_type),
            CompareOp::Ne => Ok(self.event_type != other.event_type),
            _ => Err(exceptions::PyNotImplementedError::new_err(
                "Comparison not supported",
            )),
        }
    }

    fn __str__(&self) -> String {
        format!("{:?}", self.event_type)
    }

    fn __repr__(&self) -> String {
        format!("EventType.{:?}", self.event_type)
    }
}

// ================================
// Detected Event Results
// ================================

/// Information about a detected event.
///
/// Contains all information about an event that was detected during propagation,
/// including timing, state, and event metadata.
///
/// Attributes:
///     window_open (Epoch): Start time (entry for periods, event time for instantaneous)
///     window_close (Epoch): End time (exit for periods, same as window_open for instantaneous)
///     entry_state (ndarray): State vector at window_open
///     exit_state (ndarray): State vector at window_close
///     value (float): Event function value at detection
///     name (str): Event detector name
///     action (EventAction): Action taken (STOP or CONTINUE)
///     event_type (EventType): Event type (INSTANTANEOUS or PERIOD)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # After propagation with events (requires numerical propagator binding)
///     # events_detected = propagator.event_log()
///     # for event in events_detected:
///     #     print(f"Event '{event.name}' at {event.window_open}")
///     #     print(f"State: {event.entry_state}")
///     #     if event.event_type == bh.EventType.PERIOD:
///     #         print(f"Duration: {event.window_close - event.window_open} seconds")
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "DetectedEvent")]
pub struct PyDetectedEvent {
    pub(crate) event: events::DDetectedEvent,
}

#[pymethods]
impl PyDetectedEvent {
    /// Window open time (entry for periods, event time for instantaneous).
    ///
    /// Returns:
    ///     Epoch: Start time of the event
    #[getter]
    fn window_open(&self) -> PyEpoch {
        PyEpoch {
            obj: self.event.window_open,
        }
    }

    /// Window close time (exit for periods, same as window_open for instantaneous).
    ///
    /// Returns:
    ///     Epoch: End time of the event
    #[getter]
    fn window_close(&self) -> PyEpoch {
        PyEpoch {
            obj: self.event.window_close,
        }
    }

    /// State vector at window_open.
    ///
    /// Returns:
    ///     ndarray: State at event entry
    #[getter]
    fn entry_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        self.event.entry_state.as_slice().to_pyarray(py).to_owned()
    }

    /// State vector at window_close.
    ///
    /// Returns:
    ///     ndarray: State at event exit
    #[getter]
    fn exit_state<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray<f64, Ix1>> {
        self.event.exit_state.as_slice().to_pyarray(py).to_owned()
    }

    /// Event function value at detection.
    ///
    /// Returns:
    ///     float: Event function value
    #[getter]
    fn value(&self) -> f64 {
        self.event.value
    }

    /// Event detector name.
    ///
    /// Returns:
    ///     str: Name of the event detector
    #[getter]
    fn name(&self) -> String {
        self.event.name.clone()
    }

    /// Action taken (STOP or CONTINUE).
    ///
    /// Returns:
    ///     EventAction: Action taken after event detection
    #[getter]
    fn action(&self) -> PyEventAction {
        PyEventAction {
            action: self.event.action,
        }
    }

    /// Event type (INSTANTANEOUS or PERIOD).
    ///
    /// Returns:
    ///     EventType: Type of event
    #[getter]
    fn event_type(&self) -> PyEventType {
        PyEventType {
            event_type: self.event.event_type,
        }
    }

    /// Alias for window_open.
    ///
    /// Returns:
    ///     Epoch: Start time of the event
    fn t_start(&self) -> PyEpoch {
        PyEpoch {
            obj: self.event.t_start(),
        }
    }

    /// Alias for window_close.
    ///
    /// Returns:
    ///     Epoch: End time of the event
    fn t_end(&self) -> PyEpoch {
        PyEpoch {
            obj: self.event.t_end(),
        }
    }

    /// Alias for window_open.
    ///
    /// Returns:
    ///     Epoch: Start time of the event
    fn start_time(&self) -> PyEpoch {
        PyEpoch {
            obj: self.event.start_time(),
        }
    }

    /// Alias for window_close.
    ///
    /// Returns:
    ///     Epoch: End time of the event
    fn end_time(&self) -> PyEpoch {
        PyEpoch {
            obj: self.event.end_time(),
        }
    }

    fn __str__(&self) -> String {
        format!(
            "DetectedEvent('{}', window: [{}, {}])",
            self.event.name, self.event.window_open, self.event.window_close
        )
    }

    fn __repr__(&self) -> String {
        format!(
            "DetectedEvent(name='{}', window_open={}, window_close={}, event_type={:?})",
            self.event.name, self.event.window_open, self.event.window_close, self.event.event_type
        )
    }
}

// ================================
// Event Detectors
// ================================

/// Time-based event detector.
///
/// Triggers when simulation time reaches a target epoch. Useful for scheduled
/// maneuvers or discrete events at known times.
///
/// Args:
///     target_epoch (Epoch): Target time for event detection
///     name (str): Event name for identification
///
/// Returns:
///     TimeEvent: New time event detector
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Create event at specific time
///     target = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
///     event = bh.TimeEvent(target, "Maneuver Start")
///
///     # Make it terminal (stops propagation)
///     event = event.is_terminal()
///
///     # Chain builder methods
///     event = (bh.TimeEvent(target, "Checkpoint")
///              .with_instance(1)
///              .is_terminal())
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "TimeEvent")]
pub struct PyTimeEvent {
    event: Option<events::DTimeEvent>,
}

#[pymethods]
impl PyTimeEvent {
    /// Create a new time event.
    ///
    /// Args:
    ///     target_epoch (Epoch): Target time for event detection
    ///     name (str): Event name for identification
    ///
    /// Returns:
    ///     TimeEvent: New time event detector
    #[new]
    fn new(target_epoch: PyRef<PyEpoch>, name: String) -> PyResult<Self> {
        let event = events::DTimeEvent::new(target_epoch.obj, name);
        Ok(PyTimeEvent { event: Some(event) })
    }

    /// Set instance number for display name.
    ///
    /// Appends instance number to the base name (e.g., "Maneuver" → "Maneuver 1").
    ///
    /// Args:
    ///     instance (int): Instance number to append
    ///
    /// Returns:
    ///     TimeEvent: Self for method chaining
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     event = bh.TimeEvent(epoch, "Burn").with_instance(2)
    ///     # Event name is now "Burn 2"
    ///     ```
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self {
            event: slf.event.take(),
        }
    }

    /// Set event callback.
    ///
    /// The callback is called when the event is detected and can modify the state
    /// and override the terminal action.
    ///
    /// Args:
    ///     callback (callable): Function (epoch, state) -> (Optional[state], EventAction)
    ///
    /// Returns:
    ///     TimeEvent: Self for method chaining
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///     import numpy as np
    ///
    ///     def apply_delta_v(epoch, state):
    ///         '''Apply 10 m/s delta-v in x direction'''
    ///         new_state = state.copy()
    ///         new_state[3] += 10.0  # vx += 10 m/s
    ///         return (new_state, bh.EventAction.CONTINUE)
    ///
    ///     epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     event = bh.TimeEvent(epoch + 1800, "Maneuver").with_callback(apply_delta_v)
    ///     ```
    #[allow(deprecated)]
    fn with_callback(mut slf: PyRefMut<'_, Self>, callback: Py<PyAny>) -> Self {
        let callback_clone = callback.clone_ref(slf.py());

        let rust_callback = Box::new(
            move |t: time::Epoch,
                  state: &DVector<f64>,
                  _params: Option<&DVector<f64>>|
             -> (Option<DVector<f64>>, Option<DVector<f64>>, events::EventAction) {
                Python::with_gil(|py| {
                    // Convert to Python
                    let py_epoch = Py::new(py, PyEpoch { obj: t }).ok();
                    let state_array = state.as_slice().to_pyarray(py).to_owned();

                    if py_epoch.is_none() {
                        return (None, None, events::EventAction::Continue);
                    }

                    // Call Python callback
                    let result = callback_clone.bind(py).call1((py_epoch.unwrap(), state_array));

                    match result {
                        Ok(tuple) => {
                            // Extract (Optional[state], action) tuple
                            let item0 = tuple.get_item(0).ok();
                            let new_state: Option<DVector<f64>> = item0.and_then(|item| {
                                if item.is_none() {
                                    None
                                } else {
                                    pyany_to_f64_array1(&item, None)
                                        .ok()
                                        .map(DVector::from_vec)
                                }
                            });

                            let item1 = tuple.get_item(1).ok();
                            let action: events::EventAction = item1
                                .and_then(|item| item.extract::<PyRef<PyEventAction>>().ok())
                                .map(|a| a.action)
                                .unwrap_or(events::EventAction::Continue);

                            (new_state, None, action)
                        }
                        Err(e) => {
                            eprintln!("Warning: callback failed: {}", e);
                            (None, None, events::EventAction::Continue)
                        }
                    }
                })
            },
        ) as events::DEventCallback;

        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_callback(rust_callback));
        }

        Self {
            event: slf.event.take(),
        }
    }

    /// Mark this event as terminal (stops propagation).
    ///
    /// Returns:
    ///     TimeEvent: Self for method chaining
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///     event = bh.TimeEvent(epoch, "End Condition").is_terminal()
    ///     # Propagation will stop when this event is detected
    ///     ```
    fn is_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.is_terminal());
        }
        Self {
            event: slf.event.take(),
        }
    }
}

/// Threshold event detector with custom value function.
///
/// Monitors a custom value computed by a Python function and detects when it
/// crosses a specified threshold. The value function receives the current epoch
/// and state, and returns a float value to monitor.
///
/// Args:
///     name (str): Event name for identification
///     value_fn (callable): Function (epoch, state) -> float that computes monitored value
///     threshold (float): Threshold value for crossing detection
///     direction (EventDirection): Detection direction (INCREASING, DECREASING, or ANY)
///
/// Returns:
///     ThresholdEvent: New threshold event detector
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     # Custom value function: radial distance
///     def radial_distance(epoch, state):
///         return np.linalg.norm(state[:3])
///
///     event = bh.ThresholdEvent(
///         "Altitude Check",
///         radial_distance,
///         bh.R_EARTH + 500e3,
///         bh.EventDirection.DECREASING
///     )
///
///     # Can chain builder methods
///     event = event.with_tolerances(1e-3, 1e-6).is_terminal()
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "ThresholdEvent")]
pub struct PyThresholdEvent {
    event: Option<events::DThresholdEvent>,
    // Store Python callable for potential re-wrapping if needed
    #[allow(dead_code)]
    value_fn_py: Option<Py<PyAny>>,
}

#[pymethods]
impl PyThresholdEvent {
    /// Create a new threshold event detector.
    ///
    /// Args:
    ///     name (str): Event name for identification
    ///     value_fn (callable): Function (epoch, state) -> float that computes monitored value
    ///     threshold (float): Threshold value for crossing detection
    ///     direction (EventDirection): Detection direction
    ///
    /// Returns:
    ///     ThresholdEvent: New threshold event detector
    #[new]
    #[allow(deprecated)]
    fn new(
        py: Python<'_>,
        name: String,
        value_fn: Py<PyAny>,
        threshold: f64,
        direction: PyRef<PyEventDirection>,
    ) -> PyResult<Self> {
        // Create Rust closure from Python callable
        let value_fn_clone = value_fn.clone_ref(py);
        let rust_value_fn =
            move |t: time::Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| -> f64 {
                Python::with_gil(|py| {
                    // Convert arguments to Python types
                    let py_epoch = match Py::new(py, PyEpoch { obj: t }) {
                        Ok(e) => e,
                        Err(e) => {
                            eprintln!("Warning: Failed to create PyEpoch: {}", e);
                            return 0.0; // Safe default
                        }
                    };

                    let state_array = state.as_slice().to_pyarray(py).to_owned();

                    // Call Python function
                    let result: PyResult<Bound<'_, pyo3::PyAny>> = value_fn_clone
                        .bind(py)
                        .call1((py_epoch, state_array));

                    match result {
                        Ok(val) => val.extract::<f64>().unwrap_or_else(|e| {
                            eprintln!("Warning: value_fn must return float: {}", e);
                            0.0
                        }),
                        Err(e) => {
                            eprintln!("Warning: value_fn call failed: {}", e);
                            0.0
                        }
                    }
                })
            };

        // Create Rust event with closure
        let event = events::DThresholdEvent::new(name, rust_value_fn, threshold, direction.direction);

        Ok(PyThresholdEvent {
            event: Some(event),
            value_fn_py: Some(value_fn),
        })
    }

    /// Set instance number for display name.
    ///
    /// Args:
    ///     instance (int): Instance number to append
    ///
    /// Returns:
    ///     ThresholdEvent: Self for method chaining
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self {
            event: slf.event.take(),
            value_fn_py: slf.value_fn_py.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
        }
    }

    /// Set custom tolerances for event detection.
    ///
    /// Args:
    ///     time_tol (float): Time tolerance in seconds (default: 1e-6)
    ///     value_tol (float): Value tolerance (default: 1e-9)
    ///
    /// Returns:
    ///     ThresholdEvent: Self for method chaining
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     def value_fn(epoch, state):
    ///         return state[0]  # Monitor x position
    ///
    ///     event = (bh.ThresholdEvent("X Crossing", value_fn, 0.0, bh.EventDirection.ANY)
    ///              .with_tolerances(1e-3, 1e-6))  # Looser tolerances
    ///     ```
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self {
            event: slf.event.take(),
            value_fn_py: slf.value_fn_py.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
        }
    }

    /// Set event callback.
    ///
    /// Args:
    ///     callback (callable): Function (epoch, state) -> (Optional[state], EventAction)
    ///
    /// Returns:
    ///     ThresholdEvent: Self for method chaining
    #[allow(deprecated)]
    fn with_callback(mut slf: PyRefMut<'_, Self>, callback: Py<PyAny>) -> Self {
        let callback_clone = callback.clone_ref(slf.py());

        let rust_callback = Box::new(
            move |t: time::Epoch,
                  state: &DVector<f64>,
                  _params: Option<&DVector<f64>>|
             -> (Option<DVector<f64>>, Option<DVector<f64>>, events::EventAction) {
                Python::with_gil(|py| {
                    let py_epoch = Py::new(py, PyEpoch { obj: t }).ok();
                    let state_array = state.as_slice().to_pyarray(py).to_owned();

                    if py_epoch.is_none() {
                        return (None, None, events::EventAction::Continue);
                    }

                    let result = callback_clone.bind(py).call1((py_epoch.unwrap(), state_array));

                    match result {
                        Ok(tuple) => {
                            let item0 = tuple.get_item(0).ok();
                            let new_state: Option<DVector<f64>> = item0.and_then(|item| {
                                if item.is_none() {
                                    None
                                } else {
                                    pyany_to_f64_array1(&item, None)
                                        .ok()
                                        .map(DVector::from_vec)
                                }
                            });

                            let item1 = tuple.get_item(1).ok();
                            let action: events::EventAction = item1
                                .and_then(|item| item.extract::<PyRef<PyEventAction>>().ok())
                                .map(|a| a.action)
                                .unwrap_or(events::EventAction::Continue);

                            (new_state, None, action)
                        }
                        Err(e) => {
                            eprintln!("Warning: callback failed: {}", e);
                            (None, None, events::EventAction::Continue)
                        }
                    }
                })
            },
        ) as events::DEventCallback;

        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_callback(rust_callback));
        }

        Self {
            event: slf.event.take(),
            value_fn_py: slf.value_fn_py.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
        }
    }

    /// Mark this event as terminal (stops propagation).
    ///
    /// Returns:
    ///     ThresholdEvent: Self for method chaining
    fn is_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.is_terminal());
        }
        Self {
            event: slf.event.take(),
            value_fn_py: slf.value_fn_py.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
        }
    }
}

/// Binary event detector with custom condition function.
///
/// Detects boolean condition transitions (edges). The condition function receives
/// the current epoch and state, and returns a boolean. The event fires when the
/// condition transitions according to the specified edge type.
///
/// Args:
///     name (str): Event name for identification
///     condition_fn (callable): Function (epoch, state) -> bool that returns the condition
///     edge (EdgeType): Which edge to detect (RISING_EDGE, FALLING_EDGE, or ANY_EDGE)
///
/// Returns:
///     BinaryEvent: New binary event detector
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Detect eclipse entry (sunlit → shadow)
///     def is_sunlit(epoch, state):
///         # Simplified: check if position is on sunward side
///         pos = state[:3]
///         return pos[0] > 0  # Simple hemisphere check
///
///     event = bh.BinaryEvent(
///         "Eclipse Entry",
///         is_sunlit,
///         bh.EdgeType.FALLING_EDGE  # true → false
///     )
///
///     # Chain builder methods
///     event = event.with_tolerances(1e-3, 1e-6).is_terminal()
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "BinaryEvent")]
pub struct PyBinaryEvent {
    event: Option<events::DBinaryEvent>,
    // Store Python callable for potential re-wrapping if needed
    #[allow(dead_code)]
    condition_fn_py: Option<Py<PyAny>>,
}

#[pymethods]
impl PyBinaryEvent {
    /// Create a new binary event detector.
    ///
    /// Args:
    ///     name (str): Event name for identification
    ///     condition_fn (callable): Function (epoch, state) -> bool that returns the condition
    ///     edge (EdgeType): Which edge to detect
    ///
    /// Returns:
    ///     BinaryEvent: New binary event detector
    #[new]
    #[allow(deprecated)]
    fn new(
        py: Python<'_>,
        name: String,
        condition_fn: Py<PyAny>,
        edge: PyRef<PyEdgeType>,
    ) -> PyResult<Self> {
        // Create Rust closure from Python callable
        let condition_fn_clone = condition_fn.clone_ref(py);
        let rust_condition_fn =
            move |t: time::Epoch, state: &DVector<f64>, _params: Option<&DVector<f64>>| -> bool {
                Python::with_gil(|py| {
                    // Convert arguments to Python types
                    let py_epoch = match Py::new(py, PyEpoch { obj: t }) {
                        Ok(e) => e,
                        Err(e) => {
                            eprintln!("Warning: Failed to create PyEpoch: {}", e);
                            return false; // Safe default
                        }
                    };

                    let state_array = state.as_slice().to_pyarray(py).to_owned();

                    // Call Python function
                    let result: PyResult<Bound<'_, pyo3::PyAny>> = condition_fn_clone
                        .bind(py)
                        .call1((py_epoch, state_array));

                    match result {
                        Ok(val) => val.extract::<bool>().unwrap_or_else(|e| {
                            eprintln!("Warning: condition_fn must return bool: {}", e);
                            false
                        }),
                        Err(e) => {
                            eprintln!("Warning: condition_fn call failed: {}", e);
                            false
                        }
                    }
                })
            };

        // Create Rust event with closure
        let event = events::DBinaryEvent::new(name, rust_condition_fn, edge.edge);

        Ok(PyBinaryEvent {
            event: Some(event),
            condition_fn_py: Some(condition_fn),
        })
    }

    /// Set instance number for display name.
    ///
    /// Args:
    ///     instance (int): Instance number to append
    ///
    /// Returns:
    ///     BinaryEvent: Self for method chaining
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self {
            event: slf.event.take(),
            condition_fn_py: slf.condition_fn_py.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
        }
    }

    /// Set custom tolerances for event detection.
    ///
    /// Args:
    ///     time_tol (float): Time tolerance in seconds (default: 1e-6)
    ///     value_tol (float): Value tolerance (default: 1e-9)
    ///
    /// Returns:
    ///     BinaryEvent: Self for method chaining
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self {
            event: slf.event.take(),
            condition_fn_py: slf.condition_fn_py.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
        }
    }

    /// Set event callback.
    ///
    /// Args:
    ///     callback (callable): Function (epoch, state) -> (Optional[state], EventAction)
    ///
    /// Returns:
    ///     BinaryEvent: Self for method chaining
    #[allow(deprecated)]
    fn with_callback(mut slf: PyRefMut<'_, Self>, callback: Py<PyAny>) -> Self {
        let callback_clone = callback.clone_ref(slf.py());

        let rust_callback = Box::new(
            move |t: time::Epoch,
                  state: &DVector<f64>,
                  _params: Option<&DVector<f64>>|
             -> (Option<DVector<f64>>, Option<DVector<f64>>, events::EventAction) {
                Python::with_gil(|py| {
                    let py_epoch = Py::new(py, PyEpoch { obj: t }).ok();
                    let state_array = state.as_slice().to_pyarray(py).to_owned();

                    if py_epoch.is_none() {
                        return (None, None, events::EventAction::Continue);
                    }

                    let result = callback_clone.bind(py).call1((py_epoch.unwrap(), state_array));

                    match result {
                        Ok(tuple) => {
                            let item0 = tuple.get_item(0).ok();
                            let new_state: Option<DVector<f64>> = item0.and_then(|item| {
                                if item.is_none() {
                                    None
                                } else {
                                    pyany_to_f64_array1(&item, None)
                                        .ok()
                                        .map(DVector::from_vec)
                                }
                            });

                            let item1 = tuple.get_item(1).ok();
                            let action: events::EventAction = item1
                                .and_then(|item| item.extract::<PyRef<PyEventAction>>().ok())
                                .map(|a| a.action)
                                .unwrap_or(events::EventAction::Continue);

                            (new_state, None, action)
                        }
                        Err(e) => {
                            eprintln!("Warning: callback failed: {}", e);
                            (None, None, events::EventAction::Continue)
                        }
                    }
                })
            },
        ) as events::DEventCallback;

        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_callback(rust_callback));
        }

        Self {
            event: slf.event.take(),
            condition_fn_py: slf.condition_fn_py.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
        }
    }

    /// Mark this event as terminal (stops propagation).
    ///
    /// Returns:
    ///     BinaryEvent: Self for method chaining
    fn is_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.is_terminal());
        }
        Self {
            event: slf.event.take(),
            condition_fn_py: slf.condition_fn_py.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
        }
    }
}

/// Altitude-based event detector (convenience wrapper).
///
/// Detects when geodetic altitude crosses a threshold. This is a convenience wrapper
/// that automatically handles ECI → ECEF → geodetic transformations to compute altitude
/// above the WGS84 ellipsoid.
///
/// Note: Requires EOP (Earth Orientation Parameters) to be initialized for accurate
/// transformations. Use `bh.initialize_eop()` or set a custom provider.
///
/// Args:
///     threshold_altitude (float): Geodetic altitude threshold in meters above WGS84
///     name (str): Event name for identification
///     direction (EventDirection): Detection direction (INCREASING, DECREASING, or ANY)
///
/// Returns:
///     AltitudeEvent: New altitude event detector
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Requires EOP initialization
///     bh.initialize_eop()
///
///     # Detect when altitude drops below 300 km
///     event = bh.AltitudeEvent(
///         300e3,  # 300 km in meters
///         "Low Altitude Warning",
///         bh.EventDirection.DECREASING
///     )
///
///     # Mark as terminal to stop propagation
///     event = event.is_terminal()
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "AltitudeEvent")]
pub struct PyAltitudeEvent {
    event: Option<events::DAltitudeEvent>,
}

#[pymethods]
impl PyAltitudeEvent {
    /// Create a new altitude event detector.
    ///
    /// Args:
    ///     threshold_altitude (float): Geodetic altitude threshold in meters above WGS84
    ///     name (str): Event name for identification
    ///     direction (EventDirection): Detection direction
    ///
    /// Returns:
    ///     AltitudeEvent: New altitude event detector
    #[new]
    fn new(
        threshold_altitude: f64,
        name: String,
        direction: PyRef<PyEventDirection>,
    ) -> PyResult<Self> {
        let event = events::DAltitudeEvent::new(threshold_altitude, name, direction.direction);
        Ok(PyAltitudeEvent { event: Some(event) })
    }

    /// Set instance number for display name.
    ///
    /// Args:
    ///     instance (int): Instance number to append
    ///
    /// Returns:
    ///     AltitudeEvent: Self for method chaining
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self { event: slf.event.take() }
    }

    /// Set custom tolerances for event detection.
    ///
    /// Args:
    ///     time_tol (float): Time tolerance in seconds (default: 1e-6)
    ///     value_tol (float): Value tolerance (default: 1e-9)
    ///
    /// Returns:
    ///     AltitudeEvent: Self for method chaining
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self { event: slf.event.take() }
    }

    /// Set event callback.
    ///
    /// Args:
    ///     callback (callable): Function (epoch, state) -> (Optional[state], EventAction)
    ///
    /// Returns:
    ///     AltitudeEvent: Self for method chaining
    #[allow(deprecated)]
    fn with_callback(mut slf: PyRefMut<'_, Self>, callback: Py<PyAny>) -> Self {
        let callback_clone = callback.clone_ref(slf.py());

        let rust_callback = Box::new(
            move |t: time::Epoch,
                  state: &DVector<f64>,
                  _params: Option<&DVector<f64>>|
             -> (Option<DVector<f64>>, Option<DVector<f64>>, events::EventAction) {
                Python::with_gil(|py| {
                    let py_epoch = Py::new(py, PyEpoch { obj: t }).ok();
                    let state_array = state.as_slice().to_pyarray(py).to_owned();

                    if py_epoch.is_none() {
                        return (None, None, events::EventAction::Continue);
                    }

                    let result = callback_clone.bind(py).call1((py_epoch.unwrap(), state_array));

                    match result {
                        Ok(tuple) => {
                            let new_state: Option<DVector<f64>> = tuple
                                .get_item(0)
                                .ok()
                                .and_then(|item| {
                                    if item.is_none() {
                                        None
                                    } else {
                                        pyany_to_f64_array1(&item, None)
                                            .ok()
                                            .map(DVector::from_vec)
                                    }
                                });

                            let action: events::EventAction = tuple
                                .get_item(1)
                                .ok()
                                .and_then(|item| item.extract::<PyRef<PyEventAction>>().ok())
                                .map(|a| a.action)
                                .unwrap_or(events::EventAction::Continue);

                            (new_state, None, action)
                        }
                        Err(e) => {
                            eprintln!("Warning: callback failed: {}", e);
                            (None, None, events::EventAction::Continue)
                        }
                    }
                })
            },
        ) as events::DEventCallback;

        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_callback(rust_callback));
        }

        Self { event: slf.event.take() }
    }

    /// Mark this event as terminal (stops propagation).
    ///
    /// Returns:
    ///     AltitudeEvent: Self for method chaining
    fn is_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.is_terminal());
        }
        Self { event: slf.event.take() }
    }
}
