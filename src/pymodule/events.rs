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
/// Can be set as the default action via `.set_terminal()` or returned from a callback
/// to override the default.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Terminal event (stops propagation)
///     event = bh.TimeEvent(target_epoch, "Maneuver").set_terminal()
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
    #[pyo3(name = "WINDOW")]
    fn period() -> Self {
        PyEventType {
            event_type: events::EventType::Window,
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
///     #     if event.event_type == bh.EventType.WINDOW:
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
/// Example:
///     ```python
///     import brahe as bh
///
///     target = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
///     event = bh.TimeEvent(target, "Maneuver Start")
///     event = event.set_terminal()
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "TimeEvent")]
pub struct PyTimeEvent {
    event: Option<events::DTimeEvent>,
    // Store construction parameters for D→S conversion (SGPPropagator support)
    target_time: time::Epoch,
    base_name: String,
    instance: Option<usize>,
    is_terminal: bool,
    time_tol: f64,
    step_reduction_factor: f64,
    // Store Python callback separately for S-type event creation
    py_callback: Option<Py<PyAny>>,
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
    #[pyo3(signature = (target_epoch, name))]
    fn new(target_epoch: PyRef<PyEpoch>, name: String) -> PyResult<Self> {
        let event = events::DTimeEvent::new(target_epoch.obj, name.clone());
        Ok(PyTimeEvent {
            event: Some(event),
            target_time: target_epoch.obj,
            base_name: name,
            instance: None,
            is_terminal: false,
            time_tol: 1e-6,
            step_reduction_factor: 0.2,
            py_callback: None,
        })
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
            target_time: slf.target_time,
            base_name: slf.base_name.clone(),
            instance: Some(instance),
            is_terminal: slf.is_terminal,
            time_tol: slf.time_tol,
            step_reduction_factor: slf.step_reduction_factor,
            py_callback: slf.py_callback.take(),
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

        // Store the Python callback for S-type event creation
        let py_callback_stored = callback.clone_ref(slf.py());

        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_callback(rust_callback));
        }

        Self {
            event: slf.event.take(),
            target_time: slf.target_time,
            base_name: slf.base_name.clone(),
            instance: slf.instance,
            is_terminal: slf.is_terminal,
            time_tol: slf.time_tol,
            step_reduction_factor: slf.step_reduction_factor,
            py_callback: Some(py_callback_stored),
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
    ///     event = bh.TimeEvent(epoch, "End Condition").set_terminal()
    ///     # Propagation will stop when this event is detected
    ///     ```
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self {
            event: slf.event.take(),
            target_time: slf.target_time,
            base_name: slf.base_name.clone(),
            instance: slf.instance,
            is_terminal: true,
            time_tol: slf.time_tol,
            step_reduction_factor: slf.step_reduction_factor,
            py_callback: slf.py_callback.take(),
        }
    }

    /// Set custom time tolerance for event detection.
    ///
    /// Controls the precision of the bisection search algorithm. Smaller values
    /// result in more precise event time detection at the cost of more iterations.
    ///
    /// Args:
    ///     time_tol (float): Time tolerance in seconds (default: 1e-6)
    ///
    /// Returns:
    ///     TimeEvent: Self for method chaining
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
    ///
    ///     # Use tighter tolerance for precise timing
    ///     event = bh.TimeEvent(epoch, "Precise Maneuver").with_time_tolerance(1e-9)
    ///
    ///     # Chain multiple builder methods
    ///     event = (bh.TimeEvent(epoch, "Complex")
    ///              .with_time_tolerance(1e-3)
    ///              .with_instance(1)
    ///              .set_terminal())
    ///     ```
    fn with_time_tolerance(mut slf: PyRefMut<'_, Self>, time_tol: f64) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_time_tolerance(time_tol));
        }
        Self {
            event: slf.event.take(),
            target_time: slf.target_time,
            base_name: slf.base_name.clone(),
            instance: slf.instance,
            is_terminal: slf.is_terminal,
            time_tol,
            step_reduction_factor: slf.step_reduction_factor,
            py_callback: slf.py_callback.take(),
        }
    }
}

impl PyTimeEvent {
    /// Create an S-type event from this Python event (for SGPPropagator)
    pub fn to_s_event(&self) -> events::STimeEvent<6, 0> {
        let mut event = events::STimeEvent::<6, 0>::new(self.target_time, self.base_name.clone())
            .with_time_tolerance(self.time_tol)
            .with_step_reduction_factor(self.step_reduction_factor);

        if let Some(instance) = self.instance {
            event = event.with_instance(instance);
        }

        if self.is_terminal {
            event = event.set_terminal();
        }

        // Note: py_callback is handled separately in propagators.rs since it needs
        // to create an S-type callback wrapper

        event
    }

    /// Get the stored Python callback, if any
    pub fn get_py_callback(&self) -> Option<&Py<PyAny>> {
        self.py_callback.as_ref()
    }

    /// Check if this event has been consumed (used by add_event_detector)
    pub fn is_consumed(&self) -> bool {
        self.event.is_none()
    }
}

/// Value event detector with custom value function.
///
/// Monitors a custom value computed by a Python function and detects when it
/// crosses a specified target value. The value function receives the current epoch
/// and state, and returns a float value to monitor.
///
/// Args:
///     name (str): Event name for identification
///     value_fn (callable): Function (epoch, state) -> float that computes monitored value
///     target_value (float): Target value for crossing detection
///     direction (EventDirection): Detection direction (INCREASING, DECREASING, or ANY)
///
/// Example:
///     ```python
///     import brahe as bh
///     import numpy as np
///
///     def radial_distance(epoch, state):
///         return np.linalg.norm(state[:3])
///
///     event = bh.ValueEvent(
///         "Altitude Check",
///         radial_distance,
///         bh.R_EARTH + 500e3,
///         bh.EventDirection.DECREASING
///     )
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "ValueEvent")]
pub struct PyValueEvent {
    event: Option<events::DValueEvent>,
    // Store construction parameters for D→S conversion (SGPPropagator support)
    value_fn_py: Option<Py<PyAny>>,
    base_name: String,
    target_value: f64,
    direction: events::EventDirection,
    instance: Option<usize>,
    is_terminal: bool,
    time_tol: f64,
    value_tol: f64,
    // Store Python callback separately for S-type event creation
    py_callback: Option<Py<PyAny>>,
}

#[pymethods]
impl PyValueEvent {
    /// Create a new value event detector.
    ///
    /// Args:
    ///     name (str): Event name for identification
    ///     value_fn (callable): Function (epoch, state) -> float that computes monitored value
    ///     target_value (float): Target value for crossing detection
    ///     direction (EventDirection): Detection direction
    ///
    /// Returns:
    ///     ValueEvent: New value event detector
    #[new]
    #[pyo3(signature = (name, value_fn, target_value, direction))]
    #[allow(deprecated)]
    fn new(
        py: Python<'_>,
        name: String,
        value_fn: Py<PyAny>,
        target_value: f64,
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
        let event = events::DValueEvent::new(name.clone(), rust_value_fn, target_value, direction.direction);

        Ok(PyValueEvent {
            event: Some(event),
            value_fn_py: Some(value_fn),
            base_name: name,
            target_value,
            direction: direction.direction,
            instance: None,
            is_terminal: false,
            time_tol: 1e-6,
            value_tol: 1e-9,
            py_callback: None,
        })
    }

    /// Set instance number for display name.
    ///
    /// Args:
    ///     instance (int): Instance number to append
    ///
    /// Returns:
    ///     ValueEvent: Self for method chaining
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self {
            event: slf.event.take(),
            value_fn_py: slf.value_fn_py.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
            base_name: slf.base_name.clone(),
            target_value: slf.target_value,
            direction: slf.direction,
            instance: Some(instance),
            is_terminal: slf.is_terminal,
            time_tol: slf.time_tol,
            value_tol: slf.value_tol,
            py_callback: slf.py_callback.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
        }
    }

    /// Set custom tolerances for event detection.
    ///
    /// Args:
    ///     time_tol (float): Time tolerance in seconds (default: 1e-6)
    ///     value_tol (float): Value tolerance (default: 1e-9)
    ///
    /// Returns:
    ///     ValueEvent: Self for method chaining
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     def value_fn(epoch, state):
    ///         return state[0]  # Monitor x position
    ///
    ///     event = (bh.ValueEvent("X Crossing", value_fn, 0.0, bh.EventDirection.ANY)
    ///              .with_tolerances(1e-3, 1e-6))  # Looser tolerances
    ///     ```
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self {
            event: slf.event.take(),
            value_fn_py: slf.value_fn_py.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
            base_name: slf.base_name.clone(),
            target_value: slf.target_value,
            direction: slf.direction,
            instance: slf.instance,
            is_terminal: slf.is_terminal,
            time_tol,
            value_tol,
            py_callback: slf.py_callback.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
        }
    }

    /// Set event callback.
    ///
    /// Args:
    ///     callback (callable): Function (epoch, state) -> (Optional[state], EventAction)
    ///
    /// Returns:
    ///     ValueEvent: Self for method chaining
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

        // Store the Python callback for S-type event creation
        let py_callback_stored = callback.clone_ref(slf.py());

        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_callback(rust_callback));
        }

        Self {
            event: slf.event.take(),
            value_fn_py: slf.value_fn_py.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
            base_name: slf.base_name.clone(),
            target_value: slf.target_value,
            direction: slf.direction,
            instance: slf.instance,
            is_terminal: slf.is_terminal,
            time_tol: slf.time_tol,
            value_tol: slf.value_tol,
            py_callback: Some(py_callback_stored),
        }
    }

    /// Mark this event as terminal (stops propagation).
    ///
    /// Returns:
    ///     ValueEvent: Self for method chaining
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self {
            event: slf.event.take(),
            value_fn_py: slf.value_fn_py.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
            base_name: slf.base_name.clone(),
            target_value: slf.target_value,
            direction: slf.direction,
            instance: slf.instance,
            is_terminal: true,
            time_tol: slf.time_tol,
            value_tol: slf.value_tol,
            py_callback: slf.py_callback.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
        }
    }
}

impl PyValueEvent {
    /// Get the stored Python value function
    pub fn get_value_fn_py(&self) -> Option<&Py<PyAny>> {
        self.value_fn_py.as_ref()
    }

    /// Get the stored Python callback, if any
    pub fn get_py_callback(&self) -> Option<&Py<PyAny>> {
        self.py_callback.as_ref()
    }

    /// Get the base name for this event
    pub fn get_base_name(&self) -> &str {
        &self.base_name
    }

    /// Get the target value
    pub fn get_target_value(&self) -> f64 {
        self.target_value
    }

    /// Get the detection direction
    pub fn get_direction(&self) -> events::EventDirection {
        self.direction
    }

    /// Get the instance number, if set
    pub fn get_instance(&self) -> Option<usize> {
        self.instance
    }

    /// Check if this event is terminal
    pub fn is_terminal(&self) -> bool {
        self.is_terminal
    }

    /// Get time tolerance
    pub fn get_time_tolerance(&self) -> f64 {
        self.time_tol
    }

    /// Get value tolerance
    pub fn get_value_tolerance(&self) -> f64 {
        self.value_tol
    }

    /// Check if this event has been consumed (used by add_event_detector)
    pub fn is_consumed(&self) -> bool {
        self.event.is_none()
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
/// Example:
///     ```python
///     import brahe as bh
///
///     def is_sunlit(epoch, state):
///         pos = state[:3]
///         return pos[0] > 0
///
///     event = bh.BinaryEvent(
///         "Eclipse Entry",
///         is_sunlit,
///         bh.EdgeType.FALLING_EDGE
///     )
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "BinaryEvent")]
pub struct PyBinaryEvent {
    event: Option<events::DBinaryEvent>,
    // Store construction parameters for D→S conversion (SGPPropagator support)
    condition_fn_py: Option<Py<PyAny>>,
    base_name: String,
    edge: events::EdgeType,
    instance: Option<usize>,
    is_terminal: bool,
    time_tol: f64,
    value_tol: f64,
    // Store Python callback separately for S-type event creation
    py_callback: Option<Py<PyAny>>,
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
    #[pyo3(signature = (name, condition_fn, edge))]
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
        let event = events::DBinaryEvent::new(name.clone(), rust_condition_fn, edge.edge);

        Ok(PyBinaryEvent {
            event: Some(event),
            condition_fn_py: Some(condition_fn),
            base_name: name,
            edge: edge.edge,
            instance: None,
            is_terminal: false,
            time_tol: 1e-6,
            value_tol: 1e-9,
            py_callback: None,
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
            base_name: slf.base_name.clone(),
            edge: slf.edge,
            instance: Some(instance),
            is_terminal: slf.is_terminal,
            time_tol: slf.time_tol,
            value_tol: slf.value_tol,
            py_callback: slf.py_callback.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
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
            base_name: slf.base_name.clone(),
            edge: slf.edge,
            instance: slf.instance,
            is_terminal: slf.is_terminal,
            time_tol,
            value_tol,
            py_callback: slf.py_callback.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
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

        // Store the Python callback for S-type event creation
        let py_callback_stored = callback.clone_ref(slf.py());

        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_callback(rust_callback));
        }

        Self {
            event: slf.event.take(),
            condition_fn_py: slf.condition_fn_py.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
            base_name: slf.base_name.clone(),
            edge: slf.edge,
            instance: slf.instance,
            is_terminal: slf.is_terminal,
            time_tol: slf.time_tol,
            value_tol: slf.value_tol,
            py_callback: Some(py_callback_stored),
        }
    }

    /// Mark this event as terminal (stops propagation).
    ///
    /// Returns:
    ///     BinaryEvent: Self for method chaining
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self {
            event: slf.event.take(),
            condition_fn_py: slf.condition_fn_py.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
            base_name: slf.base_name.clone(),
            edge: slf.edge,
            instance: slf.instance,
            is_terminal: true,
            time_tol: slf.time_tol,
            value_tol: slf.value_tol,
            py_callback: slf.py_callback.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
        }
    }
}

impl PyBinaryEvent {
    /// Get the stored Python condition function
    pub fn get_condition_fn_py(&self) -> Option<&Py<PyAny>> {
        self.condition_fn_py.as_ref()
    }

    /// Get the stored Python callback, if any
    pub fn get_py_callback(&self) -> Option<&Py<PyAny>> {
        self.py_callback.as_ref()
    }

    /// Get the base name for this event
    pub fn get_base_name(&self) -> &str {
        &self.base_name
    }

    /// Get the edge type
    pub fn get_edge(&self) -> events::EdgeType {
        self.edge
    }

    /// Get the instance number, if set
    pub fn get_instance(&self) -> Option<usize> {
        self.instance
    }

    /// Check if this event is terminal
    pub fn is_terminal(&self) -> bool {
        self.is_terminal
    }

    /// Get time tolerance
    pub fn get_time_tolerance(&self) -> f64 {
        self.time_tol
    }

    /// Get value tolerance
    pub fn get_value_tolerance(&self) -> f64 {
        self.value_tol
    }

    /// Check if this event has been consumed (used by add_event_detector)
    pub fn is_consumed(&self) -> bool {
        self.event.is_none()
    }
}

// ================================
// Event Query Builder
// ================================

/// Event query builder for filtering detected events.
///
/// Provides chainable filter methods for querying events with an idiomatic Python interface.
/// Filters are applied lazily and can be combined in any order.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # After propagation with events
///     # Get events from detector 0 within a time range
///     events = prop.query_events() \
///         .by_detector_index(0) \
///         .in_time_range(start, end) \
///         .collect()
///
///     # Count events by name pattern
///     count = prop.query_events() \
///         .by_name_contains("Altitude") \
///         .count()
///
///     # Get first terminal event
///     event = prop.query_events() \
///         .by_action(bh.EventAction.STOP) \
///         .first()
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "EventQuery")]
pub struct PyEventQuery {
    /// Internal storage of events being filtered
    events: Vec<events::DDetectedEvent>,
}

#[pymethods]
impl PyEventQuery {
    /// Filter by detector index.
    ///
    /// Returns events detected by the specified detector.
    ///
    /// Args:
    ///     index (int): Detector index (0-based, corresponding to add order)
    ///
    /// Returns:
    ///     EventQuery: New query with filter applied (for chaining)
    ///
    /// Example:
    ///     ```python
    ///     events = prop.query_events().by_detector_index(0).collect()
    ///     ```
    #[pyo3(text_signature = "(index)")]
    fn by_detector_index(&self, index: usize) -> Self {
        PyEventQuery {
            events: self
                .events
                .iter()
                .filter(|e| e.detector_index == index)
                .cloned()
                .collect(),
        }
    }

    /// Filter by exact detector name.
    ///
    /// Returns events where the detector name exactly matches the given string.
    ///
    /// Args:
    ///     name (str): Exact name to match
    ///
    /// Returns:
    ///     EventQuery: New query with filter applied (for chaining)
    ///
    /// Example:
    ///     ```python
    ///     events = prop.query_events().by_name_exact("Altitude Event").collect()
    ///     ```
    #[pyo3(text_signature = "(name)")]
    fn by_name_exact(&self, name: &str) -> Self {
        PyEventQuery {
            events: self
                .events
                .iter()
                .filter(|e| e.name == name)
                .cloned()
                .collect(),
        }
    }

    /// Filter by detector name substring.
    ///
    /// Returns events where the detector name contains the given substring.
    ///
    /// Args:
    ///     substring (str): Substring to search for in event names
    ///
    /// Returns:
    ///     EventQuery: New query with filter applied (for chaining)
    ///
    /// Example:
    ///     ```python
    ///     events = prop.query_events().by_name_contains("Altitude").collect()
    ///     ```
    #[pyo3(text_signature = "(substring)")]
    fn by_name_contains(&self, substring: &str) -> Self {
        PyEventQuery {
            events: self
                .events
                .iter()
                .filter(|e| e.name.contains(substring))
                .cloned()
                .collect(),
        }
    }

    /// Filter by time range (inclusive).
    ///
    /// Returns events that occurred within the specified time range.
    ///
    /// Args:
    ///     start (Epoch): Start of time range (inclusive)
    ///     end (Epoch): End of time range (inclusive)
    ///
    /// Returns:
    ///     EventQuery: New query with filter applied (for chaining)
    ///
    /// Example:
    ///     ```python
    ///     events = prop.query_events().in_time_range(start_epoch, end_epoch).collect()
    ///     ```
    #[pyo3(text_signature = "(start, end)")]
    fn in_time_range(&self, start: &PyEpoch, end: &PyEpoch) -> Self {
        PyEventQuery {
            events: self
                .events
                .iter()
                .filter(|e| e.window_open >= start.obj && e.window_open <= end.obj)
                .cloned()
                .collect(),
        }
    }

    /// Filter events after epoch (inclusive).
    ///
    /// Returns events that occurred at or after the specified epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Epoch value (inclusive)
    ///
    /// Returns:
    ///     EventQuery: New query with filter applied (for chaining)
    ///
    /// Example:
    ///     ```python
    ///     events = prop.query_events().after(cutoff_epoch).collect()
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    fn after(&self, epoch: &PyEpoch) -> Self {
        PyEventQuery {
            events: self
                .events
                .iter()
                .filter(|e| e.window_open >= epoch.obj)
                .cloned()
                .collect(),
        }
    }

    /// Filter events before epoch (inclusive).
    ///
    /// Returns events that occurred at or before the specified epoch.
    ///
    /// Args:
    ///     epoch (Epoch): Epoch value (inclusive)
    ///
    /// Returns:
    ///     EventQuery: New query with filter applied (for chaining)
    ///
    /// Example:
    ///     ```python
    ///     events = prop.query_events().before(cutoff_epoch).collect()
    ///     ```
    #[pyo3(text_signature = "(epoch)")]
    fn before(&self, epoch: &PyEpoch) -> Self {
        PyEventQuery {
            events: self
                .events
                .iter()
                .filter(|e| e.window_open <= epoch.obj)
                .cloned()
                .collect(),
        }
    }

    /// Filter by event type.
    ///
    /// Returns events of the specified type.
    ///
    /// Args:
    ///     event_type (EventType): Event type to filter by (INSTANTANEOUS or PERIOD)
    ///
    /// Returns:
    ///     EventQuery: New query with filter applied (for chaining)
    ///
    /// Example:
    ///     ```python
    ///     events = prop.query_events().by_event_type(bh.EventType.WINDOW).collect()
    ///     ```
    #[pyo3(text_signature = "(event_type)")]
    fn by_event_type(&self, event_type: &PyEventType) -> Self {
        PyEventQuery {
            events: self
                .events
                .iter()
                .filter(|e| e.event_type == event_type.event_type)
                .cloned()
                .collect(),
        }
    }

    /// Filter by event action.
    ///
    /// Returns events with the specified action.
    ///
    /// Args:
    ///     action (EventAction): Event action to filter by (STOP or CONTINUE)
    ///
    /// Returns:
    ///     EventQuery: New query with filter applied (for chaining)
    ///
    /// Example:
    ///     ```python
    ///     events = prop.query_events().by_action(bh.EventAction.STOP).collect()
    ///     ```
    #[pyo3(text_signature = "(action)")]
    fn by_action(&self, action: &PyEventAction) -> Self {
        PyEventQuery {
            events: self
                .events
                .iter()
                .filter(|e| e.action == action.action)
                .cloned()
                .collect(),
        }
    }

    /// Collect filtered events into a list.
    ///
    /// Returns:
    ///     list[DetectedEvent]: List of events matching all applied filters
    ///
    /// Example:
    ///     ```python
    ///     events = prop.query_events().by_detector_index(0).collect()
    ///     for event in events:
    ///         print(f"Event: {event.name}")
    ///     ```
    #[pyo3(text_signature = "()")]
    fn collect(&self) -> Vec<PyDetectedEvent> {
        self.events
            .iter()
            .map(|e| PyDetectedEvent { event: e.clone() })
            .collect()
    }

    /// Count filtered events.
    ///
    /// Returns:
    ///     int: Number of events matching all applied filters
    ///
    /// Example:
    ///     ```python
    ///     count = prop.query_events().by_name_contains("Altitude").count()
    ///     ```
    #[pyo3(text_signature = "()")]
    fn count(&self) -> usize {
        self.events.len()
    }

    /// Get the first matching event, if any.
    ///
    /// Returns:
    ///     DetectedEvent or None: First event matching all filters, or None if empty
    ///
    /// Example:
    ///     ```python
    ///     event = prop.query_events().by_action(bh.EventAction.STOP).first()
    ///     if event:
    ///         print(f"First terminal event: {event.name}")
    ///     ```
    #[pyo3(text_signature = "()")]
    fn first(&self) -> Option<PyDetectedEvent> {
        self.events.first().map(|e| PyDetectedEvent { event: e.clone() })
    }

    /// Get the last matching event, if any.
    ///
    /// Returns:
    ///     DetectedEvent or None: Last event matching all filters, or None if empty
    ///
    /// Example:
    ///     ```python
    ///     event = prop.query_events().by_detector_index(0).last()
    ///     ```
    #[pyo3(text_signature = "()")]
    fn last(&self) -> Option<PyDetectedEvent> {
        self.events.last().map(|e| PyDetectedEvent { event: e.clone() })
    }

    /// Check if any events match the filters.
    ///
    /// Returns:
    ///     bool: True if at least one event matches all applied filters
    ///
    /// Example:
    ///     ```python
    ///     if prop.query_events().by_action(bh.EventAction.STOP).any():
    ///         print("Found terminal events")
    ///     ```
    #[pyo3(name = "any")]
    #[pyo3(text_signature = "()")]
    fn any_matches(&self) -> bool {
        !self.events.is_empty()
    }

    /// Check if the query is empty.
    ///
    /// Returns:
    ///     bool: True if no events match all applied filters
    #[pyo3(text_signature = "()")]
    fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyEventQueryIterator {
        PyEventQueryIterator {
            events: slf.events.clone(),
            index: 0,
        }
    }

    fn __len__(&self) -> usize {
        self.events.len()
    }

    fn __repr__(&self) -> String {
        format!("EventQuery({} events)", self.events.len())
    }
}

impl PyEventQuery {
    /// Create a new EventQuery from a list of events (internal use)
    pub(crate) fn new(events: Vec<events::DDetectedEvent>) -> Self {
        PyEventQuery { events }
    }
}

/// Iterator for EventQuery
#[pyclass(module = "brahe._brahe")]
pub struct PyEventQueryIterator {
    events: Vec<events::DDetectedEvent>,
    index: usize,
}

#[pymethods]
impl PyEventQueryIterator {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(mut slf: PyRefMut<'_, Self>) -> Option<PyDetectedEvent> {
        if slf.index < slf.events.len() {
            let event = slf.events[slf.index].clone();
            slf.index += 1;
            Some(PyDetectedEvent { event })
        } else {
            None
        }
    }
}

// ================================
// Event Detectors
// ================================

/// Altitude-based event detector (convenience wrapper).
///
/// Detects when geodetic altitude crosses a value. This is a convenience wrapper
/// that automatically handles ECI → ECEF → geodetic transformations to compute altitude
/// above the WGS84 ellipsoid.
///
/// Args:
///     value_altitude (float): Geodetic altitude value in meters above WGS84
///     name (str): Event name for identification
///     direction (EventDirection): Detection direction (INCREASING, DECREASING, or ANY)
///
/// Note:
///     Requires EOP (Earth Orientation Parameters) to be initialized for accurate
///     transformations. Use `bh.initialize_eop()` or set a custom provider.
///
/// Example:
///     ```python
///     import brahe as bh
///
///     bh.initialize_eop()
///
///     event = bh.AltitudeEvent(
///         300e3,
///         "Low Altitude Warning",
///         bh.EventDirection.DECREASING
///     )
///     event = event.set_terminal()
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "AltitudeEvent")]
pub struct PyAltitudeEvent {
    event: Option<events::DAltitudeEvent>,
    // Store construction parameters for D→S conversion (SGPPropagator support)
    target_altitude: f64,
    base_name: String,
    direction: events::EventDirection,
    instance: Option<usize>,
    is_terminal: bool,
    time_tol: f64,
    value_tol: f64,
    // Store Python callback separately for S-type event creation
    py_callback: Option<Py<PyAny>>,
}

#[pymethods]
impl PyAltitudeEvent {
    /// Create a new altitude event detector.
    ///
    /// Args:
    ///     value_altitude (float): Geodetic altitude value in meters above WGS84
    ///     name (str): Event name for identification
    ///     direction (EventDirection): Detection direction
    ///
    /// Returns:
    ///     AltitudeEvent: New altitude event detector
    #[new]
    #[pyo3(signature = (value_altitude, name, direction))]
    fn new(
        value_altitude: f64,
        name: String,
        direction: PyRef<PyEventDirection>,
    ) -> PyResult<Self> {
        let event = events::DAltitudeEvent::new(value_altitude, name.clone(), direction.direction);
        Ok(PyAltitudeEvent {
            event: Some(event),
            target_altitude: value_altitude,
            base_name: name,
            direction: direction.direction,
            instance: None,
            is_terminal: false,
            time_tol: 1e-6,
            value_tol: 1e-9,
            py_callback: None,
        })
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
        Self {
            event: slf.event.take(),
            target_altitude: slf.target_altitude,
            base_name: slf.base_name.clone(),
            direction: slf.direction,
            instance: Some(instance),
            is_terminal: slf.is_terminal,
            time_tol: slf.time_tol,
            value_tol: slf.value_tol,
            py_callback: slf.py_callback.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
        }
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
        Self {
            event: slf.event.take(),
            target_altitude: slf.target_altitude,
            base_name: slf.base_name.clone(),
            direction: slf.direction,
            instance: slf.instance,
            is_terminal: slf.is_terminal,
            time_tol,
            value_tol,
            py_callback: slf.py_callback.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
        }
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

        // Store the Python callback for S-type event creation
        let py_callback_stored = callback.clone_ref(slf.py());

        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_callback(rust_callback));
        }

        Self {
            event: slf.event.take(),
            target_altitude: slf.target_altitude,
            base_name: slf.base_name.clone(),
            direction: slf.direction,
            instance: slf.instance,
            is_terminal: slf.is_terminal,
            time_tol: slf.time_tol,
            value_tol: slf.value_tol,
            py_callback: Some(py_callback_stored),
        }
    }

    /// Mark this event as terminal (stops propagation).
    ///
    /// Returns:
    ///     AltitudeEvent: Self for method chaining
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self {
            event: slf.event.take(),
            target_altitude: slf.target_altitude,
            base_name: slf.base_name.clone(),
            direction: slf.direction,
            instance: slf.instance,
            is_terminal: true,
            time_tol: slf.time_tol,
            value_tol: slf.value_tol,
            py_callback: slf.py_callback.as_ref().map(|py_obj| py_obj.clone_ref(slf.py())),
        }
    }
}

impl PyAltitudeEvent {
    /// Get the stored Python callback, if any
    pub fn get_py_callback(&self) -> Option<&Py<PyAny>> {
        self.py_callback.as_ref()
    }

    /// Get the base name for this event
    pub fn get_base_name(&self) -> &str {
        &self.base_name
    }

    /// Get the target altitude
    pub fn get_target_altitude(&self) -> f64 {
        self.target_altitude
    }

    /// Get the detection direction
    pub fn get_direction(&self) -> events::EventDirection {
        self.direction
    }

    /// Get the instance number, if set
    pub fn get_instance(&self) -> Option<usize> {
        self.instance
    }

    /// Check if this event is terminal
    pub fn is_terminal(&self) -> bool {
        self.is_terminal
    }

    /// Get time tolerance
    pub fn get_time_tolerance(&self) -> f64 {
        self.time_tol
    }

    /// Get value tolerance
    pub fn get_value_tolerance(&self) -> f64 {
        self.value_tol
    }

    /// Check if this event has been consumed (used by add_event_detector)
    pub fn is_consumed(&self) -> bool {
        self.event.is_none()
    }
}

// ================================
// Orbital Element Events
// ================================

/// Semi-major axis event detector.
///
/// Detects when orbital semi-major axis crosses a value value.
///
/// Args:
///     value (float): Semi-major axis value in meters
///     name (str): Event name for identification
///     direction (EventDirection): Detection direction
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Detect when semi-major axis drops below GEO altitude
///     event = bh.SemiMajorAxisEvent(
///         bh.R_EARTH + 35786e3,
///         "GEO value",
///         bh.EventDirection.DECREASING
///     )
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "SemiMajorAxisEvent")]
pub struct PySemiMajorAxisEvent {
    event: Option<events::DSemiMajorAxisEvent>,
}

#[pymethods]
impl PySemiMajorAxisEvent {
    /// Create a new semi-major axis event detector.
    ///
    /// Args:
    ///     value (float): Semi-major axis value in meters
    ///     name (str): Event name for identification
    ///     direction (EventDirection): Detection direction
    ///
    /// Returns:
    ///     SemiMajorAxisEvent: New semi-major axis event detector
    #[new]
    #[pyo3(signature = (value, name, direction))]
    fn new(value: f64, name: String, direction: PyRef<PyEventDirection>) -> PyResult<Self> {
        let event = events::DSemiMajorAxisEvent::new(value, name, direction.direction);
        Ok(PySemiMajorAxisEvent { event: Some(event) })
    }

    /// Set instance number for display name.
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self { event: slf.event.take() }
    }

    /// Set custom tolerances for event detection.
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self { event: slf.event.take() }
    }

    /// Mark this event as terminal (stops propagation).
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self { event: slf.event.take() }
    }
}

/// Eccentricity event detector.
///
/// Detects when orbital eccentricity crosses a value value.
///
/// Args:
///     value (float): Eccentricity value (dimensionless)
///     name (str): Event name for identification
///     direction (EventDirection): Detection direction
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Detect when orbit becomes nearly circular
///     event = bh.EccentricityEvent(
///         0.001,
///         "Near circular",
///         bh.EventDirection.DECREASING
///     )
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "EccentricityEvent")]
pub struct PyEccentricityEvent {
    event: Option<events::DEccentricityEvent>,
}

#[pymethods]
impl PyEccentricityEvent {
    /// Create a new eccentricity event detector.
    ///
    /// Args:
    ///     value (float): Eccentricity value (dimensionless)
    ///     name (str): Event name for identification
    ///     direction (EventDirection): Detection direction
    ///
    /// Returns:
    ///     EccentricityEvent: New eccentricity event detector
    #[new]
    #[pyo3(signature = (value, name, direction))]
    fn new(value: f64, name: String, direction: PyRef<PyEventDirection>) -> PyResult<Self> {
        let event = events::DEccentricityEvent::new(value, name, direction.direction);
        Ok(PyEccentricityEvent { event: Some(event) })
    }

    /// Set instance number for display name.
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self { event: slf.event.take() }
    }

    /// Set custom tolerances for event detection.
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self { event: slf.event.take() }
    }

    /// Mark this event as terminal (stops propagation).
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self { event: slf.event.take() }
    }
}

/// Inclination event detector.
///
/// Detects when orbital inclination crosses a value value.
///
/// Args:
///     value (float): Inclination value
///     name (str): Event name for identification
///     direction (EventDirection): Detection direction
///     angle_format (AngleFormat): Whether value is in degrees or radians
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Detect when inclination crosses 90 degrees (polar orbit value)
///     event = bh.InclinationEvent(
///         90.0,
///         "Polar value",
///         bh.EventDirection.ANY,
///         bh.AngleFormat.DEGREES
///     )
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "InclinationEvent")]
pub struct PyInclinationEvent {
    event: Option<events::DInclinationEvent>,
}

#[pymethods]
impl PyInclinationEvent {
    /// Create a new inclination event detector.
    ///
    /// Args:
    ///     value (float): Inclination value
    ///     name (str): Event name for identification
    ///     direction (EventDirection): Detection direction
    ///     angle_format (AngleFormat): Whether value is in degrees or radians
    ///
    /// Returns:
    ///     InclinationEvent: New inclination event detector
    #[new]
    #[pyo3(signature = (value, name, direction, angle_format))]
    fn new(
        value: f64,
        name: String,
        direction: PyRef<PyEventDirection>,
        angle_format: PyRef<PyAngleFormat>,
    ) -> PyResult<Self> {
        let event =
            events::DInclinationEvent::new(value, name, direction.direction, angle_format.value);
        Ok(PyInclinationEvent { event: Some(event) })
    }

    /// Set instance number for display name.
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self { event: slf.event.take() }
    }

    /// Set custom tolerances for event detection.
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self { event: slf.event.take() }
    }

    /// Mark this event as terminal (stops propagation).
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self { event: slf.event.take() }
    }
}

/// Argument of perigee event detector.
///
/// Detects when argument of perigee crosses a value value.
///
/// Args:
///     value (float): Argument of perigee value
///     name (str): Event name for identification
///     direction (EventDirection): Detection direction
///     angle_format (AngleFormat): Whether value is in degrees or radians
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "ArgumentOfPerigeeEvent")]
pub struct PyArgumentOfPerigeeEvent {
    event: Option<events::DArgumentOfPerigeeEvent>,
}

#[pymethods]
impl PyArgumentOfPerigeeEvent {
    /// Create a new argument of perigee event detector.
    ///
    /// Args:
    ///     value (float): Argument of perigee value
    ///     name (str): Event name for identification
    ///     direction (EventDirection): Detection direction
    ///     angle_format (AngleFormat): Whether value is in degrees or radians
    ///
    /// Returns:
    ///     ArgumentOfPerigeeEvent: New argument of perigee event detector
    #[new]
    #[pyo3(signature = (value, name, direction, angle_format))]
    fn new(
        value: f64,
        name: String,
        direction: PyRef<PyEventDirection>,
        angle_format: PyRef<PyAngleFormat>,
    ) -> PyResult<Self> {
        let event = events::DArgumentOfPerigeeEvent::new(
            value,
            name,
            direction.direction,
            angle_format.value,
        );
        Ok(PyArgumentOfPerigeeEvent { event: Some(event) })
    }

    /// Set instance number for display name.
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self { event: slf.event.take() }
    }

    /// Set custom tolerances for event detection.
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self { event: slf.event.take() }
    }

    /// Mark this event as terminal (stops propagation).
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self { event: slf.event.take() }
    }
}

/// Mean anomaly event detector.
///
/// Detects when mean anomaly crosses a value value.
///
/// Args:
///     value (float): Mean anomaly value
///     name (str): Event name for identification
///     direction (EventDirection): Detection direction
///     angle_format (AngleFormat): Whether value is in degrees or radians
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "MeanAnomalyEvent")]
pub struct PyMeanAnomalyEvent {
    event: Option<events::DMeanAnomalyEvent>,
}

#[pymethods]
impl PyMeanAnomalyEvent {
    /// Create a new mean anomaly event detector.
    ///
    /// Args:
    ///     value (float): Mean anomaly value
    ///     name (str): Event name for identification
    ///     direction (EventDirection): Detection direction
    ///     angle_format (AngleFormat): Whether value is in degrees or radians
    ///
    /// Returns:
    ///     MeanAnomalyEvent: New mean anomaly event detector
    #[new]
    #[pyo3(signature = (value, name, direction, angle_format))]
    fn new(
        value: f64,
        name: String,
        direction: PyRef<PyEventDirection>,
        angle_format: PyRef<PyAngleFormat>,
    ) -> PyResult<Self> {
        let event =
            events::DMeanAnomalyEvent::new(value, name, direction.direction, angle_format.value);
        Ok(PyMeanAnomalyEvent { event: Some(event) })
    }

    /// Set instance number for display name.
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self { event: slf.event.take() }
    }

    /// Set custom tolerances for event detection.
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self { event: slf.event.take() }
    }

    /// Mark this event as terminal (stops propagation).
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self { event: slf.event.take() }
    }
}

/// Eccentric anomaly event detector.
///
/// Detects when eccentric anomaly crosses a value value.
///
/// Args:
///     value (float): Eccentric anomaly value
///     name (str): Event name for identification
///     direction (EventDirection): Detection direction
///     angle_format (AngleFormat): Whether value is in degrees or radians
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "EccentricAnomalyEvent")]
pub struct PyEccentricAnomalyEvent {
    event: Option<events::DEccentricAnomalyEvent>,
}

#[pymethods]
impl PyEccentricAnomalyEvent {
    /// Create a new eccentric anomaly event detector.
    ///
    /// Args:
    ///     value (float): Eccentric anomaly value
    ///     name (str): Event name for identification
    ///     direction (EventDirection): Detection direction
    ///     angle_format (AngleFormat): Whether value is in degrees or radians
    ///
    /// Returns:
    ///     EccentricAnomalyEvent: New eccentric anomaly event detector
    #[new]
    #[pyo3(signature = (value, name, direction, angle_format))]
    fn new(
        value: f64,
        name: String,
        direction: PyRef<PyEventDirection>,
        angle_format: PyRef<PyAngleFormat>,
    ) -> PyResult<Self> {
        let event = events::DEccentricAnomalyEvent::new(
            value,
            name,
            direction.direction,
            angle_format.value,
        );
        Ok(PyEccentricAnomalyEvent { event: Some(event) })
    }

    /// Set instance number for display name.
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self { event: slf.event.take() }
    }

    /// Set custom tolerances for event detection.
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self { event: slf.event.take() }
    }

    /// Mark this event as terminal (stops propagation).
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self { event: slf.event.take() }
    }
}

/// True anomaly event detector.
///
/// Detects when true anomaly crosses a value value.
///
/// Args:
///     value (float): True anomaly value
///     name (str): Event name for identification
///     direction (EventDirection): Detection direction
///     angle_format (AngleFormat): Whether value is in degrees or radians
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "TrueAnomalyEvent")]
pub struct PyTrueAnomalyEvent {
    event: Option<events::DTrueAnomalyEvent>,
}

#[pymethods]
impl PyTrueAnomalyEvent {
    /// Create a new true anomaly event detector.
    ///
    /// Args:
    ///     value (float): True anomaly value
    ///     name (str): Event name for identification
    ///     direction (EventDirection): Detection direction
    ///     angle_format (AngleFormat): Whether value is in degrees or radians
    ///
    /// Returns:
    ///     TrueAnomalyEvent: New true anomaly event detector
    #[new]
    #[pyo3(signature = (value, name, direction, angle_format))]
    fn new(
        value: f64,
        name: String,
        direction: PyRef<PyEventDirection>,
        angle_format: PyRef<PyAngleFormat>,
    ) -> PyResult<Self> {
        let event = events::DTrueAnomalyEvent::new(
            value,
            name,
            direction.direction,
            angle_format.value,
        );
        Ok(PyTrueAnomalyEvent { event: Some(event) })
    }

    /// Set instance number for display name.
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self { event: slf.event.take() }
    }

    /// Set custom tolerances for event detection.
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self { event: slf.event.take() }
    }

    /// Mark this event as terminal (stops propagation).
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self { event: slf.event.take() }
    }
}

/// Argument of latitude event detector.
///
/// Detects when argument of latitude (omega + true anomaly) crosses a value value.
///
/// Args:
///     value (float): Argument of latitude value
///     name (str): Event name for identification
///     direction (EventDirection): Detection direction
///     angle_format (AngleFormat): Whether value is in degrees or radians
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "ArgumentOfLatitudeEvent")]
pub struct PyArgumentOfLatitudeEvent {
    event: Option<events::DArgumentOfLatitudeEvent>,
}

#[pymethods]
impl PyArgumentOfLatitudeEvent {
    /// Create a new argument of latitude event detector.
    ///
    /// Args:
    ///     value (float): Argument of latitude value
    ///     name (str): Event name for identification
    ///     direction (EventDirection): Detection direction
    ///     angle_format (AngleFormat): Whether value is in degrees or radians
    ///
    /// Returns:
    ///     ArgumentOfLatitudeEvent: New argument of latitude event detector
    #[new]
    #[pyo3(signature = (value, name, direction, angle_format))]
    fn new(
        value: f64,
        name: String,
        direction: PyRef<PyEventDirection>,
        angle_format: PyRef<PyAngleFormat>,
    ) -> PyResult<Self> {
        let event = events::DArgumentOfLatitudeEvent::new(
            value,
            name,
            direction.direction,
            angle_format.value,
        );
        Ok(PyArgumentOfLatitudeEvent { event: Some(event) })
    }

    /// Set instance number for display name.
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self { event: slf.event.take() }
    }

    /// Set custom tolerances for event detection.
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self { event: slf.event.take() }
    }

    /// Mark this event as terminal (stops propagation).
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self { event: slf.event.take() }
    }
}

// ================================
// Node Crossing Events
// ================================

/// Ascending node event detector.
///
/// Detects when spacecraft crosses the ascending node (equator from south to north).
///
/// Args:
///     name (str): Event name for identification
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Detect ascending node crossings
///     event = bh.AscendingNodeEvent("Ascending Node")
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "AscendingNodeEvent")]
pub struct PyAscendingNodeEvent {
    event: Option<events::DAscendingNodeEvent>,
}

#[pymethods]
impl PyAscendingNodeEvent {
    /// Create a new ascending node event detector.
    ///
    /// Args:
    ///     name (str): Event name for identification
    ///
    /// Returns:
    ///     AscendingNodeEvent: New ascending node event detector
    #[new]
    #[pyo3(signature = (name))]
    fn new(name: String) -> PyResult<Self> {
        let event = events::DAscendingNodeEvent::new(name);
        Ok(PyAscendingNodeEvent { event: Some(event) })
    }

    /// Set instance number for display name.
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self { event: slf.event.take() }
    }

    /// Set custom tolerances for event detection.
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self { event: slf.event.take() }
    }

    /// Mark this event as terminal (stops propagation).
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self { event: slf.event.take() }
    }
}

/// Descending node event detector.
///
/// Detects when spacecraft crosses the descending node (equator from north to south).
///
/// Args:
///     name (str): Event name for identification
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Detect descending node crossings
///     event = bh.DescendingNodeEvent("Descending Node")
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "DescendingNodeEvent")]
pub struct PyDescendingNodeEvent {
    event: Option<events::DDescendingNodeEvent>,
}

#[pymethods]
impl PyDescendingNodeEvent {
    /// Create a new descending node event detector.
    ///
    /// Args:
    ///     name (str): Event name for identification
    ///
    /// Returns:
    ///     DescendingNodeEvent: New descending node event detector
    #[new]
    #[pyo3(signature = (name))]
    fn new(name: String) -> PyResult<Self> {
        let event = events::DDescendingNodeEvent::new(name);
        Ok(PyDescendingNodeEvent { event: Some(event) })
    }

    /// Set instance number for display name.
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self { event: slf.event.take() }
    }

    /// Set custom tolerances for event detection.
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self { event: slf.event.take() }
    }

    /// Mark this event as terminal (stops propagation).
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self { event: slf.event.take() }
    }
}

// ================================
// State-Derived Events
// ================================

/// Speed event detector.
///
/// Detects when velocity magnitude crosses a value value.
///
/// Args:
///     value (float): Speed value in m/s
///     name (str): Event name for identification
///     direction (EventDirection): Detection direction
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Detect when speed exceeds 8 km/s
///     event = bh.SpeedEvent(
///         8000.0,
///         "High Speed",
///         bh.EventDirection.INCREASING
///     )
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "SpeedEvent")]
pub struct PySpeedEvent {
    event: Option<events::DSpeedEvent>,
}

#[pymethods]
impl PySpeedEvent {
    /// Create a new speed event detector.
    ///
    /// Args:
    ///     value (float): Speed value in m/s
    ///     name (str): Event name for identification
    ///     direction (EventDirection): Detection direction
    ///
    /// Returns:
    ///     SpeedEvent: New speed event detector
    #[new]
    #[pyo3(signature = (value, name, direction))]
    fn new(value: f64, name: String, direction: PyRef<PyEventDirection>) -> PyResult<Self> {
        let event = events::DSpeedEvent::new(value, name, direction.direction);
        Ok(PySpeedEvent { event: Some(event) })
    }

    /// Set instance number for display name.
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self { event: slf.event.take() }
    }

    /// Set custom tolerances for event detection.
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self { event: slf.event.take() }
    }

    /// Mark this event as terminal (stops propagation).
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self { event: slf.event.take() }
    }
}

/// Longitude event detector.
///
/// Detects when geodetic longitude crosses a value value.
/// Requires EOP initialization for ECI->ECEF transformation.
///
/// Args:
///     value (float): Longitude value
///     name (str): Event name for identification
///     direction (EventDirection): Detection direction
///     angle_format (AngleFormat): Whether value is in degrees or radians
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Detect when crossing prime meridian
///     event = bh.LongitudeEvent(
///         0.0,
///         "Prime Meridian",
///         bh.EventDirection.ANY,
///         bh.AngleFormat.DEGREES
///     )
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "LongitudeEvent")]
pub struct PyLongitudeEvent {
    event: Option<events::DLongitudeEvent>,
}

#[pymethods]
impl PyLongitudeEvent {
    /// Create a new longitude event detector.
    ///
    /// Args:
    ///     value (float): Longitude value
    ///     name (str): Event name for identification
    ///     direction (EventDirection): Detection direction
    ///     angle_format (AngleFormat): Whether value is in degrees or radians
    ///
    /// Returns:
    ///     LongitudeEvent: New longitude event detector
    #[new]
    #[pyo3(signature = (value, name, direction, angle_format))]
    fn new(
        value: f64,
        name: String,
        direction: PyRef<PyEventDirection>,
        angle_format: PyRef<PyAngleFormat>,
    ) -> PyResult<Self> {
        let event =
            events::DLongitudeEvent::new(value, name, direction.direction, angle_format.value);
        Ok(PyLongitudeEvent { event: Some(event) })
    }

    /// Set instance number for display name.
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self { event: slf.event.take() }
    }

    /// Set custom tolerances for event detection.
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self { event: slf.event.take() }
    }

    /// Mark this event as terminal (stops propagation).
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self { event: slf.event.take() }
    }
}

/// Latitude event detector.
///
/// Detects when geodetic latitude crosses a value value.
/// Requires EOP initialization for ECI->ECEF transformation.
///
/// Args:
///     value (float): Latitude value
///     name (str): Event name for identification
///     direction (EventDirection): Detection direction
///     angle_format (AngleFormat): Whether value is in degrees or radians
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Detect when crossing the equator
///     event = bh.LatitudeEvent(
///         0.0,
///         "Equator Crossing",
///         bh.EventDirection.ANY,
///         bh.AngleFormat.DEGREES
///     )
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "LatitudeEvent")]
pub struct PyLatitudeEvent {
    event: Option<events::DLatitudeEvent>,
}

#[pymethods]
impl PyLatitudeEvent {
    /// Create a new latitude event detector.
    ///
    /// Args:
    ///     value (float): Latitude value
    ///     name (str): Event name for identification
    ///     direction (EventDirection): Detection direction
    ///     angle_format (AngleFormat): Whether value is in degrees or radians
    ///
    /// Returns:
    ///     LatitudeEvent: New latitude event detector
    #[new]
    #[pyo3(signature = (value, name, direction, angle_format))]
    fn new(
        value: f64,
        name: String,
        direction: PyRef<PyEventDirection>,
        angle_format: PyRef<PyAngleFormat>,
    ) -> PyResult<Self> {
        let event =
            events::DLatitudeEvent::new(value, name, direction.direction, angle_format.value);
        Ok(PyLatitudeEvent { event: Some(event) })
    }

    /// Set instance number for display name.
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self { event: slf.event.take() }
    }

    /// Set custom tolerances for event detection.
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self { event: slf.event.take() }
    }

    /// Mark this event as terminal (stops propagation).
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self { event: slf.event.take() }
    }
}

// ================================
// Eclipse/Shadow Events
// ================================

/// Umbra event detector.
///
/// Detects when spacecraft enters/exits Earth's umbra (full shadow).
/// Uses the conical shadow model.
///
/// Args:
///     name (str): Event name for identification
///     edge (EdgeType): Edge type (RISING_EDGE = entering, FALLING_EDGE = exiting)
///     ephemeris_source (EphemerisSource | None): Source for sun position (None for low-precision)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Detect umbra entry with low-precision ephemeris
///     event = bh.UmbraEvent("Umbra Entry", bh.EdgeType.RISING_EDGE, None)
///
///     # Detect umbra exit with high-precision DE440s ephemeris
///     event = bh.UmbraEvent("Umbra Exit", bh.EdgeType.FALLING_EDGE, bh.EphemerisSource.DE440s)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "UmbraEvent")]
pub struct PyUmbraEvent {
    event: Option<events::DUmbraEvent>,
}

#[pymethods]
impl PyUmbraEvent {
    /// Create a new umbra event detector.
    ///
    /// Args:
    ///     name (str): Event name for identification
    ///     edge (EdgeType): Edge type (RISING_EDGE = entering, FALLING_EDGE = exiting)
    ///     ephemeris_source (EphemerisSource | None): Source for sun position (None for low-precision)
    ///
    /// Returns:
    ///     UmbraEvent: New umbra event detector
    #[new]
    #[pyo3(signature = (name, edge, ephemeris_source))]
    fn new(
        name: String,
        edge: PyRef<PyEdgeType>,
        ephemeris_source: Option<PyRef<PyEphemerisSource>>,
    ) -> PyResult<Self> {
        let source = ephemeris_source.map(|s| (*s).into());
        let event = events::DUmbraEvent::new(name, edge.edge, source);
        Ok(PyUmbraEvent { event: Some(event) })
    }

    /// Set instance number for display name.
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self { event: slf.event.take() }
    }

    /// Set custom tolerances for event detection.
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self { event: slf.event.take() }
    }

    /// Mark this event as terminal (stops propagation).
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self { event: slf.event.take() }
    }
}

/// Penumbra event detector.
///
/// Detects when spacecraft enters/exits Earth's penumbra (partial shadow).
/// Uses the conical shadow model.
///
/// Args:
///     name (str): Event name for identification
///     edge (EdgeType): Edge type (RISING_EDGE = entering, FALLING_EDGE = exiting)
///     ephemeris_source (EphemerisSource | None): Source for sun position (None for low-precision)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Detect penumbra entry
///     event = bh.PenumbraEvent("Penumbra Entry", bh.EdgeType.RISING_EDGE, None)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "PenumbraEvent")]
pub struct PyPenumbraEvent {
    event: Option<events::DPenumbraEvent>,
}

#[pymethods]
impl PyPenumbraEvent {
    /// Create a new penumbra event detector.
    ///
    /// Args:
    ///     name (str): Event name for identification
    ///     edge (EdgeType): Edge type (RISING_EDGE = entering, FALLING_EDGE = exiting)
    ///     ephemeris_source (EphemerisSource | None): Source for sun position (None for low-precision)
    ///
    /// Returns:
    ///     PenumbraEvent: New penumbra event detector
    #[new]
    #[pyo3(signature = (name, edge, ephemeris_source))]
    fn new(
        name: String,
        edge: PyRef<PyEdgeType>,
        ephemeris_source: Option<PyRef<PyEphemerisSource>>,
    ) -> PyResult<Self> {
        let source = ephemeris_source.map(|s| (*s).into());
        let event = events::DPenumbraEvent::new(name, edge.edge, source);
        Ok(PyPenumbraEvent { event: Some(event) })
    }

    /// Set instance number for display name.
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self { event: slf.event.take() }
    }

    /// Set custom tolerances for event detection.
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self { event: slf.event.take() }
    }

    /// Mark this event as terminal (stops propagation).
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self { event: slf.event.take() }
    }
}

/// Eclipse event detector.
///
/// Detects when spacecraft enters/exits eclipse (either umbra or penumbra).
/// Uses the conical shadow model.
///
/// Args:
///     name (str): Event name for identification
///     edge (EdgeType): Edge type (RISING_EDGE = entering, FALLING_EDGE = exiting)
///     ephemeris_source (EphemerisSource | None): Source for sun position (None for low-precision)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Detect any eclipse entry
///     event = bh.EclipseEvent("Eclipse Entry", bh.EdgeType.RISING_EDGE, None)
///
///     # Detect eclipse exit with high-precision ephemeris
///     event = bh.EclipseEvent("Eclipse Exit", bh.EdgeType.FALLING_EDGE, bh.EphemerisSource.DE440s)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "EclipseEvent")]
pub struct PyEclipseEvent {
    event: Option<events::DEclipseEvent>,
}

#[pymethods]
impl PyEclipseEvent {
    /// Create a new eclipse event detector.
    ///
    /// Args:
    ///     name (str): Event name for identification
    ///     edge (EdgeType): Edge type (RISING_EDGE = entering, FALLING_EDGE = exiting)
    ///     ephemeris_source (EphemerisSource | None): Source for sun position (None for low-precision)
    ///
    /// Returns:
    ///     EclipseEvent: New eclipse event detector
    #[new]
    #[pyo3(signature = (name, edge, ephemeris_source))]
    fn new(
        name: String,
        edge: PyRef<PyEdgeType>,
        ephemeris_source: Option<PyRef<PyEphemerisSource>>,
    ) -> PyResult<Self> {
        let source = ephemeris_source.map(|s| (*s).into());
        let event = events::DEclipseEvent::new(name, edge.edge, source);
        Ok(PyEclipseEvent { event: Some(event) })
    }

    /// Set instance number for display name.
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self { event: slf.event.take() }
    }

    /// Set custom tolerances for event detection.
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self { event: slf.event.take() }
    }

    /// Mark this event as terminal (stops propagation).
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self { event: slf.event.take() }
    }
}

/// Sunlit event detector.
///
/// Detects when spacecraft enters/exits sunlight (fully illuminated).
/// Uses the conical shadow model.
///
/// Args:
///     name (str): Event name for identification
///     edge (EdgeType): Edge type (RISING_EDGE = entering, FALLING_EDGE = leaving)
///     ephemeris_source (EphemerisSource | None): Source for sun position (None for low-precision)
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Detect entering sunlight
///     event = bh.SunlitEvent("Enter Sunlight", bh.EdgeType.RISING_EDGE, None)
///
///     # Detect leaving sunlight with high-precision ephemeris
///     event = bh.SunlitEvent("Leave Sunlight", bh.EdgeType.FALLING_EDGE, bh.EphemerisSource.DE440s)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "SunlitEvent")]
pub struct PySunlitEvent {
    event: Option<events::DSunlitEvent>,
}

#[pymethods]
impl PySunlitEvent {
    /// Create a new sunlit event detector.
    ///
    /// Args:
    ///     name (str): Event name for identification
    ///     edge (EdgeType): Edge type (RISING_EDGE = entering, FALLING_EDGE = leaving)
    ///     ephemeris_source (EphemerisSource | None): Source for sun position (None for low-precision)
    ///
    /// Returns:
    ///     SunlitEvent: New sunlit event detector
    #[new]
    #[pyo3(signature = (name, edge, ephemeris_source))]
    fn new(
        name: String,
        edge: PyRef<PyEdgeType>,
        ephemeris_source: Option<PyRef<PyEphemerisSource>>,
    ) -> PyResult<Self> {
        let source = ephemeris_source.map(|s| (*s).into());
        let event = events::DSunlitEvent::new(name, edge.edge, source);
        Ok(PySunlitEvent { event: Some(event) })
    }

    /// Set instance number for display name.
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self { event: slf.event.take() }
    }

    /// Set custom tolerances for event detection.
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self { event: slf.event.take() }
    }

    /// Mark this event as terminal (stops propagation).
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self { event: slf.event.take() }
    }
}

// =============================================================================
// AOI (Area of Interest) Event Detectors
// =============================================================================

/// AOI Entry event detector.
///
/// Detects when a satellite's sub-satellite point enters a polygonal Area of Interest.
/// The sub-satellite point is computed by transforming the ECI position to geodetic
/// coordinates (longitude, latitude).
///
/// The polygon can be defined either from a PolygonLocation object or from raw
/// (longitude, latitude) coordinate pairs.
///
/// Args:
///     polygon (PolygonLocation): Polygon defining the Area of Interest
///     name (str): Event name for identification
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Create from PolygonLocation
///     vertices = [
///         [10.0, 50.0, 0.0],  # [lon, lat, alt] in degrees
///         [20.0, 50.0, 0.0],
///         [20.0, 55.0, 0.0],
///         [10.0, 55.0, 0.0],
///         [10.0, 50.0, 0.0],
///     ]
///     polygon = bh.PolygonLocation(vertices)
///     entry_event = bh.AOIEntryEvent(polygon, "Europe Entry")
///
///     # Create from coordinates with angle format
///     coords = [
///         (10.0, 50.0),  # (lon, lat) in degrees
///         (20.0, 50.0),
///         (20.0, 55.0),
///         (10.0, 55.0),
///         (10.0, 50.0),
///     ]
///     entry_event = bh.AOIEntryEvent.from_coordinates(coords, "Europe Entry", bh.AngleFormat.DEGREES)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "AOIEntryEvent")]
pub struct PyAOIEntryEvent {
    pub(crate) event: Option<events::DAOIEntryEvent>,
    /// Vertices stored in radians (lon, lat) for creating S-type events
    pub(crate) vertices_rad: Vec<(f64, f64)>,
    /// Event name for creating S-type events
    pub(crate) name: String,
    /// Whether this event is terminal
    pub(crate) is_terminal: bool,
}

#[pymethods]
impl PyAOIEntryEvent {
    /// Create a new AOI entry event detector from a PolygonLocation.
    ///
    /// Args:
    ///     polygon (PolygonLocation): Polygon defining the Area of Interest
    ///     name (str): Event name for identification
    ///
    /// Returns:
    ///     AOIEntryEvent: New AOI entry event detector
    #[new]
    #[pyo3(signature = (polygon, name))]
    fn new(polygon: PyRef<PyPolygonLocation>, name: String) -> PyResult<Self> {
        // Extract vertices in radians from polygon (lon, lat, alt format in degrees)
        let vertices_rad: Vec<(f64, f64)> = polygon.location.vertices()
            .iter()
            .map(|v| (v[0].to_radians(), v[1].to_radians()))
            .collect();
        let event = events::DAOIEntryEvent::from_polygon(&polygon.location, name.clone());
        Ok(PyAOIEntryEvent {
            event: Some(event),
            vertices_rad,
            name,
            is_terminal: false,
        })
    }

    /// Create an AOI entry event from raw coordinate pairs.
    ///
    /// Args:
    ///     vertices (list[tuple[float, float]]): AOI vertices as (longitude, latitude) pairs
    ///     name (str): Event name for identification
    ///     angle_format (AngleFormat): Whether coordinates are in DEGREES or RADIANS
    ///
    /// Returns:
    ///     AOIEntryEvent: New AOI entry event detector
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     coords = [(10.0, 50.0), (20.0, 50.0), (20.0, 55.0), (10.0, 55.0), (10.0, 50.0)]
    ///     event = bh.AOIEntryEvent.from_coordinates(coords, "AOI Entry", bh.AngleFormat.DEGREES)
    ///     ```
    #[staticmethod]
    #[pyo3(signature = (vertices, name, angle_format))]
    fn from_coordinates(
        vertices: Vec<(f64, f64)>,
        name: String,
        angle_format: PyRef<PyAngleFormat>,
    ) -> PyResult<Self> {
        // Convert vertices to radians for storage
        let vertices_rad: Vec<(f64, f64)> = match angle_format.value {
            constants::AngleFormat::Degrees => vertices.iter()
                .map(|(lon, lat)| (lon.to_radians(), lat.to_radians()))
                .collect(),
            constants::AngleFormat::Radians => vertices.clone(),
        };
        let event = events::DAOIEntryEvent::from_coordinates(&vertices, name.clone(), angle_format.value);
        Ok(PyAOIEntryEvent {
            event: Some(event),
            vertices_rad,
            name,
            is_terminal: false,
        })
    }

    /// Set instance number for display name.
    ///
    /// Args:
    ///     instance (int): Instance number to append
    ///
    /// Returns:
    ///     AOIEntryEvent: Self for method chaining
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        let vertices_rad = std::mem::take(&mut slf.vertices_rad);
        let name = std::mem::take(&mut slf.name);
        let is_terminal = slf.is_terminal;
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self {
            event: slf.event.take(),
            vertices_rad,
            name,
            is_terminal,
        }
    }

    /// Set custom tolerances for event detection.
    ///
    /// Args:
    ///     time_tol (float): Time tolerance in seconds
    ///     value_tol (float): Value tolerance
    ///
    /// Returns:
    ///     AOIEntryEvent: Self for method chaining
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        let vertices_rad = std::mem::take(&mut slf.vertices_rad);
        let name = std::mem::take(&mut slf.name);
        let is_terminal = slf.is_terminal;
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self {
            event: slf.event.take(),
            vertices_rad,
            name,
            is_terminal,
        }
    }

    /// Mark this event as terminal (stops propagation).
    ///
    /// Returns:
    ///     AOIEntryEvent: Self for method chaining
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        let vertices_rad = std::mem::take(&mut slf.vertices_rad);
        let name = std::mem::take(&mut slf.name);
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self {
            event: slf.event.take(),
            vertices_rad,
            name,
            is_terminal: true,
        }
    }
}

/// AOI Exit event detector.
///
/// Detects when a satellite's sub-satellite point exits a polygonal Area of Interest.
/// The sub-satellite point is computed by transforming the ECI position to geodetic
/// coordinates (longitude, latitude).
///
/// The polygon can be defined either from a PolygonLocation object or from raw
/// (longitude, latitude) coordinate pairs.
///
/// Args:
///     polygon (PolygonLocation): Polygon defining the Area of Interest
///     name (str): Event name for identification
///
/// Example:
///     ```python
///     import brahe as bh
///
///     # Create from PolygonLocation
///     vertices = [
///         [10.0, 50.0, 0.0],  # [lon, lat, alt] in degrees
///         [20.0, 50.0, 0.0],
///         [20.0, 55.0, 0.0],
///         [10.0, 55.0, 0.0],
///         [10.0, 50.0, 0.0],
///     ]
///     polygon = bh.PolygonLocation(vertices)
///     exit_event = bh.AOIExitEvent(polygon, "Europe Exit")
///
///     # Create from coordinates with angle format
///     coords = [
///         (10.0, 50.0),  # (lon, lat) in degrees
///         (20.0, 50.0),
///         (20.0, 55.0),
///         (10.0, 55.0),
///         (10.0, 50.0),
///     ]
///     exit_event = bh.AOIExitEvent.from_coordinates(coords, "Europe Exit", bh.AngleFormat.DEGREES)
///     ```
#[pyclass(module = "brahe._brahe")]
#[pyo3(name = "AOIExitEvent")]
pub struct PyAOIExitEvent {
    pub(crate) event: Option<events::DAOIExitEvent>,
    /// Vertices stored in radians (lon, lat) for creating S-type events
    pub(crate) vertices_rad: Vec<(f64, f64)>,
    /// Event name for creating S-type events
    pub(crate) name: String,
    /// Whether this event is terminal
    pub(crate) is_terminal: bool,
}

#[pymethods]
impl PyAOIExitEvent {
    /// Create a new AOI exit event detector from a PolygonLocation.
    ///
    /// Args:
    ///     polygon (PolygonLocation): Polygon defining the Area of Interest
    ///     name (str): Event name for identification
    ///
    /// Returns:
    ///     AOIExitEvent: New AOI exit event detector
    #[new]
    #[pyo3(signature = (polygon, name))]
    fn new(polygon: PyRef<PyPolygonLocation>, name: String) -> PyResult<Self> {
        // Extract vertices in radians from polygon (lon, lat, alt format in degrees)
        let vertices_rad: Vec<(f64, f64)> = polygon.location.vertices()
            .iter()
            .map(|v| (v[0].to_radians(), v[1].to_radians()))
            .collect();
        let event = events::DAOIExitEvent::from_polygon(&polygon.location, name.clone());
        Ok(PyAOIExitEvent {
            event: Some(event),
            vertices_rad,
            name,
            is_terminal: false,
        })
    }

    /// Create an AOI exit event from raw coordinate pairs.
    ///
    /// Args:
    ///     vertices (list[tuple[float, float]]): AOI vertices as (longitude, latitude) pairs
    ///     name (str): Event name for identification
    ///     angle_format (AngleFormat): Whether coordinates are in DEGREES or RADIANS
    ///
    /// Returns:
    ///     AOIExitEvent: New AOI exit event detector
    ///
    /// Example:
    ///     ```python
    ///     import brahe as bh
    ///
    ///     coords = [(10.0, 50.0), (20.0, 50.0), (20.0, 55.0), (10.0, 55.0), (10.0, 50.0)]
    ///     event = bh.AOIExitEvent.from_coordinates(coords, "AOI Exit", bh.AngleFormat.DEGREES)
    ///     ```
    #[staticmethod]
    #[pyo3(signature = (vertices, name, angle_format))]
    fn from_coordinates(
        vertices: Vec<(f64, f64)>,
        name: String,
        angle_format: PyRef<PyAngleFormat>,
    ) -> PyResult<Self> {
        // Convert vertices to radians for storage
        let vertices_rad: Vec<(f64, f64)> = match angle_format.value {
            constants::AngleFormat::Degrees => vertices.iter()
                .map(|(lon, lat)| (lon.to_radians(), lat.to_radians()))
                .collect(),
            constants::AngleFormat::Radians => vertices.clone(),
        };
        let event = events::DAOIExitEvent::from_coordinates(&vertices, name.clone(), angle_format.value);
        Ok(PyAOIExitEvent {
            event: Some(event),
            vertices_rad,
            name,
            is_terminal: false,
        })
    }

    /// Set instance number for display name.
    ///
    /// Args:
    ///     instance (int): Instance number to append
    ///
    /// Returns:
    ///     AOIExitEvent: Self for method chaining
    fn with_instance(mut slf: PyRefMut<'_, Self>, instance: usize) -> Self {
        let vertices_rad = std::mem::take(&mut slf.vertices_rad);
        let name = std::mem::take(&mut slf.name);
        let is_terminal = slf.is_terminal;
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_instance(instance));
        }
        Self {
            event: slf.event.take(),
            vertices_rad,
            name,
            is_terminal,
        }
    }

    /// Set custom tolerances for event detection.
    ///
    /// Args:
    ///     time_tol (float): Time tolerance in seconds
    ///     value_tol (float): Value tolerance
    ///
    /// Returns:
    ///     AOIExitEvent: Self for method chaining
    fn with_tolerances(mut slf: PyRefMut<'_, Self>, time_tol: f64, value_tol: f64) -> Self {
        let vertices_rad = std::mem::take(&mut slf.vertices_rad);
        let name = std::mem::take(&mut slf.name);
        let is_terminal = slf.is_terminal;
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.with_tolerances(time_tol, value_tol));
        }
        Self {
            event: slf.event.take(),
            vertices_rad,
            name,
            is_terminal,
        }
    }

    /// Mark this event as terminal (stops propagation).
    ///
    /// Returns:
    ///     AOIExitEvent: Self for method chaining
    fn set_terminal(mut slf: PyRefMut<'_, Self>) -> Self {
        let vertices_rad = std::mem::take(&mut slf.vertices_rad);
        let name = std::mem::take(&mut slf.name);
        if let Some(event) = slf.event.take() {
            slf.event = Some(event.set_terminal());
        }
        Self {
            event: slf.event.take(),
            vertices_rad,
            name,
            is_terminal: true,
        }
    }
}
