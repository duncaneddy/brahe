"""
Shared helpers for estimation plotting functions.

Provides time-axis computation, data extraction from solver objects,
and layout/style utilities used by estimation_state, estimation_residuals,
and estimation_marginal plot modules.
"""

import math
import numpy as np

DEFAULT_COLORS = [
    "#1f77b4",
    "#d62728",
    "#2ca02c",
    "#ff7f0e",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]

_VALID_TIME_UNITS = ("seconds", "minutes", "hours", "orbits", "epoch")


def compute_time_axis(epochs, time_units="seconds", orbital_period=None):
    """Convert a list of Epoch objects to a numeric time array for plotting.

    Args:
        epochs (list[Epoch]): Ordered list of Epoch objects.
        time_units (str or callable): One of "seconds", "minutes", "hours",
            "orbits", "epoch", or a callable that maps elapsed seconds to
            the desired unit. Default: "seconds".
        orbital_period (float or None): Orbital period in seconds. Required
            when time_units="orbits". Default: None.

    Returns:
        tuple[numpy.ndarray, str]: Numeric time array and x-axis label string.

    Raises:
        ValueError: If time_units="orbits" and orbital_period is None.
        ValueError: If time_units is an unrecognised string.
    """
    if len(epochs) == 0:
        return np.array([]), "Time"

    if callable(time_units):
        return time_units(epochs)

    t0 = epochs[0]
    elapsed = np.array([float(ep - t0) for ep in epochs])

    if time_units == "seconds":
        return elapsed, "Time [s]"

    if time_units == "minutes":
        return elapsed / 60.0, "Time [min]"

    if time_units == "hours":
        return elapsed / 3600.0, "Time [hr]"

    if time_units == "orbits":
        if orbital_period is None:
            raise ValueError("orbital_period must be provided when time_units='orbits'")
        return elapsed / orbital_period, "Time [orbits]"

    if time_units == "epoch":
        return epochs, "Epoch"

    if isinstance(time_units, str):
        raise ValueError(
            f"Unknown time_units '{time_units}'. "
            f"Valid options are: {_VALID_TIME_UNITS} or a callable."
        )

    raise ValueError(f"time_units must be a string or callable, got {type(time_units)}")


def extract_state_history(solver):
    """Extract per-step state estimates from a solver.

    For BatchLeastSquares, returns one state per iteration record
    (the final corrected state at each Gauss-Newton iteration).
    For ExtendedKalmanFilter and UnscentedKalmanFilter, returns the
    post-update state from each filter record.

    Args:
        solver: A solved BatchLeastSquares, ExtendedKalmanFilter, or
            UnscentedKalmanFilter instance.

    Returns:
        tuple[list[Epoch], numpy.ndarray]: Epochs and (N, state_dim) state array.
    """
    if hasattr(solver, "iteration_records"):
        records = solver.iteration_records()
        epochs = [r.epoch for r in records]
        states = np.array([r.state for r in records])
    elif hasattr(solver, "records"):
        records = solver.records()
        epochs = [r.epoch for r in records]
        states = np.array([r.state_updated for r in records])
    else:
        raise TypeError(
            f"Unsupported solver type: {type(solver).__name__}. "
            "Expected BatchLeastSquares, ExtendedKalmanFilter, or UnscentedKalmanFilter."
        )

    return epochs, states


def extract_state_errors(solver, true_trajectory):
    """Compute state errors between solver estimates and a truth trajectory.

    Interpolates the truth trajectory at each epoch from the solver's
    state history, then returns (estimated - true).

    Args:
        solver: A solved BatchLeastSquares, ExtendedKalmanFilter, or
            UnscentedKalmanFilter instance.
        true_trajectory: An OrbitTrajectory instance representing truth.

    Returns:
        tuple[list[Epoch], numpy.ndarray]: Epochs and (N, state_dim) error array.
    """
    epochs, estimated = extract_state_history(solver)
    errors = np.empty_like(estimated)
    for i, epoch in enumerate(epochs):
        truth = true_trajectory.interpolate(epoch)
        errors[i] = estimated[i] - truth
    return epochs, errors


def extract_covariance_sigmas(solver, sigma=3):
    """Extract per-step covariance standard deviations scaled by sigma.

    For BatchLeastSquares, uses the covariance stored in iteration records.
    For ExtendedKalmanFilter and UnscentedKalmanFilter, uses the post-update
    covariance from each filter record.

    Args:
        solver: A solved BatchLeastSquares, ExtendedKalmanFilter, or
            UnscentedKalmanFilter instance.
        sigma (float): Sigma multiplier applied to sqrt(diag(P)). Default: 3.

    Returns:
        numpy.ndarray: Array of shape (N, state_dim) containing
            sigma * sqrt(diag(P)) for each step.
    """
    if hasattr(solver, "iteration_records"):
        records = solver.iteration_records()
        cov_list = [r.covariance for r in records]
    elif hasattr(solver, "records"):
        records = solver.records()
        cov_list = [r.covariance_updated for r in records]
    else:
        raise TypeError(
            f"Unsupported solver type: {type(solver).__name__}. "
            "Expected BatchLeastSquares, ExtendedKalmanFilter, or UnscentedKalmanFilter."
        )

    sigmas = np.array([sigma * np.sqrt(np.diag(P)) for P in cov_list])
    return sigmas


def extract_residuals(solver, iteration=-1, residual_type="postfit", model_name=None):
    """Extract residuals from a solver.

    For BatchLeastSquares, returns the per-observation residuals from the
    specified iteration. Negative indices follow Python conventions
    (e.g., -1 for the last iteration).

    For ExtendedKalmanFilter and UnscentedKalmanFilter, returns the residuals
    from each filter record. The ``iteration`` argument is ignored.

    Args:
        solver: A solved BatchLeastSquares, ExtendedKalmanFilter, or
            UnscentedKalmanFilter instance.
        iteration (int): Iteration index for BLS (default -1, i.e. last).
            Ignored for sequential filters.
        residual_type (str): "prefit" or "postfit". Default: "postfit".
        model_name (str or None): Filter residuals by measurement model name.
            None means include all models. Default: None.

    Returns:
        tuple[list[Epoch], numpy.ndarray, int]:
            - epochs: list of Epoch for each residual observation
            - residuals: (N, n_components) array
            - n_components: residual dimension per observation

    Raises:
        ValueError: If residual_type is not "prefit" or "postfit".
    """
    if residual_type not in ("prefit", "postfit"):
        raise ValueError(
            f"residual_type must be 'prefit' or 'postfit', got '{residual_type}'"
        )

    if hasattr(solver, "iteration_records"):
        # BatchLeastSquares path
        all_iterations = solver.observation_residuals()
        if len(all_iterations) == 0:
            return [], np.empty((0, 0)), 0

        iter_residuals = all_iterations[iteration]

        if model_name is not None:
            iter_residuals = [r for r in iter_residuals if r.model_name == model_name]

        if len(iter_residuals) == 0:
            return [], np.empty((0, 0)), 0

        epochs = [r.epoch for r in iter_residuals]
        if residual_type == "postfit":
            residuals = np.array([r.postfit_residual for r in iter_residuals])
        else:
            residuals = np.array([r.prefit_residual for r in iter_residuals])

        n_components = residuals.shape[1] if residuals.ndim == 2 else 1
        return epochs, residuals, n_components

    elif hasattr(solver, "records"):
        # EKF / UKF path
        records = solver.records()

        if model_name is not None:
            records = [r for r in records if r.measurement_name == model_name]

        if len(records) == 0:
            return [], np.empty((0, 0)), 0

        epochs = [r.epoch for r in records]
        if residual_type == "postfit":
            residuals = np.array([r.postfit_residual for r in records])
        else:
            residuals = np.array([r.prefit_residual for r in records])

        n_components = residuals.shape[1] if residuals.ndim == 2 else 1
        return epochs, residuals, n_components

    else:
        raise TypeError(
            f"Unsupported solver type: {type(solver).__name__}. "
            "Expected BatchLeastSquares, ExtendedKalmanFilter, or UnscentedKalmanFilter."
        )


def extract_sub_covariance(solver, state_indices=(0, 1)):
    """Extract a 2D sub-covariance from the solver's current covariance.

    Uses the solver's current (final) state and covariance. For BLS this
    is the formal covariance; for sequential filters it is the post-update
    covariance at the final step.

    Args:
        solver: A solved BatchLeastSquares, ExtendedKalmanFilter, or
            UnscentedKalmanFilter instance.
        state_indices (tuple[int, int]): Two state indices to extract.
            Default: (0, 1).

    Returns:
        tuple[numpy.ndarray, numpy.ndarray]:
            - mean_2d: shape (2,) sub-state vector.
            - cov_2x2: shape (2, 2) sub-covariance matrix.
    """
    i, j = state_indices
    state = solver.current_state()
    cov = solver.current_covariance()

    mean_2d = np.array([state[i], state[j]])
    cov_2x2 = np.array(
        [
            [cov[i, i], cov[i, j]],
            [cov[j, i], cov[j, j]],
        ]
    )
    return mean_2d, cov_2x2


def resolve_colors(n, colors=None):
    """Return n colors from the provided list or the default color cycle.

    If the provided (or default) color list is shorter than n, it cycles.

    Args:
        n (int): Number of colors to return.
        colors (list[str] or None): Color list to use. Default: None (uses
            DEFAULT_COLORS).

    Returns:
        list[str]: List of n color strings.
    """
    source = colors if colors is not None else DEFAULT_COLORS
    if n == 0 or len(source) == 0:
        return []
    return [source[i % len(source)] for i in range(n)]


def resolve_labels(n, labels=None):
    """Return n labels from the provided list or generate default ones.

    Default labels follow the pattern "Series 0", "Series 1", etc.

    Args:
        n (int): Number of labels to return.
        labels (list[str] or None): Label list to use. If None, generates
            default labels. Default: None.

    Returns:
        list[str]: List of n label strings.
    """
    if labels is not None:
        return list(labels)
    return [f"Series {i}" for i in range(n)]


def compute_grid_layout(n_states, ncols=3):
    """Compute subplot grid dimensions for n_states panels.

    Args:
        n_states (int): Number of state panels to display.
        ncols (int): Number of columns in the grid. Default: 3.

    Returns:
        tuple[int, int]: (nrows, ncols) grid dimensions.
    """
    nrows = math.ceil(n_states / ncols)
    return nrows, ncols
