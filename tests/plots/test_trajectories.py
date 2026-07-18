"""Tests for the Cartesian/Keplerian time-series trajectory plots."""

import numpy as np

import brahe as bh


def _raw_keplerian_array(n=5, dt=1800.0):
    """[N x 6] Keplerian element array (no time column), one row per dt seconds."""
    a = np.full(n, bh.R_EARTH + 500e3)
    e = np.full(n, 0.001)
    i = np.linspace(97.8, 98.0, n)
    raan = np.linspace(15.0, 16.0, n)
    argp = np.linspace(30.0, 31.0, n)
    anom = np.linspace(0.0, 10.0, n)
    return np.column_stack(
        (a, e, np.radians(i), np.radians(raan), np.radians(argp), np.radians(anom))
    )


def test_plot_keplerian_trajectory_times_key_plotly():
    """The "times" dict key wires up a real elapsed-time axis for [N x 6] input."""
    n = 5
    dt = 3600.0  # 1 hour between samples -> spans > 2 hours total, so hours are used
    koe = _raw_keplerian_array(n, dt)
    times = np.arange(n) * dt

    fig = bh.plot_keplerian_trajectory(
        [{"trajectory": koe, "times": times, "label": "Test"}],
        backend="plotly",
    )

    # Semi-major axis subplot (row 1, col 1) should use the supplied times
    # (converted to hours, since the span exceeds 2 hours), not bare indices.
    trace = fig.data[0]
    np.testing.assert_allclose(trace.x, times / 3600.0)

    xaxis_title = fig.layout.xaxis4.title.text
    assert xaxis_title == "Time (hours)"


def test_plot_keplerian_trajectory_without_times_key_uses_indices():
    """Without a "times" array, [N x 6] input falls back to bare sample indices."""
    koe = _raw_keplerian_array(5, 1800.0)

    fig = bh.plot_keplerian_trajectory(
        [{"trajectory": koe, "label": "Test"}],
        backend="plotly",
    )

    trace = fig.data[0]
    np.testing.assert_allclose(trace.x, np.arange(5))
    assert fig.layout.xaxis4.title.text == "Time"


def test_plot_keplerian_trajectory_time_column_seconds_uses_hours():
    """[N x 7] input's time column must be elapsed seconds: a multi-day span
    (e.g. the MRO example's 2-day propagation) is then auto-labeled in
    hours, not mislabeled "seconds" from a caller pre-converting to hours."""
    n = 5
    dt = 12 * 3600.0  # 12 hours between samples -> 48 hour span
    koe = _raw_keplerian_array(n, dt)
    times_sec = np.arange(n) * dt
    koe_with_time = np.column_stack((times_sec, koe))

    fig = bh.plot_keplerian_trajectory(
        [{"trajectory": koe_with_time, "label": "Test"}],
        backend="plotly",
    )

    trace = fig.data[0]
    np.testing.assert_allclose(trace.x, times_sec / 3600.0)
    assert fig.layout.xaxis4.title.text == "Time (hours)"
