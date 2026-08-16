"""Tests for the Gabbard diagram plot."""

import numpy as np
import pytest

import brahe as bh


class _FailingPropagator:
    """Stands in for a propagator whose trajectory does not cover the epoch."""

    def __init__(self, exc):
        self._exc = exc

    def state_koe_osc(self, epoch, angle_format):
        raise self._exc


def _keplerian_objects(n=3):
    return [
        np.array(
            [
                bh.R_EARTH + 500e3 + 50e3 * i,
                0.001 + 0.01 * i,
                np.radians(97.8),
                0.0,
                0.0,
                0.0,
            ]
        )
        for i in range(n)
    ]


def _matplotlib_point_count(fig):
    return sum(len(coll.get_offsets()) for coll in fig.axes[0].collections)


def _plotly_point_count(fig):
    return sum(len(trace.x) for trace in fig.data)


@pytest.mark.parametrize(
    "exc",
    [
        bh.BraheError("epoch outside trajectory"),
        RuntimeError("epoch outside trajectory"),
    ],
    ids=["BraheError", "RuntimeError"],
)
def test_gabbard_matplotlib_skips_objects_without_state(exc):
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.UTC)
    valid = _keplerian_objects()
    groups = [{"objects": [*valid, _FailingPropagator(exc)], "format": "Keplerian"}]

    fig = bh.plot_gabbard_diagram(groups, epoch=epoch, backend="matplotlib")

    # Apogee and perigee scatters, one point per valid object each
    assert _matplotlib_point_count(fig) == 2 * len(valid)


@pytest.mark.parametrize(
    "exc",
    [
        bh.BraheError("epoch outside trajectory"),
        RuntimeError("epoch outside trajectory"),
    ],
    ids=["BraheError", "RuntimeError"],
)
def test_gabbard_plotly_skips_objects_without_state(exc):
    epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.UTC)
    valid = _keplerian_objects()
    groups = [{"objects": [*valid, _FailingPropagator(exc)], "format": "Keplerian"}]

    fig = bh.plot_gabbard_diagram(groups, epoch=epoch, backend="plotly")

    assert _plotly_point_count(fig) == 2 * len(valid)


def test_gabbard_all_valid_objects_plot_every_point():
    valid = _keplerian_objects(4)
    fig = bh.plot_gabbard_diagram(
        [{"objects": valid, "format": "Keplerian"}], backend="matplotlib"
    )
    assert _matplotlib_point_count(fig) == 2 * len(valid)
