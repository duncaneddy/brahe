"""Tests for optional plotting dependency boundary (issue #359).

These verify that `import brahe` does not eagerly pull the visualization stack,
that the documented top-level `bh.plot_*` API still resolves, and that a missing
optional dependency produces a clear, actionable error.
"""

import importlib

import pytest

import brahe


def test_import_brahe_succeeds():
    # Importing the top-level package must not require plotting dependencies.
    assert hasattr(brahe, "__version__")


def test_plots_subpackage_importable():
    # Importing the plots package itself must not trigger third-party imports.
    plots = importlib.import_module("brahe.plots")
    assert plots is not None


def test_top_level_plot_function_forwards_to_plots():
    # The documented `bh.plot_groundtrack(...)` pattern must keep working and
    # resolve to the same object as `brahe.plots.plot_groundtrack`.
    from brahe.plots import plot_groundtrack as direct

    assert brahe.plot_groundtrack is direct


def test_unknown_top_level_attribute_raises_attributeerror():
    with pytest.raises(AttributeError):
        brahe.this_attribute_does_not_exist


def test_unknown_plots_attribute_raises_attributeerror():
    import brahe.plots as plots

    with pytest.raises(AttributeError):
        plots.this_attribute_does_not_exist


def test_all_policy_keeps_core_starimport_plot_free():
    # `plots` is exported, but individual plot names are NOT, so that
    # `from brahe import *` stays free of the visualization stack.
    assert "plots" in brahe.__all__
    assert "plot_groundtrack" not in brahe.__all__


def test_friendly_error_when_submodule_unimportable():
    # The helper reframes any ImportError raised while loading a plot submodule
    # into an actionable install hint, preserving the original via __cause__.
    import brahe.plots as plots

    with pytest.raises(ImportError) as excinfo:
        plots._import_plot_submodule("a_submodule_that_does_not_exist")

    message = str(excinfo.value)
    assert "brahe[plots]" in message
    assert "brahe[all]" in message
    assert excinfo.value.__cause__ is not None


def test_friendly_error_propagates_through_top_level_access(monkeypatch):
    # Simulate a missing optional plotting dependency at submodule import time and
    # verify the friendly hint reaches the user through the real top-level access
    # path `brahe.plot_groundtrack` -> brahe.plots.__getattr__ -> _import_plot_submodule.
    import brahe.plots as plots

    real_import_module = plots.importlib.import_module

    def fake_import_module(modname, *args, **kwargs):
        if modname.startswith("brahe.plots."):
            raise ImportError("No module named 'cartopy'")
        return real_import_module(modname, *args, **kwargs)

    monkeypatch.setattr(plots.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError) as excinfo:
        brahe.plot_groundtrack

    message = str(excinfo.value)
    assert "brahe[plots]" in message
    assert "brahe[all]" in message
    assert excinfo.value.__cause__ is not None


def test_dir_brahe_includes_plot_symbols():
    # Plot names stay out of __all__ but should remain discoverable via dir().
    names = dir(brahe)
    assert "plot_groundtrack" in names
    assert "plots" in names
    # Core symbols remain present too.
    assert "Epoch" in names
