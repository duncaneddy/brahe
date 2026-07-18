"""Tests for star catalog Python bindings."""

import inspect

import pytest
import polars as pl

import brahe as bh
import brahe.datasets as datasets


# ─── Offline tests (surface checks only) ────────────────────────────
#
# Record constructors are not exposed to Python (records only come from a
# parsed catalog), so offline coverage is limited to import/signature checks.
# Functional behavior is covered by the network-gated tests below.


def test_star_catalog_namespace_importable():
    """Test that the star_catalog namespace and its members are importable."""
    assert hasattr(datasets, "star_catalog")
    ns = datasets.star_catalog

    for attr in (
        "get_fk5",
        "get_hipparcos",
        "get_tycho2",
        "FK5Record",
        "FK5Catalog",
        "HipparcosRecord",
        "HipparcosCatalog",
        "Tycho2Record",
        "Tycho2Catalog",
    ):
        assert hasattr(ns, attr), f"star_catalog namespace missing {attr}"


@pytest.mark.parametrize(
    "func",
    [
        "get_fk5",
        "get_hipparcos",
        "get_tycho2",
    ],
)
def test_star_catalog_get_functions_signature(func):
    """Test that catalog getters accept an optional cache_max_age kwarg."""
    f = getattr(datasets.star_catalog, func)
    sig = inspect.signature(f)
    assert "cache_max_age" in sig.parameters
    assert sig.parameters["cache_max_age"].default is None


def test_star_catalog_classes_have_no_public_constructor():
    """Test record classes only come from parsed catalogs (no __init__ exposed)."""
    for cls_name in (
        "FK5Record",
        "FK5Catalog",
        "HipparcosRecord",
        "HipparcosCatalog",
        "Tycho2Record",
        "Tycho2Catalog",
    ):
        cls = getattr(datasets.star_catalog, cls_name)
        with pytest.raises(TypeError):
            cls()


# ─── Network tests (live downloads, CI-gated) ───────────────────────
#
# No Tycho-2 network test here: the full catalog is ~526 MB (mirrors the
# Rust-side `test_tycho2_full_download`, which is likewise gated behind the
# `integration` feature and not run as part of normal CI).


@pytest.mark.integration
def test_fk5_catalog_load():
    """Test downloading, parsing, and querying the FK5 catalog."""
    cat = datasets.star_catalog.get_fk5()
    assert len(cat) == 1535
    assert repr(cat).startswith("FK5Catalog(")

    rec = cat.get_by_id(699)
    assert rec is not None
    assert rec.fk5_id == 699
    assert rec.id() == "FK5 699"

    df = cat.to_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert df.height == 1535


@pytest.mark.integration
def test_fk5_radec_at_epoch_nonzero_tau():
    """Test radec_at_epoch forwards a record's fields to apply_proper_motion.

    Mirrors the Rust test of the same name: covers the tau != 0
    proper-motion-unit wiring seam by comparing radec_at_epoch's output
    against a direct apply_proper_motion call over the same interval.
    """
    cat = datasets.star_catalog.get_fk5()
    rec = cat.get_by_id(699)
    assert rec is not None

    # FK5 positions/proper motions are referred to J2000.0 (matches the
    # fixed epoch hardcoded in FK5Record::epoch on the Rust side).
    epoch_from = bh.Epoch.from_jd(2451545.0, bh.TimeSystem.TT)
    epoch_to = bh.Epoch.from_mjd(epoch_from.mjd() + 10.0 * 365.25, bh.TimeSystem.TT)

    ra, dec = rec.radec_at_epoch(epoch_to, bh.AngleFormat.DEGREES)

    expected_ra, expected_dec = bh.apply_proper_motion(
        rec.ra,
        rec.dec,
        rec.pm_ra,
        rec.pm_dec,
        rec.parallax,
        rec.radial_velocity,
        epoch_from,
        epoch_to,
        bh.AngleFormat.DEGREES,
    )

    assert ra == pytest.approx(expected_ra, abs=1e-12)
    assert dec == pytest.approx(expected_dec, abs=1e-12)


@pytest.mark.integration
def test_hipparcos_catalog_load():
    """Test downloading, parsing, and querying the Hipparcos catalog."""
    cat = datasets.star_catalog.get_hipparcos()
    assert len(cat) > 117_000

    sirius = cat.get_by_id(32349)
    assert sirius is not None
    assert sirius.vmag == pytest.approx(-1.44, abs=0.05)

    bright = cat.filter_by_magnitude(5.2)
    assert 1000 < len(bright) < 3000

    assert sirius.id() == "HIP 32349"

    ra, dec = sirius.radec_at_epoch(
        bh.Epoch.from_datetime(2026, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC),
        bh.AngleFormat.DEGREES,
    )
    # Small but nonzero proper-motion shift relative to the catalog epoch (J1991.25).
    assert abs(ra - sirius.ra) < 0.01
    assert ra != sirius.ra

    df = cat.to_dataframe()
    assert isinstance(df, pl.DataFrame)
    assert df.height == len(cat)
    assert df.width == 24
