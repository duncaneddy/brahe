"""
Tests for linear algebra utilities.

Mirrors the Rust unit tests in src/math/linalg.rs (skew_symmetric,
block_diagonal) and src/math/covariance.rs (is_symmetric, symmetrize).
"""

import numpy as np
import pytest
from pytest import approx

import brahe as bh


def test_skew_symmetric():
    """Rust: test_skew_symmetric"""
    v = np.array([0.31, -1.7, 4.2])
    w = np.array([-2.4, 0.9, 3.3])

    s = bh.skew_symmetric(v)

    # [v]x w == v x w
    np.testing.assert_allclose(s @ w, np.cross(v, w), atol=1e-12, rtol=0)

    # The matrix is skew-symmetric and annihilates its own generator.
    assert np.linalg.norm(s + s.T) < 1e-12
    assert np.linalg.norm(s @ v) < 1e-12

    # Element layout
    assert s[0, 1] == approx(-v[2])
    assert s[0, 2] == approx(v[1])
    assert s[1, 2] == approx(-v[0])


def test_block_diagonal():
    """Rust: test_block_diagonal"""
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    b = np.array([[-1.0, -2.0, -3.0], [-4.0, -5.0, -6.0], [-7.0, -8.0, -9.0]])
    m = bh.block_diagonal(a, b)

    assert m.shape == (6, 6)
    np.testing.assert_array_equal(m[0:3, 0:3], a)
    np.testing.assert_array_equal(m[3:6, 3:6], b)
    np.testing.assert_array_equal(m[0:3, 3:6], np.zeros((3, 3)))
    np.testing.assert_array_equal(m[3:6, 0:3], np.zeros((3, 3)))


def test_is_symmetric():
    """Rust: test_is_symmetric"""
    symmetric = np.array(
        [
            [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            [2.0, 7.0, 8.0, 9.0, 10.0, 11.0],
            [3.0, 8.0, 12.0, 13.0, 14.0, 15.0],
            [4.0, 9.0, 13.0, 16.0, 17.0, 18.0],
            [5.0, 10.0, 14.0, 17.0, 19.0, 20.0],
            [6.0, 11.0, 15.0, 18.0, 20.0, 21.0],
        ]
    )
    assert bh.is_symmetric(symmetric, 1e-12)

    asymmetric = symmetric.copy()
    asymmetric[0, 1] = 100.0
    assert not bh.is_symmetric(asymmetric, 1e-12)

    non_square = np.zeros((3, 4))
    assert not bh.is_symmetric(non_square, 1e-12)


def test_symmetrize():
    """Rust: test_symmetrize"""
    m = np.array([[1.0, 0.4, -2.0], [0.6, 5.0, 3.0], [0.0, 1.0, 9.0]])
    s = bh.symmetrize(m)

    assert bh.is_symmetric(s, 1e-15)
    assert s[0, 1] == approx(0.5, abs=1e-15)
    assert s[0, 2] == approx(-1.0, abs=1e-15)
    assert s[1, 2] == approx(2.0, abs=1e-15)

    # Diagonal is untouched and the operation is idempotent.
    for i in range(3):
        assert s[i, i] == approx(m[i, i], abs=1e-15)
    np.testing.assert_allclose(bh.symmetrize(s), s, atol=1e-15, rtol=0)


def test_symmetrize_non_square_panics():
    """Rust: test_symmetrize_non_square_panics"""
    with pytest.raises(bh.PanicException, match="square"):
        bh.symmetrize(np.zeros((3, 4)))
