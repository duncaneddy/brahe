"""Tests for NAIF Python bindings."""

import pytest
import tempfile
import os
import brahe as bh


@pytest.mark.ci
def test_download_de440s(naif_cache_setup):
    """Test downloading de440s kernel (smaller file for testing)."""
    kernel_path = bh.datasets.naif.download_de_kernel("de440s")

    assert isinstance(kernel_path, str)
    assert os.path.exists(kernel_path)
    assert kernel_path.endswith("de440s.bsp")
    assert "naif" in kernel_path

    # Verify file is not empty
    file_size = os.path.getsize(kernel_path)
    assert file_size > 0


@pytest.mark.ci
def test_download_with_output_path(naif_cache_setup):
    """Test downloading kernel to specific location."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = os.path.join(tmpdir, "my_kernel.bsp")

        kernel_path = bh.datasets.naif.download_de_kernel("de440s", output_path)

        assert kernel_path == output_path
        assert os.path.exists(output_path)

        # Verify file is not empty
        file_size = os.path.getsize(output_path)
        assert file_size > 0


@pytest.mark.ci
def test_caching_behavior(naif_cache_setup):
    """Test that kernel is cached and not re-downloaded."""
    # First download
    kernel_path1 = bh.datasets.naif.download_de_kernel("de440s")
    mtime1 = os.path.getmtime(kernel_path1)

    # Second download - should use cache
    kernel_path2 = bh.datasets.naif.download_de_kernel("de440s")
    mtime2 = os.path.getmtime(kernel_path2)

    # Paths should be the same
    assert kernel_path1 == kernel_path2

    # Modification times should be identical (file wasn't re-downloaded)
    assert mtime1 == mtime2


def test_unsupported_kernel():
    """Test that unsupported kernel name raises error."""
    with pytest.raises(RuntimeError, match="Unsupported kernel name"):
        bh.datasets.naif.download_de_kernel("de999")


def test_supported_kernels():
    """Test that all documented supported kernels are valid."""
    # Test a subset of supported kernels (not all to save bandwidth)
    # These should not raise validation errors
    for kernel in ["de430", "de440s"]:
        # Should not raise during validation
        # (will fail later if network is down, but that's OK for this test)
        try:
            kernel_path = bh.datasets.naif.download_de_kernel(kernel)
            assert os.path.exists(kernel_path)
        except RuntimeError as e:
            # Only allow network errors, not validation errors
            assert "Unsupported kernel" not in str(e)


@pytest.mark.ci
def test_output_path_creates_directories(naif_cache_setup):
    """Test that output_path creates parent directories if needed."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create nested directory path that doesn't exist yet
        output_path = os.path.join(tmpdir, "subdir", "nested", "my_kernel.bsp")

        kernel_path = bh.datasets.naif.download_de_kernel("de440s", output_path)

        assert kernel_path == output_path
        assert os.path.exists(output_path)
        assert os.path.exists(os.path.dirname(output_path))

        # Verify file is not empty
        file_size = os.path.getsize(output_path)
        assert file_size > 0
