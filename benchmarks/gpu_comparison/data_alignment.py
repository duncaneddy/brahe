"""Pin both backends to brahe's bundled EOP / space-weather files.

Per spike 01 (``spikes/01-eop-loader-compat.md``), astrojax exposes
``load_eop_from_file`` and ``load_sw_from_file`` as the public loaders, and
both accept brahe's bundled files directly. No format conversion is needed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any


def _enable_x64() -> None:
    """Set JAX_ENABLE_X64=1 before astrojax/jax is imported in this process."""
    os.environ.setdefault("JAX_ENABLE_X64", "1")


def load_eop_for_astrojax(path: Path) -> Any:
    """Return an astrojax EOP data object loaded from ``path``."""
    _enable_x64()
    from astrojax.eop import load_eop_from_file

    return load_eop_from_file(str(path))


def load_space_weather_for_astrojax(path: Path) -> Any:
    """Return an astrojax space-weather data object loaded from ``path``."""
    _enable_x64()
    from astrojax.space_weather import load_sw_from_file

    return load_sw_from_file(str(path))


def install_global_providers() -> None:
    """Install brahe-bundled providers as astrojax's global defaults if astrojax
    exposes set-globals.

    Cheap and idempotent. Called once per spawned-child entry and at the top of
    each in-process astrojax cell. Silent no-op if astrojax doesn't have the
    set-global APIs in this version.
    """
    _enable_x64()
    from benchmarks.gpu_comparison.config import (
        BRAHE_EOP_FILE,
        BRAHE_SPACE_WEATHER_FILE,
    )

    try:
        from astrojax.eop import set_global_eop  # type: ignore[attr-defined]

        set_global_eop(load_eop_for_astrojax(BRAHE_EOP_FILE))
    except (ImportError, AttributeError):
        pass
    try:
        from astrojax.space_weather import set_global_sw  # type: ignore[attr-defined]

        set_global_sw(load_space_weather_for_astrojax(BRAHE_SPACE_WEATHER_FILE))
    except (ImportError, AttributeError):
        pass
