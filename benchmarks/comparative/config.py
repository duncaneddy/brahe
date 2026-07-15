"""
Configuration and system info for comparative benchmarks.
"""

import os
import platform
import shutil
import subprocess
from pathlib import Path

# Paths
FRAMEWORK_DIR = Path(__file__).parent
IMPLEMENTATIONS_DIR = FRAMEWORK_DIR / "implementations"
RESULTS_DIR = FRAMEWORK_DIR / "results"
FIGURES_DIR = Path(__file__).parent.parent.parent / "docs" / "figures"

# Rust binary path
RUST_BINARY = IMPLEMENTATIONS_DIR / "rust" / "target" / "release" / "bench_comparative"

# Nyx binary path
NYX_BINARY = IMPLEMENTATIONS_DIR / "nyx" / "target" / "release" / "bench_nyx"

# Java jar path
JAVA_PROJECT_DIR = IMPLEMENTATIONS_DIR / "java"

# Brahe-bundled gravity model file (EGM2008 at degree 120, ICGEM format).
# The Java/Orekit and any other backend that supports ICGEM read this file
# so the spherical-harmonic gravity coefficients used during the
# ``RK4 + N×N + drag + SRP`` propagation tasks are byte-identical to what
# brahe integrates against. Without this, Orekit defaults to EIGEN-6S and
# the test conflates implementation differences with coefficient-source
# differences.
REPO_ROOT = FRAMEWORK_DIR.parent.parent
BRAHE_GRAVITY_FILE = REPO_ROOT / "data" / "gravity_models" / "EGM2008_120.gfc"

# Export to the process environment so any subprocess (Java, Rust, GMAT)
# launched downstream inherits the same alignment. If a user has already
# set the env var explicitly, leave it alone.
if BRAHE_GRAVITY_FILE.exists() and "BRAHE_GRAVITY_FILE" not in os.environ:
    os.environ["BRAHE_GRAVITY_FILE"] = str(BRAHE_GRAVITY_FILE)

# Defaults
DEFAULT_ITERATIONS = 100
DEFAULT_SEED = 42


def _read_lockfile_version(lock_path: Path, pkg: str) -> str:
    """Parse `name = "<pkg>"` + following `version = "..."` from a Cargo.lock.

    Returns "not installed" if the lockfile does not exist (e.g. the
    user has not run `just bench-compare-setup` yet), or "unknown" if
    the lockfile exists but the package is not present in it.
    """
    if not lock_path.exists():
        return "not installed"
    text = lock_path.read_text()
    needle = f'name = "{pkg}"'
    idx = text.find(needle)
    if idx < 0:
        return "unknown"
    tail = text[idx:]
    v_idx = tail.find('version = "')
    if v_idx < 0:
        return "unknown"
    v_start = v_idx + len('version = "')
    v_end = tail[v_start:].find('"')
    if v_end < 0:
        return "unknown"
    return tail[v_start : v_start + v_end]


def collect_system_info() -> dict:
    """Collect system information for benchmark metadata."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": platform.os.cpu_count(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
    }

    # Nyx + ANISE versions, parsed from the bench_nyx Cargo.lock if present.
    info["nyx_version"] = _read_lockfile_version(
        IMPLEMENTATIONS_DIR / "nyx" / "Cargo.lock", "nyx-space"
    )
    info["anise_version"] = _read_lockfile_version(
        IMPLEMENTATIONS_DIR / "nyx" / "Cargo.lock", "anise"
    )

    # Rust version
    if shutil.which("rustc"):
        try:
            result = subprocess.run(
                ["rustc", "--version"], capture_output=True, text=True, timeout=5
            )
            info["rust_version"] = result.stdout.strip()
        except (subprocess.TimeoutExpired, FileNotFoundError):
            info["rust_version"] = "unknown"

    # Java version
    if shutil.which("java"):
        try:
            result = subprocess.run(
                ["java", "--version"], capture_output=True, text=True, timeout=5
            )
            info["java_version"] = result.stdout.strip().split("\n")[0]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            info["java_version"] = "unknown"

    # Brahe version
    try:
        import brahe

        info["brahe_version"] = getattr(brahe, "__version__", "unknown")
    except ImportError:
        info["brahe_version"] = "not installed"

    return info
