"""
Configuration and system info for comparative benchmarks.
"""

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

# Java jar path
JAVA_PROJECT_DIR = IMPLEMENTATIONS_DIR / "java"

# Defaults
DEFAULT_ITERATIONS = 100
DEFAULT_SEED = 42


def collect_system_info() -> dict:
    """Collect system information for benchmark metadata."""
    info = {
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": platform.os.cpu_count(),
        "python_version": platform.python_version(),
        "machine": platform.machine(),
    }

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
