"""Paths, environment setup, and system-info collection for the GPU
comparison suite."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from pathlib import Path

from benchmarks.gpu_comparison.results import GPUInfo, SystemInfo

FRAMEWORK_DIR = Path(__file__).parent
REPO_ROOT = FRAMEWORK_DIR.parent.parent

BRAHE_EOP_FILE = REPO_ROOT / "data" / "eop" / "finals.all.iau2000.txt"
BRAHE_SPACE_WEATHER_FILE = REPO_ROOT / "data" / "space_weather" / "sw19571001.txt"
BRAHE_GRAVITY_FILE = REPO_ROOT / "data" / "gravity_models" / "EGM2008_120.gfc"

RESULTS_DIR = FRAMEWORK_DIR / "results"


def set_data_alignment_env() -> None:
    os.environ.setdefault("BRAHE_EOP_FILE", str(BRAHE_EOP_FILE))
    os.environ.setdefault("BRAHE_SPACE_WEATHER_FILE", str(BRAHE_SPACE_WEATHER_FILE))
    os.environ.setdefault("BRAHE_GRAVITY_FILE", str(BRAHE_GRAVITY_FILE))


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def data_alignment_record() -> dict[str, str]:
    return {
        "eop_file": str(BRAHE_EOP_FILE),
        "eop_sha256": sha256_of(BRAHE_EOP_FILE),
        "space_weather_file": str(BRAHE_SPACE_WEATHER_FILE),
        "space_weather_sha256": sha256_of(BRAHE_SPACE_WEATHER_FILE),
        "gravity_model_file": str(BRAHE_GRAVITY_FILE),
        "gravity_model_sha256": sha256_of(BRAHE_GRAVITY_FILE),
    }


def _detect_cpu_model() -> str:
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/cpuinfo").read_text().splitlines():
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        except OSError:
            pass
    if sys.platform == "darwin":
        try:
            r = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if r.returncode == 0:
                return r.stdout.strip()
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return platform.processor() or "unknown"


def _detect_physical_cores() -> int:
    try:
        import psutil

        n = psutil.cpu_count(logical=False)
        if n:
            return int(n)
    except ImportError:
        pass
    return os.cpu_count() or 1


def _detect_ram_gb() -> int:
    try:
        import psutil

        return int(psutil.virtual_memory().total / (1024**3))
    except ImportError:
        pass
    if sys.platform.startswith("linux"):
        try:
            for line in Path("/proc/meminfo").read_text().splitlines():
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb // (1024 * 1024)
        except OSError:
            pass
    return 0


def _detect_cuda_version_from_smi() -> str:
    """Parse the 'CUDA Version: X.Y' field from `nvidia-smi -q`."""
    try:
        r = subprocess.run(
            ["nvidia-smi", "-q"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode != 0:
            return "unknown"
        for line in r.stdout.splitlines():
            line = line.strip()
            if line.startswith("CUDA Version"):
                return line.split(":", 1)[1].strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return "unknown"


def _detect_gpus() -> list[GPUInfo]:
    try:
        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,memory.total,driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if r.returncode != 0:
            return []
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return []
    cuda_version = _detect_cuda_version_from_smi()
    gpus: list[GPUInfo] = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        try:
            gpus.append(
                GPUInfo(
                    index=int(parts[0]),
                    model=parts[1],
                    memory_mb=int(parts[2]),
                    driver=parts[3],
                    cuda_runtime=cuda_version,
                )
            )
        except ValueError:
            continue
    return gpus


def _detect_rust_version() -> str:
    try:
        r = subprocess.run(
            ["rustc", "--version"], capture_output=True, text=True, timeout=5
        )
        return r.stdout.strip() if r.returncode == 0 else "unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"


def _detect_jax_version() -> str:
    try:
        import jax

        return jax.__version__
    except ImportError:
        return "not installed"


def _detect_brahe_version() -> str:
    try:
        from importlib.metadata import version

        return version("brahe")
    except Exception:
        return "unknown"


def _detect_astrojax_version() -> str:
    try:
        from importlib.metadata import version

        return version("astrojax")
    except Exception:
        return "unknown"


def _git_sha(repo_path: Path) -> str | None:
    try:
        r = subprocess.run(
            ["git", "-C", str(repo_path), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if r.returncode != 0:
            return None
        return r.stdout.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None


def _astrojax_repo_path() -> Path | None:
    try:
        import astrojax

        p = Path(astrojax.__file__).resolve()
        for parent in p.parents:
            if (parent / ".git").exists():
                return parent
    except ImportError:
        pass
    return None


def _detect_rayon_threads() -> int:
    val = os.environ.get("RAYON_NUM_THREADS")
    if val and val.isdigit():
        return int(val)
    return os.cpu_count() or 1


def collect_system_info() -> SystemInfo:
    astrojax_repo = _astrojax_repo_path()
    return SystemInfo(
        cpu_model=_detect_cpu_model(),
        cpu_physical_cores=_detect_physical_cores(),
        cpu_logical_cores=os.cpu_count() or 1,
        ram_gb=_detect_ram_gb(),
        os=f"{platform.system()} {platform.release()}",
        python_version=platform.python_version(),
        rust_version=_detect_rust_version(),
        brahe_version=_detect_brahe_version(),
        brahe_git_sha=_git_sha(REPO_ROOT),
        astrojax_version=_detect_astrojax_version(),
        astrojax_git_sha=_git_sha(astrojax_repo) if astrojax_repo else None,
        jax_version=_detect_jax_version(),
        gpus=_detect_gpus(),
        rayon_threads=_detect_rayon_threads(),
    )
