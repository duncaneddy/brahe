"""
Backend detection and initialization for plotting.

Handles detection of available backends (matplotlib, plotly) and optional styling (scienceplots).
"""

import importlib.util
from loguru import logger


def is_matplotlib_available():
    """Check if matplotlib is installed."""
    return importlib.util.find_spec("matplotlib") is not None


def is_plotly_available():
    """Check if plotly is installed."""
    return importlib.util.find_spec("plotly") is not None


def is_scienceplots_available():
    """Check if scienceplots is installed."""
    return importlib.util.find_spec("scienceplots") is not None


def validate_backend(backend):
    """Validate that the requested backend is available.

    Args:
        backend (str): The backend to validate ('matplotlib' or 'plotly')

    Raises:
        ImportError: If the requested backend is not available
        ValueError: If the backend name is invalid
    """
    if backend not in ["matplotlib", "plotly"]:
        raise ValueError(
            f"Invalid backend '{backend}'. Must be 'matplotlib' or 'plotly'"
        )

    if backend == "matplotlib" and not is_matplotlib_available():
        raise ImportError(
            "matplotlib is not installed. Install with: pip install matplotlib"
        )

    if backend == "plotly" and not is_plotly_available():
        raise ImportError("plotly is not installed. Install with: pip install plotly")


def is_latex_available():
    """Check if LaTeX is installed and available.

    Returns:
        bool: True if LaTeX is available, False otherwise
    """
    import shutil

    # Check for common LaTeX executables
    return shutil.which("latex") is not None or shutil.which("pdflatex") is not None


def apply_scienceplots_style():
    """Apply scienceplots styling if available.

    Automatically detects if LaTeX is available and applies the appropriate style:
    - If LaTeX is available: applies 'science' style (with LaTeX rendering)
    - If LaTeX is not available: applies 'science' and 'no-latex' styles

    Returns:
        bool: True if scienceplots was applied, False otherwise
    """
    if is_scienceplots_available() and is_matplotlib_available():
        try:
            import matplotlib.pyplot as plt
            import scienceplots  # noqa: F401 - Import to register styles with matplotlib

            if is_latex_available():
                # LaTeX is available, use full science style
                plt.style.use("science")
                logger.debug("Applied scienceplots style with LaTeX support.")
            else:
                # No LaTeX, use no-latex variant
                plt.style.use(["science", "no-latex"])
                logger.debug("Applied scienceplots style without LaTeX support.")

            return True
        except Exception as e:
            logger.debug(f"Failed to apply scienceplots style: {e}")
            return False
    else:
        if not is_scienceplots_available():
            logger.debug("scienceplots not installed, skipping style application.")
        return False
