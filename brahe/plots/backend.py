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


def apply_scienceplots_style(dark_mode=False):
    """Apply scienceplots styling if available, with optional dark mode.

    Composes all requested styles as a single list so they layer properly.
    When *dark_mode* is ``True`` the ``"dark_background"`` style is appended
    **after** the scienceplots styles, preserving the science formatting while
    switching to a dark colour scheme.

    Args:
        dark_mode (bool): If True, append ``"dark_background"`` to the style
            list so it composes with scienceplots rather than overriding it.

    Returns:
        bool: True if any styles were applied, False otherwise.
    """
    import matplotlib.pyplot as plt

    styles = []

    if is_scienceplots_available() and is_matplotlib_available():
        try:
            import scienceplots  # noqa: F401 - Import to register styles with matplotlib

            styles.append("science")
            if not is_latex_available():
                styles.append("no-latex")
            styles.append("ieee")
        except Exception as e:
            logger.debug(f"Failed to load scienceplots: {e}")

    if dark_mode:
        styles.append("dark_background")

    if styles:
        plt.style.use(styles)
        logger.debug(f"Applied matplotlib styles: {styles}")
        return True

    logger.debug("No matplotlib styles applied.")
    return False
