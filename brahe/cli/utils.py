import re
import math
from typing import Union
import brahe


def set_cli_eop():
    """Initialize EOP data for CLI commands that require frame transformations."""
    brahe.initialize_eop()


def parse_numeric_expression(expr: Union[str, float]) -> float:
    """
    Parse a numeric expression that may contain brahe constants.

    Supports expressions like:
    - Simple numbers: 6878000.0
    - Scientific notation: 500e3
    - Expressions with constants: R_EARTH+500e3, 2*R_EARTH
    - Mathematical operations: +, -, *, /, **, ()

    Args:
        expr: Either a float or a string expression to parse

    Returns:
        The evaluated numeric value

    Raises:
        ValueError: If the expression is invalid or unsafe

    Examples:
        >>> parse_numeric_expression(500000.0)
        500000.0
        >>> parse_numeric_expression("R_EARTH+500e3")
        6878137.0
        >>> parse_numeric_expression("2*R_EARTH")
        12756274.0
    """
    # If already a number, return it
    if isinstance(expr, (int, float)):
        return float(expr)

    # Convert to string and strip whitespace
    expr_str = str(expr).strip()

    # Try to parse as simple float first
    try:
        return float(expr_str)
    except ValueError:
        pass

    # Build a safe namespace with brahe constants and math functions
    safe_namespace = {
        # Math functions
        "abs": abs,
        "sqrt": math.sqrt,
        "sin": math.sin,
        "cos": math.cos,
        "tan": math.tan,
        "asin": math.asin,
        "acos": math.acos,
        "atan": math.atan,
        "atan2": math.atan2,
        "exp": math.exp,
        "log": math.log,
        "log10": math.log10,
        "pow": pow,
        "pi": math.pi,
        "e": math.e,
        # Brahe constants - mathematical
        "DEG2RAD": brahe.DEG2RAD,
        "RAD2DEG": brahe.RAD2DEG,
        "AS2RAD": brahe.AS2RAD,
        "RAD2AS": brahe.RAD2AS,
        # Brahe constants - time
        "MJD_ZERO": brahe.MJD_ZERO,
        "MJD2000": brahe.MJD2000,
        "GPS_TAI": brahe.GPS_TAI,
        "TAI_GPS": brahe.TAI_GPS,
        "TT_TAI": brahe.TT_TAI,
        "TAI_TT": brahe.TAI_TT,
        "GPS_TT": brahe.GPS_TT,
        "TT_GPS": brahe.TT_GPS,
        "GPS_ZERO": brahe.GPS_ZERO,
        # Brahe constants - physical
        "C_LIGHT": brahe.C_LIGHT,
        "AU": brahe.AU,
        # Brahe constants - Earth
        "R_EARTH": brahe.R_EARTH,
        "WGS84_A": brahe.WGS84_A,
        "WGS84_F": brahe.WGS84_F,
        "GM_EARTH": brahe.GM_EARTH,
        "ECC_EARTH": brahe.ECC_EARTH,
        "J2_EARTH": brahe.J2_EARTH,
        "OMEGA_EARTH": brahe.OMEGA_EARTH,
        # Brahe constants - solar
        "GM_SUN": brahe.GM_SUN,
        "R_SUN": brahe.R_SUN,
        "P_SUN": brahe.P_SUN,
        # Brahe constants - lunar
        "R_MOON": brahe.R_MOON,
        "GM_MOON": brahe.GM_MOON,
        # Brahe constants - planetary
        "GM_MERCURY": brahe.GM_MERCURY,
        "GM_VENUS": brahe.GM_VENUS,
        "GM_MARS": brahe.GM_MARS,
        "GM_JUPITER": brahe.GM_JUPITER,
        "GM_SATURN": brahe.GM_SATURN,
        "GM_URANUS": brahe.GM_URANUS,
        "GM_NEPTUNE": brahe.GM_NEPTUNE,
        "GM_PLUTO": brahe.GM_PLUTO,
    }

    # Validate expression doesn't contain dangerous patterns
    # Only allow: numbers, operators, parentheses, constants, and whitespace
    if not re.match(r"^[0-9+\-*/().\s\w]+$", expr_str):
        raise ValueError(f"Invalid characters in expression: {expr_str}")

    # Check for common unsafe patterns
    unsafe_patterns = ["__", "import", "exec", "eval", "open", "file"]
    expr_lower = expr_str.lower()
    for pattern in unsafe_patterns:
        if pattern in expr_lower:
            raise ValueError(f"Unsafe pattern detected in expression: {expr_str}")

    # Evaluate the expression
    try:
        result = eval(expr_str, {"__builtins__": {}}, safe_namespace)

        # Validate result is a number
        if not isinstance(result, (int, float)):
            raise ValueError(
                f"Expression must evaluate to a number, got {type(result)}"
            )

        # Check for NaN or infinity
        if not math.isfinite(result):
            raise ValueError(f"Expression evaluated to non-finite value: {result}")

        return float(result)

    except Exception as e:
        raise ValueError(f"Failed to evaluate expression '{expr_str}': {e}")
