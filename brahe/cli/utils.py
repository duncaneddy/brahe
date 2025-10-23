import math
import brahe


def set_cli_eop():
    """Initialize EOP data for CLI commands that require frame transformations."""
    brahe.initialize_eop()


def get_time_string(t: float, short: bool = False) -> str:
    """
    Convert a time in seconds to a human-readable string.

    Args:
        t: Time duration in seconds
        short: Use short format (e.g., "6m 2s") instead of long format

    Returns:
        Human-readable time string
    """
    if short:
        # Short format: e.g., "6m 2s", "1h 30m 15s", "2d 3h 45m"
        days = math.floor(t / 86400)
        hours = math.floor(t / 3600) % 24
        minutes = math.floor(t / 60) % 60
        seconds = t % 60

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        if seconds > 0 or len(parts) == 0:
            parts.append(f"{seconds:.0f}s")

        return " ".join(parts)
    else:
        # Long format: e.g., "6 minutes and 2.00 seconds"
        if t < 60:
            return f"{t:.2f} seconds"
        elif t < 3600:
            return f"{math.floor(t / 60)} minutes and {t % 60:.2f} seconds"
        elif t < 86400:
            return (
                f"{math.floor(t / 3600)} hours, {math.floor(t / 60) % 60} minutes, "
                f"and {t % 60:.2f} seconds"
            )
        else:
            return (
                f"{math.floor(t / 86400)} days, {math.floor(t / 3600) % 24} hours, "
                f"{math.floor(t / 60) % 60} minutes, and {t % 60:.2f} seconds"
            )
