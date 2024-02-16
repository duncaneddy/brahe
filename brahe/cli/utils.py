from brahe import Epoch

def parse_float(s):
    try:
        num = float(s)
        return num
    except ValueError:
        return None

def epoch_from_epochlike(time: str) -> Epoch:
    """Attempts to convert a string to an Epoch object. Accepts any Epoch-like format.
    In particular can be a string, a modified Julian date, or a Julian date.

    Args:
        time (str): The string to convert to an Epoch object.

    Returns:
        brahe.Epoch: The Epoch object.
    """

    # Attempt to parse as MJD or JD
    if t := parse_float(time):
        # If less than 1000000, assume MJD
        if t < 1000000:
            return Epoch.from_mjd(t, "UTC")
        else:
            return Epoch.from_jd(t, "UTC")

    # Attempt to parse as string
    try:
        return Epoch.from_string(time)
    except ValueError:
        pass

    raise ValueError(f"Could not parse \"{time}\" as an Epoch-like object")