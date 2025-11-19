# flake8: noqa: PYI021
def anomaly_eccentric_to_mean(anm_ecc_or_oe, e=None, *, angle_format):
    """
    Converts eccentric anomaly into mean anomaly.

    Args:
        anm_ecc_or_oe (float or array): Either the eccentric anomaly, or a 6-element
            Keplerian elements array [a, e, i, Ω, ω, E] from which `e` and `E` will be extracted.
            The anomaly in the vector should match the `angle_format`.
        e (float, optional): The eccentricity. Required if `anm_ecc_or_oe` is a scalar, ignored if vector.
        angle_format (AngleFormat): (keyword-only) Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.

    Returns:
        float: Mean anomaly in radians or degrees.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Using scalar parameters
        E = np.pi / 4  # 45 degrees eccentric anomaly
        e = 0.1  # eccentricity
        M = bh.anomaly_eccentric_to_mean(E, e, bh.AngleFormat.RADIANS)
        print(f"Mean anomaly: {M:.4f} radians")

        # Using Keplerian elements vector (with eccentric anomaly at index 5)
        oe = [bh.R_EARTH + 500e3, 0.1, np.radians(45), 0, 0, np.pi/4]
        M = bh.anomaly_eccentric_to_mean(oe, angle_format=bh.AngleFormat.RADIANS)
        print(f"Mean anomaly: {M:.4f} radians")
        ```
    """

def anomaly_eccentric_to_true(anm_ecc_or_oe, e=None, *, angle_format):
    """
    Converts eccentric anomaly into true anomaly.

    Args:
        anm_ecc_or_oe (float or array): Either the eccentric anomaly, or a 6-element
            Keplerian elements array [a, e, i, Ω, ω, E] from which `e` and `E` will be extracted.
            The anomaly in the vector should match the `angle_format`.
        e (float, optional): The eccentricity. Required if `anm_ecc_or_oe` is a scalar, ignored if vector.
        angle_format (AngleFormat): (keyword-only) Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.

    Returns:
        float: True anomaly in radians or degrees.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Using scalar parameters
        E = np.pi / 4  # 45 degrees eccentric anomaly
        e = 0.4  # eccentricity
        nu = bh.anomaly_eccentric_to_true(E, e, bh.AngleFormat.RADIANS)
        print(f"True anomaly: {nu:.4f} radians")

        # Using Keplerian elements vector (with eccentric anomaly at index 5)
        oe = [bh.R_EARTH + 500e3, 0.4, np.radians(45), 0, 0, np.pi/4]
        nu = bh.anomaly_eccentric_to_true(oe, angle_format=bh.AngleFormat.RADIANS)
        print(f"True anomaly: {nu:.4f} radians")
        ```
    """

def anomaly_mean_to_eccentric(anm_mean_or_oe, e=None, *, angle_format):
    """
    Converts mean anomaly into eccentric anomaly.

    Args:
        anm_mean_or_oe (float or array): Either the mean anomaly, or a 6-element
            Keplerian elements array [a, e, i, Ω, ω, M] from which `e` and `M` will be extracted.
            The anomaly in the vector should match the `angle_format`.
        e (float, optional): The eccentricity. Required if `anm_mean_or_oe` is a scalar, ignored if vector.
        angle_format (AngleFormat): (keyword-only) Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.

    Returns:
        float: Eccentric anomaly in radians or degrees.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Using scalar parameters
        M = 1.5  # mean anomaly in radians
        e = 0.3  # eccentricity
        E = bh.anomaly_mean_to_eccentric(M, e, bh.AngleFormat.RADIANS)
        print(f"Eccentric anomaly: {E:.4f} radians")

        # Using Keplerian elements vector (with mean anomaly at index 5)
        oe = [bh.R_EARTH + 500e3, 0.3, np.radians(45), 0, 0, 1.5]
        E = bh.anomaly_mean_to_eccentric(oe, angle_format=bh.AngleFormat.RADIANS)
        print(f"Eccentric anomaly: {E:.4f} radians")
        ```
    """

def anomaly_mean_to_true(anm_mean_or_oe, e=None, *, angle_format):
    """
    Converts mean anomaly into true anomaly.

    Args:
        anm_mean_or_oe (float or array): Either the mean anomaly, or a 6-element
            Keplerian elements array [a, e, i, Ω, ω, M] from which `e` and `M` will be extracted.
            The anomaly in the vector should match the `angle_format`.
        e (float, optional): The eccentricity. Required if `anm_mean_or_oe` is a scalar, ignored if vector.
        angle_format (AngleFormat): (keyword-only) Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.

    Returns:
        float: True anomaly in radians or degrees.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Using scalar parameters
        M = 2.0  # mean anomaly in radians
        e = 0.25  # eccentricity
        nu = bh.anomaly_mean_to_true(M, e, bh.AngleFormat.RADIANS)
        print(f"True anomaly: {nu:.4f} radians")

        # Using Keplerian elements vector (with mean anomaly at index 5)
        oe = [bh.R_EARTH + 500e3, 0.25, np.radians(45), 0, 0, 2.0]
        nu = bh.anomaly_mean_to_true(oe, angle_format=bh.AngleFormat.RADIANS)
        print(f"True anomaly: {nu:.4f} radians")
        ```
    """

def anomaly_true_to_eccentric(anm_true_or_oe, e=None, *, angle_format):
    """
    Converts true anomaly into eccentric anomaly.

    Args:
        anm_true_or_oe (float or array): Either the true anomaly, or a 6-element
            Keplerian elements array [a, e, i, Ω, ω, ν] from which `e` and `ν` will be extracted.
            The anomaly in the vector should match the `angle_format`.
        e (float, optional): The eccentricity. Required if `anm_true_or_oe` is a scalar, ignored if vector.
        angle_format (AngleFormat): (keyword-only) Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.

    Returns:
        float: Eccentric anomaly in radians or degrees.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Using scalar parameters
        nu = np.pi / 3  # 60 degrees true anomaly
        e = 0.2  # eccentricity
        E = bh.anomaly_true_to_eccentric(nu, e, bh.AngleFormat.RADIANS)
        print(f"Eccentric anomaly: {E:.4f} radians")

        # Using Keplerian elements vector (with true anomaly at index 5)
        oe = [bh.R_EARTH + 500e3, 0.2, np.radians(45), 0, 0, np.pi/3]
        E = bh.anomaly_true_to_eccentric(oe, angle_format=bh.AngleFormat.RADIANS)
        print(f"Eccentric anomaly: {E:.4f} radians")
        ```
    """

def anomaly_true_to_mean(anm_true_or_oe, e=None, *, angle_format):
    """
    Converts true anomaly into mean anomaly.

    Args:
        anm_true_or_oe (float or array): Either the true anomaly, or a 6-element
            Keplerian elements array [a, e, i, Ω, ω, ν] from which `e` and `ν` will be extracted.
            The anomaly in the vector should match the `angle_format`.
        e (float, optional): The eccentricity. Required if `anm_true_or_oe` is a scalar, ignored if vector.
        angle_format (AngleFormat): (keyword-only) Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.

    Returns:
        float: Mean anomaly in radians or degrees.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Using scalar parameters
        nu = np.pi / 2  # 90 degrees true anomaly
        e = 0.15  # eccentricity
        M = bh.anomaly_true_to_mean(nu, e, bh.AngleFormat.RADIANS)
        print(f"Mean anomaly: {M:.4f} radians")

        # Using Keplerian elements vector (with true anomaly at index 5)
        oe = [bh.R_EARTH + 500e3, 0.15, np.radians(45), 0, 0, np.pi/2]
        M = bh.anomaly_true_to_mean(oe, angle_format=bh.AngleFormat.RADIANS)
        print(f"Mean anomaly: {M:.4f} radians")
        ```
    """

def apoapsis_altitude(a_or_oe, e=None, *, r_body):
    """
    Calculate the altitude above a body's surface at apoapsis.

    Args:
        a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
            Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` and `e` will be extracted.
        e (float, optional): The eccentricity. Required if `a_or_oe` is a scalar, ignored if vector.
        r_body (float): (keyword-only) The radius of the central body in meters.

    Returns:
        float: The altitude above the body's surface at apoapsis in meters.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Using scalar parameters
        a = bh.R_MOON + 100e3  # 100 km mean altitude
        e = 0.05  # moderate eccentricity
        alt_apo = bh.apoapsis_altitude(a, e, bh.R_MOON)
        print(f"Apoapsis altitude: {alt_apo/1000:.2f} km")

        # Using Keplerian elements vector
        oe = [bh.R_MOON + 100e3, 0.05, np.radians(30), 0, 0, 0]
        alt_apo = bh.apoapsis_altitude(oe, r_body=bh.R_MOON)
        print(f"Apoapsis altitude: {alt_apo/1000:.2f} km")
        ```
    """

def apoapsis_distance(a_or_oe, e=None):
    """
    Calculate the distance of an object at its apoapsis.

    Args:
        a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
            Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` and `e` will be extracted.
        e (float, optional): The eccentricity. Required if `a_or_oe` is a scalar, ignored if vector.

    Returns:
        float: The distance of the object at apoapsis in meters.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Using scalar parameters
        a = 8000000.0  # 8000 km semi-major axis
        e = 0.2  # moderate eccentricity
        r_apo = bh.apoapsis_distance(a, e)
        print(f"Apoapsis distance: {r_apo/1000:.2f} km")

        # Using Keplerian elements vector
        oe = [8000000.0, 0.2, np.radians(45), 0, 0, 0]
        r_apo = bh.apoapsis_distance(oe)
        print(f"Apoapsis distance: {r_apo/1000:.2f} km")
        ```
    """

def apoapsis_velocity(a_or_oe, e=None, *, gm):
    """
    Computes the apoapsis velocity of an astronomical object around a general body.

    Args:
        a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
            Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` and `e` will be extracted.
        e (float, optional): The eccentricity. Required if `a_or_oe` is a scalar, ignored if vector.
        gm (float): (keyword-only) The standard gravitational parameter of primary body in m³/s².

    Returns:
        float: The magnitude of velocity of the object at apoapsis in m/s.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Using scalar parameters
        a = 10000000.0  # 10000 km semi-major axis
        e = 0.3
        v_apo = bh.apoapsis_velocity(a, e, bh.GM_MARS)
        print(f"Apoapsis velocity: {v_apo/1000:.2f} km/s")

        # Using Keplerian elements vector
        oe = [10000000.0, 0.3, np.radians(30), 0, 0, 0]
        v_apo = bh.apoapsis_velocity(oe, gm=bh.GM_MARS)
        print(f"Apoapsis velocity: {v_apo/1000:.2f} km/s")
        ```
    """

def apogee_altitude(a_or_oe, e=None):
    """
    Calculate the altitude above Earth's surface at apogee.

    Args:
        a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
            Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` and `e` will be extracted.
        e (float, optional): The eccentricity. Required if `a_or_oe` is a scalar, ignored if vector.

    Returns:
        float: The altitude above Earth's surface at apogee in meters.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Using scalar parameters
        a = 26554000.0  # ~26554 km semi-major axis
        e = 0.7  # highly eccentric
        alt = bh.apogee_altitude(a, e)
        print(f"Apogee altitude: {alt/1000:.2f} km")

        # Using Keplerian elements vector
        oe = [26554000.0, 0.7, np.radians(63.4), 0, 0, 0]
        alt = bh.apogee_altitude(oe)
        print(f"Apogee altitude: {alt/1000:.2f} km")
        ```
    """

def apogee_velocity(a_or_oe, e=None):
    """
    Computes the apogee velocity of an astronomical object around Earth.

    Args:
        a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
            Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` and `e` will be extracted.
        e (float, optional): The eccentricity. Required if `a_or_oe` is a scalar, ignored if vector.

    Returns:
        float: The magnitude of velocity of the object at apogee in m/s.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Using scalar parameters
        a = 24400000.0  # meters
        e = 0.73  # high eccentricity
        v_apo = bh.apogee_velocity(a, e)
        print(f"Apogee velocity: {v_apo:.2f} m/s")

        # Using Keplerian elements vector
        oe = [24400000.0, 0.73, np.radians(7), 0, 0, 0]
        v_apo = bh.apogee_velocity(oe)
        print(f"Apogee velocity: {v_apo:.2f} m/s")
        ```
    """

def bias_eme2000():
    """
    Computes the frame bias matrix transforming GCRF (Geocentric Celestial Reference Frame)
    to EME2000 (Earth Mean Equator and Equinox of J2000.0).

    The bias matrix accounts for the small offset between the GCRF and the J2000.0 mean
    equator and equinox due to the difference in their definitions. This is a constant
    transformation that does not vary with time.

    Returns:
        numpy.ndarray: 3x3 rotation matrix transforming `GCRF` -> `EME2000`

    Example:
        ```python
        import brahe as bh

        # Get the bias matrix
        B = bh.bias_eme2000()
        print(f"Bias matrix shape: {B.shape}")
        # Output: Bias matrix shape: (3, 3)
        ```
    """

def bias_precession_nutation(epc):
    """
    Computes the Bias-Precession-Nutation matrix transforming the `GCRS` to the
    `CIRS` intermediate reference frame. This transformation corrects for the
    bias, precession, and nutation of Celestial Intermediate Origin (`CIO`) with
    respect to inertial space.

    This formulation computes the Bias-Precession-Nutation correction matrix
    according using a `CIO` based model using using the `IAU 2006`
    precession and `IAU 2000A` nutation models.

    The function will utilize the global Earth orientation and loaded data to
    apply corrections to the Celestial Intermediate Pole (`CIP`) derived from
    empirical observations.

    Args:
        epc (Epoch): Epoch instant for computation of transformation matrix

    Returns:
        (numpy.ndarray): 3x3 rotation matrix transforming `GCRS` -> `CIRS`

    References:
        IAU SOFA Tools For Earth Attitude, Example 5.5
        http://www.iausofa.org/2021_0512_C/sofa/sofa_pn_c.pdf
        Software Version 18, 2021-04-18
    """

def calculate_tle_line_checksum(line):
    """
    Calculate TLE line checksum.

    Args:
        line (str): TLE line.

    Returns:
        int: Checksum value.
    """

def create_tle_lines(
    epoch,
    inclination,
    raan,
    eccentricity,
    arg_perigee,
    mean_anomaly,
    mean_motion,
    norad_id,
    ephemeris_type,
    element_set_number,
    revolution_number,
    classification=None,
    intl_designator=None,
    first_derivative=None,
    second_derivative=None,
    bstar=None,
):
    """
    Create complete TLE lines from all parameters.

    Creates Two-Line Element (TLE) lines from complete set of orbital and administrative parameters.
    Provides full control over all TLE fields including derivatives and drag terms.

    Args:
        epoch (Epoch): Epoch of the elements.
        inclination (float): Inclination in degrees.
        raan (float): Right ascension of ascending node in degrees.
        eccentricity (float): Eccentricity (dimensionless).
        arg_perigee (float): Argument of periapsis in degrees.
        mean_anomaly (float): Mean anomaly in degrees.
        mean_motion (float): Mean motion in revolutions per day.
        norad_id (str): NORAD catalog number (supports numeric and Alpha-5 format).
        ephemeris_type (int): Ephemeris type (0-9).
        element_set_number (int): Element set number.
        revolution_number (int): Revolution number at epoch.
        classification (str, optional): Security classification. Defaults to ' '.
        intl_designator (str, optional): International designator. Defaults to ''.
        first_derivative (float, optional): First derivative of mean motion. Defaults to 0.0.
        second_derivative (float, optional): Second derivative of mean motion. Defaults to 0.0.
        bstar (float, optional): BSTAR drag term. Defaults to 0.0.

    Returns:
        tuple: A tuple containing (line1, line2) - the two TLE lines as strings.
    """

def datetime_to_jd(year, month, day, hour, minute, second, nanosecond):
    """
    Convert a Gregorian calendar date to the equivalent Julian Date.

    Note: Due to the ambiguity of the nature of leap second insertion, this
    method should not be used if a specific behavior for leap second insertion is expected.
    This method treats leap seconds as if they don't exist.

    Args:
        year (int): Year
        month (int): Month (1-12)
        day (int): Day of month (1-31)
        hour (int): Hour (0-23)
        minute (int): Minute (0-59)
        second (float): Second with fractional part
        nanosecond (float): Nanosecond component

    Returns:
        (float): Julian date of epoch

    Example:
        ```python
        import brahe as bh

        # Convert January 1, 2024 noon to Julian Date
        jd = bh.datetime_to_jd(2024, 1, 1, 12, 0, 0.0, 0.0)
        print(f"JD: {jd:.6f}")
        # Output: JD: 2460311.000000
        ```
    """

def datetime_to_mjd(year, month, day, hour, minute, second, nanosecond):
    """
    Convert a Gregorian calendar date to the equivalent Modified Julian Date.

    Note: Due to the ambiguity of the nature of leap second insertion, this
    method should not be used if a specific behavior for leap second insertion is expected.
    This method treats leap seconds as if they don't exist.

    Args:
        year (int): Year
        month (int): Month (1-12)
        day (int): Day of month (1-31)
        hour (int): Hour (0-23)
        minute (int): Minute (0-59)
        second (float): Second with fractional part
        nanosecond (float): Nanosecond component

    Returns:
        (float): Modified Julian date of epoch

    Example:
        ```python
        import brahe as bh

        # Convert January 1, 2024 noon to Modified Julian Date
        mjd = bh.datetime_to_mjd(2024, 1, 1, 12, 0, 0.0, 0.0)
        print(f"MJD: {mjd:.6f}")
        # Output: MJD: 60310.500000
        ```
    """

def download_c04_eop_file(filepath):
    """
    Download latest C04 Earth orientation parameter file. Will attempt to download the latest
    parameter file to the specified location. Creating any missing directories as required.

    The download source is the [IERS Earth Orientation Data Products](https://www.iers.org/IERS/EN/DataProducts/EarthOrientationData/eop.html)

    Args:
        filepath (str): Path of desired output file

    Example:
        ```python
        import brahe as bh

        # Download latest C04 EOP data
        bh.download_c04_eop_file("./eop_data/finals2000A.all.csv")
        ```
    """

def download_standard_eop_file(filepath):
    """
    Download latest standard Earth orientation parameter file. Will attempt to download the latest
    parameter file to the specified location. Creating any missing directories as required.

    The download source is the [IERS Earth Orientation Data Products](https://www.iers.org/IERS/EN/DataProducts/EarthOrientationData/eop.html)

    Args:
        filepath (str): Path of desired output file

    Example:
        ```python
        import brahe as bh

        # Download latest standard EOP data
        bh.download_standard_eop_file("./eop_data/standard_eop.txt")
        ```
    """

def earth_rotation(epc):
    """
    Computes the Earth rotation matrix transforming the `CIRS` to the `TIRS`
    intermediate reference frame. This transformation corrects for the Earth
    rotation.

    Args:
        epc (Epoch): Epoch instant for computation of transformation matrix

    Returns:
        (numpy.ndarray): 3x3 rotation matrix transforming `CIRS` -> `TIRS`
    """

def epoch_from_tle(line1):
    """
    Extract Epoch from TLE line 1

    Extracts and parses the epoch timestamp from the first line of TLE data.
    The epoch is returned in UTC time system.

    Args:
        line1 (str): First line of TLE data

    Returns:
        Epoch: Extracted epoch in UTC time system

    Examples:
        ```python
        line1 = "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997"
        epoch = epoch_from_tle(line1)
        epoch.year()
        ```
    """

def format_time_string(seconds, short=False):
    """
    Format a time duration in seconds to a human-readable string.

    Converts a duration in seconds to either a long format (e.g., "6 minutes and 2.00 seconds")
    or a short format (e.g., "6m 2s").

    Args:
        seconds (float): Time duration in seconds
        short (bool): If True, use short format; otherwise use long format (default: False)

    Returns:
        str: Human-readable string representation of the time duration

    Example:
        ```python
        import brahe as bh

        # Long format (default)
        print(bh.format_time_string(90.0))
        # Output: "1 minutes and 30.00 seconds"

        print(bh.format_time_string(3665.0))
        # Output: "1 hours, 1 minutes, and 5.00 seconds"

        # Short format
        print(bh.format_time_string(90.0, short=True))
        # Output: "1m 30s"

        print(bh.format_time_string(3665.0, short=True))
        # Output: "1h 1m 5s"
        ```
    """

def get_brahe_cache_dir():
    """
    Get the brahe cache directory path.

    The cache directory is determined by the `BRAHE_CACHE` environment variable.
    If not set, defaults to `~/.cache/brahe`.

    The directory is created if it doesn't exist.

    Returns:
        str: The full path to the cache directory.

    Raises:
        IOError: If the cache directory cannot be created or accessed.

    Example:
        ```python
        import brahe as bh

        cache_dir = bh.get_brahe_cache_dir()
        print(f"Cache directory: {cache_dir}")

        # You can also override with environment variable
        import os
        os.environ['BRAHE_CACHE'] = '/custom/cache/path'
        cache_dir = bh.get_brahe_cache_dir()
        ```

    Note:
        The directory will be created on first access if it doesn't exist.
    """

def get_brahe_cache_dir_with_subdir(subdirectory):
    """
    Get the brahe cache directory path with an optional subdirectory.

    The cache directory is determined by the `BRAHE_CACHE` environment variable.
    If not set, defaults to `~/.cache/brahe`. If a subdirectory is provided,
    it is appended to the cache path.

    The directory is created if it doesn't exist.

    Args:
        subdirectory (str or None): Optional subdirectory name to append to cache path.

    Returns:
        str: The full path to the cache directory (with subdirectory if provided).

    Raises:
        IOError: If the cache directory cannot be created or accessed.

    Example:
        ```python
        import brahe as bh

        # Get main cache directory
        cache_dir = bh.get_brahe_cache_dir_with_subdir(None)
        print(f"Cache: {cache_dir}")

        # Get custom subdirectory
        custom_cache = bh.get_brahe_cache_dir_with_subdir("my_data")
        print(f"Custom cache: {custom_cache}")
        ```

    Note:
        The directory (and subdirectory) will be created on first access if it doesn't exist.
    """

def get_celestrak_cache_dir():
    """
    Get the CelesTrak cache directory path.

    Returns the path to the CelesTrak cache subdirectory used for storing downloaded
    TLE data. Defaults to `~/.cache/brahe/celestrak` (or `$BRAHE_CACHE/celestrak`
    if environment variable is set).

    The directory is created if it doesn't exist.

    Returns:
        str: The full path to the CelesTrak cache directory.

    Raises:
        IOError: If the cache directory cannot be created or accessed.

    Example:
        ```python
        import brahe as bh

        celestrak_cache = bh.get_celestrak_cache_dir()
        print(f"CelesTrak cache: {celestrak_cache}")
        ```

    Note:
        The directory will be created on first access if it doesn't exist.
    """

def get_eop_cache_dir():
    """
    Get the EOP cache directory path.

    Returns the path to the EOP (Earth Orientation Parameters) cache subdirectory.
    Defaults to `~/.cache/brahe/eop` (or `$BRAHE_CACHE/eop` if environment variable is set).

    The directory is created if it doesn't exist.

    Returns:
        str: The full path to the EOP cache directory.

    Raises:
        IOError: If the cache directory cannot be created or accessed.

    Example:
        ```python
        import brahe as bh

        eop_cache = bh.get_eop_cache_dir()
        print(f"EOP cache: {eop_cache}")
        ```

    Note:
        The directory will be created on first access if it doesn't exist.
    """

def get_global_dxdy(mjd):
    """
    Get celestial pole offsets from the global EOP provider.

    Args:
        mjd (float): Modified Julian Date

    Returns:
        tuple[float, float]: Celestial pole offsets dx and dy in radians
    """

def get_global_eop(mjd):
    """
    Get all EOP parameters from the global EOP provider.

    Args:
        mjd (float): Modified Julian Date

    Returns:
        tuple[float, float, float, float, float, float]: UT1-UTC, pm_x, pm_y, dx, dy, lod
    """

def get_global_eop_extrapolation():
    """
    Get the extrapolation method of the global EOP provider.

    Returns:
        str: Extrapolation method string
    """

def get_global_eop_initialization():
    """
    Check if the global EOP provider is initialized.

    Returns:
        bool: True if global EOP provider is initialized
    """

def get_global_eop_interpolation():
    """
    Check if interpolation is enabled in the global EOP provider.

    Returns:
        bool: True if interpolation is enabled
    """

def get_global_eop_len():
    """
    Get the number of EOP data points in the global provider.

    Returns:
        int: Number of EOP data points
    """

def get_global_eop_mjd_last_dxdy():
    """
    Get the last Modified Julian Date with dx/dy data in the global provider.

    Returns:
        float: Last MJD with dx/dy data
    """

def get_global_eop_mjd_last_lod():
    """
    Get the last Modified Julian Date with LOD data in the global provider.

    Returns:
        float: Last MJD with LOD data
    """

def get_global_eop_mjd_max():
    """
    Get the maximum Modified Julian Date in the global EOP dataset.

    Returns:
        float: Maximum MJD
    """

def get_global_eop_mjd_min():
    """
    Get the minimum Modified Julian Date in the global EOP dataset.

    Returns:
        float: Minimum MJD
    """

def get_global_eop_type():
    """
    Get the EOP data type of the global provider.

    Returns:
        str: EOP type string
    """

def get_global_lod(mjd):
    """
    Get length of day offset from the global EOP provider.

    Args:
        mjd (float): Modified Julian Date

    Returns:
        float: Length of day offset in seconds
    """

def get_global_pm(mjd):
    """
    Get polar motion components from the global EOP provider.

    Args:
        mjd (float): Modified Julian Date

    Returns:
        tuple[float, float]: Polar motion x and y components in radians
    """

def get_global_ut1_utc(mjd):
    """
    Get UT1-UTC time difference from the global EOP provider.

    Args:
        mjd (float): Modified Julian Date

    Returns:
        float: UT1-UTC time difference in seconds
    """

def get_max_threads():
    """
    Get the current maximum number of threads for parallel computation.

    Returns the number of threads configured for the global thread pool.
    If the thread pool hasn't been initialized yet, this initializes it
    with the default (90% of available cores) and returns that value.

    Returns:
        int: Number of threads currently configured.

    Example:
        ```python
        import brahe as bh

        # Get default thread count (90% of cores, initialized on first call)
        threads = bh.get_max_threads()
        print(f"Default: {threads} threads")

        # Set to specific value and verify
        bh.set_num_threads(4)
        assert bh.get_max_threads() == 4

        # Reconfigure and verify again
        bh.set_num_threads(8)
        assert bh.get_max_threads() == 8

        # Switch to max cores
        bh.set_max_threads()
        print(f"Max cores: {bh.get_max_threads()}")
        ```

    Note:
        Calling this function will initialize the thread pool with default
        settings (90% of cores) if it hasn't been configured yet. After
        initialization, you can still reconfigure it using set_num_threads()
        or set_max_threads().
    """

def initialize_eop():
    """
    Initialize the global EOP provider with recommended default settings.

    This convenience function creates a CachingEOPProvider with sensible defaults
    and sets it as the global provider. The provider will:

    - Use StandardBulletinA EOP data format
    - Automatically download/update EOP files when older than 7 days
    - Use the default cache location (~/.cache/brahe/finals.all.iau2000.txt)
    - Enable interpolation for smooth EOP data transitions
    - Hold the last known EOP value when extrapolating beyond available data
    - NOT auto-refresh on every access (manual refresh required)

    This is the recommended way to initialize EOP data for most applications,
    balancing accuracy, performance, and ease of use.

    Raises:
        Exception: If file download or loading failed

    Example:
        ```python
        import brahe as bh

        # Initialize with recommended defaults
        bh.initialize_eop()

        # Now you can perform frame transformations that require EOP data
        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        pos_eci = [bh.R_EARTH + 500e3, 0.0, 0.0]
        pos_ecef = bh.position_eci_to_ecef(epoch, pos_eci)
        ```

    Example:
        ```python
        import brahe as bh

        # This is equivalent to:
        provider = bh.CachingEOPProvider(
            eop_type="StandardBulletinA",
            max_age_seconds=7 * 86400,
            auto_refresh=False,
            interpolate=True,
            extrapolate="Hold"
        )
        bh.set_global_eop_provider(provider)
        ```
    """

def jd_to_datetime(jd):
    """
    Convert a Julian Date to the equivalent Gregorian calendar date.

    Note: Due to the ambiguity of the nature of leap second insertion, this
    method should not be used if a specific behavior for leap second insertion is expected.
    This method treats leap seconds as if they don't exist.

    Args:
        jd (float): Julian date

    Returns:
        tuple: A tuple containing (year, month, day, hour, minute, second, nanosecond)

    Example:
        ```python
        import brahe as bh

        # Convert Julian Date to Gregorian calendar
        jd = 2460311.0
        year, month, day, hour, minute, second, nanosecond = bh.jd_to_datetime(jd)
        print(f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:06.3f}")
        # Output: 2024-01-01 12:00:00.000
        ```
    """

def keplerian_elements_from_tle(line1, line2):
    """
    Extract Keplerian orbital elements from TLE lines.

    Extracts the standard six Keplerian orbital elements from Two-Line Element (TLE) data.
    Returns elements in standard order: [a, e, i, raan, argp, M] where angles are in radians.

    Args:
        line1 (str): First line of TLE data.
        line2 (str): Second line of TLE data.

    Returns:
        tuple: A tuple containing:
            - epoch (Epoch): Epoch of the TLE data.
            - elements (numpy.ndarray): Six Keplerian elements [a, e, i, raan, argp, M] where
              a is semi-major axis in meters, e is eccentricity (dimensionless), and
              i, raan, argp, M are in radians.
    """

def keplerian_elements_to_tle(epoch, elements, norad_id):
    """
    Convert Keplerian elements to TLE lines.

    Converts standard Keplerian orbital elements to Two-Line Element (TLE) format.
    Input angles should be in degrees for compatibility with TLE format.

    Args:
        epoch (Epoch): Epoch of the elements.
        elements (numpy.ndarray): Keplerian elements [a (m), e, i (deg), raan (deg), argp (deg), M (deg)].
        norad_id (str): NORAD catalog number (supports numeric and Alpha-5 format).

    Returns:
        tuple: A tuple containing (line1, line2) - the two TLE lines as strings.
    """

def location_accesses(
    locations,
    propagators,
    search_start,
    search_end,
    constraint,
    property_computers=None,
    config=None,
    time_tolerance=None,
):
    """
    Compute access windows for locations and satellites.

    This function accepts either single items or lists for both locations and propagators,
    automatically handling all combinations. All location-satellite pairs are computed
    and results are returned sorted by window start time.

    Args:
        locations (PointLocation | PolygonLocation | List[PointLocation | PolygonLocation]):
            Single location or list of locations
        propagators (SGPPropagator | KeplerianPropagator | List[SGPPropagator | KeplerianPropagator]):
            Single propagator or list of propagators
        search_start (Epoch): Start of search window
        search_end (Epoch): End of search window
        constraint (AccessConstraint): Access constraints to evaluate
        property_computers (Optional[List[AccessPropertyComputer]]): Optional property computers
        config (Optional[AccessSearchConfig]): Search configuration (default: 60s fixed grid, no adaptation)
        time_tolerance (Optional[float]): Bisection search tolerance in seconds (default: 0.01)

    Returns:
        List[AccessWindow]: List of access windows sorted by start time

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Create a ground station
        station = bh.PointLocation(-75.0, 40.0, 0.0)  # Philadelphia

        # Create satellite propagators
        epoch = bh.Epoch(2024, 1, 1, 0, 0, 0.0)
        oe = np.array([bh.R_EARTH + 500e3, 0.01, 97.8, 15.0, 30.0, 45.0])
        state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)
        prop1 = bh.KeplerianPropagator(epoch, state)

        # Define access constraints
        constraint = bh.ElevationConstraint(10.0)  # 10 degree minimum elevation

        # Single location, single propagator
        search_end = epoch + 86400.0  # 1 day
        windows = bh.location_accesses(station, prop1, epoch, search_end, constraint)

        # Single location, multiple propagators
        prop2 = bh.KeplerianPropagator(epoch, state)  # Different satellite
        windows = bh.location_accesses(station, [prop1, prop2], epoch, search_end, constraint)

        # Multiple locations, single propagator
        station2 = bh.PointLocation(-122.0, 37.0, 0.0)  # San Francisco
        windows = bh.location_accesses([station, station2], prop1, epoch, search_end, constraint)

        # Multiple locations, multiple propagators
        windows = bh.location_accesses([station, station2], [prop1, prop2], epoch, search_end, constraint)

        # Custom search configuration
        config = bh.AccessSearchConfig(initial_time_step=30.0, adaptive_step=True)
        windows = bh.location_accesses(station, prop1, epoch, search_end, constraint, config=config)
        ```
    """

def mean_motion(a_or_oe, angle_format):
    """
    Computes the mean motion of an astronomical object around Earth.

    Args:
        a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
            Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` will be extracted.
        angle_format (AngleFormat): (keyword-only) Return output in AngleFormat.DEGREES or AngleFormat.RADIANS.

    Returns:
        float: The mean motion of the astronomical object in radians or degrees.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Using scalar semi-major axis
        a = bh.R_EARTH + 35786e3
        n = bh.mean_motion(a, bh.AngleFormat.DEGREES)
        print(f"Mean motion: {n:.6f} deg/s")

        # Using Keplerian elements vector
        oe = [bh.R_EARTH + 35786e3, 0.001, np.radians(0), 0, 0, 0]
        n = bh.mean_motion(oe, bh.AngleFormat.DEGREES)
        print(f"Mean motion: {n:.6f} deg/s")
        ```
    """

def mean_motion_general(a_or_oe, gm, angle_format):
    """
    Computes the mean motion of an astronomical object around a general body
    given a semi-major axis.

    Args:
        a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
            Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` will be extracted.
        gm (float): (keyword-only) The standard gravitational parameter of primary body in m³/s².
        angle_format (AngleFormat): (keyword-only) Return output in AngleFormat.DEGREES or AngleFormat.RADIANS.

    Returns:
        float: The mean motion of the astronomical object in radians or degrees.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Using scalar semi-major axis
        a = 4000000.0  # 4000 km semi-major axis
        n = bh.mean_motion_general(a, bh.GM_MARS, bh.AngleFormat.RADIANS)
        print(f"Mean motion: {n:.6f} rad/s")

        # Using Keplerian elements vector
        oe = [4000000.0, 0.01, np.radians(30), 0, 0, 0]
        n = bh.mean_motion_general(oe, bh.GM_MARS, bh.AngleFormat.RADIANS)
        print(f"Mean motion: {n:.6f} rad/s")
        ```
    """

def mjd_to_datetime(mjd):
    """
    Convert a Modified Julian Date to the equivalent Gregorian calendar date.

    Note: Due to the ambiguity of the nature of leap second insertion, this
    method should not be used if a specific behavior for leap second insertion is expected.
    This method treats leap seconds as if they don't exist.

    Args:
        mjd (float): Modified Julian date

    Returns:
        tuple: A tuple containing (year, month, day, hour, minute, second, nanosecond)

    Example:
        ```python
        import brahe as bh

        # Convert Modified Julian Date to Gregorian calendar
        mjd = 60310.5
        year, month, day, hour, minute, second, nanosecond = bh.mjd_to_datetime(mjd)
        print(f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:06.3f}")
        # Output: 2024-01-01 12:00:00.000
        ```
    """

def norad_id_alpha5_to_numeric(alpha5_id):
    """
    Convert Alpha-5 NORAD ID to numeric format.

    Args:
        alpha5_id (str): Alpha-5 format ID (e.g., "A0001").

    Returns:
        int: Numeric NORAD ID.
    """

def norad_id_numeric_to_alpha5(norad_id):
    """
    Convert numeric NORAD ID to Alpha-5 format or pass through if in legacy range.

    Args:
        norad_id (int): Numeric NORAD ID (0-339999). IDs 0-99999 are passed through
            as numeric strings. IDs 100000-339999 are converted to Alpha-5 format.

    Returns:
        str: For IDs 0-99999: numeric string (e.g., "42"). For IDs 100000-339999:
            Alpha-5 format ID (e.g., "A0001").
    """

def orbital_period(a_or_oe):
    """
    Computes the orbital period of an object around Earth.

    Uses rastro.constants.GM_EARTH as the standard gravitational parameter for the calculation.

    Args:
        a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
            Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` will be extracted.

    Returns:
        float: The orbital period of the astronomical object in seconds.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Using scalar semi-major axis
        a = bh.R_EARTH + 400e3
        period = bh.orbital_period(a)
        print(f"Orbital period: {period/60:.2f} minutes")

        # Using Keplerian elements vector
        oe = [bh.R_EARTH + 400e3, 0.001, np.radians(51.6), 0, 0, 0]
        period = bh.orbital_period(oe)
        print(f"Orbital period: {period/60:.2f} minutes")
        ```
    """

def orbital_period_from_state(state_eci, gm):
    """
    Computes orbital period from an ECI state vector using the vis-viva equation.

    This function uses the vis-viva equation to compute the semi-major axis from the
    position and velocity, then calculates the orbital period.

    Args:
        state_eci (np.ndarray): ECI state vector [x, y, z, vx, vy, vz] in meters and meters/second.
        gm (float): Gravitational parameter in m³/s². Use GM_EARTH for Earth orbits.

    Returns:
        float: Orbital period in seconds.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Create a circular orbit state at 500 km altitude
        r = bh.R_EARTH + 500e3
        v = np.sqrt(bh.GM_EARTH / r)
        state_eci = np.array([r, 0, 0, 0, v, 0])

        # Compute orbital period from state
        period = bh.orbital_period_from_state(state_eci, bh.GM_EARTH)
        print(f"Period: {period/60:.2f} minutes")
        ```
    """

def orbital_period_general(a_or_oe, gm):
    """
    Computes the orbital period of an astronomical object around a general body.

    Args:
        a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
            Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` will be extracted.
        gm (float): (keyword-only) The standard gravitational parameter of primary body in m³/s².

    Returns:
        float: The orbital period of the astronomical object in seconds.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Using scalar semi-major axis
        a = 1900000.0  # 1900 km semi-major axis
        period = bh.orbital_period_general(a, bh.GM_MOON)
        print(f"Lunar orbital period: {period/3600:.2f} hours")

        # Using Keplerian elements vector
        oe = [1900000.0, 0.01, np.radians(45), 0, 0, 0]
        period = bh.orbital_period_general(oe, bh.GM_MOON)
        print(f"Lunar orbital period: {period/3600:.2f} hours")
        ```
    """

def par_propagate_to(propagators, target_epoch):
    """
    Propagate multiple propagators to a target epoch in parallel.

    This function takes a list of propagators and calls `propagate_to` on each one
    in parallel using the global thread pool. Each propagator's internal state is updated
    to reflect the new epoch.

    All propagators in the list must be of the same type (either all `KeplerianPropagator`
    or all `SGPPropagator`). Mixing propagator types is not supported.

    Args:
        propagators (List[KeplerianPropagator] or List[SGPPropagator]): List of propagators to update.
        target_epoch (Epoch): The epoch to propagate all propagators to.

    Returns:
        None: Propagators are updated in place.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        bh.initialize_eop()

        epoch = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # Create multiple propagators
        propagators = []
        for i in range(10):
            oe = np.array([bh.R_EARTH + 500e3 + i*10e3, 0.001, 98.0, i*10.0, 0.0, 0.0])
            state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.DEGREES)
            prop = bh.KeplerianPropagator.from_eci(epoch, state, 60.0)
            propagators.append(prop)

        # Propagate all to target epoch in parallel
        target = epoch + 3600.0  # 1 hour later
        bh.par_propagate_to(propagators, target)

        # All propagators are now at target epoch
        for prop in propagators:
            assert prop.current_epoch() == target
        ```
    """

def parse_norad_id(norad_str):
    """
    Parse NORAD ID from string, handling both classic and Alpha-5 formats.

    Args:
        norad_str (str): NORAD ID string from TLE.

    Returns:
        int: Parsed numeric NORAD ID.
    """

def periapsis_altitude(a_or_oe, e=None, *, r_body):
    """
    Calculate the altitude above a body's surface at periapsis.

    Args:
        a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
            Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` and `e` will be extracted.
        e (float, optional): The eccentricity. Required if `a_or_oe` is a scalar, ignored if vector.
        r_body (float): (keyword-only) The radius of the central body in meters.

    Returns:
        float: The altitude above the body's surface at periapsis in meters.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Using scalar parameters
        a = bh.R_EARTH + 500e3  # 500 km mean altitude
        e = 0.01  # slight eccentricity
        alt_peri = bh.periapsis_altitude(a, e, bh.R_EARTH)
        print(f"Periapsis altitude: {alt_peri/1000:.2f} km")

        # Using Keplerian elements vector
        oe = [bh.R_EARTH + 500e3, 0.01, np.radians(45), 0, 0, 0]
        alt_peri = bh.periapsis_altitude(oe, r_body=bh.R_EARTH)
        print(f"Periapsis altitude: {alt_peri/1000:.2f} km")
        ```
    """

def periapsis_distance(a_or_oe, e=None):
    """
    Calculate the distance of an object at its periapsis.

    Args:
        a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
            Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` and `e` will be extracted.
        e (float, optional): The eccentricity. Required if `a_or_oe` is a scalar, ignored if vector.

    Returns:
        float: The distance of the object at periapsis in meters.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Using scalar parameters
        a = 8000000.0  # 8000 km semi-major axis
        e = 0.2  # moderate eccentricity
        r_peri = bh.periapsis_distance(a, e)
        print(f"Periapsis distance: {r_peri/1000:.2f} km")

        # Using Keplerian elements vector
        oe = [8000000.0, 0.2, np.radians(45), 0, 0, 0]
        r_peri = bh.periapsis_distance(oe)
        print(f"Periapsis distance: {r_peri/1000:.2f} km")
        ```
    """

def periapsis_velocity(a_or_oe, e=None, *, gm):
    """
    Computes the periapsis velocity of an astronomical object around a general body.

    Args:
        a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
            Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` and `e` will be extracted.
        e (float, optional): The eccentricity. Required if `a_or_oe` is a scalar, ignored if vector.
        gm (float): (keyword-only) The standard gravitational parameter of primary body in m³/s².

    Returns:
        float: The magnitude of velocity of the object at periapsis in m/s.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Using scalar parameters
        a = 5e11  # 5 AU semi-major axis (meters)
        e = 0.95  # highly elliptical
        v_peri = bh.periapsis_velocity(a, e, bh.GM_SUN)
        print(f"Periapsis velocity: {v_peri/1000:.2f} km/s")

        # Using Keplerian elements vector
        oe = [5e11, 0.95, np.radians(10), 0, 0, 0]
        v_peri = bh.periapsis_velocity(oe, gm=bh.GM_SUN)
        print(f"Periapsis velocity: {v_peri/1000:.2f} km/s")
        ```
    """

def perigee_altitude(a_or_oe, e=None):
    """
    Calculate the altitude above Earth's surface at perigee.

    Args:
        a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
            Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` and `e` will be extracted.
        e (float, optional): The eccentricity. Required if `a_or_oe` is a scalar, ignored if vector.

    Returns:
        float: The altitude above Earth's surface at perigee in meters.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Using scalar parameters
        a = bh.R_EARTH + 420e3  # 420 km mean altitude
        e = 0.0005  # very nearly circular
        alt = bh.perigee_altitude(a, e)
        print(f"Perigee altitude: {alt/1000:.2f} km")

        # Using Keplerian elements vector
        oe = [bh.R_EARTH + 420e3, 0.0005, np.radians(51.6), 0, 0, 0]
        alt = bh.perigee_altitude(oe)
        print(f"Perigee altitude: {alt/1000:.2f} km")
        ```
    """

def perigee_velocity(a_or_oe, e=None):
    """
    Computes the perigee velocity of an astronomical object around Earth.

    Args:
        a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
            Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` and `e` will be extracted.
        e (float, optional): The eccentricity. Required if `a_or_oe` is a scalar, ignored if vector.

    Returns:
        float: The magnitude of velocity of the object at perigee in m/s.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Using scalar parameters
        a = 26554000.0  # meters
        e = 0.72  # high eccentricity
        v_peri = bh.perigee_velocity(a, e)
        print(f"Perigee velocity: {v_peri:.2f} m/s")

        # Using Keplerian elements vector
        oe = [26554000.0, 0.72, np.radians(63.4), 0, 0, 0]
        v_peri = bh.perigee_velocity(oe)
        print(f"Perigee velocity: {v_peri:.2f} m/s")
        ```
    """

def polar_motion(epc):
    """
    Computes the Earth rotation matrix transforming the `TIRS` to the `ITRF` reference
    frame.

    The function will utilize the global Earth orientation and loaded data to
    apply corrections to compute the polar motion correction based on empirical
    observations of polar motion drift.

    Args:
        epc (Epoch): Epoch instant for computation of transformation matrix

    Returns:
        (numpy.ndarray): 3x3 rotation matrix transforming `TIRS` -> `ITRF`
    """

def position_ecef_to_eci(epc, x):
    """
    Transforms a position vector from the Earth Centered Earth Fixed (`ECEF`/`ITRF`)
    frame to the Earth Centered Inertial (`ECI`/`GCRF`) frame.

    This function is an alias for position_itrf_to_gcrf. Applies the full
    `IAU 2006/2000A` transformation including bias, precession, nutation, Earth
    rotation, and polar motion corrections using global Earth orientation parameters.

    Args:
        epc (Epoch): Epoch instant for the transformation
        x (numpy.ndarray or list): Position vector in `ECEF` frame (m), shape `(3,)`

    Returns:
        numpy.ndarray: Position vector in `ECI` frame (m), shape `(3,)`

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Create epoch
        epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # Position in ECEF (ground station)
        r_ecef = np.array([4000000.0, 3000000.0, 4000000.0])

        # Transform to ECI
        r_eci = bh.position_ecef_to_eci(epc, r_ecef)
        print(f"ECI position: {r_eci}")
        ```
    """

def position_ecef_to_geocentric(x_ecef, angle_format):
    """
    Convert `ECEF` Cartesian position to geocentric coordinates.

    Transforms a position from Earth-Centered Earth-Fixed (`ECEF`) Cartesian coordinates
    to geocentric spherical coordinates (longitude, latitude, radius).

    Args:
        x_ecef (numpy.ndarray or list): `ECEF` Cartesian position `[x, y, z]` in meters.
        angle_format (AngleFormat): Angle format for output angular coordinates (`RADIANS` or `DEGREES`).

    Returns:
        numpy.ndarray: Geocentric position `[longitude, latitude, radius]` where longitude
            is in radians or degrees, latitude is in radians or degrees, and radius is in meters.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Convert ECEF to geocentric coordinates
        x_ecef = np.array([6378137.0, 0.0, 0.0])  # Point on equator, prime meridian
        x_geoc = bh.position_ecef_to_geocentric(x_ecef, bh.AngleFormat.DEGREES)
        print(f"Geocentric: lon={x_geoc[0]:.2f}°, lat={x_geoc[1]:.2f}°, r={x_geoc[2]:.0f}m")
        ```
    """

def position_ecef_to_geodetic(x_ecef, angle_format):
    """
    Convert `ECEF` Cartesian position to geodetic coordinates.

    Transforms a position from Earth-Centered Earth-Fixed (`ECEF`) Cartesian coordinates
    to geodetic coordinates (longitude, latitude, altitude) using the `WGS84` ellipsoid model.

    Args:
        x_ecef (numpy.ndarray or list): `ECEF` Cartesian position `[x, y, z]` in meters.
        angle_format (AngleFormat): Angle format for output angular coordinates (`RADIANS` or `DEGREES`).

    Returns:
        numpy.ndarray: Geodetic position `[longitude, latitude, altitude]` where longitude
            is in radians or degrees, latitude is in radians or degrees, and altitude
            is in meters above the `WGS84` ellipsoid.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Convert ECEF to geodetic coordinates (GPS-like)
        x_ecef = np.array([-1275936.0, -4797210.0, 4020109.0])  # Example location
        x_geod = bh.position_ecef_to_geodetic(x_ecef, bh.AngleFormat.DEGREES)
        print(f"Geodetic: lon={x_geod[0]:.4f}°, lat={x_geod[1]:.4f}°, alt={x_geod[2]:.0f}m")
        ```
    """

def position_eci_to_ecef(epc, x):
    """
    Transforms a position vector from the Earth Centered Inertial (`ECI`/`GCRF`) frame
    to the Earth Centered Earth Fixed (`ECEF`/`ITRF`) frame.

    This function is an alias for position_gcrf_to_itrf. Applies the full
    `IAU 2006/2000A` transformation including bias, precession, nutation, Earth
    rotation, and polar motion corrections using global Earth orientation parameters.

    Args:
        epc (Epoch): Epoch instant for the transformation
        x (numpy.ndarray or list): Position vector in `ECI` frame (m), shape `(3,)`

    Returns:
        numpy.ndarray: Position vector in `ECEF` frame (m), shape `(3,)`

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Create epoch
        epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # Position vector in ECI (meters)
        r_eci = np.array([7000000.0, 0.0, 0.0])

        # Transform to ECEF
        r_ecef = bh.position_eci_to_ecef(epc, r_eci)
        print(f"ECEF position: {r_ecef}")
        ```
    """

def position_eme2000_to_gcrf(x):
    """
    Transforms a position vector from EME2000 (Earth Mean Equator and Equinox of J2000.0)
    to GCRF (Geocentric Celestial Reference Frame).

    Applies the inverse frame bias correction to account for the small offset between
    the J2000.0 mean equator and equinox and GCRF. This is a constant transformation
    that does not vary with time.

    Args:
        x (numpy.ndarray or list): Position vector in `EME2000` frame (m), shape `(3,)`

    Returns:
        numpy.ndarray: Position vector in `GCRF` frame (m), shape `(3,)`

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Position vector in EME2000 (meters)
        r_eme2000 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])

        # Transform to GCRF
        r_gcrf = bh.position_eme2000_to_gcrf(r_eme2000)
        print(f"GCRF position: {r_gcrf}")
        ```
    """

def position_enz_to_azel(x_enz, angle_format):
    """
    Convert position from East-North-Up (`ENZ`) frame to azimuth-elevation-range.

    Transforms a position from the local East-North-Up (`ENZ`) topocentric frame to
    azimuth-elevation-range spherical coordinates.

    Args:
        x_enz (numpy.ndarray or list): Position in `ENZ` frame `[east, north, up]` in meters.
        angle_format (AngleFormat): Angle format for output angular coordinates (`RADIANS` or `DEGREES`).

    Returns:
        numpy.ndarray: Azimuth-elevation-range `[azimuth, elevation, range]` where azimuth
            and elevation are in radians or degrees, and range is in meters.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Convert ENZ to azimuth-elevation for satellite tracking
        enz = np.array([50000.0, 100000.0, 200000.0])  # East, North, Up (meters)
        azel = bh.position_enz_to_azel(enz, bh.AngleFormat.DEGREES)
        print(f"Az={azel[0]:.1f}°, El={azel[1]:.1f}°, Range={azel[2]/1000:.1f}km")
        ```
    """

def position_gcrf_to_eme2000(x):
    """
    Transforms a position vector from GCRF (Geocentric Celestial Reference Frame)
    to EME2000 (Earth Mean Equator and Equinox of J2000.0).

    Applies the frame bias correction to account for the small offset between GCRF
    and the J2000.0 mean equator and equinox. This is a constant transformation
    that does not vary with time.

    Args:
        x (numpy.ndarray or list): Position vector in `GCRF` frame (m), shape `(3,)`

    Returns:
        numpy.ndarray: Position vector in `EME2000` frame (m), shape `(3,)`

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Position vector in GCRF (meters)
        r_gcrf = np.array([bh.R_EARTH + 500e3, 0.0, 0.0])

        # Transform to EME2000
        r_eme2000 = bh.position_gcrf_to_eme2000(r_gcrf)
        print(f"EME2000 position: {r_eme2000}")
        ```
    """

def position_gcrf_to_itrf(epc, x):
    """
    Transforms a position vector from GCRF (Geocentric Celestial Reference Frame)
    to ITRF (International Terrestrial Reference Frame).

    Applies the full `IAU 2006/2000A` transformation including bias, precession,
    nutation, Earth rotation, and polar motion corrections using global Earth
    orientation parameters.

    Args:
        epc (Epoch): Epoch instant for the transformation
        x (numpy.ndarray or list): Position vector in `GCRF` frame (m), shape `(3,)`

    Returns:
        numpy.ndarray: Position vector in `ITRF` frame (m), shape `(3,)`

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Create epoch
        epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # Position vector in GCRF (meters)
        r_gcrf = np.array([7000000.0, 0.0, 0.0])

        # Transform to ITRF
        r_itrf = bh.position_gcrf_to_itrf(epc, r_gcrf)
        print(f"ITRF position: {r_itrf}")
        ```
    """

def position_geocentric_to_ecef(x_geoc, angle_format):
    """
    Convert geocentric position to `ECEF` Cartesian coordinates.

    Transforms a position from geocentric spherical coordinates (longitude, latitude, radius)
    to Earth-Centered Earth-Fixed (`ECEF`) Cartesian coordinates.

    Args:
        x_geoc (numpy.ndarray or list): Geocentric position `[longitude, latitude, radius]` where
            longitude is in radians or degrees, latitude is in radians or degrees, and
            radius is in meters.
        angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).

    Returns:
        numpy.ndarray: `ECEF` Cartesian position `[x, y, z]` in meters.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Convert geocentric coordinates to ECEF
        lon, lat, r = 0.0, 0.0, 6378137.0  # Equator, prime meridian, Earth's radius
        x_geoc = np.array([lon, lat, r])
        x_ecef = bh.position_geocentric_to_ecef(x_geoc, bh.AngleFormat.RADIANS)
        print(f"ECEF position: {x_ecef}")
        ```
    """

def position_geodetic_to_ecef(x_geod, angle_format):
    """
    Convert geodetic position to `ECEF` Cartesian coordinates.

    Transforms a position from geodetic coordinates (longitude, latitude, altitude) using
    the `WGS84` ellipsoid model to Earth-Centered Earth-Fixed (`ECEF`) Cartesian coordinates.

    Args:
        x_geod (numpy.ndarray or list): Geodetic position `[longitude, latitude, altitude]` where
            longitude is in radians or degrees, latitude is in radians or degrees, and
            altitude is in meters above the `WGS84` ellipsoid.
        angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).

    Returns:
        numpy.ndarray: `ECEF` Cartesian position `[x, y, z]` in meters.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Convert geodetic coordinates (GPS-like) to ECEF
        lon, lat, alt = -105.0, 40.0, 1655.0  # Boulder, CO (degrees, meters)
        x_geod = np.array([lon, lat, alt])
        x_ecef = bh.position_geodetic_to_ecef(x_geod, bh.AngleFormat.DEGREES)
        print(f"ECEF position: {x_ecef}")
        ```
    """

def position_itrf_to_gcrf(epc, x):
    """
    Transforms a position vector from ITRF (International Terrestrial Reference Frame)
    to GCRF (Geocentric Celestial Reference Frame).

    Applies the full `IAU 2006/2000A` transformation including bias, precession,
    nutation, Earth rotation, and polar motion corrections using global Earth
    orientation parameters.

    Args:
        epc (Epoch): Epoch instant for the transformation
        x (numpy.ndarray or list): Position vector in `ITRF` frame (m), shape `(3,)`

    Returns:
        numpy.ndarray: Position vector in `GCRF` frame (m), shape `(3,)`

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Create epoch
        epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # Position in ITRF (ground station)
        r_itrf = np.array([4000000.0, 3000000.0, 4000000.0])

        # Transform to GCRF
        r_gcrf = bh.position_itrf_to_gcrf(epc, r_itrf)
        print(f"GCRF position: {r_gcrf}")
        ```
    """

def position_sez_to_azel(x_sez, angle_format):
    """
    Convert position from South-East-Zenith (`SEZ`) frame to azimuth-elevation-range.

    Transforms a position from the local South-East-Zenith (`SEZ`) topocentric frame to
    azimuth-elevation-range spherical coordinates.

    Args:
        x_sez (numpy.ndarray or list): Position in `SEZ` frame `[south, east, zenith]` in meters.
        angle_format (AngleFormat): Angle format for output angular coordinates (`RADIANS` or `DEGREES`).

    Returns:
        numpy.ndarray: Azimuth-elevation-range `[azimuth, elevation, range]` where azimuth
            and elevation are in radians or degrees, and range is in meters.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Convert SEZ to azimuth-elevation for satellite tracking
        sez = np.array([30000.0, 50000.0, 100000.0])  # South, East, Zenith (meters)
        azel = bh.position_sez_to_azel(sez, bh.AngleFormat.DEGREES)
        print(f"Az={azel[0]:.1f}°, El={azel[1]:.1f}°, Range={azel[2]/1000:.1f}km")
        ```
    """

def relative_position_ecef_to_enz(location_ecef, r_ecef, conversion_type):
    """
    Convert relative position from `ECEF` to East-North-Up (`ENZ`) frame.

    Transforms a relative position vector from Earth-Centered Earth-Fixed (`ECEF`) coordinates
    to the local East-North-Up (`ENZ`) topocentric frame at the specified location.

    Args:
        location_ecef (numpy.ndarray or list): Reference location in `ECEF` coordinates `[x, y, z]` in meters.
        r_ecef (numpy.ndarray or list): Position vector in `ECEF` coordinates `[x, y, z]` in meters.
        conversion_type (EllipsoidalConversionType): Type of ellipsoidal conversion (`GEOCENTRIC` or `GEODETIC`).

    Returns:
        numpy.ndarray: Relative position in `ENZ` frame `[east, north, up]` in meters.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Ground station and satellite positions
        station_ecef = np.array([4000000.0, 3000000.0, 4000000.0])
        sat_ecef = np.array([4100000.0, 3100000.0, 4100000.0])
        enz = bh.relative_position_ecef_to_enz(station_ecef, sat_ecef, bh.EllipsoidalConversionType.GEODETIC)
        print(f"ENZ: East={enz[0]/1000:.1f}km, North={enz[1]/1000:.1f}km, Up={enz[2]/1000:.1f}km")
        ```
    """

def relative_position_ecef_to_sez(location_ecef, r_ecef, conversion_type):
    """
    Convert relative position from `ECEF` to South-East-Zenith (`SEZ`) frame.

    Transforms a relative position vector from Earth-Centered Earth-Fixed (`ECEF`) coordinates
    to the local South-East-Zenith (`SEZ`) topocentric frame at the specified location.

    Args:
        location_ecef (numpy.ndarray or list): Reference location in `ECEF` coordinates `[x, y, z]` in meters.
        r_ecef (numpy.ndarray or list): Position vector in `ECEF` coordinates `[x, y, z]` in meters.
        conversion_type (EllipsoidalConversionType): Type of ellipsoidal conversion (`GEOCENTRIC` or `GEODETIC`).

    Returns:
        numpy.ndarray: Relative position in `SEZ` frame `[south, east, zenith]` in meters.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Ground station and satellite positions
        station_ecef = np.array([4000000.0, 3000000.0, 4000000.0])
        sat_ecef = np.array([4100000.0, 3100000.0, 4100000.0])
        sez = bh.relative_position_ecef_to_sez(station_ecef, sat_ecef, bh.EllipsoidalConversionType.GEODETIC)
        print(f"SEZ: South={sez[0]/1000:.1f}km, East={sez[1]/1000:.1f}km, Zenith={sez[2]/1000:.1f}km")
        ```
    """

def relative_position_enz_to_ecef(location_ecef, r_enz, conversion_type):
    """
    Convert relative position from East-North-Up (`ENZ`) frame to `ECEF`.

    Transforms a relative position vector from the local East-North-Up (`ENZ`) topocentric
    frame to Earth-Centered Earth-Fixed (`ECEF`) coordinates at the specified location.

    Args:
        location_ecef (numpy.ndarray or list): Reference location in `ECEF` coordinates `[x, y, z]` in meters.
        r_enz (numpy.ndarray or list): Relative position in `ENZ` frame `[east, north, up]` in meters.
        conversion_type (EllipsoidalConversionType): Type of ellipsoidal conversion (`GEOCENTRIC` or `GEODETIC`).

    Returns:
        numpy.ndarray: Position vector in `ECEF` coordinates `[x, y, z]` in meters.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Convert ENZ offset back to ECEF
        station_ecef = np.array([4000000.0, 3000000.0, 4000000.0])
        enz_offset = np.array([50000.0, 30000.0, 100000.0])  # 50km east, 30km north, 100km up
        target_ecef = bh.relative_position_enz_to_ecef(station_ecef, enz_offset, bh.EllipsoidalConversionType.GEODETIC)
        print(f"Target ECEF: {target_ecef}")
        ```
    """

def relative_position_sez_to_ecef(location_ecef, x_sez, conversion_type):
    """
    Convert relative position from South-East-Zenith (`SEZ`) frame to `ECEF`.

    Transforms a relative position vector from the local South-East-Zenith (`SEZ`) topocentric
    frame to Earth-Centered Earth-Fixed (`ECEF`) coordinates at the specified location.

    Args:
        location_ecef (numpy.ndarray or list): Reference location in `ECEF` coordinates `[x, y, z]` in meters.
        x_sez (numpy.ndarray or list): Relative position in `SEZ` frame `[south, east, zenith]` in meters.
        conversion_type (EllipsoidalConversionType): Type of ellipsoidal conversion (`GEOCENTRIC` or `GEODETIC`).

    Returns:
        numpy.ndarray: Position vector in `ECEF` coordinates `[x, y, z]` in meters.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Convert SEZ offset back to ECEF
        station_ecef = np.array([4000000.0, 3000000.0, 4000000.0])
        sez_offset = np.array([30000.0, 50000.0, 100000.0])  # 30km south, 50km east, 100km up
        target_ecef = bh.relative_position_sez_to_ecef(station_ecef, sez_offset, bh.EllipsoidalConversionType.GEODETIC)
        print(f"Target ECEF: {target_ecef}")
        ```
    """

def rotation_ecef_to_eci(epc):
    """
    Computes the combined rotation matrix from the Earth-fixed to the inertial
    reference frame. Applies corrections for bias, precession, nutation,
    Earth-rotation, and polar motion.

    This function is an alias for rotation_itrf_to_gcrf. `ECEF` refers to the
    `ITRF` (International Terrestrial Reference Frame) implementation, and `ECI` refers
    to the `GCRF` (Geocentric Celestial Reference Frame) implementation.

    Args:
        epc (Epoch): Epoch instant for computation of transformation matrix

    Returns:
        numpy.ndarray: 3x3 rotation matrix transforming `ECEF` (`ITRF`) -> `ECI` (`GCRF`)

    Example:
        ```python
        import brahe as bh

        # Create epoch
        epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # Get rotation matrix from ECEF to ECI
        R = bh.rotation_ecef_to_eci(epc)
        print(f"Rotation matrix shape: {R.shape}")
        ```
    """

def rotation_eci_to_ecef(epc):
    """
    Computes the combined rotation matrix from the inertial to the Earth-fixed
    reference frame. Applies corrections for bias, precession, nutation,
    Earth-rotation, and polar motion.

    This function is an alias for rotation_gcrf_to_itrf. `ECI` refers to the
    `GCRF` (Geocentric Celestial Reference Frame) implementation, and `ECEF` refers
    to the `ITRF` (International Terrestrial Reference Frame) implementation.

    Args:
        epc (Epoch): Epoch instant for computation of transformation matrix

    Returns:
        numpy.ndarray: 3x3 rotation matrix transforming `ECI` (`GCRF`) -> `ECEF` (`ITRF`)

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Create epoch
        epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # Get rotation matrix
        R = bh.rotation_eci_to_ecef(epc)
        print(f"Rotation matrix shape: {R.shape}")
        # Output: Rotation matrix shape: (3, 3)
        ```
    """

def rotation_eci_to_rtn(x_eci):
    """
    Computes the rotation matrix transforming a vector in the Earth-Centered Inertial (ECI)
    frame to the radial, along-track, cross-track (RTN) frame.

    This is the transpose (inverse) of the RTN-to-ECI rotation matrix.

    Args:
        x_eci (numpy.ndarray or list): 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,)

    Returns:
        numpy.ndarray: 3x3 rotation matrix transforming from ECI to RTN frame, shape (3, 3)

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Define satellite state
        sma = bh.R_EARTH + 700e3  # Semi-major axis in meters
        state = np.array([sma, 0.0, 0.0, 0.0, bh.perigee_velocity(sma, 0.0), 0.0])

        # Get rotation matrix
        R = bh.rotation_eci_to_rtn(state)
        print(f"ECI to RTN rotation matrix:\n{R}")
        ```
    """

def rotation_ellipsoid_to_enz(x_ellipsoid, angle_format):
    """
    Compute rotation matrix from ellipsoidal coordinates to East-North-Up (`ENZ`) frame.

    Calculates the rotation matrix that transforms vectors from an ellipsoidal coordinate
    frame (geocentric or geodetic) to the local East-North-Up (`ENZ`) topocentric frame at
    the specified location.

    Args:
        x_ellipsoid (numpy.ndarray or list): Ellipsoidal position `[latitude, longitude, altitude/radius]`
            where latitude is in radians or degrees, longitude is in radians or degrees.
        angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).

    Returns:
        numpy.ndarray: 3x3 rotation matrix from ellipsoidal frame to `ENZ` frame.
    """

def rotation_ellipsoid_to_sez(x_ellipsoid, angle_format):
    """
    Compute rotation matrix from ellipsoidal coordinates to South-East-Zenith (`SEZ`) frame.

    Calculates the rotation matrix that transforms vectors from an ellipsoidal coordinate
    frame (geocentric or geodetic) to the local South-East-Zenith (`SEZ`) topocentric frame
    at the specified location.

    Args:
        x_ellipsoid (numpy.ndarray or list): Ellipsoidal position `[latitude, longitude, altitude/radius]`
            where latitude is in radians or degrees, longitude is in radians or degrees.
        angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).

    Returns:
        numpy.ndarray: 3x3 rotation matrix from ellipsoidal frame to `SEZ` frame.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Get rotation matrix for ground station in SEZ frame
        lat, lon, alt = 0.7, -1.5, 100.0  # radians, meters
        x_geod = np.array([lat, lon, alt])
        R_sez = bh.rotation_ellipsoid_to_sez(x_geod, bh.AngleFormat.RADIANS)
        print(f"Rotation matrix shape: {R_sez.shape}")
        ```
    """

def rotation_eme2000_to_gcrf():
    """
    Computes the rotation matrix from EME2000 (Earth Mean Equator and Equinox of J2000.0)
    to GCRF (Geocentric Celestial Reference Frame).

    This transformation applies the inverse frame bias correction to account for the
    difference between EME2000 (J2000.0 mean equator/equinox) and GCRF (ICRS-aligned).
    The transformation is constant and does not depend on time.

    Returns:
        numpy.ndarray: 3x3 rotation matrix transforming `EME2000` -> `GCRF`

    Example:
        ```python
        import brahe as bh

        # Get rotation matrix
        R = bh.rotation_eme2000_to_gcrf()
        print(f"Rotation matrix shape: {R.shape}")
        # Output: Rotation matrix shape: (3, 3)
        ```
    """

def rotation_enz_to_ellipsoid(x_ellipsoid, angle_format):
    """
    Compute rotation matrix from East-North-Up (`ENZ`) frame to ellipsoidal coordinates.

    Calculates the rotation matrix that transforms vectors from the local East-North-Up
    (`ENZ`) topocentric frame to an ellipsoidal coordinate frame (geocentric or geodetic)
    at the specified location.

    Args:
        x_ellipsoid (numpy.ndarray or list): Ellipsoidal position `[latitude, longitude, altitude/radius]`
            where latitude is in radians or degrees, longitude is in radians or degrees.
        angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).

    Returns:
        numpy.ndarray: 3x3 rotation matrix from `ENZ` frame to ellipsoidal frame.
    """

def rotation_gcrf_to_eme2000():
    """
    Computes the rotation matrix from GCRF (Geocentric Celestial Reference Frame)
    to EME2000 (Earth Mean Equator and Equinox of J2000.0).

    This transformation applies the frame bias correction to account for the difference
    between GCRF (ICRS-aligned) and EME2000 (J2000.0 mean equator/equinox). The
    transformation is constant and does not depend on time.

    Returns:
        numpy.ndarray: 3x3 rotation matrix transforming `GCRF` -> `EME2000`

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Get rotation matrix
        R = bh.rotation_gcrf_to_eme2000()
        print(f"Rotation matrix shape: {R.shape}")
        # Output: Rotation matrix shape: (3, 3)
        ```
    """

def rotation_gcrf_to_itrf(epc):
    """
    Computes the combined rotation matrix from GCRF (Geocentric Celestial Reference Frame)
    to ITRF (International Terrestrial Reference Frame). Applies corrections for bias,
    precession, nutation, Earth-rotation, and polar motion.

    The transformation is accomplished using the `IAU 2006/2000A`, `CIO`-based
    theory using classical angles. The method as described in section 5.5 of
    the SOFA C transformation cookbook.

    The function will utilize the global Earth orientation and loaded data to
    apply corrections for Celestial Intermediate Pole (`CIP`) and polar motion drift
    derived from empirical observations.

    Args:
        epc (Epoch): Epoch instant for computation of transformation matrix

    Returns:
        numpy.ndarray: 3x3 rotation matrix transforming `GCRF` -> `ITRF`

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Create epoch
        epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # Get rotation matrix from GCRF to ITRF
        R = bh.rotation_gcrf_to_itrf(epc)
        print(f"Rotation matrix shape: {R.shape}")
        # Output: Rotation matrix shape: (3, 3)
        ```
    """

def rotation_itrf_to_gcrf(epc):
    """
    Computes the combined rotation matrix from ITRF (International Terrestrial Reference Frame)
    to GCRF (Geocentric Celestial Reference Frame). Applies corrections for bias,
    precession, nutation, Earth-rotation, and polar motion.

    The transformation is accomplished using the `IAU 2006/2000A`, `CIO`-based
    theory using classical angles. The method as described in section 5.5 of
    the SOFA C transformation cookbook.

    The function will utilize the global Earth orientation and loaded data to
    apply corrections for Celestial Intermediate Pole (`CIP`) and polar motion drift
    derived from empirical observations.

    Args:
        epc (Epoch): Epoch instant for computation of transformation matrix

    Returns:
        numpy.ndarray: 3x3 rotation matrix transforming `ITRF` -> `GCRF`

    Example:
        ```python
        import brahe as bh

        # Create epoch
        epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # Get rotation matrix from ITRF to GCRF
        R = bh.rotation_itrf_to_gcrf(epc)
        print(f"Rotation matrix shape: {R.shape}")
        ```
    """

def rotation_rtn_to_eci(x_eci):
    """
    Computes the rotation matrix transforming a vector in the radial, along-track, cross-track (RTN)
    frame to the Earth-Centered Inertial (ECI) frame.

    The ECI frame can be any inertial frame centered at the Earth's center, such as GCRF or EME2000.

    The RTN frame is defined as follows:
    - R (Radial): Points from the Earth's center to the satellite's position.
    - N (Cross-Track): Perpendicular to the orbital plane, defined by the angular momentum vector (cross product of position and velocity).
    - T (Along-Track): Completes the right-handed coordinate system, lying in the orbital plane and perpendicular to R and N.

    Args:
        x_eci (numpy.ndarray or list): 6D state vector in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,)

    Returns:
        numpy.ndarray: 3x3 rotation matrix transforming from RTN to ECI frame, shape (3, 3)

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Define satellite state
        sma = bh.R_EARTH + 700e3  # Semi-major axis in meters
        state = np.array([sma, 0.0, 0.0, 0.0, bh.perigee_velocity(sma, 0.0), 0.0])

        # Get rotation matrix
        R = bh.rotation_rtn_to_eci(state)
        print(f"RTN to ECI rotation matrix:\n{R}")
        ```
    """

def rotation_sez_to_ellipsoid(x_ellipsoid, angle_format):
    """
    Compute rotation matrix from South-East-Zenith (`SEZ`) frame to ellipsoidal coordinates.

    Calculates the rotation matrix that transforms vectors from the local South-East-Zenith
    (`SEZ`) topocentric frame to an ellipsoidal coordinate frame (geocentric or geodetic)
    at the specified location.

    Args:
        x_ellipsoid (numpy.ndarray or list): Ellipsoidal position `[latitude, longitude, altitude/radius]`
            where latitude is in radians or degrees, longitude is in radians or degrees.
        angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).

    Returns:
        numpy.ndarray: 3x3 rotation matrix from `SEZ` frame to ellipsoidal frame.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Get inverse rotation matrix from SEZ to ellipsoidal
        lat, lon, alt = 0.7, -1.5, 100.0  # radians, meters
        x_geod = np.array([lat, lon, alt])
        R_ellipsoid = bh.rotation_sez_to_ellipsoid(x_geod, bh.AngleFormat.RADIANS)
        print(f"Rotation matrix shape: {R_ellipsoid.shape}")
        ```
    """

def semimajor_axis(n, angle_format):
    """
    Computes the semi-major axis of an astronomical object from Earth
    given the object's mean motion.

    Args:
        n (float): The mean motion of the astronomical object in radians or degrees.
        angle_format (AngleFormat): Interpret mean motion as AngleFormat.DEGREES or AngleFormat.RADIANS.

    Returns:
        float: The semi-major axis of the astronomical object in meters.

    Example:
        ```python
        import brahe as bh

        # Calculate semi-major axis from mean motion (typical LEO satellite)
        n = 0.001027  # radians/second (~15 revolutions/day)
        a = bh.semimajor_axis(n, bh.AngleFormat.RADIANS)
        print(f"Semi-major axis: {a/1000:.2f} km")
        ```
    """

def semimajor_axis_from_orbital_period(period):
    """
    Computes the semi-major axis from orbital period around Earth.

    Args:
        period (float): The orbital period in seconds.

    Returns:
        float: The semi-major axis in meters.

    Example:
        ```python
        import brahe as bh

        # Calculate semi-major axis for a 90-minute orbit
        period = 90 * 60.0  # 90 minutes in seconds
        a = bh.semimajor_axis_from_orbital_period(period)
        print(f"Semi-major axis: {a/1000:.2f} km")
        ```
    """

def semimajor_axis_from_orbital_period_general(period, gm):
    """
    Computes the semi-major axis from orbital period for a general body.

    Args:
        period (float): The orbital period in seconds.
        gm (float): (keyword-only) The standard gravitational parameter of primary body in m³/s².

    Returns:
        float: The semi-major axis in meters.

    Example:
        ```python
        import brahe as bh

        # Calculate semi-major axis for 2-hour Venus orbit
        period = 2 * 3600.0  # 2 hours in seconds
        a = bh.semimajor_axis_from_orbital_period_general(period, bh.GM_VENUS)
        print(f"Semi-major axis: {a/1000:.2f} km")
        ```
    """

def semimajor_axis_general(n, gm, angle_format):
    """
    Computes the semi-major axis of an astronomical object from a general body
    given the object's mean motion.

    Args:
        n (float): The mean motion of the astronomical object in radians or degrees.
        gm (float): (keyword-only) The standard gravitational parameter of primary body in m³/s².
        angle_format (AngleFormat): Interpret mean motion as AngleFormat.DEGREES or AngleFormat.RADIANS.

    Returns:
        float: The semi-major axis of the astronomical object in meters.

    Example:
        ```python
        import brahe as bh

        # Calculate semi-major axis for Jupiter orbiter
        n = 0.0001  # radians/second
        a = bh.semimajor_axis_general(n, bh.GM_JUPITER, bh.AngleFormat.RADIANS)
        print(f"Semi-major axis: {a/1000:.2f} km")
        ```
    """

def set_global_eop_provider(provider):
    """
    Set the global EOP provider using any supported provider type.

    This function accepts any of the three EOP provider types: StaticEOPProvider,
    FileEOPProvider, or CachingEOPProvider. This is the recommended way to set
    the global EOP provider.

    Args:
        provider (StaticEOPProvider | FileEOPProvider | CachingEOPProvider): EOP provider to set globally

    Example:
        ```python
        import brahe as bh

        # Use with StaticEOPProvider
        provider = bh.StaticEOPProvider.from_zero()
        bh.set_global_eop_provider(provider)

        # Use with FileEOPProvider
        provider = bh.FileEOPProvider.from_default_standard(True, "Hold")
        bh.set_global_eop_provider(provider)

        # Use with CachingEOPProvider
        provider = bh.CachingEOPProvider(
            eop_type="StandardBulletinA",
            max_age_seconds=7 * 86400,
            auto_refresh=False,
            interpolate=True,
            extrapolate="Hold"
        )
        bh.set_global_eop_provider(provider)
        ```
    """

def set_global_eop_provider_from_caching_provider(provider):
    """
    Set the global EOP provider using a caching provider.

    Args:
        provider (CachingEOPProvider): Caching EOP provider to set globally

    Example:
        ```python
        import brahe as bh

        provider = bh.CachingEOPProvider(
            eop_type="StandardBulletinA",
            max_age_seconds=7 * 86400,
            auto_refresh=False,
            interpolate=True,
            extrapolate="Hold"
        )
        bh.set_global_eop_provider_from_caching_provider(provider)
        ```
    """

def set_global_eop_provider_from_file_provider(provider):
    """
    Set the global EOP provider using a file-based provider.

    Args:
        provider (FileEOPProvider): File-based EOP provider to set globally

    Example:
        ```python
        import brahe as bh

        provider = bh.FileEOPProvider.from_default_standard(True, "Hold")
        bh.set_global_eop_provider_from_file_provider(provider)
        ```
    """

def set_global_eop_provider_from_static_provider(provider):
    """
    Set the global EOP provider using a static provider.

    Args:
        provider (StaticEOPProvider): Static EOP provider to set globally

    Example:
        ```python
        import brahe as bh

        provider = bh.StaticEOPProvider.from_zero()
        bh.set_global_eop_provider_from_static_provider(provider)
        ```
    """

def set_ludicrous_speed():
    """
    LUDICROUS SPEED! GO!

    Set the thread pool to use all available CPU cores (alias for `set_max_threads`).

    This is a fun alias for `set_max_threads()` that sets the number of threads
    to 100% of available CPU cores for maximum performance. Can be called multiple
    times to dynamically reinitialize the thread pool.

    Raises:
        RuntimeError: If thread pool fails to build.

    Example:
        ```python
        import brahe as bh

        # MAXIMUM POWER! Use all available CPU cores
        bh.set_ludicrous_speed()
        print(f"Going ludicrous with {bh.get_max_threads()} threads!")

        # Throttle down for testing
        bh.set_num_threads(1)

        # ENGAGE LUDICROUS SPEED again - no error!
        bh.set_ludicrous_speed()
        ```

    Note:
        This function can be called at any time to reconfigure the thread pool
        to use maximum available cores, regardless of previous configuration.
    """

def set_max_threads():
    """
    Set the thread pool to use all available CPU cores.

    This is a convenience function that sets the number of threads to 100%
    of available CPU cores. Can be called multiple times to reinitialize the
    thread pool dynamically.

    Raises:
        RuntimeError: If thread pool fails to build.

    Example:
        ```python
        import brahe as bh

        # Use all available CPU cores
        bh.set_max_threads()
        print(f"Using all {bh.get_max_threads()} cores")

        # Switch to 2 threads
        bh.set_num_threads(2)

        # Switch back to max - no error!
        bh.set_max_threads()
        print(f"Back to {bh.get_max_threads()} cores")
        ```

    Note:
        This function can be called at any time, even after the thread pool
        has been initialized with a different configuration.
    """

def set_num_threads(n):
    """
    Set the number of threads for parallel computation.

    Configures the global thread pool used by Brahe for parallel operations such as
    access computations. This function can be called multiple times to dynamically
    change the thread pool configuration - each call will reinitialize the pool with
    the new thread count.

    Args:
        n (int): Number of threads to use. Must be at least 1.

    Raises:
        ValueError: If n < 1.
        RuntimeError: If thread pool fails to build.

    Example:
        ```python
        import brahe as bh

        # Set to 4 threads initially
        bh.set_num_threads(4)
        print(f"Threads: {bh.get_max_threads()}")  # Output: 4

        # Reinitialize with 8 threads - no error!
        bh.set_num_threads(8)
        print(f"Threads: {bh.get_max_threads()}")  # Output: 8

        # All parallel operations (e.g., location_accesses) will now use
        # 8 threads unless overridden with AccessSearchConfig.num_threads
        ```

    Note:
        Unlike earlier versions, this function no longer raises an error if the
        thread pool has already been initialized. You can safely call it at any
        time to reconfigure the thread pool.
    """

def state_cartesian_to_osculating(x_cart, angle_format):
    """
    Convert Cartesian state to osculating orbital elements.

    Transforms a state vector from Cartesian position and velocity coordinates to
    osculating Keplerian orbital elements.

    Args:
        x_cart (numpy.ndarray or list): Cartesian state `[x, y, z, vx, vy, vz]` where position
            is in meters and velocity is in meters per second.
        angle_format (AngleFormat): Angle format for output angular elements (`RADIANS` or `DEGREES`).

    Returns:
        numpy.ndarray: Osculating orbital elements `[a, e, i, RAAN, omega, M]` where `a` is
            semi-major axis (meters), `e` is eccentricity (dimensionless), `i` is inclination
            (radians or degrees), `RAAN` is right ascension of ascending node (radians or degrees),
            `omega` is argument of periapsis (radians or degrees), and `M` is mean anomaly
            (radians or degrees).

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Cartesian state vector
        x_cart = np.array([7000000.0, 0.0, 0.0, 0.0, 7546.0, 0.0])  # [x, y, z, vx, vy, vz]
        oe = bh.state_cartesian_to_osculating(x_cart, bh.AngleFormat.RADIANS)
        print(f"Orbital elements: a={oe[0]:.0f}m, e={oe[1]:.6f}, i={oe[2]:.6f} rad")
        ```
    """

def state_ecef_to_eci(epc, x_ecef):
    """
    Transforms a state vector (position and velocity) from the Earth Centered
    Earth Fixed (`ECEF`/`ITRF`) frame to the Earth Centered Inertial (`ECI`/`GCRF`) frame.

    This function is an alias for state_itrf_to_gcrf. Applies the full
    `IAU 2006/2000A` transformation including bias, precession, nutation, Earth
    rotation, and polar motion corrections using global Earth orientation parameters.
    The velocity transformation accounts for the Earth's rotation rate.

    Args:
        epc (Epoch): Epoch instant for the transformation
        x_ecef (numpy.ndarray or list): State vector in `ECEF` frame `[position (m), velocity (m/s)]`, shape `(6,)`

    Returns:
        numpy.ndarray: State vector in `ECI` frame `[position (m), velocity (m/s)]`, shape `(6,)`

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Create epoch
        epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # State vector in ECEF [x, y, z, vx, vy, vz] (meters, m/s)
        state_ecef = np.array([4000000.0, 3000000.0, 4000000.0, 100.0, -50.0, 200.0])

        # Transform to ECI
        state_eci = bh.state_ecef_to_eci(epc, state_ecef)
        print(f"ECI state: {state_eci}")
        ```
    """

def state_eci_to_ecef(epc, x_eci):
    """
    Transforms a state vector (position and velocity) from the Earth Centered
    Inertial (`ECI`/`GCRF`) frame to the Earth Centered Earth Fixed (`ECEF`/`ITRF`) frame.

    This function is an alias for state_gcrf_to_itrf. Applies the full
    `IAU 2006/2000A` transformation including bias, precession, nutation, Earth
    rotation, and polar motion corrections using global Earth orientation parameters.
    The velocity transformation accounts for the Earth's rotation rate.

    Args:
        epc (Epoch): Epoch instant for the transformation
        x_eci (numpy.ndarray or list): State vector in `ECI` frame `[position (m), velocity (m/s)]`, shape `(6,)`

    Returns:
        numpy.ndarray: State vector in `ECEF` frame `[position (m), velocity (m/s)]`, shape `(6,)`

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Create epoch
        epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # State vector in ECI [x, y, z, vx, vy, vz] (meters, m/s)
        state_eci = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])

        # Transform to ECEF
        state_ecef = bh.state_eci_to_ecef(epc, state_eci)
        print(f"ECEF state: {state_ecef}")
        ```
    """

def state_eci_to_rtn(x_chief, x_deputy):
    """
    Transforms the absolute states of a chief and deputy satellite from the Earth-Centered Inertial (ECI)
    frame to the relative state of the deputy with respect to the chief in the rotating
    Radial, Along-Track, Cross-Track (RTN) frame.

    Args:
        x_chief (numpy.ndarray or list): 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,)
        x_deputy (numpy.ndarray or list): 6D state vector of the deputy satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,)

    Returns:
        numpy.ndarray: 6D relative state vector of the deputy with respect to the chief in the RTN frame [ρ_R, ρ_T, ρ_N, ρ̇_R, ρ̇_T, ρ̇_N] (m, m/s), shape (6,)

    Example:
        ```python
        import brahe as bh
        import numpy as np

        bh.initialize_eop()

        # Define chief and deputy orbital elements
        oe_chief = np.array([bh.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        oe_deputy = np.array([bh.R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])

        # Convert to Cartesian states
        x_chief = bh.state_osculating_to_cartesian(oe_chief, bh.AngleFormat.DEGREES)
        x_deputy = bh.state_osculating_to_cartesian(oe_deputy, bh.AngleFormat.DEGREES)

        # Transform to relative RTN state
        x_rel_rtn = bh.state_eci_to_rtn(x_chief, x_deputy)
        print(f"Relative state in RTN: {x_rel_rtn}")
        ```
    """

def state_eme2000_to_gcrf(x_eme2000):
    """
    Transforms a state vector (position and velocity) from EME2000 (Earth Mean Equator
    and Equinox of J2000.0) to GCRF (Geocentric Celestial Reference Frame).

    Applies the inverse frame bias correction to both position and velocity. Because
    the transformation does not vary with time, the velocity is directly rotated without
    additional correction terms.

    Args:
        x_eme2000 (numpy.ndarray or list): State vector in `EME2000` frame `[position (m), velocity (m/s)]`, shape `(6,)`

    Returns:
        numpy.ndarray: State vector in `GCRF` frame `[position (m), velocity (m/s)]`, shape `(6,)`

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # State vector in EME2000 [x, y, z, vx, vy, vz] (meters, m/s)
        state_eme2000 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])

        # Transform to GCRF
        state_gcrf = bh.state_eme2000_to_gcrf(state_eme2000)
        print(f"GCRF state: {state_gcrf}")
        ```
    """

def state_gcrf_to_eme2000(x_gcrf):
    """
    Transforms a state vector (position and velocity) from GCRF (Geocentric Celestial
    Reference Frame) to EME2000 (Earth Mean Equator and Equinox of J2000.0).

    Applies the frame bias correction to both position and velocity. Because the
    transformation does not vary with time, the velocity is directly rotated without
    additional correction terms.

    Args:
        x_gcrf (numpy.ndarray or list): State vector in `GCRF` frame `[position (m), velocity (m/s)]`, shape `(6,)`

    Returns:
        numpy.ndarray: State vector in `EME2000` frame `[position (m), velocity (m/s)]`, shape `(6,)`

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # State vector in GCRF [x, y, z, vx, vy, vz] (meters, m/s)
        state_gcrf = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])

        # Transform to EME2000
        state_eme2000 = bh.state_gcrf_to_eme2000(state_gcrf)
        print(f"EME2000 state: {state_eme2000}")
        ```
    """

def state_gcrf_to_itrf(epc, x_gcrf):
    """
    Transforms a state vector (position and velocity) from GCRF (Geocentric Celestial
    Reference Frame) to ITRF (International Terrestrial Reference Frame).

    Applies the full `IAU 2006/2000A` transformation including bias, precession,
    nutation, Earth rotation, and polar motion corrections using global Earth
    orientation parameters. The velocity transformation accounts for the Earth's
    rotation rate.

    Args:
        epc (Epoch): Epoch instant for the transformation
        x_gcrf (numpy.ndarray or list): State vector in `GCRF` frame `[position (m), velocity (m/s)]`, shape `(6,)`

    Returns:
        numpy.ndarray: State vector in `ITRF` frame `[position (m), velocity (m/s)]`, shape `(6,)`

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Create epoch
        epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # State vector in GCRF [x, y, z, vx, vy, vz] (meters, m/s)
        state_gcrf = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])

        # Transform to ITRF
        state_itrf = bh.state_gcrf_to_itrf(epc, state_gcrf)
        print(f"ITRF state: {state_itrf}")
        ```
    """

def state_itrf_to_gcrf(epc, x_itrf):
    """
    Transforms a state vector (position and velocity) from ITRF (International Terrestrial
    Reference Frame) to GCRF (Geocentric Celestial Reference Frame).

    Applies the full `IAU 2006/2000A` transformation including bias, precession,
    nutation, Earth rotation, and polar motion corrections using global Earth
    orientation parameters. The velocity transformation accounts for the Earth's
    rotation rate.

    Args:
        epc (Epoch): Epoch instant for the transformation
        x_itrf (numpy.ndarray or list): State vector in `ITRF` frame `[position (m), velocity (m/s)]`, shape `(6,)`

    Returns:
        numpy.ndarray: State vector in `GCRF` frame `[position (m), velocity (m/s)]`, shape `(6,)`

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Create epoch
        epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)

        # State vector in ITRF [x, y, z, vx, vy, vz] (meters, m/s)
        state_itrf = np.array([4000000.0, 3000000.0, 4000000.0, 100.0, -50.0, 200.0])

        # Transform to GCRF
        state_gcrf = bh.state_itrf_to_gcrf(epc, state_itrf)
        print(f"GCRF state: {state_gcrf}")
        ```
    """

def state_oe_to_roe(oe_chief, oe_deputy, angle_format):
    """
    Converts chief and deputy satellite orbital elements (OE) to quasi-nonsingular relative orbital elements (ROE).

    The ROE formulation provides a mean description of relative motion that is nonsingular for
    circular and near-circular orbits. The ROE vector contains:
    - da: Relative semi-major axis (dimensionless)
    - dλ: Relative mean longitude (degrees or radians)
    - dex: x-component of relative eccentricity vector (dimensionless)
    - dey: y-component of relative eccentricity vector (dimensionless)
    - dix: x-component of relative inclination vector (degrees or radians)
    - diy: y-component of relative inclination vector (degrees or radians)

    Args:
        oe_chief (numpy.ndarray or list): Chief satellite orbital elements [a, e, i, Ω, ω, M] shape (6,)
        oe_deputy (numpy.ndarray or list): Deputy satellite orbital elements [a, e, i, Ω, ω, M] shape (6,)
        angle_format (AngleFormat): Format of angular elements (DEGREES or RADIANS)

    Returns:
        numpy.ndarray: Relative orbital elements [da, dλ, dex, dey, dix, diy] shape (6,)

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Define chief and deputy orbital elements (degrees)
        oe_chief = np.array([bh.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        oe_deputy = np.array([bh.R_EARTH + 701e3, 0.0015, 97.85, 15.05, 30.05, 45.05])

        # Convert to ROE
        roe = bh.state_oe_to_roe(oe_chief, oe_deputy, bh.AngleFormat.DEGREES)
        print(f"Relative orbital elements: {roe}")
        # Relative orbital elements: [1.413e-4, 9.321e-2, 4.324e-4, 2.511e-4, 5.0e-2, 4.954e-2]
        ```
    """

def state_osculating_to_cartesian(x_oe, angle_format):
    """
    Convert osculating orbital elements to Cartesian state.

    Transforms a state vector from osculating Keplerian orbital elements to Cartesian
    position and velocity coordinates.

    Args:
        x_oe (numpy.ndarray or list): Osculating orbital elements `[a, e, i, RAAN, omega, M]` where
            `a` is semi-major axis (meters), `e` is eccentricity (dimensionless), `i` is
            inclination (radians or degrees), `RAAN` is right ascension of ascending node
            (radians or degrees), `omega` is argument of periapsis (radians or degrees),
            and `M` is mean anomaly (radians or degrees).
        angle_format (AngleFormat): Angle format for angular elements (`RADIANS` or `DEGREES`).

    Returns:
        numpy.ndarray: Cartesian state `[x, y, z, vx, vy, vz]` where position is in meters
            and velocity is in meters per second.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Orbital elements for a circular orbit
        oe = np.array([7000000.0, 0.0, 0.0, 0.0, 0.0, 0.0])  # a, e, i, RAAN, omega, M
        x_cart = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
        print(f"Cartesian state: {x_cart}")
        ```
    """

def state_roe_to_oe(oe_chief, roe, angle_format):
    """
    Converts chief satellite orbital elements (OE) and quasi-nonsingular relative orbital elements (ROE)
    to deputy satellite orbital elements.

    This is the inverse transformation of `state_oe_to_roe`, converting from ROE representation
    back to classical orbital elements for the deputy satellite.

    Args:
        oe_chief (numpy.ndarray or list): Chief satellite orbital elements [a, e, i, Ω, ω, M] shape (6,)
        roe (numpy.ndarray or list): Relative orbital elements [da, dλ, dex, dey, dix, diy] shape (6,)
        angle_format (AngleFormat): Format of angular elements (DEGREES or RADIANS)

    Returns:
        numpy.ndarray: Deputy satellite orbital elements [a, e, i, Ω, ω, M] shape (6,)

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Define chief orbital elements and ROE (degrees)
        oe_chief = np.array([bh.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        roe = np.array([1.413e-4, 9.321e-2, 4.324e-4, 2.511e-4, 5.0e-2, 4.954e-2])

        # Convert to deputy OE
        oe_deputy = bh.state_roe_to_oe(oe_chief, roe, bh.AngleFormat.DEGREES)
        print(f"Deputy orbital elements: {oe_deputy}")
        # Deputy orbital elements: [7.079e6, 1.5e-3, 97.85, 15.05, 30.05, 45.05]
        ```
    """

def state_rtn_to_eci(x_chief, x_rel_rtn):
    """
    Transforms the relative state of a deputy satellite with respect to a chief satellite
    from the rotating Radial, Along-Track, Cross-Track (RTN) frame to the absolute state
    of the deputy in the Earth-Centered Inertial (ECI) frame.

    Args:
        x_chief (numpy.ndarray or list): 6D state vector of the chief satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,)
        x_rel_rtn (numpy.ndarray or list): 6D relative state vector of the deputy with respect to the chief in the RTN frame [ρ_R, ρ_T, ρ_N, ρ̇_R, ρ̇_T, ρ̇_N] (m, m/s), shape (6,)

    Returns:
        numpy.ndarray: 6D state vector of the deputy satellite in the ECI frame [x, y, z, vx, vy, vz] (m, m/s), shape (6,)

    Example:
        ```python
        import brahe as bh
        import numpy as np

        bh.initialize_eop()

        # Define chief state and relative RTN state
        oe_chief = np.array([bh.R_EARTH + 700e3, 0.001, 97.8, 15.0, 30.0, 45.0])
        x_chief = bh.state_osculating_to_cartesian(oe_chief, bh.AngleFormat.DEGREES)

        # Relative state: 1km radial, 0.5km along-track, -0.3km cross-track
        x_rel_rtn = np.array([1000.0, 500.0, -300.0, 0.0, 0.0, 0.0])

        # Transform to absolute deputy ECI state
        x_deputy = bh.state_rtn_to_eci(x_chief, x_rel_rtn)
        print(f"Deputy state in ECI: {x_deputy}")
        ```
    """

def sun_synchronous_inclination(a_or_oe, e=None, *, angle_format):
    """
    Computes the inclination for a Sun-synchronous orbit around Earth based on
    the J2 gravitational perturbation.

    Args:
        a_or_oe (float or array): Either the semi-major axis in meters, or a 6-element
            Keplerian elements array [a, e, i, Ω, ω, ν] from which `a` and `e` will be extracted.
        e (float, optional): The eccentricity. Required if `a_or_oe` is a scalar, ignored if vector.
        angle_format (AngleFormat): (keyword-only) Return output in AngleFormat.DEGREES or AngleFormat.RADIANS.

    Returns:
        float: Inclination for a Sun synchronous orbit in degrees or radians.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Using scalar parameters
        a = bh.R_EARTH + 600e3
        e = 0.001  # nearly circular
        inc = bh.sun_synchronous_inclination(a, e, bh.AngleFormat.DEGREES)
        print(f"Sun-synchronous inclination: {inc:.2f} degrees")

        # Using Keplerian elements vector
        oe = [bh.R_EARTH + 600e3, 0.001, np.radians(97.8), 0, 0, 0]
        inc = bh.sun_synchronous_inclination(oe, angle_format=bh.AngleFormat.DEGREES)
        print(f"Sun-synchronous inclination: {inc:.2f} degrees")
        ```
    """

def time_system_offset_for_datetime(
    year, month, day, hour, minute, second, nanosecond, time_system_src, time_system_dst
):
    """
    Calculate the offset between two time systems for a given Gregorian calendar date.

    Args:
        year (int): Year
        month (int): Month (1-12)
        day (int): Day of month (1-31)
        hour (int): Hour (0-23)
        minute (int): Minute (0-59)
        second (float): Second with fractional part
        nanosecond (float): Nanosecond component
        time_system_src (TimeSystem): Source time system
        time_system_dst (TimeSystem): Destination time system

    Returns:
        float: Offset between time systems in seconds

    Example:
        ```python
        import brahe as bh

        # Get offset from TT to TAI on January 1, 2024
        offset = bh.time_system_offset_for_datetime(
            2024, 1, 1, 0, 0, 0.0, 0.0,
            bh.TimeSystem.TT, bh.TimeSystem.TAI
        )
        print(f"TT to TAI offset: {offset} seconds")
        # Output: TT to TAI offset: -32.184 seconds
        ```
    """

def time_system_offset_for_jd(jd, time_system_src, time_system_dst):
    """
    Calculate the offset between two time systems for a given Julian Date.

    Args:
        jd (float): Julian date
        time_system_src (TimeSystem): Source time system
        time_system_dst (TimeSystem): Destination time system

    Returns:
        float: Offset between time systems in seconds

    Example:
        ```python
        import brahe as bh

        # Get offset from GPS to UTC at a specific Julian Date
        jd = 2460000.0
        offset = bh.time_system_offset_for_jd(jd, bh.TimeSystem.GPS, bh.TimeSystem.UTC)
        print(f"GPS to UTC offset: {offset} seconds")
        # Output: GPS to UTC offset: -18.0 seconds
        ```
    """

def time_system_offset_for_mjd(mjd, time_system_src, time_system_dst):
    """
    Calculate the offset between two time systems for a given Modified Julian Date.

    Args:
        mjd (float): Modified Julian date
        time_system_src (TimeSystem): Source time system
        time_system_dst (TimeSystem): Destination time system

    Returns:
        float: Offset between time systems in seconds

    Example:
        ```python
        import brahe as bh

        # Get offset from UTC to TAI at J2000 epoch
        mjd_j2000 = 51544.0
        offset = bh.time_system_offset_for_mjd(mjd_j2000, bh.TimeSystem.UTC, bh.TimeSystem.TAI)
        print(f"UTC to TAI offset: {offset} seconds")
        # Output: UTC to TAI offset: 32.0 seconds
        ```
    """

def validate_tle_line(line):
    """
    Validate single TLE line.

    Args:
        line (str): TLE line to validate.

    Returns:
        bool: True if the line is valid.
    """

def validate_tle_lines(line1, line2):
    """
    Validate TLE lines.

    Args:
        line1 (str): First line of TLE data.
        line2 (str): Second line of TLE data.

    Returns:
        bool: True if both lines are valid.
    """
