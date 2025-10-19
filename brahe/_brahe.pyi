"""Type stubs for brahe._brahe module - AUTO-GENERATED"""

from typing import Any, List, Tuple, Optional, Union
import numpy as np

# Type aliases for access constraints
AccessConstraint = Union[
    "AscDscConstraint",
    "ConstraintAll",
    "ConstraintAny",
    "ConstraintNot",
    "ElevationConstraint",
    "ElevationMaskConstraint",
    "LocalTimeConstraint",
    "LookDirectionConstraint",
    "OffNadirConstraint",
]

# Classes

class AccessProperties:
    """Properties computed for an access window.

    AccessProperties contains geometric properties (azimuth, elevation, off-nadir angles,
    local time, look direction, ascending/descending) computed over an access window,
    plus a dictionary of additional custom properties.

    Attributes:
        azimuth_open (float): Azimuth angle at window opening (degrees, 0-360)
        azimuth_close (float): Azimuth angle at window closing (degrees, 0-360)
        elevation_min (float): Minimum elevation angle (degrees)
        elevation_max (float): Maximum elevation angle (degrees)
        off_nadir_min (float): Minimum off-nadir angle (degrees)
        off_nadir_max (float): Maximum off-nadir angle (degrees)
        local_time (float): Local solar time (seconds since midnight, 0-86400)
        look_direction (LookDirection): Required look direction (Left or Right)
        asc_dsc (AscDsc): Pass type (Ascending or Descending)

    Example:
        ```python
        import brahe as bh

        # Access properties are typically computed by the access computation system
        # This example shows accessing the properties
        props = ...  # From access computation

        print(f"Azimuth at open: {props.azimuth_open}°")
        print(f"Max elevation: {props.elevation_max}°")
        print(f"Look direction: {props.look_direction}")

        # Access additional custom properties
        if "signal_strength" in props.additional:
            print(f"Signal: {props.additional['signal_strength']}")
        ```
    """

    def __init__(self) -> None:
        """Initialize instance."""
        ...

    @property
    def additional(self) -> AdditionalPropertiesDict:
        """Get additional properties as a dict-like wrapper.

        Returns a dictionary-like object that automatically converts between
        Python types and internal PropertyValue representation.

        Supported Python types:
        - float -> Scalar
        - list[float] -> Vector
        - bool -> Boolean
        - str -> String
        - dict -> Json

        Returns:
            AdditionalPropertiesDict: Dict-like wrapper for additional properties

        Example:
            ```python
            # Dict-style assignment
            props.additional["doppler_shift"] = 2500.0
            props.additional["snr_values"] = [10.5, 12.3, 15.1]
            props.additional["has_eclipse"] = False

            # Dict-style access
            print(props.additional["doppler_shift"])  # 2500.0

            # Dict methods
            if "doppler_shift" in props.additional:
                del props.additional["doppler_shift"]

            # Iteration
            for key in props.additional.keys():
                print(key, props.additional[key])
            ```
        """
        ...

    @property
    def asc_dsc(self) -> Any:
        """TODO: Add docstring"""
        ...

    @property
    def azimuth_close(self) -> Any:
        """TODO: Add docstring"""
        ...

    @property
    def azimuth_open(self) -> Any:
        """TODO: Add docstring"""
        ...

    @property
    def elevation_max(self) -> Any:
        """TODO: Add docstring"""
        ...

    @property
    def elevation_min(self) -> Any:
        """TODO: Add docstring"""
        ...

    @property
    def local_time(self) -> Any:
        """TODO: Add docstring"""
        ...

    @property
    def look_direction(self) -> Any:
        """TODO: Add docstring"""
        ...

    @property
    def off_nadir_max(self) -> Any:
        """TODO: Add docstring"""
        ...

    @property
    def off_nadir_min(self) -> Any:
        """TODO: Add docstring"""
        ...

class AccessPropertyComputer:
    """Base class for custom access property computers.

    Subclass this class and implement the `compute` and `property_names` methods
    to create custom property calculations that can be applied to access windows.

    The compute method is called for each access window and should return a dictionary
    of property names to values. Properties can be scalars, vectors, time series,
    booleans, strings, or any JSON-serializable value.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        class DopplerComputer(bh.AccessPropertyComputer):
            '''Computes Doppler shift at window midtime.'''

            def compute(self, window, satellite_state_ecef, location_ecef):
                '''
                Args:
                    window: AccessWindow with timing information
                    satellite_state_ecef: Satellite state [x,y,z,vx,vy,vz] in ECEF (m, m/s)
                    location_ecef: Location position [x,y,z] in ECEF (m)

                Returns:
                    dict: Property name -> value
                '''
                # Extract velocity
                vx, vy, vz = satellite_state_ecef[3:6]

                # Line-of-sight vector
                sat_pos = satellite_state_ecef[:3]
                los = location_ecef - sat_pos
                los_unit = los / np.linalg.norm(los)

                # Radial velocity
                sat_vel = np.array([vx, vy, vz])
                radial_velocity = np.dot(sat_vel, los_unit)

                # Doppler shift (L-band)
                freq_hz = 1.57542e9  # GPS L1
                doppler_hz = -radial_velocity * freq_hz / bh.C_LIGHT

                return {"doppler_shift": doppler_hz}

            def property_names(self):
                '''Return list of property names this computer produces.'''
                return ["doppler_shift"]

        # Use with access computation (future)
        computer = DopplerComputer()
        # accesses = bh.compute_accesses(..., property_computers=[computer])
        ```

    Notes:
        - The `compute` method receives ECEF coordinates in SI units (meters, m/s)
        - Property values are automatically converted to appropriate Rust types
        - The window parameter provides access to timing via:
          - `window.window_open`: Start epoch
          - `window.window_close`: End epoch
          - `window.midtime()`: Midpoint epoch
          - `window.duration()`: Duration in seconds
    """

    def __init__(self) -> None:
        """Initialize instance."""
        ...

    def compute(
        self,
        window: AccessWindow,
        satellite_state_ecef: np.ndarray,
        location_ecef: np.ndarray,
    ) -> dict:
        """Compute custom properties for an access window.

        Override this method in your subclass to implement custom property calculations.

        Args:
            window (AccessWindow): Access window with timing information
            satellite_state_ecef (ndarray): Satellite state in ECEF [x,y,z,vx,vy,vz] (meters, m/s)
            location_ecef (ndarray): Location position in ECEF [x,y,z] (meters)

        Returns:
            dict: Dictionary mapping property names (str) to values (scalar, list, dict, etc.)
        """
        ...

    def property_names(self) -> list[str]:
        """Return list of property names this computer will produce.

        Override this method to return the list of property names that your
        compute() method will include in its returned dictionary.

        Returns:
            list[str]: List of property names
        """
        ...

class AccessSearchConfig:
    """Configuration for access search grid parameters.

    Controls the time step and adaptive stepping behavior for access window finding.

    Args:
        initial_time_step (float): Initial time step in seconds for grid search (default: 60.0)
        adaptive_step (bool): Enable adaptive stepping after first access (default: False)
        adaptive_fraction (float): Fraction of orbital period to use for adaptive step (default: 0.75)

    Example:
        ```python
        import brahe as bh

        # Create a config with custom parameters
        config = bh.AccessSearchConfig(
            initial_time_step=30.0,
            adaptive_step=True,
            adaptive_fraction=0.5
        )

        # Use config with location_accesses
        windows = bh.location_accesses(
            station, prop, search_start, search_end,
            constraint, config=config
        )
        ```
    """

    def __init__(
        self, initial_time_step: float, adaptive_step: bool, adaptive_fraction: float
    ) -> None:
        """Initialize instance."""
        ...

    @property
    def adaptive_fraction(self) -> float:
        """Get the adaptive fraction (fraction of orbital period).

        Returns:
            float: Adaptive fraction
        """
        ...

    @property
    def adaptive_step(self) -> bool:
        """Get whether adaptive stepping is enabled.

        Returns:
            bool: Adaptive stepping flag
        """
        ...

    @property
    def initial_time_step(self) -> float:
        """Get the initial time step in seconds.

        Returns:
            float: Initial time step
        """
        ...

class AccessWindow:
    """An access window representing a period of time when access constraints are satisfied.

    AccessWindow stores the opening and closing times of an access period, along with
    computed properties for that window.

    Args:
        window_open (Epoch): Opening time of the access window
        window_close (Epoch): Closing time of the access window

    Example:
        ```python
        import brahe as bh

        # Create an access window
        t_open = bh.Epoch(2024, 1, 1, 12, 0, 0.0)
        t_close = bh.Epoch(2024, 1, 1, 12, 10, 0.0)
        window = bh.AccessWindow(t_open, t_close)

        # Access window properties
        print(f"Duration: {window.duration()} seconds")
        print(f"Midpoint: {window.midtime()}")
        ```
    """

    def __init__(self, window_open: Epoch, window_close: Epoch) -> None:
        """Initialize instance."""
        ...

    def duration(self) -> float:
        """Get the duration of the access window in seconds.

        Returns:
            float: Duration in seconds
        """
        ...

    def end(self) -> Epoch:
        """Get the end time of the access window.

        Returns:
            Epoch: Closing time of the window
        """
        ...

    def midtime(self) -> Epoch:
        """Get the midpoint time of the access window.

        Returns:
            Epoch: Midpoint time (average of start and end)
        """
        ...

    def start(self) -> Epoch:
        """Get the start time of the access window.

        Returns:
            Epoch: Opening time of the window
        """
        ...

    @property
    def location_id(self) -> Optional[int]:
        """Get the location ID if available.

        Returns:
            Optional[int]: ID of the location, or None if not set
        """
        ...

    @property
    def location_name(self) -> Optional[str]:
        """Get the location name if available.

        Returns:
            Optional[str]: Name of the location, or None if not set
        """
        ...

    @property
    def properties(self) -> AccessProperties:
        """Get the access properties.

        Returns:
            AccessProperties: Computed properties for this access window
        """
        ...

    @property
    def satellite_id(self) -> Optional[int]:
        """Get the satellite/object ID if available.

        Returns:
            Optional[int]: ID of the satellite, or None if not set
        """
        ...

    @property
    def satellite_name(self) -> Optional[str]:
        """Get the satellite/object name if available.

        Returns:
            Optional[str]: Name of the satellite, or None if not set
        """
        ...

class AdditionalPropertiesDict:
    """Python dictionary interface for additional access properties.

    Provides dict-like access to additional properties with automatic type conversion.
    """

    def __init__(self) -> None:
        """Initialize instance."""
        ...

    def clear(self) -> Any:
        """Remove all properties."""
        ...

    def get(self) -> Any:
        """Get property value with optional default."""
        ...

    def items(self) -> Any:
        """Return a list of (key, value) tuples."""
        ...

    def keys(self) -> Any:
        """Return a list of property keys."""
        ...

    def update(self) -> Tuple[int, ...]:
        """Update properties from another dict."""
        ...

    def values(self) -> Any:
        """Return a list of property values."""
        ...

class AngleFormat:
    """Python wrapper for AngleFormat enum"""

    def __init__(self) -> None:
        """Initialize instance."""
        ...

    @property
    def DEGREES(self) -> Any:
        """Python wrapper for AngleFormat enum"""
        ...

    @property
    def RADIANS(self) -> Any:
        """Python wrapper for AngleFormat enum"""
        ...

class AscDsc:
    """Ascending or descending pass type for satellite orbits.

    Indicates whether a satellite is moving from south to north (ascending) or
    north to south (descending) in its orbit. This is determined by the sign of
    the Z-component of the velocity vector in ECEF coordinates.

    This is useful for:
    - Sun-synchronous orbits that prefer specific pass types
    - Minimizing lighting variation between passes
    - Coordinating multi-satellite observations

    Attributes:
        ASCENDING: Satellite moving from south to north (vz > 0 in ECEF)
        DESCENDING: Satellite moving from north to south (vz < 0 in ECEF)
        EITHER: Either ascending or descending is acceptable

    Example:
        ```python
        import brahe as bh

        # Create a constraint for ascending passes only
        constraint = bh.AscDscConstraint(allowed=bh.AscDsc.ASCENDING)

        # Create a constraint for descending passes only
        constraint = bh.AscDscConstraint(allowed=bh.AscDsc.DESCENDING)

        # Accept either type
        constraint = bh.AscDscConstraint(allowed=bh.AscDsc.EITHER)

        # Compare pass types
        assert bh.AscDsc.ASCENDING != bh.AscDsc.DESCENDING
        assert bh.AscDsc.ASCENDING == bh.AscDsc.ASCENDING
        ```
    """

    def __init__(self) -> None:
        """Initialize instance."""
        ...

    @property
    def ASCENDING(self) -> Any:
        """Ascending or descending pass type for satellite orbits.

        Indicates whether a satellite is moving from south to north (ascending) or
        north to south (descending) in its orbit. This is determined by the sign of
        the Z-component of the velocity vector in ECEF coordinates.

        This is useful for:
        - Sun-synchronous orbits that prefer specific pass types
        - Minimizing lighting variation between passes
        - Coordinating multi-satellite observations

        Attributes:
            ASCENDING: Satellite moving from south to north (vz > 0 in ECEF)
            DESCENDING: Satellite moving from north to south (vz < 0 in ECEF)
            EITHER: Either ascending or descending is acceptable

        Example:
            ```python
            import brahe as bh

            # Create a constraint for ascending passes only
            constraint = bh.AscDscConstraint(allowed=bh.AscDsc.ASCENDING)

            # Create a constraint for descending passes only
            constraint = bh.AscDscConstraint(allowed=bh.AscDsc.DESCENDING)

            # Accept either type
            constraint = bh.AscDscConstraint(allowed=bh.AscDsc.EITHER)

            # Compare pass types
            assert bh.AscDsc.ASCENDING != bh.AscDsc.DESCENDING
            assert bh.AscDsc.ASCENDING == bh.AscDsc.ASCENDING
            ```
        """
        ...

    @property
    def DESCENDING(self) -> Any:
        """Ascending or descending pass type for satellite orbits.

        Indicates whether a satellite is moving from south to north (ascending) or
        north to south (descending) in its orbit. This is determined by the sign of
        the Z-component of the velocity vector in ECEF coordinates.

        This is useful for:
        - Sun-synchronous orbits that prefer specific pass types
        - Minimizing lighting variation between passes
        - Coordinating multi-satellite observations

        Attributes:
            ASCENDING: Satellite moving from south to north (vz > 0 in ECEF)
            DESCENDING: Satellite moving from north to south (vz < 0 in ECEF)
            EITHER: Either ascending or descending is acceptable

        Example:
            ```python
            import brahe as bh

            # Create a constraint for ascending passes only
            constraint = bh.AscDscConstraint(allowed=bh.AscDsc.ASCENDING)

            # Create a constraint for descending passes only
            constraint = bh.AscDscConstraint(allowed=bh.AscDsc.DESCENDING)

            # Accept either type
            constraint = bh.AscDscConstraint(allowed=bh.AscDsc.EITHER)

            # Compare pass types
            assert bh.AscDsc.ASCENDING != bh.AscDsc.DESCENDING
            assert bh.AscDsc.ASCENDING == bh.AscDsc.ASCENDING
            ```
        """
        ...

    @property
    def EITHER(self) -> Any:
        """Ascending or descending pass type for satellite orbits.

        Indicates whether a satellite is moving from south to north (ascending) or
        north to south (descending) in its orbit. This is determined by the sign of
        the Z-component of the velocity vector in ECEF coordinates.

        This is useful for:
        - Sun-synchronous orbits that prefer specific pass types
        - Minimizing lighting variation between passes
        - Coordinating multi-satellite observations

        Attributes:
            ASCENDING: Satellite moving from south to north (vz > 0 in ECEF)
            DESCENDING: Satellite moving from north to south (vz < 0 in ECEF)
            EITHER: Either ascending or descending is acceptable

        Example:
            ```python
            import brahe as bh

            # Create a constraint for ascending passes only
            constraint = bh.AscDscConstraint(allowed=bh.AscDsc.ASCENDING)

            # Create a constraint for descending passes only
            constraint = bh.AscDscConstraint(allowed=bh.AscDsc.DESCENDING)

            # Accept either type
            constraint = bh.AscDscConstraint(allowed=bh.AscDsc.EITHER)

            # Compare pass types
            assert bh.AscDsc.ASCENDING != bh.AscDsc.DESCENDING
            assert bh.AscDsc.ASCENDING == bh.AscDsc.ASCENDING
            ```
        """
        ...

class AscDscConstraint:
    """Ascending/descending pass constraint.

    Constrains access based on whether the satellite is on an ascending or
    descending pass (moving north or south).

    Args:
        allowed (AscDsc): Required pass type (ASCENDING, DESCENDING, or EITHER)

    Example:
        ```python
        import brahe as bh

        # Only ascending passes
        constraint = bh.AscDscConstraint(allowed=bh.AscDsc.ASCENDING)

        # Either type is acceptable
        constraint = bh.AscDscConstraint(allowed=bh.AscDsc.EITHER)
        ```
    """

    def __init__(self, allowed: AscDsc) -> None:
        """Initialize instance."""
        ...

    def evaluate(
        self, epoch: Epoch, sat_state_ecef: np.ndarray, location_ecef: np.ndarray
    ) -> bool:
        """Evaluate whether the constraint is satisfied.

        Args:
            epoch (Epoch): Time of evaluation
            sat_state_ecef (ndarray): Satellite state in ECEF [x, y, z, vx, vy, vz] (meters, m/s)
            location_ecef (ndarray): Ground location in ECEF [x, y, z] (meters)

        Returns:
            bool: True if constraint is satisfied, False otherwise
        """
        ...

    def name(self) -> Any:
        """Get the constraint name"""
        ...

class CachingEOPProvider:
    """Caching EOP provider that automatically downloads updated files when stale.

    This provider wraps a FileEOPProvider and adds automatic cache management.
    It checks the age of the EOP file and downloads updated versions when the file
    exceeds the maximum age threshold. If the file doesn't exist, it will be
    downloaded on initialization.

    Args:
        filepath (str): Path to the EOP file (will be created if it doesn't exist)
        eop_type (str): Type of EOP file - "C04" for IERS C04 format or
            "StandardBulletinA" for IERS finals2000A.all format
        max_age_seconds (int): Maximum age of file in seconds before triggering
            a refresh. Common values: 86400 (1 day), 604800 (7 days)
        auto_refresh (bool): If True, automatically checks file age and refreshes
            on every data access. If False, only checks on initialization and
            manual refresh() calls
        interpolate (bool): Enable linear interpolation between tabulated EOP
            values. Recommended: True for smoother data
        extrapolate (str): Behavior for dates outside EOP data range:
            "Hold" (use last known value), "Zero" (return 0.0), or "Error" (raise exception)

    Raises:
        RuntimeError: If file download fails or file is invalid

    Example:
        ```python
        import brahe as bh

        # Manual refresh mode (recommended for performance)
        provider = bh.CachingEOPProvider(
            filepath="./eop_data/finals.all.iau2000.txt",
            eop_type="StandardBulletinA",
            max_age_seconds=7 * 86400,  # 7 days
            auto_refresh=False,
            interpolate=True,
            extrapolate="Hold"
        )
        bh.set_global_eop_provider_from_caching_provider(provider)

        # Check and refresh as needed
        provider.refresh()

        # Auto-refresh mode (convenience)
        auto_provider = bh.CachingEOPProvider(
            filepath="./eop_data/finals.all.iau2000.txt",
            eop_type="StandardBulletinA",
            max_age_seconds=24 * 3600,  # 24 hours
            auto_refresh=True,  # Checks on every access
            interpolate=True,
            extrapolate="Hold"
        )
        ```
    """

    def __init__(
        self,
        filepath: str,
        eop_type: str,
        max_age_seconds: int,
        auto_refresh: bool,
        interpolate: bool,
        extrapolate: str,
    ) -> None:
        """Initialize instance."""
        ...

    def eop_type(self) -> str:
        """Get the EOP file type.

        Returns:
            str: EOP type ("C04", "StandardBulletinA", etc.)
        """
        ...

    def extrapolation(self) -> str:
        """Get the extrapolation method.

        Returns:
            str: Extrapolation method ("Hold", "Zero", or "Error")
        """
        ...

    def file_age(self) -> float:
        """Get the age of the currently loaded EOP file in seconds.

        Returns:
            float: Age of the loaded file in seconds

        Example:
            ```python
            import brahe as bh

            provider = bh.CachingEOPProvider(
                "./eop_data/finals.all.iau2000.txt",
                "StandardBulletinA",
                7 * 86400,
                False,
                True,
                "Hold"
            )

            age = provider.file_age()
            print(f"EOP file age: {age:.2f} seconds")
            ```
        """
        ...

    def file_epoch(self) -> Epoch:
        """Get the epoch when the EOP file was last loaded.

        Returns:
            Epoch: Epoch in UTC when file was loaded

        Example:
            ```python
            import brahe as bh

            provider = bh.CachingEOPProvider(
                "./eop_data/finals.all.iau2000.txt",
                "StandardBulletinA",
                7 * 86400,
                False,
                True,
                "Hold"
            )

            file_epoch = provider.file_epoch()
            print(f"EOP file loaded at: {file_epoch}")
            ```
        """
        ...

    def get_dxdy(self, mjd: float) -> tuple[float, float]:
        """Get celestial pole offsets for a given MJD.

        Args:
            mjd (float): Modified Julian Date

        Returns:
            tuple[float, float]: Celestial pole offsets dx and dy in radians
        """
        ...

    def get_eop(self, mjd: float) -> Tuple:
        """Get all EOP parameters for a given MJD.

        Args:
            mjd (float): Modified Julian Date

        Returns:
            tuple: (pm_x, pm_y, ut1_utc, dx, dy, lod)
        """
        ...

    def get_lod(self, mjd: float) -> float:
        """Get length of day offset for a given MJD.

        Args:
            mjd (float): Modified Julian Date

        Returns:
            float: Length of day offset in seconds
        """
        ...

    def get_pm(self, mjd: float) -> tuple[float, float]:
        """Get polar motion components for a given MJD.

        Args:
            mjd (float): Modified Julian Date

        Returns:
            tuple[float, float]: Polar motion x and y components in radians
        """
        ...

    def get_ut1_utc(self, mjd: float) -> float:
        """Get UT1-UTC time difference for a given MJD.

        Args:
            mjd (float): Modified Julian Date

        Returns:
            float: UT1-UTC time difference in seconds
        """
        ...

    def interpolation(self) -> bool:
        """Check if interpolation is enabled.

        Returns:
            bool: True if interpolation is enabled
        """
        ...

    def is_initialized(self) -> bool:
        """Check if the provider is initialized.

        Returns:
            bool: True if initialized
        """
        ...

    def len(self) -> int:
        """Get the number of EOP data points.

        Returns:
            int: Number of EOP data points
        """
        ...

    def mjd_last_dxdy(self) -> float:
        """Get the last MJD with valid celestial pole offset data.

        Returns:
            float: Last MJD with dX/dY data
        """
        ...

    def mjd_last_lod(self) -> float:
        """Get the last MJD with valid LOD data.

        Returns:
            float: Last MJD with length of day data
        """
        ...

    def mjd_max(self) -> float:
        """Get the maximum MJD in the dataset.

        Returns:
            float: Maximum Modified Julian Date
        """
        ...

    def mjd_min(self) -> float:
        """Get the minimum MJD in the dataset.

        Returns:
            float: Minimum Modified Julian Date
        """
        ...

    def refresh(self) -> Any:
        """Manually refresh the cached EOP data.

        Checks if the file needs updating and downloads a new version if necessary.

        Example:
            ```python
            import brahe as bh

            provider = bh.CachingEOPProvider(
                "./eop_data/finals.all.iau2000.txt",
                "StandardBulletinA",
                7 * 86400,
                False,
                True,
                "Hold"
            )

            # Later, manually force a refresh check
            provider.refresh()
            ```
        """
        ...

class ConstraintAll:
    """Composite constraint combining multiple constraints with AND logic.

    All constraints must be satisfied for the composite to evaluate to true.

    Args:
        constraints (list): List of constraint objects to combine with AND logic

    Example:
        ```python
        import brahe as bh

        # Ground station with multiple requirements
        elev = bh.ElevationConstraint(min_elevation_deg=5.0, max_elevation_deg=None)
        time = bh.LocalTimeConstraint(time_windows=[(600, 1800)])
        combined = bh.ConstraintAll(constraints=[elev, time])
        ```
    """

    def __init__(self, constraints: List) -> None:
        """Initialize instance."""
        ...

    def evaluate(
        self, epoch: Epoch, sat_state_ecef: np.ndarray, location_ecef: np.ndarray
    ) -> bool:
        """Evaluate whether the constraint is satisfied.

        Args:
            epoch (Epoch): Time of evaluation
            sat_state_ecef (ndarray): Satellite state in ECEF [x, y, z, vx, vy, vz] (meters, m/s)
            location_ecef (ndarray): Ground location in ECEF [x, y, z] (meters)

        Returns:
            bool: True if ALL constraints are satisfied, False otherwise
        """
        ...

    def name(self) -> Any:
        """Get the constraint name"""
        ...

class ConstraintAny:
    """Composite constraint combining multiple constraints with OR logic.

    At least one constraint must be satisfied for the composite to evaluate to true.

    Args:
        constraints (list): List of constraint objects to combine with OR logic

    Example:
        ```python
        import brahe as bh

        # Accept either high elevation or specific time window
        elev = bh.ElevationConstraint(min_elevation_deg=60.0, max_elevation_deg=None)
        time = bh.LocalTimeConstraint(time_windows=[(1200, 1400)])
        combined = bh.ConstraintAny(constraints=[elev, time])
        ```
    """

    def __init__(self, constraints: List) -> None:
        """Initialize instance."""
        ...

    def evaluate(
        self, epoch: Epoch, sat_state_ecef: np.ndarray, location_ecef: np.ndarray
    ) -> bool:
        """Evaluate whether the constraint is satisfied.

        Args:
            epoch (Epoch): Time of evaluation
            sat_state_ecef (ndarray): Satellite state in ECEF [x, y, z, vx, vy, vz] (meters, m/s)
            location_ecef (ndarray): Ground location in ECEF [x, y, z] (meters)

        Returns:
            bool: True if AT LEAST ONE constraint is satisfied, False otherwise
        """
        ...

    def name(self) -> Any:
        """Get the constraint name"""
        ...

class ConstraintNot:
    """Composite constraint negating another constraint with NOT logic.

    The negated constraint must NOT be satisfied for this to evaluate to true.

    Args:
        constraint: Constraint object to negate

    Example:
        ```python
        import brahe as bh

        # Avoid low elevation angles (i.e., require high elevation)
        low_elev = bh.ElevationConstraint(min_elevation_deg=None, max_elevation_deg=10.0)
        high_elev = bh.ConstraintNot(constraint=low_elev)
        ```
    """

    def __init__(self) -> None:
        """Initialize instance."""
        ...

    def evaluate(
        self, epoch: Epoch, sat_state_ecef: np.ndarray, location_ecef: np.ndarray
    ) -> bool:
        """Evaluate whether the constraint is satisfied.

        Args:
            epoch (Epoch): Time of evaluation
            sat_state_ecef (ndarray): Satellite state in ECEF [x, y, z, vx, vy, vz] (meters, m/s)
            location_ecef (ndarray): Ground location in ECEF [x, y, z] (meters)

        Returns:
            bool: True if the negated constraint is NOT satisfied, False otherwise
        """
        ...

    def name(self) -> Any:
        """Get the constraint name"""
        ...

class DTrajectory:
    """Dynamic-dimension trajectory container.

    Stores a sequence of N-dimensional states at specific epochs with support
    for interpolation and automatic state eviction policies. Dimension is
    determined at runtime.
    """

    def __init__(self) -> None:
        """Initialize instance."""
        ...

    @classmethod
    def from_data(
        cls,
        epochs: list[Epoch],
        states: np.ndarray,
        interpolation_method: InterpolationMethod,
    ) -> DTrajectory:
        """Create a trajectory from existing data.

        Args:
            epochs (list[Epoch]): List of time epochs
            states (numpy.ndarray): 2D array of states with shape (num_epochs, dimension)
                where each row is a state vector
            interpolation_method (InterpolationMethod): Interpolation method (default Linear)

        Returns:
            DTrajectory: New trajectory instance populated with data
        """
        ...

    def add(self, epoch: Epoch, state: np.ndarray) -> Any:
        """Add a state to the trajectory.

        Args:
            epoch (Epoch): Time of the state
            state (numpy.ndarray): N-element state vector where N is the trajectory dimension

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.DTrajectory(6)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            ```
        """
        ...

    def clear(self) -> Any:
        """Clear all states from the trajectory."""
        ...

    def dimension(self) -> int:
        """Get the trajectory dimension (method form).

        Returns:
            int: Dimension of the trajectory

        Example:
            ```python
            import brahe as bh

            traj = bh.DTrajectory(6)
            print(f"Dimension: {traj.dimension()}")
            ```
        """
        ...

    def end_epoch(self) -> Any:
        """Get end epoch of trajectory"""
        ...

    def epoch_at_idx(self, index: int) -> Epoch:
        """Get epoch at a specific index

        Arguments:
            index (int): Index of the epoch

        Returns:
            Epoch: Epoch at index

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.DTrajectory(6)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            retrieved_epc = traj.epoch_at_idx(0)
            ```
        """
        ...

    def first(self) -> Tuple:
        """Get the first (epoch, state) tuple in the trajectory, if any exists.

        Returns:
            tuple or None: Tuple of (Epoch, numpy.ndarray) for first state, or None if empty

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            first_epc, first_state = traj.first()
            ```
        """
        ...

    def get(self, index: int) -> Tuple:
        """Get both epoch and state at a specific index.

        Args:
            index (int): Index to retrieve

        Returns:
            tuple: Tuple of (Epoch, numpy.ndarray) for epoch and state at the index

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            ret_epc, ret_state = traj.get(0)
            ```
        """
        ...

    def get_eviction_policy(self) -> str:
        """Get current eviction policy.

        Returns:
            str: String representation of eviction policy

        Example:
            ```python
            import brahe as bh

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            policy = traj.get_eviction_policy()
            ```
        """
        ...

    def get_interpolation_method(self) -> InterpolationMethod:
        """Get interpolation method.

        Returns:
            InterpolationMethod: Current interpolation method

        Example:
            ```python
            import brahe as bh

            traj = bh.DTrajectory(6)
            method = traj.get_interpolation_method()
            ```
        """
        ...

    def index_after_epoch(self, epoch: Epoch) -> int:
        """Get the index of the state at or after the given epoch.

        Args:
            epoch (Epoch): Target epoch

        Returns:
            int: Index of the state at or after the target epoch

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc1, state)
            epc2 = bh.Epoch.from_datetime(2024, 1, 1, 11, 59, 0.0, 0.0, bh.TimeSystem.UTC)
            index = traj.index_after_epoch(epc2)
            ```
        """
        ...

    def index_before_epoch(self, epoch: Epoch) -> int:
        """Get the index of the state at or before the given epoch.

        Args:
            epoch (Epoch): Target epoch

        Returns:
            int: Index of the state at or before the target epoch

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc1, state)
            epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
            index = traj.index_before_epoch(epc2)
            ```
        """
        ...

    def interpolate(self, epoch: Epoch) -> np.ndarray:
        """Interpolate state at a given epoch using the configured interpolation method.

        Args:
            epoch (Epoch): Target epoch

        Returns:
            numpy.ndarray: Interpolated state vector

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state1 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc1, state1)
            epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 2, 0.0, 0.0, bh.TimeSystem.UTC)
            state2 = np.array([bh.R_EARTH + 510e3, 0.0, 0.0, 0.0, 7650.0, 0.0])
            traj.add(epc2, state2)
            epc_mid = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
            state_interp = traj.interpolate(epc_mid)
            ```
        """
        ...

    def interpolate_linear(self, epoch: Epoch) -> np.ndarray:
        """Interpolate state at a given epoch using linear interpolation.

        Args:
            epoch (Epoch): Target epoch

        Returns:
            numpy.ndarray: Linearly interpolated state vector

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state1 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc1, state1)
            epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 2, 0.0, 0.0, bh.TimeSystem.UTC)
            state2 = np.array([bh.R_EARTH + 510e3, 0.0, 0.0, 0.0, 7650.0, 0.0])
            traj.add(epc2, state2)
            epc_mid = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
            state_interp = traj.interpolate_linear(epc_mid)
            ```
        """
        ...

    def is_empty(self) -> bool:
        """Check if trajectory is empty.

        Returns:
            bool: True if trajectory contains no states, False otherwise

        Example:
            ```python
            import brahe as bh

            traj = bh.DTrajectory(6)
            print(f"Is empty: {traj.is_empty()}")
            ```
        """
        ...

    def last(self) -> Tuple:
        """Get the last (epoch, state) tuple in the trajectory, if any exists.

        Returns:
            tuple or None: Tuple of (Epoch, numpy.ndarray) for last state, or None if empty

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            last_epc, last_state = traj.last()
            ```
        """
        ...

    def len(self) -> int:
        """Get the number of states in the trajectory (alias for length).

        Returns:
            int: Number of states in the trajectory

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.DTrajectory(6)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            print(f"Number of states: {traj.len()}")
            ```
        """
        ...

    def nearest_state(self, epoch: Epoch) -> Tuple:
        """Get the nearest state to a given epoch.

        Args:
            epoch (Epoch): Target epoch

        Returns:
            tuple: Tuple of (Epoch, numpy.ndarray) containing the nearest state

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc1, state)
            epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 30.0, 0.0, bh.TimeSystem.UTC)
            nearest_epc, nearest_state = traj.nearest_state(epc2)
            ```
        """
        ...

    def remove(self, index: int) -> Tuple:
        """Remove a state at a specific index.

        Args:
            index (int): Index of the state to remove

        Returns:
            tuple: Tuple of (Epoch, numpy.ndarray) for the removed epoch and state

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            removed_epc, removed_state = traj.remove(0)
            ```
        """
        ...

    def remove_epoch(self, epoch: Epoch) -> np.ndarray:
        """Remove a state at a specific epoch.

        Args:
            epoch (Epoch): Epoch of the state to remove

        Returns:
            numpy.ndarray: The removed state vector

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            removed_state = traj.remove_epoch(epc)
            ```
        """
        ...

    def set_eviction_policy_max_age(self, max_age: float) -> Any:
        """Set maximum age for trajectory states.

        Args:
            max_age (float): Maximum age in seconds relative to most recent state
        """
        ...

    def set_eviction_policy_max_size(self, max_size: int) -> Any:
        """Set maximum trajectory size.

        Args:
            max_size (int): Maximum number of states to retain
        """
        ...

    def set_interpolation_method(self, method: InterpolationMethod) -> Any:
        """Set interpolation method.

        Args:
            method (InterpolationMethod): New interpolation method

        Example:
            ```python
            import brahe as bh

            traj = bh.DTrajectory(6)
            method = bh.InterpolationMethod.LINEAR
            traj.set_interpolation_method(method)
            ```
        """
        ...

    def start_epoch(self) -> Any:
        """Get start epoch of trajectory"""
        ...

    def state_after_epoch(self, epoch: Epoch) -> Tuple:
        """Get the state at or after the given epoch.

        Args:
            epoch (Epoch): Target epoch

        Returns:
            tuple: Tuple of (Epoch, numpy.ndarray) containing state at or after the target epoch

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc1, state)
            epc2 = bh.Epoch.from_datetime(2024, 1, 1, 11, 59, 0.0, 0.0, bh.TimeSystem.UTC)
            ret_epc, ret_state = traj.state_after_epoch(epc2)
            ```
        """
        ...

    def state_at_idx(self, index: int) -> np.ndarray:
        """Get state at a specific index

        Arguments:
            index (int): Index of the state

        Returns:
            numpy.ndarray: State vector at index

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.DTrajectory(6)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            retrieved_state = traj.state_at_idx(0)
            ```
        """
        ...

    def state_before_epoch(self, epoch: Epoch) -> Tuple:
        """Get the state at or before the given epoch.

        Args:
            epoch (Epoch): Target epoch

        Returns:
            tuple: Tuple of (Epoch, numpy.ndarray) containing state at or before the target epoch

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc1, state)
            epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
            ret_epc, ret_state = traj.state_before_epoch(epc2)
            ```
        """
        ...

    def timespan(self) -> Any:
        """Get time span of trajectory in seconds"""
        ...

    def to_matrix(self) -> np.ndarray:
        """Get all states as a numpy array"""
        ...

    def with_eviction_policy_max_age(self, max_age: float) -> DTrajectory:
        """Set eviction policy to keep states within maximum age using builder pattern

        Arguments:
            max_age (float): Maximum age of states in seconds

        Returns:
            DTrajectory: Self with updated eviction policy

        Example:
            ```python
            import brahe as bh

            traj = bh.DTrajectory(6)
            traj = traj.with_eviction_policy_max_age(3600.0)
            ```
        """
        ...

    def with_eviction_policy_max_size(self, max_size: int) -> DTrajectory:
        """Set eviction policy to keep maximum number of states using builder pattern

        Arguments:
            max_size (int): Maximum number of states to retain

        Returns:
            DTrajectory: Self with updated eviction policy

        Example:
            ```python
            import brahe as bh

            traj = bh.DTrajectory(6)
            traj = traj.with_eviction_policy_max_size(1000)
            ```
        """
        ...

    def with_interpolation_method(
        self, interpolation_method: InterpolationMethod
    ) -> DTrajectory:
        """Set interpolation method using builder pattern

        Arguments:
            interpolation_method (InterpolationMethod): Interpolation method to use

        Returns:
            DTrajectory: Self with updated interpolation method

        Example:
            ```python
            import brahe as bh

            traj = bh.DTrajectory(6)
            traj = traj.with_interpolation_method(bh.InterpolationMethod.LINEAR)
            ```
        """
        ...

    @property
    def length(self) -> int:
        """Get the number of states in the trajectory.

        Returns:
            int: Number of states in the trajectory

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.DTrajectory(6)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            print(f"Trajectory length: {traj.length}")
            ```
        """
        ...

class ElevationConstraint:
    """Elevation angle constraint for satellite visibility.

    Constrains access based on the elevation angle of the satellite above
    the local horizon at the ground location.

    Args:
        min_elevation_deg (float | None): Minimum elevation angle in degrees, or None for no minimum
        max_elevation_deg (float | None): Maximum elevation angle in degrees, or None for no maximum

    Raises:
        ValueError: If both min and max are None (unbounded constraint is meaningless)

    Example:
        ```python
        import brahe as bh

        # Typical ground station constraint: 5° minimum elevation
        constraint = bh.ElevationConstraint(min_elevation_deg=5.0, max_elevation_deg=None)

        # Both bounds specified
        constraint = bh.ElevationConstraint(min_elevation_deg=5.0, max_elevation_deg=85.0)

        # Only maximum (e.g., avoid zenith)
        constraint = bh.ElevationConstraint(min_elevation_deg=None, max_elevation_deg=85.0)
        ```
    """

    def __init__(
        self, min_elevation_deg: float | None, max_elevation_deg: float | None
    ) -> None:
        """Initialize instance."""
        ...

    def evaluate(
        self, epoch: Epoch, sat_state_ecef: np.ndarray, location_ecef: np.ndarray
    ) -> bool:
        """Evaluate whether the constraint is satisfied.

        Args:
            epoch (Epoch): Time of evaluation
            sat_state_ecef (ndarray): Satellite state in ECEF [x, y, z, vx, vy, vz] (meters, m/s)
            location_ecef (ndarray): Ground location in ECEF [x, y, z] (meters)

        Returns:
            bool: True if constraint is satisfied, False otherwise
        """
        ...

    def name(self) -> Any:
        """Get the constraint name"""
        ...

class ElevationMaskConstraint:
    """Azimuth-dependent elevation mask constraint.

    Constrains access based on azimuth-dependent elevation masks.
    Useful for ground stations with terrain obstructions or antenna limitations.

    The mask is defined as a list of (azimuth, elevation) pairs in degrees.
    Linear interpolation is used between points, and the mask wraps at 0°/360°.

    Args:
        mask (list[tuple[float, float]]): List of (azimuth_deg, min_elevation_deg) pairs

    Example:
        ```python
        import brahe as bh

        # Ground station with terrain obstruction to the north
        mask = [
            (0.0, 15.0),     # North: 15° minimum
            (90.0, 5.0),     # East: 5° minimum
            (180.0, 5.0),    # South: 5° minimum
            (270.0, 5.0),    # West: 5° minimum
        ]
        constraint = bh.ElevationMaskConstraint(mask)
        ```
    """

    def __init__(self, mask: list[tuple[float, float]]) -> None:
        """Initialize instance."""
        ...

    def evaluate(
        self, epoch: Epoch, sat_state_ecef: np.ndarray, location_ecef: np.ndarray
    ) -> bool:
        """Evaluate whether the constraint is satisfied.

        Args:
            epoch (Epoch): Time of evaluation
            sat_state_ecef (ndarray): Satellite state in ECEF [x, y, z, vx, vy, vz] (meters, m/s)
            location_ecef (ndarray): Ground location in ECEF [x, y, z] (meters)

        Returns:
            bool: True if constraint is satisfied, False otherwise
        """
        ...

    def name(self) -> Any:
        """Get the constraint name"""
        ...

class EllipsoidalConversionType:
    """Python wrapper for EllipsoidalConversionType enum

    Specifies the type of ellipsoidal conversion used in coordinate transformations.
    """

    def __init__(self) -> None:
        """Initialize instance."""
        ...

    @property
    def GEOCENTRIC(self) -> Any:
        """Python wrapper for EllipsoidalConversionType enum

        Specifies the type of ellipsoidal conversion used in coordinate transformations.
        """
        ...

    @property
    def GEODETIC(self) -> Any:
        """Python wrapper for EllipsoidalConversionType enum

        Specifies the type of ellipsoidal conversion used in coordinate transformations.
        """
        ...

class Epoch:
    """Represents a specific instant in time.

    Epoch is the primary and preferred mechanism for representing time in brahe.
    It accurately represents, tracks, and compares instants in time with nanosecond precision.

    Internally, Epoch stores time in terms of days, seconds, and nanoseconds. This representation
    was chosen to enable accurate time system conversions using the IAU SOFA library (which operates
    in days and fractional days) while maintaining high precision for small time differences.
    The structure uses Kahan summation to accurately handle running sums over long periods without
    losing accuracy to floating-point rounding errors.

    All arithmetic operations (addition, subtraction) use seconds as the default unit and return
    time differences in seconds.

    The Epoch constructor accepts multiple input formats for convenience:

    - **Date components**: `Epoch(year, month, day)` - creates epoch at midnight
    - **Full datetime**: `Epoch(year, month, day, hour, minute, second, nanosecond)` - full precision
    - **Partial datetime**: `Epoch(year, month, day, hour)` or `Epoch(year, month, day, hour, minute)` etc.
    - **ISO 8601 string**: `Epoch("2024-01-01T12:00:00Z")` - parse from string
    - **Python datetime**: `Epoch(datetime_obj)` - convert from Python datetime
    - **Copy constructor**: `Epoch(other_epoch)` - create a copy
    - **Time system**: All constructors accept optional `time_system=` keyword argument (default: UTC)

    Example:
        ```python
        import brahe as bh
        from datetime import datetime

        # Multiple ways to create the same epoch
        epc1 = bh.Epoch(2024, 1, 1, 12, 0, 0.0, 0.0)
        epc2 = bh.Epoch("2024-01-01 12:00:00.000 UTC")
        epc3 = bh.Epoch(datetime(2024, 1, 1, 12, 0, 0))
        print(epc1)
        # Output: 2024-01-01 12:00:00.000 UTC

        # Create epoch at midnight
        midnight = bh.Epoch(2024, 1, 1)
        print(midnight)
        # Output: 2024-01-01 00:00:00.000 UTC

        # Use different time systems
        gps_time = bh.Epoch(2024, 1, 1, 12, 0, 0.0, 0.0, time_system=bh.GPS)
        print(gps_time)
        # Output: 2024-01-01 12:00:00.000 GPS

        # Perform arithmetic operations
        epoch2 = epc1 + 3600.0  # Add one hour (in seconds)
        diff = epoch2 - epc1     # Difference in seconds
        print(f"Time difference: {diff} seconds")
        # Output: Time difference: 3600.0 seconds

        # Legacy constructors still available
        epc4 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.UTC)
        epc5 = bh.Epoch.from_jd(2460310.0, bh.UTC)
        ```
    """

    def __init__(self) -> None:
        """Initialize instance."""
        ...

    @classmethod
    def from_date(
        cls, year: int, month: int, day: int, time_system: TimeSystem
    ) -> Epoch:
        """Create an Epoch from a calendar date at midnight.

        Args:
            year (int): Gregorian calendar year
            month (int): Month (1-12)
            day (int): Day of month (1-31)
            time_system (TimeSystem): Time system

        Returns:
            Epoch: The epoch representing midnight on the specified date

        Example:
            ```python
            import brahe as bh

            # Create an epoch at midnight on January 1, 2024 UTC
            epc = bh.Epoch.from_date(2024, 1, 1, bh.TimeSystem.UTC)
            print(epc)
            # Output: 2024-01-01T00:00:00.000000000 UTC

            # Create epoch in different time system
            epc_tai = bh.Epoch.from_date(2024, 6, 15, bh.TimeSystem.TAI)
            print(epc_tai)
            # Output: 2024-06-15T00:00:00.000000000 TAI
            ```
        """
        ...

    @classmethod
    def from_datetime(
        cls,
        year: int,
        month: int,
        day: int,
        hour: int,
        minute: int,
        second: float,
        nanosecond: float,
        time_system: TimeSystem,
    ) -> Epoch:
        """Create an Epoch from a complete Gregorian calendar date and time.

        Args:
            year (int): Gregorian calendar year
            month (int): Month (1-12)
            day (int): Day of month (1-31)
            hour (int): Hour (0-23)
            minute (int): Minute (0-59)
            second (float): Second with fractional part
            nanosecond (float): Nanosecond component
            time_system (TimeSystem): Time system

        Returns:
            Epoch: The epoch representing the specified date and time

        Example:
            ```python
            import brahe as bh

            # Create epoch for January 1, 2024 at 12:30:45.5 UTC
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 30, 45.5, 0.0, bh.TimeSystem.UTC)
            print(epc)
            # Output: 2024-01-01T12:30:45.500000000 UTC

            # With nanosecond precision
            epc_ns = bh.Epoch.from_datetime(2024, 6, 15, 14, 30, 0.0, 123456789.0, bh.TimeSystem.TAI)
            print(epc_ns)
            # Output: 2024-06-15T14:30:00.123456789 TAI
            ```
        """
        ...

    @classmethod
    def from_day_of_year(
        cls, year: int, day_of_year: float, time_system: TimeSystem
    ) -> Epoch:
        """Create an Epoch from a year and floating-point day-of-year.

        Args:
            year (int): Gregorian calendar year
            day_of_year (float): Day of year as a floating-point number
                (1.0 = January 1st, 1.5 = January 1st noon, etc.)
            time_system (TimeSystem): Time system

        Returns:
            Epoch: The epoch representing the specified day of year

        Example:
            ```python
            import brahe as bh

            # Create epoch for day 100 of 2024 at midnight
            epc = bh.Epoch.from_day_of_year(2024, 100.0, bh.TimeSystem.UTC)
            print(epc)
            # Output: 2024-04-09T00:00:00.000000000 UTC

            # Create epoch for day 100.5 (noon on day 100)
            epc_noon = bh.Epoch.from_day_of_year(2024, 100.5, bh.TimeSystem.UTC)
            year, month, day, hour, minute, second, ns = epc_noon.to_datetime()
            print(f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:06.3f}")
            # Output: 2024-04-09 12:00:00.000
            ```
        """
        ...

    @classmethod
    def from_gps_date(cls, week: int, seconds: float) -> Epoch:
        """Create an Epoch from GPS week and seconds.

        Args:
            week (int): GPS week number since GPS epoch (January 6, 1980)
            seconds (float): Seconds into the GPS week

        Returns:
            Epoch: The epoch in GPS time system

        Example:
            ```python
            import brahe as bh

            # Create epoch from GPS week 2200, day 3, noon
            week = 2200
            seconds = 3 * 86400 + 12 * 3600  # 3 days + 12 hours
            epc = bh.Epoch.from_gps_date(week, seconds)
            print(epc)

            # Verify GPS week extraction
            week_out, sec_out = epc.gps_date()
            print(f"GPS Week: {week_out}, Seconds: {sec_out}")
            ```
        """
        ...

    @classmethod
    def from_gps_nanoseconds(cls, gps_nanoseconds: int) -> Epoch:
        """Create an Epoch from GPS nanoseconds since the GPS epoch.

        Args:
            gps_nanoseconds (int): Nanoseconds since GPS epoch (January 6, 1980, 00:00:00 UTC)

        Returns:
            Epoch: The epoch in GPS time system

        Example:
            ```python
            import brahe as bh

            # Create epoch from GPS nanoseconds with high precision
            gps_ns = 1234567890123456789
            epc = bh.Epoch.from_gps_nanoseconds(gps_ns)
            print(f"Epoch: {epc}")
            ```
        """
        ...

    @classmethod
    def from_gps_seconds(cls, gps_seconds: float) -> Epoch:
        """Create an Epoch from GPS seconds since the GPS epoch.

        Args:
            gps_seconds (float): Seconds since GPS epoch (January 6, 1980, 00:00:00 UTC)

        Returns:
            Epoch: The epoch in GPS time system

        Example:
            ```python
            import brahe as bh

            # Create epoch from GPS seconds
            gps_seconds = 1234567890.5
            epc = bh.Epoch.from_gps_seconds(gps_seconds)
            print(f"Epoch: {epc}")
            print(f"GPS seconds: {epc.gps_seconds()}")
            ```
        """
        ...

    @classmethod
    def from_jd(cls, jd: float, time_system: TimeSystem) -> Epoch:
        """Create an Epoch from a Julian Date.

        Args:
            jd (float): Julian date
            time_system (TimeSystem): Time system

        Returns:
            Epoch: The epoch representing the Julian date

        Example:
            ```python
            import brahe as bh

            # Create epoch from Julian Date
            jd = 2460000.0
            epc = bh.Epoch.from_jd(jd, bh.TimeSystem.UTC)
            print(epc)

            # Verify round-trip conversion
            jd_out = epc.jd()
            print(f"JD: {jd_out:.10f}")
            # Output: JD: 2460000.0000000000
            ```
        """
        ...

    @classmethod
    def from_mjd(cls, mjd: float, time_system: TimeSystem) -> Epoch:
        """Create an Epoch from a Modified Julian Date.

        Args:
            mjd (float): Modified Julian date
            time_system (TimeSystem): Time system

        Returns:
            Epoch: The epoch representing the Modified Julian date

        Example:
            ```python
            import brahe as bh

            # Create epoch from Modified Julian Date
            mjd = 60000.0
            epc = bh.Epoch.from_mjd(mjd, bh.TimeSystem.UTC)
            print(epc)

            # MJD is commonly used in astronomy
            mjd_j2000 = 51544.0  # J2000 epoch
            epc_j2000 = bh.Epoch.from_mjd(mjd_j2000, bh.TimeSystem.TT)
            print(f"J2000: {epc_j2000}")
            ```
        """
        ...

    @classmethod
    def from_string(cls, datestr: str) -> Epoch:
        """Create an Epoch from an ISO 8601 formatted string.

        Args:
            datestr (str): ISO 8601 formatted date string (e.g., "2024-01-01T12:00:00.000000000 UTC")

        Returns:
            Epoch: The epoch representing the parsed date and time

        Example:
            ```python
            import brahe as bh

            # Parse ISO 8601 string with full precision
            epc = bh.Epoch.from_string("2024-01-01T12:00:00.000000000 UTC")
            print(epc)
            # Output: 2024-01-01T12:00:00.000000000 UTC

            # Parse different time systems
            epc_tai = bh.Epoch.from_string("2024-06-15T14:30:45.123456789 TAI")
            print(epc_tai.time_system)
            # Output: TimeSystem.TAI
            ```
        """
        ...

    def day(self) -> int:
        """Returns the day component of the epoch in the epoch's time system.

        Returns:
            int: The day of the month as an integer from 1 to 31
        """
        ...

    def day_of_year(self) -> float:
        """Returns the day of year as a floating-point number in the epoch's time system.

        The day of year is computed such that January 1st at midnight is 1.0,
        January 1st at noon is 1.5, January 2nd at midnight is 2.0, etc.

        Returns:
            float: The day of year as a floating-point number (1.0 to 366.999...)

        Example:
            >>> epoch = brahe.Epoch.from_datetime(2023, 4, 10, 12, 0, 0.0, 0.0, "UTC")
            >>> doy = epoch.day_of_year()
            >>> print(f"Day of year: {doy}")
            Day of year: 100.5
        """
        ...

    def day_of_year_as_time_system(self, time_system: TimeSystem) -> float:
        """Returns the day of year as a floating-point number in the specified time system.

        The day of year is computed such that January 1st at midnight is 1.0,
        January 1st at noon is 1.5, January 2nd at midnight is 2.0, etc.

        Args:
            time_system (TimeSystem): The time system to use for the calculation

        Returns:
            float: The day of year as a floating-point number (1.0 to 366.999...)

        Example:
            >>> epoch = brahe.Epoch.from_datetime(2023, 4, 10, 12, 0, 0.0, 0.0, brahe.TimeSystem.UTC)
            >>> doy_tai = epoch.day_of_year_as_time_system(brahe.TimeSystem.TAI)
            >>> print(f"Day of year in TAI: {doy_tai}")
            Day of year in TAI: 100.50042824074075
        """
        ...

    def gast(self, angle_format: AngleFormat) -> float:
        """Get the Greenwich Apparent Sidereal Time (GAST) for this epoch.

        Args:
            angle_format (AngleFormat): Format for the returned angle (radians or degrees)

        Returns:
            float: GAST angle

        Example:
            ```python
            import brahe as bh

            epc = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            gast_rad = epc.gast(bh.AngleFormat.RADIANS)
            gast_deg = epc.gast(bh.AngleFormat.DEGREES)
            print(f"GAST: {gast_rad:.6f} rad = {gast_deg:.6f} deg")
            ```
        """
        ...

    def gmst(self, angle_format: AngleFormat) -> float:
        """Get the Greenwich Mean Sidereal Time (GMST) for this epoch.

        Args:
            angle_format (AngleFormat): Format for the returned angle (radians or degrees)

        Returns:
            float: GMST angle

        Example:
            ```python
            import brahe as bh

            epc = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            gmst_rad = epc.gmst(bh.AngleFormat.RADIANS)
            gmst_deg = epc.gmst(bh.AngleFormat.DEGREES)
            print(f"GMST: {gmst_rad:.6f} rad = {gmst_deg:.6f} deg")
            ```
        """
        ...

    def gps_date(self) -> Tuple:
        """Get the GPS week number and seconds into the week.

        Returns:
            tuple: A tuple containing (week, seconds_into_week)

        Example:
            ```python
            import brahe as bh

            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.GPS)
            week, seconds = epc.gps_date()
            print(f"GPS Week: {week}, Seconds: {seconds:.3f}")
            ```
        """
        ...

    def gps_nanoseconds(self) -> float:
        """Get the nanoseconds since GPS epoch (January 6, 1980, 00:00:00 UTC).

        Returns:
            float: GPS nanoseconds

        Example:
            ```python
            import brahe as bh

            epc = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 123456789.0, bh.TimeSystem.GPS)
            gps_ns = epc.gps_nanoseconds()
            print(f"GPS nanoseconds: {gps_ns:.0f}")
            ```
        """
        ...

    def gps_seconds(self) -> float:
        """Get the seconds since GPS epoch (January 6, 1980, 00:00:00 UTC).

        Returns:
            float: GPS seconds

        Example:
            ```python
            import brahe as bh

            epc = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.GPS)
            gps_sec = epc.gps_seconds()
            print(f"GPS seconds: {gps_sec:.3f}")
            ```
        """
        ...

    def hour(self) -> int:
        """Returns the hour component of the epoch in the epoch's time system.

        Returns:
            int: The hour as an integer from 0 to 23
        """
        ...

    def isostring(self) -> str:
        """Convert the epoch to an ISO 8601 formatted string.

        Returns:
            str: ISO 8601 formatted date string with full nanosecond precision

        Example:
            ```python
            import brahe as bh

            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 30, 45.123456789, 0.0, bh.TimeSystem.UTC)
            iso = epc.isostring()
            print(iso)
            # Output: 2024-01-01T12:30:45.123456789Z
            ```
        """
        ...

    def isostring_with_decimals(self, decimals: int) -> str:
        """Convert the epoch to an ISO 8601 formatted string with specified decimal precision.

        Args:
            decimals (int): Number of decimal places for the seconds field

        Returns:
            str: ISO 8601 formatted date string

        Example:
            ```python
            import brahe as bh

            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 30, 45.123456789, 0.0, bh.TimeSystem.UTC)
            iso3 = epc.isostring_with_decimals(3)
            iso6 = epc.isostring_with_decimals(6)
            print(iso3)  # Output: 2024-01-01T12:30:45.123Z
            print(iso6)  # Output: 2024-01-01T12:30:45.123457Z
            ```
        """
        ...

    def jd(self) -> float:
        """Get the Julian Date in the epoch's time system.

        Returns:
            float: Julian date

        Example:
            ```python
            import brahe as bh

            epc = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            jd = epc.jd()
            print(f"JD: {jd:.6f}")
            # Output: JD: 2460310.500000
            ```
        """
        ...

    def jd_as_time_system(self, time_system: TimeSystem) -> float:
        """Get the Julian Date in a specified time system.

        Args:
            time_system (TimeSystem): Target time system for the conversion

        Returns:
            float: Julian date in the specified time system

        Example:
            ```python
            import brahe as bh

            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            jd_utc = epc.jd()
            jd_tai = epc.jd_as_time_system(bh.TimeSystem.TAI)
            print(f"JD UTC: {jd_utc:.10f}")
            print(f"JD TAI: {jd_tai:.10f}")
            ```
        """
        ...

    def minute(self) -> int:
        """Returns the minute component of the epoch in the epoch's time system.

        Returns:
            int: The minute as an integer from 0 to 59
        """
        ...

    def mjd(self) -> float:
        """Get the Modified Julian Date in the epoch's time system.

        Returns:
            float: Modified Julian date

        Example:
            ```python
            import brahe as bh

            epc = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            mjd = epc.mjd()
            print(f"MJD: {mjd:.6f}")
            # Output: MJD: 60310.000000
            ```
        """
        ...

    def mjd_as_time_system(self, time_system: TimeSystem) -> float:
        """Get the Modified Julian Date in a specified time system.

        Args:
            time_system (TimeSystem): Target time system for the conversion

        Returns:
            float: Modified Julian date in the specified time system

        Example:
            ```python
            import brahe as bh

            epc = bh.Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            mjd_utc = epc.mjd()
            mjd_gps = epc.mjd_as_time_system(bh.TimeSystem.GPS)
            print(f"MJD UTC: {mjd_utc:.6f}")
            print(f"MJD GPS: {mjd_gps:.6f}")
            ```
        """
        ...

    def month(self) -> int:
        """Returns the month component of the epoch in the epoch's time system.

        Returns:
            int: The month as an integer from 1 to 12
        """
        ...

    def nanosecond(self) -> float:
        """Returns the nanosecond component of the epoch in the epoch's time system.

        Returns:
            float: The nanosecond component as a floating-point number
        """
        ...

    def second(self) -> float:
        """Returns the second component of the epoch in the epoch's time system.

        Returns:
            float: The second as a floating-point number from 0.0 to 59.999...
        """
        ...

    def to_datetime(self) -> Tuple:
        """Convert the epoch to Gregorian calendar date and time in the epoch's time system.

        Returns:
            tuple: A tuple containing (year, month, day, hour, minute, second, nanosecond)

        Example:
            ```python
            import brahe as bh

            epc = bh.Epoch.from_datetime(2024, 6, 15, 14, 30, 45.5, 0.0, bh.TimeSystem.UTC)
            year, month, day, hour, minute, second, ns = epc.to_datetime()
            print(f"{year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:06.3f}")
            # Output: 2024-06-15 14:30:45.500
            ```
        """
        ...

    def to_datetime_as_time_system(self, time_system: TimeSystem) -> Tuple:
        """Convert the epoch to Gregorian calendar date and time in a specified time system.

        Args:
            time_system (TimeSystem): Target time system for the conversion

        Returns:
            tuple: A tuple containing (year, month, day, hour, minute, second, nanosecond)

        Example:
            ```python
            import brahe as bh

            # Create epoch in UTC and convert to TAI
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            year, month, day, hour, minute, second, ns = epc.to_datetime_as_time_system(bh.TimeSystem.TAI)
            print(f"TAI: {year}-{month:02d}-{day:02d} {hour:02d}:{minute:02d}:{second:06.3f}")
            # Output: TAI: 2024-01-01 12:00:37.000
            ```
        """
        ...

    def to_string_as_time_system(self, time_system: TimeSystem) -> str:
        """Convert the epoch to a string representation in a specified time system.

        Args:
            time_system (TimeSystem): Target time system for the conversion

        Returns:
            str: String representation of the epoch

        Example:
            ```python
            import brahe as bh

            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            print(epc.to_string_as_time_system(bh.TimeSystem.UTC))
            print(epc.to_string_as_time_system(bh.TimeSystem.TAI))
            # Shows same instant in different time systems
            ```
        """
        ...

    def year(self) -> int:
        """Returns the year component of the epoch in the epoch's time system.

        Returns:
            int: The year as a 4-digit integer
        """
        ...

    @property
    def time_system(self) -> TimeSystem:
        """Time system of the epoch.

        Returns:
            TimeSystem: The time system used by this epoch
        """
        ...

class EulerAngle:
    """Represents a rotation using Euler angles.

    Euler angles describe rotations as a sequence of three rotations about
    specified axes. The rotation sequence is specified by the order parameter
    (e.g., "XYZ", "ZYX").

    Args:
        order (str): Rotation sequence (e.g., "XYZ", "ZYX", "ZXZ")
        phi (float): First rotation angle in radians or degrees
        theta (float): Second rotation angle in radians or degrees
        psi (float): Third rotation angle in radians or degrees
        angle_format (AngleFormat): Units of input angles (RADIANS or DEGREES)

    Example:
        ```python
        import brahe as bh

        # Create Euler angle rotation (roll, pitch, yaw in ZYX order)
        e = bh.EulerAngle("ZYX", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
        print(f"Roll={e.phi}, Pitch={e.theta}, Yaw={e.psi}")

        # Convert to quaternion
        q = e.to_quaternion()

        # Convert to rotation matrix
        dcm = e.to_rotation_matrix()
        ```
    """

    def __init__(
        self,
        order: str,
        phi: float,
        theta: float,
        psi: float,
        angle_format: AngleFormat,
    ) -> None:
        """Initialize instance."""
        ...

    @classmethod
    def from_euler_angle(cls, e: EulerAngle, order: str) -> EulerAngle:
        """Create Euler angles from another Euler angle with different order.

        Args:
            e (EulerAngle): Source Euler angles
            order (str): Desired rotation sequence (e.g., "XYZ", "ZYX")

        Returns:
            EulerAngle: Equivalent Euler angles with new order

        Example:
            ```python
            import brahe as bh

            e1 = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
            e2 = bh.EulerAngle.from_euler_angle(e1, "ZYX")
            ```
        """
        ...

    @classmethod
    def from_euler_axis(cls, e: EulerAxis, order: str) -> EulerAngle:
        """Create Euler angles from an Euler axis representation.

        Args:
            e (EulerAxis): Euler axis representation
            order (str): Desired rotation sequence (e.g., "XYZ", "ZYX")

        Returns:
            EulerAngle: Equivalent Euler angles

        Example:
            ```python
            import brahe as bh
            import numpy as np

            axis = np.array([0.0, 0.0, 1.0])
            ea = bh.EulerAxis(axis, 1.5708, bh.AngleFormat.RADIANS)
            e = bh.EulerAngle.from_euler_axis(ea, "XYZ")
            ```
        """
        ...

    @classmethod
    def from_quaternion(cls, q: Quaternion, order: str) -> EulerAngle:
        """Create Euler angles from a quaternion.

        Args:
            q (Quaternion): Source quaternion
            order (str): Desired rotation sequence (e.g., "XYZ", "ZYX")

        Returns:
            EulerAngle: Equivalent Euler angles

        Example:
            ```python
            import brahe as bh

            q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
            e = bh.EulerAngle.from_quaternion(q, "XYZ")
            ```
        """
        ...

    @classmethod
    def from_rotation_matrix(cls, r: RotationMatrix, order: str) -> EulerAngle:
        """Create Euler angles from a rotation matrix.

        Args:
            r (RotationMatrix): Rotation matrix
            order (str): Desired rotation sequence (e.g., "XYZ", "ZYX")

        Returns:
            EulerAngle: Equivalent Euler angles

        Example:
            ```python
            import brahe as bh
            import numpy as np

            r = bh.RotationMatrix.from_array(np.eye(3))
            e = bh.EulerAngle.from_rotation_matrix(r, "XYZ")
            ```
        """
        ...

    @classmethod
    def from_vector(
        cls, v: np.ndarray, order: str, angle_format: AngleFormat
    ) -> EulerAngle:
        """Create Euler angles from a numpy array.

        Args:
            v (numpy.ndarray): 3-element array [phi, theta, psi]
            order (str): Rotation sequence (e.g., "XYZ", "ZYX")
            angle_format (AngleFormat): Units of input angles (RADIANS or DEGREES)

        Returns:
            EulerAngle: New Euler angle instance

        Example:
            ```python
            import brahe as bh
            import numpy as np

            v = np.array([0.1, 0.2, 0.3])
            euler = bh.EulerAngle.from_vector(v, "XYZ", bh.AngleFormat.RADIANS)
            ```
        """
        ...

    def to_euler_angle(self, order: str) -> EulerAngle:
        """Convert to Euler angles with different rotation sequence.

        Args:
            order (str): Desired rotation sequence (e.g., "XYZ", "ZYX")

        Returns:
            EulerAngle: Equivalent Euler angles with new order

        Example:
            ```python
            import brahe as bh

            e1 = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
            e2 = e1.to_euler_angle("ZYX")
            ```
        """
        ...

    def to_euler_axis(self) -> EulerAxis:
        """Convert to Euler axis representation.

        Returns:
            EulerAxis: Equivalent Euler axis

        Example:
            ```python
            import brahe as bh

            e = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
            ea = e.to_euler_axis()
            ```
        """
        ...

    def to_quaternion(self) -> Quaternion:
        """Convert to quaternion representation.

        Returns:
            Quaternion: Equivalent quaternion

        Example:
            ```python
            import brahe as bh

            e = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
            q = e.to_quaternion()
            ```
        """
        ...

    def to_rotation_matrix(self) -> RotationMatrix:
        """Convert to rotation matrix representation.

        Returns:
            RotationMatrix: Equivalent rotation matrix

        Example:
            ```python
            import brahe as bh

            e = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
            r = e.to_rotation_matrix()
            ```
        """
        ...

    @property
    def order(self) -> str:
        """Get the rotation sequence order.

        Returns:
            str: Rotation sequence (e.g., "XYZ", "ZYX")

        Example:
            ```python
            import brahe as bh

            e = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
            print(f"Order: {e.order}")
            ```
        """
        ...

    @property
    def phi(self) -> float:
        """Get the first rotation angle (phi) in radians.

        Returns:
            float: First rotation angle in radians

        Example:
            ```python
            import brahe as bh

            e = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
            print(f"Phi: {e.phi}")
            ```
        """
        ...

    @property
    def psi(self) -> float:
        """Get the third rotation angle (psi) in radians.

        Returns:
            float: Third rotation angle in radians

        Example:
            ```python
            import brahe as bh

            e = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
            print(f"Psi: {e.psi}")
            ```
        """
        ...

    @property
    def theta(self) -> float:
        """Get the second rotation angle (theta) in radians.

        Returns:
            float: Second rotation angle in radians

        Example:
            ```python
            import brahe as bh

            e = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
            print(f"Theta: {e.theta}")
            ```
        """
        ...

class EulerAxis:
    """Represents a rotation using Euler axis-angle representation.

    The Euler axis-angle representation describes a rotation as a single rotation
    about a specified axis by a given angle. This is also known as the axis-angle
    or rotation vector representation.

    Args:
        axis (numpy.ndarray): 3-element unit vector specifying rotation axis
        angle (float): Rotation angle in radians or degrees
        angle_format (AngleFormat): Units of input angle (RADIANS or DEGREES)

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Rotation of 90 degrees about z-axis
        axis = np.array([0.0, 0.0, 1.0])
        e = bh.EulerAxis(axis, np.pi/2, bh.AngleFormat.RADIANS)
        print(f"Angle: {e.angle} rad")

        # Convert to quaternion
        q = e.to_quaternion()
        ```
    """

    def __init__(
        self, axis: np.ndarray, angle: float, angle_format: AngleFormat
    ) -> None:
        """Initialize instance."""
        ...

    @classmethod
    def from_euler_angle(cls, e: EulerAngle) -> EulerAxis:
        """Create an Euler axis from Euler angles.

        Args:
            e (EulerAngle): Euler angle representation

        Returns:
            EulerAxis: Equivalent Euler axis

        Example:
            ```python
            import brahe as bh

            euler = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
            e = bh.EulerAxis.from_euler_angle(euler)
            ```
        """
        ...

    @classmethod
    def from_euler_axis(cls, e: EulerAxis) -> EulerAxis:
        """Create an Euler axis from another Euler axis (copy constructor).

        Args:
            e (EulerAxis): Source Euler axis

        Returns:
            EulerAxis: New Euler axis instance

        Example:
            ```python
            import brahe as bh
            import numpy as np

            axis = np.array([0.0, 0.0, 1.0])
            e1 = bh.EulerAxis(axis, 1.5708, bh.AngleFormat.RADIANS)
            e2 = bh.EulerAxis.from_euler_axis(e1)
            ```
        """
        ...

    @classmethod
    def from_quaternion(cls, q: Quaternion) -> EulerAxis:
        """Create an Euler axis from a quaternion.

        Args:
            q (Quaternion): Source quaternion

        Returns:
            EulerAxis: Equivalent Euler axis

        Example:
            ```python
            import brahe as bh

            q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
            e = bh.EulerAxis.from_quaternion(q)
            ```
        """
        ...

    @classmethod
    def from_rotation_matrix(cls, r: RotationMatrix) -> EulerAxis:
        """Create an Euler axis from a rotation matrix.

        Args:
            r (RotationMatrix): Rotation matrix

        Returns:
            EulerAxis: Equivalent Euler axis

        Example:
            ```python
            import brahe as bh
            import numpy as np

            r = bh.RotationMatrix.from_array(np.eye(3))
            e = bh.EulerAxis.from_rotation_matrix(r)
            ```
        """
        ...

    @classmethod
    def from_values(
        cls, x: float, y: float, z: float, angle: float, angle_format: AngleFormat
    ) -> EulerAxis:
        """Create an Euler axis from individual axis components and angle.

        Args:
            x (float): X component of rotation axis
            y (float): Y component of rotation axis
            z (float): Z component of rotation axis
            angle (float): Rotation angle in radians or degrees
            angle_format (AngleFormat): Units of input angle (RADIANS or DEGREES)

        Returns:
            EulerAxis: New Euler axis instance

        Example:
            ```python
            import brahe as bh

            e = bh.EulerAxis.from_values(0.0, 0.0, 1.0, 1.5708, bh.AngleFormat.RADIANS)
            ```
        """
        ...

    @classmethod
    def from_vector(
        cls, v: np.ndarray, angle_format: AngleFormat, vector_first: bool
    ) -> EulerAxis:
        """Create an Euler axis from a numpy array.

        Args:
            v (numpy.ndarray): 4-element array containing axis and angle
            angle_format (AngleFormat): Units of angle (RADIANS or DEGREES)
            vector_first (bool): If True, array is [x, y, z, angle], else [angle, x, y, z]

        Returns:
            EulerAxis: New Euler axis instance

        Example:
            ```python
            import brahe as bh
            import numpy as np

            v = np.array([0.0, 0.0, 1.0, 1.5708])
            e = bh.EulerAxis.from_vector(v, bh.AngleFormat.RADIANS, True)
            ```
        """
        ...

    def to_euler_angle(self, order: str) -> EulerAngle:
        """Convert to Euler angle representation.

        Args:
            order (str): Desired rotation sequence (e.g., "XYZ", "ZYX")

        Returns:
            EulerAngle: Equivalent Euler angles

        Example:
            ```python
            import brahe as bh
            import numpy as np

            axis = np.array([0.0, 0.0, 1.0])
            ea = bh.EulerAxis(axis, 1.5708, bh.AngleFormat.RADIANS)
            e = ea.to_euler_angle("XYZ")
            ```
        """
        ...

    def to_euler_axis(self) -> EulerAxis:
        """Convert to Euler axis representation (returns self).

        Returns:
            EulerAxis: This Euler axis

        Example:
            ```python
            import brahe as bh
            import numpy as np

            axis = np.array([0.0, 0.0, 1.0])
            e1 = bh.EulerAxis(axis, 1.5708, bh.AngleFormat.RADIANS)
            e2 = e1.to_euler_axis()
            ```
        """
        ...

    def to_quaternion(self) -> Quaternion:
        """Convert to quaternion representation.

        Returns:
            Quaternion: Equivalent quaternion

        Example:
            ```python
            import brahe as bh
            import numpy as np

            axis = np.array([0.0, 0.0, 1.0])
            e = bh.EulerAxis(axis, 1.5708, bh.AngleFormat.RADIANS)
            q = e.to_quaternion()
            ```
        """
        ...

    def to_rotation_matrix(self) -> RotationMatrix:
        """Convert to rotation matrix representation.

        Returns:
            RotationMatrix: Equivalent rotation matrix

        Example:
            ```python
            import brahe as bh
            import numpy as np

            axis = np.array([0.0, 0.0, 1.0])
            e = bh.EulerAxis(axis, 1.5708, bh.AngleFormat.RADIANS)
            r = e.to_rotation_matrix()
            ```
        """
        ...

    def to_vector(self, angle_format: AngleFormat, vector_first: bool) -> np.ndarray:
        """Convert Euler axis to a numpy array.

        Args:
            angle_format (AngleFormat): Units for output angle (RADIANS or DEGREES)
            vector_first (bool): If True, returns [x, y, z, angle], else [angle, x, y, z]

        Returns:
            numpy.ndarray: 4-element array containing axis and angle
        """
        ...

    @property
    def angle(self) -> float:
        """Get the rotation angle in radians.

        Returns:
            float: Rotation angle in radians

        Example:
            ```python
            import brahe as bh
            import numpy as np

            axis = np.array([0.0, 0.0, 1.0])
            e = bh.EulerAxis(axis, 1.5708, bh.AngleFormat.RADIANS)
            print(f"Angle: {e.angle}")
            ```
        """
        ...

    @property
    def axis(self) -> np.ndarray:
        """Get the rotation axis as a numpy array.

        Returns:
            numpy.ndarray: 3-element unit vector specifying rotation axis

        Example:
            ```python
            import brahe as bh
            import numpy as np

            axis = np.array([0.0, 0.0, 1.0])
            e = bh.EulerAxis(axis, 1.5708, bh.AngleFormat.RADIANS)
            print(f"Axis: {e.axis}")
            ```
        """
        ...

class FileEOPProvider:
    """File-based Earth Orientation Parameter provider.

    Loads EOP data from files in standard IERS formats and provides
    interpolation and extrapolation capabilities.

    Example:
        ```python
        import brahe as bh

        # Create from C04 file with interpolation
        eop = bh.FileEOPProvider.from_c04_file(
            "./eop_data/finals2000A.all.csv",
            interpolate=True,
            extrapolate="Hold"
        )

        # Create from standard file
        eop = bh.FileEOPProvider.from_standard_file(
            "./eop_data/finals.all",
            interpolate=True,
            extrapolate="Zero"
        )

        # Use default file location
        eop = bh.FileEOPProvider.from_default_c04(True, "Hold")

        # Set as global provider
        bh.set_global_eop_provider_from_file_provider(eop)

        # Get EOP data for a specific MJD
        mjd = 60310.0
        ut1_utc, pm_x, pm_y, dx, dy, lod = eop.get_eop(mjd)
        ```
    """

    def __init__(self) -> None:
        """Initialize instance."""
        ...

    @classmethod
    def from_c04_file(
        cls, filepath: str, interpolate: bool, extrapolate: str
    ) -> FileEOPProvider:
        """Create provider from a C04 format EOP file.

        Args:
            filepath (str): Path to C04 EOP file
            interpolate (bool): Enable interpolation between data points
            extrapolate (str): Extrapolation method ("Hold", "Zero", or "Error")

        Returns:
            FileEOPProvider: Provider initialized with C04 file data

        Example:
            ```python
            import brahe as bh

            eop = bh.FileEOPProvider.from_c04_file("./eop_data/finals2000A.all.csv", True, "Hold")
            bh.set_global_eop_provider_from_file_provider(eop)
            ```
        """
        ...

    @classmethod
    def from_default_c04(cls, interpolate: bool, extrapolate: str) -> FileEOPProvider:
        """Create provider from the default C04 EOP file location.

        Args:
            interpolate (bool): Enable interpolation between data points
            extrapolate (str): Extrapolation method ("Hold", "Zero", or "Error")

        Returns:
            FileEOPProvider: Provider initialized with default C04 file

        Example:
            ```python
            import brahe as bh

            eop = bh.FileEOPProvider.from_default_c04(True, "Hold")
            bh.set_global_eop_provider_from_file_provider(eop)
            ```
        """
        ...

    @classmethod
    def from_default_file(
        cls, eop_type: str, interpolate: bool, extrapolate: str
    ) -> FileEOPProvider:
        """Create provider from default EOP file location with specified type.

        Args:
            eop_type (str): EOP file type ("C04" or "StandardBulletinA")
            interpolate (bool): Enable interpolation between data points
            extrapolate (str): Extrapolation method ("Hold", "Zero", or "Error")

        Returns:
            FileEOPProvider: Provider initialized with default file of specified type

        Example:
            ```python
            import brahe as bh

            eop = bh.FileEOPProvider.from_default_file("C04", True, "Hold")
            bh.set_global_eop_provider_from_file_provider(eop)
            ```
        """
        ...

    @classmethod
    def from_default_standard(
        cls, interpolate: bool, extrapolate: str
    ) -> FileEOPProvider:
        """Create provider from the default standard IERS EOP file location.

        Args:
            interpolate (bool): Enable interpolation between data points
            extrapolate (str): Extrapolation method ("Hold", "Zero", or "Error")

        Returns:
            FileEOPProvider: Provider initialized with default standard file

        Example:
            ```python
            import brahe as bh

            eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
            bh.set_global_eop_provider_from_file_provider(eop)
            ```
        """
        ...

    @classmethod
    def from_file(
        cls, filepath: str, interpolate: bool, extrapolate: str
    ) -> FileEOPProvider:
        """Create provider from an EOP file with automatic format detection.

        Args:
            filepath (str): Path to EOP file
            interpolate (bool): Enable interpolation between data points
            extrapolate (str): Extrapolation method ("Hold", "Zero", or "Error")

        Returns:
            FileEOPProvider: Provider initialized with file data

        Example:
            ```python
            import brahe as bh

            eop = bh.FileEOPProvider.from_file("./eop_data/eop.txt", True, "Hold")
            bh.set_global_eop_provider_from_file_provider(eop)
            ```
        """
        ...

    @classmethod
    def from_standard_file(
        cls, filepath: str, interpolate: bool, extrapolate: str
    ) -> FileEOPProvider:
        """Create provider from a standard IERS format EOP file.

        Args:
            filepath (str): Path to standard IERS EOP file
            interpolate (bool): Enable interpolation between data points
            extrapolate (str): Extrapolation method ("Hold", "Zero", or "Error")

        Returns:
            FileEOPProvider: Provider initialized with standard file data

        Example:
            ```python
            import brahe as bh

            eop = bh.FileEOPProvider.from_standard_file("./eop_data/standard_eop.txt", True, "Hold")
            bh.set_global_eop_provider_from_file_provider(eop)
            ```
        """
        ...

    def eop_type(self) -> str:
        """Get the EOP data type.

        Returns:
            str: EOP type string

        Example:
            ```python
            import brahe as bh

            eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
            print(f"EOP type: {eop.eop_type()}")
            ```
        """
        ...

    def extrapolation(self) -> str:
        """Get the extrapolation method.

        Returns:
            str: Extrapolation method string

        Example:
            ```python
            import brahe as bh

            eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
            print(f"Extrapolation: {eop.extrapolation()}")
            ```
        """
        ...

    def get_dxdy(self, mjd: float) -> tuple[float, float]:
        """Get celestial pole offsets for a given MJD.

        Args:
            mjd (float): Modified Julian Date

        Returns:
            tuple[float, float]: Celestial pole offsets dx and dy in radians

        Example:
            ```python
            import brahe as bh

            eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
            dx, dy = eop.get_dxdy(58849.0)
            print(f"Celestial pole offsets: dx={dx} rad, dy={dy} rad")
            ```
        """
        ...

    def get_eop(self, mjd: float) -> tuple[float, float, float, float, float, float]:
        """Get all EOP parameters for a given MJD.

        Args:
            mjd (float): Modified Julian Date

        Returns:
            tuple[float, float, float, float, float, float]: UT1-UTC, pm_x, pm_y, dx, dy, lod

        Example:
            ```python
            import brahe as bh

            eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
            ut1_utc, pm_x, pm_y, dx, dy, lod = eop.get_eop(58849.0)
            print(f"EOP: UT1-UTC={ut1_utc}s, PM=({pm_x},{pm_y})rad")
            ```
        """
        ...

    def get_lod(self, mjd: float) -> float:
        """Get length of day offset for a given MJD.

        Args:
            mjd (float): Modified Julian Date

        Returns:
            float: Length of day offset in seconds

        Example:
            ```python
            import brahe as bh

            eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
            lod = eop.get_lod(58849.0)
            print(f"Length of day offset: {lod} seconds")
            ```
        """
        ...

    def get_pm(self, mjd: float) -> tuple[float, float]:
        """Get polar motion components for a given MJD.

        Args:
            mjd (float): Modified Julian Date

        Returns:
            tuple[float, float]: Polar motion x and y components in radians

        Example:
            ```python
            import brahe as bh

            eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
            pm_x, pm_y = eop.get_pm(58849.0)
            print(f"Polar motion: x={pm_x} rad, y={pm_y} rad")
            ```
        """
        ...

    def get_ut1_utc(self, mjd: float) -> float:
        """Get UT1-UTC time difference for a given MJD.

        Args:
            mjd (float): Modified Julian Date

        Returns:
            float: UT1-UTC time difference in seconds

        Example:
            ```python
            import brahe as bh

            eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
            ut1_utc = eop.get_ut1_utc(58849.0)
            print(f"UT1-UTC: {ut1_utc} seconds")
            ```
        """
        ...

    def interpolation(self) -> bool:
        """Check if interpolation is enabled.

        Returns:
            bool: True if interpolation is enabled

        Example:
            ```python
            import brahe as bh

            eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
            print(f"interpolation: {eop.interpolation()}")
            ```
        """
        ...

    def is_initialized(self) -> bool:
        """Check if the provider is initialized.

        Returns:
            bool: True if initialized

        Example:
            ```python
            import brahe as bh

            eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
            print(f"is_initialized: {eop.is_initialized()}")
            ```
        """
        ...

    def len(self) -> int:
        """Get the number of EOP data points.

        Returns:
            int: Number of EOP data points

        Example:
            ```python
            import brahe as bh

            eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
            print(f"EOP data points: {eop.len()}")
            ```
        """
        ...

    def mjd_last_dxdy(self) -> float:
        """Get the last Modified Julian Date with dx/dy data.

        Returns:
            float: Last MJD with dx/dy data

        Example:
            ```python
            import brahe as bh

            eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
            print(f"mjd_last_dxdy: {eop.mjd_last_dxdy()}")
            ```
        """
        ...

    def mjd_last_lod(self) -> float:
        """Get the last Modified Julian Date with LOD data.

        Returns:
            float: Last MJD with LOD data

        Example:
            ```python
            import brahe as bh

            eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
            print(f"Last MJD with LOD: {eop.mjd_last_lod()}")
            ```
        """
        ...

    def mjd_max(self) -> float:
        """Get the maximum Modified Julian Date in the dataset.

        Returns:
            float: Maximum MJD

        Example:
            ```python
            import brahe as bh

            eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
            print(f"mjd_max: {eop.mjd_max()}")
            ```
        """
        ...

    def mjd_min(self) -> float:
        """Get the minimum Modified Julian Date in the dataset.

        Returns:
            float: Minimum MJD

        Example:
            ```python
            import brahe as bh

            eop = bh.FileEOPProvider.from_default_standard(True, "Hold")
            print(f"Minimum MJD: {eop.mjd_min()}")
            ```
        """
        ...

class InterpolationMethod:
    """Python bindings for the new trajectory architecture
    Interpolation method for trajectory state estimation.

    Specifies the algorithm used to estimate states at epochs between
    discrete trajectory points.
    """

    def __init__(self) -> None:
        """Initialize instance."""
        ...

    @property
    def LINEAR(self) -> Any:
        """Python bindings for the new trajectory architecture
        Interpolation method for trajectory state estimation.

        Specifies the algorithm used to estimate states at epochs between
        discrete trajectory points.
        """
        ...

class KeplerianPropagator:
    """Python wrapper for KeplerianPropagator (new architecture)
    Keplerian orbit propagator using two-body dynamics.

    The Keplerian propagator implements ideal two-body orbital mechanics without
    perturbations. It's fast and accurate for short time spans but doesn't account
    for real-world effects like drag, J2, solar radiation pressure, etc.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Initial epoch and orbital elements
        epc0 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        oe = np.array([7000000.0, 0.001, 0.9, 0.0, 0.0, 0.0])  # a, e, i, RAAN, omega, M

        # Create propagator from Keplerian elements
        prop = bh.KeplerianPropagator.from_keplerian(
            epc0, oe, bh.AngleFormat.RADIANS, step_size=60.0
        )

        # Propagate forward one orbit
        period = bh.orbital_period(oe[0])
        epc_future = epc0 + period
        state = prop.state(epc_future)
        print(f"State after one orbit: {state}")

        # Create from Cartesian state
        x_cart = np.array([7000000.0, 0.0, 0.0, 0.0, 7546.0, 0.0])
        prop2 = bh.KeplerianPropagator(
            epc0, x_cart, bh.OrbitFrame.ECI,
            bh.OrbitRepresentation.CARTESIAN,
            bh.AngleFormat.RADIANS, 60.0
        )
        ```
    """

    def __init__(self) -> None:
        """Initialize instance."""
        ...

    @classmethod
    def from_ecef(
        cls, epoch: Epoch, state: np.ndarray, step_size: float
    ) -> KeplerianPropagator:
        """Create a new Keplerian propagator from Cartesian state in ECEF frame.

        Args:
            epoch (Epoch): Initial epoch.
            state (numpy.ndarray): 6-element Cartesian state [x, y, z, vx, vy, vz] in ECEF frame.
            step_size (float): Step size in seconds for propagation.

        Returns:
            KeplerianPropagator: New propagator instance.
        """
        ...

    @classmethod
    def from_eci(
        cls, epoch: Epoch, state: np.ndarray, step_size: float
    ) -> KeplerianPropagator:
        """Create a new Keplerian propagator from Cartesian state in ECI frame.

        Args:
            epoch (Epoch): Initial epoch.
            state (numpy.ndarray): 6-element Cartesian state [x, y, z, vx, vy, vz] in ECI frame.
            step_size (float): Step size in seconds for propagation.

        Returns:
            KeplerianPropagator: New propagator instance.
        """
        ...

    @classmethod
    def from_keplerian(
        cls,
        epoch: Epoch,
        elements: np.ndarray,
        angle_format: AngleFormat,
        step_size: float,
    ) -> KeplerianPropagator:
        """Create a new Keplerian propagator from Keplerian orbital elements.

        Args:
            epoch (Epoch): Initial epoch.
            elements (numpy.ndarray): 6-element Keplerian elements [a, e, i, raan, argp, mean_anomaly].
            angle_format (AngleFormat): Angle format (Degrees or Radians).
            step_size (float): Step size in seconds for propagation.

        Returns:
            KeplerianPropagator: New propagator instance.
        """
        ...

    def current_state(self) -> np.ndarray:
        """Get current state vector.

        Returns:
            numpy.ndarray: Current state vector.
        """
        ...

    def generate_uuid(self) -> Any:
        """Generate a new UUID and set it in-place (mutating)."""
        ...

    def initial_state(self) -> np.ndarray:
        """Get initial state.

        Returns:
            numpy.ndarray: Initial state vector.
        """
        ...

    def propagate_steps(self, num_steps: int) -> Any:
        """Propagate forward by specified number of steps.

        Args:
            num_steps (int): Number of steps to take.

        Example:
            ```python
            import brahe as bh
            import numpy as np

            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
            state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
            prop = bh.KeplerianPropagator(epc, state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, 60.0)
            prop.propagate_steps(10)  # Take 10 steps (600 seconds total)
            print(f"Advanced to: {prop.current_epoch}")
            ```
        """
        ...

    def propagate_to(self, target_epoch: Epoch) -> Any:
        """Propagate to a specific target epoch.

        Args:
            target_epoch (Epoch): The epoch to propagate to.

        Example:
            ```python
            import brahe as bh
            import numpy as np

            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
            state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
            prop = bh.KeplerianPropagator(epc, state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, 60.0)
            target = epc + 3600.0  # Propagate to 1 hour ahead
            prop.propagate_to(target)
            print(f"Propagated to: {prop.current_epoch}")
            ```
        """
        ...

    def reset(self) -> Any:
        """Reset propagator to initial conditions.

        Example:
            ```python
            import brahe as bh
            import numpy as np

            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
            state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
            prop = bh.KeplerianPropagator(epc, state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, 60.0)
            prop.propagate_steps(10)
            prop.reset()  # Return to initial epoch and state
            print(f"Reset to: {prop.current_epoch}")
            ```
        """
        ...

    def set_eviction_policy_max_age(self, max_age: float) -> Any:
        """Set eviction policy to keep states within maximum age.

        Args:
            max_age (float): Maximum age in seconds.

        Example:
            ```python
            import brahe as bh
            import numpy as np

            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
            state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
            prop = bh.KeplerianPropagator(epc, state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, 60.0)
            prop.set_eviction_policy_max_age(3600.0)  # Keep only states within 1 hour
            prop.propagate_to(epc + 7200.0)  # Propagate 2 hours
            print(f"Trajectory length: {prop.trajectory.len()}")
            ```
        """
        ...

    def set_eviction_policy_max_size(self, max_size: int) -> Any:
        """Set eviction policy to keep maximum number of states.

        Args:
            max_size (int): Maximum number of states to retain.

        Example:
            ```python
            import brahe as bh
            import numpy as np

            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
            state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
            prop = bh.KeplerianPropagator(epc, state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, 60.0)
            prop.set_eviction_policy_max_size(100)  # Keep only 100 most recent states
            prop.propagate_steps(200)
            print(f"Trajectory length: {prop.trajectory.len()}")
            ```
        """
        ...

    def set_id(self, id: int or None) -> Any:
        """Set the numeric ID in-place (mutating).

        Args:
            id (int or None): Numeric ID to assign, or None to clear.
        """
        ...

    def set_identity(
        self, name: str or None, uuid_str: str or None, id: int or None
    ) -> Any:
        """Set all identity fields in-place (mutating).

        Args:
            name (str or None): Optional name to assign.
            uuid_str (str or None): Optional UUID string to assign.
            id (int or None): Optional numeric ID to assign.
        """
        ...

    def set_initial_conditions(
        self,
        epoch: Epoch,
        state: np.ndarray,
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: AngleFormat,
    ) -> Any:
        """Set initial conditions.

        Args:
            epoch (Epoch): Initial epoch.
            state (numpy.ndarray): Initial state vector.
            frame (OrbitFrame): Reference frame.
            representation (OrbitRepresentation): State representation.
            angle_format (AngleFormat): Angle format.

        Example:
            ```python
            import brahe as bh
            import numpy as np

            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
            state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
            prop = bh.KeplerianPropagator(epc, state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, 60.0)

            # Change initial conditions to a different orbit
            new_oe = np.array([bh.R_EARTH + 800e3, 0.02, 1.2, 0.5, 0.3, 0.0])
            new_state = bh.state_osculating_to_cartesian(new_oe, bh.AngleFormat.RADIANS)
            new_epc = bh.Epoch.from_datetime(2024, 1, 2, 0, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            prop.set_initial_conditions(new_epc, new_state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, bh.AngleFormat.RADIANS)
            print(f"New initial epoch: {prop.initial_epoch}")
            ```
        """
        ...

    def set_name(self, name: str or None) -> Any:
        """Set the name in-place (mutating).

        Args:
            name (str or None): Name to assign, or None to clear.
        """
        ...

    def state(self, epoch: Epoch) -> np.ndarray:
        """Compute state at a specific epoch.

        Args:
            epoch (Epoch): Target epoch for state computation.

        Returns:
            numpy.ndarray: State vector in the propagator's native format.
        """
        ...

    def state_as_osculating_elements(
        self, epoch: Epoch, angle_format: AngleFormat
    ) -> np.ndarray:
        """Compute state as osculating elements at a specific epoch.

        Args:
            epoch (Epoch): Target epoch for state computation.
            angle_format (AngleFormat): If AngleFormat.DEGREES, angular elements are returned in degrees, otherwise in radians.

        Returns:
            numpy.ndarray: Osculating elements [a, e, i, raan, argp, mean_anomaly].
        """
        ...

    def state_ecef(self, epoch: Epoch) -> np.ndarray:
        """Compute state at a specific epoch in ECEF coordinates.

        Args:
            epoch (Epoch): Target epoch for state computation.

        Returns:
            numpy.ndarray: State vector [x, y, z, vx, vy, vz] in ECEF frame.
        """
        ...

    def state_eci(self, epoch: Epoch) -> np.ndarray:
        """Compute state at a specific epoch in ECI coordinates.

        Args:
            epoch (Epoch): Target epoch for state computation.

        Returns:
            numpy.ndarray: State vector [x, y, z, vx, vy, vz] in ECI frame.
        """
        ...

    def states(self, epochs: list[Epoch]) -> List:
        """Compute states at multiple epochs.

        Args:
            epochs (list[Epoch]): List of epochs for state computation.

        Returns:
            list[numpy.ndarray]: List of state vectors in the propagator's native format.
        """
        ...

    def states_as_osculating_elements(
        self, epochs: list[Epoch], angle_format: AngleFormat
    ) -> List:
        """Compute states as osculating elements at multiple epochs.

        Args:
            epochs (list[Epoch]): List of epochs for state computation.
            angle_format (AngleFormat): If AngleFormat.DEGREES, angular elements are returned in degrees, otherwise in radians.

        Returns:
            list[numpy.ndarray]: List of osculating element vectors.
        """
        ...

    def states_ecef(self, epochs: list[Epoch]) -> List:
        """Compute states at multiple epochs in ECEF coordinates.

        Args:
            epochs (list[Epoch]): List of epochs for state computation.

        Returns:
            list[numpy.ndarray]: List of ECEF state vectors.
        """
        ...

    def states_eci(self, epochs: list[Epoch]) -> List:
        """Compute states at multiple epochs in ECI coordinates.

        Args:
            epochs (list[Epoch]): List of epochs for state computation.

        Returns:
            list[numpy.ndarray]: List of ECI state vectors.
        """
        ...

    def step(self) -> Any:
        """Step forward by the default step size.

        Example:
            ```python
            import brahe as bh
            import numpy as np

            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
            state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
            prop = bh.KeplerianPropagator(epc, state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, 60.0)
            prop.step()  # Advance by default step_size (60 seconds)
            print(f"Advanced to: {prop.current_epoch}")
            ```
        """
        ...

    def step_by(self, step_size: float) -> Any:
        """Step forward by a specified time duration.

        Args:
            step_size (float): Time step in seconds.

        Example:
            ```python
            import brahe as bh
            import numpy as np

            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
            state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
            prop = bh.KeplerianPropagator(epc, state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, 60.0)
            prop.step_by(120.0)  # Advance by 120 seconds
            print(f"Advanced to: {prop.current_epoch}")
            ```
        """
        ...

    def step_past(self, target_epoch: Epoch) -> Any:
        """Step past a specified target epoch.

        Args:
            target_epoch (Epoch): The epoch to step past.

        Example:
            ```python
            import brahe as bh
            import numpy as np

            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
            state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
            prop = bh.KeplerianPropagator(epc, state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, 60.0)
            target = epc + 300.0  # Target 5 minutes ahead
            prop.step_past(target)
            print(f"Advanced to: {prop.current_epoch}")
            ```
        """
        ...

    def with_id(self, id: int) -> KeplerianPropagator:
        """Set the numeric ID and return self (consuming constructor pattern).

        Args:
            id (int): Numeric ID to assign to this propagator.

        Returns:
            KeplerianPropagator: Self with ID set.
        """
        ...

    def with_identity(
        self, name: str or None, uuid_str: str or None, id: int or None
    ) -> KeplerianPropagator:
        """Set all identity fields at once and return self (consuming constructor pattern).

        Args:
            name (str or None): Optional name to assign.
            uuid_str (str or None): Optional UUID string to assign.
            id (int or None): Optional numeric ID to assign.

        Returns:
            KeplerianPropagator: Self with identity set.
        """
        ...

    def with_name(self, name: str) -> KeplerianPropagator:
        """Set the name and return self (consuming constructor pattern).

        Args:
            name (str): Name to assign to this propagator.

        Returns:
            KeplerianPropagator: Self with name set.

        Example:
            ```python
            import brahe as bh
            import numpy as np

            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            oe = np.array([7000e3, 0.001, 0.9, 0.0, 0.0, 0.0])
            prop = bh.KeplerianPropagator.from_keplerian(
                epc, oe, bh.AngleFormat.RADIANS, 60.0
            ).with_name("My Orbit")
            print(f"Name: {prop.name}")
            ```
        """
        ...

    def with_new_uuid(self) -> KeplerianPropagator:
        """Generate a new UUID, set it, and return self (consuming constructor pattern).

        Returns:
            KeplerianPropagator: Self with new UUID set.
        """
        ...

    def with_uuid(self, uuid_str: str) -> KeplerianPropagator:
        """Set the UUID and return self (consuming constructor pattern).

        Args:
            uuid_str (str): UUID string to assign to this propagator.

        Returns:
            KeplerianPropagator: Self with UUID set.
        """
        ...

    @property
    def current_epoch(self) -> Epoch:
        """Get current epoch.

        Returns:
            Epoch: Current propagator epoch.
        """
        ...

    @property
    def id(self) -> int:
        """Get the current numeric ID.

        Returns:
            int or None: The numeric ID, or None if not set.
        """
        ...

    @property
    def initial_epoch(self) -> Epoch:
        """Get initial epoch.

        Returns:
            Epoch: Initial propagator epoch.
        """
        ...

    @property
    def name(self) -> str:
        """Get the current name.

        Returns:
            str or None: The name, or None if not set.
        """
        ...

    @property
    def step_size(self) -> float:
        """Get step size in seconds.

        Returns:
            float: Step size in seconds.
        """
        ...

    @property
    def trajectory(self) -> OrbitTrajectory:
        """Get accumulated trajectory.

        Returns:
            OrbitalTrajectory: The accumulated trajectory.

        Example:
            ```python
            import brahe as bh
            import numpy as np

            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            oe = np.array([bh.R_EARTH + 500e3, 0.01, 0.9, 1.0, 0.5, 0.0])
            state = bh.state_osculating_to_cartesian(oe, bh.AngleFormat.RADIANS)
            prop = bh.KeplerianPropagator(epc, state, bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None, 60.0)
            prop.propagate_steps(10)
            traj = prop.trajectory
            print(f"Trajectory contains {traj.len()} states")
            ```
        """
        ...

    @property
    def uuid(self) -> str:
        """Get the current UUID.

        Returns:
            str or None: The UUID as a string, or None if not set.
        """
        ...

class LocalTimeConstraint:
    """Local solar time constraint.

    Constrains access based on the local solar time at the ground location.
    Useful for sun-synchronous orbits or daytime-only imaging.

    Time windows are specified in military time format (HHMM).
    Wrap-around windows (e.g., 2200-0200) are supported.

    Args:
        time_windows (list[tuple[int, int]]): List of (start_military, end_military) tuples (0-2400)

    Raises:
        ValueError: If any military time is invalid (>2400 or minutes >=60)

    Example:
        ```python
        import brahe as bh

        # Only daytime (6 AM to 6 PM local time)
        constraint = bh.LocalTimeConstraint(time_windows=[(600, 1800)])

        # Two windows: morning (6-9 AM) and evening (4-7 PM)
        constraint = bh.LocalTimeConstraint(time_windows=[(600, 900), (1600, 1900)])

        # Overnight window (10 PM to 2 AM) - handles wrap-around
        constraint = bh.LocalTimeConstraint(time_windows=[(2200, 200)])
        ```
    """

    def __init__(self, time_windows: list[tuple[int, int]]) -> None:
        """Initialize instance."""
        ...

    @classmethod
    def from_hours(cls, time_windows: list[tuple[float, float]]) -> LocalTimeConstraint:
        """Create from decimal hour windows instead of military time.

        Args:
            time_windows (list[tuple[float, float]]): List of (start_hour, end_hour) tuples [0, 24)

        Returns:
            LocalTimeConstraint: The constraint instance

        Example:
            ```python
            import brahe as bh

            # Only daytime (6 AM to 6 PM local time)
            constraint = bh.LocalTimeConstraint.from_hours([(6.0, 18.0)])

            # Overnight window (10 PM to 2 AM)
            constraint = bh.LocalTimeConstraint.from_hours([(22.0, 2.0)])
            ```
        """
        ...

    def evaluate(
        self, epoch: Epoch, sat_state_ecef: np.ndarray, location_ecef: np.ndarray
    ) -> bool:
        """Evaluate whether the constraint is satisfied.

        Args:
            epoch (Epoch): Time of evaluation
            sat_state_ecef (ndarray): Satellite state in ECEF [x, y, z, vx, vy, vz] (meters, m/s)
            location_ecef (ndarray): Ground location in ECEF [x, y, z] (meters)

        Returns:
            bool: True if constraint is satisfied, False otherwise
        """
        ...

    def name(self) -> Any:
        """Get the constraint name"""
        ...

class LookDirection:
    """Look direction of a satellite relative to its velocity vector.

    Indicates whether a satellite is looking to the left (counterclockwise from velocity),
    right (clockwise from velocity), or either direction.

    This is commonly used for imaging satellites with side-looking sensors or SAR systems
    that have a preferred look direction.

    Attributes:
        LEFT: Left-looking (counterclockwise from velocity vector)
        RIGHT: Right-looking (clockwise from velocity vector)
        EITHER: Either left or right is acceptable

    Example:
        ```python
        import brahe as bh

        # Create a constraint for right-looking only satellites
        constraint = bh.LookDirectionConstraint(allowed=bh.LookDirection.RIGHT)

        # Create a constraint accepting either direction
        constraint = bh.LookDirectionConstraint(allowed=bh.LookDirection.EITHER)

        # Compare look directions
        assert bh.LookDirection.LEFT != bh.LookDirection.RIGHT
        assert bh.LookDirection.LEFT == bh.LookDirection.LEFT
        ```
    """

    def __init__(self) -> None:
        """Initialize instance."""
        ...

    @property
    def EITHER(self) -> Any:
        """Look direction of a satellite relative to its velocity vector.

        Indicates whether a satellite is looking to the left (counterclockwise from velocity),
        right (clockwise from velocity), or either direction.

        This is commonly used for imaging satellites with side-looking sensors or SAR systems
        that have a preferred look direction.

        Attributes:
            LEFT: Left-looking (counterclockwise from velocity vector)
            RIGHT: Right-looking (clockwise from velocity vector)
            EITHER: Either left or right is acceptable

        Example:
            ```python
            import brahe as bh

            # Create a constraint for right-looking only satellites
            constraint = bh.LookDirectionConstraint(allowed=bh.LookDirection.RIGHT)

            # Create a constraint accepting either direction
            constraint = bh.LookDirectionConstraint(allowed=bh.LookDirection.EITHER)

            # Compare look directions
            assert bh.LookDirection.LEFT != bh.LookDirection.RIGHT
            assert bh.LookDirection.LEFT == bh.LookDirection.LEFT
            ```
        """
        ...

    @property
    def LEFT(self) -> Any:
        """Look direction of a satellite relative to its velocity vector.

        Indicates whether a satellite is looking to the left (counterclockwise from velocity),
        right (clockwise from velocity), or either direction.

        This is commonly used for imaging satellites with side-looking sensors or SAR systems
        that have a preferred look direction.

        Attributes:
            LEFT: Left-looking (counterclockwise from velocity vector)
            RIGHT: Right-looking (clockwise from velocity vector)
            EITHER: Either left or right is acceptable

        Example:
            ```python
            import brahe as bh

            # Create a constraint for right-looking only satellites
            constraint = bh.LookDirectionConstraint(allowed=bh.LookDirection.RIGHT)

            # Create a constraint accepting either direction
            constraint = bh.LookDirectionConstraint(allowed=bh.LookDirection.EITHER)

            # Compare look directions
            assert bh.LookDirection.LEFT != bh.LookDirection.RIGHT
            assert bh.LookDirection.LEFT == bh.LookDirection.LEFT
            ```
        """
        ...

    @property
    def RIGHT(self) -> Any:
        """Look direction of a satellite relative to its velocity vector.

        Indicates whether a satellite is looking to the left (counterclockwise from velocity),
        right (clockwise from velocity), or either direction.

        This is commonly used for imaging satellites with side-looking sensors or SAR systems
        that have a preferred look direction.

        Attributes:
            LEFT: Left-looking (counterclockwise from velocity vector)
            RIGHT: Right-looking (clockwise from velocity vector)
            EITHER: Either left or right is acceptable

        Example:
            ```python
            import brahe as bh

            # Create a constraint for right-looking only satellites
            constraint = bh.LookDirectionConstraint(allowed=bh.LookDirection.RIGHT)

            # Create a constraint accepting either direction
            constraint = bh.LookDirectionConstraint(allowed=bh.LookDirection.EITHER)

            # Compare look directions
            assert bh.LookDirection.LEFT != bh.LookDirection.RIGHT
            assert bh.LookDirection.LEFT == bh.LookDirection.LEFT
            ```
        """
        ...

class LookDirectionConstraint:
    """Look direction constraint (left/right relative to velocity).

    Constrains access based on the look direction of the satellite relative
    to its velocity vector.

    Args:
        allowed (LookDirection): Required look direction (LEFT, RIGHT, or EITHER)

    Example:
        ```python
        import brahe as bh

        # Satellite can only look right
        constraint = bh.LookDirectionConstraint(allowed=bh.LookDirection.RIGHT)

        # Either direction is acceptable
        constraint = bh.LookDirectionConstraint(allowed=bh.LookDirection.EITHER)
        ```
    """

    def __init__(self, allowed: LookDirection) -> None:
        """Initialize instance."""
        ...

    def evaluate(
        self, epoch: Epoch, sat_state_ecef: np.ndarray, location_ecef: np.ndarray
    ) -> bool:
        """Evaluate whether the constraint is satisfied.

        Args:
            epoch (Epoch): Time of evaluation
            sat_state_ecef (ndarray): Satellite state in ECEF [x, y, z, vx, vy, vz] (meters, m/s)
            location_ecef (ndarray): Ground location in ECEF [x, y, z] (meters)

        Returns:
            bool: True if constraint is satisfied, False otherwise
        """
        ...

    def name(self) -> Any:
        """Get the constraint name"""
        ...

class OffNadirConstraint:
    """Off-nadir angle constraint for satellite imaging.

    Constrains access based on the off-nadir angle (angle between the satellite's
    nadir vector and the line-of-sight to the location).

    Args:
        min_off_nadir_deg (float | None): Minimum off-nadir angle in degrees, or None for no minimum
        max_off_nadir_deg (float | None): Maximum off-nadir angle in degrees, or None for no maximum

    Raises:
        ValueError: If both min and max are None, or if any angle is negative

    Example:
        ```python
        import brahe as bh

        # Imaging satellite with 45° maximum slew angle
        constraint = bh.OffNadirConstraint(min_off_nadir_deg=None, max_off_nadir_deg=45.0)

        # Minimum 10° to avoid nadir (e.g., for oblique imaging)
        constraint = bh.OffNadirConstraint(min_off_nadir_deg=10.0, max_off_nadir_deg=45.0)
        ```
    """

    def __init__(
        self, min_off_nadir_deg: float | None, max_off_nadir_deg: float | None
    ) -> None:
        """Initialize instance."""
        ...

    def evaluate(
        self, epoch: Epoch, sat_state_ecef: np.ndarray, location_ecef: np.ndarray
    ) -> bool:
        """Evaluate whether the constraint is satisfied.

        Args:
            epoch (Epoch): Time of evaluation
            sat_state_ecef (ndarray): Satellite state in ECEF [x, y, z, vx, vy, vz] (meters, m/s)
            location_ecef (ndarray): Ground location in ECEF [x, y, z] (meters)

        Returns:
            bool: True if constraint is satisfied, False otherwise
        """
        ...

    def name(self) -> Any:
        """Get the constraint name"""
        ...

class OrbitFrame:
    """Reference frame for orbital trajectory representation.

    Specifies the coordinate reference frame for position and velocity states.
    """

    def __init__(self) -> None:
        """Initialize instance."""
        ...

    def name(self) -> str:
        """Get the full name of the reference frame.

        Returns:
            str: Human-readable frame name
        """
        ...

    @property
    def ECEF(self) -> Any:
        """Reference frame for orbital trajectory representation.

        Specifies the coordinate reference frame for position and velocity states.
        """
        ...

    @property
    def ECI(self) -> Any:
        """Reference frame for orbital trajectory representation.

        Specifies the coordinate reference frame for position and velocity states.
        """
        ...

class OrbitRepresentation:
    """Orbital state representation format.

    Specifies how orbital states are represented in the trajectory.
    """

    def __init__(self) -> None:
        """Initialize instance."""
        ...

    @property
    def CARTESIAN(self) -> Any:
        """Orbital state representation format.

        Specifies how orbital states are represented in the trajectory.
        """
        ...

    @property
    def KEPLERIAN(self) -> Any:
        """Orbital state representation format.

        Specifies how orbital states are represented in the trajectory.
        """
        ...

class OrbitTrajectory:
    """Orbital trajectory with frame and representation awareness.

    Stores a sequence of orbital states at specific epochs with support for
    interpolation, frame conversions, and representation transformations.
    """

    def __init__(self) -> None:
        """Initialize instance."""
        ...

    @classmethod
    def default(cls) -> OrbitTrajectory:
        """Create a default empty orbital trajectory (ECI Cartesian).

        Returns:
            OrbitTrajectory: New trajectory with ECI frame and Cartesian representation
        """
        ...

    @classmethod
    def from_orbital_data(
        cls,
        epochs: list[Epoch],
        states: np.ndarray,
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: AngleFormat or None,
    ) -> OrbitTrajectory:
        """Create orbital trajectory from existing data.

        Args:
            epochs (list[Epoch]): List of time epochs for each state
            states (numpy.ndarray): Flattened 1D array of 6-element state vectors
                with total length N*6 where N is the number of epochs
            frame (OrbitFrame): Reference frame for the states
            representation (OrbitRepresentation): State representation format
            angle_format (AngleFormat or None): Angle format for Keplerian states,
                must be None for Cartesian representation

        Returns:
            OrbitTrajectory: New trajectory instance populated with data
        """
        ...

    def add(self, epoch: Epoch, state: np.ndarray) -> Any:
        """Add a state to the trajectory.

        Args:
            epoch (Epoch): Time of the state
            state (numpy.ndarray): 6-element state vector

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            ```
        """
        ...

    def clear(self) -> Any:
        """Clear all states from the trajectory.

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            traj.clear()
            ```
        """
        ...

    def dimension(self) -> int:
        """Get trajectory dimension (always 6 for orbital trajectories).

        Returns:
            int: Dimension of the trajectory (always 6)

        Example:
            ```python
            import brahe as bh

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            print(f"Dimension: {traj.dimension()}")
            ```
        """
        ...

    def end_epoch(self) -> Epoch:
        """Get end epoch of trajectory.

        Returns:
            Epoch or None: Last epoch if trajectory is not empty, None otherwise

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            print(f"End epoch: {traj.end_epoch()}")
            ```
        """
        ...

    def epoch_at_idx(self, index: int) -> Epoch:
        """Get the epoch at a specific index.

        Args:
            index (int): Index of the epoch to retrieve

        Returns:
            Epoch: Epoch at the specified index

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.UTC)
            state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
            traj.add(epc, state)

            # Get epoch at index
            epoch_0 = traj.epoch_at_idx(0)
            ```
        """
        ...

    def epochs(self) -> np.ndarray:
        """Get all epochs as a numpy array.

        Returns:
            numpy.ndarray: 1D array of Julian dates for all epochs

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            traj.add(epc + 60.0, state)
            epochs_array = traj.epochs()
            ```
        """
        ...

    def first(self) -> Tuple:
        """Get the first (epoch, state) tuple in the trajectory, if any exists.

        Returns:
            tuple or None: Tuple of (Epoch, numpy.ndarray) for first state, or None if empty

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            first_epc, first_state = traj.first()
            ```
        """
        ...

    def get(self, index: int) -> Tuple:
        """Get both epoch and state at a specific index.

        Args:
            index (int): Index to retrieve

        Returns:
            tuple: Tuple of (Epoch, numpy.ndarray) for epoch and state at the index

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            ret_epc, ret_state = traj.get(0)
            ```
        """
        ...

    def get_eviction_policy(self) -> str:
        """Get current eviction policy.

        Returns:
            str: String representation of eviction policy

        Example:
            ```python
            import brahe as bh

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            policy = traj.get_eviction_policy()
            ```
        """
        ...

    def get_interpolation_method(self) -> InterpolationMethod:
        """Get the current interpolation method.

        Returns:
            InterpolationMethod: Current interpolation method

        Example:
            ```python
            import brahe as bh

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            method = traj.get_interpolation_method()
            ```
        """
        ...

    def index_after_epoch(self, epoch: Epoch) -> int:
        """Get the index of the state at or after the given epoch.

        Args:
            epoch (Epoch): Target epoch

        Returns:
            int: Index of the state at or after the target epoch

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc1, state)
            epc2 = bh.Epoch.from_datetime(2024, 1, 1, 11, 59, 0.0, 0.0, bh.TimeSystem.UTC)
            index = traj.index_after_epoch(epc2)
            ```
        """
        ...

    def index_before_epoch(self, epoch: Epoch) -> int:
        """Get the index of the state at or before the given epoch.

        Args:
            epoch (Epoch): Target epoch

        Returns:
            int: Index of the state at or before the target epoch

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc1, state)
            epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
            index = traj.index_before_epoch(epc2)
            ```
        """
        ...

    def interpolate(self, epoch: Epoch) -> np.ndarray:
        """Interpolate state at a given epoch using the configured interpolation method.

        Args:
            epoch (Epoch): Target epoch

        Returns:
            numpy.ndarray: Interpolated state vector

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state1 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc1, state1)
            epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 2, 0.0, 0.0, bh.TimeSystem.UTC)
            state2 = np.array([bh.R_EARTH + 510e3, 0.0, 0.0, 0.0, 7650.0, 0.0])
            traj.add(epc2, state2)
            epc_mid = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
            state_interp = traj.interpolate(epc_mid)
            ```
        """
        ...

    def interpolate_linear(self, epoch: Epoch) -> np.ndarray:
        """Interpolate state at a given epoch using linear interpolation.

        Args:
            epoch (Epoch): Target epoch

        Returns:
            numpy.ndarray: Linearly interpolated state vector

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state1 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc1, state1)
            epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 2, 0.0, 0.0, bh.TimeSystem.UTC)
            state2 = np.array([bh.R_EARTH + 510e3, 0.0, 0.0, 0.0, 7650.0, 0.0])
            traj.add(epc2, state2)
            epc_mid = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
            state_interp = traj.interpolate_linear(epc_mid)
            ```
        """
        ...

    def is_empty(self) -> bool:
        """Check if trajectory is empty.

        Returns:
            bool: True if trajectory contains no states, False otherwise

        Example:
            ```python
            import brahe as bh

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            print(f"Is empty: {traj.is_empty()}")
            ```
        """
        ...

    def last(self) -> Tuple:
        """Get the last (epoch, state) tuple in the trajectory, if any exists.

        Returns:
            tuple or None: Tuple of (Epoch, numpy.ndarray) for last state, or None if empty

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            last_epc, last_state = traj.last()
            ```
        """
        ...

    def len(self) -> int:
        """Get the number of states in the trajectory (alias for length).

        Returns:
            int: Number of states in the trajectory

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.DTrajectory(6)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            print(f"Number of states: {traj.len()}")
            ```
        """
        ...

    def nearest_state(self, epoch: Epoch) -> Tuple:
        """Get the nearest state to a given epoch.

        Args:
            epoch (Epoch): Target epoch

        Returns:
            tuple: Tuple of (Epoch, numpy.ndarray) containing the nearest state

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc1, state)
            epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 30.0, 0.0, bh.TimeSystem.UTC)
            nearest_epc, nearest_state = traj.nearest_state(epc2)
            ```
        """
        ...

    def remove(self, index: int) -> Tuple:
        """Remove a state at a specific index.

        Args:
            index (int): Index of the state to remove

        Returns:
            tuple: Tuple of (Epoch, numpy.ndarray) for the removed epoch and state

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            removed_epc, removed_state = traj.remove(0)
            ```
        """
        ...

    def remove_epoch(self, epoch: Epoch) -> np.ndarray:
        """Remove a state at a specific epoch.

        Args:
            epoch (Epoch): Epoch of the state to remove

        Returns:
            numpy.ndarray: The removed state vector

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            removed_state = traj.remove_epoch(epc)
            ```
        """
        ...

    def set_eviction_policy_max_age(self, max_age: float) -> Any:
        """Set eviction policy to keep states within maximum age.

        Args:
            max_age (float): Maximum age in seconds relative to most recent state

        Example:
            ```python
            import brahe as bh

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            traj.set_eviction_policy_max_age(3600.0)
            ```
        """
        ...

    def set_eviction_policy_max_size(self, max_size: int) -> Any:
        """Set eviction policy to keep maximum number of states.

        Args:
            max_size (int): Maximum number of states to retain

        Example:
            ```python
            import brahe as bh

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            traj.set_eviction_policy_max_size(1000)
            ```
        """
        ...

    def set_interpolation_method(self, method: InterpolationMethod) -> Any:
        """Set the interpolation method for the trajectory.

        Args:
            method (InterpolationMethod): New interpolation method

        Example:
            ```python
            import brahe as bh

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            traj.set_interpolation_method(bh.InterpolationMethod.LINEAR)
            ```
        """
        ...

    def start_epoch(self) -> Epoch:
        """Get start epoch of trajectory.

        Returns:
            Epoch or None: First epoch if trajectory is not empty, None otherwise

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            print(f"Start epoch: {traj.start_epoch()}")
            ```
        """
        ...

    def state(self, epoch: Epoch) -> np.ndarray:
        """Get state at specified epoch (in native frame/representation).

        Args:
            epoch (Epoch): Time for state query

        Returns:
            numpy.ndarray: State vector in trajectory's native frame and representation

        Example:
            ```python
            import brahe as bh
            import numpy as np

            # Create ECI Cartesian trajectory
            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.UTC)
            state1 = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
            traj.add(epc1, state1)

            # Query state at epoch
            state = traj.state(epc1)
            ```
        """
        ...

    def state_after_epoch(self, epoch: Epoch) -> Tuple:
        """Get the state at or after the given epoch.

        Args:
            epoch (Epoch): Target epoch

        Returns:
            tuple: Tuple of (Epoch, numpy.ndarray) containing state at or after the target epoch

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc1, state)
            epc2 = bh.Epoch.from_datetime(2024, 1, 1, 11, 59, 0.0, 0.0, bh.TimeSystem.UTC)
            ret_epc, ret_state = traj.state_after_epoch(epc2)
            ```
        """
        ...

    def state_as_osculating_elements(
        self, epoch: Epoch, angle_format: AngleFormat
    ) -> np.ndarray:
        """Get state as osculating Keplerian elements at specified epoch.

        Args:
            epoch (Epoch): Time for state query
            angle_format (AngleFormat): Desired angle format for output

        Returns:
            numpy.ndarray: Osculating Keplerian elements [a, e, i, raan, argp, M]

        Example:
            ```python
            import brahe as bh
            import numpy as np

            # Create Cartesian trajectory
            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.UTC)
            state_cart = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
            traj.add(epc, state_cart)

            # Get osculating elements in degrees
            elements = traj.state_as_osculating_elements(epc, bh.AngleFormat.DEGREES)
            print(f"Semi-major axis: {elements[0]/1000:.2f} km")
            print(f"Inclination: {elements[2]:.2f} degrees")
            ```
        """
        ...

    def state_at_idx(self, index: int) -> np.ndarray:
        """Get the state vector at a specific index.

        Args:
            index (int): Index of the state to retrieve

        Returns:
            numpy.ndarray: State vector at the specified index

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.UTC)
            state = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
            traj.add(epc, state)

            # Get state at index
            state_0 = traj.state_at_idx(0)
            ```
        """
        ...

    def state_before_epoch(self, epoch: Epoch) -> Tuple:
        """Get the state at or before the given epoch.

        Args:
            epoch (Epoch): Target epoch

        Returns:
            tuple: Tuple of (Epoch, numpy.ndarray) containing state at or before the target epoch

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc1, state)
            epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
            ret_epc, ret_state = traj.state_before_epoch(epc2)
            ```
        """
        ...

    def state_ecef(self, epoch: Epoch) -> np.ndarray:
        """Get state in ECEF Cartesian frame at specified epoch.

        Args:
            epoch (Epoch): Time for state query

        Returns:
            numpy.ndarray: State vector in ECEF Cartesian [x, y, z, vx, vy, vz] (meters, m/s)

        Example:
            ```python
            import brahe as bh
            import numpy as np

            # Create ECI trajectory
            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.UTC)
            state_eci = np.array([7000e3, 0.0, 0.0, 0.0, 7.5e3, 0.0])
            traj.add(epc, state_eci)

            # Get ECEF state (automatically converted from ECI)
            state_ecef = traj.state_ecef(epc)
            ```
        """
        ...

    def state_eci(self, epoch: Epoch) -> np.ndarray:
        """Get state in ECI Cartesian frame at specified epoch.

        Args:
            epoch (Epoch): Time for state query

        Returns:
            numpy.ndarray: State vector in ECI Cartesian [x, y, z, vx, vy, vz] (meters, m/s)

        Example:
            ```python
            import brahe as bh
            import numpy as np

            # Create trajectory in any frame/representation
            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.KEPLERIAN, bh.AngleFormat.DEGREES)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.UTC)
            oe = np.array([bh.R_EARTH + 500e3, 0.001, 98.0, 15.0, 30.0, 45.0])
            traj.add(epc, oe)

            # Get ECI Cartesian state (automatically converted from Keplerian)
            state_eci = traj.state_eci(epc)
            ```
        """
        ...

    def states(self) -> np.ndarray:
        """Get all states as a numpy array.

        Returns:
            numpy.ndarray: 2D array of states with shape (6, N) where N is the number of states

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            traj.add(epc + 60.0, state)
            states_array = traj.states()
            ```
        """
        ...

    def timespan(self) -> float:
        """Get time span of trajectory in seconds.

        Returns:
            float or None: Time span between first and last epochs, or None if less than 2 states

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            traj.add(epc + 3600.0, state)
            print(f"Timespan: {traj.timespan()} seconds")
            ```
        """
        ...

    def to_ecef(self) -> OrbitTrajectory:
        """Convert to ECEF (Earth-Centered Earth-Fixed) frame in Cartesian representation.

        Returns:
            OrbitTrajectory: Trajectory in ECEF Cartesian frame

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            traj_ecef = traj.to_ecef()
            ```
        """
        ...

    def to_eci(self) -> OrbitTrajectory:
        """Convert to ECI (Earth-Centered Inertial) frame in Cartesian representation.

        Returns:
            OrbitTrajectory: Trajectory in ECI Cartesian frame

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECEF, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 0.0, 0.0])
            traj.add(epc, state)
            traj_eci = traj.to_eci()
            ```
        """
        ...

    def to_keplerian(self, angle_format: AngleFormat) -> OrbitTrajectory:
        """Convert to Keplerian representation in ECI frame.

        Args:
            angle_format (AngleFormat): Angle format for the result (Radians or Degrees)

        Returns:
            OrbitTrajectory: Trajectory in ECI Keplerian representation

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            traj_kep = traj.to_keplerian(bh.AngleFormat.RADIANS)
            ```
        """
        ...

    def to_matrix(self) -> np.ndarray:
        """Convert trajectory to matrix representation.

        Returns:
            numpy.ndarray: 2D array with shape (6, N) where N is number of states

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            matrix = traj.to_matrix()
            ```
        """
        ...

    def with_eviction_policy_max_age(self, max_age: float) -> OrbitTrajectory:
        """Set eviction policy to keep states within maximum age using builder pattern.

        Args:
            max_age (float): Maximum age of states in seconds

        Returns:
            OrbitTrajectory: Self with updated eviction policy

        Example:
            ```python
            import brahe as bh

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            traj = traj.with_eviction_policy_max_age(3600.0)
            ```
        """
        ...

    def with_eviction_policy_max_size(self, max_size: int) -> OrbitTrajectory:
        """Set eviction policy to keep maximum number of states using builder pattern.

        Args:
            max_size (int): Maximum number of states to retain

        Returns:
            OrbitTrajectory: Self with updated eviction policy

        Example:
            ```python
            import brahe as bh

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            traj = traj.with_eviction_policy_max_size(1000)
            ```
        """
        ...

    def with_interpolation_method(
        self, interpolation_method: InterpolationMethod
    ) -> OrbitTrajectory:
        """Set interpolation method using builder pattern.

        Args:
            interpolation_method (InterpolationMethod): Interpolation method to use

        Returns:
            OrbitTrajectory: Self with updated interpolation method

        Example:
            ```python
            import brahe as bh

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            traj = traj.with_interpolation_method(bh.InterpolationMethod.LINEAR)
            ```
        """
        ...

    @property
    def angle_format(self) -> AngleFormat:
        """Get trajectory angle format for Keplerian states.

        Returns:
            AngleFormat or None: Angle format for Keplerian representation, None for Cartesian

        Example:
            ```python
            import brahe as bh

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            print(f"Angle format: {traj.angle_format}")
            ```
        """
        ...

    @property
    def frame(self) -> OrbitFrame:
        """Get trajectory reference frame.

        Returns:
            OrbitFrame: Reference frame of the trajectory

        Example:
            ```python
            import brahe as bh

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            print(f"Frame: {traj.frame}")
            ```
        """
        ...

    @property
    def length(self) -> int:
        """Get the number of states in the trajectory.

        Returns:
            int: Number of states in the trajectory

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.DTrajectory(6)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            print(f"Trajectory length: {traj.length}")
            ```
        """
        ...

    @property
    def representation(self) -> OrbitRepresentation:
        """Get trajectory state representation.

        Returns:
            OrbitRepresentation: State representation format of the trajectory

        Example:
            ```python
            import brahe as bh

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            print(f"Representation: {traj.representation}")
            ```
        """
        ...

class PanicException:
    """The exception raised when Rust code called from Python panics.

    Like SystemExit, this exception is derived from BaseException so that
    it will typically propagate all the way through the stack and cause the
    Python interpreter to exit.
    """

    def __init__(self) -> None:
        """Initialize instance."""
        ...

    def add_note(self) -> Any:
        """Exception.add_note(note) --
        add a note to the exception
        """
        ...

    def with_traceback(self) -> Any:
        """Exception.with_traceback(tb) --
        set self.__traceback__ to tb and return self.
        """
        ...

    @property
    def args(self) -> Any:
        """TODO: Add docstring"""
        ...

class PointLocation:
    """A single point location on Earth's surface.
    
    Represents a discrete point with geodetic coordinates (longitude, latitude, altitude).
    Commonly used for ground stations, imaging targets, or tessellated polygon tiles.
    
    Args:
        lon (float): Longitude in degrees (-180 to 180)
        lat (float): Latitude in degrees (-90 to 90)
        alt (float): Altitude above ellipsoid in meters (default: 0.0)
    
    Example:
        ```python
        import brahe as bh
    
        # Create a ground station in Svalbard
        svalbard = bh.PointLocation(lon=15.4, lat=78.2, alt=0.0)
    
        # With identity
        svalbard = bh.PointLocation(lon=15.4, lat=78.2, alt=0.0) \\
            .with_name("Svalbard Ground Station") \\
            .with_id(1)
    
        # With custom properties
        svalbard = bh.PointLocation(lon=15.4, lat=78.2, alt=0.0) \\
            .add_property("country", "Norway") \\
            .add_property("min_elevation_deg", 5.0)
    
        # Access coordinates
        lon = svalbard.lon()  # Quick accessor (always degrees)
        lat_rad = svalbard.latitude(bh.AngleFormat.RADIANS)  # Format-aware
        ```
    """

    def __init__(self, lon: float, lat: float, alt: float) -> None:
        """Initialize instance."""
        ...

    @classmethod
    def from_geojson(cls, geojson: dict) -> PointLocation:
        """Create from GeoJSON Point Feature.

        Args:
            geojson (dict): GeoJSON Feature object with Point geometry

        Returns:
            PointLocation: New location instance

        Raises:
            ValueError: If GeoJSON is invalid or not a Point Feature

        Example:
            ```python
            import brahe as bh

            geojson = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [15.4, 78.2, 0.0]
                },
                "properties": {
                    "name": "Svalbard"
                }
            }

            location = bh.PointLocation.from_geojson(geojson)
            ```
        """
        ...

    def add_property(self, key: str) -> PointLocation:
        """Add a custom property (builder pattern).
        
        Args:
            key (str): Property name
            value: Property value (must be JSON-serializable)
        
        Returns:
            PointLocation: Self for chaining
        
        Example:
            ```python
            import brahe as bh
        
            location = bh.PointLocation(lon=15.4, lat=78.2, alt=0.0) \\
                .add_property("country", "Norway") \\
                .add_property("elevation_mask_deg", 5.0)
            ```
        """
        ...

    def alt(self) -> float:
        """Get altitude in meters (quick accessor).

        Returns:
            float: Altitude in meters
        """
        ...

    def altitude(self) -> float:
        """Get altitude in meters.

        Returns:
            float: Altitude in meters
        """
        ...

    def center_ecef(self) -> np.ndarray:
        """Get center position in ECEF coordinates [x, y, z].

        Returns:
            ndarray: ECEF position in meters [x, y, z]
        """
        ...

    def center_geodetic(self) -> np.ndarray:
        """Get center coordinates in geodetic format [lon, lat, alt].

        Returns:
            ndarray: Geodetic coordinates [longitude_deg, latitude_deg, altitude_m]
        """
        ...

    def generate_uuid(self) -> Any:
        """Generate a new UUID (mutating)."""
        ...

    def get_id(self) -> int:
        """Get the numeric ID.

        Returns:
            int | None: ID if set, None otherwise
        """
        ...

    def get_name(self) -> str:
        """Get the name.

        Returns:
            str | None: Name if set, None otherwise
        """
        ...

    def get_uuid(self) -> str:
        """Get the UUID as a string.

        Returns:
            str | None: UUID string if set, None otherwise
        """
        ...

    def lat(self) -> float:
        """Get latitude in degrees (quick accessor).

        Returns:
            float: Latitude in degrees
        """
        ...

    def latitude(self, angle_format: AngleFormat) -> float:
        """Get latitude with angle format conversion.

        Args:
            angle_format (AngleFormat): Desired output format (DEGREES or RADIANS)

        Returns:
            float: Latitude in specified format
        """
        ...

    def lon(self) -> float:
        """Get longitude in degrees (quick accessor).

        Returns:
            float: Longitude in degrees
        """
        ...

    def longitude(self, angle_format: AngleFormat) -> float:
        """Get longitude with angle format conversion.

        Args:
            angle_format (AngleFormat): Desired output format (DEGREES or RADIANS)

        Returns:
            float: Longitude in specified format
        """
        ...

    def set_id(self, id: int | None) -> Any:
        """Set the numeric ID (mutating).

        Args:
            id (int | None): ID to set, or None to clear
        """
        ...

    def set_name(self, name: str | None) -> Any:
        """Set the name (mutating).

        Args:
            name (str | None): Name to set, or None to clear
        """
        ...

    def to_geojson(self) -> dict:
        """Export to GeoJSON Feature format.
        
        Returns:
            dict: GeoJSON Feature object
        
        Example:
            ```python
            import brahe as bh
        
            location = bh.PointLocation(lon=15.4, lat=78.2, alt=0.0) \\
                .with_name("Svalbard")
        
            geojson = location.to_geojson()
            # Returns:
            # {
            #     "type": "Feature",
            #     "geometry": {
            #         "type": "Point",
            #         "coordinates": [15.4, 78.2, 0.0]
            #     },
            #     "properties": {
            #         "name": "Svalbard"
            #     }
            # }
            ```
        """
        ...

    def with_id(self, id: int) -> PointLocation:
        """Set the numeric ID (builder pattern).

        Args:
            id (int): Numeric identifier

        Returns:
            PointLocation: Self for chaining
        """
        ...

    def with_name(self, name: str) -> PointLocation:
        """Set the name (builder pattern).

        Args:
            name (str): Human-readable name

        Returns:
            PointLocation: Self for chaining
        """
        ...

    def with_new_uuid(self) -> PointLocation:
        """Generate a new UUID (builder pattern).

        Returns:
            PointLocation: Self for chaining
        """
        ...

    def with_uuid(self, uuid_str: str) -> PointLocation:
        """Set the UUID from a string (builder pattern).

        Args:
            uuid_str (str): UUID string

        Returns:
            PointLocation: Self for chaining

        Raises:
            ValueError: If UUID string is invalid
        """
        ...

    @property
    def properties(self) -> PropertiesDict:
        """Get custom properties dictionary.

        Returns:
            PropertiesDict: Dictionary-like wrapper for properties that supports assignment

        Example:
            ```python
            import brahe as bh

            loc = bh.PointLocation(15.4, 78.2, 0.0)

            # Dict-style assignment
            loc.properties["climate"] = "Arctic"
            loc.properties["country"] = "Norway"

            # Dict-style access
            print(loc.properties["climate"])  # "Arctic"

            # Dict methods
            if "country" in loc.properties:
                del loc.properties["country"]

            # Iteration
            for key in loc.properties.keys():
                print(key, loc.properties[key])
            ```
        """
        ...

class PolygonLocation:
    """A polygonal area on Earth's surface.
    
    Represents a closed polygon with multiple vertices.
    Commonly used for areas of interest, no-fly zones, or imaging footprints.
    
    The polygon is automatically closed if the first and last vertices don't match.
    
    Args:
        vertices (list[list[float]]): List of [lon, lat, alt] vertices in degrees and meters
    
    Raises:
        ValueError: If polygon has fewer than 4 vertices or has validation errors
    
    Example:
        ```python
        import brahe as bh
    
        # Define a rectangular area
        vertices = [
            [10.0, 50.0, 0.0],  # lon, lat, alt
            [11.0, 50.0, 0.0],
            [11.0, 51.0, 0.0],
            [10.0, 51.0, 0.0],
            [10.0, 50.0, 0.0],  # Closed (first == last)
        ]
        polygon = bh.PolygonLocation(vertices)
    
        # With identity
        polygon = bh.PolygonLocation(vertices) \\
            .with_name("AOI-1") \\
            .add_property("region", "Europe")
        ```
    """

    def __init__(self, vertices: list[list[float]]) -> None:
        """Initialize instance."""
        ...

    @classmethod
    def from_geojson(cls, geojson: dict) -> PolygonLocation:
        """Create from GeoJSON Polygon Feature.

        Args:
            geojson (dict): GeoJSON Feature object with Polygon geometry

        Returns:
            PolygonLocation: New polygon instance

        Raises:
            ValueError: If GeoJSON is invalid or not a Polygon Feature

        Example:
            ```python
            import brahe as bh

            geojson = {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[
                        [10.0, 50.0, 0.0],
                        [11.0, 50.0, 0.0],
                        [11.0, 51.0, 0.0],
                        [10.0, 51.0, 0.0],
                        [10.0, 50.0, 0.0]
                    ]]
                },
                "properties": {
                    "name": "AOI-1"
                }
            }

            polygon = bh.PolygonLocation.from_geojson(geojson)
            ```
        """
        ...

    def add_property(self, key: str) -> PolygonLocation:
        """Add a custom property (builder pattern).

        Args:
            key (str): Property name
            value: Property value (must be JSON-serializable)

        Returns:
            PolygonLocation: Self for chaining
        """
        ...

    def alt(self) -> float:
        """Get center altitude in meters (quick accessor).

        Returns:
            float: Center altitude in meters
        """
        ...

    def altitude(self) -> float:
        """Get center altitude in meters.

        Returns:
            float: Center altitude in meters
        """
        ...

    def center_ecef(self) -> np.ndarray:
        """Get center position in ECEF coordinates [x, y, z].

        Returns:
            ndarray: ECEF position in meters [x, y, z]
        """
        ...

    def center_geodetic(self) -> np.ndarray:
        """Get center coordinates in geodetic format [lon, lat, alt].

        Returns:
            ndarray: Geodetic coordinates [longitude_deg, latitude_deg, altitude_m]
        """
        ...

    def generate_uuid(self) -> Any:
        """Generate a new UUID (mutating)."""
        ...

    def get_id(self) -> int:
        """Get the numeric ID.

        Returns:
            int | None: ID if set, None otherwise
        """
        ...

    def get_name(self) -> str:
        """Get the name.

        Returns:
            str | None: Name if set, None otherwise
        """
        ...

    def get_uuid(self) -> str:
        """Get the UUID as a string.

        Returns:
            str | None: UUID string if set, None otherwise
        """
        ...

    def lat(self) -> float:
        """Get center latitude in degrees (quick accessor).

        Returns:
            float: Center latitude in degrees
        """
        ...

    def latitude(self, angle_format: AngleFormat) -> float:
        """Get center latitude with angle format conversion.

        Args:
            angle_format (AngleFormat): Desired output format (DEGREES or RADIANS)

        Returns:
            float: Center latitude in specified format
        """
        ...

    def lon(self) -> float:
        """Get center longitude in degrees (quick accessor).

        Returns:
            float: Center longitude in degrees
        """
        ...

    def longitude(self, angle_format: AngleFormat) -> float:
        """Get center longitude with angle format conversion.

        Args:
            angle_format (AngleFormat): Desired output format (DEGREES or RADIANS)

        Returns:
            float: Center longitude in specified format
        """
        ...

    def num_vertices(self) -> int:
        """Get number of unique vertices (excluding closure).

        Returns:
            int: Number of unique vertices
        """
        ...

    def set_id(self, id: int | None) -> Any:
        """Set the numeric ID (mutating).

        Args:
            id (int | None): ID to set, or None to clear
        """
        ...

    def set_name(self, name: str | None) -> Any:
        """Set the name (mutating).

        Args:
            name (str | None): Name to set, or None to clear
        """
        ...

    def to_geojson(self) -> dict:
        """Export to GeoJSON Feature format.

        Returns:
            dict: GeoJSON Feature object
        """
        ...

    def vertices(self) -> np.ndarray:
        """Get polygon vertices.

        Returns all vertices including the closure vertex (first == last).

        Returns:
            ndarray: Vertices as Nx3 array [[lon, lat, alt], ...]
        """
        ...

    def with_id(self, id: int) -> PolygonLocation:
        """Set the numeric ID (builder pattern).

        Args:
            id (int): Numeric identifier

        Returns:
            PolygonLocation: Self for chaining
        """
        ...

    def with_name(self, name: str) -> PolygonLocation:
        """Set the name (builder pattern).

        Args:
            name (str): Human-readable name

        Returns:
            PolygonLocation: Self for chaining
        """
        ...

    def with_new_uuid(self) -> PolygonLocation:
        """Generate a new UUID (builder pattern).

        Returns:
            PolygonLocation: Self for chaining
        """
        ...

    def with_uuid(self, uuid_str: str) -> PolygonLocation:
        """Set the UUID from a string (builder pattern).

        Args:
            uuid_str (str): UUID string

        Returns:
            PolygonLocation: Self for chaining

        Raises:
            ValueError: If UUID string is invalid
        """
        ...

    @property
    def properties(self) -> PropertiesDict:
        """Get custom properties dictionary.

        Returns:
            PropertiesDict: Dictionary-like wrapper for properties that supports assignment

        Example:
            ```python
            import brahe as bh

            verts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
            poly = bh.PolygonLocation(verts)

            # Dict-style assignment
            poly.properties["region"] = "Test Area"
            poly.properties["area_km2"] = 123.45

            # Dict-style access
            print(poly.properties["region"])  # "Test Area"

            # Dict methods
            if "area_km2" in poly.properties:
                del poly.properties["area_km2"]
            ```
        """
        ...

class PropertiesDict:
    """A dictionary-like wrapper for Location properties that supports dict-style assignment.

    This class provides a Pythonic dict interface for accessing and modifying location properties.
    Changes are automatically synchronized with the underlying Location object.

    Example:
        ```python
        import brahe as bh

        loc = bh.PointLocation(15.4, 78.2, 0.0)

        # Dict-style assignment
        loc.properties["climate"] = "Arctic"
        loc.properties["country"] = "Norway"

        # Dict-style access
        climate = loc.properties["climate"]

        # Dict methods work
        if "country" in loc.properties:
            print(loc.properties["country"])

        # Iteration
        for key in loc.properties:
            print(key, loc.properties[key])
        ```
    """

    def __init__(self) -> None:
        """Initialize instance."""
        ...

    def clear(self) -> Any:
        """Remove all properties."""
        ...

    def get(self) -> Any:
        """Get property value with optional default."""
        ...

    def items(self) -> Any:
        """Return a list of (key, value) tuples."""
        ...

    def keys(self) -> Any:
        """Return a list of property keys."""
        ...

    def update(self) -> Tuple[int, ...]:
        """Update properties from another dict."""
        ...

    def values(self) -> Any:
        """Return a list of property values."""
        ...

class Quaternion:
    """Represents a quaternion for 3D rotations.

    Quaternions provide a compact, singularity-free representation of rotations.
    The quaternion is stored as [w, x, y, z] where w is the scalar part and
    [x, y, z] is the vector part.

    Args:
        w (float): Scalar component
        x (float): X component of vector part
        y (float): Y component of vector part
        z (float): Z component of vector part

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Create identity quaternion
        q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
        print(f"Norm: {q.norm()}")

        # Create from array
        q_vec = np.array([1.0, 0.0, 0.0, 0.0])
        q2 = bh.Quaternion.from_vector(q_vec, scalar_first=True)

        # Convert to rotation matrix
        dcm = q.to_rotation_matrix()

        # Quaternion multiplication
        q3 = q * q2

        # Normalize
        q3.normalize()
        ```
    """

    def __init__(self, w: float, x: float, y: float, z: float) -> None:
        """Initialize instance."""
        ...

    @classmethod
    def from_euler_angle(cls, e: EulerAngle) -> Quaternion:
        """Create a quaternion from an Euler angle representation.

        Args:
            e (EulerAngle): Euler angle representation

        Returns:
            Quaternion: Equivalent quaternion

        Example:
            ```python
            import brahe as bh

            euler = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
            q = bh.Quaternion.from_euler_angle(euler)
            ```
        """
        ...

    @classmethod
    def from_euler_axis(cls, e: EulerAxis) -> Quaternion:
        """Create a quaternion from an Euler axis representation.

        Args:
            e (EulerAxis): Euler axis representation

        Returns:
            Quaternion: Equivalent quaternion

        Example:
            ```python
            import brahe as bh
            import numpy as np

            axis = np.array([0.0, 0.0, 1.0])
            ea = bh.EulerAxis(axis, 1.5708, bh.AngleFormat.RADIANS)
            q = bh.Quaternion.from_euler_axis(ea)
            ```
        """
        ...

    @classmethod
    def from_quaternion(cls, q: Quaternion) -> Quaternion:
        """Create a quaternion from another quaternion (copy constructor).

        Args:
            q (Quaternion): Source quaternion

        Returns:
            Quaternion: New quaternion instance

        Example:
            ```python
            import brahe as bh

            q1 = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
            q2 = bh.Quaternion.from_quaternion(q1)
            ```
        """
        ...

    @classmethod
    def from_rotation_matrix(cls, r: RotationMatrix) -> Quaternion:
        """Create a quaternion from a rotation matrix.

        Args:
            r (RotationMatrix): Rotation matrix

        Returns:
            Quaternion: Equivalent quaternion

        Example:
            ```python
            import brahe as bh
            import numpy as np

            mat = np.eye(3)
            rm = bh.RotationMatrix.from_matrix(mat)
            q = bh.Quaternion.from_rotation_matrix(rm)
            ```
        """
        ...

    @classmethod
    def from_vector(cls, v: np.ndarray, scalar_first: bool) -> Quaternion:
        """Create a quaternion from a numpy array.

        Args:
            v (numpy.ndarray): 4-element array containing quaternion components
            scalar_first (bool): If True, array is [w, x, y, z], else [x, y, z, w]

        Returns:
            Quaternion: New quaternion instance

        Example:
            ```python
            import brahe as bh
            import numpy as np

            v = np.array([1.0, 0.0, 0.0, 0.0])
            q = bh.Quaternion.from_vector(v, scalar_first=True)
            ```
        """
        ...

    def conjugate(self) -> Quaternion:
        """Compute the conjugate of the quaternion.

        Returns:
            Quaternion: Conjugate quaternion with negated vector part

        Example:
            ```python
            import brahe as bh

            q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
            q_conj = q.conjugate()
            ```
        """
        ...

    def inverse(self) -> Quaternion:
        """Compute the inverse of the quaternion.

        Returns:
            Quaternion: Inverse quaternion

        Example:
            ```python
            import brahe as bh

            q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
            q_inv = q.inverse()
            ```
        """
        ...

    def norm(self) -> float:
        """Calculate the norm (magnitude) of the quaternion.

        Returns:
            float: Euclidean norm of the quaternion

        Example:
            ```python
            import brahe as bh

            q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
            norm = q.norm()
            ```
        """
        ...

    def normalize(self) -> Any:
        """Normalize the quaternion in-place to unit length.

        Example:
            ```python
            import brahe as bh

            q = bh.Quaternion(2.0, 0.0, 0.0, 0.0)
            q.normalize()
            ```
        """
        ...

    def slerp(self, other: Quaternion, t: float) -> Quaternion:
        """Perform spherical linear interpolation (SLERP) between two quaternions.

        Args:
            other (Quaternion): Target quaternion
            t (float): Interpolation parameter in [0, 1]

        Returns:
            Quaternion: Interpolated quaternion

        Example:
            ```python
            import brahe as bh

            q1 = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
            q2 = bh.Quaternion(0.707, 0.707, 0.0, 0.0)
            q_mid = q1.slerp(q2, 0.5)
            ```
        """
        ...

    def to_euler_angle(self, order: str) -> EulerAngle:
        """Convert to Euler angle representation.

        Args:
            order (str): Rotation sequence (e.g., "XYZ", "ZYX")

        Returns:
            EulerAngle: Equivalent Euler angles

        Example:
            ```python
            import brahe as bh

            q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
            euler = q.to_euler_angle("XYZ")
            ```
        """
        ...

    def to_euler_axis(self) -> EulerAxis:
        """Convert to Euler axis representation.

        Returns:
            EulerAxis: Equivalent Euler axis
        """
        ...

    def to_quaternion(self) -> Quaternion:
        """Convert to quaternion representation (returns self).

        Returns:
            Quaternion: This quaternion
        """
        ...

    def to_rotation_matrix(self) -> RotationMatrix:
        """Convert to rotation matrix representation.

        Returns:
            RotationMatrix: Equivalent rotation matrix
        """
        ...

    def to_vector(self, scalar_first: bool) -> np.ndarray:
        """Convert quaternion to a numpy array.

        Args:
            scalar_first (bool): If True, returns [w, x, y, z], else [x, y, z, w]

        Returns:
            numpy.ndarray: 4-element array containing quaternion components

        Example:
            ```python
            import brahe as bh

            q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
            v = q.to_vector(scalar_first=True)
            ```
        """
        ...

    @property
    def data(self) -> np.ndarray:
        """Get the quaternion components as a numpy array [w, x, y, z].

        Returns:
            numpy.ndarray: 4-element array containing quaternion components
        """
        ...

class RotationMatrix:
    """Represents a rotation using a 3x3 rotation matrix (Direction Cosine Matrix).

    A rotation matrix is an orthogonal 3x3 matrix with determinant +1 that
    represents rotation in 3D space. Also known as a Direction Cosine Matrix (DCM).

    Args:
        r11 (float): Element at row 1, column 1
        r12 (float): Element at row 1, column 2
        r13 (float): Element at row 1, column 3
        r21 (float): Element at row 2, column 1
        r22 (float): Element at row 2, column 2
        r23 (float): Element at row 2, column 3
        r31 (float): Element at row 3, column 1
        r32 (float): Element at row 3, column 2
        r33 (float): Element at row 3, column 3

    Raises:
        BraheError: If the matrix is not a valid rotation matrix

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Create identity rotation
        dcm = bh.RotationMatrix(1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0)

        # Create from numpy array
        R = np.eye(3)
        dcm2 = bh.RotationMatrix.from_matrix(R)

        # Convert to quaternion
        q = dcm.to_quaternion()

        # Rotate a vector
        v = np.array([1.0, 0.0, 0.0])
        v_rot = dcm.rotate_vector(v)
        ```
    """

    def __init__(
        self,
        r11: float,
        r12: float,
        r13: float,
        r21: float,
        r22: float,
        r23: float,
        r31: float,
        r32: float,
        r33: float,
    ) -> None:
        """Initialize instance."""
        ...

    @classmethod
    def Rx(cls, angle: float, angle_format: AngleFormat) -> RotationMatrix:
        """Create a rotation matrix for rotation about the X axis.

        Args:
            angle (float): Rotation angle in radians or degrees
            angle_format (AngleFormat): Units of input angle (RADIANS or DEGREES)

        Returns:
            RotationMatrix: Rotation matrix for X-axis rotation
        """
        ...

    @classmethod
    def Ry(cls, angle: float, angle_format: AngleFormat) -> RotationMatrix:
        """Create a rotation matrix for rotation about the Y axis.

        Args:
            angle (float): Rotation angle in radians or degrees
            angle_format (AngleFormat): Units of input angle (RADIANS or DEGREES)

        Returns:
            RotationMatrix: Rotation matrix for Y-axis rotation
        """
        ...

    @classmethod
    def Rz(cls, angle: float, angle_format: AngleFormat) -> RotationMatrix:
        """Create a rotation matrix for rotation about the Z axis.

        Args:
            angle (float): Rotation angle in radians or degrees
            angle_format (AngleFormat): Units of input angle (RADIANS or DEGREES)

        Returns:
            RotationMatrix: Rotation matrix for Z-axis rotation

        Example:
            ```python
            import brahe as bh

            r = bh.RotationMatrix.Rz(1.5708, bh.AngleFormat.RADIANS)
            ```
        """
        ...

    @classmethod
    def from_euler_angle(cls, e: EulerAngle) -> RotationMatrix:
        """Create a rotation matrix from Euler angles.

        Args:
            e (EulerAngle): Euler angle representation

        Returns:
            RotationMatrix: Equivalent rotation matrix

        Example:
            ```python
            import brahe as bh

            euler = bh.EulerAngle("XYZ", 0.1, 0.2, 0.3, bh.AngleFormat.RADIANS)
            r = bh.RotationMatrix.from_euler_angle(euler)
            ```
        """
        ...

    @classmethod
    def from_euler_axis(cls, e: EulerAxis) -> RotationMatrix:
        """Create a rotation matrix from an Euler axis.

        Args:
            e (EulerAxis): Euler axis representation

        Returns:
            RotationMatrix: Equivalent rotation matrix

        Example:
            ```python
            import brahe as bh
            import numpy as np

            axis = np.array([0.0, 0.0, 1.0])
            ea = bh.EulerAxis(axis, 1.5708, bh.AngleFormat.RADIANS)
            r = bh.RotationMatrix.from_euler_axis(ea)
            ```
        """
        ...

    @classmethod
    def from_matrix(cls, m: np.ndarray) -> RotationMatrix:
        """Create a rotation matrix from a 3x3 numpy array.

        Args:
            m (numpy.ndarray): 3x3 rotation matrix

        Returns:
            RotationMatrix: New rotation matrix instance

        Raises:
            BraheError: If the matrix is not a valid rotation matrix

        Example:
            ```python
            import brahe as bh
            import numpy as np

            mat = np.eye(3)
            r = bh.RotationMatrix.from_matrix(mat)
            ```
        """
        ...

    @classmethod
    def from_quaternion(cls, q: Quaternion) -> RotationMatrix:
        """Create a rotation matrix from a quaternion.

        Args:
            q (Quaternion): Source quaternion

        Returns:
            RotationMatrix: Equivalent rotation matrix

        Example:
            ```python
            import brahe as bh

            q = bh.Quaternion(1.0, 0.0, 0.0, 0.0)
            r = bh.RotationMatrix.from_quaternion(q)
            ```
        """
        ...

    @classmethod
    def from_rotation_matrix(cls, r: RotationMatrix) -> RotationMatrix:
        """Create a rotation matrix from another rotation matrix (copy constructor).

        Args:
            r (RotationMatrix): Source rotation matrix

        Returns:
            RotationMatrix: New rotation matrix instance

        Example:
            ```python
            import brahe as bh
            import numpy as np

            r1 = bh.RotationMatrix.from_array(np.eye(3))
            r2 = bh.RotationMatrix.from_rotation_matrix(r1)
            ```
        """
        ...

    def to_euler_angle(self, order: str) -> EulerAngle:
        """Convert to Euler angle representation.

        Args:
            order (str): Desired rotation sequence (e.g., "XYZ", "ZYX")

        Returns:
            EulerAngle: Equivalent Euler angles

        Example:
            ```python
            import brahe as bh
            import numpy as np

            r = bh.RotationMatrix.from_array(np.eye(3))
            euler = r.to_euler_angle("XYZ")
            ```
        """
        ...

    def to_euler_axis(self) -> EulerAxis:
        """Convert to Euler axis representation.

        Returns:
            EulerAxis: Equivalent Euler axis

        Example:
            ```python
            import brahe as bh
            import numpy as np

            r = bh.RotationMatrix.from_array(np.eye(3))
            e = r.to_euler_axis()
            ```
        """
        ...

    def to_matrix(self) -> np.ndarray:
        """Convert rotation matrix to a 3x3 numpy array.

        Returns:
            numpy.ndarray: 3x3 rotation matrix
        """
        ...

    def to_quaternion(self) -> Quaternion:
        """Convert to quaternion representation.

        Returns:
            Quaternion: Equivalent quaternion

        Example:
            ```python
            import brahe as bh
            import numpy as np

            r = bh.RotationMatrix.from_array(np.eye(3))
            q = r.to_quaternion()
            ```
        """
        ...

    def to_rotation_matrix(self) -> RotationMatrix:
        """Convert to rotation matrix representation (returns self).

        Returns:
            RotationMatrix: This rotation matrix

        Example:
            ```python
            import brahe as bh
            import numpy as np

            r1 = bh.RotationMatrix.from_array(np.eye(3))
            r2 = r1.to_rotation_matrix()
            ```
        """
        ...

class SGPPropagator:
    """Python wrapper for SGPPropagator (replaces TLE)
    SGP4/SDP4 satellite propagator using TLE data.

    The SGP (Simplified General Perturbations) propagator implements the SGP4/SDP4 models
    for propagating satellites using Two-Line Element (TLE) orbital data. This is the standard
    model used for tracking objects in Earth orbit.

    Example:
        ```python
        import brahe as bh

        # ISS TLE data (example)
        line1 = "1 25544U 98067A   24001.50000000  .00016717  00000-0  30000-3 0  9005"
        line2 = "2 25544  51.6400 150.0000 0003000 100.0000 260.0000 15.50000000300000"

        # Create propagator
        prop = bh.SGPPropagator.from_tle(line1, line2, step_size=60.0)

        # Propagate to a specific epoch
        epc = bh.Epoch.from_datetime(2024, 1, 2, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
        state_eci = prop.state(epc)
        print(f"Position: {state_eci[:3]}")
        print(f"Velocity: {state_eci[3:]}")

        # Propagate multiple epochs
        epochs = [epc + i*60.0 for i in range(10)]  # 10 minutes
        states = prop.states(epochs)
        ```
    """

    def __init__(self) -> None:
        """Initialize instance."""
        ...

    @classmethod
    def from_3le(
        cls, name: str, line1: str, line2: str, step_size: float
    ) -> SGPPropagator:
        """Create a new SGP propagator from 3-line TLE format (with satellite name).

        Args:
            name (str): Satellite name (line 0).
            line1 (str): First line of TLE data.
            line2 (str): Second line of TLE data.
            step_size (float): Step size in seconds for propagation. Defaults to 60.0.

        Returns:
            SGPPropagator: New SGP propagator instance.
        """
        ...

    @classmethod
    def from_tle(cls, line1: str, line2: str, step_size: float) -> SGPPropagator:
        """Create a new SGP propagator from TLE lines.

        Args:
            line1 (str): First line of TLE data.
            line2 (str): Second line of TLE data.
            step_size (float): Step size in seconds for propagation. Defaults to 60.0.

        Returns:
            SGPPropagator: New SGP propagator instance.
        """
        ...

    def current_state(self) -> np.ndarray:
        """Get current state vector.

        Returns:
            numpy.ndarray: Current state vector in the propagator's output format.
        """
        ...

    def generate_uuid(self) -> Any:
        """Generate a new UUID and set it in-place (mutating)."""
        ...

    def get_elements(self, angle_format: AngleFormat) -> np.ndarray:
        """Get Keplerian orbital elements from TLE data.

        Extracts the Keplerian elements directly from the TLE lines used to
        initialize this propagator.

        Args:
            angle_format (AngleFormat): Format for angular elements (DEGREES or RADIANS).

        Returns:
            numpy.ndarray: Keplerian elements [a, e, i, Ω, ω, M] where:
                - a: semi-major axis [m]
                - e: eccentricity [dimensionless]
                - i: inclination [rad or deg]
                - Ω: right ascension of ascending node [rad or deg]
                - ω: argument of periapsis [rad or deg]
                - M: mean anomaly [rad or deg]

        Example:
            ```python
            import brahe as bh

            line1 = "1 25544U 98067A   08264.51782528 -.00002182  00000-0 -11606-4 0  2927"
            line2 = "2 25544  51.6416 247.4627 0006703 130.5360 325.0288 15.72125391563537"
            prop = bh.SGPPropagator.from_tle(line1, line2)

            # Get elements in degrees
            oe_deg = prop.get_elements(bh.AngleFormat.DEGREES)
            print(f"Inclination: {oe_deg[2]:.4f} degrees")

            # Get elements in radians
            oe_rad = prop.get_elements(bh.AngleFormat.RADIANS)
            print(f"Inclination: {oe_rad[2]:.4f} radians")
            ```
        """
        ...

    def initial_state(self) -> np.ndarray:
        """Get initial state vector.

        Returns:
            numpy.ndarray: Initial state vector in the propagator's output format.
        """
        ...

    def propagate_steps(self, num_steps: int) -> Any:
        """Propagate forward by specified number of steps.

        Args:
            num_steps (int): Number of steps to take.

        Example:
            ```python
            import brahe as bh

            line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
            line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
            prop = bh.SGPPropagator.from_tle(line1, line2, step_size=60.0)
            prop.propagate_steps(10)  # Advance by 10 steps (600 seconds)
            print(f"After 10 steps: {prop.current_epoch}")
            ```
        """
        ...

    def propagate_to(self, target_epoch: Epoch) -> Any:
        """Propagate to a specific target epoch.

        Args:
            target_epoch (Epoch): The epoch to propagate to.

        Example:
            ```python
            import brahe as bh

            line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
            line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
            prop = bh.SGPPropagator.from_tle(line1, line2)
            target = prop.epoch + 7200.0  # 2 hours later
            prop.propagate_to(target)
            print(f"Propagated to: {prop.current_epoch}")
            ```
        """
        ...

    def reset(self) -> Any:
        """Reset propagator to initial conditions.

        Example:
            ```python
            import brahe as bh

            line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
            line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
            prop = bh.SGPPropagator.from_tle(line1, line2)
            initial_epoch = prop.epoch
            prop.propagate_steps(100)
            prop.reset()
            print(f"Reset to: {prop.current_epoch == initial_epoch}")
            ```
        """
        ...

    def set_eviction_policy_max_age(self, max_age: float) -> Any:
        """Set trajectory eviction policy based on maximum age.

        Args:
            max_age (float): Maximum age in seconds to keep states in trajectory.

        Example:
            ```python
            import brahe as bh

            line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
            line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
            prop = bh.SGPPropagator.from_tle(line1, line2)
            prop.set_eviction_policy_max_age(86400.0)  # Keep 1 day of history
            print("Trajectory limited to 24 hours of states")
            ```
        """
        ...

    def set_eviction_policy_max_size(self, max_size: int) -> Any:
        """Set trajectory eviction policy based on maximum size.

        Args:
            max_size (int): Maximum number of states to keep in trajectory.

        Example:
            ```python
            import brahe as bh

            line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
            line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
            prop = bh.SGPPropagator.from_tle(line1, line2)
            prop.set_eviction_policy_max_size(1000)
            print("Trajectory limited to 1000 states")
            ```
        """
        ...

    def set_id(self, id: int or None) -> Any:
        """Set the numeric ID in-place (mutating).

        Args:
            id (int or None): Numeric ID to assign, or None to clear.
        """
        ...

    def set_identity(
        self, name: str or None, uuid_str: str or None, id: int or None
    ) -> Any:
        """Set all identity fields in-place (mutating).

        Args:
            name (str or None): Optional name to assign.
            uuid_str (str or None): Optional UUID string to assign.
            id (int or None): Optional numeric ID to assign.
        """
        ...

    def set_name(self, name: str or None) -> Any:
        """Set the name in-place (mutating).

        Args:
            name (str or None): Name to assign, or None to clear.
        """
        ...

    def set_output_format(
        self,
        frame: OrbitFrame,
        representation: OrbitRepresentation,
        angle_format: AngleFormat or None,
    ) -> Any:
        """Set output format (frame, representation, and angle format).

        Args:
            frame (OrbitFrame): Output frame (ECI or ECEF).
            representation (OrbitRepresentation): Output representation (Cartesian or Keplerian).
            angle_format (AngleFormat or None): Angle format for Keplerian (None for Cartesian).
        """
        ...

    def state(self, epoch: Epoch) -> np.ndarray:
        """Compute state at a specific epoch.

        Args:
            epoch (Epoch): Target epoch for state computation.

        Returns:
            numpy.ndarray: State vector in the propagator's current output format.
        """
        ...

    def state_ecef(self, epoch: Epoch) -> np.ndarray:
        """Compute state at a specific epoch in ECEF coordinates.

        Args:
            epoch (Epoch): Target epoch for state computation.

        Returns:
            numpy.ndarray: State vector [x, y, z, vx, vy, vz] in ECEF frame.
        """
        ...

    def state_eci(self, epoch: Epoch) -> np.ndarray:
        """Compute state at a specific epoch in ECI coordinates.

        Args:
            epoch (Epoch): Target epoch for state computation.

        Returns:
            numpy.ndarray: State vector [x, y, z, vx, vy, vz] in ECI frame.
        """
        ...

    def state_pef(self, epoch: Epoch) -> np.ndarray:
        """Compute state at a specific epoch in PEF coordinates.

        Args:
            epoch (Epoch): Target epoch for state computation.

        Returns:
            numpy.ndarray: State vector [x, y, z, vx, vy, vz] in PEF frame.
        """
        ...

    def states(self, epochs: list[Epoch]) -> List:
        """Compute states at multiple epochs.

        Args:
            epochs (list[Epoch]): List of epochs for state computation.

        Returns:
            list[numpy.ndarray]: List of state vectors in the propagator's current output format.
        """
        ...

    def states_eci(self, epochs: list[Epoch]) -> List:
        """Compute states at multiple epochs in ECI coordinates.

        Args:
            epochs (list[Epoch]): List of epochs for state computation.

        Returns:
            list[numpy.ndarray]: List of ECI state vectors.
        """
        ...

    def step(self) -> Any:
        """Step forward by the default step size.

        Example:
            ```python
            import brahe as bh

            line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
            line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
            prop = bh.SGPPropagator.from_tle(line1, line2)
            prop.step()  # Advance by default step_size
            print(f"Advanced to: {prop.current_epoch}")
            ```
        """
        ...

    def step_by(self, step_size: float) -> Any:
        """Step forward by a specified time duration.

        Args:
            step_size (float): Time step in seconds.

        Example:
            ```python
            import brahe as bh

            line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
            line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
            prop = bh.SGPPropagator.from_tle(line1, line2)
            prop.step_by(120.0)  # Advance by 2 minutes
            print(f"Advanced to: {prop.current_epoch}")
            ```
        """
        ...

    def step_past(self, target_epoch: Epoch) -> Any:
        """Step past a specified target epoch.

        Args:
            target_epoch (Epoch): The epoch to step past.

        Example:
            ```python
            import brahe as bh

            line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
            line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
            prop = bh.SGPPropagator.from_tle(line1, line2)
            target = prop.epoch + 3600.0  # 1 hour later
            prop.step_past(target)
            print(f"Stepped past target")
            ```
        """
        ...

    def with_id(self, id: int) -> SGPPropagator:
        """Set the numeric ID and return self (consuming constructor pattern).

        Args:
            id (int): Numeric ID to assign to this propagator.

        Returns:
            SGPPropagator: Self with ID set.
        """
        ...

    def with_identity(
        self, name: str or None, uuid_str: str or None, id: int or None
    ) -> SGPPropagator:
        """Set all identity fields at once and return self (consuming constructor pattern).

        Args:
            name (str or None): Optional name to assign.
            uuid_str (str or None): Optional UUID string to assign.
            id (int or None): Optional numeric ID to assign.

        Returns:
            SGPPropagator: Self with identity set.

        Example:
            ```python
            import brahe as bh
            import uuid

            line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
            line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
            my_uuid = str(uuid.uuid4())
            prop = bh.SGPPropagator.from_tle(line1, line2).with_identity("ISS", my_uuid, 25544)
            print(f"Name: {prop.name}, ID: {prop.id}, UUID: {prop.uuid}")
            ```
        """
        ...

    def with_name(self, name: str) -> SGPPropagator:
        """Set the name and return self (consuming constructor pattern).

        Args:
            name (str): Name to assign to this propagator.

        Returns:
            SGPPropagator: Self with name set.

        Example:
            ```python
            import brahe as bh

            line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
            line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
            prop = bh.SGPPropagator.from_tle(line1, line2).with_name("My Satellite")
            print(f"Name: {prop.name}")
            ```
        """
        ...

    def with_new_uuid(self) -> SGPPropagator:
        """Generate a new UUID, set it, and return self (consuming constructor pattern).

        Returns:
            SGPPropagator: Self with new UUID set.
        """
        ...

    def with_uuid(self, uuid_str: str) -> SGPPropagator:
        """Set the UUID and return self (consuming constructor pattern).

        Args:
            uuid_str (str): UUID string to assign to this propagator.

        Returns:
            SGPPropagator: Self with UUID set.
        """
        ...

    @property
    def current_epoch(self) -> Epoch:
        """Get current epoch.

        Returns:
            Epoch: Current propagator epoch.

        Example:
            ```python
            import brahe as bh

            line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
            line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
            propagator = bh.SGPPropagator.from_tle(line1, line2)
            propagator.step()
            print(f"Current epoch: {propagator.current_epoch}")
            ```
        """
        ...

    @property
    def epoch(self) -> Epoch:
        """Get TLE epoch.

        Returns:
            Epoch: Epoch of the TLE data.
        """
        ...

    @property
    def id(self) -> int:
        """Get the current numeric ID.

        Returns:
            int or None: The numeric ID, or None if not set.
        """
        ...

    @property
    def name(self) -> str:
        """Get the current name.

        Returns:
            str or None: The name, or None if not set.
        """
        ...

    @property
    def norad_id(self) -> int:
        """Get NORAD ID.

        Returns:
            int: NORAD catalog ID.
        """
        ...

    @property
    def satellite_name(self) -> str:
        """Get satellite name (if available).

        Returns:
            str or None: Satellite name if provided.

        Example:
            ```python
            import brahe as bh

            name = "ISS (ZARYA)"
            line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
            line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
            propagator = bh.SGPPropagator.from_3le(name, line1, line2)
            print(f"Satellite: {propagator.satellite_name}")
            ```
        """
        ...

    @property
    def step_size(self) -> float:
        """Get step size in seconds.

        Returns:
            float: Step size in seconds.

        Example:
            ```python
            import brahe as bh

            line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
            line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
            propagator = bh.SGPPropagator.from_tle(line1, line2)
            print(f"Step size: {propagator.step_size} seconds")
            ```
        """
        ...

    @property
    def trajectory(self) -> OrbitTrajectory:
        """Get accumulated trajectory.

        Returns:
            OrbitalTrajectory: The accumulated trajectory.

        Example:
            ```python
            import brahe as bh

            line1 = "1 25544U 98067A   21027.77992426  .00003336  00000-0  68893-4 0  9990"
            line2 = "2 25544  51.6461 339.8014 0002571  24.9690  60.4407 15.48919393267689"
            prop = bh.SGPPropagator.from_tle(line1, line2)
            prop.propagate_steps(100)
            traj = prop.trajectory
            print(f"Trajectory has {traj.len()} states")
            ```
        """
        ...

    @property
    def uuid(self) -> str:
        """Get the current UUID.

        Returns:
            str or None: The UUID as a string, or None if not set.
        """
        ...

class STrajectory6:
    """Static-dimension 6D trajectory container.

    Stores a sequence of 6-dimensional states at specific epochs with support
    for interpolation and automatic state eviction policies. Dimension is fixed
    at compile time for performance.
    """

    def __init__(self) -> None:
        """Initialize instance."""
        ...

    @classmethod
    def from_data(
        cls,
        epochs: list[Epoch],
        states: np.ndarray,
        interpolation_method: InterpolationMethod,
    ) -> STrajectory6:
        """Create a trajectory from existing data.

        Args:
            epochs (list[Epoch]): List of time epochs
            states (numpy.ndarray): Flattened 1D array of 6D state vectors with total
                length N*6 where N is the number of epochs
            interpolation_method (InterpolationMethod): Interpolation method (default Linear)

        Returns:
            STrajectory6: New 6D trajectory instance populated with data

        Example:
            ```python
            import brahe as bh
            import numpy as np

            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
            states = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0,
                               bh.R_EARTH + 510e3, 0.0, 0.0, 0.0, 7650.0, 0.0])
            traj = bh.STrajectory6.from_data([epc1, epc2], states)
            ```
        """
        ...

    def add(self, epoch: Epoch, state: np.ndarray) -> Any:
        """Add a state to the trajectory.

        Args:
            epoch (Epoch): Time of the state
            state (numpy.ndarray): 6-element state vector
        """
        ...

    def clear(self) -> Any:
        """Clear all states from the trajectory."""
        ...

    def dimension(self) -> int:
        """Get trajectory dimension (always 6).

        Returns:
            int: Dimension of the trajectory (always 6)
        """
        ...

    def epoch_at_idx(self, index: int) -> Epoch:
        """Get epoch at a specific index

        Arguments:
            index (int): Index of the epoch

        Returns:
            Epoch: Epoch at index

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.DTrajectory(6)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            retrieved_epc = traj.epoch_at_idx(0)
            ```
        """
        ...

    def first(self) -> Tuple:
        """Get the first (epoch, state) tuple in the trajectory, if any exists.

        Returns:
            tuple or None: Tuple of (Epoch, numpy.ndarray) for first state, or None if empty

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            first_epc, first_state = traj.first()
            ```
        """
        ...

    def get(self, index: int) -> Tuple:
        """Get both epoch and state at a specific index.

        Args:
            index (int): Index to retrieve

        Returns:
            tuple: Tuple of (Epoch, numpy.ndarray) for epoch and state at the index

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            ret_epc, ret_state = traj.get(0)
            ```
        """
        ...

    def get_eviction_policy(self) -> str:
        """Get current eviction policy.

        Returns:
            str: String representation of eviction policy

        Example:
            ```python
            import brahe as bh

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            policy = traj.get_eviction_policy()
            ```
        """
        ...

    def index_after_epoch(self, epoch: Epoch) -> int:
        """Get the index of the state at or after the given epoch.

        Args:
            epoch (Epoch): Target epoch

        Returns:
            int: Index of the state at or after the target epoch

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc1, state)
            epc2 = bh.Epoch.from_datetime(2024, 1, 1, 11, 59, 0.0, 0.0, bh.TimeSystem.UTC)
            index = traj.index_after_epoch(epc2)
            ```
        """
        ...

    def index_before_epoch(self, epoch: Epoch) -> int:
        """Get the index of the state at or before the given epoch.

        Args:
            epoch (Epoch): Target epoch

        Returns:
            int: Index of the state at or before the target epoch

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc1, state)
            epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
            index = traj.index_before_epoch(epc2)
            ```
        """
        ...

    def interpolate(self, epoch: Epoch) -> np.ndarray:
        """Interpolate state at a given epoch using the configured interpolation method.

        Args:
            epoch (Epoch): Target epoch

        Returns:
            numpy.ndarray: Interpolated state vector

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state1 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc1, state1)
            epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 2, 0.0, 0.0, bh.TimeSystem.UTC)
            state2 = np.array([bh.R_EARTH + 510e3, 0.0, 0.0, 0.0, 7650.0, 0.0])
            traj.add(epc2, state2)
            epc_mid = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
            state_interp = traj.interpolate(epc_mid)
            ```
        """
        ...

    def interpolate_linear(self, epoch: Epoch) -> np.ndarray:
        """Interpolate state at a given epoch using linear interpolation.

        Args:
            epoch (Epoch): Target epoch

        Returns:
            numpy.ndarray: Linearly interpolated state vector

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state1 = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc1, state1)
            epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 2, 0.0, 0.0, bh.TimeSystem.UTC)
            state2 = np.array([bh.R_EARTH + 510e3, 0.0, 0.0, 0.0, 7650.0, 0.0])
            traj.add(epc2, state2)
            epc_mid = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
            state_interp = traj.interpolate_linear(epc_mid)
            ```
        """
        ...

    def is_empty(self) -> bool:
        """Check if trajectory is empty.

        Returns:
            bool: True if trajectory contains no states, False otherwise

        Example:
            ```python
            import brahe as bh

            traj = bh.DTrajectory(6)
            print(f"Is empty: {traj.is_empty()}")
            ```
        """
        ...

    def last(self) -> Tuple:
        """Get the last (epoch, state) tuple in the trajectory, if any exists.

        Returns:
            tuple or None: Tuple of (Epoch, numpy.ndarray) for last state, or None if empty

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            last_epc, last_state = traj.last()
            ```
        """
        ...

    def len(self) -> int:
        """Get the number of states in the trajectory (alias for length).

        Returns:
            int: Number of states in the trajectory

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.DTrajectory(6)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            print(f"Number of states: {traj.len()}")
            ```
        """
        ...

    def nearest_state(self, epoch: Epoch) -> Tuple:
        """Get the nearest state to a given epoch.

        Args:
            epoch (Epoch): Target epoch

        Returns:
            tuple: Tuple of (Epoch, numpy.ndarray) containing the nearest state

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc1, state)
            epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 30.0, 0.0, bh.TimeSystem.UTC)
            nearest_epc, nearest_state = traj.nearest_state(epc2)
            ```
        """
        ...

    def remove(self, index: int) -> Tuple:
        """Remove a state at a specific index.

        Args:
            index (int): Index of the state to remove

        Returns:
            tuple: Tuple of (Epoch, numpy.ndarray) for the removed epoch and state

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            removed_epc, removed_state = traj.remove(0)
            ```
        """
        ...

    def remove_epoch(self, epoch: Epoch) -> np.ndarray:
        """Remove a state at a specific epoch.

        Args:
            epoch (Epoch): Epoch of the state to remove

        Returns:
            numpy.ndarray: The removed state vector

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            removed_state = traj.remove_epoch(epc)
            ```
        """
        ...

    def set_eviction_policy_max_age(self, max_age: float) -> Any:
        """Set maximum age for trajectory states.

        Args:
            max_age (float): Maximum age in seconds relative to most recent state
        """
        ...

    def set_eviction_policy_max_size(self, max_size: int) -> Any:
        """Set maximum trajectory size.

        Args:
            max_size (int): Maximum number of states to retain
        """
        ...

    def set_interpolation_method(self, method: InterpolationMethod) -> Any:
        """Set the interpolation method for the trajectory.

        Args:
            method (InterpolationMethod): New interpolation method

        Example:
            ```python
            import brahe as bh

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            traj.set_interpolation_method(bh.InterpolationMethod.LINEAR)
            ```
        """
        ...

    def state_after_epoch(self, epoch: Epoch) -> Tuple:
        """Get the state at or after the given epoch.

        Args:
            epoch (Epoch): Target epoch

        Returns:
            tuple: Tuple of (Epoch, numpy.ndarray) containing state at or after the target epoch

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc1, state)
            epc2 = bh.Epoch.from_datetime(2024, 1, 1, 11, 59, 0.0, 0.0, bh.TimeSystem.UTC)
            ret_epc, ret_state = traj.state_after_epoch(epc2)
            ```
        """
        ...

    def state_at_idx(self, index: int) -> np.ndarray:
        """Get state at a specific index

        Arguments:
            index (int): Index of the state

        Returns:
            numpy.ndarray: State vector at index

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.DTrajectory(6)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            retrieved_state = traj.state_at_idx(0)
            ```
        """
        ...

    def state_before_epoch(self, epoch: Epoch) -> Tuple:
        """Get the state at or before the given epoch.

        Args:
            epoch (Epoch): Target epoch

        Returns:
            tuple: Tuple of (Epoch, numpy.ndarray) containing state at or before the target epoch

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.OrbitTrajectory(bh.OrbitFrame.ECI, bh.OrbitRepresentation.CARTESIAN, None)
            epc1 = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc1, state)
            epc2 = bh.Epoch.from_datetime(2024, 1, 1, 12, 1, 0.0, 0.0, bh.TimeSystem.UTC)
            ret_epc, ret_state = traj.state_before_epoch(epc2)
            ```
        """
        ...

    def to_matrix(self) -> np.ndarray:
        """Get all states as a numpy array"""
        ...

    def with_eviction_policy_max_age(self, max_age: float) -> STrajectory6:
        """Set eviction policy to keep states within maximum age using builder pattern

        Arguments:
            max_age (float): Maximum age of states in seconds

        Returns:
            STrajectory6: Self with updated eviction policy
        """
        ...

    def with_eviction_policy_max_size(self, max_size: int) -> STrajectory6:
        """Set eviction policy to keep maximum number of states using builder pattern

        Arguments:
            max_size (int): Maximum number of states to retain

        Returns:
            STrajectory6: Self with updated eviction policy
        """
        ...

    def with_interpolation_method(
        self, interpolation_method: InterpolationMethod
    ) -> STrajectory6:
        """Set interpolation method using builder pattern

        Arguments:
            interpolation_method (InterpolationMethod): Interpolation method to use

        Returns:
            STrajectory6: Self with updated interpolation method
        """
        ...

    @property
    def end_epoch(self) -> Epoch:
        """Get end epoch of trajectory.

        Returns:
            Epoch or None: Last epoch if trajectory is not empty, None otherwise
        """
        ...

    @property
    def interpolation_method(self) -> InterpolationMethod:
        """Get interpolation method.

        Returns:
            InterpolationMethod: Current interpolation method
        """
        ...

    @property
    def length(self) -> int:
        """Get the number of states in the trajectory.

        Returns:
            int: Number of states in the trajectory

        Example:
            ```python
            import brahe as bh
            import numpy as np

            traj = bh.DTrajectory(6)
            epc = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0.0, 0.0, bh.TimeSystem.UTC)
            state = np.array([bh.R_EARTH + 500e3, 0.0, 0.0, 0.0, 7600.0, 0.0])
            traj.add(epc, state)
            print(f"Trajectory length: {traj.length}")
            ```
        """
        ...

    @property
    def start_epoch(self) -> Epoch:
        """Get start epoch of trajectory.

        Returns:
            Epoch or None: First epoch if trajectory is not empty, None otherwise
        """
        ...

    @property
    def time_span(self) -> float:
        """Get time span of trajectory in seconds.

        Returns:
            float or None: Time span between first and last epochs, or None if less than 2 states
        """
        ...

class StaticEOPProvider:
    """Static Earth Orientation Parameter provider with constant values.

    Provides EOP data using fixed values that don't change with time.
    Useful for testing or scenarios where time-varying EOP data is not needed.

    Example:
        ```python
        import brahe as bh

        # Create static EOP provider with default values
        eop = bh.StaticEOPProvider()

        # Create static EOP provider with zero values
        eop_zero = bh.StaticEOPProvider.from_zero()

        # Create with custom values
        eop_custom = bh.StaticEOPProvider.from_values(0.1, 0.0, 0.0, 0.0, 0.0, 0.0)

        # Set as global provider
        bh.set_global_eop_provider_from_static_provider(eop_custom)
        ```
    """

    def __init__(self) -> None:
        """Initialize instance."""
        ...

    @classmethod
    def from_values(
        cls, ut1_utc: float, pm_x: float, pm_y: float, dx: float, dy: float, lod: float
    ) -> StaticEOPProvider:
        """Create a static EOP provider with specified values.

        Args:
            ut1_utc (float): UT1-UTC time difference in seconds
            pm_x (float): Polar motion x-component in radians
            pm_y (float): Polar motion y-component in radians
            dx (float): Celestial pole offset dx in radians
            dy (float): Celestial pole offset dy in radians
            lod (float): Length of day offset in seconds

        Returns:
            StaticEOPProvider: Provider with specified EOP values

        Example:
            ```python
            import brahe as bh

            # Create EOP provider with custom values
            eop = bh.StaticEOPProvider.from_values(
                ut1_utc=0.1,
                pm_x=1e-6,
                pm_y=2e-6,
                dx=1e-7,
                dy=1e-7,
                lod=0.001
            )
            bh.set_global_eop_provider_from_static_provider(eop)
            ```
        """
        ...

    @classmethod
    def from_zero(cls) -> StaticEOPProvider:
        """Create a static EOP provider with all values set to zero.

        Returns:
            StaticEOPProvider: Provider with all EOP values set to zero

        Example:
            ```python
            import brahe as bh

            # Create EOP provider with all zeros (no corrections)
            eop = bh.StaticEOPProvider.from_zero()
            bh.set_global_eop_provider_from_static_provider(eop)
            ```
        """
        ...

    def eop_type(self) -> str:
        """Get the EOP data type.

        Returns:
            str: EOP type string

        Example:
            ```python
            import brahe as bh

            eop = bh.StaticEOPProvider.from_zero()
            print(f"EOP type: {eop.eop_type()}")
            ```
        """
        ...

    def extrapolation(self) -> str:
        """Get the extrapolation method.

        Returns:
            str: Extrapolation method string

        Example:
            ```python
            import brahe as bh

            eop = bh.StaticEOPProvider.from_zero()
            print(f"Extrapolation method: {eop.extrapolation()}")
            ```
        """
        ...

    def get_dxdy(self, mjd: float) -> tuple[float, float]:
        """Get celestial pole offsets for a given MJD.

        Args:
            mjd (float): Modified Julian Date

        Returns:
            tuple[float, float]: Celestial pole offsets dx and dy in radians

        Example:
            ```python
            import brahe as bh

            eop = bh.StaticEOPProvider.from_zero()
            dx, dy = eop.get_dxdy(58849.0)
            print(f"Celestial pole offsets: dx={dx} rad, dy={dy} rad")
            ```
        """
        ...

    def get_eop(self, mjd: float) -> tuple[float, float, float, float, float, float]:
        """Get all EOP parameters for a given MJD.

        Args:
            mjd (float): Modified Julian Date

        Returns:
            tuple[float, float, float, float, float, float]: UT1-UTC, pm_x, pm_y, dx, dy, lod

        Example:
            ```python
            import brahe as bh

            eop = bh.StaticEOPProvider()
            ut1_utc, pm_x, pm_y, dx, dy, lod = eop.get_eop(58849.0)
            print(f"EOP: UT1-UTC={ut1_utc}s, PM=({pm_x},{pm_y})rad")
            ```
        """
        ...

    def get_lod(self, mjd: float) -> float:
        """Get length of day offset for a given MJD.

        Args:
            mjd (float): Modified Julian Date

        Returns:
            float: Length of day offset in seconds

        Example:
            ```python
            import brahe as bh

            eop = bh.StaticEOPProvider.from_zero()
            lod = eop.get_lod(58849.0)
            print(f"Length of day offset: {lod} seconds")
            ```
        """
        ...

    def get_pm(self, mjd: float) -> tuple[float, float]:
        """Get polar motion components for a given MJD.

        Args:
            mjd (float): Modified Julian Date

        Returns:
            tuple[float, float]: Polar motion x and y components in radians

        Example:
            ```python
            import brahe as bh

            eop = bh.StaticEOPProvider.from_zero()
            pm_x, pm_y = eop.get_pm(58849.0)
            print(f"Polar motion: x={pm_x} rad, y={pm_y} rad")
            ```
        """
        ...

    def get_ut1_utc(self, mjd: float) -> float:
        """Get UT1-UTC time difference for a given MJD.

        Args:
            mjd (float): Modified Julian Date

        Returns:
            float: UT1-UTC time difference in seconds

        Example:
            ```python
            import brahe as bh

            eop = bh.StaticEOPProvider.from_zero()
            ut1_utc = eop.get_ut1_utc(58849.0)
            print(f"UT1-UTC: {ut1_utc} seconds")
            ```
        """
        ...

    def interpolation(self) -> bool:
        """Check if interpolation is enabled.

        Returns:
            bool: True if interpolation is enabled

        Example:
            ```python
            import brahe as bh

            eop = bh.StaticEOPProvider.from_zero()
            print(f"Interpolation enabled: {eop.interpolation()}")
            ```
        """
        ...

    def is_initialized(self) -> bool:
        """Check if the provider is initialized.

        Returns:
            bool: True if initialized

        Example:
            ```python
            import brahe as bh

            eop = bh.StaticEOPProvider.from_zero()
            print(f"Is initialized: {eop.is_initialized()}")
            ```
        """
        ...

    def len(self) -> int:
        """Get the number of EOP data points.

        Returns:
            int: Number of EOP data points

        Example:
            ```python
            import brahe as bh

            eop = bh.StaticEOPProvider.from_zero()
            print(f"EOP data points: {eop.len()}")
            ```
        """
        ...

    def mjd_last_dxdy(self) -> float:
        """Get the last Modified Julian Date with dx/dy data.

        Returns:
            float: Last MJD with dx/dy data

        Example:
            ```python
            import brahe as bh

            eop = bh.StaticEOPProvider.from_zero()
            print(f"Last MJD with dx/dy: {eop.mjd_last_dxdy()}")
            ```
        """
        ...

    def mjd_last_lod(self) -> float:
        """Get the last Modified Julian Date with LOD data.

        Returns:
            float: Last MJD with LOD data

        Example:
            ```python
            import brahe as bh

            eop = bh.StaticEOPProvider.from_zero()
            print(f"Last MJD with LOD: {eop.mjd_last_lod()}")
            ```
        """
        ...

    def mjd_max(self) -> float:
        """Get the maximum Modified Julian Date in the dataset.

        Returns:
            float: Maximum MJD

        Example:
            ```python
            import brahe as bh

            eop = bh.StaticEOPProvider.from_zero()
            print(f"Maximum MJD: {eop.mjd_max()}")
            ```
        """
        ...

    def mjd_min(self) -> float:
        """Get the minimum Modified Julian Date in the dataset.

        Returns:
            float: Minimum MJD

        Example:
            ```python
            import brahe as bh

            eop = bh.StaticEOPProvider.from_zero()
            print(f"Minimum MJD: {eop.mjd_min()}")
            ```
        """
        ...

class TimeRange:
    """Iterator that generates a sequence of epochs over a time range.

    TimeRange creates an iterator that yields epochs from a start time to an end time
    with a specified step size in seconds. This is useful for propagating orbits,
    sampling trajectories, or generating time grids for analysis.

    Args:
        epoch_start (Epoch): Starting epoch for the range
        epoch_end (Epoch): Ending epoch for the range
        step (float): Time step in seconds between consecutive epochs

    Examples:
        >>> from brahe import Epoch, TimeRange, TimeSystem
        >>> start = Epoch.from_datetime(2024, 1, 1, 0, 0, 0.0, 0.0, TimeSystem.UTC)
        >>> end = start + 3600.0  # One hour later
        >>> time_range = TimeRange(start, end, 60.0)  # 60-second steps
        >>> for epoch in time_range:
        ...     print(epoch)
    """

    def __init__(self, epoch_start: Epoch, epoch_end: Epoch, step: float) -> None:
        """Initialize instance."""
        ...

class TimeSystem:
    """Enumeration of supported time systems.

    Time systems define different conventions for measuring and representing time.
    Each system has specific uses in astrodynamics and timekeeping applications.
    """

    def __init__(self) -> None:
        """Initialize instance."""
        ...

    @property
    def GPS(self) -> Any:
        """Enumeration of supported time systems.

        Time systems define different conventions for measuring and representing time.
        Each system has specific uses in astrodynamics and timekeeping applications.
        """
        ...

    @property
    def TAI(self) -> Any:
        """Enumeration of supported time systems.

        Time systems define different conventions for measuring and representing time.
        Each system has specific uses in astrodynamics and timekeeping applications.
        """
        ...

    @property
    def TT(self) -> Any:
        """Enumeration of supported time systems.

        Time systems define different conventions for measuring and representing time.
        Each system has specific uses in astrodynamics and timekeeping applications.
        """
        ...

    @property
    def UT1(self) -> Any:
        """Enumeration of supported time systems.

        Time systems define different conventions for measuring and representing time.
        Each system has specific uses in astrodynamics and timekeeping applications.
        """
        ...

    @property
    def UTC(self) -> Any:
        """Enumeration of supported time systems.

        Time systems define different conventions for measuring and representing time.
        Each system has specific uses in astrodynamics and timekeeping applications.
        """
        ...

# Functions

def anomaly_eccentric_to_mean(
    anm_ecc: float, e: float, angle_format: AngleFormat
) -> float:
    """Converts eccentric anomaly into mean anomaly.

    Args:
        anm_ecc (float): Eccentric anomaly in radians or degrees.
        e (float): The eccentricity of the astronomical object's orbit (dimensionless).
        angle_format (AngleFormat): Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.

    Returns:
        float: Mean anomaly in radians or degrees.

    Example:
        ```python
        import brahe as bh
        import math

        # Convert eccentric to mean anomaly
        E = math.pi / 4  # 45 degrees eccentric anomaly
        e = 0.1  # eccentricity
        M = bh.anomaly_eccentric_to_mean(E, e, bh.AngleFormat.RADIANS)
        print(f"Mean anomaly: {M:.4f} radians")
        ```
    """
    ...

def anomaly_eccentric_to_true(
    anm_ecc: float, e: float, angle_format: AngleFormat
) -> float:
    """Converts eccentric anomaly into true anomaly.

    Args:
        anm_ecc (float): Eccentric anomaly in radians or degrees.
        e (float): The eccentricity of the astronomical object's orbit (dimensionless).
        angle_format (AngleFormat): Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.

    Returns:
        float: True anomaly in radians or degrees.

    Example:
        ```python
        import brahe as bh
        import math

        # Convert eccentric to true anomaly
        E = math.pi / 4  # 45 degrees eccentric anomaly
        e = 0.4  # eccentricity
        nu = bh.anomaly_eccentric_to_true(E, e, bh.AngleFormat.RADIANS)
        print(f"True anomaly: {nu:.4f} radians")
        ```
    """
    ...

def anomaly_mean_to_eccentric(
    anm_mean: float, e: float, angle_format: AngleFormat
) -> float:
    """Converts mean anomaly into eccentric anomaly.

    Args:
        anm_mean (float): Mean anomaly in radians or degrees.
        e (float): The eccentricity of the astronomical object's orbit (dimensionless).
        angle_format (AngleFormat): Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.

    Returns:
        float: Eccentric anomaly in radians or degrees.

    Example:
        ```python
        import brahe as bh

        # Convert mean to eccentric anomaly (solves Kepler's equation)
        M = 1.5  # mean anomaly in radians
        e = 0.3  # eccentricity
        E = bh.anomaly_mean_to_eccentric(M, e, bh.AngleFormat.RADIANS)
        print(f"Eccentric anomaly: {E:.4f} radians")
        ```
    """
    ...

def anomaly_mean_to_true(anm_mean: float, e: float, angle_format: AngleFormat) -> float:
    """Converts mean anomaly into true anomaly.

    Args:
        anm_mean (float): Mean anomaly in radians or degrees.
        e (float): The eccentricity of the astronomical object's orbit (dimensionless).
        angle_format (AngleFormat): Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.

    Returns:
        float: True anomaly in radians or degrees.

    Example:
        ```python
        import brahe as bh

        # Convert mean to true anomaly (combines Kepler's equation + eccentric anomaly conversion)
        M = 2.0  # mean anomaly in radians
        e = 0.25  # eccentricity
        nu = bh.anomaly_mean_to_true(M, e, bh.AngleFormat.RADIANS)
        print(f"True anomaly: {nu:.4f} radians")
        ```
    """
    ...

def anomaly_true_to_eccentric(
    anm_true: float, e: float, angle_format: AngleFormat
) -> float:
    """Converts true anomaly into eccentric anomaly.

    Args:
        anm_true (float): True anomaly in radians or degrees.
        e (float): The eccentricity of the astronomical object's orbit (dimensionless).
        angle_format (AngleFormat): Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.

    Returns:
        float: Eccentric anomaly in radians or degrees.

    Example:
        ```python
        import brahe as bh
        import math

        # Convert true to eccentric anomaly
        nu = math.pi / 3  # 60 degrees true anomaly
        e = 0.2  # eccentricity
        E = bh.anomaly_true_to_eccentric(nu, e, bh.AngleFormat.RADIANS)
        print(f"Eccentric anomaly: {E:.4f} radians")
        ```
    """
    ...

def anomaly_true_to_mean(anm_true: float, e: float, angle_format: AngleFormat) -> float:
    """Converts true anomaly into mean anomaly.

    Args:
        anm_true (float): True anomaly in radians or degrees.
        e (float): The eccentricity of the astronomical object's orbit (dimensionless).
        angle_format (AngleFormat): Interprets input and returns output in AngleFormat.DEGREES or AngleFormat.RADIANS.

    Returns:
        float: Mean anomaly in radians or degrees.

    Example:
        ```python
        import brahe as bh
        import math

        # Convert true to mean anomaly
        nu = math.pi / 2  # 90 degrees true anomaly
        e = 0.15  # eccentricity
        M = bh.anomaly_true_to_mean(nu, e, bh.AngleFormat.RADIANS)
        print(f"Mean anomaly: {M:.4f} radians")
        ```
    """
    ...

def apoapsis_distance(a: float, e: float) -> float:
    """Calculate the distance of an object at its apoapsis.

    Args:
        a (float): The semi-major axis of the astronomical object in meters.
        e (float): The eccentricity of the astronomical object's orbit (dimensionless).

    Returns:
        float: The distance of the object at apoapsis in meters.

    Example:
        ```python
        import brahe as bh

        # Calculate apoapsis distance
        a = 8000000.0  # 8000 km semi-major axis
        e = 0.2  # moderate eccentricity
        r_apo = bh.apoapsis_distance(a, e)
        print(f"Apoapsis distance: {r_apo/1000:.2f} km")
        ```
    """
    ...

def apoapsis_velocity(a: float, e: float, gm: float) -> float:
    """Computes the apoapsis velocity of an astronomical object around a general body.

    Args:
        a (float): The semi-major axis of the astronomical object in meters.
        e (float): The eccentricity of the astronomical object's orbit (dimensionless).
        gm (float): The standard gravitational parameter of primary body in m³/s².

    Returns:
        float: The magnitude of velocity of the object at apoapsis in m/s.

    Example:
        ```python
        import brahe as bh

        # Calculate apoapsis velocity for a Martian satellite
        a = 10000000.0  # 10000 km semi-major axis
        e = 0.3
        v_apo = bh.apoapsis_velocity(a, e, bh.GM_MARS)
        print(f"Apoapsis velocity: {v_apo/1000:.2f} km/s")
        ```
    """
    ...

def apogee_velocity(a: float, e: float) -> float:
    """Computes the apogee velocity of an astronomical object around Earth.

    Args:
        a (float): The semi-major axis of the astronomical object in meters.
        e (float): The eccentricity of the astronomical object's orbit (dimensionless).

    Returns:
        float: The magnitude of velocity of the object at apogee in m/s.

    Example:
        ```python
        import brahe as bh

        # Calculate apogee velocity for GTO (Geostationary Transfer Orbit)
        a = 24400000.0  # meters
        e = 0.73  # high eccentricity
        v_apo = bh.apogee_velocity(a, e)
        print(f"Apogee velocity: {v_apo:.2f} m/s")
        ```
    """
    ...

def bias_precession_nutation(epc: Epoch) -> Any:
    """Computes the Bias-Precession-Nutation matrix transforming the `GCRS` to the
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
    ...

def calculate_tle_line_checksum(line: str) -> int:
    """Calculate TLE line checksum.

    Args:
        line (str): TLE line.

    Returns:
        int: Checksum value.
    """
    ...

def celestrak_download_ephemeris(
    group: str, filepath: str, content_format: str, file_format: str
) -> Any:
    """Download satellite ephemeris from CelesTrak and save to file

    Downloads 3LE data from CelesTrak and serializes to the specified file format.
    The file can contain either 2-line elements (TLE, without names) or 3-line elements
    (3LE, with satellite names), and can be saved as plain text, CSV, or JSON.

    Args:
        group (str): Satellite group name (e.g., "active", "stations", "gnss", "last-30-days").
        filepath (str): Output file path. Parent directories will be created if needed.
        content_format (str): Content format - "tle" (2-line without names) or "3le" (3-line with names).
        file_format (str): File format - "txt" (plain text), "csv" (comma-separated), or "json" (JSON array).

    Raises:
        RuntimeError: If download fails, format is invalid, or file cannot be written.

    Example:
        ```python
        import brahe as bh

        # Download GNSS satellites as 3LE in JSON format
        bh.datasets.celestrak.download_ephemeris("gnss", "gnss_sats.json", "3le", "json")

        # Download active satellites as 2LE in plain text
        bh.datasets.celestrak.download_ephemeris("active", "active.txt", "tle", "txt")

        # Download stations as 3LE in CSV format
        bh.datasets.celestrak.download_ephemeris("stations", "stations.csv", "3le", "csv")
        ```
    """
    ...

def celestrak_get_ephemeris(group: str) -> list[tuple[str, str, str]]:
    """Get satellite ephemeris data from CelesTrak

    Downloads and parses 3LE (three-line element) data for the specified satellite group
    from CelesTrak (https://celestrak.org).

    Args:
        group (str): Satellite group name (e.g., "active", "stations", "gnss", "last-30-days").
            See https://celestrak.org/NORAD/elements/ for available groups.

    Returns:
        list[tuple[str, str, str]]: List of (name, line1, line2) tuples containing satellite
            names and TLE lines.

    Raises:
        RuntimeError: If download fails or data cannot be parsed.

    Example:
        ```python
        import brahe as bh

        # Download ephemeris for ground stations
        ephemeris = bh.datasets.celestrak.get_ephemeris("stations")

        # Print first 5 satellites
        for name, line1, line2 in ephemeris[:5]:
            print(f"Satellite: {name}")
            print(f"  Line 1: {line1[:20]}...")
        ```
    """
    ...

def celestrak_get_ephemeris_as_propagators(
    group: str, step_size: float
) -> list[SGPPropagator]:
    """Get satellite ephemeris as SGP propagators from CelesTrak

    Downloads and parses 3LE data from CelesTrak, then creates SGP4/SDP4 propagators
    for each satellite. This is a convenient way to get ready-to-use propagators.

    Args:
        group (str): Satellite group name (e.g., "active", "stations", "gnss", "last-30-days").
        step_size (float): Default step size for propagators in seconds.

    Returns:
        list[SGPPropagator]: List of configured SGP propagators (PySGPPropagator), one per satellite.

    Raises:
        RuntimeError: If download fails or no valid propagators can be created.

    Note:
        Satellites with invalid TLE data will be skipped with a warning printed to stderr.
        The function will only raise an error if NO valid propagators can be created.

    Example:
        ```python
        import brahe as bh

        # Get propagators for GNSS satellites with 60-second step size
        propagators = bh.datasets.celestrak.get_ephemeris_as_propagators("gnss", 60.0)
        print(f"Loaded {len(propagators)} GNSS satellites")

        # Propagate first satellite
        epoch = bh.Epoch.from_datetime(2024, 1, 1, 12, 0, 0, tsys="UTC")
        state = propagators[0].propagate(epoch)
        ```
    """
    ...

def create_tle_lines(
    epoch: Epoch,
    inclination: float,
    raan: float,
    eccentricity: float,
    arg_perigee: float,
    mean_anomaly: float,
    mean_motion: float,
    norad_id: str,
    ephemeris_type: int,
    element_set_number: int,
    revolution_number: int,
    classification: str = ...,
    intl_designator: str = ...,
    first_derivative: float = ...,
    second_derivative: float = ...,
    bstar: float = ...,
) -> Tuple:
    """Create complete TLE lines from all parameters.

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
    ...

def datetime_to_jd(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: float,
    nanosecond: float,
) -> float:
    """Convert a Gregorian calendar date to the equivalent Julian Date.

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
    ...

def datetime_to_mjd(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: float,
    nanosecond: float,
) -> float:
    """Convert a Gregorian calendar date to the equivalent Modified Julian Date.

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
    ...

def download_c04_eop_file(filepath: str) -> Any:
    """Download latest C04 Earth orientation parameter file. Will attempt to download the latest
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
    ...

def download_standard_eop_file(filepath: str) -> Any:
    """Download latest standard Earth orientation parameter file. Will attempt to download the latest
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
    ...

def earth_rotation(epc: Epoch) -> Any:
    """Computes the Earth rotation matrix transforming the `CIRS` to the `TIRS`
    intermediate reference frame. This transformation corrects for the Earth
    rotation.

    Args:
        epc (Epoch): Epoch instant for computation of transformation matrix

    Returns:
        (numpy.ndarray): 3x3 rotation matrix transforming `CIRS` -> `TIRS`
    """
    ...

def epoch_from_tle(line1: str) -> Epoch:
    """Extract Epoch from TLE line 1

    Extracts and parses the epoch timestamp from the first line of TLE data.
    The epoch is returned in UTC time system.

    Args:
        line1 (str): First line of TLE data

    Returns:
        Epoch: Extracted epoch in UTC time system

    Examples:
        >>> line1 = "1 25544U 98067A   21001.50000000  .00001764  00000-0  40967-4 0  9997"
        >>> epoch = epoch_from_tle(line1)
        >>> epoch.year()
        2021
    """
    ...

def get_global_dxdy(mjd: float) -> tuple[float, float]:
    """Get celestial pole offsets from the global EOP provider.

    Args:
        mjd (float): Modified Julian Date

    Returns:
        tuple[float, float]: Celestial pole offsets dx and dy in radians
    """
    ...

def get_global_eop(mjd: float) -> tuple[float, float, float, float, float, float]:
    """Get all EOP parameters from the global EOP provider.

    Args:
        mjd (float): Modified Julian Date

    Returns:
        tuple[float, float, float, float, float, float]: UT1-UTC, pm_x, pm_y, dx, dy, lod
    """
    ...

def get_global_eop_extrapolation() -> str:
    """Get the extrapolation method of the global EOP provider.

    Returns:
        str: Extrapolation method string
    """
    ...

def get_global_eop_initialization() -> bool:
    """Check if the global EOP provider is initialized.

    Returns:
        bool: True if global EOP provider is initialized
    """
    ...

def get_global_eop_interpolation() -> bool:
    """Check if interpolation is enabled in the global EOP provider.

    Returns:
        bool: True if interpolation is enabled
    """
    ...

def get_global_eop_len() -> int:
    """Get the number of EOP data points in the global provider.

    Returns:
        int: Number of EOP data points
    """
    ...

def get_global_eop_mjd_last_dxdy() -> float:
    """Get the last Modified Julian Date with dx/dy data in the global provider.

    Returns:
        float: Last MJD with dx/dy data
    """
    ...

def get_global_eop_mjd_last_lod() -> float:
    """Get the last Modified Julian Date with LOD data in the global provider.

    Returns:
        float: Last MJD with LOD data
    """
    ...

def get_global_eop_mjd_max() -> float:
    """Get the maximum Modified Julian Date in the global EOP dataset.

    Returns:
        float: Maximum MJD
    """
    ...

def get_global_eop_mjd_min() -> float:
    """Get the minimum Modified Julian Date in the global EOP dataset.

    Returns:
        float: Minimum MJD
    """
    ...

def get_global_eop_type() -> str:
    """Get the EOP data type of the global provider.

    Returns:
        str: EOP type string
    """
    ...

def get_global_lod(mjd: float) -> float:
    """Get length of day offset from the global EOP provider.

    Args:
        mjd (float): Modified Julian Date

    Returns:
        float: Length of day offset in seconds
    """
    ...

def get_global_pm(mjd: float) -> tuple[float, float]:
    """Get polar motion components from the global EOP provider.

    Args:
        mjd (float): Modified Julian Date

    Returns:
        tuple[float, float]: Polar motion x and y components in radians
    """
    ...

def get_global_ut1_utc(mjd: float) -> float:
    """Get UT1-UTC time difference from the global EOP provider.

    Args:
        mjd (float): Modified Julian Date

    Returns:
        float: UT1-UTC time difference in seconds
    """
    ...

def jd_to_datetime(jd: float) -> Tuple:
    """Convert a Julian Date to the equivalent Gregorian calendar date.

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
    ...

def keplerian_elements_from_tle(line1: str, line2: str) -> Tuple:
    """Extract Keplerian orbital elements from TLE lines.

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
    ...

def keplerian_elements_to_tle(
    epoch: Epoch, elements: np.ndarray, norad_id: str
) -> Tuple:
    """Convert Keplerian elements to TLE lines.

    Converts standard Keplerian orbital elements to Two-Line Element (TLE) format.
    Input angles should be in degrees for compatibility with TLE format.

    Args:
        epoch (Epoch): Epoch of the elements.
        elements (numpy.ndarray): Keplerian elements [a (m), e, i (deg), raan (deg), argp (deg), M (deg)].
        norad_id (str): NORAD catalog number (supports numeric and Alpha-5 format).

    Returns:
        tuple: A tuple containing (line1, line2) - the two TLE lines as strings.
    """
    ...

def location_accesses(
    locations: PointLocation | PolygonLocation | List[PointLocation | PolygonLocation],
    propagators: SGPPropagator
    | KeplerianPropagator
    | List[SGPPropagator | KeplerianPropagator],
    search_start: Epoch,
    search_end: Epoch,
    constraint: AccessConstraint,
    property_computers: [List[AccessPropertyComputer]] = ...,
    config: [AccessSearchConfig] = ...,
    time_tolerance: [float] = ...,
) -> List[AccessWindow]:
    """Compute access windows for locations and satellites.

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
    ...

def mean_motion(a: float, angle_format: AngleFormat) -> float:
    """Computes the mean motion of an astronomical object around Earth.

    Args:
        a (float): The semi-major axis of the astronomical object in meters.
        angle_format (AngleFormat): Return output in AngleFormat.DEGREES or AngleFormat.RADIANS.

    Returns:
        float: The mean motion of the astronomical object in radians or degrees.

    Example:
        ```python
        import brahe as bh

        # Calculate mean motion for geostationary orbit (35786 km altitude)
        a = bh.R_EARTH + 35786e3
        n = bh.mean_motion(a, bh.AngleFormat.DEGREES)
        print(f"Mean motion: {n:.6f} deg/s")
        ```
    """
    ...

def mean_motion_general(a: float, gm: float, angle_format: AngleFormat) -> float:
    """Computes the mean motion of an astronomical object around a general body
    given a semi-major axis.

    Args:
        a (float): The semi-major axis of the astronomical object in meters.
        gm (float): The standard gravitational parameter of primary body in m³/s².
        angle_format (AngleFormat): Return output in AngleFormat.DEGREES or AngleFormat.RADIANS.

    Returns:
        float: The mean motion of the astronomical object in radians or degrees.

    Example:
        ```python
        import brahe as bh

        # Calculate mean motion for a Mars orbiter
        a = 4000000.0  # 4000 km semi-major axis
        n = bh.mean_motion_general(a, bh.GM_MARS, bh.AngleFormat.RADIANS)
        print(f"Mean motion: {n:.6f} rad/s")
        ```
    """
    ...

def mjd_to_datetime(mjd: float) -> Tuple:
    """Convert a Modified Julian Date to the equivalent Gregorian calendar date.

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
    ...

def norad_id_alpha5_to_numeric(alpha5_id: str) -> int:
    """Convert Alpha-5 NORAD ID to numeric format.

    Args:
        alpha5_id (str): Alpha-5 format ID (e.g., "A0001").

    Returns:
        int: Numeric NORAD ID.
    """
    ...

def norad_id_numeric_to_alpha5(norad_id: int) -> str:
    """Convert numeric NORAD ID to Alpha-5 format.

    Args:
        norad_id (int): Numeric NORAD ID (100000-339999).

    Returns:
        str: Alpha-5 format ID (e.g., "A0001").
    """
    ...

def orbital_period(a: float) -> Any:
    """Computes the orbital period of an object around Earth.

    Uses rastro.constants.GM_EARTH as the standard gravitational parameter for the calculation.

    Args:
        a (float): The semi-major axis of the astronomical object in meters.

    Returns:
        (float): The orbital period of the astronomical object in seconds.

    Example:
        ```python
        import brahe as bh

        # Calculate orbital period for ISS-like orbit (400 km altitude)
        a = bh.R_EARTH + 400e3
        period = bh.orbital_period(a)
        print(f"Orbital period: {period/60:.2f} minutes")
        ```
    """
    ...

def orbital_period_from_state(state_eci: np.ndarray, gm: float) -> float:
    """Computes orbital period from an ECI state vector using the vis-viva equation.

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
    ...

def orbital_period_general(a: float, gm: float) -> float:
    """Computes the orbital period of an astronomical object around a general body.

    Args:
        a (float): The semi-major axis of the astronomical object in meters.
        gm (float): The standard gravitational parameter of primary body in m³/s².

    Returns:
        float: The orbital period of the astronomical object in seconds.

    Example:
        ```python
        import brahe as bh

        # Calculate orbital period around the Moon
        a = 1900000.0  # 1900 km semi-major axis
        period = bh.orbital_period_general(a, bh.GM_MOON)
        print(f"Lunar orbital period: {period/3600:.2f} hours")
        ```
    """
    ...

def parse_norad_id(norad_str: str) -> int:
    """Parse NORAD ID from string, handling both classic and Alpha-5 formats.

    Args:
        norad_str (str): NORAD ID string from TLE.

    Returns:
        int: Parsed numeric NORAD ID.
    """
    ...

def periapsis_distance(a: float, e: float) -> float:
    """Calculate the distance of an object at its periapsis.

    Args:
        a (float): The semi-major axis of the astronomical object in meters.
        e (float): The eccentricity of the astronomical object's orbit (dimensionless).

    Returns:
        float: The distance of the object at periapsis in meters.

    Example:
        ```python
        import brahe as bh

        # Calculate periapsis distance for an elliptical orbit
        a = 8000000.0  # 8000 km semi-major axis
        e = 0.2  # moderate eccentricity
        r_peri = bh.periapsis_distance(a, e)
        print(f"Periapsis distance: {r_peri/1000:.2f} km")
        ```
    """
    ...

def periapsis_velocity(a: float, e: float, gm: float) -> float:
    """Computes the periapsis velocity of an astronomical object around a general body.

    Args:
        a (float): The semi-major axis of the astronomical object in meters.
        e (float): The eccentricity of the astronomical object's orbit (dimensionless).
        gm (float): The standard gravitational parameter of primary body in m³/s².

    Returns:
        float: The magnitude of velocity of the object at periapsis in m/s.

    Example:
        ```python
        import brahe as bh

        # Calculate periapsis velocity for a comet around the Sun
        a = 5e11  # 5 AU semi-major axis (meters)
        e = 0.95  # highly elliptical
        v_peri = bh.periapsis_velocity(a, e, bh.GM_SUN)
        print(f"Periapsis velocity: {v_peri/1000:.2f} km/s")
        ```
    """
    ...

def perigee_velocity(a: float, e: float) -> float:
    """Computes the perigee velocity of an astronomical object around Earth.

    Args:
        a (float): The semi-major axis of the astronomical object in meters.
        e (float): The eccentricity of the astronomical object's orbit (dimensionless).

    Returns:
        float: The magnitude of velocity of the object at perigee in m/s.

    Example:
        ```python
        import brahe as bh

        # Calculate perigee velocity for Molniya orbit (highly elliptical)
        a = 26554000.0  # meters
        e = 0.72  # high eccentricity
        v_peri = bh.perigee_velocity(a, e)
        print(f"Perigee velocity: {v_peri:.2f} m/s")
        ```
    """
    ...

def polar_motion(epc: Epoch) -> Any:
    """Computes the Earth rotation matrix transforming the `TIRS` to the `ITRF` reference
    frame.

    The function will utilize the global Earth orientation and loaded data to
    apply corrections to compute the polar motion correction based on empirical
    observations of polar motion drift.

    Args:
        epc (Epoch): Epoch instant for computation of transformation matrix

    Returns:
        (numpy.ndarray): 3x3 rotation matrix transforming `TIRS` -> `ITRF`
    """
    ...

def position_ecef_to_eci(epc: Epoch, x: np.ndarray) -> Any:
    """Transforms a position vector from the Earth Centered Earth Fixed (`ECEF`/`ITRF`)
    frame to the Earth Centered Inertial (`ECI`/`GCRF`) frame.

    Applies the full `IAU 2006/2000A` transformation including bias, precession,
    nutation, Earth rotation, and polar motion corrections using global Earth
    orientation parameters.

    Args:
        epc (Epoch): Epoch instant for the transformation
        x (numpy.ndarray): Position vector in `ECEF` frame (m), shape `(3,)`

    Returns:
        (numpy.ndarray): Position vector in `ECI` frame (m), shape `(3,)`

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
    ...

def position_ecef_to_geocentric(x_ecef: np.ndarray, angle_format: AngleFormat) -> Any:
    """Convert `ECEF` Cartesian position to geocentric coordinates.

    Transforms a position from Earth-Centered Earth-Fixed (`ECEF`) Cartesian coordinates
    to geocentric spherical coordinates (latitude, longitude, radius).

    Args:
        x_ecef (numpy.ndarray): `ECEF` Cartesian position `[x, y, z]` in meters.
        angle_format (AngleFormat): Angle format for output angular coordinates (`RADIANS` or `DEGREES`).

    Returns:
        (numpy.ndarray): Geocentric position `[latitude, longitude, radius]` where latitude
            is in radians or degrees, longitude is in radians or degrees, and radius is in meters.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Convert ECEF to geocentric coordinates
        x_ecef = np.array([6378137.0, 0.0, 0.0])  # Point on equator, prime meridian
        x_geoc = bh.position_ecef_to_geocentric(x_ecef, bh.AngleFormat.DEGREES)
        print(f"Geocentric: lat={x_geoc[0]:.2f}°, lon={x_geoc[1]:.2f}°, r={x_geoc[2]:.0f}m")
        ```
    """
    ...

def position_ecef_to_geodetic(x_ecef: np.ndarray, angle_format: AngleFormat) -> Any:
    """Convert `ECEF` Cartesian position to geodetic coordinates.

    Transforms a position from Earth-Centered Earth-Fixed (`ECEF`) Cartesian coordinates
    to geodetic coordinates (latitude, longitude, altitude) using the `WGS84` ellipsoid model.

    Args:
        x_ecef (numpy.ndarray): `ECEF` Cartesian position `[x, y, z]` in meters.
        angle_format (AngleFormat): Angle format for output angular coordinates (`RADIANS` or `DEGREES`).

    Returns:
        (numpy.ndarray): Geodetic position `[latitude, longitude, altitude]` where latitude
            is in radians or degrees, longitude is in radians or degrees, and altitude
            is in meters above the `WGS84` ellipsoid.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Convert ECEF to geodetic coordinates (GPS-like)
        x_ecef = np.array([-1275936.0, -4797210.0, 4020109.0])  # Example location
        x_geod = bh.position_ecef_to_geodetic(x_ecef, bh.AngleFormat.DEGREES)
        print(f"Geodetic: lat={x_geod[0]:.4f}°, lon={x_geod[1]:.4f}°, alt={x_geod[2]:.0f}m")
        ```
    """
    ...

def position_eci_to_ecef(epc: Epoch, x: np.ndarray) -> Any:
    """Transforms a position vector from the Earth Centered Inertial (`ECI`/`GCRF`) frame
    to the Earth Centered Earth Fixed (`ECEF`/`ITRF`) frame.

    Applies the full `IAU 2006/2000A` transformation including bias, precession,
    nutation, Earth rotation, and polar motion corrections using global Earth
    orientation parameters.

    Args:
        epc (Epoch): Epoch instant for the transformation
        x (numpy.ndarray): Position vector in `ECI` frame (m), shape `(3,)`

    Returns:
        (numpy.ndarray): Position vector in `ECEF` frame (m), shape `(3,)`

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
    ...

def position_enz_to_azel(x_enz: np.ndarray, angle_format: AngleFormat) -> Any:
    """Convert position from East-North-Up (`ENZ`) frame to azimuth-elevation-range.

    Transforms a position from the local East-North-Up (`ENZ`) topocentric frame to
    azimuth-elevation-range spherical coordinates.

    Args:
        x_enz (numpy.ndarray): Position in `ENZ` frame `[east, north, up]` in meters.
        angle_format (AngleFormat): Angle format for output angular coordinates (`RADIANS` or `DEGREES`).

    Returns:
        (numpy.ndarray): Azimuth-elevation-range `[azimuth, elevation, range]` where azimuth
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
    ...

def position_geocentric_to_ecef(x_geoc: np.ndarray, angle_format: AngleFormat) -> Any:
    """Convert geocentric position to `ECEF` Cartesian coordinates.

    Transforms a position from geocentric spherical coordinates (latitude, longitude, radius)
    to Earth-Centered Earth-Fixed (`ECEF`) Cartesian coordinates.

    Args:
        x_geoc (numpy.ndarray): Geocentric position `[latitude, longitude, radius]` where
            latitude is in radians or degrees, longitude is in radians or degrees, and
            radius is in meters.
        angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).

    Returns:
        (numpy.ndarray): `ECEF` Cartesian position `[x, y, z]` in meters.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Convert geocentric coordinates to ECEF
        lat, lon, r = 0.0, 0.0, 6378137.0  # Equator, prime meridian, Earth's radius
        x_geoc = np.array([lat, lon, r])
        x_ecef = bh.position_geocentric_to_ecef(x_geoc, bh.AngleFormat.RADIANS)
        print(f"ECEF position: {x_ecef}")
        ```
    """
    ...

def position_geodetic_to_ecef(x_geod: np.ndarray, angle_format: AngleFormat) -> Any:
    """Convert geodetic position to `ECEF` Cartesian coordinates.

    Transforms a position from geodetic coordinates (latitude, longitude, altitude) using
    the `WGS84` ellipsoid model to Earth-Centered Earth-Fixed (`ECEF`) Cartesian coordinates.

    Args:
        x_geod (numpy.ndarray): Geodetic position `[latitude, longitude, altitude]` where
            latitude is in radians or degrees, longitude is in radians or degrees, and
            altitude is in meters above the `WGS84` ellipsoid.
        angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).

    Returns:
        (numpy.ndarray): `ECEF` Cartesian position `[x, y, z]` in meters.

    Example:
        ```python
        import brahe as bh
        import numpy as np

        # Convert geodetic coordinates (GPS-like) to ECEF
        lat, lon, alt = 40.0, -105.0, 1655.0  # Boulder, CO (degrees, meters)
        x_geod = np.array([lat, lon, alt])
        x_ecef = bh.position_geodetic_to_ecef(x_geod, bh.AngleFormat.DEGREES)
        print(f"ECEF position: {x_ecef}")
        ```
    """
    ...

def position_sez_to_azel(x_sez: np.ndarray, angle_format: AngleFormat) -> Any:
    """Convert position from South-East-Zenith (`SEZ`) frame to azimuth-elevation-range.

    Transforms a position from the local South-East-Zenith (`SEZ`) topocentric frame to
    azimuth-elevation-range spherical coordinates.

    Args:
        x_sez (numpy.ndarray): Position in `SEZ` frame `[south, east, zenith]` in meters.
        angle_format (AngleFormat): Angle format for output angular coordinates (`RADIANS` or `DEGREES`).

    Returns:
        (numpy.ndarray): Azimuth-elevation-range `[azimuth, elevation, range]` where azimuth
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
    ...

def relative_position_ecef_to_enz(
    location_ecef: np.ndarray,
    r_ecef: np.ndarray,
    conversion_type: EllipsoidalConversionType,
) -> Any:
    """Convert relative position from `ECEF` to East-North-Up (`ENZ`) frame.

    Transforms a relative position vector from Earth-Centered Earth-Fixed (`ECEF`) coordinates
    to the local East-North-Up (`ENZ`) topocentric frame at the specified location.

    Args:
        location_ecef (numpy.ndarray): Reference location in `ECEF` coordinates `[x, y, z]` in meters.
        r_ecef (numpy.ndarray): Position vector in `ECEF` coordinates `[x, y, z]` in meters.
        conversion_type (EllipsoidalConversionType): Type of ellipsoidal conversion (`GEOCENTRIC` or `GEODETIC`).

    Returns:
        (numpy.ndarray): Relative position in `ENZ` frame `[east, north, up]` in meters.

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
    ...

def relative_position_ecef_to_sez(
    location_ecef: np.ndarray,
    r_ecef: np.ndarray,
    conversion_type: EllipsoidalConversionType,
) -> Any:
    """Convert relative position from `ECEF` to South-East-Zenith (`SEZ`) frame.

    Transforms a relative position vector from Earth-Centered Earth-Fixed (`ECEF`) coordinates
    to the local South-East-Zenith (`SEZ`) topocentric frame at the specified location.

    Args:
        location_ecef (numpy.ndarray): Reference location in `ECEF` coordinates `[x, y, z]` in meters.
        r_ecef (numpy.ndarray): Position vector in `ECEF` coordinates `[x, y, z]` in meters.
        conversion_type (EllipsoidalConversionType): Type of ellipsoidal conversion (`GEOCENTRIC` or `GEODETIC`).

    Returns:
        (numpy.ndarray): Relative position in `SEZ` frame `[south, east, zenith]` in meters.

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
    ...

def relative_position_enz_to_ecef(
    location_ecef: np.ndarray,
    r_enz: np.ndarray,
    conversion_type: EllipsoidalConversionType,
) -> Any:
    """Convert relative position from East-North-Up (`ENZ`) frame to `ECEF`.

    Transforms a relative position vector from the local East-North-Up (`ENZ`) topocentric
    frame to Earth-Centered Earth-Fixed (`ECEF`) coordinates at the specified location.

    Args:
        location_ecef (numpy.ndarray): Reference location in `ECEF` coordinates `[x, y, z]` in meters.
        r_enz (numpy.ndarray): Relative position in `ENZ` frame `[east, north, up]` in meters.
        conversion_type (EllipsoidalConversionType): Type of ellipsoidal conversion (`GEOCENTRIC` or `GEODETIC`).

    Returns:
        (numpy.ndarray): Position vector in `ECEF` coordinates `[x, y, z]` in meters.

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
    ...

def relative_position_sez_to_ecef(
    location_ecef: np.ndarray,
    x_sez: np.ndarray,
    conversion_type: EllipsoidalConversionType,
) -> Any:
    """Convert relative position from South-East-Zenith (`SEZ`) frame to `ECEF`.

    Transforms a relative position vector from the local South-East-Zenith (`SEZ`) topocentric
    frame to Earth-Centered Earth-Fixed (`ECEF`) coordinates at the specified location.

    Args:
        location_ecef (numpy.ndarray): Reference location in `ECEF` coordinates `[x, y, z]` in meters.
        x_sez (numpy.ndarray): Relative position in `SEZ` frame `[south, east, zenith]` in meters.
        conversion_type (EllipsoidalConversionType): Type of ellipsoidal conversion (`GEOCENTRIC` or `GEODETIC`).

    Returns:
        (numpy.ndarray): Position vector in `ECEF` coordinates `[x, y, z]` in meters.

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
    ...

def rotation_ecef_to_eci(epc: Epoch) -> Any:
    """Computes the combined rotation matrix from the Earth-fixed to the inertial
    reference frame. Applies corrections for bias, precession, nutation,
    Earth-rotation, and polar motion.

    The transformation is accomplished using the `IAU 2006/2000A`, `CIO`-based
    theory using classical angles. The method as described in section 5.5 of
    the SOFA C transformation cookbook.

    The function will utilize the global Earth orientation and loaded data to
    apply corrections for Celestial Intermidate Pole (`CIP`) and polar motion drift
    derived from empirical observations.

    Args:
        epc (Epoch): Epoch instant for computation of transformation matrix

    Returns:
        (numpy.ndarray): 3x3 rotation matrix transforming `ITRF` -> `GCRF`

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
    ...

def rotation_eci_to_ecef(epc: Epoch) -> Any:
    """Computes the combined rotation matrix from the inertial to the Earth-fixed
    reference frame. Applies corrections for bias, precession, nutation,
    Earth-rotation, and polar motion.

    The transformation is accomplished using the `IAU 2006/2000A`, `CIO`-based
    theory using classical angles. The method as described in section 5.5 of
    the SOFA C transformation cookbook.

    The function will utilize the global Earth orientation and loaded data to
    apply corrections for Celestial Intermidate Pole (`CIP`) and polar motion drift
    derived from empirical observations.

    Args:
        epc (Epoch): Epoch instant for computation of transformation matrix

    Returns:
        (numpy.ndarray): 3x3 rotation matrix transforming `GCRF` -> `ITRF`

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
    ...

def rotation_ellipsoid_to_enz(
    x_ellipsoid: np.ndarray, angle_format: AngleFormat
) -> Any:
    """Compute rotation matrix from ellipsoidal coordinates to East-North-Up (`ENZ`) frame.

    Calculates the rotation matrix that transforms vectors from an ellipsoidal coordinate
    frame (geocentric or geodetic) to the local East-North-Up (`ENZ`) topocentric frame at
    the specified location.

    Args:
        x_ellipsoid (numpy.ndarray): Ellipsoidal position `[latitude, longitude, altitude/radius]`
            where latitude is in radians or degrees, longitude is in radians or degrees.
        angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).

    Returns:
        (numpy.ndarray): 3x3 rotation matrix from ellipsoidal frame to `ENZ` frame.
    """
    ...

def rotation_ellipsoid_to_sez(
    x_ellipsoid: np.ndarray, angle_format: AngleFormat
) -> Any:
    """Compute rotation matrix from ellipsoidal coordinates to South-East-Zenith (`SEZ`) frame.

    Calculates the rotation matrix that transforms vectors from an ellipsoidal coordinate
    frame (geocentric or geodetic) to the local South-East-Zenith (`SEZ`) topocentric frame
    at the specified location.

    Args:
        x_ellipsoid (numpy.ndarray): Ellipsoidal position `[latitude, longitude, altitude/radius]`
            where latitude is in radians or degrees, longitude is in radians or degrees.
        angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).

    Returns:
        (numpy.ndarray): 3x3 rotation matrix from ellipsoidal frame to `SEZ` frame.

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
    ...

def rotation_enz_to_ellipsoid(
    x_ellipsoid: np.ndarray, angle_format: AngleFormat
) -> Any:
    """Compute rotation matrix from East-North-Up (`ENZ`) frame to ellipsoidal coordinates.

    Calculates the rotation matrix that transforms vectors from the local East-North-Up
    (`ENZ`) topocentric frame to an ellipsoidal coordinate frame (geocentric or geodetic)
    at the specified location.

    Args:
        x_ellipsoid (numpy.ndarray): Ellipsoidal position `[latitude, longitude, altitude/radius]`
            where latitude is in radians or degrees, longitude is in radians or degrees.
        angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).

    Returns:
        (numpy.ndarray): 3x3 rotation matrix from `ENZ` frame to ellipsoidal frame.
    """
    ...

def rotation_sez_to_ellipsoid(
    x_ellipsoid: np.ndarray, angle_format: AngleFormat
) -> Any:
    """Compute rotation matrix from South-East-Zenith (`SEZ`) frame to ellipsoidal coordinates.

    Calculates the rotation matrix that transforms vectors from the local South-East-Zenith
    (`SEZ`) topocentric frame to an ellipsoidal coordinate frame (geocentric or geodetic)
    at the specified location.

    Args:
        x_ellipsoid (numpy.ndarray): Ellipsoidal position `[latitude, longitude, altitude/radius]`
            where latitude is in radians or degrees, longitude is in radians or degrees.
        angle_format (AngleFormat): Angle format for input angular coordinates (`RADIANS` or `DEGREES`).

    Returns:
        (numpy.ndarray): 3x3 rotation matrix from `SEZ` frame to ellipsoidal frame.

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
    ...

def semimajor_axis(n: float, angle_format: AngleFormat) -> float:
    """Computes the semi-major axis of an astronomical object from Earth
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
    ...

def semimajor_axis_from_orbital_period(period: float) -> float:
    """Computes the semi-major axis from orbital period around Earth.

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
    ...

def semimajor_axis_from_orbital_period_general(period: float, gm: float) -> float:
    """Computes the semi-major axis from orbital period for a general body.

    Args:
        period (float): The orbital period in seconds.
        gm (float): The standard gravitational parameter of primary body in m³/s².

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
    ...

def semimajor_axis_general(n: float, gm: float, angle_format: AngleFormat) -> float:
    """Computes the semi-major axis of an astronomical object from a general body
    given the object's mean motion.

    Args:
        n (float): The mean motion of the astronomical object in radians or degrees.
        gm (float): The standard gravitational parameter of primary body in m³/s².
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
    ...

def set_global_eop_provider_from_caching_provider(provider: CachingEOPProvider) -> Any:
    """Set the global EOP provider using a caching provider.

    Args:
        provider (CachingEOPProvider): Caching EOP provider to set globally

    Example:
        ```python
        import brahe as bh

        provider = bh.CachingEOPProvider(
            "./eop_data/finals.all.iau2000.txt",
            "StandardBulletinA",
            7 * 86400,
            False,
            True,
            "Hold"
        )
        bh.set_global_eop_provider_from_caching_provider(provider)
        ```
    """
    ...

def set_global_eop_provider_from_file_provider(provider: FileEOPProvider) -> Any:
    """Set the global EOP provider using a file-based provider.

    Args:
        provider (FileEOPProvider): File-based EOP provider to set globally

    Example:
        ```python
        import brahe as bh

        provider = bh.FileEOPProvider.from_default_standard(True, "Hold")
        bh.set_global_eop_provider_from_file_provider(provider)
        ```
    """
    ...

def set_global_eop_provider_from_static_provider(provider: StaticEOPProvider) -> Any:
    """Set the global EOP provider using a static provider.

    Args:
        provider (StaticEOPProvider): Static EOP provider to set globally

    Example:
        ```python
        import brahe as bh

        provider = bh.StaticEOPProvider.from_zero()
        bh.set_global_eop_provider_from_static_provider(provider)
        ```
    """
    ...

def state_cartesian_to_osculating(
    x_cart: np.ndarray, angle_format: AngleFormat
) -> np.ndarray:
    """Convert Cartesian state to osculating orbital elements.

    Transforms a state vector from Cartesian position and velocity coordinates to
    osculating Keplerian orbital elements.

    Args:
        x_cart (numpy.ndarray): Cartesian state `[x, y, z, vx, vy, vz]` where position
            is in meters and velocity is in meters per second.
        angle_format (AngleFormat): Angle format for output angular elements (`RADIANS` or `DEGREES`).

    Returns:
        (numpy.ndarray): Osculating orbital elements `[a, e, i, RAAN, omega, M]` where `a` is
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
    ...

def state_ecef_to_eci(epc: Epoch, x_ecef: np.ndarray) -> np.ndarray:
    """Transforms a state vector (position and velocity) from the Earth Centered
    Earth Fixed (`ECEF`/`ITRF`) frame to the Earth Centered Inertial (`ECI`/`GCRF`) frame.

    Applies the full `IAU 2006/2000A` transformation including bias, precession,
    nutation, Earth rotation, and polar motion corrections using global Earth
    orientation parameters. The velocity transformation accounts for the Earth's
    rotation rate.

    Args:
        epc (Epoch): Epoch instant for the transformation
        x_ecef (numpy.ndarray): State vector in `ECEF` frame `[position (m), velocity (m/s)]`, shape `(6,)`

    Returns:
        (numpy.ndarray): State vector in `ECI` frame `[position (m), velocity (m/s)]`, shape `(6,)`

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
    ...

def state_eci_to_ecef(epc: Epoch, x_eci: np.ndarray) -> np.ndarray:
    """Transforms a state vector (position and velocity) from the Earth Centered
    Inertial (`ECI`/`GCRF`) frame to the Earth Centered Earth Fixed (`ECEF`/`ITRF`) frame.

    Applies the full `IAU 2006/2000A` transformation including bias, precession,
    nutation, Earth rotation, and polar motion corrections using global Earth
    orientation parameters. The velocity transformation accounts for the Earth's
    rotation rate.

    Args:
        epc (Epoch): Epoch instant for the transformation
        x_eci (numpy.ndarray): State vector in `ECI` frame `[position (m), velocity (m/s)]`, shape `(6,)`

    Returns:
        (numpy.ndarray): State vector in `ECEF` frame `[position (m), velocity (m/s)]`, shape `(6,)`

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
    ...

def state_osculating_to_cartesian(
    x_oe: np.ndarray, angle_format: AngleFormat
) -> np.ndarray:
    """Convert osculating orbital elements to Cartesian state.

    Transforms a state vector from osculating Keplerian orbital elements to Cartesian
    position and velocity coordinates.

    Args:
        x_oe (numpy.ndarray): Osculating orbital elements `[a, e, i, RAAN, omega, M]` where
            `a` is semi-major axis (meters), `e` is eccentricity (dimensionless), `i` is
            inclination (radians or degrees), `RAAN` is right ascension of ascending node
            (radians or degrees), `omega` is argument of periapsis (radians or degrees),
            and `M` is mean anomaly (radians or degrees).
        angle_format (AngleFormat): Angle format for angular elements (`RADIANS` or `DEGREES`).

    Returns:
        (numpy.ndarray): Cartesian state `[x, y, z, vx, vy, vz]` where position is in meters
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
    ...

def sun_synchronous_inclination(a: float, e: float, angle_format: AngleFormat) -> float:
    """Computes the inclination for a Sun-synchronous orbit around Earth based on
    the J2 gravitational perturbation.

    Args:
        a (float): The semi-major axis of the astronomical object in meters.
        e (float): The eccentricity of the astronomical object's orbit (dimensionless).
        angle_format (AngleFormat): Return output in AngleFormat.DEGREES or AngleFormat.RADIANS.

    Returns:
        float: Inclination for a Sun synchronous orbit in degrees or radians.

    Example:
        ```python
        import brahe as bh

        # Calculate sun-synchronous inclination for typical Earth observation satellite (600 km)
        a = bh.R_EARTH + 600e3
        e = 0.001  # nearly circular
        inc = bh.sun_synchronous_inclination(a, e, bh.AngleFormat.DEGREES)
        print(f"Sun-synchronous inclination: {inc:.2f} degrees")
        ```
    """
    ...

def time_system_offset_for_datetime(
    year: int,
    month: int,
    day: int,
    hour: int,
    minute: int,
    second: float,
    nanosecond: float,
    time_system_src: TimeSystem,
    time_system_dst: TimeSystem,
) -> float:
    """Calculate the offset between two time systems for a given Gregorian calendar date.

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
    ...

def time_system_offset_for_jd(
    jd: float, time_system_src: TimeSystem, time_system_dst: TimeSystem
) -> float:
    """Calculate the offset between two time systems for a given Julian Date.

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
    ...

def time_system_offset_for_mjd(
    mjd: float, time_system_src: TimeSystem, time_system_dst: TimeSystem
) -> float:
    """Calculate the offset between two time systems for a given Modified Julian Date.

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
    ...

def validate_tle_line(line: str) -> bool:
    """Validate single TLE line.

    Args:
        line (str): TLE line to validate.

    Returns:
        bool: True if the line is valid.
    """
    ...

def validate_tle_lines(line1: str, line2: str) -> bool:
    """Validate TLE lines.

    Args:
        line1 (str): First line of TLE data.
        line2 (str): Second line of TLE data.

    Returns:
        bool: True if both lines are valid.
    """
    ...

# Module constants

AS2RAD: float
AU: float
C_LIGHT: float
DEG2RAD: float
ECC_EARTH: float
GM_EARTH: float
GM_JUPITER: float
GM_MARS: float
GM_MERCURY: float
GM_MOON: float
GM_NEPTUNE: float
GM_PLUTO: float
GM_SATURN: float
GM_SUN: float
GM_URANUS: float
GM_VENUS: float
GPS: Any
GPS_TAI: float
GPS_TT: float
GPS_ZERO: float
J2_EARTH: float
MJD2000: float
MJD_ZERO: float
OMEGA_EARTH: float
P_SUN: float
RAD2AS: float
RAD2DEG: float
R_EARTH: float
R_MOON: float
R_SUN: float
TAI: Any
TAI_GPS: float
TAI_TT: float
TT: Any
TT_GPS: float
TT_TAI: float
UT1: Any
UTC: Any
WGS84_A: float
WGS84_F: float
__version__: str
