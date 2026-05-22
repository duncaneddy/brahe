"""GMAT benchmark implementations for access computation.

Implementation uses GMAT's ContactLocator with the SPICESGP4 propagator.
A single script is generated for each run containing all 100 GroundStation
objects, a single Spacecraft propagated with SPICESGP4 over the search window,
and one ContactLocator that searches all observers in one pass.

Output format: [[{"start_jd": float, "end_jd": float}, ...], ...]
One list per ground location (100 total), each containing zero or more windows.

GMAT ContactLocator:
- Uses SPICE-based line-of-sight algorithm
- GroundStation.MinimumElevationAngle controls the 10 deg elevation constraint
- UseLightTimeDelay = false, UseStellarAberration = false for simple geometric
  visibility (matching brahe/Rust/Java implementations)
- StepSize = 30 s gives ~30 s resolution; ContactLocator internally refines
  event boundaries using the SPICE event algorithm

Parse pattern (Legacy report format):
  Target: Sat
  <blank>
  Observer: GS0
  Start Time (UTC)  Stop Time (UTC)  Duration (s)
  <rows>
  <blank>
  Number of events : N
  <blank>
  Observer: GS1
  ...
"""

import calendar
import os
import re
import tempfile
from datetime import datetime

from benchmarks.comparative.implementations.gmat.base import (
    build_task_result,
    gmat_clear,
    time_iterations,
)

# GMAT UTCMJD reference: JD_UTC - this constant = UTCMJD
_JD_TO_UTCMJD = 2430000.0


def _parse_tle_epoch_jd(line1: str) -> float:
    """Parse epoch from TLE line 1 and return Julian Date (UTC).

    TLE epoch field (columns 19-32): YYDDD.DDDDDDDD
      YY  — 2-digit year (57-99 -> 1957-1999; 00-56 -> 2000-2056)
      DDD — day of year, 1-based
      .DD — fractional day
    """
    epoch_str = line1[18:32].strip()
    yy = int(epoch_str[:2])
    day_frac = float(epoch_str[2:])
    year = (1900 + yy) if yy >= 57 else (2000 + yy)

    # JD of Jan 1 00:00:00 UTC for that year
    from datetime import date
    jan1_ts = calendar.timegm(date(year, 1, 1).timetuple())
    # day_frac is 1-based: day 1.0 = Jan 1 00:00:00 UTC
    jd_utc = jan1_ts / 86400.0 + 2440587.5 + (day_frac - 1.0)
    return jd_utc


def _jd_to_utcmjd(jd: float) -> float:
    """Convert Julian Date (UTC) to GMAT UTCModJulian."""
    return jd - _JD_TO_UTCMJD


def _gmat_utcgreg_to_jd(s: str) -> float:
    """Parse GMAT UTC Gregorian timestamp to Julian Date.

    Input format: 'DD Mon YYYY HH:MM:SS.mmm'  (e.g. '20 Sep 2008 22:52:03.361')
    """
    s = s.strip()
    dot_pos = s.rfind('.')
    frac = 0.0
    if dot_pos > 0:
        frac = float('0' + s[dot_pos:])
        s_whole = s[:dot_pos]
    else:
        s_whole = s
    dt = datetime.strptime(s_whole, '%d %b %Y %H:%M:%S')
    ts = calendar.timegm(dt.timetuple())
    # Unix epoch (1970-01-01 00:00:00 UTC) = JD 2440587.5
    return ts / 86400.0 + 2440587.5 + frac / 86400.0


def _build_contact_script(
    tle_file: str,
    report_file: str,
    locations: list[dict],
    min_elevation_deg: float,
    utcmjd_start: float,
    utcmjd_end: float,
) -> str:
    """Build a GMAT script that runs SPICESGP4 + ContactLocator for all locations."""
    lines = []

    # Spacecraft
    lines.append("Create Spacecraft Sat")
    lines.append(f"Sat.EphemerisName = '{tle_file}'")
    lines.append("Sat.Id = 'ISS'")
    lines.append("")

    # Propagator
    lines.append("Create Propagator TLEProp")
    lines.append("TLEProp.Type = SPICESGP4")
    lines.append("TLEProp.InitialStepSize = 60")
    lines.append("")

    # Ground stations
    gs_names = []
    for i, loc in enumerate(locations):
        name = f"GS{i}"
        gs_names.append(name)
        lat = float(loc["lat"])
        # GMAT Location2 for Spherical state is East longitude (0-360)
        lon_east = float(loc["lon"]) % 360.0
        alt_km = float(loc["alt"]) / 1000.0  # m -> km (GMAT uses km)
        lines.append(f"Create GroundStation {name}")
        lines.append(f"{name}.CentralBody = Earth")
        lines.append(f"{name}.StateType = Spherical")
        lines.append(f"{name}.Location1 = {lat:.10f}")
        lines.append(f"{name}.Location2 = {lon_east:.10f}")
        lines.append(f"{name}.Location3 = {alt_km:.10f}")
        lines.append(f"{name}.MinimumElevationAngle = {min_elevation_deg:.4f}")
        lines.append("")

    # ContactLocator
    observers_str = ", ".join(gs_names)
    lines.append("Create ContactLocator CL")
    lines.append("CL.Target = Sat")
    lines.append(f"CL.Observers = {{{observers_str}}}")
    lines.append(f"CL.Filename = '{report_file}'")
    lines.append("CL.StepSize = 30")
    lines.append("CL.UseLightTimeDelay = false")
    lines.append("CL.UseStellarAberration = false")
    lines.append("CL.UseEntireInterval = false")
    lines.append("CL.InputEpochFormat = UTCModJulian")
    lines.append(f"CL.InitialEpoch = '{utcmjd_start:.10f}'")
    lines.append(f"CL.FinalEpoch = '{utcmjd_end:.10f}'")
    lines.append("")

    # Mission sequence: propagate over the search window
    elapsed_days = (utcmjd_end - utcmjd_start)
    lines.append("BeginMissionSequence")
    lines.append(f"Propagate TLEProp(Sat) {{Sat.ElapsedDays = {elapsed_days:.15f}}}")
    lines.append("")

    return "\n".join(lines)


def _parse_contact_report(report_file: str, n_locations: int) -> list[list[dict]]:
    """Parse a GMAT Legacy ContactLocator report.

    Returns a list of n_locations entries, each a list of
    {'start_jd': float, 'end_jd': float} dicts.

    Legacy format per-observer block:
      Observer: <name>
      Start Time (UTC)            Stop Time (UTC)               Duration (s)
      <timestamp>    <timestamp>      <seconds>
      ...
      [blank line]
      Number of events : N
    """
    results: list[list[dict]] = [[] for _ in range(n_locations)]

    # Map GS name -> location index
    gs_re = re.compile(r'^Observer:\s+(GS(\d+))')
    row_re = re.compile(
        r'^(\d{1,2}\s+\w{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}\.\d+)'
        r'\s+'
        r'(\d{1,2}\s+\w{3}\s+\d{4}\s+\d{2}:\d{2}:\d{2}\.\d+)'
    )

    current_idx: int | None = None

    try:
        with open(report_file) as f:
            for line in f:
                line = line.rstrip('\n')
                # Check for observer header
                m_gs = gs_re.match(line.strip())
                if m_gs:
                    current_idx = int(m_gs.group(2))
                    continue
                # Check for a data row
                if current_idx is not None:
                    m_row = row_re.match(line.strip())
                    if m_row:
                        start_jd = _gmat_utcgreg_to_jd(m_row.group(1))
                        end_jd = _gmat_utcgreg_to_jd(m_row.group(2))
                        if current_idx < n_locations:
                            results[current_idx].append(
                                {"start_jd": start_jd, "end_jd": end_jd}
                            )
    except FileNotFoundError:
        # No contacts found or GMAT did not write the report (zero-event case)
        pass

    return results


def sgp4_access(params: dict, iterations: int):
    """Benchmark SGP4 access computation for 100 ground locations via GMAT.

    Uses GMAT's ContactLocator with SPICESGP4 propagation. All 100 ground
    stations are processed in a single GMAT script run. The ContactLocator
    searches the 86400 s window from the TLE epoch using a 30 s step size.

    Parameters:
      - line1                : TLE line 1
      - line2                : TLE line 2
      - locations            : list of 100 {lon, lat, alt} dicts
      - min_elevation_deg    : minimum elevation angle (degrees)
      - search_duration_seconds : search window length (seconds)

    Returns: list of 100 lists, each containing zero or more
      {"start_jd": float, "end_jd": float} windows (JD UTC).
    """
    line1 = params["line1"]
    line2 = params["line2"]
    locations = params["locations"]
    min_el = float(params["min_elevation_deg"])
    duration = float(params["search_duration_seconds"])

    # Parse TLE epoch to get search window bounds
    jd_start = _parse_tle_epoch_jd(line1)
    jd_end = jd_start + duration / 86400.0
    utcmjd_start = _jd_to_utcmjd(jd_start)
    utcmjd_end = _jd_to_utcmjd(jd_end)

    tmpdir = os.environ.get("TMPDIR", tempfile.gettempdir())
    tle_file = os.path.join(tmpdir, "gmat_access.tle")
    script_file = os.path.join(tmpdir, "gmat_access.script")
    report_file = os.path.join(tmpdir, "gmat_access_contacts.txt")

    # Write TLE file once (outside timed region)
    with open(tle_file, "w") as f:
        f.write("ISS\n")
        f.write(line1 + "\n")
        f.write(line2 + "\n")

    # Build the script once (outside timed region); it's deterministic
    script_content = _build_contact_script(
        tle_file, report_file, locations, min_el, utcmjd_start, utcmjd_end
    )
    with open(script_file, "w") as f:
        f.write(script_content)

    n_locations = len(locations)

    def run():
        import gmatpy as gmat

        if os.path.exists(report_file):
            os.remove(report_file)

        gmat_clear()
        gmat.LoadScript(script_file)
        gmat.RunScript()

        return _parse_contact_report(report_file, n_locations)

    times, results = time_iterations(run, iterations)

    return build_task_result(
        "access.sgp4_access",
        iterations,
        times,
        results,
        extra_metadata={
            "propagator": "SPICESGP4",
            "contact_locator": "GMAT ContactLocator",
            "step_size_s": 30,
            "light_time": False,
            "stellar_aberration": False,
            "n_locations": n_locations,
        },
    )
