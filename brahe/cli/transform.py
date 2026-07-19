from enum import Enum
import typer
from typing_extensions import Annotated, Tuple
from loguru import logger
import numpy as np

import brahe
from brahe.cli.utils import set_cli_eop, parse_numeric_expression

app = typer.Typer()


class Frame(str, Enum):
    """Named reference frames accepted by the transform commands.

    Values mirror the strings understood by
    [`ReferenceFrame.from_string`](../library_api/frames/router.md#brahe.ReferenceFrame.from_string).
    """

    ECI = "ECI"
    ECEF = "ECEF"
    GCRF = "GCRF"
    ITRF = "ITRF"
    EME2000 = "EME2000"
    LCI = "LCI"
    LFPA = "LFPA"
    LFME = "LFME"
    MCI = "MCI"
    MCMF = "MCMF"
    EMBI = "EMBI"
    SSBI = "SSBI"
    EMR = "EMR"
    SER = "SER"
    GSE = "GSE"


class CartesianFrame(str, Enum):
    """Reference frame for cartesian coordinate input/output (ECI or ECEF)."""

    ECI = "ECI"
    ECEF = "ECEF"


def _resolve_frame(frame: Frame) -> "brahe.ReferenceFrame":
    """Convert a CLI `Frame` value into a core `ReferenceFrame`.

    Raises a Typer error with a clean message if the frame string is not
    recognized by the core router.
    """
    try:
        return brahe.ReferenceFrame.from_string(frame.value)
    except Exception as e:  # noqa: BLE001 - surface as CLI error
        typer.echo(f"ERROR: unknown reference frame '{frame.value}': {e}")
        raise typer.Exit(code=1)


class StateRepresentation(str, Enum):
    """State representation format (extends brahe.OrbitRepresentation with geodetic/geocentric)"""

    keplerian = "keplerian"  # Keplerian orbital elements [a, e, i, Ω, ω, ν]
    cartesian = "cartesian"  # Cartesian state [x, y, z, vx, vy, vz]
    geodetic = "geodetic"  # Geodetic coordinates [lat, lon, alt, 0, 0, 0]
    geocentric = "geocentric"  # Geocentric coordinates [lat, lon, radius, 0, 0, 0]


def _parse_vector(values, n: int):
    """Parse `n` state components, each allowing constant expressions."""
    try:
        return np.array([parse_numeric_expression(values[i]) for i in range(n)])
    except ValueError as e:
        typer.echo(f"Error parsing state values: {e}")
        raise typer.Exit(code=1)


def _echo_vector(x, format_string: str):
    typer.echo("[" + ", ".join(f"{v:{format_string}}" for v in x) + "]")


@app.command()
def frame(
    from_frame: Annotated[
        Frame, typer.Argument(help="The reference frame to convert from")
    ],
    to_frame: Annotated[
        Frame, typer.Argument(help="The reference frame to convert to")
    ],
    epoch: Annotated[str, typer.Argument(help="Epoch to perform the conversion at")],
    state: Annotated[
        Tuple[float, float, float, float, float, float],
        typer.Argument(..., help="The state to convert [x, y, z, vx, vy, vz]"),
    ],
    format_string: Annotated[
        str, typer.Option("--format", help="The format of the output")
    ] = "f",
):
    """Transform a Cartesian state vector between reference frames."""
    logger.info(
        f"Converting state between frames: {from_frame.value} -> {to_frame.value}"
    )
    logger.debug(f"Epoch: {epoch}, State: {state}")
    set_cli_eop()
    epc = brahe.Epoch(epoch)
    x = _parse_vector(state, 6)

    if from_frame == to_frame:
        _echo_vector(x, format_string)
        return

    src = _resolve_frame(from_frame)
    dst = _resolve_frame(to_frame)
    try:
        y = brahe.state_frame_to_frame(src, dst, epc, x)
    except Exception as e:  # noqa: BLE001 - surface as CLI error
        typer.echo(f"ERROR: frame transform failed: {e}")
        raise typer.Exit(code=1)
    _echo_vector(y, format_string)


@app.command()
def position(
    from_frame: Annotated[
        Frame, typer.Argument(help="The reference frame to convert from")
    ],
    to_frame: Annotated[
        Frame, typer.Argument(help="The reference frame to convert to")
    ],
    epoch: Annotated[str, typer.Argument(help="Epoch to perform the conversion at")],
    pos: Annotated[
        Tuple[float, float, float],
        typer.Argument(..., help="The position to convert [x, y, z]"),
    ],
    format_string: Annotated[
        str, typer.Option("--format", help="The format of the output")
    ] = "f",
):
    """Transform a position vector between reference frames."""
    logger.info(
        f"Converting position between frames: {from_frame.value} -> {to_frame.value}"
    )
    logger.debug(f"Epoch: {epoch}, Position: {pos}")
    set_cli_eop()
    epc = brahe.Epoch(epoch)
    x = _parse_vector(pos, 3)

    if from_frame == to_frame:
        _echo_vector(x, format_string)
        return

    src = _resolve_frame(from_frame)
    dst = _resolve_frame(to_frame)
    try:
        y = brahe.position_frame_to_frame(src, dst, epc, x)
    except Exception as e:  # noqa: BLE001 - surface as CLI error
        typer.echo(f"ERROR: position transform failed: {e}")
        raise typer.Exit(code=1)
    _echo_vector(y, format_string)


@app.command()
def rotation(
    from_frame: Annotated[
        Frame, typer.Argument(help="The reference frame to convert from")
    ],
    to_frame: Annotated[
        Frame, typer.Argument(help="The reference frame to convert to")
    ],
    epoch: Annotated[str, typer.Argument(help="Epoch to perform the conversion at")],
    format_string: Annotated[
        str, typer.Option("--format", help="The format of the output")
    ] = "f",
):
    """Emit the 3x3 rotation matrix between two reference frames."""
    logger.info(
        f"Computing rotation between frames: {from_frame.value} -> {to_frame.value}"
    )
    logger.debug(f"Epoch: {epoch}")
    set_cli_eop()
    epc = brahe.Epoch(epoch)

    src = _resolve_frame(from_frame)
    dst = _resolve_frame(to_frame)
    try:
        r = brahe.rotation_frame_to_frame(src, dst, epc)
    except Exception as e:  # noqa: BLE001 - surface as CLI error
        typer.echo(f"ERROR: rotation transform failed: {e}")
        raise typer.Exit(code=1)
    for row in np.asarray(r):
        _echo_vector(row, format_string)


@app.command()
def coordinates(
    from_system: Annotated[
        StateRepresentation,
        typer.Argument(help="The state representation to convert from"),
    ],
    to_system: Annotated[
        StateRepresentation,
        typer.Argument(help="The state representation to convert to"),
    ],
    epoch: Annotated[
        str,
        typer.Argument(
            help="Epoch (ISO-8601 format). Use empty string '' if not needed for conversion"
        ),
    ],
    state: Annotated[
        Tuple[float, float, float, float, float, float],
        typer.Argument(..., help="The state to convert"),
    ],
    from_frame: Annotated[
        CartesianFrame,
        typer.Option(help="Reference frame for cartesian input (ECI or ECEF)"),
    ] = CartesianFrame.ECI,
    to_frame: Annotated[
        CartesianFrame,
        typer.Option(help="Reference frame for cartesian output (ECI or ECEF)"),
    ] = CartesianFrame.ECI,
    as_degrees: Annotated[
        bool, typer.Option(help="Output format in degrees if applicable")
    ] = True,
    format_string: Annotated[
        str, typer.Option("--format", help="The format of the output")
    ] = "f",
):
    """Convert between state representations (keplerian, cartesian, geodetic, geocentric)."""
    logger.info(f"Converting coordinates: {from_system.value} -> {to_system.value}")
    logger.debug(f"State: {state}, Frames: {from_frame.value} -> {to_frame.value}")
    set_cli_eop()

    # Parse state values (may contain constant expressions)
    try:
        x = np.array(
            [
                parse_numeric_expression(state[0]),
                parse_numeric_expression(state[1]),
                parse_numeric_expression(state[2]),
                parse_numeric_expression(state[3]),
                parse_numeric_expression(state[4]),
                parse_numeric_expression(state[5]),
            ]
        )
    except ValueError as e:
        typer.echo(f"Error parsing state values: {e}")
        raise typer.Exit(code=1)

    angle_format = (
        brahe.AngleFormat.DEGREES if as_degrees else brahe.AngleFormat.RADIANS
    )

    # Same system, just return
    if from_system == to_system and (
        from_system != StateRepresentation.cartesian or from_frame == to_frame
    ):
        typer.echo(
            f"[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}, {x[3]:{format_string}}, {x[4]:{format_string}}, {x[5]:{format_string}}]"
        )
        return

    # === Conversions FROM Keplerian ===
    if from_system == StateRepresentation.keplerian:
        x_eci = brahe.state_koe_to_eci(x, angle_format)

        if to_system == StateRepresentation.cartesian:
            if to_frame == CartesianFrame.ECEF:
                if not epoch:
                    typer.echo("ERROR: --epoch required for frame conversion to ECEF")
                    raise typer.Exit(code=1)
                x = brahe.state_eci_to_ecef(brahe.Epoch(epoch), x_eci)
            else:
                x = x_eci
        elif to_system == StateRepresentation.geodetic:
            if not epoch:
                typer.echo("ERROR: --epoch required for geodetic conversion")
                raise typer.Exit(code=1)
            x_ecef = brahe.state_eci_to_ecef(brahe.Epoch(epoch), x_eci)
            x = brahe.position_ecef_to_geodetic(x_ecef[0:3], angle_format)
        elif to_system == StateRepresentation.geocentric:
            if not epoch:
                typer.echo("ERROR: --epoch required for geocentric conversion")
                raise typer.Exit(code=1)
            x_ecef = brahe.state_eci_to_ecef(brahe.Epoch(epoch), x_eci)
            x = brahe.position_ecef_to_geocentric(x_ecef[0:3], angle_format)

    # === Conversions FROM Cartesian ===
    elif from_system == StateRepresentation.cartesian:
        if to_system == StateRepresentation.keplerian:
            # Need ECI for Keplerian
            x_eci = (
                x
                if from_frame == CartesianFrame.ECI
                else brahe.state_ecef_to_eci(brahe.Epoch(epoch), x)
            )
            x = brahe.state_eci_to_koe(x_eci, angle_format)
        elif to_system == StateRepresentation.cartesian:
            # Frame conversion
            epc = brahe.Epoch(epoch)
            if from_frame == CartesianFrame.ECI and to_frame == CartesianFrame.ECEF:
                x = brahe.state_eci_to_ecef(epc, x)
            elif from_frame == CartesianFrame.ECEF and to_frame == CartesianFrame.ECI:
                x = brahe.state_ecef_to_eci(epc, x)
        elif to_system in (
            StateRepresentation.geodetic,
            StateRepresentation.geocentric,
        ):
            # Need ECEF for geodetic/geocentric
            x_ecef = (
                brahe.state_eci_to_ecef(brahe.Epoch(epoch), x)
                if from_frame == CartesianFrame.ECI
                else x
            )
            if to_system == StateRepresentation.geodetic:
                x = brahe.position_ecef_to_geodetic(x_ecef[0:3], angle_format)
            else:
                x = brahe.position_ecef_to_geocentric(x_ecef[0:3], angle_format)

    # === Conversions FROM Geocentric ===
    elif from_system == StateRepresentation.geocentric:
        x_ecef = brahe.position_geocentric_to_ecef(x[0:3], angle_format)

        if to_system == StateRepresentation.geodetic:
            x = brahe.position_ecef_to_geodetic(x_ecef, angle_format)
        elif to_system == StateRepresentation.cartesian:
            # Geodetic->ECEF gives position only
            x = (
                brahe.position_ecef_to_eci(brahe.Epoch(epoch), x_ecef)
                if to_frame == CartesianFrame.ECI
                else x_ecef
            )
        elif to_system == StateRepresentation.keplerian:
            typer.echo(
                "ERROR: Cannot convert from Geocentric to Keplerian (position-only to orbit elements)"
            )
            raise typer.Exit(code=1)

    # === Conversions FROM Geodetic ===
    elif from_system == StateRepresentation.geodetic:
        x_ecef = brahe.position_geodetic_to_ecef(x[0:3], angle_format)

        if to_system == StateRepresentation.geocentric:
            x = brahe.position_ecef_to_geocentric(x_ecef, angle_format)
        elif to_system == StateRepresentation.cartesian:
            # Geodetic->ECEF gives position only
            x = (
                brahe.position_ecef_to_eci(brahe.Epoch(epoch), x_ecef)
                if to_frame == CartesianFrame.ECI
                else x_ecef
            )
        elif to_system == StateRepresentation.keplerian:
            typer.echo(
                "ERROR: Cannot convert from Geodetic to Keplerian (position-only to orbit elements)"
            )
            raise typer.Exit(code=1)

    # Output result
    if len(x) == 3:
        typer.echo(
            f"[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}]"
        )
    else:
        typer.echo(
            f"[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}, {x[3]:{format_string}}, {x[4]:{format_string}}, {x[5]:{format_string}}]"
        )
