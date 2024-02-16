from enum import Enum
import typer
from typing_extensions import Annotated, Tuple
import numpy as np

import brahe
from brahe.cli.utils import epoch_from_epochlike, set_cli_eop

app = typer.Typer()

class ReferenceFrame(str, Enum):
    ECI = "ECI"
    ECEF = "ECEF"
    # ITRF = "ITRF"
    # ICRF = "ICRF"
    # TOD = "TOD"
    # MOD = "MOD"
    # TEME = "TEME"
    # GCRF = "GCRF"

class CoordinateSystem(str, Enum):
    keplerian = "keplerian"
    cartesian = "cartesian"
    eci = "eci"
    ecef = "ecef"
    geodetic = "geodetic"
    geocentric = "geocentric"

class AttitudeRepresentation(str, Enum):
    quaternion = "quaternion"
    euler_angles = "euler_angles"
    euler_axis = "euler_axis"
    rotation_matrix = "rotation_matrix"

@app.command()
def frame(
        epoch: Annotated[str, typer.Argument(help="Epoch to perform the conversion at if required")],
          state: Annotated[Tuple[float, float, float, float, float, float], typer.Argument(..., help="The state to convert")],
          from_frame: Annotated[ReferenceFrame, typer.Argument(help="The reference frame to convert from")],
          to_frame: Annotated[ReferenceFrame, typer.Argument(help="The reference frame to convert to")],
        format_string: Annotated[str, typer.Option("--format", help="The format of the output")] = "f"):

    set_cli_eop()
    epc = epoch_from_epochlike(epoch)

    x = np.array([state[0], state[1], state[2], state[3], state[4], state[5]])

    if from_frame == to_frame:
        typer.echo(f'[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}, {x[3]:{format_string}}, {x[4]:{format_string}}, {x[5]:{format_string}}]')

    if from_frame == ReferenceFrame.ECI and to_frame == ReferenceFrame.ECEF:
        x = brahe.state_eci_to_ecef(epc, x)
        typer.echo(f'[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}, {x[3]:{format_string}}, {x[4]:{format_string}}, {x[5]:{format_string}}]')

    if from_frame == ReferenceFrame.ECEF and to_frame == ReferenceFrame.ECI:
        x = brahe.state_ecef_to_eci(epc, x)
        typer.echo(f'[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}, {x[3]:{format_string}}, {x[4]:{format_string}}, {x[5]:{format_string}}]')


@app.command()
def coordinates(state: Annotated[Tuple[float, float, float, float, float, float], typer.Argument(..., help="The state to convert")],
               from_system: Annotated[CoordinateSystem, typer.Argument(help="The coordinate system to convert from")],
               to_system: Annotated[CoordinateSystem, typer.Argument(help="The coordinate system to convert to")],
                epoch: Annotated[str, typer.Option(help="Epoch to perform the conversion at if required")] = "",
                as_degrees: Annotated[bool, typer.Option(help="Output format in degrees if applicable")] = True,
                format_string: Annotated[str, typer.Option("--format", help="The format of the output")] = "f"):
    # Set the EOP provider - Technically not needed for all conversions but we do it
    # anyways to reduce the number of times we need to set it.
    set_cli_eop()

    x = np.array([state[0], state[1], state[2], state[3], state[4], state[5]])

    if from_system == to_system:
        typer.echo(f'[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}, {x[3]:{format_string}}, {x[4]:{format_string}}, {x[5]:{format_string}}]')
        return

    # Conversions starting from Keplerian elements
    if from_system == CoordinateSystem.keplerian:
        x_eci = brahe.state_osculating_to_cartesian(x, as_degrees)

        if to_system == CoordinateSystem.cartesian or to_system == CoordinateSystem.eci:
            x = x_eci
            typer.echo(f'[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}, {x[3]:{format_string}}, {x[4]:{format_string}}, {x[5]:{format_string}}]')
            return

        epc = epoch_from_epochlike(epoch)

        x_ecef = brahe.state_eci_to_ecef(epc, x_eci)

        if to_system == CoordinateSystem.ecef:
            x = x_ecef
            typer.echo(f'[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}, {x[3]:{format_string}}, {x[4]:{format_string}}, {x[5]:{format_string}}]')
            return

        if to_system == CoordinateSystem.geocentric:
            x = brahe.position_ecef_to_geocentric(x_ecef[0:3], as_degrees)
            typer.echo(f'[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}')
            return

        if to_system == CoordinateSystem.geodetic:
            x = brahe.position_ecef_to_geodetic(x_ecef[0:3], as_degrees)
            typer.echo(f'[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}]')
            return

    # Conversions starting from Cartesian / ECI coordinates
    if from_system == CoordinateSystem.cartesian or from_system == CoordinateSystem.eci:
        if to_system == CoordinateSystem.keplerian:
            x = brahe.state_cartesian_to_osculating(x, as_degrees)
            typer.echo(f'[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}, {x[3]:{format_string}}, {x[4]:{format_string}}, {x[5]:{format_string}}]')
            return

        if to_system == CoordinateSystem.eci:
            # Already addressed
            return

        epc = epoch_from_epochlike(epoch)
        x_ecef = brahe.state_eci_to_ecef(epc, x)

        if to_system == CoordinateSystem.ecef:
            x = x_ecef
            typer.echo(f'[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}, {x[3]:{format_string}}, {x[4]:{format_string}}, {x[5]:{format_string}}]')
            return

        if to_system == CoordinateSystem.geocentric:
            x = brahe.position_ecef_to_geocentric(x_ecef[0:3], as_degrees)
            typer.echo(f'[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}')
            return

        if to_system == CoordinateSystem.geodetic:
            x = brahe.position_ecef_to_geodetic(x_ecef[0:3], as_degrees)
            typer.echo(f'[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}]')
            return

    # Conversions starting from ECEF coordinates
    if from_system == CoordinateSystem.ecef:
        if to_system == CoordinateSystem.geocentric:
            x = brahe.position_ecef_to_geocentric(x[0:3], as_degrees)
            typer.echo(f'[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}')
            return

        if to_system == CoordinateSystem.geodetic:
            x = brahe.position_ecef_to_geodetic(x[0:3], as_degrees)
            typer.echo(f'[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}]')
            return

        epc = epoch_from_epochlike(epoch)
        x_eci = brahe.state_ecef_to_eci(epc, x)


        if to_system == CoordinateSystem.cartesian or to_system == CoordinateSystem.eci:
            x = x_eci
            typer.echo(f'[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}, {x[3]:{format_string}}, {x[4]:{format_string}}, {x[5]:{format_string}}]')
            return

        if to_system == CoordinateSystem.keplerian:
            x = brahe.state_cartesian_to_osculating(x_eci, as_degrees)
            typer.echo(f'[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}, {x[3]:{format_string}}, {x[4]:{format_string}}, {x[5]:{format_string}}]')
            return

    # Conversions starting from Geocentric coordinates
    if from_system == CoordinateSystem.geocentric:
        x_ecef = brahe.position_geocentric_to_ecef(x, as_degrees)

        if to_system == CoordinateSystem.geodetic:
            x = brahe.position_ecef_to_geodetic(x_ecef, as_degrees)
            typer.echo(f'[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}]')
            return

        if to_system == CoordinateSystem.ecef:
            x = brahe.position_geocentric_to_ecef(x, as_degrees)
            typer.echo(f'[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}]')
            return

        if to_system == CoordinateSystem.cartesian or to_system == CoordinateSystem.eci:
            epc = epoch_from_epochlike(epoch)
            x = brahe.position_ecef_to_eci(epc, x_ecef)
            typer.echo(f'[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}]')
            return

        if to_system == CoordinateSystem.keplerian:
            typer.echo("ERROR: Unable to convert from Geocentric to Keplerian elements directly")
            typer.Exit(code=1)

    # Conversions starting from Geodetic coordinates
    if from_system == CoordinateSystem.geodetic:
        x_ecef = brahe.position_geodetic_to_ecef(x, as_degrees)

        if to_system == CoordinateSystem.geocentric:
            x = brahe.position_ecef_to_geocentric(x_ecef, as_degrees)
            typer.echo(f'[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}')
            return

        if to_system == CoordinateSystem.ecef:
            x = x_ecef
            typer.echo(f'[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}]')
            return

        if to_system == CoordinateSystem.cartesian or to_system == CoordinateSystem.eci:
            epc = epoch_from_epochlike(epoch)
            x = brahe.position_ecef_to_eci(epc, x_ecef)
            typer.echo(f'[{x[0]:{format_string}}, {x[1]:{format_string}}, {x[2]:{format_string}}]')
            return

        if to_system == CoordinateSystem.keplerian:
            typer.echo("ERROR: Unable to convert from Geodetic to Keplerian elements directly")
            typer.Exit(code=1)

@app.command()
def attitude():
    typer.echo("Not implemented yet")