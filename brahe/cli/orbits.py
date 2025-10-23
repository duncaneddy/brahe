from enum import Enum
import typer
from typing_extensions import Annotated
from loguru import logger

import brahe
from brahe.cli.utils import parse_numeric_expression

app = typer.Typer()


class OrbitalAnomaly(str, Enum):
    mean = "mean"
    eccentric = "eccentric"
    true = "true"


class TimeUnit(str, Enum):
    seconds = "seconds"
    minutes = "minutes"
    hours = "hours"
    days = "days"
    years = "years"


@app.command()
def orbital_period(
    semi_major_axis: Annotated[
        str,
        typer.Argument(
            help="The semi-major axis of the orbit (supports constants like R_EARTH+500e3)"
        ),
    ],
    gm: Annotated[
        str, typer.Option(help="The gravitational parameter of the central body")
    ] = None,
    units: Annotated[
        TimeUnit, typer.Option(help="The time units of the output")
    ] = TimeUnit.seconds,
    format_string: Annotated[
        str, typer.Option("--format", help="The format of the output")
    ] = "f",
):
    # Parse numeric arguments
    try:
        sma = parse_numeric_expression(semi_major_axis)
        gm_val = parse_numeric_expression(gm) if gm is not None else None
    except ValueError as e:
        typer.echo(f"Error parsing numeric values: {e}")
        raise typer.Exit(code=1)

    logger.info(f"Computing orbital period for semi-major axis {sma}m")
    divisor = 1.0
    if units == TimeUnit.seconds:
        divisor = 1.0
    elif units == TimeUnit.minutes:
        divisor = 60.0
    elif units == TimeUnit.hours:
        divisor = 3600.0
    elif units == TimeUnit.days:
        divisor = 86400.0
    elif units == TimeUnit.years:
        divisor = 86400.0 * 365.25

    if gm_val is not None:
        typer.echo(
            f"{brahe.orbital_period_general(sma, gm_val) / divisor:{format_string}}"
        )
    else:
        typer.echo(f"{brahe.orbital_period(sma) / divisor:{format_string}}")


@app.command()
def sma_from_period(
    period: Annotated[
        str, typer.Argument(help="The orbital period (supports expressions)")
    ],
    units: Annotated[
        TimeUnit, typer.Option(help="The time units of the input")
    ] = TimeUnit.seconds,
    gm: Annotated[
        str, typer.Option(help="The gravitational parameter of the central body")
    ] = None,
    format_string: Annotated[
        str, typer.Option("--format", help="The format of the output")
    ] = "f",
):
    # Parse numeric arguments
    try:
        period_val = parse_numeric_expression(period)
        gm_val = parse_numeric_expression(gm) if gm is not None else None
    except ValueError as e:
        typer.echo(f"Error parsing numeric values: {e}")
        raise typer.Exit(code=1)

    multiplier = 1.0
    if units == TimeUnit.seconds:
        multiplier = 1.0
    elif units == TimeUnit.minutes:
        multiplier = 60.0
    elif units == TimeUnit.hours:
        multiplier = 3600.0
    elif units == TimeUnit.days:
        multiplier = 86400.0
    elif units == TimeUnit.years:
        multiplier = 86400.0 * 365.25

    if gm_val is not None:
        typer.echo(
            f"{brahe.semimajor_axis_from_orbital_period_general(period_val * multiplier, gm_val):{format_string}}"
        )
    else:
        typer.echo(
            f"{brahe.semimajor_axis_from_orbital_period(period_val * multiplier):{format_string}}"
        )


@app.command()
def mean_motion(
    semi_major_axis: Annotated[
        str,
        typer.Argument(help="The semi-major axis of the orbit (supports constants)"),
    ],
    gm: Annotated[
        str, typer.Option(help="The gravitational parameter of the central body")
    ] = None,
    format_string: Annotated[
        str, typer.Option("--format", help="The format of the output")
    ] = "f",
):
    # Parse numeric arguments
    try:
        sma = parse_numeric_expression(semi_major_axis)
        gm_val = parse_numeric_expression(gm) if gm is not None else None
    except ValueError as e:
        typer.echo(f"Error parsing numeric values: {e}")
        raise typer.Exit(code=1)

    if gm_val is not None:
        typer.echo(f"{brahe.mean_motion_general(sma, gm_val):{format_string}}")
    else:
        typer.echo(f"{brahe.mean_motion(sma):{format_string}}")


@app.command()
def anomaly_conversion(
    anomaly: Annotated[
        str, typer.Argument(help="The anomaly to convert from (supports expressions)")
    ],
    eccentricty: Annotated[
        str, typer.Argument(help="The eccentricity of the orbit (supports expressions)")
    ],
    input_anomaly: Annotated[
        OrbitalAnomaly, typer.Argument(help="The anomaly to convert from")
    ],
    output_anomaly: Annotated[
        OrbitalAnomaly, typer.Argument(help="The anomaly to convert to")
    ],
    as_degrees: Annotated[
        bool, typer.Option(help="Convert the anomaly to degrees")
    ] = False,
    format_string: Annotated[
        str, typer.Option("--format", help="The format of the output")
    ] = "f",
):
    # Parse numeric arguments
    try:
        anom = parse_numeric_expression(anomaly)
        ecc = parse_numeric_expression(eccentricty)
    except ValueError as e:
        typer.echo(f"Error parsing numeric values: {e}")
        raise typer.Exit(code=1)

    angle_format = (
        brahe.AngleFormat.DEGREES if as_degrees else brahe.AngleFormat.RADIANS
    )

    if input_anomaly == output_anomaly:
        typer.echo(f"{anom:{format_string}}")
        return

    if input_anomaly == OrbitalAnomaly.mean:
        if output_anomaly == OrbitalAnomaly.eccentric:
            typer.echo(
                f"{brahe.anomaly_mean_to_eccentric(anom, ecc, angle_format):{format_string}}"
            )
            return
        elif output_anomaly == OrbitalAnomaly.true:
            typer.echo(
                f"{brahe.anomaly_mean_to_true(anom, ecc, angle_format):{format_string}}"
            )
            return
    elif input_anomaly == OrbitalAnomaly.eccentric:
        if output_anomaly == OrbitalAnomaly.mean:
            typer.echo(
                f"{brahe.anomaly_eccentric_to_mean(anom, ecc, angle_format):{format_string}}"
            )
            return
        elif output_anomaly == OrbitalAnomaly.true:
            typer.echo(
                f"{brahe.anomaly_eccentric_to_true(anom, ecc, angle_format):{format_string}}"
            )
            return
    elif input_anomaly == OrbitalAnomaly.true:
        if output_anomaly == OrbitalAnomaly.mean:
            typer.echo(
                f"{brahe.anomaly_true_to_mean(anom, ecc, angle_format):{format_string}}"
            )
            return
        elif output_anomaly == OrbitalAnomaly.eccentric:
            typer.echo(
                f"{brahe.anomaly_true_to_eccentric(anom, ecc, angle_format):{format_string}}"
            )
            return


@app.command()
def sun_sync_inclination(
    semi_major_axis: Annotated[
        str,
        typer.Argument(help="The semi-major axis of the orbit (supports constants)"),
    ],
    eccentricity: Annotated[
        str, typer.Argument(help="The eccentricity of the orbit (supports expressions)")
    ],
    as_degrees: Annotated[bool, typer.Option(help="Output format in degrees")] = True,
    format_string: Annotated[
        str, typer.Option("--format", help="The format of the output")
    ] = "f",
):
    # Parse numeric arguments
    try:
        sma = parse_numeric_expression(semi_major_axis)
        ecc = parse_numeric_expression(eccentricity)
    except ValueError as e:
        typer.echo(f"Error parsing numeric values: {e}")
        raise typer.Exit(code=1)

    angle_format = (
        brahe.AngleFormat.DEGREES if as_degrees else brahe.AngleFormat.RADIANS
    )
    typer.echo(
        f"{brahe.sun_synchronous_inclination(sma, ecc, angle_format):{format_string}}"
    )


@app.command()
def perigee_velocity(
    semi_major_axis: Annotated[
        str,
        typer.Argument(help="The semi-major axis of the orbit (supports constants)"),
    ],
    eccentricity: Annotated[
        str, typer.Argument(help="The eccentricity of the orbit (supports expressions)")
    ],
    format_string: Annotated[
        str, typer.Option("--format", help="The format of the output")
    ] = "f",
):
    # Parse numeric arguments
    try:
        sma = parse_numeric_expression(semi_major_axis)
        ecc = parse_numeric_expression(eccentricity)
    except ValueError as e:
        typer.echo(f"Error parsing numeric values: {e}")
        raise typer.Exit(code=1)

    typer.echo(f"{brahe.perigee_velocity(sma, ecc):{format_string}}")


@app.command()
def apogee_velocity(
    semi_major_axis: Annotated[
        str,
        typer.Argument(help="The semi-major axis of the orbit (supports constants)"),
    ],
    eccentricity: Annotated[
        str, typer.Argument(help="The eccentricity of the orbit (supports expressions)")
    ],
    format_string: Annotated[
        str, typer.Option("--format", help="The format of the output")
    ] = "f",
):
    # Parse numeric arguments
    try:
        sma = parse_numeric_expression(semi_major_axis)
        ecc = parse_numeric_expression(eccentricity)
    except ValueError as e:
        typer.echo(f"Error parsing numeric values: {e}")
        raise typer.Exit(code=1)

    typer.echo(f"{brahe.apogee_velocity(sma, ecc):{format_string}}")
