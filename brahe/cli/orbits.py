from enum import Enum
import typer
from typing_extensions import Annotated

import brahe

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
def orbital_period(semi_major_axis: Annotated[float, typer.Argument(help="The semi-major axis of the orbit")],
                   gm: Annotated[float, typer.Option(help="The gravitational parameter of the central body")] = None,
                   units: Annotated[TimeUnit, typer.Option(help="The time units of the output")] = TimeUnit.seconds,
                   format_string: Annotated[str, typer.Option("--format", help="The format of the output")] = "f"):

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

    if gm is not None:
        typer.echo(f"{brahe.orbital_period_general(semi_major_axis, gm)/divisor:{format_string}}")
    else:
        typer.echo(f"{brahe.orbital_period(semi_major_axis)/divisor:{format_string}}")

@app.command()
def sma_from_period(period: Annotated[float, typer.Argument(help="The orbital period")],
                    units: Annotated[TimeUnit, typer.Option(help="The time units of the input")] = TimeUnit.seconds,
                    gm: Annotated[float, typer.Option(help="The gravitational parameter of the central body")] = None,
                    format_string: Annotated[str, typer.Option("--format", help="The format of the output")] = "f"):

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

    if gm is not None:
        typer.echo(f"{brahe.semimajor_axis_from_orbital_period_general(period*multiplier, gm):{format_string}}")
    else:
        typer.echo(f"{brahe.semimajor_axis_from_orbital_period(period*multiplier):{format_string}}")

@app.command()
def mean_motion(semi_major_axis: Annotated[float, typer.Argument(help="The semi-major axis of the orbit")],
                   gm: Annotated[float, typer.Option(help="The gravitational parameter of the central body")] = None,
                   format_string: Annotated[str, typer.Option("--format", help="The format of the output")] = "f"):

    if gm is not None:
        typer.echo(f"{brahe.mean_motion_general(semi_major_axis, gm):{format_string}}")
    else:
        typer.echo(f"{brahe.mean_motion(semi_major_axis):{format_string}}")


@app.command()
def anomaly_conversion(
        anomaly: Annotated[float, typer.Argument(help="The anomaly to convert from")],
        eccentricty: Annotated[float, typer.Argument(help="The eccentricity of the orbit")],
        input_anomaly: Annotated[OrbitalAnomaly, typer.Argument(help="The anomaly to convert from")],
        output_anomaly: Annotated[OrbitalAnomaly, typer.Argument(help="The anomaly to convert to")],
        as_degrees: Annotated[bool, typer.Option(help="Convert the anomaly to degrees")] = False,
        format_string: Annotated[str, typer.Option("--format", help="The format of the output")] = "f"):

    if input_anomaly == output_anomaly:
        typer.echo(f"{anomaly:{format_string}}")
        return

    if input_anomaly == OrbitalAnomaly.mean:
        if output_anomaly == OrbitalAnomaly.eccentric:
            typer.echo(f"{brahe.anomaly_mean_to_eccentric(anomaly, eccentricty, as_degrees):{format_string}}")
            return
        elif output_anomaly == OrbitalAnomaly.true:
            typer.echo(f"{brahe.anomaly_mean_to_true(anomaly, eccentricty, as_degrees):{format_string}}")
            return
    elif input_anomaly == OrbitalAnomaly.eccentric:
        if output_anomaly == OrbitalAnomaly.mean:
            typer.echo(f"{brahe.anomaly_eccentric_to_mean(anomaly, eccentricty, as_degrees):{format_string}}")
            return
        elif output_anomaly == OrbitalAnomaly.true:
            typer.echo(f"{brahe.anomaly_eccentric_to_true(anomaly, eccentricty, as_degrees):{format_string}}")
            return
    elif input_anomaly == OrbitalAnomaly.true:
        if output_anomaly == OrbitalAnomaly.mean:
            typer.echo(f"{brahe.anomaly_true_to_mean(anomaly, eccentricty, as_degrees):{format_string}}")
            return
        elif output_anomaly == OrbitalAnomaly.eccentric:
            typer.echo(f"{brahe.anomaly_true_to_eccentric(anomaly, eccentricty, as_degrees):{format_string}}")
            return


@app.command()
def sun_sync_inclination(
        semi_major_axis: Annotated[float, typer.Argument(help="The semi-major axis of the orbit")],
        eccentricity: Annotated[float, typer.Argument(help="The eccentricity of the orbit")],
        as_degrees: Annotated[bool, typer.Option(help="Output format in degrees")] = True,
        format_string: Annotated[str, typer.Option("--format", help="The format of the output")] = "f"):

    typer.echo(f"{brahe.sun_synchronous_inclination(semi_major_axis, eccentricity, as_degrees):{format_string}}")

@app.command()
def perigee_velocity(
        semi_major_axis: Annotated[float, typer.Argument(help="The semi-major axis of the orbit")],
        eccentricity: Annotated[float, typer.Argument(help="The eccentricity of the orbit")],
        format_string: Annotated[str, typer.Option("--format", help="The format of the output")] = "f"):
    typer.echo(f"{brahe.perigee_velocity(semi_major_axis, eccentricity):{format_string}}")

@app.command()
def apogee_velocity(
        semi_major_axis: Annotated[float, typer.Argument(help="The semi-major axis of the orbit")],
        eccentricity: Annotated[float, typer.Argument(help="The eccentricity of the orbit")],
        format_string: Annotated[str, typer.Option("--format", help="The format of the output")] = "f"):
    typer.echo(f"{brahe.apogee_velocity(semi_major_axis, eccentricity):{format_string}}")