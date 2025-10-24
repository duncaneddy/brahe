from enum import Enum
import typer
from typing_extensions import Annotated
from loguru import logger

import brahe

app = typer.Typer()


class TimeSystem(str, Enum):
    UTC = "UTC"
    GPS = "GPS"
    TAI = "TAI"
    UT1 = "UT1"
    TT = "TT"


class EpochFormat(str, Enum):
    mjd = "mjd"
    jd = "jd"
    string = "string"
    gps_date = "gps_date"
    gps_nanoseconds = "gps_nanoseconds"


@app.command()
def convert(
    epoch: Annotated[
        str, typer.Argument(..., help="Epoch to convert the time representation for")
    ],
    input_format: Annotated[
        EpochFormat, typer.Argument(..., help="Input format of the epoch")
    ],
    output_format: Annotated[
        EpochFormat, typer.Argument(..., help="Desired output format of the epoch")
    ],
    input_time_system: Annotated[
        TimeSystem, typer.Option(help="Time system of the input epoch")
    ] = None,
    output_time_system: Annotated[
        TimeSystem, typer.Option(help="Time system of the output epoch")
    ] = None,
):
    logger.info(f"Converting epoch from {input_format.value} to {output_format.value}")
    logger.debug(f"Input: {epoch}")
    if input_time_system and input_format in [
        EpochFormat.gps_date,
        EpochFormat.gps_nanoseconds,
    ]:
        typer.echo("Input time system is not supported for GPS date or nanoseconds")
        typer.Exit(code=1)

    if output_time_system and output_format in [
        EpochFormat.gps_date,
        EpochFormat.gps_nanoseconds,
    ]:
        typer.echo("Output time system is not supported for GPS date or nanoseconds")
        typer.Exit(code=1)

    # Set input time system if unset
    if not input_time_system:
        input_time_system = TimeSystem.TAI

    # Set output time system if unset
    if not output_time_system:
        output_time_system = TimeSystem.TAI

    if input_format == EpochFormat.string:
        epc = brahe.Epoch.from_string(epoch)
    elif input_format == EpochFormat.mjd:
        epc = brahe.Epoch.from_mjd(float(epoch), input_time_system.value)
    elif input_format == EpochFormat.jd:
        epc = brahe.Epoch.from_jd(float(epoch), input_time_system.value)
    elif input_format == EpochFormat.gps_date:
        epc = brahe.Epoch.from_gps_date(int(epoch))
    elif input_format == EpochFormat.gps_nanoseconds:
        epc = brahe.Epoch.from_gps_nanoseconds(int(epoch))
    else:
        typer.echo("Invalid input format")
        typer.Exit(code=1)

    # Convert CLI TimeSystem enum to brahe TimeSystem enum
    brahe_output_ts = getattr(brahe.TimeSystem, output_time_system.value)

    if output_format == EpochFormat.string:
        typer.echo(epc.to_string_as_time_system(time_system=brahe_output_ts))
    elif output_format == EpochFormat.mjd:
        typer.echo(epc.mjd_as_time_system(time_system=brahe_output_ts))
    elif output_format == EpochFormat.jd:
        typer.echo(epc.jd_as_time_system(time_system=brahe_output_ts))
    elif output_format == EpochFormat.gps_date:
        typer.echo(epc.gps_date())
    elif output_format == EpochFormat.gps_nanoseconds:
        typer.echo(epc.gps_nanoseconds())
    else:
        typer.echo("Invalid output format")
        typer.Exit(code=1)


@app.command()
def add(
    epoch: Annotated[str, typer.Argument(..., help="Epoch to add time to")],
    seconds: Annotated[float, typer.Argument(..., help="Seconds to add")],
    output_format: Annotated[
        EpochFormat, typer.Option(help="Desired output format of the epoch")
    ] = EpochFormat.string,
    output_time_system: Annotated[
        TimeSystem, typer.Option(help="Time system of the output epoch")
    ] = TimeSystem.UTC,
):
    epc = brahe.Epoch(epoch)

    epc += seconds

    # Convert CLI TimeSystem enum to brahe TimeSystem enum
    brahe_output_ts = getattr(brahe.TimeSystem, output_time_system.value)

    if output_format == EpochFormat.string:
        typer.echo(epc.to_string_as_time_system(time_system=brahe_output_ts))
    elif output_format == EpochFormat.mjd:
        typer.echo(epc.mjd_as_time_system(time_system=brahe_output_ts))
    elif output_format == EpochFormat.jd:
        typer.echo(epc.jd_as_time_system(time_system=brahe_output_ts))
    elif output_format == EpochFormat.gps_date:
        typer.echo(epc.gps_date())
    elif output_format == EpochFormat.gps_nanoseconds:
        typer.echo(epc.gps_nanoseconds())
    else:
        typer.echo("Invalid output format")
        typer.Exit(code=1)


@app.command()
def time_system_offset(
    epoch: Annotated[str, typer.Argument(..., help="Epoch-like to get EOP data for")],
    source: Annotated[
        TimeSystem, typer.Argument(..., help="Time system to convert from")
    ],
    target: Annotated[
        TimeSystem, typer.Argument(..., help="Time system to convert to")
    ],
):
    epc = brahe.Epoch(epoch)

    # Convert CLI TimeSystem enums to brahe TimeSystem enums
    brahe_source = getattr(brahe.TimeSystem, source.value)
    brahe_target = getattr(brahe.TimeSystem, target.value)

    offset = brahe.time_system_offset_for_mjd(epc.mjd(), brahe_source, brahe_target)

    typer.echo(f"{offset}")


@app.command(name="range")
def time_range(
    epoch_start: Annotated[
        str, typer.Argument(..., help="Epoch-like for start of time range")
    ],
    epoch_end: Annotated[
        str, typer.Argument(..., help="Epoch-like for end of time range")
    ],
    step: Annotated[float, typer.Argument(..., help="Step size in seconds")],
):
    epc_start = brahe.Epoch(epoch_start)
    epc_end = brahe.Epoch(epoch_end)

    for epc in brahe.TimeRange(epc_start, epc_end, step):
        typer.echo(epc)
