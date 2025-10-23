# Command Line Interface

Brahe includes a command-line interface (CLI) for common astrodynamics operations. The CLI provides quick access to calculations, data downloads, and conversions without writing Python code.

## Installation

The CLI is automatically installed when you install the Brahe Python package:

```bash
pip install brahe
```

## Basic Usage

The CLI is invoked using the `brahe` command followed by a subcommand:

```bash
brahe <subcommand> <command> [arguments] [options]
```

## Available Subcommands

Brahe's CLI is organized into the following subcommands:

- **[`eop`](eop.md)** - Earth Orientation Parameter (EOP) data management
- **[`time`](time.md)** - Time system conversions and operations
- **[`orbits`](orbits.md)** - Orbital mechanics calculations
- **[`convert`](convert.md)** - Coordinate and reference frame conversions
- **[`datasets`](datasets.md)** - Download satellite data and groundstation information
- **[`access`](access.md)** - Access window computation (future)

## Getting Help

View available subcommands:

```bash
brahe --help
```

View help for a specific subcommand:

```bash
brahe eop --help
brahe time --help
brahe orbits --help
```

View help for a specific command:

```bash
brahe eop download --help
brahe time convert --help
```

## Common Patterns

### Piping and Output

CLI commands output results to stdout, making them easy to use in shell pipelines:

```bash
# Calculate orbital period and save to file
brahe orbits orbital-period 7000000 > period.txt

# Convert time format and use in another command
epoch=$(brahe time convert "2024-01-01T00:00:00" string mjd)
echo "MJD: $epoch"
```

### Using with Scripts

The CLI integrates seamlessly with shell scripts:

```bash
#!/bin/bash
# Download latest TLE data
brahe datasets celestrak download satellites.json --group active --file-format json

# Calculate orbital periods for different altitudes
for alt in 400000 500000 600000; do
    sma=$((6378137 + alt))
    period=$(brahe orbits orbital-period $sma --units minutes)
    echo "Altitude ${alt}m: ${period} minutes"
done
```

### Working with Epochs

Many commands accept "epoch-like" strings that are automatically parsed:

```bash
# ISO 8601 string
brahe time convert "2024-01-01T12:00:00" string mjd

# Modified Julian Date
brahe time convert 60310.5 mjd string

# Julian Date
brahe time convert 2460310.5 jd string
```

## Output Formatting

Many commands support the `--format` option to control numeric output:

```bash
# Default floating point
brahe orbits orbital-period 7000000
# Output: 5933.015427

# Scientific notation
brahe orbits orbital-period 7000000 --format e
# Output: 5.933015e+03

# Fixed precision
brahe orbits orbital-period 7000000 --format .2f
# Output: 5933.02
```

## Examples

### Download EOP Data

```bash
# Download standard EOP data
brahe eop download eop_data.txt --product standard

# Download C04 EOP data
brahe eop download eop_c04.txt --product c04
```

### Time Conversions

```bash
# Convert UTC to GPS time
brahe time convert "2024-01-01T00:00:00" string mjd --output-time-system GPS

# Add time to an epoch
brahe time add "2024-01-01T00:00:00" 3600 --output-format string
```

### Orbital Calculations

```bash
# Calculate orbital period (in minutes)
brahe orbits orbital-period 7000000 --units minutes

# Calculate semi-major axis from period
brahe orbits sma-from-period 90 --units minutes

# Sun-synchronous inclination
brahe orbits sun-sync-inclination 7000000 0.001
```

### Coordinate Conversions

```bash
# Convert Keplerian to Cartesian
brahe convert coordinates 7000000 0.001 97.8 0 0 0 keplerian cartesian

# Convert geodetic to ECEF
brahe convert coordinates 40.0 -105.0 1000 0 0 0 geodetic ecef
```

### Dataset Downloads

```bash
# Download active satellites from CelesTrak
brahe datasets celestrak download active.json --group active --file-format json

# Show KSAT groundstations
brahe datasets groundstations show ksat

# List all groundstation providers
brahe datasets groundstations list
```

## Error Handling

The CLI returns appropriate exit codes:

- `0` - Success
- `1` - Error (with error message printed to stderr)

Example error handling in scripts:

```bash
if brahe eop download eop.txt --product standard; then
    echo "Download successful"
else
    echo "Download failed" >&2
    exit 1
fi
```

## Next Steps

- Explore individual subcommands in the navigation menu
- Check out [Examples](../../examples/index.md) for complete workflows
- See the [Python API Reference](../../library_api/index.md) for programmatic access
