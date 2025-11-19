# Space Weather Data

Space weather data provides geomagnetic indices and solar flux measurements that are essential for atmospheric density models used in satellite drag calculations. These parameters capture solar activity and geomagnetic disturbances that directly affect upper atmosphere density and satellite orbital decay.

!!! warning "Experimental API"
    The space weather module is currently experimental. While the core functionality should be stable, the API may change in future **MINOR** releases as we refine the design and add features.

## Overview

The primary space weather parameters used in astrodynamics are:

<div class="center-table" markdown="1">
| Parameter | Description | Range | Units |
|-----------|-------------|-------|-------|
| **Kp** | Planetary geomagnetic index | 0-9 | dimensionless |
| **Ap** | Daily amplitude geomagnetic index | 0-400 | nT (nanotesla) |
| **F10.7** | 10.7 cm solar radio flux | 60-400 | sfu (solar flux units) |
| **ISN** | International Sunspot Number | 0-400 | count |
</div>

## F10.7 Solar Radio Flux

The F10.7 index measures solar radio emissions at 10.7 cm wavelength and serves as a proxy for solar radiation that heats the upper atmosphere. Higher F10.7 values indicate increased solar activity and higher atmospheric density.

The plot below shows the historical F10.7 observed flux from 1957 to present, with future predicted values shown in red.

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/fig_f107_history_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/fig_f107_history_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="fig_f107_history.py"
    --8<-- "./plots/fig_f107_history.py:10"
    ```

The approximately 11-year solar cycle is clearly visible in the data, with peaks corresponding to solar maximum periods.

## Geomagnetic Indices

**Kp** and **Ap** indices measure geomagnetic activity caused by solar wind interactions with Earth's magnetosphere.
## Data Source

Brahe uses CSSI space weather data files provided by [CelesTrak](https://celestrak.com/SpaceData/). The data includes:

- **Historical observations** from October 1957 to present
- **Daily predictions** for the near term
- **Monthly predictions** extending further into the future

## Managing Space Weather in Brahe

Brahe provides three provider types for space weather data:

- **CachingSpaceWeatherProvider**: Auto-downloads and caches latest data (recommended)
- **FileSpaceWeatherProvider**: Loads from CSSI format files
- **StaticSpaceWeatherProvider**: Uses fixed values for testing

See: [Managing Space Weather Data](managing_space_weather_data.md)

---

## See Also

- [Managing Space Weather Data](managing_space_weather_data.md) - Practical guide to providers
- [Space Weather API Reference](../../library_api/space_weather/index.md) - Complete API documentation
