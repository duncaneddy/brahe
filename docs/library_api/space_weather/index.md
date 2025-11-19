# Space Weather Data

**Module**: `brahe.space_weather`

Space weather data provides geomagnetic indices and solar flux values required for atmospheric density models used in satellite drag calculations and orbit propagation with atmospheric perturbations.

## Overview

Space weather data includes:
- **Kp/Ap indices**: Geomagnetic activity indices (8 values per day for 3-hour intervals)
- **F10.7**: 10.7 cm solar radio flux (solar flux units, sfu)
- **Sunspot Number**: International Sunspot Number (ISN)

## Global Space Weather Management

Space weather data is managed globally to avoid passing providers through every function call.

### [Functions](functions.md)
- Setting global space weather providers
- Querying Kp/Ap indices
- Querying F10.7 solar flux
- Accessing historical data

## Space Weather Providers

Brahe supports three types of space weather providers:

### [CachingSpaceWeatherProvider](caching_provider.md)
Automatically download and cache the latest CSSI space weather data from CelesTrak with configurable refresh intervals.

### [FileSpaceWeatherProvider](file_provider.md)
Load space weather data from CSSI format files for production applications with current data.

### [StaticSpaceWeatherProvider](static_provider.md)
Use user-defined fixed values, ideal for testing, offline use, or applications with known constant conditions.

---

## See Also

- [Managing Space Weather Data](../../learn/space_weather/managing_space_weather_data.md) - Practical guide to providers
- [Space Weather Learn](../../learn/space_weather/index.md) - Conceptual overview
