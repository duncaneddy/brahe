# Earth Models

**Module**: `brahe.earth_models`

Earth models provide atmospheric density and other environmental data required for accurate orbit propagation and satellite operations.

## Atmospheric Density Models

Atmospheric density models compute the density of the Earth's atmosphere at a given location and time, which is essential for drag calculations in low Earth orbit.

### [Harris-Priester](harris_priester.md)
Simple empirical atmospheric density model that accounts for altitude and diurnal variations.

### [NRLMSISE-00](nrlmsise00.md)
Empirical atmospheric model providing temperature and density profiles using space weather data.

---

## See Also

- [Atmospheric Drag (Learn)](../../learn/orbital_dynamics/drag.md) - Conceptual explanation of drag modeling
- [Space Weather Data](../space_weather/index.md) - Required for NRLMSISE-00
- [Orbital Dynamics Module](../orbit_dynamics/index.md) - Drag acceleration calculations
