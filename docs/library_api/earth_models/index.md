# Earth Models

**Module**: `brahe.earth_models`

Earth models provide atmospheric density and geomagnetic field data required for accurate orbit propagation and satellite operations.

## Atmospheric Density Models

Atmospheric density models compute the density of the Earth's atmosphere at a given location and time, which is essential for drag calculations in low Earth orbit.

### [Harris-Priester](harris_priester.md)
Simple empirical atmospheric density model that accounts for altitude and diurnal variations.

### [NRLMSISE-00](nrlmsise00.md)
Empirical atmospheric model providing temperature and density profiles using space weather data.

## Magnetic Field Models

Geomagnetic field models compute Earth's magnetic field vector at a given location and time, which is essential for magnetorquer control, magnetic attitude determination, and space weather analysis.

### [IGRF-14](igrf.md)
International Geomagnetic Reference Field. Spherical harmonic degree 13, covering 1900--2030.

### [WMMHR-2025](wmmhr.md)
World Magnetic Model High Resolution. Spherical harmonic degree 133 with crustal field detail, covering 2025--2030.

---

## See Also

- [Magnetic Field Models (Learn)](../../learn/earth_models/magnetic_field.md) - Usage guide for IGRF and WMMHR
- [Atmospheric Drag (Learn)](../../learn/orbital_dynamics/drag.md) - Conceptual explanation of drag modeling
- [Space Weather Data](../space_weather/index.md) - Required for NRLMSISE-00
- [Orbital Dynamics Module](../orbit_dynamics/index.md) - Drag acceleration calculations
