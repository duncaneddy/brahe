# Reference Frames

Reference frame transformations are a fundamental aspect of astrodynamics. Different tasks require working in different reference frames, and accurate transformations between these frames are essential for precise calculations.

Brahe uses the [IAU SOFA](https://www.iausofa.org/) (Standards of Fundamental Astronomy) C library for reference frame transformations to provide speed, accuracy, and reliability. To learn more about these models, refer to the [IERS Conventions (2010)](https://www.iers.org/IERS/EN/Publications/TechnicalNotes/tn36.html).

## Reference Frame Types

### Inertial Frames (Non-Rotating)

Inertial reference frames are fixed with respect to distant stars and do not rotate. They are ideal for integrating equations of motion as they do not require the addition of fictitious forces.

Inertial reference frames currently supported in Brahe are:

- **GCRF (Geocentric Celestial Reference Frame)**: The standard modern inertial reference frame for Earth-orbiting satellites, aligned with the International Celestial Reference Frame (ICRF)
- **EME2000 (Earth Mean Equator and Equinox of J2000.0)**: Classical J2000.0 mean equator and mean equinox inertial frame. Derived from the FK5 catalog and widely used in older systems

### Earth-Fixed Frames (Rotating)

Earth-fixed reference frames rotate with the Earth and are ideal for computing positions and motions relative to terrestrial locations and observers.

Earth-fixed reference frames currently supported in Brahe are:

- **ITRF (International Terrestrial Reference Frame)**: The standard Earth-fixed reference frame maintained by IERS, rotating with the Earth and aligned with geographic coordinates

## Available Transformations

We can visualize the relations between the relevants reference frames and their transformations as follows

<div class="plotly-embed">
    <img class="only-light" src="../../assets/reference_frames_light.svg" alt="Frame transformation chains" loading="lazy">
    <img class="only-dark" src="../../assets/reference_frames_dark.svg" alt="Frame transformation chains" loading="lazy">
</div>

The CIRS and TIRS frames are intermediate steps in the transformation process and are not directly exposed in Brahe's API. If you need to work with these frames, you can construct them manually using the provided functions for bias-precession-nutation, Earth rotation, and polar motion. Refer directly to the source code for details.

!!! tip "Which Methods Should I Use?"

    If you always want to use the most accurate and up-to-date reference frame transformations, use the **ECI ↔ ECEF** functions. These functions will always map to the best available transformations in Brahe.

    If you want to make sure your results are reproducible and consistent over time, use the explicit **GCRF ↔ ITRF** functions. This ensures that your code will always use the same transformation models, even if Brahe introduces improved models in the future.

### ECI ↔ ECEF (Common Naming)

Generic "Earth-Centered Inertial" and "Earth-Centered Earth-Fixed" naming convention that currently maps to GCRF and ITRF using conventions. This naming is widely used in the astrodynamics community.

The ECI/ECEF naming is provided as a convenient alias for the commonly used terminology, and seeks to always provide the "best" available transformation between inertial and Earth-fixed frames. The transformation that underpins ECI ↔ ECEF conversions is currently the IAU 2010 GCRF ↔ ITRF transformations, but may differ in the future if improved reference frame models are introduced and adopted.

Learn more in [ECI ↔ ECEF Naming Convention](eci_ecef.md)

### GCRF ↔ ITRF

The primary transformation for modern applications, converting between the inertial GCRF and Earth-fixed ITRF frames. The transformation is accomplished using the IAU 2006/2000A, CIO-based theory using classical angles. The method as described in section 5.5 of the [SOFA C transformation cookbook](https://www.iausofa.org/s/sofa_pn_c.pdf). This transformation accounts for:

- Earth's rotation
- Polar motion
- Precession and nutation effects

Learn more in [GCRF ↔ ITRF Transformations](gcrf_itrf.md)

### EME2000 ↔ GCRF

A constant frame bias transformation between the classical J2000.0 frame (Earth Equator and Mean Equinox) and the modern ICRS-aligned GCRF. The transformation is accomplished using the second-order frame bias rotation matrix as described in [Astrodynamics Convention and Modeling Reference for Lunar, Cislunar, and Libration Point Orbits by Folta et al.](https://ntrs.nasa.gov/api/citations/20220014814/downloads/NASA%20TP%2020220014814%20final.pdf), section 4.3.5.

Learn more in [EME2000 ↔ GCRF Transformations](eme2000_gcrf.md)
