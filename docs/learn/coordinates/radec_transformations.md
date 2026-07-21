# Right Ascension / Declination Transformations

Right ascension ($\alpha$) and declination ($\delta$) are the standard spherical coordinates for describing directions in an inertial frame: $\alpha$ is measured eastward in the fundamental plane from a reference direction, and $\delta$ is measured perpendicular to that plane. Brahe's RA/Dec functions are **frame-agnostic** - they convert between the spherical $(\alpha, \delta, r)$ representation and whatever Cartesian inertial frame the caller supplies. For star catalog positions (FK5, Hipparcos, Tycho-2) that frame is ICRS/GCRF; for other applications it can be any inertial frame consistent with the input Cartesian state.

For complete API details, see the [RA/Dec Coordinates API Reference](../../library_api/coordinates/radec.md).

## Position Conversions

A position at range $r$ with right ascension $\alpha$ and declination $\delta$ maps to the Cartesian inertial position:

$$
\vec{r} = r \begin{bmatrix} \cos\delta\cos\alpha \\ \cos\delta\sin\alpha \\ \sin\delta \end{bmatrix}
$$

and the inverse recovers $(\alpha, \delta, r)$ from $\vec{r} = [x, y, z]$:

$$
r = \lVert \vec{r} \rVert, \qquad \delta = \arcsin\left(\frac{z}{r}\right), \qquad \alpha = \operatorname{atan2}(y, x)
$$

Right ascension is normalized to $[0, 360)$ degrees (or $[0, 2\pi)$ radians). At the polar singularity ($x = y = 0$), $\alpha$ is indeterminate from position alone and `position_inertial_to_radec` returns `0`; use the state-based conversion below to resolve it from velocity instead.

=== "Python"

    ``` python
    --8<-- "./examples/coordinates/radec_to_inertial.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/coordinates/radec_to_inertial.rs:4"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/coordinates/radec_to_inertial.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/coordinates/radec_to_inertial.rs.txt"
        ```

## State Conversions

Including velocity extends the position relations to their rates. Converting $(\alpha, \delta, r, \dot\alpha, \dot\delta, \dot r)$ to Cartesian state:

$$
\dot{x} = \dot r \cos\delta\cos\alpha - r\sin\delta\cos\alpha\,\dot\delta - r\cos\delta\sin\alpha\,\dot\alpha
$$

$$
\dot{y} = \dot r \cos\delta\sin\alpha - r\sin\delta\sin\alpha\,\dot\delta + r\cos\delta\cos\alpha\,\dot\alpha
$$

$$
\dot{z} = \dot r \sin\delta + r\cos\delta\,\dot\delta
$$

with $x, y, z$ given by the Eq. 4-1 position relation above. The inverse (Cartesian state to $\alpha, \delta, r$ and rates) resolves the polar singularity using velocity rather than returning $\alpha = 0$:

$$
\delta = \arcsin\left(\frac{r_k}{r}\right), \qquad
\alpha = \begin{cases}
\operatorname{atan2}(r_j, r_i) & r_i^2 + r_j^2 > \epsilon \\[4pt]
\operatorname{atan2}(v_j, v_i) & r_i^2 + r_j^2 \le \epsilon \ \text{(polar singularity)}
\end{cases}
$$

$$
\dot r = \frac{r_i v_i + r_j v_j + r_k v_k}{r}, \qquad
\dot\alpha = \frac{v_j r_i - v_i r_j}{r_i^2 + r_j^2}, \qquad
\dot\delta = \frac{v_k - \dot r \, (r_k / r)}{\sqrt{r_i^2 + r_j^2}}
$$

=== "Python"

    ``` python
    --8<-- "./examples/coordinates/radec_state_transforms.py:state_transforms"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/coordinates/radec_state_transforms.rs:state_transforms"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/coordinates/radec_state_transforms.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/coordinates/radec_state_transforms.rs.txt"
        ```

## Topocentric Right Ascension/Declination

The RA/Dec conversions above assume the input position/state is already relative to the frame's origin. For a ground-based observation, the object's position must first be made relative to the observing site: subtract the site's inertial position (or state) from the object's before converting.

This subtract-then-convert pattern is the vector form of Vallado Algorithm 26 (*Topocentric*): applying `state_inertial_to_radec` to the slant-range vector $\vec{r}_{\text{sat}} - \vec{r}_{\text{site}}$ is equivalent to running Algorithm 25 directly on that vector, because the topocentric frame's axes are parallel to the geocentric inertial frame - only the origin is translated to the site.

=== "Python"

    ``` python
    --8<-- "./examples/coordinates/radec_state_transforms.py:topocentric"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/coordinates/radec_state_transforms.rs:topocentric"
    ```

This continues the same `radec_state_transforms` example above - see its output for the resulting topocentric line of sight.

For stars, which are effectively at infinite range, this correction is unnecessary: the geocentric catalog $(\alpha, \delta)$ and the topocentric $(\alpha, \delta)$ are the same to any achievable precision, so [`position_radec_to_azel`](../../library_api/coordinates/radec.md) can be called directly on the catalog values.

## Right Ascension/Declination ↔ Azimuth-Elevation

`position_radec_to_azel` and `position_azel_to_radec` rotate a topocentric line-of-sight direction between the equatorial $(\alpha, \delta)$ representation and the local horizon azimuth-elevation representation. Both conversions are **direction-only**: no parallax translation between the geocenter and the site is applied (the site's altitude does not affect the result), `range` passes through unchanged, and a global EOP provider must be initialized for the inertial ↔ Earth-fixed rotation.

=== "Python"

    ``` python
    --8<-- "./examples/coordinates/radec_to_azel.py:9"
    ```

=== "Rust"

    ``` rust
    --8<-- "./examples/coordinates/radec_to_azel.rs:5"
    ```

??? example "Output"
    === "Python"
        ```
        --8<-- "./docs/outputs/coordinates/radec_to_azel.py.txt"
        ```

    === "Rust"
        ```
        --8<-- "./docs/outputs/coordinates/radec_to_azel.rs.txt"
        ```

!!! info
    Azimuth is measured clockwise from North. This is the same convention used by the [Topocentric Coordinates](topocentric_transformations.md) ENZ/SEZ-to-azimuth-elevation functions, so `position_radec_to_azel` results are directly comparable to `position_enz_to_azel`/`position_sez_to_azel` results.

## Proper Motion

Catalog positions are only valid at their reference epoch. `apply_proper_motion` propagates $(\alpha, \delta)$ from a catalog epoch to a target epoch using IAU SOFA's `iauPmsafe` space-motion routine, given the star's proper motion and, when available, its parallax and radial velocity.

`iauPmsafe` reconstructs the star's full barycentric position/velocity (pv-)state from the catalog $(\alpha, \delta)$, proper motion, parallax, and radial velocity; advances it assuming **straight-line motion at constant velocity** - including a light-time correction and the special-relativistic (Doppler) treatment of Stumpff (1985) - and reduces the result back to catalog $(\alpha, \delta)$ at the target epoch. The `pm_ra` argument follows the standard catalog convention $\mu_{\alpha*} = \mu_\alpha \cos\delta$ (matching the `pmRA`/`pmDE` columns of Hipparcos, Tycho-2, Gaia, and most other catalogs), not the raw coordinate rate $\dot\alpha$. When `parallax` or `radial_velocity` is unavailable it is treated as zero; `iauPmsafe` additionally applies a proper-motion-scaled minimum-parallax guard so a star with a missing or tiny parallax still propagates correctly rather than being clamped to a no-op.

### Underlying model

To first order the direction change `iauPmsafe` produces is the rigorous epoch transformation of ESA SP-1200 §1.5.5. The star's unit direction $\hat{u}_0$ moves in the tangent plane spanned by

$$
\hat{u}_0 = \begin{bmatrix} \cos\delta\cos\alpha \\ \cos\delta\sin\alpha \\ \sin\delta \end{bmatrix}, \qquad
\hat{p} = \begin{bmatrix} -\sin\alpha \\ \cos\alpha \\ 0 \end{bmatrix}, \qquad
\hat{q} = \begin{bmatrix} -\sin\delta\cos\alpha \\ -\sin\delta\sin\alpha \\ \cos\delta \end{bmatrix}
$$

driven by the tangential proper-motion vector $\vec{\mu} = \hat{p}\,\mu_{\alpha*} + \hat{q}\,\mu_\delta$ (with $\mu_{\alpha*}$, $\mu_\delta$ converted from mas/yr to rad/yr) and a radial "perspective-acceleration" term $\mu_r$ that captures the change in angular rate as the star's line-of-sight distance changes:

$$
\mu_r = \frac{v_r \, \varpi_{\text{rad}}}{4.740470446\ \text{km/s per AU/yr}} \qquad \left[\text{yr}^{-1}\right]
$$

where $v_r$ is the radial velocity (km/s), $\varpi_{\text{rad}}$ is the parallax in radians, and $4.740470446\ \text{km/s}$ is the speed of one astronomical unit per year. The perspective term is significant only for high radial-velocity, high-parallax stars such as Barnard's Star; when parallax or radial velocity is unknown it vanishes and the propagation reduces to purely linear proper motion. `iauPmsafe` goes beyond this first-order model by carrying the full pv-state and adding the light-time and Doppler corrections noted above.

The worked example above (Position Conversions) continues past the round-trip check to call `apply_proper_motion` on Barnard's Star, propagating it 10 years forward using its Hipparcos catalog proper motion, parallax, and radial velocity - see its output for the resulting $(\alpha, \delta)$ shift.

!!! info "Reference"
    Proper motion propagation is performed by IAU SOFA's `iauPmsafe` (SOFA Tools for Earth Attitude, 2023); the underlying epoch-transformation theory is ESA, *The Hipparcos and Tycho Catalogues*, ESA SP-1200, Vol. 1, §1.5.5, 1997. The RA/Dec position, state, and topocentric conversions follow D. Vallado, *Fundamentals of Astrodynamics and Applications*, 4th Ed., §4.4 (Eq. 4-1, Eq. 4-2, Algorithm 25, Algorithm 26), 2013.

    Star catalog records expose this transformation directly via `radec_at_epoch` - see [Star Catalogs](../datasets/star_catalogs.md).

---

## See Also

- [RA/Dec Coordinates API Reference](../../library_api/coordinates/radec.md) - Complete function documentation
- [Star Catalogs](../datasets/star_catalogs.md) - FK5, Hipparcos, and Tycho-2 catalog records that use these conversions
- [Topocentric Transformations](topocentric_transformations.md) - ENZ/SEZ frames and azimuth-elevation conversions
- [Cartesian Transformations](cartesian_transformations.md) - Orbital elements and Cartesian states
