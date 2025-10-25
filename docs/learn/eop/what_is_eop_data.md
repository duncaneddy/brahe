# What is EOP Data?

Earth Orientation Parameters are empirically observed, estimated parameters that describe the irregularities in Earth's rotation in space. When combined with their specific related models they provide the mechanism to transform between an Earth-Centered Earth-Fixed (ECEF) reference frame and an Earth-Centered Inertial (ECI) reference frame. These transformations are essential for accurate orbit propagation, coordinate transformations, and other space-related applications.

Earth Orientation Parameters are _stochastic_ meaning that they are random and cannot be predicted with perfect accuracy into the future. Therefore, Earth orientation data is continually observed, estimated, and updated by various international organizations. The International Earth Rotation and Reference Systems Service (IERS) is the primary organization responsible for providing Earth orientation data products and maintaining the associated reference frames and systems.

For example the predicted evolution of the offset between solar time (UT1) and Coordinated Universal Time (UTC) is show below. The difference between UT1 and UTC is primarily driven by variations in Earth's rotation rate, which are influenced by factors such as tidal forces, atmospheric dynamics, and core-mantle interactions. As a result, the UT1-UTC offset exhibits irregular fluctuations that cannot be precisely predicted far into the future.

<!-- <figure markdown="span">
  ![UTC-UT1 Offset](../../figures/fig_ut1_utc_offset_light.html#only-light){ width="600" }
  ![UTC-UT1 Offset](../../figures/fig_ut1_utc_offset_dark.html#only-dark){ width="600" }
  <figcaption>UTC - UT1 Offset</figcaption>
</figure> -->

<!-- ![UTC-UT1 Offset](../../figures/fig_ut1_utc_offset_light.html#only-light){ width="600" }
![UTC-UT1 Offset](../../figures/fig_ut1_utc_offset_dark.html#only-dark){ width="600" } -->

<!-- <picture>
  <source srcset="../../figures/fig_ut1_utc_offset_dark.html" media="(prefers-color-scheme: dark)">
  <source srcset="../../figures/fig_ut1_utc_offset_light.html" media="(prefers-color-scheme: light)">
  <img src="../../figures/fig_ut1_utc_offset_light.html">
</picture> -->

<div class="plotly-embed">
  <iframe class="only-light" src="../../figures/fig_ut1_utc_offset_light.html" loading="lazy"></iframe>
  <iframe class="only-dark"  src="../../figures/fig_ut1_utc_offset_dark.html"  loading="lazy"></iframe>
</div>

??? "Plot Source"

    ``` python title="fig_ut1_utc_offset.py"
    --8<-- "./plots/fig_ut1_utc_offset.py:8"
    ```

## IERS

The [International Earth Rotation Service (IERS)](https://www.iers.org/) was established in 1987 by the International Astronomical Union and the International Union of Geodesy and Geophysics. The IERS provides data on Earth orientation, on the International Celestial Reference System/Frame, and on the International Terrestrial Reference System/Frame. The IERS also maintains conventions containing models, constants, and standards used for modeling Earth orientation.

The IERS deals with reference _systems_ and reference _frames_. A _reference system_ is an idealized mathematical concept for defining a reference used to represent the state of objects in that system. The two primary reference systems developed by the IERS are the International Celestial Reference System (ICRS) and International Terrestrial Reference System (ITRS). The ICRS is an inertial reference system and the one we used to define Earth-centered inertial (**ECI**) reference frames in brahe. The ITRS is a rotating reference system fixed to the Earth and is used to define Earth-centered, Earth-fixed (**ECEF***) reference frames in the packge.

A reference system is a concept and cannot be used directly. For example you can say that you'll represent all coordinates in the world with respect to the the North Star, but to actually use that reference you need to define how to measure positions with respect to it. Therefore the IERS develops _reference frames_, which are specific realizations of a given reference system. A reference frame realization defines the models, standards, and associated data products for users to actually interact and usethat reference system. The primary reference frames of the IERS are the International Celestial Reference Frame (ICRF) and International Terrestrial Reference Frame (ITRF).

The ICRS and ITRS models are defined with respect to the solar system barycenter[^1]. However, for many satellite-specific engineering applications we are primarily concerned with _geocentric_ references, centered at Earth. Therefore, brahe primarily deals with the Geocentric Celestial Reference Frame (GCRF) and Geocentric Terrestrial Reference Frame (GTRF). For most intents and purposes the international and geocentric references are identical as there is no rotation component between ICRS and GCRS (or ITRF and GCRF)[^2]. The transformation between the two reference systems and frames is simply a pure translation.

For a more detailed discussion of reference frames and systems please read [IERS Technical Note 36](https://www.iers.org/SharedDocs/Publikationen/EN/IERS/Publications/tn/TechnNote36/tn36_174.pdf?__blob=publicationFile&v=1) provides an in-depth discussion of the concepts presented and discussed here.

## Earth Orientation Products

The IERS provides various Earth orientation products which are derived from Very Long Baseline Interferometry (VLBI) or a network of terrestrial GPS[^3]  reference stations. The continual observations made by these stations are  combined with specific reference frame realizations (e.g. the IAU 2010  conventions) to model Earth orientation and enable the transformation between  inertial and Earth-fixed reference frames.

The Earth orientation parameter products come in multiple variations, all of which can be found at the [IERS data products site](https://www.iers.org/IERS/EN/DataProducts/EarthOrientationData/eop.html). These variations arise from the selection of precession-nutation model, ITRF realization, the data sources, and data processing time span. There are two precession-nutation models widely in use today: IAU 1980 nutation theory andthe IAU2006/2000A precession-nutation model. The ITRF 2014 realization is the most recent realization and preferred in most cases.

For data products there are two primary distinctions: standard products and long term products. Standard products, which are produced daily, to provide a daily estimate of the past Earth orientation along with forward-looking predictions available for use in planning. Long term data products are only available for past days, and are produced less frequently, but provider higher accurate estimates of Earth orientation. 

For most purposes the standard products provide sufficient accuracy along with the benefit of having fairly accurate forward-looking predictions. Therefore, brahe defaults to using standard Earth Orientation data products wherever possible. Unless otherwise stated or specified, brahe uses IERS standard product generated with respect to IAU 2006/2000A precession-nutation model and consistent with ITRF2014.

## Earth Orientation Parameters

The primary Earth orientation parameters provided by the IERS are _polar motion coefficients_ ($x_p$, $y_p$), UTC-UT1 _time system offset_ ($\Delta_{UTC}$), _celestial pole offsets_ ($dX$, $dY$), and _length of day_ ($LOD$) corrections. These parameters are used in combination with specific models to compute the transformation between ECEF and ECI reference frames.

Brahe defines the `EarthOrientationProvider` trait to provide a common interface for accessing Earth orientation data. There are multiple different types of providers, each with their own use cases. The package includes default data files for ease of use that are sufficient for most purposes.

There is a single, global Earth orientation provider used internally by brahe functions. This global provider can be initialized using one of the provided loading functions. See the [Managing EOP Data](managing_eop_data.md) page for more information on loading and managing Earth orientation data in brahe.

[^1]: A barycenter is the center of mass of two or more bodies. The solar 
system barycenter is the center of mass of the entire solar system. Due to 
significant mass contributions and distances of Jupiter and Saturn, the 
solar system barycenter evolves in time and is sometimes outside of the 
Sun's outer radius.
[^2]: For applications requiring the highest levels of fidelity, the 
equations of motion of an Earth satellite, with respect to the 
GCRS will contain a relativistic Coriolis force due to geodesic precession 
not present in the ICRS. 
[^3]: Now frequently GNSS receivers