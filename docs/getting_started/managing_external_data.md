# Managing External Data

Modern astrodynamics applications often require the use of external data for precision modeling of common phenomena. In particular, information on Earth reference frames, space weather, and planetary ephemerides (positions) are commonly provided through external data files calculated regularly by _product centers_ (e.g. NASA JPL, ESA, etc.) and made available to the public.

While some applications don't need the level of accuracy provided by these data files, they are necessary for working with the current de facto standards in astrodynamics and needed to make your results comparable to others in the field.

For example, say you wanted to compare the computed positions of a satellite in Earth orbit, to do that comparison properly, the position needs to be expressed in the same reference frame. In most cases, that will be the GCRF (Geocentric Celestial Reference Frame) which is defined by the IAU and maintained by the IERS. To use that reference frame, you need to use the latest EOP (Earth Orientation Parameters) data provided by the IERS, which is updated regularly as Earth orientation changes over time. If you don't use EOP data, your computed positions will be in a different reference frame and won't be directly comparable to the standard GCRF positions used by others in the field.

Historically, managing these data files has been a significant source of friction for users of astrodynamics software. You need to know where to get the data, how to download it, configure your software to use it, as well as regularly check for updates and download new versions of the data as they are released. For long-running applications, if you don't regularly update the data, the results degrade in accuracy over time or the application may stop working entirely if there are no data entries for the needed date.

Brahe provides a built-in data management system to handle all of this for you. However, the library still requires the user to be _aware_ of the existence of these data files because it affects modeling outcomes. As such, Brahe requires the user to explicitly load the data files before use, but generally the library handles the rest of the management process for you.

## Types of Data

## Initializing Data Files

## Learning More