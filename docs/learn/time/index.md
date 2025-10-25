# Time Systems and Representations

!!! quote ""
    In the beginning the Universe was created. This has made a lot of people very angry and has been widely regarded as a bad move

    - Douglas Adams

Astrodynamics is the study of space motion, which inherently involves the concept of time. Accurate timekeeping and conversions between different time systems are crucial for precise calculations in orbital mechanics.

The _time_ module provides functions for the handling of time. The package makes the distinction between the _representation_ of a specific time and the _conversion_ between different time scales. Precise specification of an instant in time requires the specification of both a time representation and time scale.

## Time Representation

A single instance in time can be represented in multiple different formats. For examples The J2000 Epoch can be represented as a calendar date in terms of years, months, days, hours, minutes, and seconds as `2000-01-01T12:00:00`. The same instant can also be represented in terms of Modified Julian Days as `51544.5`. Both of these representations refer to the same _instant_ in time.

## Time Scales

In addition to representing time in different manners, there are also different time scales. A _time scale_ is a standard to reckoning and resolving instances in time. Multimple time scales have been introduced due to the criticality of being able to correctly measure and understand when specific events occur in science and engineering.  

Within a time scale it is possible to compare different instances in time to determine if one is before, after, or at the same time as another instant. It is also possible to compare between time scales, however you must know how to properly convert between them. It is assumed that all time scales use the same definition of the SI second, and therefore advance at the same rate.

!!! info ""

    Athough the calendar date representations of time `2000-01-01T12:00:00 UTC` and `2000-01-01T12:00:00 GPS` have the same values, they are actually  _different instances in time!_. This is because while the calendar date representations are the same there are actually offsets between the different time scales.

The time scales currently supported by brahe are

| Time Scale | Description                                                                                                                                                                                                                                                                                           |
|------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `GPS`      | `GPS` stands for Global Positioning System time. It is specifically, the time scale used by the US GPS satellites. It is different from TAI and TT by constant, fixed offsets.                                                                                                                        |
| `TAI`      | `TAI` is a French acronym for _*temps atomique international*_. It is an atomic time scale meant to track the proper time on the Earth's surface.                                                                                                                                                     | 
| `UTC`      | `UTC` stands for Universal Coordinated Time. `UTC` tracks the solar day, accounting for long term variations due to changes in Earth's rotation to within +/- 1 second. Tracking the solar day in this manner introduces an offset between `TAI` and `UTC` of a fixed number of leap seconds.         |
| `UT1`      | `UT1` stands for Universal Time 1. `UT1` represents the time as determined by the true solar day. Due to Earth's rotation rate constantly changing UT1 itself is constantly changing. The difference between `UT1` and `UTC` is empirically estimated on a daily basis as an Earth orientation parameter. |
| `TT`       | `TT` is Terrestrial Time, a time scale used historically to model the motion of planets and other solar system bodies. These models are still in wide use.                                                                                                                                            | 

## Epoch

The `Epoch` type represents a specific instant in time, defined by both a time representation and a time scale. The `Epoch` type provides methods for converting between different time representations and time scales, as well as for performing arithmetic operations on time instances.

It is the core type used throughout the brahe package to represent time and provides many advandages as 

## See Also

- [Time Systems](time_systems.md) - Detailed explanation of each time system
- [Time Conversions](time_conversions.md) - Conversion functions and algorithms
- [Epoch](epoch.md) - Complete guide to the Epoch type
- [Time API Reference](../../library_api/time/index.md) - Complete time function documentation
- [Time Constants](../../library_api/constants/time.md) - Important time-related constants
