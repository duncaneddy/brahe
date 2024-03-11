//------------------------------------------------------------------------------
//
// SAT_Const.h
// 
// Purpose:
//
//   Definition of astronomical and mathematical constants (in MKS units)
//
// Notes:
//
//   This software is protected by national and international copyright. 
//   Any unauthorized use, reproduction or modificaton is unlawful and 
//   will be prosecuted. Commercial and non-private application of the 
//   software in any form is strictly prohibited unless otherwise granted
//   by the authors.
//
//   The code is provided without any warranty; without even the implied 
//   warranty of merchantibility or fitness for a particular purpose.
//
// Last modified:
//
//   2000/03/04  OMO  Final version (1st edition)
//   2012/07/01  OMO  Final version (3rd reprint)
//  
// (c) 1999-2012  O. Montenbruck, E. Gill
//
//------------------------------------------------------------------------------

#ifndef INC_SAT_CONST_H
#define INC_SAT_CONST_H


//
// Mathematical constants
//

const double pi        = 3.14159265358979324;
const double pi2       = 2.0*pi;              // 2pi
const double Rad       = pi / 180.0;          // Radians per degree
const double Deg       = 180.0 / pi;          // Degrees per radian
const double Arcs      = 3600.0*180.0/pi;     // Arcseconds per radian


//
// General
//

const double MJD_J2000 = 51544.5;             // Modif. Julian Date of J2000.0

const double AU        = 149597870000.0;      // Astronomical unit [m]; IAU 1976
const double c_light   = 299792458.0;         // Speed of light  [m/s]; IAU 1976


//
// Physical parameters of the Earth, Sun and Moon
//

// Equatorial radius and flattening

const double R_Earth     =   6378.137e3;      // Radius Earth [m]; WGS-84
const double f_Earth     = 1.0/298.257223563; // Flattening; WGS-84   
const double R_Sun       = 696000.0e3;        // Radius Sun [m]; Seidelmann 1992
const double R_Moon      =   1738.0e3;        // Radius Moon [m]

// Earth rotation (derivative of GMST at J2000; differs from inertial period by precession)

const double omega_Earth = 7.2921158553e-5;   // [rad/s]; Aoki 1982, NIMA 1997

// Gravitational coefficient

const double GM_Earth    = 398600.4415e+9;    // [m^3/s^2]; JGM3
const double GM_Sun      = 1.32712438e+20;    // [m^3/s^2]; IAU 1976
const double GM_Moon     = GM_Earth/81.300587;// [m^3/s^2]; DE200


// Solar radiation pressure at 1 AU 

const double P_Sol       = 4.560E-6;          // [N/m^2] (~1367 W/m^2); IERS 96


#endif  // include blocker
