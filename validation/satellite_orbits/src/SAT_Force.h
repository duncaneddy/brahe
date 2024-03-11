//------------------------------------------------------------------------------
//
// SAT_Force.h
// 
// Purpose:
//
//    Force model for Earth orbiting satellites
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


#ifndef INC_SAT_FORCE_H
#define INC_SAT_FORCE_H


#include "SAT_VecMat.h"


// Gravity model record

struct GravModel {
  // Elements
  int     n_max;   // Degree
  int     m_max;   // Order
  double  GM;      // Gravitational coefficient [m^3/s^2]
  double  R_ref;   // Reference radius [m]
  Matrix  CS;      // Unnormalized harmonic coefficients (C_nm, S_nm)

  // Constructor
  GravModel ( int n, double gm, double r_ref, const double* p_cs)
    : n_max(n), m_max(n), GM(gm), R_ref(r_ref)
  {
    CS = Matrix(p_cs,n+1,n+1);
  }

};


// Add external definition of Grav variable allocated in SAT_Force.cpp

#ifndef SAT_FORCE_CPP

extern  GravModel Grav;

#endif


//------------------------------------------------------------------------------
//
// Sun
//
// Purpose:
//
//   Computes the Sun's geocentric position using a low precision 
//   analytical series
//
// Input/Output:
//
//   Mjd_TT    Terrestrial Time (Modified Julian Date)
//   <return>  Solar position vector [m] with respect to the 
//             mean equator and equinox of J2000 (EME2000, ICRF)
//
//------------------------------------------------------------------------------

Vector Sun (double Mjd_TT);


//------------------------------------------------------------------------------
//
// Moon
//
// Purpose:
//
//   Computes the Moon's geocentric position using a low precision
//   analytical series
//
// Input/Output:
//
//   Mjd_TT    Terrestrial Time (Modified Julian Date)
//   <return>  Lunar position vector [m] with respect to the 
//             mean equator and equinox of J2000 (EME2000, ICRF)
//
//------------------------------------------------------------------------------

Vector Moon (double Mjd_TT);



//------------------------------------------------------------------------------
// 
// Illumination
//
// Purpose:
//
//   Computes the fractional illumination of a spacecraft in the 
//   vicinity of the Earth assuming a cylindrical shadow model
// 
// Input/output:
// 
//   r               Spacecraft position vector [m]
//   r_Sun           Sun position vector [m]
//   <return>        Illumination factor:
//                     nu=0   Spacecraft in Earth shadow 
//                     nu=1   Spacecraft fully illuminated by the Sun
//
//------------------------------------------------------------------------------

double Illumination ( const Vector& r, const Vector& r_Sun );


//------------------------------------------------------------------------------
//
// AccelHarmonic
//
// Purpose:
//
//   Computes the acceleration due to the harmonic gravity field of the 
//   central body
//
// Input/Output:
//
//   r           Satellite position vector in the inertial system
//   E           Transformation matrix to body-fixed system
//   GM          Gravitational coefficient
//   R_ref       Reference radius 
//   CS          Spherical harmonics coefficients (un-normalized)
//   n_max       Maximum degree 
//   m_max       Maximum order (m_max<=n_max; m_max=0 for zonals, only)
//   <return>    Acceleration (a=d^2r/dt^2)
//
//------------------------------------------------------------------------------

Vector AccelHarmonic (const Vector& r, const Matrix& E, 
                      double GM, double R_ref, const Matrix& CS,
                      int n_max, int m_max );


//------------------------------------------------------------------------------
//
// AccelPointMass
//
// Purpose:
//
//   Computes the perturbational acceleration due to a point mass
//
// Input/Output:
//
//   r           Satellite position vector (r)
//   s           Point mass position vector (s)
//   GM          Gravitational coefficient of point mass
//   <return>    Acceleration (a=d^2r/dt^2)
//
//------------------------------------------------------------------------------

Vector AccelPointMass (const Vector& r, const Vector& s, double GM);


//------------------------------------------------------------------------------
//
// AccelSolrad
//
// Purpose:
//
//   Computes the acceleration due to solar radiation pressure assuming 
//   the spacecraft surface normal to the Sun direction
//
// Input/Output:
//
//   r           Spacecraft position vector 
//   r_Sun       Sun position vector 
//   Area        Cross-section 
//   mass        Spacecraft mass
//   CR          Solar radiation pressure coefficient
//   P0          Solar radiation pressure at 1 AU 
//   AU          Length of one Astronomical Unit 
//   <return>    Acceleration (a=d^2r/dt^2)
//
// Notes:
//
//   r, r_sun, Area, mass, P0 and AU must be given in consistent units,
//   e.g. m, m^2, kg and N/m^2. 
//
//------------------------------------------------------------------------------

Vector AccelSolrad (const Vector& r, const Vector& r_Sun,
                    double Area, double mass, double CR, 
                    double P0, double AU );


//------------------------------------------------------------------------------
//
// AccelDrag
//
// Purpose:
//
//   Computes the acceleration due to the atmospheric drag.
//
// Input/Output:
//
//   Mjd_TT      Terrestrial Time (Modified Julian Date)
//   r           Satellite position vector in the inertial system [m]
//   v           Satellite velocity vector in the inertial system [m/s]
//   T           Transformation matrix to true-of-date inertial system
//   Area        Cross-section [m^2]
//   mass        Spacecraft mass [kg]
//   CD          Drag coefficient
//   <return>    Acceleration (a=d^2r/dt^2) [m/s^2]
//
//------------------------------------------------------------------------------

Vector AccelDrag ( double Mjd_TT, const Vector& r, const Vector& v,
                   const Matrix& T, 
                   double Area, double mass, double CD );
                       

//------------------------------------------------------------------------------
//
// Density_HP
//
// Purpose:
//
//   Computes the atmospheric density for the modified Harris-Priester model.
//
// Input/Output:
//
//   Mjd_TT      Terrestrial Time (Modified Julian Date)
//   r_tod       Satellite position vector in the inertial system [m]
//   <return>    Density [kg/m^3]
//
//------------------------------------------------------------------------------

double Density_HP ( double Mjd_TT, const Vector& r_tod );


//------------------------------------------------------------------------------
//
// AccelMain
//
// Purpose:
//
//   Computes the acceleration of an Earth orbiting satellite due to 
//    - the Earth's harmonic gravity field, 
//    - the gravitational perturbations of the Sun and Moon
//    - the solar radiation pressure and
//    - the atmospheric drag
//
// Input/Output:
//
//   Mjd_TT      Terrestrial Time (Modified Julian Date)
//   r           Satellite position vector in the ICRF/EME2000 system
//   v           Satellite velocity vector in the ICRF/EME2000 system
//   Area        Cross-section 
//   mass        Spacecraft mass
//   CR          Radiation pressure coefficient
//   CD          Drag coefficient
//   <return>    Acceleration (a=d^2r/dt^2) in the ICRF/EME2000 system
//
//------------------------------------------------------------------------------

Vector AccelMain ( double Mjd_TT, const Vector& r, const Vector& v, 
                   double Area, double mass, double CR, double CD );


#endif  // include blocker

