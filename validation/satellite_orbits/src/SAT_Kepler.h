//------------------------------------------------------------------------------
//
// SAT_Kepler.h
// 
// Purpose:
//
//    Keplerian orbit computation
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

#ifndef INC_SAT_KEPLER_H
#define INC_SAT_KEPLER_H


#include "SAT_VecMat.h"



//------------------------------------------------------------------------------
//
// EccAnom
//
// Purpose:
//
//   Computes the eccentric anomaly for elliptic orbits
//
// Input/Output:
//
//   M         Mean anomaly in [rad]
//   e         Eccentricity of the orbit [0,1[
//   <return>  Eccentric anomaly in [rad]
//
//------------------------------------------------------------------------------

double EccAnom (double M, double e);


//------------------------------------------------------------------------------
//
// FindEta 
//
//   Computes the sector-triangle ratio from two position vectors and 
//   the intermediate time 
//
// Input/Output:
//
//   r_a        Position at time t_a
//   r_a        Position at time t_b
//   tau        Normalized time (sqrt(GM)*(t_a-t_b))
//   <return>   Sector-triangle ratio
//
//------------------------------------------------------------------------------

double FindEta (const Vector& r_a, const Vector& r_b, double tau);


//------------------------------------------------------------------------------
//
// State
//
// Purpose:
//
//   Computes the satellite state vector from osculating Keplerian elements 
//   for elliptic orbits
//
// Input/Output:
//
//   GM        Gravitational coefficient
//             (gravitational constant * mass of central body)
//   Kep       Keplerian elements (a,e,i,Omega,omega,M)
//               a      Semimajor axis 
//               e      Eccentricity 
//               i      Inclination [rad]
//               Omega  Longitude of the ascending node [rad]
//               omega  Argument of pericenter  [rad]
//               M      Mean anomaly at epoch [rad]
//   dt        Time since epoch
//   <return>  State vector (x,y,z,vx,vy,vz)
//
// Notes:
//
//   The semimajor axis a=Kep(0), dt and GM must be given in consistent units, 
//   e.g. [m], [s] and [m^3/s^2]. The resulting units of length and velocity  
//   are implied by the units of GM, e.g. [m] and [m/s].
//
//------------------------------------------------------------------------------

Vector State ( double GM, const Vector& Kep, double dt=0.0 );


//------------------------------------------------------------------------------
//
// StatePartials
//
// Purpose:
//
//   Computes the partial derivatives of the satellite state vector with respect
//   to the orbital elements for elliptic, Keplerian orbits
//
// Input/Output:
//
//   GM        Gravitational coefficient
//             (gravitational constant * mass of central body)
//   Kep       Keplerian elements (a,e,i,Omega,omega,M) with
//               a      Semimajor axis 
//               e      Eccentricity 
//               i      Inclination [rad]
//               Omega  Longitude of the ascending node [rad]
//               omega  Argument of pericenter  [rad]
//               M      Mean anomaly at epoch [rad]
//   dt        Time since epoch
//   <return>  Partials derivatives of the state vector (x,y,z,vx,vy,vz) at time
//             dt with respect to the epoch orbital elements
//
// Notes:
//
//   The semimajor axis a=Kep(0), dt and GM must be given in consistent units, 
//   e.g. [m], [s] and [m^3/s^2]. The resulting units of length and velocity  
//   are implied by the units of GM, e.g. [m] and [m/s].
//
//   The function cannot be used with circular or non-inclined orbit.
//
//------------------------------------------------------------------------------

Matrix StatePartials ( double GM, const Vector& Kep, double dt=0.0 );


//------------------------------------------------------------------------------
//
// Elements
//
// Purpose:
//
//   Computes the osculating Keplerian elements from the satellite state vector
//   for elliptic orbits
//
// Input/Output:
//
//   GM        Gravitational coefficient
//             (gravitational constant * mass of central body)
//   y         State vector (x,y,z,vx,vy,vz)
//   <return>  Keplerian elements (a,e,i,Omega,omega,M) with
//               a      Semimajor axis 
//               e      Eccentricity 
//               i      Inclination [rad]
//               Omega  Longitude of the ascending node [rad]
//               omega  Argument of pericenter  [rad]
//               M      Mean anomaly  [rad]
//
// Notes:
//
//   The state vector and GM must be given in consistent units, 
//   e.g. [m], [m/s] and [m^3/s^2]. The resulting unit of the semimajor
//   axis is implied by the unity of y, e.g. [m].
//
//   The function cannot be used with state vectors describing a circular
//   or non-inclined orbit.
//
//------------------------------------------------------------------------------

Vector Elements ( double GM, const Vector& y );


//------------------------------------------------------------------------------
//
// Elements 
//
// Purpose:
//
//   Computes orbital elements from two given position vectors and 
//   associated times 
//
// Input/Output:
//
//   GM        Gravitational coefficient
//             (gravitational constant * mass of central body)
//   Mjd_a     Time t_a (Modified Julian Date)
//   Mjd_b     Time t_b (Modified Julian Date)
//   r_a       Position vector at time t_a
//   r_b       Position vector at time t_b
//   <return>  Keplerian elements (a,e,i,Omega,omega,M)
//               a      Semimajor axis 
//               e      Eccentricity 
//               i      Inclination [rad]
//               Omega  Longitude of the ascending node [rad]
//               omega  Argument of pericenter  [rad]
//               M      Mean anomaly  [rad]
//             at time t_a 
//
// Notes:
//
//   The function cannot be used with state vectors describing a circular
//   or non-inclined orbit.
//
//------------------------------------------------------------------------------

Vector Elements ( double GM, double Mjd_a, double Mjd_b, 
                  const Vector& r_a, const Vector& r_b );


//------------------------------------------------------------------------------
//
// TwoBody
//
// Purpose:
//
//   Propagates a given state vector and computes the state transition matrix 
//   for elliptical Keplerian orbits
//
// Input/Output:
//
//   GM        Gravitational coefficient
//             (gravitational constant * mass of central body)
//   Y0        Epoch state vector (x,y,z,vx,vy,vz)_0
//   dt        Time since epoch
//   Y         State vector (x,y,z,vx,vy,vz)
//   dYdY0     State transition matrix d(x,y,z,vx,vy,vz)/d(x,y,z,vx,vy,vz)_0
//
// Notes:
//
//   The state vector, dt and GM must be given in consistent units, 
//   e.g. [m], [m/s] and [m^3/s^2]. The resulting units of length and velocity  
//   are implied by the units of GM, e.g. [m] and [m/s].
//
//   Due to the internal use of Keplerian elements, the function cannot be 
//   used with epoch state vectors describing a circular or non-inclined orbit.
//
//------------------------------------------------------------------------------

void TwoBody ( double GM, const Vector& Y0, double dt, 
               Vector& Y, Matrix& dYdY0 );

#endif  // include blocker

