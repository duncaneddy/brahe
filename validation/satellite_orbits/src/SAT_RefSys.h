//------------------------------------------------------------------------------
//
// SAT_RefSys.h
// 
// Purpose: 
//
//   Transformations betweeen celestial and terrestrial reference systems
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

#ifndef INC_SAT_REFSYS_H
#define INC_SAT_REFSYS_H


#include "SAT_Const.h"
#include "SAT_VecMat.h"


//------------------------------------------------------------------------------
//
// IERS (Class definition)
//
// Purpose:
//
//   Management of IERS time and polar motion data
//  
//------------------------------------------------------------------------------

class IERS {

  public:

    static void Set (double UT1_UTC,          // Set Earth rotation parameters
                     double UTC_TAI,          // (UT1-UTC [s],UTC-TAI [s],
                     double x_pole,           //  x ["], y ["])
                     double y_pole);

    static const double  TT_TAI;              //  TT-TAI time difference 32.184s
    static const double GPS_TAI;              // GPS-TAI time difference -19s
    static double UTC_TAI(double Mjd_UTC);    // UTC_TAI time difference [s]
    static double UT1_TAI(double Mjd_UTC);    // UT1-UTC time difference [s]

    static const double  TT_GPS;              //  TT-GPS time difference 51.184s
    static const double TAI_GPS;              // TAI-GPS time difference 19s
    static double UTC_GPS(double Mjd_UTC);    // UTC_GPS time difference [s]
    static double UT1_GPS(double Mjd_UTC);    // UT1-GPS time difference [s]

    static double  TT_UTC(double Mjd_UTC);    //  TT-UTC time difference [s]
    static double TAI_UTC(double Mjd_UTC);    // TAI-UTC time difference [s]
    static double GPS_UTC(double Mjd_UTC);    // GPS-UTC time difference [s]
    static double UT1_UTC(double Mjd_UTC);    // UT1-UTC time difference [s]

    static double x_pole(double Mjd_UTC);     // Pole coordinate [rad]
    static double y_pole(double Mjd_UTC);     // Pole coordinate [rad]

  private:
    static double UT1_TAI_;                   // UT1-TAI time difference [s]
    static double UTC_TAI_;                   // UTC-TAI time difference [s]
    static double x_pole_;                    // Pole coordinate [rad]
    static double y_pole_;                    // Pole coordinate [rad]
};



//------------------------------------------------------------------------------
//
// MeanObliquity
//
// Purpose:
//
//   Computes the mean obliquity of the ecliptic
//
// Input/Output:
//
//   Mjd_TT    Modified Julian Date (Terrestrial Time)
//   <return>  Mean obliquity of the ecliptic
//
//------------------------------------------------------------------------------

double MeanObliquity (double Mjd_TT);


//------------------------------------------------------------------------------
//
// EclMatrix
//
// Purpose:
//
//   Transformation of equatorial to ecliptical coordinates
//
// Input/Output:
//
//   Mjd_TT    Modified Julian Date (Terrestrial Time)
//   <return>  Transformation matrix
//
//------------------------------------------------------------------------------

Matrix EclMatrix (double Mjd_TT);


//------------------------------------------------------------------------------
//
// PrecMatrix
//
// Purpose:
//
//   Precession transformation of equatorial coordinates
//
// Input/Output:
//
//   Mjd_1     Epoch given (Modified Julian Date TT)
//   MjD_2     Epoch to precess to (Modified Julian Date TT)
//   <return>  Precession transformation matrix
//
//------------------------------------------------------------------------------

Matrix PrecMatrix (double Mjd_1, double Mjd_2);


//------------------------------------------------------------------------------
//
// NutMatrix 
//
// Purpose:
//
//   Transformation from mean to true equator and equinox
//
// Input/Output:
//
//   Mjd_TT    Modified Julian Date (Terrestrial Time)
//   <return>  Nutation matrix
//
//------------------------------------------------------------------------------

Matrix NutMatrix (double Mjd_TT);


//------------------------------------------------------------------------------
//
// EqnEquinox 
//
// Purpose:
//
//   Computation of the equation of the equinoxes
//
// Input/Output:
//
//   Mjd_TT    Modified Julian Date (Terrestrial Time)
//   <return>  Equation of the equinoxes
//
// Notes:
//
//   The equation of the equinoxes dpsi*cos(eps) is the right ascension of the 
//   mean equinox referred to the true equator and equinox and is equal to the 
//   difference between apparent and mean sidereal time.
//
//------------------------------------------------------------------------------

double EqnEquinox (double Mjd_TT);


//------------------------------------------------------------------------------
//
// GMST
//
// Purpose:
//
//   Greenwich Mean Sidereal Time
//
// Input/Output:
//
//   Mjd_UT1   Modified Julian Date UT1
//   <return>  GMST in [rad]
//
//------------------------------------------------------------------------------

double GMST (double Mjd_UT1);


//------------------------------------------------------------------------------
//
// GAST
//
// Purpose:
//
//   Greenwich Apparent Sidereal Time
//
// Input/Output:
//
//   Mjd_UT1   Modified Julian Date UT1
//   <return>  GMST in [rad]
//
//------------------------------------------------------------------------------

double GAST (double Mjd_UT1);


//------------------------------------------------------------------------------
//
// GHAMatrix
//
// Purpose:
//
//   Transformation from true equator and equinox to Earth equator and 
//   Greenwich meridian system 
//
// Input/Output:
//
//   Mjd_UT1   Modified Julian Date UT1
//   <return>  Greenwich Hour Angle matrix
//
//------------------------------------------------------------------------------

Matrix GHAMatrix (double Mjd_UT1);


//------------------------------------------------------------------------------
//
// PoleMatrix
//
// Purpose:
//
//   Transformation from pseudo Earth-fixed to Earth-fixed coordinates
//   for a given date
//
// Input/Output:
//
//   Mjd_UTC   Modified Julian Date UTC
//   <return>  Pole matrix
//
//------------------------------------------------------------------------------

Matrix PoleMatrix (double Mjd_UTC);


//------------------------------------------------------------------------------
//
// Geodetic (class definition)
//
// Purpose:
//
//   Class (with all public elements) for handling geodetic coordinates
//
//------------------------------------------------------------------------------

class Geodetic 
{
  public:

    // Elements
    double lon;                                        // Longitude [rad]
    double lat;                                        // Latitude [rad]
    double h;                                          // Altitude [m]

    // Constructors
    Geodetic ();                                       // Default constructor
    Geodetic (double lambda, double phi, double alt);   
    Geodetic (Vector r,                                // Position vector [m]
              double R_equ=R_Earth,                    // Equator radius [m]
              double f    =f_Earth);                   // Flattening
  
    // Position vector [m] from geodetic coordinates
    Vector Position (double R_equ=R_Earth,             // Equator radius [m]
                     double f    =f_Earth) const;      // Flattening

    // Transformation to local tangent coordinates
    Matrix LTC_Matrix () const;
};


//------------------------------------------------------------------------------
//
// LTCMatrix
//
// Purpose:
//
//   Transformation from Greenwich meridian system to local tangent coordinates
//
// Input/Output:
//
//   lambda    Geodetic East longitude [rad]
//   phi       Geodetic latitude [rad]
//   <return>  Rotation matrix from the Earth equator and Greenwich meridian
//             to the local tangent (East-North-Zenith) coordinate system
//
//------------------------------------------------------------------------------

Matrix LTCMatrix (double lambda, double phi);


//------------------------------------------------------------------------------
//
// AzEl
//
// Purpose:
//
//   Computes azimuth and elevation from local tangent coordinates
//
// Input/Output:
//
//   s   Topocentric local tangent coordinates (East-North-Zenith frame)
//   A   Azimuth [rad]
//   E   Elevation [rad]
//
//------------------------------------------------------------------------------

void AzEl ( const Vector& s, double& A, double& E ); 


//------------------------------------------------------------------------------
//
// AzEl
//
// Purpose:
//
//   Computes azimuth, elevation and partials from local tangent coordinates
//
// Input/Output:
//
//   s      Topocentric local tangent coordinates (East-North-Zenith frame)
//   A      Azimuth [rad]
//   E      Elevation [rad]
//   dAds   Partials of azimuth w.r.t. s
//   dEds   Partials of elevation w.r.t. s
//
//------------------------------------------------------------------------------

void AzEl( const Vector& s, double& A, double& E, Vector& dAds, Vector& dEds );


#endif  // include-Blocker
