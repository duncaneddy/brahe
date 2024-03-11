//------------------------------------------------------------------------------
//
// SAT_RefSys.cpp
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

#include <cmath>
#include <iostream>

#include <limits>

#include "SAT_Const.h"
#include "SAT_RefSys.h"
#include "SAT_VecMat.h"

using std::ostream;
using std::cerr;
using std::endl;



//
// Local declarations
//

namespace { 
  // Machine accuracy
  const double eps_mach = std::numeric_limits<double>::epsilon();
  // Fractional part of a number (y=x-[x])
  double Frac (double x) { return x-floor(x); };
  // x mod y
  double Modulo (double x, double y) { return y*Frac(x/y); }
}            


//------------------------------------------------------------------------------
//
// IERS (Class implementation)
//
// Purpose:
//
//   Management of IERS time and polar motion data
//  
//------------------------------------------------------------------------------

// Class constants

const double IERS::TT_TAI  = +32.184;          //  TT-TAI time difference [s]
const double IERS::GPS_TAI = -19.0;            // GPS-TAI time difference [s]

const double IERS::TT_GPS  =  IERS::TT_TAI     //  TT-GPS time difference [s]
                             -IERS::GPS_TAI; 
const double IERS::TAI_GPS = -IERS::GPS_TAI;   // TAI-GPS time difference [s]

// Default values of Earth rotation parameters

double IERS::UT1_TAI_ = 0.0;          // UT1-TAI time difference [s]
double IERS::UTC_TAI_ = 0.0;          // UTC-TAI time difference [s] 
double IERS::x_pole_  = 0.0;          // Pole coordinate [rad]
double IERS::y_pole_  = 0.0;          // Pole coordinate [rad]

// Setting of IERS Earth rotation parameters
// (UT1-UTC [s], UTC-TAI [s], x ["], y ["])

void IERS::Set(double UT1_UTC, double UTC_TAI,
               double x_pole, double y_pole) 
{
   UT1_TAI_ = UT1_UTC+UTC_TAI;
   UTC_TAI_ = UTC_TAI;
   x_pole_  = x_pole/Arcs;
   y_pole_  = y_pole/Arcs;
};

// Time differences [s]

double IERS::UTC_TAI(double Mjd_UTC){return UTC_TAI_;};
double IERS::UT1_TAI(double Mjd_UTC){return UT1_TAI_;};

double IERS::UTC_GPS(double Mjd_UTC){return UTC_TAI(Mjd_UTC)-GPS_TAI;};
double IERS::UT1_GPS(double Mjd_UTC){return UT1_TAI(Mjd_UTC)-GPS_TAI;};

double IERS::TT_UTC (double Mjd_UTC){return  TT_TAI-UTC_TAI(Mjd_UTC);};
double IERS::GPS_UTC(double Mjd_UTC){return GPS_TAI-UTC_TAI(Mjd_UTC);};
double IERS::UT1_UTC(double Mjd_UTC){return UT1_TAI(Mjd_UTC)-UTC_TAI(Mjd_UTC);};




// Pole coordinate [rad]

double IERS::x_pole(double Mjd_UTC) { return x_pole_; };

// Pole coordinate [rad]

double IERS::y_pole(double Mjd_UTC) { return y_pole_; };



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

double MeanObliquity (double Mjd_TT)
{
  const double T = (Mjd_TT-MJD_J2000)/36525.0;

  return 
    Rad *( 23.43929111-(46.8150+(0.00059-0.001813*T)*T)*T/3600.0 );
}


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

Matrix EclMatrix (double Mjd_TT)
{
  return R_x ( MeanObliquity(Mjd_TT) );
}


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

Matrix PrecMatrix (double Mjd_1, double Mjd_2)
{

  // Constants

  const double T  = (Mjd_1-MJD_J2000)/36525.0;
  const double dT = (Mjd_2-Mjd_1)/36525.0;
  
  // Variables

  double zeta,z,theta;

  // Precession angles
  
  zeta  =  ( (2306.2181+(1.39656-0.000139*T)*T)+
                ((0.30188-0.000344*T)+0.017998*dT)*dT )*dT/Arcs;
  z     =  zeta + ( (0.79280+0.000411*T)+0.000205*dT)*dT*dT/Arcs;
  theta =  ( (2004.3109-(0.85330+0.000217*T)*T)-
                ((0.42665+0.000217*T)+0.041833*dT)*dT )*dT/Arcs;

  // Precession matrix
  
  return R_z(-z) * R_y(theta) * R_z(-zeta);

}    


//------------------------------------------------------------------------------
//
// NutAngles 
//
// Purpose:
//
//   Nutation in longitude and obliquity
//
// Input/Output:
//
//   Mjd_TT    Modified Julian Date (Terrestrial Time)
//   <return>  Nutation matrix
//
//------------------------------------------------------------------------------

void NutAngles (double Mjd_TT, double& dpsi, double& deps)
{

  // Constants

  const double T  = (Mjd_TT-MJD_J2000)/36525.0;
  const double T2 = T*T;
  const double T3 = T2*T;
  const double rev = 360.0*3600.0;  // arcsec/revolution

  const int  N_coeff = 106;
  const long C[N_coeff][9] =
  {
   //
   // l  l' F  D Om    dpsi    *T     deps     *T       #
   //
    {  0, 0, 0, 0, 1,-1719960,-1742,  920250,   89 },   //   1
    {  0, 0, 0, 0, 2,   20620,    2,   -8950,    5 },   //   2
    { -2, 0, 2, 0, 1,     460,    0,    -240,    0 },   //   3
    {  2, 0,-2, 0, 0,     110,    0,       0,    0 },   //   4
    { -2, 0, 2, 0, 2,     -30,    0,      10,    0 },   //   5
    {  1,-1, 0,-1, 0,     -30,    0,       0,    0 },   //   6
    {  0,-2, 2,-2, 1,     -20,    0,      10,    0 },   //   7
    {  2, 0,-2, 0, 1,      10,    0,       0,    0 },   //   8
    {  0, 0, 2,-2, 2, -131870,  -16,   57360,  -31 },   //   9
    {  0, 1, 0, 0, 0,   14260,  -34,     540,   -1 },   //  10
    {  0, 1, 2,-2, 2,   -5170,   12,    2240,   -6 },   //  11
    {  0,-1, 2,-2, 2,    2170,   -5,    -950,    3 },   //  12
    {  0, 0, 2,-2, 1,    1290,    1,    -700,    0 },   //  13
    {  2, 0, 0,-2, 0,     480,    0,      10,    0 },   //  14
    {  0, 0, 2,-2, 0,    -220,    0,       0,    0 },   //  15
    {  0, 2, 0, 0, 0,     170,   -1,       0,    0 },   //  16
    {  0, 1, 0, 0, 1,    -150,    0,      90,    0 },   //  17
    {  0, 2, 2,-2, 2,    -160,    1,      70,    0 },   //  18
    {  0,-1, 0, 0, 1,    -120,    0,      60,    0 },   //  19
    { -2, 0, 0, 2, 1,     -60,    0,      30,    0 },   //  20
    {  0,-1, 2,-2, 1,     -50,    0,      30,    0 },   //  21
    {  2, 0, 0,-2, 1,      40,    0,     -20,    0 },   //  22
    {  0, 1, 2,-2, 1,      40,    0,     -20,    0 },   //  23
    {  1, 0, 0,-1, 0,     -40,    0,       0,    0 },   //  24
    {  2, 1, 0,-2, 0,      10,    0,       0,    0 },   //  25
    {  0, 0,-2, 2, 1,      10,    0,       0,    0 },   //  26
    {  0, 1,-2, 2, 0,     -10,    0,       0,    0 },   //  27
    {  0, 1, 0, 0, 2,      10,    0,       0,    0 },   //  28
    { -1, 0, 0, 1, 1,      10,    0,       0,    0 },   //  29
    {  0, 1, 2,-2, 0,     -10,    0,       0,    0 },   //  30
    {  0, 0, 2, 0, 2,  -22740,   -2,    9770,   -5 },   //  31
    {  1, 0, 0, 0, 0,    7120,    1,     -70,    0 },   //  32
    {  0, 0, 2, 0, 1,   -3860,   -4,    2000,    0 },   //  33
    {  1, 0, 2, 0, 2,   -3010,    0,    1290,   -1 },   //  34
    {  1, 0, 0,-2, 0,   -1580,    0,     -10,    0 },   //  35
    { -1, 0, 2, 0, 2,    1230,    0,    -530,    0 },   //  36
    {  0, 0, 0, 2, 0,     630,    0,     -20,    0 },   //  37
    {  1, 0, 0, 0, 1,     630,    1,    -330,    0 },   //  38
    { -1, 0, 0, 0, 1,    -580,   -1,     320,    0 },   //  39
    { -1, 0, 2, 2, 2,    -590,    0,     260,    0 },   //  40
    {  1, 0, 2, 0, 1,    -510,    0,     270,    0 },   //  41
    {  0, 0, 2, 2, 2,    -380,    0,     160,    0 },   //  42
    {  2, 0, 0, 0, 0,     290,    0,     -10,    0 },   //  43
    {  1, 0, 2,-2, 2,     290,    0,    -120,    0 },   //  44
    {  2, 0, 2, 0, 2,    -310,    0,     130,    0 },   //  45
    {  0, 0, 2, 0, 0,     260,    0,     -10,    0 },   //  46
    { -1, 0, 2, 0, 1,     210,    0,    -100,    0 },   //  47
    { -1, 0, 0, 2, 1,     160,    0,     -80,    0 },   //  48
    {  1, 0, 0,-2, 1,    -130,    0,      70,    0 },   //  49
    { -1, 0, 2, 2, 1,    -100,    0,      50,    0 },   //  50
    {  1, 1, 0,-2, 0,     -70,    0,       0,    0 },   //  51
    {  0, 1, 2, 0, 2,      70,    0,     -30,    0 },   //  52
    {  0,-1, 2, 0, 2,     -70,    0,      30,    0 },   //  53
    {  1, 0, 2, 2, 2,     -80,    0,      30,    0 },   //  54
    {  1, 0, 0, 2, 0,      60,    0,       0,    0 },   //  55
    {  2, 0, 2,-2, 2,      60,    0,     -30,    0 },   //  56
    {  0, 0, 0, 2, 1,     -60,    0,      30,    0 },   //  57
    {  0, 0, 2, 2, 1,     -70,    0,      30,    0 },   //  58
    {  1, 0, 2,-2, 1,      60,    0,     -30,    0 },   //  59
    {  0, 0, 0,-2, 1,     -50,    0,      30,    0 },   //  60
    {  1,-1, 0, 0, 0,      50,    0,       0,    0 },   //  61
    {  2, 0, 2, 0, 1,     -50,    0,      30,    0 },   //  62
    {  0, 1, 0,-2, 0,     -40,    0,       0,    0 },   //  63
    {  1, 0,-2, 0, 0,      40,    0,       0,    0 },   //  64
    {  0, 0, 0, 1, 0,     -40,    0,       0,    0 },   //  65
    {  1, 1, 0, 0, 0,     -30,    0,       0,    0 },   //  66
    {  1, 0, 2, 0, 0,      30,    0,       0,    0 },   //  67
    {  1,-1, 2, 0, 2,     -30,    0,      10,    0 },   //  68
    { -1,-1, 2, 2, 2,     -30,    0,      10,    0 },   //  69
    { -2, 0, 0, 0, 1,     -20,    0,      10,    0 },   //  70
    {  3, 0, 2, 0, 2,     -30,    0,      10,    0 },   //  71
    {  0,-1, 2, 2, 2,     -30,    0,      10,    0 },   //  72
    {  1, 1, 2, 0, 2,      20,    0,     -10,    0 },   //  73
    { -1, 0, 2,-2, 1,     -20,    0,      10,    0 },   //  74
    {  2, 0, 0, 0, 1,      20,    0,     -10,    0 },   //  75
    {  1, 0, 0, 0, 2,     -20,    0,      10,    0 },   //  76
    {  3, 0, 0, 0, 0,      20,    0,       0,    0 },   //  77
    {  0, 0, 2, 1, 2,      20,    0,     -10,    0 },   //  78
    { -1, 0, 0, 0, 2,      10,    0,     -10,    0 },   //  79
    {  1, 0, 0,-4, 0,     -10,    0,       0,    0 },   //  80
    { -2, 0, 2, 2, 2,      10,    0,     -10,    0 },   //  81
    { -1, 0, 2, 4, 2,     -20,    0,      10,    0 },   //  82
    {  2, 0, 0,-4, 0,     -10,    0,       0,    0 },   //  83
    {  1, 1, 2,-2, 2,      10,    0,     -10,    0 },   //  84
    {  1, 0, 2, 2, 1,     -10,    0,      10,    0 },   //  85
    { -2, 0, 2, 4, 2,     -10,    0,      10,    0 },   //  86
    { -1, 0, 4, 0, 2,      10,    0,       0,    0 },   //  87
    {  1,-1, 0,-2, 0,      10,    0,       0,    0 },   //  88
    {  2, 0, 2,-2, 1,      10,    0,     -10,    0 },   //  89
    {  2, 0, 2, 2, 2,     -10,    0,       0,    0 },   //  90
    {  1, 0, 0, 2, 1,     -10,    0,       0,    0 },   //  91
    {  0, 0, 4,-2, 2,      10,    0,       0,    0 },   //  92
    {  3, 0, 2,-2, 2,      10,    0,       0,    0 },   //  93
    {  1, 0, 2,-2, 0,     -10,    0,       0,    0 },   //  94
    {  0, 1, 2, 0, 1,      10,    0,       0,    0 },   //  95
    { -1,-1, 0, 2, 1,      10,    0,       0,    0 },   //  96
    {  0, 0,-2, 0, 1,     -10,    0,       0,    0 },   //  97
    {  0, 0, 2,-1, 2,     -10,    0,       0,    0 },   //  98
    {  0, 1, 0, 2, 0,     -10,    0,       0,    0 },   //  99
    {  1, 0,-2,-2, 0,     -10,    0,       0,    0 },   // 100
    {  0,-1, 2, 0, 1,     -10,    0,       0,    0 },   // 101
    {  1, 1, 0,-2, 1,     -10,    0,       0,    0 },   // 102
    {  1, 0,-2, 2, 0,     -10,    0,       0,    0 },   // 103
    {  2, 0, 0, 2, 0,      10,    0,       0,    0 },   // 104
    {  0, 0, 2, 4, 2,     -10,    0,       0,    0 },   // 105
    {  0, 1, 0, 1, 0,      10,    0,       0,    0 }    // 106
   };

  // Variables

  double  l, lp, F, D, Om;
  double  arg;
  

  // Mean arguments of luni-solar motion
  //
  //   l   mean anomaly of the Moon
  //   l'  mean anomaly of the Sun
  //   F   mean argument of latitude
  //   D   mean longitude elongation of the Moon from the Sun 
  //   Om  mean longitude of the ascending node
  
  l  = Modulo (  485866.733 + (1325.0*rev +  715922.633)*T    
                                 + 31.310*T2 + 0.064*T3, rev );
  lp = Modulo ( 1287099.804 + (  99.0*rev + 1292581.224)*T             
                                 -  0.577*T2 - 0.012*T3, rev );
  F  = Modulo (  335778.877 + (1342.0*rev +  295263.137)*T    
                                 - 13.257*T2 + 0.011*T3, rev );
  D  = Modulo ( 1072261.307 + (1236.0*rev + 1105601.328)*T    
                                 -  6.891*T2 + 0.019*T3, rev );
  Om = Modulo (  450160.280 - (   5.0*rev +  482890.539)*T    
                                 +  7.455*T2 + 0.008*T3, rev );

  // Nutation in longitude and obliquity [rad]

  deps = dpsi = 0.0;
  for (int i=0; i<N_coeff; i++) {
    arg  =  ( C[i][0]*l+C[i][1]*lp+C[i][2]*F+C[i][3]*D+C[i][4]*Om ) / Arcs;
    dpsi += ( C[i][5]+C[i][6]*T ) * sin(arg);
    deps += ( C[i][7]+C[i][8]*T ) * cos(arg);
  };
      
  dpsi = 1.0E-5 * dpsi/Arcs;
  deps = 1.0E-5 * deps/Arcs;

}


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

Matrix NutMatrix (double Mjd_TT)
{

  double dpsi, deps, eps;
  
  // Mean obliquity of the ecliptic

  eps = MeanObliquity(Mjd_TT);

  // Nutation in longitude and obliquity

  NutAngles (Mjd_TT, dpsi,deps);

  // Transformation from mean to true equator and equinox

  return  R_x(-eps-deps)*R_z(-dpsi)*R_x(+eps);

}


//------------------------------------------------------------------------------
//
// NutMatrixSimple
//
// Purpose:
//
//   Transformation from mean to true equator and equinox (low precision)
//
// Input/Output:
//
//   Mjd_TT    Modified Julian Date (Terrestrial Time)
//   <return>  Nutation matrix
//
//------------------------------------------------------------------------------

Matrix NutMatrixSimple (double Mjd_TT)
{

  // Constants

  const double T  = (Mjd_TT-MJD_J2000)/36525.0;

  // Variables

  double  ls, D, F, N;
  double  eps, dpsi, deps;

  // Mean arguments of luni-solar motion
  
  ls = pi2*Frac(0.993133+  99.997306*T);   // mean anomaly Sun          
  D  = pi2*Frac(0.827362+1236.853087*T);   // diff. longitude Moon-Sun  
  F  = pi2*Frac(0.259089+1342.227826*T);   // mean argument of latitude 
  N  = pi2*Frac(0.347346-   5.372447*T);   // longit. ascending node    

  // Nutation angles
  
  dpsi = ( -17.200*sin(N)   - 1.319*sin(2*(F-D+N)) - 0.227*sin(2*(F+N))
           + 0.206*sin(2*N) + 0.143*sin(ls) ) / Arcs;
  deps = ( + 9.203*cos(N)   + 0.574*cos(2*(F-D+N)) + 0.098*cos(2*(F+N))
           - 0.090*cos(2*N)                 ) / Arcs;

  // Mean obliquity of the ecliptic

  eps  = 0.4090928-2.2696E-4*T;    

  return  R_x(-eps-deps)*R_z(-dpsi)*R_x(+eps);

}


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

double EqnEquinox (double Mjd_TT)
{
  double dpsi, deps;              // Nutation angles

  // Nutation in longitude and obliquity 
      
  NutAngles (Mjd_TT, dpsi,deps );

  // Equation of the equinoxes

  return  dpsi * cos ( MeanObliquity(Mjd_TT) );

};


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

double GMST (double Mjd_UT1)
{

  // Constants

  const double Secs = 86400.0;        // Seconds per day

  // Variables

  double Mjd_0,UT1,T_0,T,gmst;

  // Mean Sidereal Time
  
  Mjd_0 = floor(Mjd_UT1);
  UT1   = Secs*(Mjd_UT1-Mjd_0);          // [s]
  T_0   = (Mjd_0  -MJD_J2000)/36525.0; 
  T     = (Mjd_UT1-MJD_J2000)/36525.0; 

  gmst  = 24110.54841 + 8640184.812866*T_0 + 1.002737909350795*UT1
          + (0.093104-6.2e-6*T)*T*T; // [s]

  return  pi2*Frac(gmst/Secs);       // [rad], 0..2pi

}


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

double GAST (double Mjd_UT1)
{
  return Modulo ( GMST(Mjd_UT1) + EqnEquinox(Mjd_UT1), pi2 );
}


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

Matrix GHAMatrix (double Mjd_UT1)
{
  return  R_z ( GAST(Mjd_UT1) );
}


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

Matrix PoleMatrix (double Mjd_UTC)
{
   return  R_y(-IERS::x_pole(Mjd_UTC)) * R_x(-IERS::y_pole(Mjd_UTC));
}


//------------------------------------------------------------------------------
//
// Geodetic (class implementation)
//
// Purpose:
//
//   Class (with all public elements) for handling geodetic coordinates
//
//------------------------------------------------------------------------------

// Default constructor

Geodetic::Geodetic ()
 : lon(0.0), lat(0.0), h(0.0)
{
}

// Simple constructor

Geodetic::Geodetic (double lambda, double phi, double alt)
{
  lon=lambda; lat=phi; h=alt;
}

// Constructor for geodetic coordinates from given position

Geodetic::Geodetic (Vector r,                         // Position vector [m]
                    double R_equ,                     // Equator radius [m]
                    double f)                         // Flattening
{

  const double  eps     = 1.0e3*eps_mach;   // Convergence criterion 
  const double  epsRequ = eps*R_equ;
  const double  e2      = f*(2.0-f);        // Square of eccentricity
  
  const double  X = r(0);                   // Cartesian coordinates
  const double  Y = r(1);
  const double  Z = r(2);
  const double  rho2 = X*X + Y*Y;           // Square of distance from z-axis
  
  // Check validity of input data
  
  if (Norm(r)==0.0) {
    cerr << " invalid input in Geodetic constructor" << endl;
    lon=0.0; lat=0.0; h=-R_Earth;
    return;
  }

  // Iteration 

  double  dZ, dZ_new, SinPhi;
  double  ZdZ, Nh, N;

  dZ = e2*Z;
  for(;;) {
    ZdZ    =  Z + dZ;
    Nh     =  sqrt ( rho2 + ZdZ*ZdZ ); 
    SinPhi =  ZdZ / Nh;                    // Sine of geodetic latitude
    N      =  R_equ / sqrt(1.0-e2*SinPhi*SinPhi); 
    dZ_new =  N*e2*SinPhi;
    if ( fabs(dZ-dZ_new) < epsRequ ) break;
    dZ = dZ_new;
  }
    
  // Longitude, latitude, altitude

  lon = atan2 ( Y, X );
  lat = atan2 ( ZdZ, sqrt(rho2) );
  h   = Nh - N;

}

// Position vector [m] from geodetic coordinates

Vector Geodetic::Position (double R_equ,   // Equator radius [m]
                           double f    )   // Flattening
  const
{  

  const double  e2     = f*(2.0-f);        // Square of eccentricity
  const double  CosLat = cos(lat);         // (Co)sine of geodetic latitude
  const double  SinLat = sin(lat);

  double  N;
  Vector  r(3);
      
  // Position vector 

  N = R_equ / sqrt(1.0-e2*SinLat*SinLat);

  r(0) =  (         N+h)*CosLat*cos(lon);
  r(1) =  (         N+h)*CosLat*sin(lon);
  r(2) =  ((1.0-e2)*N+h)*SinLat;

  return r;

}

// Transformation to local tangent coordinates

Matrix Geodetic::LTC_Matrix () const
{
  return LTCMatrix(lon,lat);
}


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

Matrix LTCMatrix (double lambda, double phi)
{
  
  Matrix  M(3,3);
  double  Aux;
  
  // Transformation to Zenith-East-North System
  M = R_y(-phi)*R_z(lambda);
  
  // Cyclic shift of rows 0,1,2 to 1,2,0 to obtain East-North-Zenith system
  for (int j=0; j<3; j++) {
    Aux=M(0,j); M(0,j)=M(1,j); M(1,j)=M(2,j); M(2,j)= Aux;
  }
  
  return  M;
  
}


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

void AzEl ( const Vector& s, double& A, double& E ) 
{
  A = atan2(s(0),s(1));
  A = ((A<0.0)? A+pi2 : A);
  E = atan ( s(2) / sqrt(s(0)*s(0)+s(1)*s(1)) );
}


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

void AzEl( const Vector& s, double& A, double& E, Vector& dAds, Vector& dEds ) 
{
  const double rho = sqrt(s(0)*s(0)+s(1)*s(1));
  // Angles
  A = atan2(s(0),s(1));
  A = ((A<0.0)? A+pi2 : A);
  E = atan ( s(2) / rho );
  // Partials
  dAds = Vector( s(1)/(rho*rho), -s(0)/(rho*rho), 0.0 );
  dEds = Vector( -s(0)*s(2)/rho, -s(1)*s(2)/rho , rho ) / Dot(s,s);
}
