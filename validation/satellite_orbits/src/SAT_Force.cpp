//------------------------------------------------------------------------------
//
// SAT_Force.cpp
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

#define SAT_FORCE_CPP

#include <cmath>

#include "SAT_Const.h"
#include "SAT_Force.h"
#include "SAT_RefSys.h"
#include "SAT_VecMat.h"


// Local funtions

namespace 
{
  // Fractional part of a number (y=x-[x])
  double Frac (double x) { return x-floor(x); };

  // Earth gravity field JGM3

  // Gravitational coefficients C, S are efficiently stored in a single
  // array CS. The lower triangle matrix CS holds the non-sectorial C
  // coefficients C_n,m (n!=m). Sectorial C coefficients C_n,n are the 
  // diagonal elements of CS and the upper triangular matrix stores
  // the S_n,m (m!=0) coefficients in columns, for the same degree n.
  // Mapping of CS to C, S is achieved through 
  // C_n,m = CS(n,m), S_n,m = CS(m-1,n)

  const double R_JGM3 = 6378.1363e3;           // Radius Earth [m]; JGM3
  const int    N_JGM3 = 20;
  const double CS_JGM3[N_JGM3+1][N_JGM3+1] = {
    { 1.000000e+00, 0.000000e+00, 1.543100e-09, 2.680119e-07,-4.494599e-07,     
     -8.066346e-08, 2.116466e-08, 6.936989e-08, 4.019978e-08, 1.423657e-08,     
     -8.128915e-08,-1.646546e-08,-2.378448e-08, 2.172109e-08, 1.443750e-08,     
      4.154186e-09, 1.660440e-08,-1.427822e-08,-1.817656e-08, 7.160542e-11,     
      2.759192e-09                                                       },     
    { 0.000000e+00, 0.000000e+00,-9.038681e-07,-2.114024e-07, 1.481555e-07,     
     -5.232672e-08,-4.650395e-08, 9.282314e-09, 5.381316e-09,-2.228679e-09,     
     -3.057129e-09,-5.097360e-09, 1.416422e-09,-2.545587e-09,-1.089217e-10,     
     -1.045474e-09, 7.856272e-10, 2.522818e-10, 3.427413e-10,-1.008909e-10,     
      3.216826e-10                                                       },     
    {-1.082627e-03,-2.414000e-10, 1.574536e-06, 1.972013e-07,-1.201129e-08,     
     -7.100877e-09, 1.843134e-10,-3.061150e-09,-8.723520e-10,-5.633921e-10,     
     -8.989333e-10,-6.863521e-10, 9.154575e-11, 3.005522e-10, 5.182512e-11,     
      3.265044e-11,-4.271981e-11, 1.297841e-11,-4.278803e-12,-1.190759e-12,     
      3.778260e-11                                                       },     
    { 2.532435e-06, 2.192799e-06, 3.090160e-07, 1.005589e-07, 6.525606e-09,     
      3.873005e-10,-1.784491e-09,-2.636182e-10, 9.117736e-11, 1.717309e-11,     
     -4.622483e-11,-2.677798e-11, 9.170517e-13,-2.960682e-12,-3.750977e-12,     
      1.116419e-12, 5.250141e-12, 2.159727e-12, 1.105860e-13,-3.556436e-13,     
     -1.178441e-12                                                       },     
    { 1.619331e-06,-5.087253e-07, 7.841223e-08, 5.921574e-08,-3.982396e-09,     
     -1.648204e-09,-4.329182e-10, 6.397253e-12, 1.612521e-11,-5.550919e-12,     
     -3.122269e-12, 1.982505e-12, 2.033249e-13, 1.214266e-12,-2.217440e-13,     
      8.637823e-14,-1.205563e-14, 2.923804e-14, 1.040715e-13, 9.006136e-14,     
     -1.823414e-14                                                       },     
    { 2.277161e-07,-5.371651e-08, 1.055905e-07,-1.492615e-08,-2.297912e-09,     
      4.304768e-10,-5.527712e-11, 1.053488e-11, 8.627743e-12, 2.940313e-12,     
     -5.515591e-13, 1.346234e-13, 9.335408e-14,-9.061871e-15, 2.365713e-15,     
     -2.505252e-14,-1.590014e-14,-9.295650e-15,-3.743268e-15, 3.176649e-15,     
     -5.637288e-17                                                       },     
    {-5.396485e-07,-5.987798e-08, 6.012099e-09, 1.182266e-09,-3.264139e-10,     
     -2.155771e-10, 2.213693e-12, 4.475983e-13, 3.814766e-13,-1.846792e-13,     
     -2.650681e-15,-3.728037e-14, 7.899913e-15,-9.747983e-16,-3.193839e-16,     
      2.856094e-16,-2.590259e-16,-1.190467e-16, 8.666599e-17,-8.340023e-17,     
     -8.899420e-19                                                       },     
    { 3.513684e-07, 2.051487e-07, 3.284490e-08, 3.528541e-09,-5.851195e-10,     
      5.818486e-13,-2.490718e-11, 2.559078e-14, 1.535338e-13,-9.856184e-16,     
     -1.052843e-14, 1.170448e-15, 3.701523e-16,-1.095673e-16,-9.074974e-17,     
      7.742869e-17, 1.086771e-17, 4.812890e-18, 2.015619e-18,-5.594661e-18,     
      1.459810e-18                                                       },     
    { 2.025187e-07, 1.603459e-08, 6.576542e-09,-1.946358e-10,-3.189358e-10,     
     -4.615173e-12,-1.839364e-12, 3.429762e-13,-1.580332e-13, 7.441039e-15,     
     -7.011948e-16, 2.585245e-16, 6.136644e-17, 4.870630e-17, 1.489060e-17,     
      1.015964e-17,-5.700075e-18,-2.391386e-18, 1.794927e-18, 1.965726e-19,     
     -1.128428e-19                                                       },     
    { 1.193687e-07, 9.241927e-08, 1.566874e-09,-1.217275e-09,-7.018561e-12,     
     -1.669737e-12, 8.296725e-13,-2.251973e-13, 6.144394e-14,-3.676763e-15,     
     -9.892610e-17,-1.736649e-17, 9.242424e-18,-4.153238e-18,-6.937464e-20,     
      3.275583e-19, 1.309613e-19, 1.026767e-19,-1.437566e-20,-1.268576e-20,     
     -6.100911e-21                                                       },     
    { 2.480569e-07, 5.175579e-08,-5.562846e-09,-4.195999e-11,-4.967025e-11,     
     -3.074283e-12,-2.597232e-13, 6.909154e-15, 4.635314e-15, 2.330148e-15,     
      4.170802e-16,-1.407856e-17,-2.790078e-19,-6.376262e-20,-1.849098e-19,     
      3.595115e-20,-2.537013e-21, 4.480853e-21, 4.348241e-22, 1.197796e-21,     
     -1.138734e-21                                                       },     
    {-2.405652e-07, 9.508428e-09, 9.542030e-10,-1.409608e-10,-1.685257e-11,     
      1.489441e-12,-5.754671e-15, 1.954262e-15,-2.924949e-16,-1.934320e-16,     
     -4.946396e-17, 9.351706e-18,-9.838299e-20, 1.643922e-19,-1.658377e-20,     
      2.905537e-21, 4.983891e-22, 6.393876e-22,-2.294907e-22, 6.437043e-23,     
      6.435154e-23                                                       },     
    { 1.819117e-07,-3.068001e-08, 6.380398e-10, 1.451918e-10,-2.123815e-11,     
      8.279902e-13, 7.883091e-15,-4.131557e-15,-5.708254e-16, 1.012728e-16,     
     -1.840173e-18, 4.978700e-19,-2.108949e-20, 2.503221e-20, 3.298844e-21,     
     -8.660491e-23, 6.651727e-24, 5.110031e-23,-3.635064e-23,-1.311958e-23,     
      1.534228e-24                                                       },     
    { 2.075677e-07,-2.885131e-08, 2.275183e-09,-6.676768e-11,-3.452537e-13,     
      1.074251e-12,-5.281862e-14, 3.421269e-16,-1.113494e-16, 2.658019e-17,     
      4.577888e-18,-5.902637e-19,-5.860603e-20,-2.239852e-20,-6.914977e-23,     
     -6.472496e-23,-2.741331e-23, 2.570941e-24,-1.074458e-24,-4.305386e-25,     
     -2.046569e-25                                                       },     
    {-1.174174e-07,-9.997710e-09,-1.347496e-09, 9.391106e-11, 3.104170e-13,     
      3.932888e-13,-1.902110e-14, 2.787457e-15,-2.125248e-16, 1.679922e-17,     
      1.839624e-18, 7.273780e-20, 4.561174e-21, 2.347631e-21,-7.142240e-22,     
     -2.274403e-24,-2.929523e-24, 1.242605e-25,-1.447976e-25,-3.551992e-26,     
     -7.473051e-28                                                       },     
    { 1.762727e-08, 6.108862e-09,-7.164511e-10, 1.128627e-10,-6.013879e-12,     
      1.293499e-13, 2.220625e-14, 2.825477e-15,-1.112172e-16, 3.494173e-18,     
      2.258283e-19,-1.828153e-21,-6.049406e-21,-5.705023e-22, 1.404654e-23,     
     -9.295855e-24, 5.687404e-26, 1.057368e-26, 4.931703e-27,-1.480665e-27,     
      2.400400e-29                                                       },     
    {-3.119431e-08, 1.356279e-08,-6.713707e-10,-6.451812e-11, 4.698674e-12,     
     -9.690791e-14, 6.610666e-15,-2.378057e-16,-4.460480e-17,-3.335458e-18,     
     -1.316568e-19, 1.643081e-20, 1.419788e-21, 9.260416e-23,-1.349210e-23,     
     -1.295522e-24,-5.943715e-25,-9.608698e-27, 3.816913e-28,-3.102988e-28,     
     -8.192994e-29                                                       },     
    { 1.071306e-07,-1.262144e-08,-4.767231e-10, 1.175560e-11, 6.946241e-13,     
     -9.316733e-14,-4.427290e-15, 4.858365e-16, 4.814810e-17, 2.752709e-19,     
     -2.449926e-20,-6.393665e-21, 8.842755e-22, 4.178428e-23,-3.177778e-24,     
      1.229862e-25,-8.535124e-26,-1.658684e-26,-1.524672e-28,-2.246909e-29,     
     -5.508346e-31                                                       },     
    { 4.421672e-08, 1.958333e-09, 3.236166e-10,-5.174199e-12, 4.022242e-12,     
      3.088082e-14, 3.197551e-15, 9.009281e-17, 2.534982e-17,-9.526323e-19,     
      1.741250e-20,-1.569624e-21,-4.195542e-22,-6.629972e-24,-6.574751e-25,     
     -2.898577e-25, 7.555273e-27, 3.046776e-28, 3.696154e-29, 1.845778e-30,     
      6.948820e-31                                                       },     
    {-2.197334e-08,-3.156695e-09, 7.325272e-10,-1.192913e-11, 9.941288e-13,     
      3.991921e-14,-4.220405e-16, 7.091584e-17, 1.660451e-17, 9.233532e-20,     
     -5.971908e-20, 1.750987e-21,-2.066463e-23,-3.440194e-24,-1.487095e-25,     
     -4.491878e-26,-4.558801e-27, 5.960375e-28, 8.263952e-29,-9.155723e-31,     
     -1.237749e-31                                                       },     
    { 1.203146e-07, 3.688524e-09, 4.328972e-10,-6.303973e-12, 2.869669e-13,     
     -3.011115e-14, 1.539793e-15,-1.390222e-16, 1.766707e-18, 3.471731e-19,     
     -3.447438e-20, 8.760347e-22,-2.271884e-23, 5.960951e-24, 1.682025e-25,     
     -2.520877e-26,-8.774566e-28, 2.651434e-29, 8.352807e-30,-1.878413e-31,     
      4.054696e-32                                                       }     
  };


}


// Earth gravity model (JGM3)

GravModel Grav ( N_JGM3,            // Degree and order
                 GM_Earth,          // Gravitational coefficient [m^3/s^2]
                 R_JGM3,            // Reference radius [m]
                 &CS_JGM3[0][0] );  // Unnormalized harmonic coefficients



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

Vector Sun (double Mjd_TT)
{
  // Constants

  const double eps = 23.43929111*Rad;             // Obliquity of J2000 ecliptic
  const double T   = (Mjd_TT-MJD_J2000)/36525.0;  // Julian cent. since J2000

  // Variables

  double L, M, r;
  Vector r_Sun(3);

  // Mean anomaly, ecliptic longitude and radius

  M = pi2 * Frac ( 0.9931267 + 99.9973583*T);                    // [rad]
  L = pi2 * Frac ( 0.7859444 + M/pi2 + 
                    (6892.0*sin(M)+72.0*sin(2.0*M)) / 1296.0e3); // [rad]
  r = 149.619e9 - 2.499e9*cos(M) - 0.021e9*cos(2*M);             // [m]
  
  // Equatorial position vector

  r_Sun = R_x(-eps) * Vector(r*cos(L),r*sin(L),0.0);
  
  return r_Sun;
  
}


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

Vector Moon (double Mjd_TT)
{
  // Constants

  const double eps = 23.43929111*Rad;             // Obliquity of J2000 ecliptic
  const double T   = (Mjd_TT-MJD_J2000)/36525.0;  // Julian cent. since J2000

  // Variables

  double  L_0, l,lp, F, D, dL, S, h, N;
  double  L, B, R, cosB;
  Vector  r_Moon(3);
  
  
  // Mean elements of lunar orbit

  L_0 =     Frac ( 0.606433 + 1336.851344*T );     // Mean longitude [rev]
                                                   // w.r.t. J2000 equinox
  l   = pi2*Frac ( 0.374897 + 1325.552410*T );     // Moon's mean anomaly [rad]
  lp  = pi2*Frac ( 0.993133 +   99.997361*T );     // Sun's mean anomaly [rad]
  D   = pi2*Frac ( 0.827361 + 1236.853086*T );     // Diff. long. Moon-Sun [rad]
  F   = pi2*Frac ( 0.259086 + 1342.227825*T );     // Argument of latitude 
    
  
  // Ecliptic longitude (w.r.t. equinox of J2000)

  dL = +22640*sin(l) - 4586*sin(l-2*D) + 2370*sin(2*D) +  769*sin(2*l) 
       -668*sin(lp) - 412*sin(2*F) - 212*sin(2*l-2*D) - 206*sin(l+lp-2*D)
       +192*sin(l+2*D) - 165*sin(lp-2*D) - 125*sin(D) - 110*sin(l+lp)
       +148*sin(l-lp) - 55*sin(2*F-2*D);

  L = pi2 * Frac( L_0 + dL/1296.0e3 );  // [rad]

  // Ecliptic latitude

  S  = F + (dL+412*sin(2*F)+541*sin(lp)) / Arcs; 
  h  = F-2*D;
  N  = -526*sin(h) + 44*sin(l+h) - 31*sin(-l+h) - 23*sin(lp+h) 
       +11*sin(-lp+h) - 25*sin(-2*l+F) + 21*sin(-l+F);

  B = ( 18520.0*sin(S) + N ) / Arcs;   // [rad]
    
  cosB = cos(B);

  // Distance [m]

  R = 385000e3 - 20905e3*cos(l) - 3699e3*cos(2*D-l) - 2956e3*cos(2*D)
      -570e3*cos(2*l) + 246e3*cos(2*l-2*D) - 205e3*cos(lp-2*D) 
      -171e3*cos(l+2*D) - 152e3*cos(l+lp-2*D);   

  // Equatorial coordinates

  r_Moon = R_x(-eps) * Vector ( R*cos(L)*cosB, R*sin(L)*cosB, R*sin(B) );
    
  return r_Moon;
  
}


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

double Illumination ( const Vector& r, const Vector& r_Sun )
{                      

  Vector e_Sun = r_Sun / Norm(r_Sun);   // Sun direction unit vector
  double s     = Dot ( r, e_Sun );      // Projection of s/c position 

  return ( ( s>0 || Norm(r-s*e_Sun)>R_Earth ) ?  1.0 : 0.0 );
}


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
                      int n_max, int m_max )
{

  // Local variables
  
  int     n,m;                           // Loop counters
  double  r_sqr, rho, Fac;               // Auxiliary quantities
  double  x0,y0,z0;                      // Normalized coordinates
  double  ax,ay,az;                      // Acceleration vector 
  double  C,S;                           // Gravitational coefficients
  Vector  r_bf(3);                       // Body-fixed position
  Vector  a_bf(3);                       // Body-fixed acceleration
  
  Matrix  V(n_max+2,n_max+2);            // Harmonic functions
  Matrix  W(n_max+2,n_max+2);            // work array (0..n_max+1,0..n_max+1)
       
  
  // Body-fixed position 
  
  r_bf = E * r;
  
  // Auxiliary quantities
  
  r_sqr =  Dot(r_bf,r_bf);               // Square of distance
  rho   =  R_ref*R_ref / r_sqr;
    
  x0 = R_ref * r_bf(0) / r_sqr;          // Normalized
  y0 = R_ref * r_bf(1) / r_sqr;          // coordinates
  z0 = R_ref * r_bf(2) / r_sqr;
  
  //
  // Evaluate harmonic functions 
  //   V_nm = (R_ref/r)^(n+1) * P_nm(sin(phi)) * cos(m*lambda)
  // and 
  //   W_nm = (R_ref/r)^(n+1) * P_nm(sin(phi)) * sin(m*lambda)
  // up to degree and order n_max+1
  //
  
  // Calculate zonal terms V(n,0); set W(n,0)=0.0
  
  V(0,0) = R_ref / sqrt(r_sqr);
  W(0,0) = 0.0;
        
  V(1,0) = z0 * V(0,0);
  W(1,0) = 0.0;
  
  for (n=2; n<=n_max+1; n++) {
    V(n,0) = ( (2*n-1) * z0 * V(n-1,0) - (n-1) * rho * V(n-2,0) ) / n;
    W(n,0) = 0.0;
  };
  
  // Calculate tesseral and sectorial terms 
  
  for (m=1; m<=m_max+1; m++) {
      
    // Calculate V(m,m) .. V(n_max+1,m)
  
    V(m,m) = (2*m-1) * ( x0*V(m-1,m-1) - y0*W(m-1,m-1) );
    W(m,m) = (2*m-1) * ( x0*W(m-1,m-1) + y0*V(m-1,m-1) );
  
    if (m<=n_max) {
      V(m+1,m) = (2*m+1) * z0 * V(m,m);
      W(m+1,m) = (2*m+1) * z0 * W(m,m);
    };
  
    for (n=m+2; n<=n_max+1; n++) {
      V(n,m) = ( (2*n-1)*z0*V(n-1,m) - (n+m-1)*rho*V(n-2,m) ) / (n-m);
      W(n,m) = ( (2*n-1)*z0*W(n-1,m) - (n+m-1)*rho*W(n-2,m) ) / (n-m);
    };
  
  };
  
  //
  // Calculate accelerations ax,ay,az
  //
  
  ax = ay = az = 0.0;
  
  for (m=0; m<=m_max; m++)
    for (n=m; n<=n_max ; n++)
      if (m==0) {
        C = CS(n,0);   // = C_n,0
        ax -=       C * V(n+1,1);
        ay -=       C * W(n+1,1);
        az -= (n+1)*C * V(n+1,0);
      }
      else { 
        C = CS(n,m);   // = C_n,m 
        S = CS(m-1,n); // = S_n,m 
        Fac = 0.5 * (n-m+1) * (n-m+2);
        ax +=   + 0.5 * ( - C * V(n+1,m+1) - S * W(n+1,m+1) )
                + Fac * ( + C * V(n+1,m-1) + S * W(n+1,m-1) );
        ay +=   + 0.5 * ( - C * W(n+1,m+1) + S * V(n+1,m+1) ) 
                + Fac * ( - C * W(n+1,m-1) + S * V(n+1,m-1) );
        az += (n-m+1) * ( - C * V(n+1,m)   - S * W(n+1,m)   );
      };
  
  // Body-fixed acceleration

  a_bf = (GM/(R_ref*R_ref)) * Vector(ax,ay,az);
  
  // Inertial acceleration 
  
  return  Transp(E)*a_bf;
         
}


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
//   r           Satellite position vector 
//   s           Point mass position vector
//   GM          Gravitational coefficient of point mass
//   <return>    Acceleration (a=d^2r/dt^2)
//
//------------------------------------------------------------------------------

Vector AccelPointMass (const Vector& r, const Vector& s, double GM)
{    

   Vector d(3);
  
   //  Relative position vector of satellite w.r.t. point mass 
  
   d = r - s;
  
   // Acceleration 
  
   return  (-GM) * ( d/pow(Norm(d),3) + s/pow(Norm(s),3) );

}


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
                    double P0, double AU )
{
    
  Vector d(3);
  
  // Relative position vector of spacecraft w.r.t. Sun
  
  d = r - r_Sun;
  
  // Acceleration 
  
  return  CR*(Area/mass)*P0*(AU*AU) * d / pow(Norm(d),3); 
  
}


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
                   double Area, double mass, double CD )
{
    
  // Constants

  // Earth angular velocity vector [rad/s]
  const double Data_omega[3]= { 0.0, 0.0, 7.29212e-5 };
  const Vector omega ( &Data_omega[0], 3); 
 

  // Variables

  double v_abs, dens;
  Vector r_tod(3), v_tod(3);
  Vector v_rel(3), a_tod(3);
  Matrix T_trp(3,3);

  
  // Transformation matrix to ICRF/EME2000 system
  
  T_trp = Transp(T);


  // Position and velocity in true-of-date system

  r_tod = T * r;
  v_tod = T * v;


  // Velocity relative to the Earth's atmosphere

  v_rel = v_tod - Cross(omega,r_tod);
  v_abs = Norm(v_rel);


  // Atmospheric density due to modified Harris-Priester model

  dens = Density_HP (Mjd_TT,r_tod);

  // Acceleration 
  
  a_tod = -0.5*CD*(Area/mass)*dens*v_abs*v_rel;

  return T_trp * a_tod;  
  
}


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

double Density_HP ( double Mjd_TT, const Vector& r_tod )
{
  // Constants

  const double upper_limit =   1000.0;           // Upper height limit [km]
  const double lower_limit =    100.0;           // Lower height limit [km]
  const double ra_lag      = 0.523599;           // Right ascension lag [rad]
  const int    n_prm       =        3;           // Harris-Priester parameter 
                                                 // 2(6) low(high) inclination

  // Harris-Priester atmospheric density model parameters 
  // Height [km], minimum density, maximum density [gm/km^3]

  const int    N_Coef = 50;
  const double Data_h[N_Coef]= {
      100.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0, 200.0,     
      210.0, 220.0, 230.0, 240.0, 250.0, 260.0, 270.0, 280.0, 290.0, 300.0,     
      320.0, 340.0, 360.0, 380.0, 400.0, 420.0, 440.0, 460.0, 480.0, 500.0,     
      520.0, 540.0, 560.0, 580.0, 600.0, 620.0, 640.0, 660.0, 680.0, 700.0,     
      720.0, 740.0, 760.0, 780.0, 800.0, 840.0, 880.0, 920.0, 960.0,1000.0};    
  const double Data_c_min[N_Coef] = {
      4.974e+05, 2.490e+04, 8.377e+03, 3.899e+03, 2.122e+03, 1.263e+03,         
      8.008e+02, 5.283e+02, 3.617e+02, 2.557e+02, 1.839e+02, 1.341e+02,         
      9.949e+01, 7.488e+01, 5.709e+01, 4.403e+01, 3.430e+01, 2.697e+01,         
      2.139e+01, 1.708e+01, 1.099e+01, 7.214e+00, 4.824e+00, 3.274e+00,         
      2.249e+00, 1.558e+00, 1.091e+00, 7.701e-01, 5.474e-01, 3.916e-01,         
      2.819e-01, 2.042e-01, 1.488e-01, 1.092e-01, 8.070e-02, 6.012e-02,         
      4.519e-02, 3.430e-02, 2.632e-02, 2.043e-02, 1.607e-02, 1.281e-02,         
      1.036e-02, 8.496e-03, 7.069e-03, 4.680e-03, 3.200e-03, 2.210e-03,         
      1.560e-03, 1.150e-03                                            };        
  const double Data_c_max[N_Coef] = {
      4.974e+05, 2.490e+04, 8.710e+03, 4.059e+03, 2.215e+03, 1.344e+03,         
      8.758e+02, 6.010e+02, 4.297e+02, 3.162e+02, 2.396e+02, 1.853e+02,         
      1.455e+02, 1.157e+02, 9.308e+01, 7.555e+01, 6.182e+01, 5.095e+01,         
      4.226e+01, 3.526e+01, 2.511e+01, 1.819e+01, 1.337e+01, 9.955e+00,         
      7.492e+00, 5.684e+00, 4.355e+00, 3.362e+00, 2.612e+00, 2.042e+00,         
      1.605e+00, 1.267e+00, 1.005e+00, 7.997e-01, 6.390e-01, 5.123e-01,         
      4.121e-01, 3.325e-01, 2.691e-01, 2.185e-01, 1.779e-01, 1.452e-01,         
      1.190e-01, 9.776e-02, 8.059e-02, 5.741e-02, 4.210e-02, 3.130e-02,         
      2.360e-02, 1.810e-02                                            };        

  const Vector h ( &Data_h[0], N_Coef);
  const Vector c_min ( &Data_c_min[0], N_Coef);
  const Vector c_max ( &Data_c_max[0], N_Coef);


  // Variables

  int    i, ih;                              // Height section variables        
  double height;                             // Earth flattening
  double dec_Sun, ra_Sun, c_dec;             // Sun declination, right asc.
  double c_psi2;                             // Harris-Priester modification
  double density, h_min, h_max, d_min, d_max;// Height, density parameters
  Vector r_Sun(3);                           // Sun position
  Vector u(3);                               // Apex of diurnal bulge

 
  // Satellite height
 
  height = Geodetic(r_tod).h/1000.0;              //  [km]
 

  // Exit with zero density outside height model limits

  if ( height >= upper_limit || height <= lower_limit ) 
  {  return 0.0;
  }


  // Sun right ascension, declination
 
  r_Sun = Sun ( Mjd_TT );
  ra_Sun  = atan2( r_Sun(1), r_Sun(0) ); 
  dec_Sun = atan2( r_Sun(2), sqrt( pow(r_Sun(0),2)+pow(r_Sun(1),2) ) );


  // Unit vector u towards the apex of the diurnal bulge
  // in inertial geocentric coordinates
 
  c_dec = cos(dec_Sun);
  u(0) = c_dec * cos(ra_Sun + ra_lag);
  u(1) = c_dec * sin(ra_Sun + ra_lag);
  u(2) = sin(dec_Sun);


  // Cosine of half angle between satellite position vector and
  // apex of diurnal bulge

  c_psi2 = 0.5 + 0.5 * Dot(r_tod,u)/Norm(r_tod);


  // Height index search and exponential density interpolation
 
  ih = 0;                           // section index reset
  for ( i=0; i<N_Coef-1; i++)       // loop over N_Coef height regimes
  {
    if ( height >= h(i) && height < h(i+1) ) 
    {
      ih = i;                       // ih identifies height section
      break;
    }
  }

  h_min = ( h(ih) - h(ih+1) )/log( c_min(ih+1)/c_min(ih) );
  h_max = ( h(ih) - h(ih+1) )/log( c_max(ih+1)/c_max(ih) );

  d_min = c_min(ih) * exp( (h(ih)-height)/h_min );
  d_max = c_max(ih) * exp( (h(ih)-height)/h_max );

  // Density computation

  density = d_min + (d_max-d_min)*pow(c_psi2,n_prm);


  return density * 1.0e-12;       // [kg/m^3]
                       
}


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
                   double Area, double mass, double CR, double CD )
{

  double Mjd_UT1;
  Vector a(3), r_Sun(3), r_Moon(3);
  Matrix T(3,3), E(3,3);

  // Acceleration due to harmonic gravity field

  Mjd_UT1 = Mjd_TT;

  T = NutMatrix(Mjd_TT) * PrecMatrix(MJD_J2000,Mjd_TT);
  E = GHAMatrix(Mjd_UT1) * T;

  a = AccelHarmonic ( r,E, Grav.GM,Grav.R_ref,Grav.CS, Grav.n_max,Grav.m_max );

  // Luni-solar perturbations 

  r_Sun  = Sun(Mjd_TT);
  r_Moon = Moon(Mjd_TT);

  a += AccelPointMass ( r, r_Sun,  GM_Sun  ); 
  a += AccelPointMass ( r, r_Moon, GM_Moon ); 

  // Solar radiation pressure

  a += Illumination ( r, r_Sun )
       * AccelSolrad ( r, r_Sun, Area, mass, CR, P_Sol, AU );

  // Atmospheric drag

  a += AccelDrag ( Mjd_TT, r, v, T, Area, mass, CD );

  // Acceleration
  
  return a;

}

