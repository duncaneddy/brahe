//------------------------------------------------------------------------------
//
// SAT_Kepler.cpp
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

#include <cmath>
#include <iostream>
#include <limits>

#include "SAT_Const.h"
#include "SAT_Kepler.h"
#include "SAT_VecMat.h"


using std::ostream;
using std::cerr;
using std::endl;


// Local constants and functions

namespace // Unnamed namespace
{
  
  // Machine accuracy
  
  const double eps_mach = std::numeric_limits<double>::epsilon();

  //
  // Fractional part of a number (y=x-[x])
  //
  
  double Frac (double x) { return x-floor(x); };

  //
  // x mod y
  //

  double Modulo (double x, double y) { return y*Frac(x/y); }
  
  //
  // F : local function for use by FindEta()
  // F = 1 - eta +(m/eta**2)*W(m/eta**2-l)
  //

  double F (double eta, double m, double l)
  {
    // Constants
    const double eps = 100.0 * eps_mach;

    // Variables
    double  w,W,a,n,g;
    
    w = m/(eta*eta)-l; 

    if (fabs(w)<0.1) { // Series expansion
      W = a = 4.0/3.0; n = 0.0;
      do {
        n += 1.0;  a *= w*(n+2.0)/(n+1.5);  W += a; 
      }
      while (fabs(a) >= eps);
    }
    else {
      if (w > 0.0) {
        g = 2.0*asin(sqrt(w));  
        W = (2.0*g - sin(2.0*g)) / pow(sin(g), 3);
      }
      else {
        g = 2.0*log(sqrt(-w)+sqrt(1.0-w));  // =2.0*arsinh(sqrt(-w))
        W = (sinh(2.0*g) - 2.0*g) / pow(sinh(g), 3);
      }
    }
  
    return ( 1.0 - eta + (w+l)*W );
  }   // End of function F


} // End of unnamed namespace



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

double EccAnom (double M, double e)
{

  // Constants

  const int maxit = 15;
  const double eps = 100.0*eps_mach;

  // Variables

  int    i=0;
  double E, f;

  // Starting value

  M = Modulo(M, 2.0*pi);   
  if (e<0.8) E=M; else E=pi;

  // Iteration

  do {
    f = E - e*sin(E) - M;
    E = E - f / ( 1.0 - e*cos(E) );
    ++i;
    if (i==maxit) {
      cerr << " convergence problems in EccAnom" << endl;
      break;
    }
  }
  while (fabs(f) > eps);

  return E;

}


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

double FindEta (const Vector& r_a, const Vector& r_b, double tau)
{
  // Constants

  const int maxit = 30;
  const double delta = 100.0*eps_mach;  

  // Variables

  int    i;
  double kappa, m, l, s_a, s_b, eta_min, eta1, eta2, F1, F2, d_eta;


  // Auxiliary quantities

  s_a = Norm(r_a);  
  s_b = Norm(r_b);  

  kappa = sqrt ( 2.0*(s_a*s_b+Dot(r_a,r_b)) );

  m = tau*tau / pow(kappa,3);   
  l = (s_a+s_b) / (2.0*kappa) - 0.5;

  eta_min = sqrt(m/(l+1.0));

  // Start with Hansen's approximation

  eta2 = ( 12.0 + 10.0*sqrt(1.0+(44.0/9.0)*m /(l+5.0/6.0)) ) / 22.0;
  eta1 = eta2 + 0.1;   

  // Secant method
  
  F1 = F(eta1, m, l);   
  F2 = F(eta2, m, l);  

  i = 0;

  while (fabs(F2-F1) > delta)
  {
    d_eta = -F2*(eta2-eta1)/(F2-F1);  
    eta1 = eta2; F1 = F2; 
    while (eta2+d_eta<=eta_min)  d_eta *= 0.5;
    eta2 += d_eta;  
    F2 = F(eta2,m,l); ++i;
  
    if ( i == maxit ) {
      cerr << "WARNING: Convergence problems in FindEta" << endl;
      break;
    }
  }

  return eta2;
}


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
//   Kep       Keplerian elements (a,e,i,Omega,omega,M) with
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

Vector State ( double GM, const Vector& Kep, double dt )
{

  // Variables

  double  a,e,i,Omega,omega,M,M0,n;
  double  E,cosE,sinE, fac, R,V;
  Vector  r(3),v(3);
  Matrix  PQW(3,3);

  // Keplerian elements at epoch
  
  a = Kep(0);  Omega = Kep(3);
  e = Kep(1);  omega = Kep(4); 
  i = Kep(2);  M0    = Kep(5);

  // Mean anomaly

  if (dt==0.0) {
    M = M0;
  }
  else {
    n = sqrt (GM/(a*a*a));
    M = M0 +n*dt;
  };

  // Eccentric anomaly
  
  E  = EccAnom(M,e);   

  cosE = cos(E); 
  sinE = sin(E);

  // Perifocal coordinates

  fac = sqrt ( (1.0-e)*(1.0+e) );  

  R = a*(1.0-e*cosE);  // Distance
  V = sqrt(GM*a)/R;    // Velocity

  r = Vector ( a*(cosE-e), a*fac*sinE , 0.0 );
  v = Vector ( -V*sinE   , +V*fac*cosE, 0.0 ); 

  // Transformation to reference system (Gaussian vectors)
  
  PQW = R_z(-Omega) * R_x(-i) * R_z(-omega);

  r = PQW*r;
  v = PQW*v;

  // State vector 
  
  return Stack(r,v);

}


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
//   Kep       Keplerian elements (a,e,i,Omega,omega,M) at epoch with
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

Matrix StatePartials ( double GM, const Vector& Kep, double dt )
{

  // Variables

  int     k;
  double  a,e,i,Omega,omega,M,M0,n,dMda;
  double  E,cosE,sinE, fac, r,v, x,y,vx,vy;
  Matrix  PQW(3,3);
  Vector  P(3),Q(3),W(3),e_z(3),N(3);
  Vector  dPdi(3),dPdO(3),dPdo(3),dQdi(3),dQdO(3),dQdo(3); 
  Vector  dYda(6),dYde(6),dYdi(6),dYdO(6),dYdo(6),dYdM(6);
  Matrix  dYdA(6,6);
  
  // Keplerian elements at epoch
  
  a = Kep(0);  Omega = Kep(3);
  e = Kep(1);  omega = Kep(4); 
  i = Kep(2);  M0    = Kep(5);

  // Mean and eccentric anomaly

  n = sqrt (GM/(a*a*a));
  M = M0 +n*dt;
  E = EccAnom(M,e);   

  // Perifocal coordinates

  cosE = cos(E); 
  sinE = sin(E);
  fac  = sqrt((1.0-e)*(1.0+e));  

  r = a*(1.0-e*cosE);  // Distance
  v = sqrt(GM*a)/r;    // Velocity

  x  = +a*(cosE-e); y  = +a*fac*sinE;
  vx = -v*sinE;     vy = +v*fac*cosE; 

  // Transformation to reference system (Gaussian vectors) and partials
  
  PQW = R_z(-Omega) * R_x(-i) * R_z(-omega);

  P = PQW.Col(0);  Q = PQW.Col(1);  W = PQW.Col(2);

  e_z = Vector(0,0,1);  N = Cross(e_z,W); N = N/Norm(N);

  dPdi = Cross(N,P);  dPdO = Cross(e_z,P); dPdo =  Q;
  dQdi = Cross(N,Q);  dQdO = Cross(e_z,Q); dQdo = -P;
  
  // Partials w.r.t. semimajor axis, eccentricity and mean anomaly at time dt

  dYda = Stack ( (x/a)*P + (y/a)*Q,
                 (-vx/(2*a))*P + (-vy/(2*a))*Q );

  dYde = Stack ( (-a-pow(y/fac,2)/r)*P + (x*y/(r*fac*fac))*Q , 
                 (vx*(2*a*x+e*pow(y/fac,2))/(r*r))*P
                 + ((n/fac)*pow(a/r,2)*(x*x/r-pow(y/fac,2)/a))*Q );

  dYdM = Stack ( (vx*P+vy*Q)/n, (-n*pow(a/r,3))*(x*P+y*Q) );

  // Partials w.r.t. inlcination, node and argument of pericenter

  dYdi = Stack ( x*dPdi+y*dQdi, vx*dPdi+vy*dQdi ); 
  dYdO = Stack ( x*dPdO+y*dQdO, vx*dPdO+vy*dQdO ); 
  dYdo = Stack ( x*dPdo+y*dQdo, vx*dPdo+vy*dQdo ); 

  // Derivative of mean anomaly at time dt w.r.t. the semimajor axis at epoch

  dMda = -1.5*(n/a)*dt;  

  // Combined partial derivative matrix of state with respect to epoch elements
  
  for (k=0;k<6;k++) {
    dYdA(k,0) = dYda(k) + dYdM(k)*dMda;  
    dYdA(k,1) = dYde(k); 
    dYdA(k,2) = dYdi(k); 
    dYdA(k,3) = dYdO(k);
    dYdA(k,4) = dYdo(k);
    dYdA(k,5) = dYdM(k);
  }

  return dYdA;

}



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

Vector Elements ( double GM, const Vector& y )
{

  // Variables

  Vector  r(3),v(3),h(3);
  double  H, u, R;
  double  eCosE, eSinE, e2, E, nu;
  double  a,e,i,Omega,omega,M;

  r = y.slice(0,2);                                  // Position
  v = y.slice(3,5);                                  // Velocity
  
  h = Cross(r,v);                                    // Areal velocity
  H = Norm(h);

  Omega = atan2 ( h(0), -h(1) );                     // Long. ascend. node 
  Omega = Modulo(Omega,pi2);
  i     = atan2 ( sqrt(h(0)*h(0)+h(1)*h(1)), h(2) ); // Inclination        
  u     = atan2 ( r(2)*H, -r(0)*h(1)+r(1)*h(0) );    // Arg. of latitude   

  R  = Norm(r);                                      // Distance           

  a = 1.0 / (2.0/R-Dot(v,v)/GM);                     // Semi-major axis    

  eCosE = 1.0-R/a;                                   // e*cos(E)           
  eSinE = Dot(r,v)/sqrt(GM*a);                       // e*sin(E)           

  e2 = eCosE*eCosE +eSinE*eSinE;
  e  = sqrt(e2);                                     // Eccentricity 
  E  = atan2(eSinE,eCosE);                           // Eccentric anomaly  

  M  = Modulo(E-eSinE,pi2);                          // Mean anomaly

  nu = atan2(sqrt(1.0-e2)*eSinE, eCosE-e2);          // True anomaly

  omega = Modulo(u-nu,pi2);                          // Arg. of perihelion 
 
  // Keplerian elements vector

  return Vector(a,e,i,Omega,omega,M);

}


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
                  const Vector& r_a, const Vector& r_b )
{
  
  // Variables
  
  double  tau, eta, p;
  double  n, nu, E, u;
  double  s_a, s_b, s_0, fac, sinhH;
  double  cos_dnu, sin_dnu, ecos_nu, esin_nu;
  double  a, e, i, Omega, omega, M;
  Vector  e_a(3), r_0(3), e_0(3), W(3);

  // Calculate vector r_0 (fraction of r_b perpendicular to r_a) 
  // and the magnitudes of r_a,r_b and r_0

  s_a = Norm(r_a);  e_a = r_a/s_a;
  s_b = Norm(r_b); 
  fac = Dot(r_b,e_a); r_0 = r_b-fac*e_a;
  s_0 = Norm(r_0);  e_0 = r_0/s_0;
  
  // Inclination and ascending node 

  W     = Cross(e_a,e_0);
  Omega = atan2 ( W(0), -W(1) );                     // Long. ascend. node 
  Omega = Modulo(Omega,pi2);
  i     = atan2 ( sqrt(W(0)*W(0)+W(1)*W(1)), W(2) ); // Inclination        
  if (i==0.0) 
    u = atan2 ( r_a(1), r_a(0) );
  else 
    u = atan2 ( +e_a(2) , -e_a(0)*W(1)+e_a(1)*W(0) );
  
  // Semilatus rectum
  
  tau = sqrt(GM) * 86400.0*fabs(Mjd_b-Mjd_a);   
  eta = FindEta ( r_a, r_b, tau );
  p   = pow ( s_a*s_0*eta/tau, 2 );   

  // Eccentricity, true anomaly and argument of perihelion

  cos_dnu = fac / s_b;    
  sin_dnu = s_0 / s_b;

  ecos_nu = p / s_a - 1.0;  
  esin_nu = ( ecos_nu * cos_dnu - (p/s_b-1.0) ) / sin_dnu;

  e  = sqrt ( ecos_nu*ecos_nu + esin_nu*esin_nu );
  nu = atan2(esin_nu,ecos_nu);

  omega = Modulo(u-nu,pi2);

  // Perihelion distance, semimajor axis and mean motion
  
  a = p/(1.0-e*e);
  n = sqrt ( GM / fabs(a*a*a) );

  // Mean anomaly and time of perihelion passage

  if (e<1.0) {
    E = atan2 ( sqrt((1.0-e)*(1.0+e)) * esin_nu,  ecos_nu + e*e );
    M = Modulo ( E - e*sin(E), pi2 );
  }
  else 
  {
    sinhH = sqrt((e-1.0)*(e+1.0)) * esin_nu / ( e + e * ecos_nu );
    M = e * sinhH - log ( sinhH + sqrt(1.0+sinhH*sinhH) );
  }

  // Keplerian elements vector

  return Vector(a,e,i,Omega,omega,M);

}


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
               Vector& Y, Matrix& dYdY0 )
{

  // Variables
  
  int     k;
  double  a,e,i,n, sqe2,naa;
  double  P_aM, P_eM, P_eo, P_io, P_iO;
  Vector  A0(6);
  Matrix  dY0dA0(6,6), dYdA0(6,6), dA0dY0(6,6);

  // Orbital elements at epoch

  A0 = Elements(GM,Y0);

  a = A0(0);  e = A0(1);  i = A0(2);

  n = sqrt (GM/(a*a*a));

  // Propagated state 

  Y = State(GM,A0,dt);

  // State vector partials w.r.t epoch elements

  dY0dA0 = StatePartials(GM,A0,0.0);
  dYdA0  = StatePartials(GM,A0,dt);

  // Poisson brackets

  sqe2 = sqrt((1.0-e)*(1.0+e));
  naa  = n*a*a;
      
  P_aM = -2.0/(n*a);                   // P(a,M)     = -P(M,a)
  P_eM = -(1.0-e)*(1.0+e)/(naa*e);     // P(e,M)     = -P(M,e)
  P_eo = +sqe2/(naa*e);                // P(e,omega) = -P(omega,e)
  P_io = -1.0/(naa*sqe2*tan(i));       // P(i,omega) = -P(omega,i)
  P_iO = +1.0/(naa*sqe2*sin(i));       // P(i,Omega) = -P(Omega,i)

  // Partials of epoch elements w.r.t. epoch state

  for (k=0;k<3;k++) {
      
    dA0dY0(0,k)   = + P_aM*dY0dA0(k+3,5);
    dA0dY0(0,k+3) = - P_aM*dY0dA0(k  ,5);
        
    dA0dY0(1,k)   = + P_eo*dY0dA0(k+3,4) + P_eM*dY0dA0(k+3,5);
    dA0dY0(1,k+3) = - P_eo*dY0dA0(k  ,4) - P_eM*dY0dA0(k  ,5);

    dA0dY0(2,k)   = + P_iO*dY0dA0(k+3,3) + P_io*dY0dA0(k+3,4);
    dA0dY0(2,k+3) = - P_iO*dY0dA0(k  ,3) - P_io*dY0dA0(k  ,4);
        
    dA0dY0(3,k)   = - P_iO*dY0dA0(k+3,2);
    dA0dY0(3,k+3) = + P_iO*dY0dA0(k  ,2);

    dA0dY0(4,k)   = - P_eo*dY0dA0(k+3,1) - P_io*dY0dA0(k+3,2);
    dA0dY0(4,k+3) = + P_eo*dY0dA0(k  ,1) + P_io*dY0dA0(k  ,2);
     
    dA0dY0(5,k)   = - P_aM*dY0dA0(k+3,0) - P_eM*dY0dA0(k+3,1);
    dA0dY0(5,k+3) = + P_aM*dY0dA0(k  ,0) + P_eM*dY0dA0(k  ,1);

  };

  // State transition matrix

  dYdY0 = dYdA0 * dA0dY0;

}
