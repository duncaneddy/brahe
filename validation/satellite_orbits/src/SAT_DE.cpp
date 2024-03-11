//------------------------------------------------------------------------------
//
// SAT_DE.cpp
// 
// Purpose:
//
//   Numerical integration methods for ordinaray differential equations
//
//   This module provides implemenations of the 4th-order Runge-Kutta method
//   and the variable order variable stepsize multistep method of Shampine &
//   Gordon.
// 
// Reference:
//
//   Shampine, Gordon: "Computer solution of Ordinary Differential Equations",
//   Freeman and Comp., San Francisco (1975).
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
#include <cstdlib>
#include <iostream>

#include <limits>

#include "SAT_DE.h"


namespace // Unnamed namespace
{
  // Constants

  const int    maxnum = 500;  // Maximum number of steps to take

  const double umach = std::numeric_limits<double>::epsilon();
  const double twou   = 2.0*umach;   
  const double fouru  = 4.0*umach;   


  // Auxiliary functions (min, max, sign)

  template <class T>
  T max (T a, T b) { return ((a>b) ? a : b); };

  template <class T>
  T min (T a, T b) { return ((a<b) ? a : b); };

  // sign: returns absolute value of a with sign of b
  double sign(double a, double b)
  {
   return (b>=0.0) ? fabs(a) : - fabs(a);
  };

}


//------------------------------------------------------------------------------
//
// RK4 class (implementation)
//
//------------------------------------------------------------------------------

// Step

void RK4::Step (double& t, Vector& y, double h) {

  // Elementary RK4 step

  f( t      , y            , k_1, pAux );
  f( t+h/2.0, y+(h/2.0)*k_1, k_2, pAux );
  f( t+h/2.0, y+(h/2.0)*k_2, k_3, pAux );
  f( t+h    , y+h*k_3      , k_4, pAux );
  
  y = y + (h/6.0)*( k_1 + 2.0*k_2 + 2.0*k_3 + k_4 );

  // Update independent variable

  t = t + h;

};


//------------------------------------------------------------------------------
//
// DE class (implementation)
//
//------------------------------------------------------------------------------


//
// Default constructor
//

DE::DE ()
: f(0), 
  n_eqn(0),  
  pAux(0)
{
  State      = DE_INVPARAM;     // Status flag 
  PermitTOUT = true;            // Allow integration past tout by default
  t          = 0.0;
  relerr     = 0.0;             // Accuracy requirements
  abserr     = 0.0;
  kmax       = 12;
};


//
// Constructor
//

DE::DE (
      DEfunct    f_,            // Differential equation
      int        n_eqn_,        // Dimension
      void*      pAux_          // Pointer to auxiliary data
    )
: f(f_),
  n_eqn(n_eqn_), 
  pAux(pAux_)
{
  yy         = Vector(n_eqn);   // Allocate vectors with proper dimension
  wt         = Vector(n_eqn);
  p          = Vector(n_eqn);
  yp         = Vector(n_eqn);
  ypout      = Vector(n_eqn);
  phi        = Matrix(n_eqn,17);
  State      = DE_INVPARAM;     // Status flag 
  PermitTOUT = true;            // Allow integration past tout by default
  t          = 0.0;
  relerr     = 0.0;             // Accuracy requirements
  abserr     = 0.0;
  kmax       = 12;
};


//
// Constructor
//

void DE::Define (
      DEfunct    f_,            // Differential equation
      int        n_eqn_,        // Dimension
      void*      pAux_          // Pointer to auxiliary data
    )
{
  n_eqn      = n_eqn_; 
  f          = f_; 
  pAux       = pAux_;
  yy         = Vector(n_eqn);   // Allocate vectors with proper dimension
  wt         = Vector(n_eqn);
  p          = Vector(n_eqn);
  yp         = Vector(n_eqn);
  ypout      = Vector(n_eqn);
  phi        = Matrix(n_eqn,17);
  State      = DE_INVPARAM;     // Status flag 
  PermitTOUT = true;            // Allow integration past tout by default
  t          = 0.0;
  relerr     = 0.0;             // Accuracy requirements
  abserr     = 0.0;
  kmax       = 12;
};


//
// Integration step
//

void DE::Step (double& x, Vector& y, double& eps, bool& crash)
{

  // Constants

  // Powers of two (two(n)=2**n)
  static const double two[14] = 
     { 1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0,     
       256.0, 512.0, 1024.0, 2048.0, 4096.0, 8192.0 };

  static const double gstr[14] =  
     {1.0, 0.5, 0.0833, 0.0417, 0.0264, 0.0188,
      0.0143, 0.0114, 0.00936, 0.00789, 0.00679,   
      0.00592, 0.00524, 0.00468 };

  

  // Variables

  bool    success;
  int     i,ifail, im1, ip1, iq, j, km1, km2, knew, kp1, kp2;
  int     l, limit1, limit2, nsm2, nsp1, nsp2;
  double  absh, erk, erkm1, erkm2, erkp1, err, hnew;
  double  p5eps, r, reali, realns, rho, round, sum, tau;
  double  temp1, temp2, temp3, temp4, temp5, temp6, xold;


  //                                                                   
  // Begin block 0                                                     
  //                                                                   
  // Check if step size or error tolerance is too small for machine    
  // precision.  If first step, initialize phi array and estimate a    
  // starting step size. If step size is too small, determine an       
  // acceptable one.                                                   
  //                                                                   
  
  if (fabs(h) < fouru*fabs(x)) {
    h = sign(fouru*fabs(x),h);
    crash = true;
    return;                      // Exit 
  };

  p5eps  = 0.5*eps;
  crash  = false;
  g[1]   = 1.0;
  g[2]   = 0.5;
  sig[1] = 1.0;
  
  ifail = 0;

  // If error tolerance is too small, increase it to an 
  // acceptable value.                                  

  round = 0.0;
  for (l=0;l<n_eqn;l++) round += (y(l)*y(l))/(wt(l)*wt(l));
  round = twou*sqrt(round);
  if (p5eps<round) {
    eps = 2.0*round*(1.0+fouru);
    crash = true;
    return;
  };

  
  if (start) {
    // Initialize. Compute appropriate step size for first step. 
    f(x, y, yp, pAux);
    sum = 0.0;
    for (l=0;l<n_eqn;l++) {
      phi(l,1) = yp(l);
      phi(l,2) = 0.0;
      sum += (yp(l)*yp(l))/(wt(l)*wt(l));
    }
    sum  = sqrt(sum);
    absh = fabs(h);
    if (eps<16.0*sum*h*h) absh=0.25*sqrt(eps/sum);
    h    = sign(max(absh, fouru*fabs(x)), h);
    hold = 0.0;
    hnew = 0.0;
    k    = 1;
    kold = 0;
    start  = false;
    phase1 = true;
    nornd  = true;
    if (p5eps<=100.0*round) {
      nornd = false;
      for (l=0;l<n_eqn;l++) phi(l,15)=0.0;
    };
  };

  //                                                                   
  // End block 0                                                       
  //                                                                   


  //                                                                   
  // Repeat blocks 1, 2 (and 3) until step is successful               
  //                                                                   
  do {
    
    //                                                                 
    // Begin block 1                                                   
    //                                                                 
    // Compute coefficients of formulas for this step. Avoid computing 
    // those quantities not changed when step size is not changed.     
    //                                                                 

    kp1 = k+1;
    kp2 = k+2;
    km1 = k-1;
    km2 = k-2;
  
    // ns is the number of steps taken with size h, including the 
    // current one. When k<ns, no coefficients change.           

    if (h !=hold)  ns=0;
    if (ns<=kold)  ns=ns+1;
    nsp1 = ns+1;
  
    if (k>=ns) {

      // Compute those components of alpha[*],beta[*],psi[*],sig[*] 
      // which are changed                                          
      beta[ns] = 1.0;
      realns = ns;
      alpha[ns] = 1.0/realns;
      temp1 = h*realns;
      sig[nsp1] = 1.0;
      if (k>=nsp1) {
        for (i=nsp1;i<=k;i++) {
          im1   = i-1;
          temp2 = psi[im1];
          psi[im1] = temp1;
          beta[i]  = beta[im1]*psi[im1]/temp2;
          temp1    = temp2 + h;
          alpha[i] = h/temp1;
          reali = i;
          sig[i+1] = reali*alpha[i]*sig[i];
        };
      };
      psi[k] = temp1;
        
      // Compute coefficients g[*]; initialize v[*] and set w[*]. 
      if (ns>1) {
        // If order was raised, update diagonal part of v[*] 
        if (k>kold) {
          temp4 = k*kp1;
          v[k] = 1.0/temp4;
          nsm2 = ns-2;
          for (j=1;j<=nsm2;j++) {
            i = k-j;
            v[i] = v[i] - alpha[j+1]*v[i+1];
          };
        };
        // Update V[*] and set W[*] 
        limit1 = kp1 - ns;
        temp5  = alpha[ns];
        for (iq=1;iq<=limit1;iq++) {
          v[iq] = v[iq] - temp5*v[iq+1];
          w[iq] = v[iq];
        };
        
        g[nsp1] = w[1];
      }
      else {
        for (iq=1;iq<=k;iq++) {
          temp3 = iq*(iq+1);
          v[iq] = 1.0/temp3;
          w[iq] = v[iq];
        };
      };
  
      // Compute the g[*] in the work vector w[*] 
      nsp2 = ns + 2;
      if (kp1>=nsp2) {
        for (i=nsp2;i<=kp1;i++) {
          limit2 = kp2 - i;
          temp6  = alpha[i-1];
          for (iq=1;iq<=limit2;iq++) w[iq] = w[iq] - temp6*w[iq+1];
          g[i] = w[1];
        };
      };
  
    }; // if K>=NS  
  
    //                                                                 
    // End block 1                                                     
    //                                                                 
  
  
    //                                                                 
    // Begin block 2                                                   
    //                                                                 
    // Predict a solution p[*], evaluate derivatives using predicted   
    // solution, estimate local error at order k and errors at orders  
    // k, k-1, k-2 as if constant step size were used.                 
    //                                                                 
  
    // Change phi to phi star 
    if (k>=nsp1) {
      for (i=nsp1;i<=k;i++) {
        temp1 = beta[i];
        for (l=0;l<n_eqn;l++) phi(l,i) = temp1 * phi(l,i);
      }
    };

    // Predict solution and differences 
    for (l=0;l<n_eqn;l++) {
      phi(l,kp2) = phi(l,kp1);
      phi(l,kp1) = 0.0;
      p(l)       = 0.0;
    };
    for (j=1;j<=k;j++) {
      i     = kp1 - j;
      ip1   = i+1;
      temp2 = g[i];
      for (l=0; l<n_eqn;l++) {
        p(l)     = p(l) + temp2*phi(l,i);
        phi(l,i) = phi(l,i) + phi(l,ip1);
      };
    };
    if (nornd) {
      p = y + h*p;
    }
    else {
      for (l=0;l<n_eqn;l++) {
        tau = h*p(l) - phi(l,15);
        p(l) = y(l) + tau;
        phi(l,16) = (p(l) - y(l)) - tau;
      };
    };
    xold = x;
    x = x + h;
    absh = fabs(h);
    f(x, p, yp, pAux);
  
    // Estimate errors at orders k, k-1, k-2 
    erkm2 = 0.0;
    erkm1 = 0.0;
    erk = 0.0;
    
    for (l=0;l<n_eqn;l++) {
      temp3 = 1.0/wt(l);
      temp4 = yp(l) - phi(l,1);
      if (km2> 0) erkm2 = erkm2 + ((phi(l,km1)+temp4)*temp3)
                                 *((phi(l,km1)+temp4)*temp3);
      if (km2>=0) erkm1 = erkm1 + ((phi(l,k)+temp4)*temp3)
                                 *((phi(l,k)+temp4)*temp3);
      erk = erk + (temp4*temp3)*(temp4*temp3);
    };
    
    if (km2> 0)  erkm2 = absh*sig[km1]*gstr[km2]*sqrt(erkm2);
    if (km2>=0)  erkm1 = absh*sig[k]*gstr[km1]*sqrt(erkm1);
    
    temp5 = absh*sqrt(erk);
    err = temp5*(g[k]-g[kp1]);
    erk = temp5*sig[kp1]*gstr[k];
    knew = k;
  
    // Test if order should be lowered 
    if (km2 >0) if (max(erkm1,erkm2)<=erk) knew=km1;
    if (km2==0) if (erkm1<=0.5*erk) knew=km1;
  
    //                                                                 
    // End block 2                                                     
    //                                                                 
  

    //                                                                 
    // If step is successful continue with block 4, otherwise repeat   
    // blocks 1 and 2 after executing block 3                          
    //                                                                 
  
    success = (err<=eps);
    
    if (!success) {

      //                                                             
      // Begin block 3                                               
      //                                                             

      // The step is unsuccessful. Restore x, phi[*,*], psi[*]. If   
      // 3rd consecutive failure, set order to 1. If step fails more 
      // than 3 times, consider an optimal step size. Double error   
      // tolerance and return if estimated step size is too small    
      // for machine precision.                                      
      //                                                             
      
      // Restore x, phi[*,*] and psi[*] 
      phase1 = false; 
      x = xold;
      for (i=1;i<=k;i++) {
        temp1 = 1.0/beta[i];
        ip1 = i+1;
        for (l=0;l<n_eqn;l++) phi(l,i)=temp1*(phi(l,i)-phi(l,ip1));
      };
      
      if (k>=2)  
        for (i=2;i<=k;i++) psi[i-1] = psi[i] - h;

      // On third failure, set order to one. 
      // Thereafter, use optimal step size   
      ifail++;
      temp2 = 0.5;
      if (ifail>3) 
        if (p5eps < 0.25*erk) temp2 = sqrt(p5eps/erk);
      if (ifail>=3) knew = 1;
      h = temp2*h;
      k = knew;
      if (fabs(h)<fouru*fabs(x)) {
        crash = true;
        h = sign(fouru*fabs(x), h);
        eps *= 2.0;
        return;                     // Exit 
      };
      
      //                                                             
      // End block 3, return to start of block 1                     
      //                                                             
  
    };  // end if(success) 
  
  }
  while (!success);


  //                                                                   
  // Begin block 4                                                     
  //                                                                   
  // The step is successful. Correct the predicted solution, evaluate  
  // the derivatives using the corrected solution and update the       
  // differences. Determine best order and step size for next step.    
  //                                                                   

  kold = k;
  hold = h;


  // Correct and evaluate 
  temp1 = h*g[kp1];
  if (nornd) 
    for (l=0;l<n_eqn;l++) y(l) = p(l) + temp1*(yp(l) - phi(l,1));
  else 
    for (l=0;l<n_eqn;l++) {
      rho = temp1*(yp(l) - phi(l,1)) - phi(l,16);
      y(l) = p(l) + rho;
      phi(l,15) = (y(l) - p(l)) - rho;
    };
  
  f(x,y,yp,pAux);

  
  // Update differences for next step 
  for (l=0;l<n_eqn;l++) {
    phi(l,kp1) = yp(l) - phi(l,1);
    phi(l,kp2) = phi(l,kp1) - phi(l,kp2);
  };
  for (i=1;i<=k;i++) 
    for (l=0;l<n_eqn;l++)
      phi(l,i) = phi(l,i) + phi(l,kp1);


  // Estimate error at order k+1 unless               
  // - in first phase when always raise order,        
  // - already decided to lower order,                
  // - step size not constant so estimate unreliable  
  erkp1 = 0.0;
  if ( (knew==km1) || (k==kmax) )  phase1=false;

  if (phase1) {
    k = kp1;
    erk = erkp1;
  }
  else {
    if (knew==km1) {
       // lower order 
       k = km1;
       erk = erkm1;
    }
    else {
       
      if (kp1<=ns) {
        for (l=0;l<n_eqn;l++)
          erkp1 = erkp1 + (phi(l,kp2)/wt(l))*(phi(l,kp2)/wt(l));
        erkp1 = absh*gstr[kp1]*sqrt(erkp1);

        // Using estimated error at order k+1, determine 
        // appropriate order for next step               
        if (k>1) {
          if ( erkm1<=min(erk,erkp1)) {
            // lower order
            k=km1; erk=erkm1;
          }
          else {
            if ( (erkp1<erk) && (k!=kmax) ) {
               // raise order 
               k=kp1; erk=erkp1;
            };
          };
        }
        else {
          if (erkp1<0.5*erk) {
            // raise order 
            // Here erkp1 < erk < max(erkm1,ermk2) else    
            // order would have been lowered in block 2.   
            // Thus order is to be raised                  
            k = kp1;
            erk = erkp1;
          };
        };

      }; // end if kp1<=ns 

    }; // end if knew!=km1 

  }; // end if !phase1 
  

  // With new order determine appropriate step size for next step 
  if ( phase1 || (p5eps>=erk*two[k+1]) ) 
    hnew = 2.0*h;
  else {
    if (p5eps<erk) {
      temp2 = k+1;
      r = pow(p5eps/erk, 1.0/temp2);
      hnew = absh*max(0.5, min(0.9,r));
      hnew = sign(max(hnew, fouru*fabs(x)), h);
    }
    else hnew = h;
  };
  
  h = hnew;

  //                                                                   
  // End block 4                                                       
  //                                                                   
};


//
// Interpolation
//

void DE::Intrp ( double xout, Vector& yout, Vector& ypout )
{

  // Variables

  int     i, j, ki;
  double  eta, gamma, hi, psijm1;
  double  temp1, term;
  double  g[14], rho[14], w[14];


  g[1]   = 1.0;
  rho[1] = 1.0;

  hi = xout - x;
  ki = kold + 1;

  // Initialize w[*] for computing g[*] 
  for (i=1;i<=ki;i++) {
    temp1 = i;
    w[i] = 1.0/temp1;
  }

  // Compute g[*] 
  term = 0.0;
  for (j=2;j<=ki;j++) {
    psijm1 = psi[j-1];
    gamma = (hi + term)/psijm1;
    eta = hi/psijm1;
    for (i=1;i<=ki+1-j;i++) w[i] = gamma*w[i] - eta*w[i+1];
    g[j] = w[1];
    rho[j] = gamma*rho[j-1];
    term = psijm1;
  };

  // Interpolate for the solution yout and for 
  // the derivative of the solution ypout      
  ypout = 0.0;
  yout  = 0.0;
  for (j=1;j<=ki;j++){
    i = ki+1-j;
    yout  = yout  + g[i]  *phi.Col(i);
    ypout = ypout + rho[i]*phi.Col(i);
  };
  yout = yy + hi*yout; 

};


//
// DE integration
// (with full control of warnings and errros status codes)
//

void DE::Integ_ ( 
           double&    t,          // Value of independent variable
           double     tout,       // Desired output point
           Vector&    y           // Solution vector
         )
{

  // Variables

  bool    stiff, crash;           // Flags
  int     nostep;                 // Step count
  int     kle4 = 0;
  double  releps, abseps, tend;
  double  absdel, del, eps;

   
  // Return, if output time equals input time

  if (t==tout) return;    // No integration


  // Test for improper parameters

  eps   = max(relerr,abserr);

  if ( ( relerr <  0.0         ) ||      // Negative relative error bound
       ( abserr <  0.0         ) ||      // Negative absolute error bound
       ( eps    <= 0.0         ) ||      // Both error bounds are non-positive
       ( State  >  DE_INVPARAM ) ||      // Invalid status flag
       ( (State != DE_INIT) && 
         (t != told)           ) ) 
  {
    State = DE_INVPARAM;                 // Set error code
    return;                              // Exit
  };


  // On each call set interval of integration and counter for
  // number of steps. Adjust input error tolerances to define
  // weight vector for subroutine STEP.

  del    = tout - t;
  absdel = fabs(del);

  tend   = t + 100.0*del;
  if (!PermitTOUT) tend = tout;
    
  nostep = 0;
  kle4   = 0;
  stiff  = false;
  releps = relerr/eps;
  abseps = abserr/eps;
    
  if  ( (State==DE_INIT) || (!OldPermit) || (delsgn*del<=0.0) ) {
    // On start and restart also set the work variables x and yy(*),
    // store the direction of integration and initialize the step size
    start  = true;
    x      = t;
    yy     = y;
    delsgn = sign(1.0, del);
    h      = sign( max(fouru*fabs(x), fabs(tout-x)), tout-x );
  }

  while (true) {  // Start step loop
  
    // If already past output point, interpolate solution and return
    if (fabs(x-t) >= absdel) {
      Intrp (tout, y, ypout);
      State     = DE_DONE;          // Set return code
      t         = tout;             // Set independent variable
      told      = t;                // Store independent variable
      OldPermit = PermitTOUT;
      return;                       // Normal exit
    };                         

    // If cannot go past output point and sufficiently close,
    // extrapolate and return
    if ( !PermitTOUT && ( fabs(tout-x) < fouru*fabs(x) ) ) {
      h = tout - x;
      f(x,yy,yp,pAux);              // Compute derivative yp(x)
      y = yy + h*yp;                // Extrapolate vector from x to tout
      State     = DE_DONE;          // Set return code
      t         = tout;             // Set independent variable
      told      = t;                // Store independent variable
      OldPermit = PermitTOUT;
      return;                       // Normal exit
    };

    // Test for too much work
    if (nostep >= maxnum) {
      State = DE_NUMSTEPS;          // Too many steps
      if (stiff) State = DE_STIFF;  // Stiffness suspected
      y         = yy;               // Copy last step
      t         = x;
      told      = t;
      OldPermit = true;
      return;                       // Weak failure exit
    };

    // Limit step size, set weight vector and take a step
    h  = sign(min(fabs(h), fabs(tend-x)), h);
    for (int l=0; l<n_eqn; l++) 
      wt(l) = releps*fabs(yy(l)) + abseps;

    Step ( x, yy, eps, crash );


    // Test for too small tolerances
    if (crash) {
      State     = DE_BADACC;
      relerr    = eps*releps;       // Modify relative and absolute
      abserr    = eps*abseps;       // accuracy requirements
      y         = yy;               // Copy last step
      t         = x;
      told      = t;
      OldPermit = true;
      return;                       // Weak failure exit
    }

    nostep++;  // Count total number of steps

    // Count number of consecutive steps taken with the order of
    // the method being less or equal to four and test for stiffness
    kle4++;
    if (kold>  4) kle4=0;
    if (kle4>=50) stiff=true;

  } // End step loop 


};


//
// Initialization
//

void DE::Init ( 
           double     t0,         // Initial value of the independent variable
           double     rel,        // Relative accuracy requirement
           double     abs,        // Absolute accuracy requirement
           int        maxord      // Maximum order
          )
{
  t      = t0;
  relerr = rel;
  abserr = abs;
  kmax   = maxord; if (kmax>12) kmax=12;
  State  = DE_INIT;
};


//
// DE integration with simplified state code handling
// (skips over warnings, aborts in case of error)
//

void DE::Integ ( 
           double     tout,       // Desired output point
           Vector&    y           // Solution vector
         )
{
  do {
    Integ_ (t,tout,y);
    if ( State==DE_INVPARAM ) { 
      std::cerr << "ERROR: invalid parameters in DE::Integ" 
                << std::endl; exit(1); 
    }
    if ( State==DE_BADACC ) { 
      std::cerr << "WARNING: Accuracy requirement not achieved in DE::Integ" 
                << std::endl;
    }
    if ( State==DE_STIFF ) { 
      std::cerr << "WARNING: Stiff problem suspected in DE::Integ" 
                << std::endl;
    }
  }
  while ( State > DE_DONE );
};


//
// Interpolation
//

void DE::Intrp ( 
      double     tout,           // Desired output point
      Vector&    y               // Solution vector
    )
{
   Intrp ( tout, y, ypout );     // Interpolate and discard interpolated
                                 // derivative ypout
};

