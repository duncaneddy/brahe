//------------------------------------------------------------------------------
//
// SAT_DE.h
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

#ifndef INC_SAT_DE_H
#define INC_SAT_DE_H

#include "SAT_VecMat.h"


//------------------------------------------------------------------------------
//
// RK4 class (specification)
//
//------------------------------------------------------------------------------

// Function prototype for first order differential equations
// void f (double x, const Vector& y, Vector& yp )

typedef void (*RK4funct)(
  double        x,     // Independent variable
  const Vector& y,     // State vector 
  Vector&       yp,    // Derivative y'=f(x,y)
  void*         pAux   // Pointer to auxiliary data used within f
);


// RK4 class specification

class RK4
{
  public:

    // Constructor
    RK4 (
      RK4funct f_,        // Differential equation
      int      n_eqn_,    // Dimension
      void*    pAux_      // Pointer to auxiliary data
      ) 
    : f(f_), n_eqn(n_eqn_), pAux(pAux_)
    { k_4=k_3=k_2=k_1=Vector(n_eqn_); };
    
    // Integration step
    void Step (         
      double&  t,         // Value of the independent variable; updated by t+h
      Vector&  y,         // Value of y(t); updated by y(t+h)
      double   h          // Step size
    );

  private:

    // Elements
    RK4funct  f;
    int       n_eqn;
    void*     pAux;
    Vector    k_1,k_2,k_3,k_4;

};


//------------------------------------------------------------------------------
//
// DE class (specification)
//
//------------------------------------------------------------------------------

// Function prototype for first order differential equations
// void f (double x, const Vector y, Vector yp[])

typedef void (*DEfunct)(
  double        x,     // Independent variable
  const Vector& y,     // State vector 
  Vector&       yp,    // Derivative y'=f(x,y)
  void*         pAux   // Pointer to auxiliary data used within f
);


// State codes 

enum DE_STATE {
  DE_INIT     = 1,   // Restart integration
  DE_DONE     = 2,   // Successful step
  DE_BADACC   = 3,   // Accuracy requirement could not be achieved
  DE_NUMSTEPS = 4,   // Permitted number of steps exceeded
  DE_STIFF    = 5,   // Stiff problem suspected
  DE_INVPARAM = 6    // Invalid input parameters
};



// DE integrator class specification

class DE
{
  public:

    // Elements
    
    bool     PermitTOUT;      // Flag for integrating past tout
                              // (default = true)
    DE_STATE State;           // State code (default = DE_INIT)
    int      kmax;            // Maximum order
    double   relerr;          // Desired relative accuracy of the solution
    double   abserr;          // Desired absolute accuracy of the solution
    double   t;               // Value of independent variable

  public:

    // Constructor
    DE ();                    // Default constructor
    DE (
      DEfunct    f_,          // Differential equation
      int        n_eqn_,      // Dimension
      void*      pAux_        // Pointer to auxiliary data
    );
    
    // Definition
    void Define (
      DEfunct    f_,          // Differential equation
      int        n_eqn_,      // Dimension
      void*      pAux_        // Pointer to auxiliary data
    );

    // Initialization
    void Init ( 
      double     t0,          // Initial value of the independent variable
      double     rel,         // Relative accuracy requirement
      double     abs,         // Absolute accuracy requirement
      int        maxord=12    // Maximum order
    );

    // Integration (skips over warnings, aborts in case of error)
    void Integ ( 
      double     tout,        // Desired output point
      Vector&    y            // Solution vector
    );

    // Interpolation
    void Intrp ( 
      double     tout,        // Desired output point
      Vector&    y            // Solution vector
    );

  private:

    // Elements

    DEfunct  f;
    int      n_eqn;
    void*    pAux;
    Vector   yy,wt,p,yp,ypout;
    Matrix   phi;
    double   alpha[13],beta[13],v[13],w[13],psi[13];
    double   sig[14],g[14];
    double   x,h,hold,told,delsgn;
    int      ns,k,kold;
    bool     OldPermit, phase1,start,nornd;   
    bool     init;

    // Elementary integration step
    void Step (double& x, Vector& y, double& eps, bool& crash);

    // Interpolation
    void Intrp ( double xout, Vector& yout, Vector& ypout );

    // Integration (with full control of warnings and error status codes)
    void Integ_ ( 
      double&    t,           // Initial value of the independent variable
      double     tout,        // Desired output point
      Vector&    y            // Solution vector
    );
};

#endif   // include blocker
