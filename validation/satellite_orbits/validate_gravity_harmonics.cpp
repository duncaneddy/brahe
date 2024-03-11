//------------------------------------------------------------------------------
//
// validate_gravity_harmonics.cpp
// 
// Purpose: 
//
//   Exercise the reference implementaiton of spherical harmonic gravity
//   field acceleration to provide reference values for the validation of
//   the Rust implementation.
//
//------------------------------------------------------------------------------

#include <ctime>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "SAT_Const.h"
#include "SAT_Force.h"
#include "SAT_VecMat.h"

using namespace std;


//------------------------------------------------------------------------------
//
// Main program
//
//------------------------------------------------------------------------------

int main() {

  // Constants
  
  const int    n_max  =     20;

  const Vector r(6525.919e3, 1710.416e3, 2508.886e3);  // Position [m]

  // Variables

  int       i,n;                 // Loop counters
  clock_t   start,end;           // Processor time at start and end 
  double    duration;  
  Vector    a(3);

  // Header 
  
  cout << "Validate Gravity Harmonics" << endl << endl;

  cout << "(degree,order,a_x,a_y,a_z)" << endl << endl;

  // Outer loop [2,4,...,n_max]

  for (n=2;n<=n_max;n+=1) {

    // Evaluate gravitational acceleration N_Step times
    a = AccelHarmonic (r,Id(3),Grav.GM,Grav.R_ref,Grav.CS,n,n);
    cout << "(" << n << "," << n << "," << setprecision(12) << a(0) << "," << a(1) << "," << a(2) << ")" << endl;
  };

  return 0;

}
