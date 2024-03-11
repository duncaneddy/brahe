//------------------------------------------------------------------------------
//
// Exercise_3_1.cpp
// 
// Purpose: 
//
//   Satellite Orbits - Models, Methods, and Applications
//   Exercise 3-1: Gravity field
//
// Hint:
//
//   The number of evaluations (N_Step) should be suitably adjusted for 
//   different platforms. The numerical results may differ, depending 
//   on the specific run-time environment.
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
//   2000/03/04  EGO  Final version (1st edition)
//   2012/07/01  OMO  Final version (3rd reprint)
//
// (c) 1999-2012  O. Montenbruck, E. Gill
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
  
  const int    N_Step =  10000;  // Recommended for 0.01 sec timer (Linux)
  const int    n_max  =     20;

  const Vector r(6525.919e3, 1710.416e3, 2508.886e3);  // Position [m]

  // Variables

  int       i,n;                 // Loop counters
  clock_t   start,end;           // Processor time at start and end 
  double    duration;  
  Vector    a(3);

  // Header 
  
  cout << "Exercise 3-1: Gravity Field Computation " << endl << endl;

  cout << " Order   CPU Time [s]" << endl << endl; 

  // Outer loop [2,4,...,n_max]

  for (n=2;n<=n_max;n+=2) {

    // Start timing
    start = clock();

    // Evaluate gravitational acceleration N_Step times
    for (i=0;i<=N_Step;i++) 
      a = AccelHarmonic (r,Id(3),Grav.GM,Grav.R_ref,Grav.CS,n,n);

    // Stop CPU time measurement
    end = clock();

    duration = (double)(end-start) / (double)(CLOCKS_PER_SEC);

    cout << setw(4) << n 
         << setprecision(2) << fixed << setw(13) << duration << endl; 

  };

  return 0;

}
