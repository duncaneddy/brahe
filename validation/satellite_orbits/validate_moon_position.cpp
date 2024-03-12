//------------------------------------------------------------------------------
//
// validate_sun_position.cpp
//
// Purpose:
//
//   Exercise the reference implementation of Moon position function to validate
//   the Rust implementaiton
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

  const double MJD_TT_BASE = 60310; // 2024-01-01

  // Variables

  int       n;
  double    mjd_tt;
  Vector    p(3);

  // Header

  cout << "Validate Moon Position" << endl << endl;

  cout << "(mjd_tt,p_x,p_y,p_z)" << endl << endl;

  for (n=0;n<=24;n+=1) {

    // Evaluate Sun Position
    mjd_tt = MJD_TT_BASE + n/24.0;
    p = Moon(mjd_tt);
    cout << "(" << mjd_tt << "," << setprecision(15) << p(0) << "," << p(1) << "," << p(2) << ")" << endl;
  };

  return 0;

}
