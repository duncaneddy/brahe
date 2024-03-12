//------------------------------------------------------------------------------
//
// validate_eclipse_cylindrical.cpp
//
// Purpose:
//
//   Exercise the reference implementation of cylindrical eclipse models.
//
//------------------------------------------------------------------------------

#include <ctime>
#include <iostream>
#include <iomanip>
#include <cmath>

#include "SAT_Const.h"
#include "SAT_Force.h"
#include "SAT_Kepler.h"
#include "SAT_VecMat.h"

using namespace std;


//------------------------------------------------------------------------------
//
// Main program
//
//------------------------------------------------------------------------------

int main() {

    // Constants

    const int       steps = 360;
    const double    MJD_TT = 60310.0; // 2024-01-01
    const double    sma = R_Earth+500e3;
    const double    e = 0.01;
    const double    i = 23.0*Rad;
    const double    Omega = 15.0*Rad;
    const double    omega = 30.0*Rad;
    const double    area = 1.0;
    const double    CR = 1.8;
    const double    mass = 100.0;

    // Variables

    int       n;
    double    M;
    double    illumination;
    Vector    r_Sun(3), a(3), x(6), r(3);

    // Header

    cout << "Validate Third-Body Sun Acceleration" << endl << endl;

    cout << "(mjd_tt,r_x,r_y,r_z,illumination)" << endl << endl;

    for (n=0;n<steps;n+=1) {

        // Compute Satellite Position
        M = n*2.0*pi/steps;
        x = State(GM_Earth, Vector(sma, e, i, Omega, omega, M), 0.0 );
        r = Vector(x(0), x(1), x(2));

        // Evaluate Sun Position
        r_Sun = Sun(MJD_TT);

        // Compute Illumination
        illumination = Illumination (r, r_Sun);

        // Print test case
        cout << "(" << MJD_TT << "," << setprecision(15) << x(0) << "," << x(1) << "," << x(2) << "," << illumination << ")" << endl;
    };

    return 0;

}
