//------------------------------------------------------------------------------
//
// validate_sun_position.cpp
//
// Purpose:
//
//   Exercise the reference implementation of Sun position and third-body
//   acceleration to validate the Rust implementation.
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

    const double MJD_TT = 60310.0; // 2024-01-01
    const double sma = R_Earth+500e3;
    const double e = 0.01;
    const double i = 23.0*Rad;
    const double Omega = 15.0*Rad;
    const double omega = 30.0*Rad;

    // Variables

    int       n;
    double    M;
    Vector    r_Sun(3), a(3), x(6);

    // Header

    cout << "Validate Third-Body Sun Acceleration" << endl << endl;

    cout << "(mjd_tt,r_x,r_y,r_z,a_x,a_y,a_z)" << endl << endl;

    for (n=0;n<16;n+=1) {

        // Compute Satellite Position
        M = n*2.0*pi/16.0;
        x = State(GM_Earth, Vector(sma, e, i, Omega, omega, M), 0.0 );


        // Evaluate Sun Position
        r_Sun = Sun(MJD_TT);

        // Compute third-body acceleration
        a = AccelPointMass (Vector(x(0), x(1), x(2)), r_Sun,  GM_Sun );
        cout << "(" << MJD_TT << "," << setprecision(15) << x(0) << "," << x(1) << "," << x(2) << "," << a(0) << "," << a(1) << "," << a(2) << ")" << endl;
    };

    return 0;

}
