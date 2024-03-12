//------------------------------------------------------------------------------
//
// validate_harris_priester_density.cpp
//
// Purpose:
//
//   Exercise the reference implementation of the Harris-Priester density model
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
#include "SAT_RefSys.h"

using namespace std;


//------------------------------------------------------------------------------
//
// Main program
//
//------------------------------------------------------------------------------

int main() {

    // Constants

    const double    MJD_TT = 60310.0; // 2024-01-01

    // Variables

    double    lon, lat, h, rho;
    Vector    r_Sun(3), r(3);

    // Header

    cout << "Validate Harris-Priester Density" << endl << endl;

    cout << "(rs_x,rs_y,rs_z,r_x,r_y,r_z,rho)" << endl << endl;

    for (lon=0.0;lon<360.0;lon+=45.0) {
        for (lat=-90.0;lat<90.0;lat+=45.0) {
            for (h=110.0;h<1000.0;h+=100.0) {

                // Compute Sun Position
                r_Sun = Sun(MJD_TT);

                // Compute Position
                r = Geodetic(lon*Rad, lat*Rad, h*1.0e+3).Position();

                // Compute Density
                rho = Density_HP(MJD_TT, r);

                // Print test case
                cout << "(" << fixed << setprecision(3) << r_Sun(0) << "," << r_Sun(1) << "," << r_Sun(2) << "," << r(0) << "," << r(1) << "," << r(2) << "," << defaultfloat << setprecision(6) << rho << ")" << endl;
            };
        };
    };

    return 0;

}
