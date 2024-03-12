//------------------------------------------------------------------------------
//
// validate_drag.cpp
//
// Purpose:
//
//   Exercise the reference implementation of the atmospheric drag perturbation
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

        const int       steps = 32;
        const double    MJD_TT = 60310.0; // 2024-01-01
        const double    sma = R_Earth+500e3;
        const double    e = 0.01;
        const double    i = 23.0*Rad;
        const double    Omega = 15.0*Rad;
        const double    omega = 30.0*Rad;
        const double    area = 1.0;
        const double    Cd = 2.3;
        const double    mass = 100.0;

        // Variables

        int       n;
        double    M;
        double    illumination;
        Vector    r_Sun(3), a(3), x(6), r(3), v(3);

        // Header

        cout << "Validate Atmospheric Drag Acceleration" << endl << endl;

        cout << "(mjd_tt,area,mass,Cd,r_x,r_y,r_z,a_x,a_y,a_z)" << endl << endl;

        for (n=0;n<steps;n+=1) {

            // Compute Satellite Position
            M = n*2.0*pi/steps;
            x = State(GM_Earth, Vector(sma, e, i, Omega, omega, M), 0.0 );
            r = Vector(x(0), x(1), x(2));
            v = Vector(x(3), x(4), x(5));

            // Compute Drag Acceleration
            a = AccelDrag(MJD_TT, r, v, Id(3), area, mass, Cd);

            // Print test case
            cout << "(" << fixed << setprecision(3) << MJD_TT << "," << area << "," << mass << "," << Cd << scientific << setprecision(12) << "," << x(0) << "," << x(1) << "," << x(2) << "," << x(3) << "," << x(4) << "," << x(5) << "," << a(0) << "," << a(1) << "," << a(2) << ")" << endl;
        };

        return 0;

}
