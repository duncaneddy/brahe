//------------------------------------------------------------------------------
//
// SAT_Time.cpp
// 
// Purpose:
//
//    Time and date computation
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
#include <iomanip>

#include "SAT_Time.h"

using namespace std;


//------------------------------------------------------------------------------
//
// Mjd
//
// Purpose:
//
//   Modified Julian Date from calendar date and time
//
// Input/output:
//
//   Year      Calendar date components
//   Month
//   Day
//   Hour      Time components (optional)
//   Min
//   Sec
//   <return>  Modified Julian Date
//
//------------------------------------------------------------------------------

double Mjd ( int Year, int Month, int Day, int Hour, int Min, double Sec )
{
  // Variables

  long    MjdMidnight;
  double  FracOfDay;
  int     b;


  if (Month<=2) { Month+=12; --Year;}
  
  if ( (10000L*Year+100L*Month+Day) <= 15821004L )
    b = -2 + ((Year+4716)/4) - 1179;     // Julian calendar 
  else
    b = (Year/400)-(Year/100)+(Year/4);  // Gregorian calendar 
    
  MjdMidnight = 365L*Year - 679004L + b + int(30.6001*(Month+1)) + Day;
  FracOfDay   = (Hour+Min/60.0+Sec/3600.0) / 24.0; 

  return MjdMidnight + FracOfDay;
}


//------------------------------------------------------------------------------
//
// CalDat
//
// Purpose:
//
//   Calendar date and time from Modified Julian Date
//
// Input/output:
//
//   Mjd       Modified Julian Date
//   Year      Calendar date components
//   Month
//   Day
//   Hour      Time components
//   Min
//   Sec
//
//------------------------------------------------------------------------------

void CalDat ( double Mjd, 
              int& Year, int& Month, int& Day,  
              int& Hour, int& Min, double& Sec )
{
  // Variables
  long    a,b,c,d,e,f;
  double  Hours,x;

  // Convert Julian day number to calendar date
  a = long(Mjd+2400001.0);

  if ( a < 2299161 ) {  // Julian calendar
    b = 0;
    c = a + 1524;
  }
  else {                // Gregorian calendar
    b = long((a-1867216.25)/36524.25);
    c = a +  b - (b/4) + 1525;
  }

  d     = long ( (c-122.1)/365.25 );
  e     = 365*d + d/4;
  f     = long ( (c-e)/30.6001 );

  Day   = c - e - int(30.6001*f);
  Month = f - 1 - 12*(f/14);
  Year  = d - 4715 - ((7+Month)/10);

  Hours = 24.0*(Mjd-floor(Mjd));

  Hour = int(Hours);
  x = (Hours-Hour)*60.0; Min = int(x);  Sec = (x-Min)*60.0;

}


//------------------------------------------------------------------------------
//
// Date (class implementation)
//
// Purpose:
//
//   Auxiliary class for date and time output
//
//------------------------------------------------------------------------------

// Constructor

Date::Date (double Mjd)
 : mjd(Mjd)
{
}

// Output operator

ostream& operator << (ostream& os, const Date& D)
{
  // Constants

  const double mSecs = 86400.0e3;   // Milliseconds per day
  const double eps  = 0.1/mSecs;    // 0.1 msec
  
  // Variables
  double MjdRound;
  int    Year, Month, Day;
  int    H, M;
  double S; 

  // Round to 1 msec

  MjdRound = (floor(mSecs*D.mjd+0.5)/mSecs)+eps;

  CalDat (MjdRound, Year, Month, Day, H, M, S);
      
  os << setfill('0') 
     << setw(4) << Year  << "/" 
     << setw(2) << Month << "/" 
     << setw(2) << Day   << "  "; 
  os << setw(2) << H << ":" 
     << setw(2) << M << ":" 
     << fixed << setprecision(3)
     << setw(6) << S
     << setfill(' ');
   
  return os;
}
