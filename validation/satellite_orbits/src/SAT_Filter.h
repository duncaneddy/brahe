//------------------------------------------------------------------------------
//
// SAT_Filter.h
// 
// Purpose:
//
//    Batch least squares and Kalman filtering
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

#ifndef INC_SAT_FILTER_H
#define INC_SAT_FILTER_H


#include "SAT_VecMat.h"


//------------------------------------------------------------------------------
//
// InvUpper
//
// Purpose:
//
//   Inversion of an upper right triangular matrix
//
// Input/Output:
//
//   R    Upper triangular square matrix
//   T    Inverse of R 
//
// Note:
//
//   This function may be called with the same actual parameter for R and T
//
//------------------------------------------------------------------------------

void InvUpper(const Matrix& R, Matrix& T);


//------------------------------------------------------------------------------
//
// LSQ class (specification)
//
//------------------------------------------------------------------------------

// Least squares estimation class 

class LSQ
{
  public:

    // Constructor
    LSQ();         
    LSQ(int nEst); // nEst: Number of estimation parameters

    // Re-dimensioning
    LSQ &resize(int nEst); 
    
    // Number of data equations
    int nData() const { return n; };

    // Reset to solve a new problem
    void Init();  
    void Init(const Vector& x,   // A priori parameters
              const Matrix& P);  // A priori covariance
        
    // Add a data equation of form Ax=b to a least squares system
    // (performs a row-wise QR transformation using Givens rotations)
    void Accumulate (const Vector& A, double b, double sigma = 1.0);
    
    // Solve the LSQ problem for vector x by backsubstitution
    void Solve (Vector& x) const;

    // Covariance matrix
    Matrix Cov();

    // Standard deviation
    Vector StdDev();

    // RMS of normalized post-fit residuals (square-root of loss function)
    double Res();
    
    // Square-root information matrix and data vector
    Matrix SRIM();      // Copy of R
    Vector Data();      // Copy of d

  protected:

    // Elements
    int      N;         // Number of estimation parameters
    int      n;         // Number of data equations 
    double   r_sqr;     // Square sum of post-fit residuals
    Vector   d;         // Right hand side of transformed equations
    Matrix   R;         // Square-root information matrix 
                        // (Upper right triangular matrix)

};


//------------------------------------------------------------------------------
//
// EKF class (specification)
//
//------------------------------------------------------------------------------

// Extended Kalman Filter class 

class EKF
{
  public:

    // Constructor
    
    EKF();         
    EKF(int n_);                  // n: Number of estimation parameters

    // Re-dimensioning
    EKF &resize(int n_); 

    // Initialization of a new problem
    
    void Init(double t_, Vector x_, Matrix P_);  
    void Init(double t_, Vector x_, Vector sigma);  
    
    // Update of filter parameters

    void TimeUpdate(double        t_,    // New epoch
                    const Vector& x_,    // Propagetd state 
                    const Matrix& Phi);  // State transition matrix 
    void TimeUpdate(double        t_,    // New epoch
                    const Vector& x_,    // Propagated state 
                    const Matrix& Phi,   // State transition matrix 
                    const Matrix& Qdt);  // Accumulated process noise

    void MeasUpdate(double        z,     // Measurement at new epoch
                    double        g,     // Modelled measurement
                    double        sigma, // Standard deviation
                    const Vector& G);    // Partials dg/dx

    void MeasUpdate(const Vector& z,     // Measurement at new epoch
                    const Vector& g,     // Modelled measurement
                    const Vector& s,     // Standard deviation
                    const Matrix& G);    // Partials dg/dx
    
    void MeasUpdate(const Vector& z,     // Measurement at new epoch
                    const Vector& g,     // Modelled measurement
                    const Matrix& Inv_W, // Measurement covariance
                    const Matrix& G);    // Partials dg/dx

    // Access to filter parameters
    
    double   Time();                     // Time    
    Vector   State();                    // State parameters
    Matrix   Cov();                      // Covariance matrix
    Vector   StdDev();                   // Standard deviation

  protected:

    // Elements
    int      n;                          // Number of state parameters
    double   t;                          // Time 
    Vector   x;                          // State parameters
    Matrix   P;                          // Covariance matrix
    
};

#endif   // include blocker
