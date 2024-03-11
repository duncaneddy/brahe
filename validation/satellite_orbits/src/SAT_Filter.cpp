//------------------------------------------------------------------------------
//
// SAT_Filter.cpp
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


#include <cmath>
#include <cstdlib>
#include <iostream>

#include "SAT_Filter.h"
#include "SAT_VecMat.h"


using namespace std;


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

void InvUpper(const Matrix& R, Matrix& T)
{
  
  const int N = R.size1();   // Dimension of R and T
  
  int    i,j,k;
  double Sum;

  if ( R.size2()!=N || T.size1()!=N || T.size2()!=N ) {
    cerr << " ERROR: Incompatible shapes in InvUpper" << endl;
    exit(1);
  }

  // Check diagonal elements

  for (i=0; i<N; i++)
    if ( R(i,i) == 0.0 ) {
      cerr << " ERROR: Singular matrix in InvUpper" << endl;
      exit(1);
    }
    else {
      // Compute the inverse of i-th diagonal element.
      T(i,i) = 1.0/R(i,i);
    };
 
  // Calculate the inverse T = R^(-1)

  for (i=0; i<N-1; i++) 
    for (j=i+1; j<N; j++) {
      Sum = 0.0; for (k=i; k<=j-1; k++)  Sum += T(i,k)*R(k,j);
      T(i,j) = -T(j,j)*Sum;
    };


}



//------------------------------------------------------------------------------
//
// LSQ class (implementation)
//
//------------------------------------------------------------------------------


//
// Constructor; n: Number of estimation parameters
//

LSQ::LSQ () 
  : N(0), n(0), r_sqr(0.0)
{
}

LSQ::LSQ (int nEst) 
  : N(nEst), n(0), r_sqr(0.0)
{
  // Allocate storage for R and d and initialize to zero
  d = Vector(N);
  R = Matrix(N,N);
}

//
// Re-dimensioning
//

LSQ &LSQ::resize(int nEst) {
  // Allocate storage for R and d and initialize to zero
  N     = nEst;
  n     = 0;
  r_sqr = 0.0;       
  d     = Vector(N);
  R     = Matrix(N,N);
  return (*this);
}; 

//
// Initialize new problem
//

// Initialization without apriori information

void LSQ::Init()
{
  // Reset number of data equations
  n     = 0;
  // Reset all elements of R and d
  R     = 0.0;
  d     = 0.0;
  r_sqr = 0.0;
}

// Initialization with apriori information

void LSQ::Init(const Vector& x,   // a priori parameters
               const Matrix& P)   // a priori covariance
{
  
  // Variables

  int     i,j,k;
  double  Sum;

  
  // Reset number of data equations
  
  n = 0;
  
  // Start the factorization of matrix P. Compute upper triangular 
  // factor R of P, where P = (R)*(R^T). Proceed backward column 
  // by column.
      
  for (j=N-1;j>=0;j--) {

    // Compute j-th diagonal element.

    Sum = 0.0;
    for (k=j+1;k<=N-1;k++) Sum += R(j,k)*R(j,k);
    R(j,j) = sqrt(P(j,j)-Sum);

    // Complete factorization of j-th column.

    for (i=j-1;i>=0;i--) {
      Sum = 0.0;
      for (k=j+1;k<=N-1;k++) Sum += R(i,k)*R(j,k);
      R(i,j) = (P(i,j)-Sum)/R(j,j);
    };

  };

  // Replace R by its inverse R^(-1)
  
  InvUpper(R,R);

  // Initialize right hand side

  d = R*x;

}


//
//  Add a data equation of form Ax=b to the least squares system
//  (performs a row-wise QR transformation using Givens rotations)
//

void LSQ::Accumulate (const Vector& A, double b, double sigma)
{
  // Variables
  int      i,j;
  double   c,s,h;
  Vector   a(N);

  // Weighting
  a = A/sigma;  // Normalize A 
  b = b/sigma;  // Normalize b

  // Construct and apply Givens plane rotation.
  for (i=0; i<N; i++)
  {
    // Construct the rotation and apply it to
    // eliminate the i-th element of a.
    if ( R(i,i)==0.0 &&  a(i)==0.0 ) {
      c = 1.0; s = 0.0; R(i,i) = 0.0;
    }
    else {
      h = sqrt ( R(i,i)*R(i,i) + a(i)*a(i) );
      if (R(i,i)<0.0) h=-h; 
      c = R(i,i)/h;
      s = a(i)/h;
      R(i,i) = h;
    };

    a(i) = 0.0;

    // Apply the rotation to the remaining elements of a
    for (j=i+1; j<N; j++) {
      h       = +c*R(i,j)+s*a(j);
      a(j)    = -s*R(i,j)+c*a(j);
      R(i,j) = h;
    }

    // Apply the rotation to the i-th element of d
    h    = +c*d(i)+s*b;
    b    = -s*d(i)+c*b;
    d(i) = h;
  }
  
  // Square sum of post-fit residuals
  r_sqr += b*b;
  
  // Increment number of data equations
  n++;

}


//
// Solve the LSQ problem for vector x[] by backsubstitution
//

void LSQ::Solve (Vector& x) const
{
  // Variables
  int i,j; i=j=0;
  double Sum=0.0;

  // Check for singular matrix 
  for (i=0;i<N;i++)
    if ( R(i,i) == 0.0 ) {
      cerr << " ERROR: Singular matrix R in LSQ::Solve()" << endl;
      exit(1);
    };

  //  Solve Rx=d for x_n,...,x_1 by backsubstitution
  x(N-1) = d(N-1) / R(N-1,N-1);
  for (i=N-2;i>=0;i--) {
    Sum = 0.0;
    for (j=i+1;j<N;j++) Sum += R(i,j)*x(j);
    x(i) = ( d(i) - Sum ) / R(i,i);
  };
}


//
// Covariance matrix
//

Matrix LSQ::Cov()
{
  // Variables
  int     i,j,k;
  double  Sum;
  Matrix  T(N,N);

  // Calculate the inverse T = R^(-1)
  
  InvUpper(R,T);

  // Replace T by the covariance matrix C=T*T^t
  
  for (i=0; i<N; i++) 
    for (j=i; j<N; j++) {
      Sum = 0.0; for (k=j; k<N; k++)  Sum += T(i,k)*T(j,k);
      T(i,j) = Sum;
      T(j,i) = Sum;
    };

  // Result
  
  return T;
   
}


//
// Standard deviation
//

Vector LSQ::StdDev() {

  // Variables
  int     i;
  Vector  Sigma(N);
  Matrix  C(N,N);

  // Covariance
  C = Cov();

  // Standard deviation
  for (i=0; i<N; i++) Sigma(i)=sqrt(C(i,i));

  return Sigma;

}

//
// RMS of normalized post-fit residuals (square-root of loss function)
//

double LSQ::Res() {
  return (n==0? 0.0 : sqrt(r_sqr/n) );
};

//
// Square-root information matrix and data vector
//

Matrix LSQ::SRIM() { return R; };      // Copy of R
Vector LSQ::Data() { return d; };      // Copy of d



//------------------------------------------------------------------------------
//
// EKF class (implementation)
//
//------------------------------------------------------------------------------

//
// Constructor
//

EKF::EKF()
 : n(0),   
   t(0.0) 
{
}

EKF::EKF(int n_)
 : n(n_),             // Number of estimation parameters
   t(0.0)             // Epoch
{
  // Allocate storage and initialize to zero
  x = Vector(n);
  P = Matrix(n,n);
}

//
// Re-dimensioning
//

EKF &EKF::resize(int n_) {
  // Allocate storage and initialize to zero
  n = n_;         
  t = 0.0;            
  x = Vector(n);
  P = Matrix(n,n);
  return (*this);
}; 


//
// Initialization of a new problem
//

void EKF::Init(double t_, Vector x_, Matrix P_) 
{
  t = t_; x = x_; P = P_;
}  

void EKF::Init(double t_, Vector x_, Vector sigma)
{
  t = t_; x = x_; 
  P = 0.0;
  for (int i=0; i<n; i++) P(i,i)=sigma(i)*sigma(i);
}  
    

//
// Access to filter parameters
//

double EKF::Time()  { return t; };     // Time    

Vector EKF::State() { return x; };     // State parameters

Matrix EKF::Cov()   { return P; };     // Covariance matrix

Vector EKF::StdDev() {                 // Standard deviation
  Vector Sigma(n);
  for (int i=0; i<n; i++) Sigma(i)=sqrt(P(i,i));
  return Sigma;
}

//
// Time Update
//

void EKF::TimeUpdate(double t_, const Vector& x_, const Matrix& Phi)
{
  t = t_;                    // Next time step
  x = x_;                    // Propagated state
  P = Phi*P*Transp(Phi);     // Propagated covariance 
}  

void EKF::TimeUpdate(double        t_, 
                     const Vector& x_, 
                     const Matrix& Phi, 
                     const Matrix& Qdt)
{
  t = t_;                          // Next time step
  x = x_;                          // Propagated state
  P = Phi*P*Transp(Phi) + Qdt;     // Propagated covariance + noise
}  

//
// Scalar Measurement Update 
//

void EKF::MeasUpdate(double z, double g, double sigma, const Vector& G)
{
  Vector K(n);                   // Kalman gain
  double Inv_W = sigma*sigma;    // Inverse weight (measurement covariance)

  // Kalman gain

  K = P*G/(Inv_W+Dot(G,P*G));

  // State update

  x = x + K*(z-g);

  // Covariance update

//P = (Id(n)-Dyadic(K,G))*P;                           // Kalman
  P = (Id(n)-Dyadic(K,G))*P*Transp(Id(n)-Dyadic(K,G))  // Joseph
      + Dyadic(K*sigma,K*sigma);

}

//
// Vector Measurement Update 
//

void EKF::MeasUpdate(const Vector& z,
                     const Vector& g,
                     const Vector& s,
                     const Matrix& G ) 
{
  
  int    i,m = z.size();
  Matrix K    (n,m);    // Kalman gain
  Matrix Inv_W(m,m);    // Measurement covariance
  
  for(i=0;i<m;i++) Inv_W(i,i) = s(i)*s(i);
 
  /*
  MeasUpdate(z,g,Inv_W,G);
  */  
  
  // Kalman gain

  K = P*Transp(G)*Inv(Inv_W+G*P*Transp(G));

  // State update

  x = x + K*(z-g);
  
  // Covariance update

//P = (Id(n)-K*G)*P;
  P = (Id(n)-K*G)*P*Transp(Id(n)-K*G) + K*Inv_W*Transp(K) ;
  
}

//
// Vector Measurement Update 
//

void EKF::MeasUpdate(const Vector& z, 
                     const Vector& g, 
                     const Matrix& Inv_W, 
                     const Matrix& G      )
{
  
  Matrix K(n,z.size());                 // Kalman gain
  
  // Kalman gain

  K = P*Transp(G)*Inv(Inv_W+G*P*Transp(G));

  // State update

  x = x + K*(z-g);
  
  // Covariance update

//P = (Id(n)-K*G)*P;                                         // Kalman
  P = (Id(n)-K*G)*P*Transp(Id(n)-K*G) + K*Inv_W*Transp(K) ;  // Joseph
  
}
