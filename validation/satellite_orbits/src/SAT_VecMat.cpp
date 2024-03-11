//------------------------------------------------------------------------------
//
// SAT_VecMat.cpp
// 
// Purpose: 
//
//   Vector/matrix operations
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
#include <iomanip>

#include "SAT_VecMat.h"


using std::ostream;
using std::cerr;
using std::endl;
using std::setw;


//------------------------------------------------------------------------------
//
// Vector (class implementation)
//
// Purpose:
//
//   Vector data type and associated operations
//
//------------------------------------------------------------------------------


// Constructors, destructor

Vector::Vector ()                              // Vector without elements
  : n(0) 
{
  v = 0;
}

Vector::Vector (int Size)                      // Creates null vector
 : n(Size)
{
  v = new double [Size];
  for (int i=0; i<Size; i++) v[i]=0.0;
}

Vector::Vector (const Vector& V)               // Vector copy
 : n(V.n)
{
  v = new double [V.n];
  for (int i=0; i<V.n; i++) v[i]=V.v[i];
}

Vector::Vector (const double* p, int N)        // Array copy
  : n(N)
{
  v = new double [N];
  for (int i=0; i<N; i++) v[i]=p[i];
}

Vector::Vector (double x, double y, double z)  // 3dim-Vector
  : n(3)
{
  v = new double [3];
  v[0]=x; v[1]=y; v[2]=z;
}

Vector::Vector (double x, double y, double z,   // 6dim-Vector
                double X, double Y, double Z)  
  : n(6)
{
  v = new double [6];
  v[0]=x; v[1]=y; v[2]=z;
  v[3]=X; v[4]=Y; v[5]=Z;
}

Vector::~Vector() 
{ 
  delete [] v; 
}

// Size

Vector& Vector::resize(int Size) {
  if (n==Size) return (*this);
  int    i,i_max;
  double *v_new = new double[Size];
  // Copy existing elements to new vector; pad with zeroes
  i_max = ((Size<n)? Size : n);
  for (i=0;i<i_max;i++)    v_new[i]=v[i];
  for (i=i_max;i<Size;i++) v_new[i]=0.0;
  // Dispose unused memory
  delete [] v;
  // Copy vector pointer and set dimension
  v = v_new;
  n = Size;
  return (*this);
};

// Component access

Vector Vector::slice (int first, int last) const
{
  Vector Aux(last-first+1);
  for (int i=first; i<=last; i++) Aux.v[i-first]=v[i];
  return Aux;
}


// Square root of vector elements

Vector Vector::Sqrt() const
{
  Vector Aux(n);
  for (int i=0; i<n; i++) Aux.v[i]=sqrt(v[i]);
  return Aux;
}


// Assignment

Vector& Vector::operator=(const double value)
{
  for (int i=0; i<n; i++) v[i]=value;
  return (*this);
}

Vector& Vector::operator=(const Vector& V)
{
  if (this == &V) return (*this);
  // Allocate vector if still empty
  if (n==0) {
    n = V.n; 
    v = new double [V.n];
  };
  // Check dimension
  if (n!=V.n) {
    cerr << "ERROR: Incompatible sizes in Vector operator=(Vector)" << endl;
    exit(1);
  };
  // Copy elements
  for (int i=0; i<n; i++) v[i]=V.v[i];
  return (*this);
}


// Concatenation

Vector Stack (const Vector& a, const Vector& b)
{
  int    i;
  Vector c(a.size()+b.size());
  for (i=0;i<a.size();i++) c(i)=a(i);
  for (i=0;i<b.size();i++) c(i+a.size())=b(i);
  return c;
}

Vector operator &(const Vector& a, double b) {
  int    n=a.n;
  Vector tmp(n+1);
  for (int i=0;i<n;i++) tmp.v[i]=a.v[i];
  tmp.v[n] = b;
  return tmp;
}

Vector operator &(double a, const Vector& b) {
  int    n=b.n;
  Vector tmp(n+1);
  tmp.v[0] = a;
  for (int i=1;i<n+1;i++) tmp.v[i]=b.v[i];
  return tmp;
}

Vector operator &(const Vector& a, const Vector& b)
{
  int    i;
  Vector c(a.n+b.n);
  for (i=0;i<a.n;i++) c.v[i]=a.v[i];
  for (i=0;i<b.n;i++) c.v[i+a.n]=b.v[i];
  return c;
}

// Vector from polar angles

Vector VecPolar (double azim, double elev, double r)
{
  return Vector(r*cos(azim)*cos(elev),r*sin(azim)*cos(elev),r*sin(elev));
}


// Vector addition/subtraction with assignment

void Vector::operator += (const Vector& V)
{
  if (n!=V.n) {
    cerr << "ERROR: Incompatible shape in Vector operator+=(Vector)" << endl;
    exit(1);
  };
  for (int i=0; i<n; i++) v[i]+=V.v[i];
}

void Vector::operator -= (const Vector& V)
{
  if (n!=V.n) {
    cerr << "ERROR: Incompatible shape in Vector operator-=(Vector)" << endl;
    exit(1);
  };
  for (int i=0; i<n; i++) v[i]-=V.v[i];
}


// Dot product, norm, cross product

double Dot (const Vector& left, const Vector& right)
{
  if (left.n!=right.n) {
    cerr << "ERROR: Incompatible shape in Dot(Vector,Vector)" << endl;
    exit(1);
  };
  double Sum = 0.0;
  for (int i=0; i<left.n; i++) Sum+=left.v[i]*right.v[i];
  return Sum;
}

double Norm (const Vector& V)
{
  return sqrt(Dot(V,V));
}

Vector Cross (const Vector& left, const Vector& right)
{
  if ( (left.n!=3) || (right.n!=3) ) {
    cerr << "ERROR: Invalid dimension in Cross(Vector,Vector)" << endl;
    exit(1);
  };
  Vector Result(3);
  Result.v[0] = left.v[1]*right.v[2] - left.v[2]*right.v[1];
  Result.v[1] = left.v[2]*right.v[0] - left.v[0]*right.v[2];
  Result.v[2] = left.v[0]*right.v[1] - left.v[1]*right.v[0];
  return Result;
}


// Scalar multiplication and division of a vector

Vector operator * (double value, const Vector& V)
{
  Vector Aux(V.n);
  for (int i=0; i<V.n; i++) Aux.v[i]=value*V.v[i];
  return Aux;
}

Vector operator * (const Vector& V, double value)
{
  return value*V;
}

Vector operator / (const Vector& V, double value)
{
  Vector Aux(V.n);
  for (int i=0; i<V.n; i++) Aux.v[i]=V.v[i]/value;
  return Aux;
}


// Negation of a vector (unary minus)

Vector operator - (const Vector& V)
{
  Vector Aux(V.n);
  for (int i=0; i<V.n; i++) Aux.v[i]=-V.v[i];
  return Aux;
}


// Vector addition and subtraction

Vector operator + (const Vector& left, const Vector& right)
{
  if (left.n!=right.n) {
    cerr << "ERROR: Incompatible shape in +(Vector,Vector)" << endl;
    exit(1);
  };
  Vector Aux(left.n);
  for (int i=0; i<left.n; i++) Aux.v[i]=left.v[i]+right.v[i];
  return Aux;
}

Vector operator - (const Vector& left, const Vector& right)
{
  if (left.n!=right.n) {
    cerr << "ERROR: Incompatible shape in -(Vector,Vector)" << endl;
    exit(1);
  };
  Vector Aux(left.n);
  for (int i=0; i<left.n; i++) Aux.v[i]=left.v[i]-right.v[i];
  return Aux;
}

// Vector output

ostream& operator << (ostream& os, const Vector& Vec)
{
  int w = os.width();
  for (int i=0; i<Vec.size(); i++)
    os << setw(w) << Vec(i);
//os << endl;
  return os;
}



//------------------------------------------------------------------------------
//
// Matrix (class implementation)
//
// Purpose:
//
//   Matrix data type and associated operations
//
//------------------------------------------------------------------------------


// Constructors

Matrix::Matrix ()                          // Matrix without elements
  : n(0), m(0) 
{
  M = 0;
}

Matrix::Matrix (int dim1, int dim2)        // Nullmatrix of specified shape 
  : n(dim1), m(dim2)
{
  int i,j;
  // Memory allocation
  M = new double*[dim1];
  for (i=0; i<dim1; i++) M[i] = new double[dim2];
  // Initialization
  for (i=0; i<dim1; i++) {
    for (j=0; j<dim2; j++) M[i][j]=0.0;
  }
}

Matrix::Matrix (const Matrix& M_)          // Copy
{
  int i,j;
  n = M_.n;
  m = M_.m;
  // Memory allocation
  M = new double*[n];
  for (i=0; i<n; i++) M[i] = new double[m];
  // Initialization
  for (i=0; i<n; i++) {
    for (j=0; j<m; j++) M[i][j]=M_.M[i][j];
  }
}

Matrix::Matrix (const double* p, int dim1, int dim2)   // Array copy
{
  int i,j;
  n = dim1;
  m = dim2;
  // Memory allocation
  M = new double*[n];
  for (i=0; i<n; i++) M[i] = new double[m];
  // Initialization
  for (i=0; i<n; i++) {
    for (j=0; j<m; j++) M[i][j]=p[i*dim2+j];
  }
  
}

Matrix::~Matrix() 
{ 
  for (int i=0; i<n; i++) delete[] M[i];
  delete [] M; 
};

// Size

Matrix& Matrix::resize(int dim1, int dim2) {
  if (n==dim1 && m==dim2) return (*this);
  int    i,j,i_max,j_max;
  // Alocate new matrix
  double **M_new = new double*[dim1];
  for (i=0; i<dim1; i++) M_new[i] = new double[dim2];
  // Copy existing elements to new matrix; pad with zeroes
  i_max = ((dim1<n)? dim1 : n);
  j_max = ((dim2<m)? dim2 : m);
  for (i=0;i<i_max;i++)    for (j=0;j<j_max;j++) M_new[i][j]=M[i][j];
  for (i=i_max;i<dim1;i++) for (j=0;j<dim2 ;j++) M_new[i][j]=0.0;
  for (i=0;i<i_max;i++) for (j=j_max;j<dim2;j++) M_new[i][j]=0.0;
  // Dispose unused memory
  for (i=0; i<n; i++) delete[] M[i];
  delete [] M; 
  // Copy vector pointer and set dimensions
  M = M_new;
  n = dim1; 
  m = dim2;
  return (*this);
};


// Assignment

Matrix& Matrix::operator=(const double value)
{
  for (int i=0; i<n; i++) 
    for (int j=0; j<m; j++) 
      M[i][j]=value;
  return (*this);
}

Matrix& Matrix::operator=(const Matrix& M_)
{
  int i,j;
  if (this == &M_) return (*this);
  // Allocate matrix if still empty
  if (n==0 && m==0) {
    n = M_.n; 
    m = M_.m;
    M = new double*[n];
    for (i=0; i<n; i++) M[i] = new double[m];
  };
  if ( (n!=M_.n) || (m!=M_.m) ) {
    cerr << "ERROR: Incompatible shapes in Matrix operator=(Matrix)" << endl;
    exit(1);
  };
  for (i=0; i<n; i++) 
    for (j=0; j<m; j++) 
      M[i][j]=M_.M[i][j];
  return (*this);
}


// Component access

Vector Matrix::Col(int j) const
{
  Vector Res(n);
  for (int i=0; i<n; i++)  Res.v[i]=M[i][j];
  return Res;
}

Vector Matrix::Row(int i) const
{
  Vector Res(m);
  for (int j=0; j<m; j++)  Res.v[j]=M[i][j];
  return Res;
}

Vector Matrix::Diag() const
{
  if (n!=m) {
    cerr << "ERROR: Invalid shape in Matrix.Diag()" << endl;
    exit(1);
  };
  Vector Vec(n);
  for (int i=0; i<n; i++) Vec.v[i] = M[i][i];
  return Vec;
}

double Matrix::Trace() const {
  return this->Trace(0,n-1);
};

double Matrix::Trace(int low, int upp) const {
  double tmp = 0.0;
  if (n!=m) {
    cerr << "ERROR: Invalid shape in Matrix.Trace()" << endl;
    exit(1);
  };
  if (low<0 || n<=upp) {
    cerr << "ERROR: Invalid arguments in Matrix.Trace()" << endl;
    exit(1);
  };
  for (int i=low; i<=upp; i++) tmp += M[i][i];
  return tmp;
};


Matrix Matrix::slice(int first_row, int last_row, int first_col, int last_col)
{
  if (first_row<0 || last_row<first_row || n-1<last_row ||
      first_col<0 || last_col<first_col || m-1<last_col) {
    cerr << "ERROR: Invalid arguments in Matrix.slice()" << endl;
    exit(1);
  };
  Matrix Aux(last_row-first_row+1,last_col-first_col+1);
  for (int i=0;i<=last_row-first_row;i++)
    for (int j=0;j<=last_col-first_col;j++)
       Aux(i,j) = M[i+first_row][j+first_col];
  return Aux;
}


void Matrix::SetCol(int j, const Vector& Col) 
{
  if (Col.size()!=n) {
    cerr << "ERROR: Incompatible shapes in Matrix.SetCol()" << endl;
    exit(1);
  };
  if (j<0 || m<=j) {
    cerr << "ERROR: Column index out of range in Matrix.SetCol()" << endl;
    exit(1);
  };
  for (int i=0; i<n; i++) M[i][j]=Col(i);
}

void Matrix::SetRow(int i, const Vector& Row)
{
  if (Row.size()!=m) {
    cerr << "ERROR: Incompatible shapes in Matrix.SetRow()" << endl;
    exit(1);
  };
  if (i<0 || n<=i) {
    cerr << "ERROR: Row index out of range in Matrix.SetRow()" << endl;
    exit(1);
  };
  for (int j=0; j<m; j++) M[i][j]=Row(j);
}


// Concatenation

Matrix operator &(const Matrix& A, const Vector& Row) {
  int    n=A.n;
  int    m=A.m;
  Matrix tmp(n+1,m);
  if ( m!=Row.size() ) {
    cerr << "ERROR: Incompatible shape in Matrix&Vector concatenation" << endl;
    exit(1);
  };
  for (int j=0;j<m;j++) {
    for (int i=0;i<n;i++) tmp.M[i][j]=A.M[i][j];
    tmp.M[n][j] = Row(j);
  };
  return tmp;
}

Matrix operator &(const Vector& Row, const Matrix& A) {
  int    n=A.n;
  int    m=A.m;
  Matrix tmp(n+1,m);
  if ( m!=Row.size() ) {
    cerr << "ERROR: Incompatible shape in Vector&Matrix concatenation" << endl;
    exit(1);
  };
  for (int j=0;j<m;j++) {
    tmp.M[0][j] = Row(j);
    for (int i=0;i<n;i++) tmp.M[i+1][j]=A.M[i][j];
  };
  return tmp;
}

Matrix operator &(const Matrix& A, const Matrix& B) {
  int    i;
  Matrix tmp(A.n+B.n,A.m);
  if ( A.m!=B.m ) {
    cerr << "ERROR: Incompatible shape in Matrix&Matrix concatenation" << endl;
    exit(1);
  };
  for (int j=0;j<A.m;j++) {
    for (i=0;i<A.n;i++) tmp.M[i    ][j]=A.M[i][j];
    for (i=0;i<B.n;i++) tmp.M[i+A.n][j]=B.M[i][j];
  };
  return tmp;
};


Matrix operator |(const Matrix& A, const Vector& Col) {
  int    n=A.n;
  int    m=A.m;
  Matrix tmp(n,m+1);
  if ( n!=Col.size() ) {
    cerr << "ERROR: Incompatible shape in Matrix|Vector concatenation" << endl;
    exit(1);
  };
  for (int i=0;i<n;i++) {
    for (int j=0;j<m;j++) tmp.M[i][j]=A.M[i][j];
    tmp.M[i][m] = Col(i);
  };
  return tmp;
}

Matrix operator |(const Vector& Col, const Matrix& A) {
  int    n=A.n;
  int    m=A.m;
  Matrix tmp(n,m+1);
  if ( n!=Col.size() ) {
    cerr << "ERROR: Incompatible shape in Vector|Matrix concatenation" << endl;
    exit(1);
  };
  for (int i=0;i<n;i++) {
    tmp.M[i][0] = Col(i);
    for (int j=0;j<m;j++) tmp.M[i][j+1]=A.M[i][j];
  };
  return tmp;
};

Matrix operator |(const Matrix& A, const Matrix& B) {

  int    j;
  Matrix tmp(A.n,A.m+B.m);
  if ( A.n!=B.n ) {
    cerr << "ERROR: Incompatible shape in Matrix|Matrix concatenation" << endl;
    exit(1);
  };
  for (int i=0;i<A.n;i++) {
    for (j=0;j<A.m;j++) tmp.M[i][j    ]=A.M[i][j];
    for (j=0;j<B.m;j++) tmp.M[i][j+A.m]=B.M[i][j];
  };
  return tmp;


};



// Matrix addition/subtraction with assignment

void Matrix::operator += (const Matrix& M)
{
  if ( (n!=M.n) || (m!=M.m) ) {
    cerr << "ERROR: Incompatible shape in Matrix operator+=(Matrix)" << endl;
    exit(1);
  };
  for (int i=0; i<n; i++) 
    for (int j=0; j<m; j++) 
      this->M[i][j]+=M.M[i][j];
}

void Matrix::operator -= (const Matrix& M)
{
  if ( (n!=M.n) || (m!=M.m) ) {
    cerr << "ERROR: Incompatible shape in Matrix operator-=(Matrix)" << endl;
    exit(1);
  };
  for (int i=0; i<n; i++) 
    for (int j=0; j<m; j++) 
      this->M[i][j]-=M.M[i][j];
}


// Unit matrix

Matrix Id(int Size)
{
  Matrix Aux(Size,Size);      
  for (int i=0; i<Size; i++) Aux.M[i][i] = 1.0;
  return Aux;
}


// Diagonal matrix

Matrix Diag(const Vector& Vec)
{
  Matrix Mat(Vec.n,Vec.n);
  for (int i=0; i<Vec.n; i++) Mat.M[i][i] = Vec.v[i];
  return Mat;
}


// Elementary rotations

Matrix R_x(double Angle)
{
  const double C = cos(Angle);
  const double S = sin(Angle);
  Matrix U(3,3);
  U.M[0][0] = 1.0;  U.M[0][1] = 0.0;  U.M[0][2] = 0.0;
  U.M[1][0] = 0.0;  U.M[1][1] =  +C;  U.M[1][2] =  +S;
  U.M[2][0] = 0.0;  U.M[2][1] =  -S;  U.M[2][2] =  +C;
  return U;
}

Matrix R_y(double Angle)
{
  const double C = cos(Angle);
  const double S = sin(Angle);
  Matrix U(3,3);
  U.M[0][0] =  +C;  U.M[0][1] = 0.0;  U.M[0][2] =  -S;
  U.M[1][0] = 0.0;  U.M[1][1] = 1.0;  U.M[1][2] = 0.0;
  U.M[2][0] =  +S;  U.M[2][1] = 0.0;  U.M[2][2] =  +C;
  return U;
}

Matrix R_z(double Angle)
{
  const double C = cos(Angle);
  const double S = sin(Angle);
  Matrix U(3,3);
  U.M[0][0] =  +C;  U.M[0][1] =  +S;  U.M[0][2] = 0.0;
  U.M[1][0] =  -S;  U.M[1][1] =  +C;  U.M[1][2] = 0.0;
  U.M[2][0] = 0.0;  U.M[2][1] = 0.0;  U.M[2][2] = 1.0;
  return U;
}


// Transposition

Matrix Transp(const Matrix& Mat)
{
  Matrix T(Mat.m,Mat.n);
  for ( int i=0; i<T.n; i++ )
    for ( int j=0; j<T.m; j++ )
      T.M[i][j] = Mat.M[j][i];
  return T;
}


// Inverse

Matrix Inv(const Matrix& Mat)
{
  const int n = Mat.n;

  Matrix LU(n,n), Inverse(n,n);
  Vector b(n), Indx(n);

  if (Mat.m!=Mat.n) {
    cerr << "ERROR: Invalid shape in Inv(Matrix)" << endl;
    exit(1);
  };

  // LU decomposition 

  LU = Mat;
  LU_Decomp ( LU, Indx );

  // Solve Ax=b for  unit vectors b_1..b_n

  for (int j=0; j<n; j++ ) {
    b=0.0; b(j)= 1.0;                     // Set b to j-th unit vector 
    LU_BackSub ( LU, Indx, b );           // Solve Ax=b 
    Inverse.SetCol(j,b);                  // Copy result
  };

  return Inverse;

}


// Scalar multiplication and division of a vector

Matrix operator * (double value, const Matrix& Mat)
{
  Matrix Aux(Mat.n,Mat.m);
  for (int i=0; i<Mat.n; i++) 
    for (int j=0; j<Mat.m; j++) 
      Aux.M[i][j]=value*Mat.M[i][j];
  return Aux;
}

Matrix operator * (const Matrix& Mat, double value)
{
  return value*Mat;
}

Matrix operator / (const Matrix& Mat, double value)
{
  Matrix Aux(Mat.n,Mat.m);
  for (int i=0; i<Mat.n; i++) 
    for (int j=0; j<Mat.m; j++) 
      Aux.M[i][j]=Mat.M[i][j]/value;
  return Aux;
}


// Unary minus

Matrix operator - (const Matrix& Mat)
{
  Matrix Aux(Mat.n,Mat.m);
  for (int i=0; i<Mat.n; i++) 
    for (int j=0; j<Mat.m; j++) 
      Aux.M[i][j]=-Mat.M[i][j];
  return Aux;
}


// Matrix addition and subtraction

Matrix operator + (const Matrix& left, const Matrix& right)
{
  if ( (left.n!=right.n) || (left.m!=right.m) ) {
    cerr << "ERROR: Incompatible shape in +(Matrix,Matrix)" << endl;
    exit(1);
  };
  Matrix Aux(left.n,left.m);
  for (int i=0; i<left.n; i++) 
    for (int j=0; j<left.m; j++) 
      Aux.M[i][j] = left.M[i][j] + right.M[i][j];
  return Aux;
}

Matrix operator - (const Matrix& left, const Matrix& right)    
{
  if ( (left.n!=right.n) || (left.m!=right.m) ) {
    cerr << "ERROR: Incompatible shape in -(Matrix,Matrix)" << endl;
    exit(1);
  };
  Matrix Aux(left.n,left.m);
  for (int i=0; i<left.n; i++) 
    for (int j=0; j<left.m; j++) 
      Aux.M[i][j] = left.M[i][j] - right.M[i][j];
  return Aux;
}


// Matrix product

Matrix operator * (const Matrix& left, const Matrix& right)
{
  if (left.m!=right.n) {
    cerr << "ERROR: Incompatible shape in *(Matrix,Matrix)" << endl;
    exit(1);
  };
  Matrix Aux(left.n,right.m);
  double Sum;
  for (int i=0; i<left.n; i++) 
    for (int j=0; j<right.m; j++) {
      Sum = 0.0;
      for (int k=0; k<left.m; k++) 
        Sum += left.M[i][k] * right.M[k][j];
      Aux.M[i][j] = Sum;
    }
  return Aux;
}


// Vector/matrix product

Vector operator * (const Matrix& Mat, const Vector& Vec)
{
  if (Mat.m!=Vec.n) {
    cerr << "ERROR: Incompatible shape in *(Matrix,Vector)" << endl;
    exit(1);
  };
  Vector Aux(Mat.n);
  double Sum;
  for (int i=0; i<Mat.n; i++) {
    Sum = 0.0;
    for (int j=0; j<Mat.m; j++) 
      Sum += Mat.M[i][j] * Vec.v[j];
    Aux.v[i] = Sum;
  }
  return Aux;
}

Vector operator * (const Vector& Vec, const Matrix& Mat)
{
  if (Mat.n!=Vec.n) {
    cerr << "ERROR: Incompatible shape in *(Vector,Matrix)" << endl;
    exit(1);
  };
  Vector Aux(Mat.m);
  double Sum;
  for (int j=0; j<Mat.m; j++) {
    Sum = 0.0;
    for (int i=0; i<Mat.n; i++) 
      Sum += Vec.v[i] * Mat.M[i][j];
    Aux.v[j] = Sum;
  }
  return Aux;
}


// Dyadic product

Matrix Dyadic (const Vector& left, const Vector& right)
{
  Matrix Mat(left.n,right.n);
  for (int i=0;i<left.n;i++)
    for (int j=0;j<right.n;j++)
      Mat.M[i][j] = left.v[i]*right.v[j];
  return Mat;
}


// Matrix output

ostream& operator << (ostream& os, const Matrix& Mat)
{
  int w = os.width();
  for (int i=0; i<Mat.size1(); i++) {
    for (int j=0; j<Mat.size2(); j++)
      os << setw(w) << Mat(i,j);
    os << endl;
  }
  return os;
}


//------------------------------------------------------------------------------
//
// Basic Linear Algebra
//
//------------------------------------------------------------------------------


//------------------------------------------------------------------------------
// 
// LU_Decomp
//
// Purpose:
//
//   LU-Decomposition.
//
//   Given an nxn matrix A, this routine replaces it by the LU decomposition
//   of a rowwise permutation of itself. A is output, arranged as in 
//   equation (2.3.14) of Press et al. (1986); Indx is an ouput vector which
//   records the row permutation effected by partial pivoting. This routine is 
//   used in combination with LU_BackSub to solve linear equations or invert 
//   a matrix.
//
// Input/output:
//
//   A       Square matrix; replaced by LU decomposition of permutation of A
//           on output
//   Indx    Permutation index vector
//
// Note:
//
//   Adapted from LUDCMP of Press et al. (1986).
//
//------------------------------------------------------------------------------

void LU_Decomp ( Matrix& A, Vector& Indx )
{

  // Constants
      
  const int    n    = A.size1();
  const double tiny = 1.0e-20;       // A small number

  // Variables
      
  int     imax=0;
  int     i,j,k;
  double  aAmax, Sum, Dum;
  Vector  V(n); 

  // Loop over rows to get scaling information

  for (i=0; i<n; i++) {
    aAmax = 0.0;
    for (j=0;j<n;j++) if (fabs(A(i,j)) > aAmax ) aAmax=fabs(A(i,j));
    if (aAmax==0.0) {
      // No nonzero largest element
      cerr << "ERROR: Singular matrix A in LU_Decomp";
      exit(1);
    };
    V(i) = 1.0/aAmax;           // V stores the implicit scaling of each row
  };

  // Loop over columns of Crout's method
      
  for ( j=0; j<n; j++ ) {
      
    if (j > 0) {
      for ( i=0; i<j; i++ ) {   // This is equation 2.3.12 except for i=j
        Sum = A(i,j);
        if (i>0) {
          for ( k=0; k<i; k++ )  Sum -= A(i,k)*A(k,j);
          A(i,j) = Sum;
        };
      };
    };
      
    aAmax=0.0;                  // Initialize for the search of the largest
                                // pivot element
      
    for ( i=j; i<n; i++ ) {     // This is i=j of equation 2.3.12 and 
      Sum = A(i,j);             // i=j+1..N of equation 2.3.13
      if (j > 0) {
        for ( k=0; k<j; k++ ) Sum -= A(i,k)*A(k,j);
        A(i,j) = Sum;
      };
      Dum = V(i)*fabs(Sum);     // Figure of merit for the pivot
      if (Dum >= aAmax) {       // Is it better than the best so far ?
        imax  = i;
        aAmax = Dum;
      };
    };
      
    if (j != imax) {            // Do we need to interchange rows?
      for ( k=0; k<n; k++) {    // Yes, do so ...
        Dum = A(imax,k);
        A(imax,k) = A(j,k);
        A(j,k) = Dum;
      }
      V(imax) = V(j);           // Also interchange the scale factor 
    };
      
    Indx(j) = imax;
      
    if (j != n-1) {             // Now finally devide by the pivot element
      if (A(j,j) == 0.0) {      // If the pivot element is zero the matrix 
        A(j,j) = tiny;          // is singular (at least to the precision of
      };                        // the algorithm). For some applications on
      Dum=1.0/A(j,j);           // singular matrices, it is desirable to 
      for (i=j+1;i<n;i++) {     // substitude tiny for zero. 
        A(i,j)=A(i,j)*Dum;
      };
    };

  };   // Go back for the next column in the reduction

  if (A(n-1,n-1)==0.0) A(n-1,n-1)=tiny; 

};


//------------------------------------------------------------------------------
//
// LU_BackSub
//
// Purpose:
//
//   LU Backsubstitution
//
//   Solves the set of n linear equations Ax=b. Here A is input, not as the 
//   matrix A but rather as its LU decomposition, determined by the function
//   LU_Decomp. b is input as the right-hand side vector b, and returns with
//   the solution vector x. A and Indx are not modified by this function and 
//   can be left in place for successive calls with different right-hand 
//   sides b. This routine takes into account the posssibility that B will  
//   begin with many zero elements, so it is efficient for use in matrix
//   inversions.
//
// Input/output:
//
//   A       LU decomposition of permutation of A
//   Indx    Permutation index vector
//   b       Right-hand side vector b; replaced by solution x of Ax=b on output
//
//------------------------------------------------------------------------------

void LU_BackSub ( Matrix& A, Vector& Indx, Vector& b )
{ 

  // Constants
      
  const int  n = A.size1();

  // Local variables

  int     ii,i,ll,j;
  double  Sum;

  //
  // Start
  //

  ii = -1;                      // When ii is set to a nonegative value, it will
                                // become the first nonvanishing element of B. 
  for (i=0; i<n; i++) {         // We now do the forward substitution.
    ll = (int) Indx(i);         // The only wrinkle is to unscramble the 
    Sum = b(ll);                // permutation as we go.
    b(ll) = b(i);
    if (ii != -1) {
      for (j=ii; j<i; j++) Sum -= A(i,j)*b(j);
    }
    else {
      if (Sum != 0.0) ii = i;   // A nonzero element was encountered, so from 
    };                          // now on we will have to do the sums in the
    b(i) = Sum;                 // loop above.
   };
      
   for (i=n-1; i>=0; i--) {     // Now we do the backsubstitution, eqn 2.3.7.
     Sum=b(i);
     if (i<n-1) {
       for (j=i+1;j<n;j++) {
          Sum = Sum-A(i,j)*b(j);
       };
     };
     b(i) = Sum/A(i,i);         // Store a component of the solution vector X.
   };

};


