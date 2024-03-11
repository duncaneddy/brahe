//------------------------------------------------------------------------------
//
// SAT_VecMat.h
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

#ifndef INC_SAT_VECMAT_H
#define INC_SAT_VECMAT_H

#include <iostream>

class Matrix;

//------------------------------------------------------------------------------
//
// Vector (class definition)
//
// Purpose:
//
//   Vector data type and associated operations
//
//------------------------------------------------------------------------------

class Vector
{

  public:
    
    friend class Matrix;
    
    // Constructors
    Vector ();                              // Vector without elements
    Vector (int Size);                      // Nullvector of specified size 
    Vector (const Vector& V);               // Vector copy
    Vector (const double* p, int N);        // Array copy
    Vector (double x, double y, double z);  // 3dim-Vector
    Vector (double x, double y, double z,   // 6dim-Vector
            double X, double Y, double Z);  

    // Destructor
    ~Vector();
    
    // Size
    int size() const { return n; };
    Vector& resize(int Size);
    
    // Assignment
    Vector& operator=(const double value);
    Vector& operator=(const Vector& V);

    // Component access (Fortran notation)
    double  operator () (int i) const { return v[i]; };
    double& operator () (int i)       { return v[i]; };
    Vector slice (int first, int last) const;

    // Square root of vector elements
    Vector Sqrt() const;

    // Vector addition/subtraction with assignment
    void operator += (const Vector& V);
    void operator -= (const Vector& V);
    
    // Friends 
    friend Vector operator & (const Vector& a, double b);
    friend Vector operator & (double a, const Vector& b);
    friend Vector operator & (const Vector& a, const Vector& b);
    friend Vector Stack      (const Vector& a, const Vector& b);
    friend Vector VecPolar   (double azim, double elev, double);
    friend double Dot        (const Vector& left, const Vector& right);
    friend double Norm       (const Vector& V);
    friend Vector Cross      (const Vector& left, const Vector& right);
    friend Vector operator * (double value, const Vector& V);
    friend Vector operator * (const Vector& V, double value);
    friend Vector operator / (const Vector& V, double value);
    friend Vector operator - (const Vector& V);
    friend Vector operator + (const Vector& left, const Vector& right);    
    friend Vector operator - (const Vector& left, const Vector& right);    
    friend Matrix Diag       (const Vector& Vec);
    friend Vector operator * (const Matrix& Mat, const Vector& Vec);
    friend Vector operator * (const Vector& Vec, const Matrix& Mat);
    friend Matrix Dyadic     (const Vector& left, const Vector& right);
    friend std::ostream& operator << (std::ostream& os, const Vector& Vec);

  private:
        
    // Elements
    int    n;      // Dimension
    double *v;     // Vector v(n)

};


//------------------------------------------------------------------------------
//
// Matrix (class definition)
//
// Purpose:
//
//   Matrix data type and associated operations
//
//------------------------------------------------------------------------------

class Matrix
{

  public:

    // Constructors
    Matrix ();                                      // Matrix without elements
    Matrix (int dim1, int dim2);                    // Nullmatrix 
    Matrix (const Matrix& M_);                      // Matrix copy
    Matrix (const double* p, int dim1, int dim2);   // Array copy

    // Destructor
    ~Matrix();

    // Assignment
    Matrix& operator=(const double value);
    Matrix& operator=(const Matrix& M_);

    // Size
    int size1() const { return n; };
    int size2() const { return m; };
    Matrix& resize(int dim1, int dim2);
    
    // Component access (Fortran notation)
    double  operator () (int i, int j) const { return M[i][j]; };   
    double& operator () (int i, int j)       { return M[i][j]; };   
    Vector Col(int j) const;      
    Vector Row(int i) const;      
    Vector Diag() const;
    double Trace() const;
    double Trace(int low, int upp) const;
    Matrix slice(int first_row, int last_row, int first_col, int last_col);
    void SetCol(int j, const Vector& Col);
    void SetRow(int i, const Vector& Row);

    // Matrix addition/subtraction with assignment
    void operator += (const Matrix& V);
    void operator -= (const Matrix& V);

    // Friends 
    friend Matrix operator & (const Matrix& A, const Vector& Row);
    friend Matrix operator & (const Vector& Row, const Matrix& A);
    friend Matrix operator & (const Matrix& A, const Matrix& B);
    friend Matrix operator | (const Matrix& A, const Vector& Col);
    friend Matrix operator | (const Vector& Col, const Matrix& A);
    friend Matrix operator | (const Matrix& A, const Matrix& B);
    friend Matrix Id         (int Size);
    friend Matrix Diag       (const Vector& Vec);
    friend Matrix R_x        (double Angle);
    friend Matrix R_y        (double Angle);
    friend Matrix R_z        (double Angle);
    friend Matrix Transp     (const Matrix& Mat);
    friend Matrix Inv        (const Matrix& Mat);
    friend Matrix operator * (double value, const Matrix& Mat);
    friend Matrix operator * (const Matrix& Mat, double value);
    friend Matrix operator / (const Matrix& Mat, double value);
    friend Matrix operator - (const Matrix& Mat);
    friend Matrix operator + (const Matrix& left, const Matrix& right);
    friend Matrix operator - (const Matrix& left, const Matrix& right);    
    friend Matrix operator * (const Matrix& left, const Matrix& right);
    friend Vector operator * (const Matrix& Mat, const Vector& Vec);
    friend Vector operator * (const Vector& Vec, const Matrix& Mat);
    friend Matrix Dyadic     (const Vector& left, const Vector& right);
    friend std::ostream& operator << (std::ostream& os, const Matrix& Mat);

  private:

    // Elements
    int      n;                       // First dimension (number of rows)
    int      m;                       // Second dimension (number of columns)
    double **M;                       // Matrix M(n,m)

};

//------------------------------------------------------------------------------
//
// Vector/Matrix functions and operators
//
//------------------------------------------------------------------------------

// Concatenation 
Vector operator &(const Vector& a, double b);
Vector operator &(double a, const Vector& b);
Vector operator &(const Vector& a, const Vector& b);
Vector Stack     (const Vector& a, const Vector& b);

// Vector from polar angles
Vector VecPolar (double azim, double elev, double r=1.0);

// Dot product, norm, cross product
double Dot (const Vector& left, const Vector& right);
double Norm (const Vector& V);
Vector Cross (const Vector& left, const Vector& right);

// Scalar multiplication and division of a vector
Vector operator * (double value, const Vector& V);
Vector operator * (const Vector& V, double value);
Vector operator / (const Vector& V, double value);

// Negation of a vector (unary minus)
Vector operator - (const Vector& V);

// Vector addition and subtraction
Vector operator + (const Vector& left, const Vector& right);    
Vector operator - (const Vector& left, const Vector& right);    

// Diagonal matrix
Matrix Diag(const Vector& Vec);

// Vector/matrix product
Vector operator * (const Matrix& Mat, const Vector& Vec);
Vector operator * (const Vector& Vec, const Matrix& Mat);

// Dyadic product
Matrix Dyadic (const Vector& left, const Vector& right);

// Output
std::ostream& operator << (std::ostream& os, const Vector& Vec);


// Concatenation 
Matrix operator &(const Matrix& A, const Vector& Row);
Matrix operator &(const Vector& Row, const Matrix& A);
Matrix operator &(const Matrix& A, const Matrix& B);
Matrix operator |(const Matrix& A, const Vector& Col);
Matrix operator |(const Vector& Col, const Matrix& A);
Matrix operator |(const Matrix& A, const Matrix& B);

// Unit matrix
Matrix Id(int Size);

// Diagonal matrix
Matrix Diag(const Vector& Vec);

// Elementary rotations
Matrix R_x(double Angle);
Matrix R_y(double Angle);
Matrix R_z(double Angle);

// Transposition and inverse
Matrix Transp(const Matrix& Mat);
Matrix Inv(const Matrix& Mat);

// Scalar multiplication and division of a matrix
Matrix operator * (double value, const Matrix& Mat);
Matrix operator * (const Matrix& Mat, double value);
Matrix operator / (const Matrix& Mat, double value);

// Unary minus
Matrix operator - (const Matrix& Mat);

// Matrix addition and subtraction
Matrix operator + (const Matrix& left, const Matrix& right);
Matrix operator - (const Matrix& left, const Matrix& right);    

// Matrix product
Matrix operator * (const Matrix& left, const Matrix& right);

// Vector/matrix product
Vector operator * (const Matrix& Mat, const Vector& Vec);
Vector operator * (const Vector& Vec, const Matrix& Mat);

// Dyadic product
Matrix Dyadic (const Vector& left, const Vector& right);

// Output
std::ostream& operator << (std::ostream& os, const Matrix& Mat);



//------------------------------------------------------------------------------
//
// Basic Linear Algebra
//
//------------------------------------------------------------------------------

void LU_Decomp ( Matrix& A, Vector& Indx );
void LU_BackSub ( Matrix& A, Vector& Indx, Vector& b );


#endif  // include-Blocker
