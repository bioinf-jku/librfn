
typedef int spmat_t;

#ifndef MKL
/* this part wraps and implements the NIST interface */

/*
*
* Sparse BLAS (Basic Linear Algebra Subprograms) Library
*
* A C++ implementation of the routines specified by the ANSI C
* interface specification of the Sparse BLAS in the BLAS Technical
* Forum Standard[1].   For details, see [2].
*
* Mathematical and Computational Sciences Division
* National Institute of Technology,
* Gaithersburg, MD USA
*
*
* [1] BLAS Technical Forum: www.netlib.org/blas/blast-forum/
* [2] I. S. Duff, M. A. Heroux, R. Pozo, "An Overview of the Sparse Basic
*     Linear Algebra Subprograms: The new standard of the BLAS Techincal
*     Forum,"  Vol. 28, No. 2, pp. 239-267,ACM Transactions on Mathematical
*     Software (TOMS), 2002.
*
*
* DISCLAIMER:
*
* This software was developed at the National Institute of Standards and
* Technology (NIST) by employees of the Federal Government in the course
* of their official duties. Pursuant to title 17 Section 105 of the
* United States Code, this software is not subject to copyright protection
* and is in the public domain. NIST assumes no responsibility whatsoever for
* its use by other parties, and makes no guarantees, expressed or implied,
* about its quality, reliability, or any other characteristic.
*
*
*/


/* numeric is for accumulate() below */
#include <iostream>
#include <complex>
#include <numeric>
#include <vector>
#include <utility>
  /* pair defined here */

#include "blas_enum.h"
#include "blas_sparse_proto.h"

#include <cstring> /* for memset */
#include <cassert> /* for assert */


#ifdef SPBLAS_ERROR_FATAL
#include <cassert>
#define ASSERT_RETURN(x, ret_val) assert(x)
#define ERROR_RETURN(ret_val)  assert(0)
#else
#define ASSERT_RETURN(x, ret_val) {if (!(x)) return ret_val;}
#define ERROR_RETURN(ret_val) return ret_val
#endif



using namespace std;

namespace NIST_SPBLAS
{



/**
   Generic sparse matrix (base) class: defines only the structure
   (size, symmetry, etc.) and maintains state during construction,
   but does not specify the actual nonzero values, or their type.

*/
class Sp_mat
{
  private:

    int num_rows_;
    int num_cols_;
    int num_nonzeros_;

    /* ... */

    int void_;
    int nnew_;      /* avoid using "new" since it is a C++ keyword */
    int open_;
    int valid_;

    int unit_diag_ ;
    int complex_;
    int real_;
    int single_precision_;
    int double_precision_;
    int upper_triangular_;
    int lower_triangular_;
    int upper_symmetric_;
    int lower_symmetric_;
    int upper_hermitian_;
    int lower_hermitian_;
    int general_;

    int one_base_;


        /* optional block information */

    int Mb_;                /* matrix is partitioned into Mb x Nb blocks    */
    int Nb_;                /* otherwise 0, if regular (non-blocked) matrix */
    int k_;                 /* for constant blocks, each block is k x l     */
    int l_;                 /* otherwise 0, if variable blocks are used.   */

    int rowmajor_;          /* 1,if block storage is rowm major.  */
    int colmajor_;          /* 1,if block storage is column major. */

    /* unused optimization paramters */

    int opt_regular_;
    int opt_irregular_;
    int opt_block_;
    int opt_unassembled_;

    vector<int> K_; /* these are GLOBAL index of starting point of block     */
    vector<int> L_; /* i.e. block(i,j) starts at global location (K[i],L[i]) */
                    /* and of size (K[i+1]-K[i] x L[i+1]-L[i])               */

  public:

    Sp_mat(int M, int N) :
      num_rows_(M),         /* default construction */
      num_cols_(N),
      num_nonzeros_(0),

      void_(0),
      nnew_(1),
      open_(0),
      valid_(0),

      unit_diag_(0),
      complex_(0),
      real_(0),
      single_precision_(0),
      double_precision_(0),
      upper_triangular_(0),
      lower_triangular_(0),
      upper_symmetric_(0),
      lower_symmetric_(0),
      upper_hermitian_(0),
      lower_hermitian_(0),
      general_(0),
      one_base_(0),
      Mb_(0),
      Nb_(0),
      k_(0),
      l_(0),
      rowmajor_(0),
      colmajor_(0),
      opt_regular_(0),
      opt_irregular_(1),
      opt_block_(0),
      opt_unassembled_(0),
      K_(),
      L_()
      {}


    int& num_rows()           { return num_rows_; }
    int& num_cols()           { return num_cols_; }
    int& num_nonzeros()         { return num_nonzeros_;}

    int num_rows() const        { return num_rows_; }
    int num_cols() const        { return num_cols_; }
    int num_nonzeros() const      { return num_nonzeros_;}

    int is_one_base() const     { return (one_base_ ? 1 : 0); }
    int is_zero_base() const    { return (one_base_ ? 0 : 1); }
    int is_void() const         { return void_; }
    int is_new() const          { return nnew_; }
    int is_open() const         { return open_; }
    int is_valid() const        { return valid_; }

    int is_unit_diag() const    { return unit_diag_; }
    int is_complex() const        { return complex_;}
    int is_real() const         { return real_;}
    int is_single_precision() const   { return single_precision_;}
    int is_double_precision() const   { return double_precision_;}
    int is_upper_triangular() const   { return upper_triangular_;}
    int is_lower_triangular() const   { return lower_triangular_;}
    int is_triangular() const     { return upper_triangular_ ||
                           lower_triangular_; }


    int is_lower_symmetric() const    { return lower_symmetric_; }
    int is_upper_symmetric() const    { return upper_symmetric_; }
    int is_symmetric() const      { return upper_symmetric_ ||
                           lower_symmetric_; }

    int is_lower_hermitian() const    { return lower_hermitian_; }
    int is_upper_hermitian() const    { return upper_hermitian_; }
    int is_hermitian() const  { return lower_hermitian_ ||
                                       upper_hermitian_; }
    int is_general() const { return !( is_hermitian() || is_symmetric()) ; }

    int is_lower_storage() const { return is_lower_triangular() ||
                                          is_lower_symmetric()  ||
                                          is_lower_hermitian() ; }

    int is_upper_storage() const { return is_upper_triangular() ||
                                          is_upper_symmetric()  ||
                                          is_upper_hermitian() ; }


    int is_opt_regular() const { return opt_regular_; }
    int is_opt_irregular() const { return opt_irregular_; }
    int is_opt_block() const { return opt_block_;}
    int is_opt_unassembled() const { return opt_unassembled_;}

    int K(int i) const { return (k_ ? i*k_ : K_[i] ); }
    int L(int i) const { return (l_ ? i*l_ : L_[i] ); }

    int is_rowmajor() const { return rowmajor_; }
    int is_colmajor() const { return colmajor_; }


    void set_one_base()   { one_base_ = 1; }
    void set_zero_base()  { one_base_ = 0; }


    void set_void()       { void_ = 1;  nnew_ = open_ =  valid_ = 0;}
    void set_new()        { nnew_ = 1;  void_ = open_ =  valid_ = 0;}
    void set_open()       { open_ = 1;  void_ = nnew_  = valid_ = 0;}
    void set_valid()      { valid_ = 1; void_ = nnew_ =  open_ = 0; }


    void set_unit_diag()    { unit_diag_ = 1;}
    void set_complex()        {complex_ = 1; }
    void set_real()         { real_ = 1; }
    void set_single_precision()   { single_precision_ = 1; }
    void set_double_precision()   { double_precision_ = 1; }
    void set_upper_triangular()   { upper_triangular_ = 1; }
    void set_lower_triangular()   { lower_triangular_ = 1; }
    void set_upper_symmetric()  { upper_symmetric_ = 1; }
    void set_lower_symmetric()  { lower_symmetric_ = 1; }
    void set_upper_hermitian()  { upper_hermitian_ = 1; }
    void set_lower_hermitian()  { lower_hermitian_ = 1; }


    void set_const_block_parameters(int Mb, int Nb, int k, int l)
    {
      Mb_ = Mb;
      Nb_ = Nb;
      k_ = k;
      l_ = l;
    }


    void set_var_block_parameters(int Mb, int Nb, const int *k, const int *l)
    {
      Mb_ = Mb;
      Nb_ = Nb;
      k_ = 0;
      l_ = 0;

      K_.resize(Mb+1);
      K_[0] = 0;
      for (int i=0; i<Mb; i++)
        K_[i+1] = k[i] + K_[i];

      L_.resize(Nb+1);
      L_[0] = 0;
      for (int j=0; j<Mb; j++)
        K_[j+1] = k[j] + K_[j];
    }


    virtual int end_construction()
    {
      if (is_open() || is_new())
      {
        set_valid();

        return 0;
      }
      else
        ERROR_RETURN(1);
    }
    virtual void print() const;

    virtual void destroy() {};

    virtual ~Sp_mat() {};

};


template <class T>
class TSp_mat : public Sp_mat
{
  private:
    vector< vector< pair<T, int> > > S;
    vector<T> diag;                 /* optional diag if matrix is
                        triangular. Created
                        at end_construction() phase */
  private:

    inline T sp_dot_product( const vector< pair<T, int> > &r,
        const T* x, int incx ) const
    {
        T sum(0);

        if (incx == 1)
        {
          for ( typename vector< pair<T,int> >::const_iterator p = r.begin();
            p < r.end(); p++)
          {
            //sum = sum + p->first * x[p->second];
            sum += p->first * x[p->second];
          }
        }
        else /* incx != 1 */
        {
          for ( typename vector< pair<T,int> >::const_iterator p = r.begin();
            p < r.end(); p++)
           {
            //sum = sum + p->first * x[p->second * incx];
            sum += p->first * x[p->second * incx];
            }
        }

        return sum;
  }

    inline T sp_conj_dot_product( const vector< pair<T, int> > &r,
        const T* x, int incx ) const
    {
        T sum(0);

        if (incx == 1)
        {
          for ( typename vector< pair<T,int> >::const_iterator p = r.begin();
            p < r.end(); p++)
          {
            sum += conj(p->first) * x[p->second];
          }
        }
        else /* incx != 1 */
        {
          for ( typename vector< pair<T,int> >::const_iterator p = r.begin();
            p < r.end(); p++)
           {
            //sum = sum + p->first * x[p->second * incx];
            sum += conj(p->first) * x[p->second * incx];
            }
        }

        return sum;
  }


  inline void sp_axpy( const T& alpha, const vector< pair<T,int> > &r,
      T*  y, int incy) const
  {
    if (incy == 1)
    {
      for (typename vector< pair<T,int> >::const_iterator p = r.begin();
          p < r.end(); p++)
       y[p->second] += alpha * p->first;

    }
    else /* incy != 1 */
    {
    for (typename vector< pair<T,int> >::const_iterator p = r.begin();
        p < r.end(); p++)
      y[incy * p->second] += alpha * p->first;
    }
  }

  inline void sp_conj_axpy( const T& alpha, const vector< pair<T,int> > &r,
      T*  y, int incy) const
  {
    if (incy == 1)
    {
      for (typename vector< pair<T,int> >::const_iterator p = r.begin();
          p < r.end(); p++)
       y[p->second] += alpha * conj(p->first);

    }
    else /* incy != 1 */
    {
    for (typename vector< pair<T,int> >::const_iterator p = r.begin();
        p < r.end(); p++)
      y[incy * p->second] += alpha * conj(p->first);
    }
  }

  void mult_diag(const T& alpha, const T* x, int incx, T* y, int incy)
      const
  {
    const T* X = x;
    T* Y = y;
    typename vector<T>::const_iterator d= diag.begin();
    for ( ; d < diag.end(); X+=incx, d++, Y+=incy)
    {
      *Y += alpha * *d * *X;
    }
  }

  void mult_conj_diag(const T& alpha, const T* x, int incx, T* y, int incy)
      const
  {
    const T* X = x;
    T* Y = y;
    typename vector<T>::const_iterator d= diag.begin();
    for ( ; d < diag.end(); X+=incx, d++, Y+=incy)
    {
      *Y += alpha * conj(*d) * *X;
    }
  }


  void nondiag_mult_vec(const T& alpha, const T* x, int incx,
      T* y, int incy) const
  {

    int M = num_rows();

    if (incy == 1)
    {
      for (int i=0; i<M; i++)
        y[i] += alpha * sp_dot_product(S[i], x, incx);
    }
    else
    {
      for (int i=0; i<M; i++)
        y[i * incy] += alpha * sp_dot_product(S[i], x, incx);
    }
  }

  void nondiag_mult_vec_conj(const T& alpha, const T* x, int incx,
      T* y, int incy) const
  {

    int M = num_rows();

    if (incy == 1)
    {
      for (int i=0; i<M; i++)
        y[i] += alpha * sp_conj_dot_product(S[i], x, incx);
    }
    else
    {
      for (int i=0; i<M; i++)
        y[i * incy] += alpha * sp_conj_dot_product(S[i], x, incx);
    }
  }

  void nondiag_mult_vec_transpose(const T& alpha, const T* x, int incx,
      T* y, int incy) const
  {
    /* saxpy: y += (alpha * x[i]) row[i]  */

    int M = num_rows();
    const T* X = x;
    for (int i=0; i<M; i++, X += incx)
      sp_axpy( alpha * *X, S[i], y, incy);
  }

  void nondiag_mult_vec_conj_transpose(const T& alpha, const T* x, int incx,
      T* y, int incy) const
  {
    /* saxpy: y += (alpha * x[i]) row[i]  */

    int M = num_rows();
    const T* X = x;
    for (int i=0; i<M; i++, X += incx)
      sp_conj_axpy( alpha * *X, S[i], y, incy);
  }

  void mult_vec(const T& alpha, const T* x, int incx, T* y, int incy)
      const
  {
    nondiag_mult_vec(alpha, x, incx, y, incy);

    if (is_triangular() || is_symmetric())
      mult_diag(alpha, x, incx, y, incy);

    if (is_symmetric())
      nondiag_mult_vec_transpose(alpha, x, incx, y, incy);
  }


  void mult_vec_transpose(const T& alpha, const T* x, int incx, T* y,
      int incy) const
  {

    nondiag_mult_vec_transpose(alpha, x, incx, y, incy);

    if (is_triangular() || is_symmetric())
      mult_diag(alpha, x, incx, y, incy);

    if (is_symmetric())
      nondiag_mult_vec(alpha, x, incx, y, incy);
  }


  void mult_vec_conj_transpose(const T& alpha, const T* x, int incx, T* y,
      int incy) const
  {

    nondiag_mult_vec_conj_transpose(alpha, x, incx, y, incy);

    if (is_triangular() || is_symmetric())
      mult_conj_diag(alpha, x, incx, y, incy);

    if (is_symmetric())
      nondiag_mult_vec_conj(alpha, x, incx, y, incy);
  }







  int triangular_solve(T alpha, T* x, int incx ) const
  {
    if (alpha == (T) 0.0)
      ERROR_RETURN(1);

    if ( ! is_triangular() )
      ERROR_RETURN(1);

    int N = num_rows();

    if (is_lower_triangular())
    {
        for (int i=0, ii=0; i<N; i++, ii += incx)
        {
            x[ii] = (x[ii] - sp_dot_product(S[i], x, incx)) / diag[i];
        }
       if (alpha != (T) 1.0)
       {
        for (int i=0, ii=0; i<N; i++, ii += incx)
            x[ii] /= alpha;
       }
    }
    else if (is_upper_triangular())
    {

      for (int i=N-1, ii=(N-1)*incx ;   0<=i ;    i--, ii-=incx)
      {
         x[ii] = (x[ii] - sp_dot_product(S[i],x, incx)) / diag[i];
      }
      if (alpha != (T) 1.0)
      {
        for (int i=N-1, ii=(N-1)*incx ;   0<=i ;    i--, ii-=incx)
          x[ii] /= alpha;
      }

    }
    else
        ERROR_RETURN(1);

    return 0;
  }

  int transpose_triangular_solve(T alpha, T* x, int incx) const
  {
    if ( ! is_triangular())
      return -1;

    int N = num_rows();

    if (is_lower_triangular())
    {

      for (int j=N-1, jj=(N-1)*incx; 0<=j; j--, jj -= incx)
      {
        x[jj] /= diag[j] ;
        sp_axpy( -x[jj], S[j], x, incx);
      }
      if (alpha != (T) 1.0)
      {
        for (int jj=(N-1)*incx; 0<=jj; jj -=incx)
          x[jj] /= alpha;
      }
    }
    else if (is_upper_triangular())
    {

      for (int j=0, jj=0; j<N; j++, jj += incx)
      {
        x[jj] /= diag[j];
        sp_axpy(- x[jj], S[j], x, incx);
      }
      if (alpha != (T) 1.0)
      {
        for (int jj=(N-1)*incx; 0<=jj; jj -=incx)
          x[jj] /= alpha;
      }
    }
    else
         ERROR_RETURN(1);

    return 0;
  }



  int transpose_triangular_conj_solve(T alpha, T* x, int incx) const
  {
    if ( ! is_triangular())
      return -1;

    int N = num_rows();

    if (is_lower_triangular())
    {

      for (int j=N-1, jj=(N-1)*incx; 0<=j; j--, jj -= incx)
      {
        x[jj] /= conj(diag[j]) ;
        sp_conj_axpy( -x[jj], S[j], x, incx);
      }
      if (alpha != (T) 1.0)
      {
        for (int jj=(N-1)*incx; 0<=jj; jj -=incx)
          x[jj] /= alpha;
      }
    }
    else if (is_upper_triangular())
    {

      for (int j=0, jj=0; j<N; j++, jj += incx)
      {
        x[jj] /= conj(diag[j]);
        sp_conj_axpy(- x[jj], S[j], x, incx);
      }
      if (alpha != (T) 1.0)
      {
        for (int jj=(N-1)*incx; 0<=jj; jj -=incx)
          x[jj] /= alpha;
      }
    }
    else
         ERROR_RETURN(1);

    return 0;
  }




 public:

  inline T& val(pair<T, int> &VP) { return VP.first; }
  inline int& col_index(pair<T,int> &VP) { return VP.second; }

  inline const T& val(pair<T, int> const &VP) const { return VP.first; }
  inline int col_index(pair<T,int> const &VP) const { return VP.second; }


  TSp_mat( int M, int N) : Sp_mat(M,N), S(M), diag() {}

  /* custom ctors */
  /* row select ctor */
  typedef typename vector< vector< pair<T, int> > >::const_iterator Siter_t;
  TSp_mat( int N, Siter_t first, Siter_t last) : Sp_mat(last-first,N), S(first, last), diag() { }

   /* csr ctor */
   TSp_mat( int M, int N, T *x, int *row, int *col ) : Sp_mat(M, N), S(M), diag()
   {
      for (int i = 0; i < M; i++)
         for (int j = 0; j < (row[i+1] - row[i]); j++)
         {
            S[i].push_back(make_pair(x[row[i] + j], col[row[i] + j]));
         }

      num_nonzeros() = row[M];
   }

  void destroy()
  {
    // set vector sizes to zero
    (vector<T>(0)).swap(diag);
    (vector< vector< pair<T, int> > > (0) ).swap(S);
  }

/**

    This function is the entry point for all of the insert routines in
    this implementation.  It fills the sparse matrix, one entry at a time.
    If matrix is declared unit_diagonal, then inserting any diagonal
    values is ignored.  If it is symmetric (upper/lower) or triangular
    (upper/lower) inconsistent values are not caught.  (That is, entries
    into the upper region of a lower triangular matrix is not reported.)

    [NOTE: the base is determined at the creation phase, and can be determined
    by testing whether  BLAS_usgp(A, blas_one_base) returns 1.  If it returns 0,
    then offsets are zero based.]

    @param val  the numeric value of entry A(i,j)
    @param i  the row index of A(i,j)
    @param j  the column index of A(i,j)

    @return 0 if succesful, 1 otherwise
*/
  int insert_entry(T val, int i, int j)
  {
    if (is_one_base())
    {
      i--;
      j--;
    }

    /* make sure the indices are in range */
    ASSERT_RETURN(i >= 0, 1);
    ASSERT_RETURN(i < num_rows(), 1);
    ASSERT_RETURN(j >= 0, 1);
    ASSERT_RETURN(j < num_cols(), 1);

    /* allocate space for the diagonal, if this is the first time
     * trying to insert values.
    */
    if (is_new())
    {
      set_open();

      if (is_triangular() || is_symmetric())
      {
        diag.resize(num_rows());

        if (is_unit_diag())
        {
          for (unsigned int ii=0; ii< diag.size(); ii++)
              diag[ii] = T(1.0);
        }
        else
        {
          for (unsigned int ii=0; ii< diag.size(); ii++)
              diag[ii] = (T) 0.0;
        }
      }

    }
    if (is_open())
    {

      if (i==j && (is_triangular() || is_symmetric() || is_hermitian()) )
      {
        if (!is_unit_diag())
        {
          diag[i] += val;
        }
        else /* if unit diagonal */
        {
          if (val != (T) 1)
            ERROR_RETURN(0);    /* tries to insert non-unit diagonal */
        }

        if (is_upper_storage() && i > j)
            ERROR_RETURN(0);    /* tries to fill lower-triangular region */
        else

          if (is_lower_storage() && i < j)
            ERROR_RETURN(0);  /* tries to fill upper-triangular region */

      }
      else
      {
        S[i].push_back( make_pair(val, j) );
      }

      num_nonzeros() ++;
    }


    return 0;
  }

  int insert_entries( int nz, const T* Val, const int *I, const int *J)
  {
    for (int i=0; i<nz; i++)
    {
      insert_entry(Val[i], I[i], J[i]) ;
    }
    return 0;

  }

  int insert_row(int k, int nz, const T* Val, const int *J)
  {
    for (int i=0; i<nz; i++)
      insert_entry(Val[i], k, J[i]);
    return 0;
  }

  int insert_col(int k, int nz, const T* Val, const int *I)
  {
    for (int i=0; i<nz; i++)
      insert_entry(Val[i], I[i], k);
    return 0;
  }

  int insert_block(const T* Val, int row_stride,
        int col_stride, int bi, int bj)
  {
    /* translate from block index to global indices */
    int Iend = K(bi+1);
    int Jend = L(bj+1);
    for (int i=K(bi), r=0; i<Iend; i++, r += row_stride)
      for (int j=L(bi); j<Jend; j++, r += col_stride)
        insert_entry( Val[r], i, j );

    return 0;
  }

  int end_construction()
  {
    return Sp_mat::end_construction();
  }



  int usmv(enum blas_trans_type transa, const T& alpha, const  T* x , int incx,
    T* y, int incy) const
  {

  ASSERT_RETURN(is_valid(), -1);

  if (transa == blas_no_trans)
    mult_vec(alpha, x, incx, y, incy);
  else
  if (transa == blas_conj_trans)
    mult_vec_conj_transpose(alpha, x, incx, y, incy);
  else
  if ( transa == blas_trans)
    mult_vec_transpose(alpha, x, incx, y, incy);
  else
    ERROR_RETURN(1);


    return 0;

  }


  int usmm(enum blas_order_type ordera, enum blas_trans_type transa,
    int nrhs, const T& alpha, const  T* b, int ldb, T* C, int ldC) const
  {
    if (ordera == blas_rowmajor)
    {
      /* for each column of C, perform a mat_vec */
      for (int i=0; i<nrhs; i++)
      {
        usmv( transa, alpha, &b[i], ldb, &C[i], ldC );
      }
      return 0;
    }
    else
    if (ordera == blas_colmajor)
    {
      /* for each column of C, perform a mat_vec */
      for (int i=0; i<nrhs; i++)
      {
        usmv( transa, alpha, &b[i*ldb], 1, &C[i*ldC], 1 );
      }
      return 0;
    }
    else
      ERROR_RETURN(1);
  }

  int ussv( enum blas_trans_type transa, const T& alpha,  T* x, int incx) const
  {
      if (transa == blas_trans)
        return transpose_triangular_solve(alpha, x, incx);
      else
      if (transa == blas_conj_trans)
        return transpose_triangular_conj_solve(alpha, x, incx);
      else
      if (transa == blas_no_trans)
        return triangular_solve(alpha, x, incx);
      else
        ERROR_RETURN(1);


  }

  int ussm( enum blas_order_type ordera, enum blas_trans_type transa, int nrhs,
      const T& alpha, T* C, int ldC) const
  {
    if (ordera == blas_rowmajor)
    {
      /* for each column of C, perform a usmv */
      for (int i=0; i<nrhs; i++)
      {
        ussv(
            transa, alpha, &C[i], ldC );
      }
      return 0;
    }
    else
    if (ordera == blas_colmajor)
    {
      /* for each column of C, perform a mat_vec */
      for (int i=0; i<nrhs; i++)
      {
        ussv( transa, alpha, &C[i*ldC], 1 );
      }
      return 0;
    }
    else
      ERROR_RETURN(1);
  }



  void print() const
  {
    Sp_mat::print();  /* print matrix header info */

    /* if there is actual data, print out contents */
    for (int i=0; i<num_rows(); i++)
      for (unsigned int j=0; j< S[i].size(); j++)
        cout << i << "    " << col_index(S[i][j]) <<
              "        "  << val(S[i][j]) << "\n";

    /* if matrix is triangular, print out diagonals */
    if (is_upper_triangular() || is_lower_triangular())
    {
      for (unsigned int i=0; i< diag.size(); i++)
        cout << i << "    " << i << "     " << diag[i] << "\n";
    }
  }

  /* custom, these functions assume irregular structure */

   inline void colmeans( T *means )
   {
      for (typename vector< vector< pair<T, int> > >::const_iterator i = S.begin(); i < S.end(); i++)
         for (typename vector< pair<T, int> >::const_iterator v = i->begin(); v < i->end(); v++)
            means[v->second] += v->first;

      for (int j = 0; j < num_cols(); j++)
         means[j] /= num_rows();
   }

   inline void rowmeans( T *means )
   {
      for (int i = 0; i < num_rows(); i++)
         for (typename vector< pair<T, int> >::const_iterator v = S[i].begin(); v < S[i].end(); v++)
            means[i] += v->first;

      for (int i = 0; i < num_rows(); i++)
         means[i] /= num_cols();
   }

   inline void colvars( T *vars )
   {
      vector<T> means(num_cols());
      vector<int> nnz(num_cols());

      colmeans(&means[0]);

      for (typename vector< vector< pair<T, int> > >::const_iterator i = S.begin(); i < S.end(); i++)
         for (typename vector< pair<T, int> >::const_iterator v = i->begin(); v < i->end(); v++)
         {
            const T x = v->first - means[v->second];
            vars[v->second] += x * x;
            nnz[v->second]++;
         }

      for (int j = 0; j < num_cols(); j++)
      {
         vars[j] += means[j] * means[j] * (num_rows() - nnz[j]);
         vars[j] /= num_rows();
      }
   }

   inline void rowvars( T * vars )
   {
      vector<T> means(num_rows());
      vector<int> nnz(num_rows());

      rowmeans(&means[0]);

      for (int i = 0; i < num_rows(); i++)
         for (typename vector< pair<T, int> >::const_iterator v = S[i].begin(); v < S[i].end(); v++)
         {
            const T x = v->first - means[i];
            vars[i] += x * x;
            nnz[i]++;
         }

      for (int i = 0; i < num_rows(); i++)
      {
         vars[i] += means[i] * means[i] * (num_cols() - nnz[i]);
         vars[i] /= num_cols();
      }
   }

   inline void scalecols( T *s )
   {
      for (typename vector< vector< pair<T, int> > >::iterator i = S.begin(); i < S.end(); i++)
         for (typename vector< pair<T, int> >::iterator v = i->begin(); v < i->end(); v++)
            v->first *= s[v->second];
   }

   inline void scalerows( T *s )
   {
      for (int i = 0; i < num_rows(); i++)
         for (typename vector< pair<T, int> >::iterator v = S[i].begin(); v < S[i].end(); v++)
            v->first *= s[i];
   }

   inline TSp_mat<T> *rowsubset( int first_row, int nrow )
   {
      TSp_mat<T> *copy = new TSp_mat<T>(num_cols(), S.begin() + first_row, S.begin() + first_row + nrow);
      copy->set_open();

      int nnz = 0;

      for (typename vector< vector< pair<T, int> > >::iterator i = copy->S.begin(); i < copy->S.end(); i++)
         nnz += i->size();

      copy->num_nonzeros() = nnz;
      return copy;
   }

   inline void setelement( int row, int col, T val )
   {
      vector< pair<T, int> > &rowv = S[row];
      int i;

      for (i = 0; i < rowv.size(); i++)
      {
         if (rowv[i].second < col)
         {
            continue;
         }
         else if (rowv[i].second == col) /* nnz entry affected */
         {
            if (val == 0) /* FIXME: won't work for complex instanciation */
            {
               rowv.erase(rowv.begin() + i);
               num_nonzeros()--;
            }
            else
            {
               rowv[i].first = val;
            }
            break;
         }
         else if (rowv[i].second > col && val != 0) /* zero entry affected, FIXME: complex!!! */
         {
            rowv.insert(rowv.begin() + i, make_pair(val, col));
            num_nonzeros()++;
            break;
         }
      }

      if (i == rowv.size()) /* insertion at end of row */
      {
         rowv.push_back(make_pair(val, col));
         num_nonzeros()++;
      }
   }

   inline T &getelement( int row, int col )
   {
      vector< pair<T, int> > &rowv = S[row];
      int i;

      for (i = 0; i < rowv.size(); i++)
      {
         if (rowv[i].second < col)
         {
            continue;
         }
         else if (rowv[i].second == col) /* nnz entry affected */
         {
            return rowv[i].first;
         }
         else if (rowv[i].second > col) /* zero entry affected */
         {
            rowv.insert(rowv.begin() + i, make_pair(T(), col));
            num_nonzeros()++;
            return rowv[i].first;
         }
      }

      if (i == rowv.size()) /* insertion at end of row */
      {
         rowv.push_back(make_pair(T(), col));
         num_nonzeros()++;
         return rowv[i].first;
      }
   }

   /* returns pointer to element if allocated, NULL otherwise */
   inline T *getelementp(int row, int col)
   {
      vector< pair<T, int> > &rowv = S[row];

      for (int i = 0; i < rowv.size(); i++)
      {
         if (rowv[i].second < col)
         {
            continue;
         }
         else if (rowv[i].second == col) /* nnz entry affected */
         {
            return &rowv[i].first;
         }
         else if (rowv[i].second > col) /* zero entry affected */
         {
            return NULL;
         }
      }
   }

   /* rowmajor only */
   void usgemm(enum blas_side_type sidea, enum blas_trans_type transa, enum blas_trans_type transb,
      int nohs, const T &alpha, const T *B, int ldB, const T &beta, T *C, int ldC)
   {
      const bool ra = (sidea == blas_right_side);
      const bool ta = (transa == blas_trans);
      const bool tb = (transb == blas_trans);

      const int idxb = (ra == tb ? 1 : ldB);
      const int incb = (ra == tb ? ldB : 1);
      const int idxc = (ra ? ldC : 1); /* if A is left, compute C's columns, otherwise C's rows */
      const int incc = (ra ? 1 : ldC); /* with matrix-vetor products */
      const int lenc = (ra == ta ? num_rows() : num_cols()) * nohs;

      /* set A's transposition to have A's rows (no trans) or cols (trans) used in usmv */
      transa = (ra == ta ? blas_no_trans : blas_trans);

      if (beta == 0)
         memset(&C[0], 0, lenc * sizeof(T));
      else
         for (int i = 0; i < lenc; i++)
            C[i] *= beta;

      for (int i = 0; i < nohs; i++)
         usmv( transa, alpha, &B[i * idxb], incb, &C[i * idxc], incc );
   }

   int usgemma(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb,
      int nrhs, const T &alpha, const T *B, int ldB, const T &beta, T *C, int ldC)
   {
      const int idxb = (transb == blas_no_trans ? 1 : ldB);
      const int incb = (transb == blas_no_trans ? ldB : 1);
      const int lenc = (transa == blas_no_trans ? num_rows() : num_cols()) * nrhs;

      for (int i = 0; i < lenc; i++)
         C[i] *= beta;

      if (order == blas_rowmajor)
         for (int i = 0; i < nrhs; i++)
            usmv( transa, alpha, &B[i * idxb], incb, &C[i], ldC );
      else
         for (int i = 0; i < nrhs; i++)
            usmv( transa, alpha, &B[i * incb], idxb, &C[i * ldC], 1 );
      return 0;
   }

   int usgemmb(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb,
      int nlhs, const T &alpha, const T *A, int ldA, const T &beta, T *C, int ldC)
   {
      const int idxa = (transa == blas_no_trans ? ldA : 1);
      const int inca = (transa == blas_no_trans ? 1 : ldA);
      const int lenc = (transb == blas_no_trans ? num_cols() : num_rows()) * nlhs;

      /* we need to build dot products with B's columns (not rows as usmv defaults);
       * passing inverse transb to usmv recovers */
      transb = (transb == blas_no_trans ? blas_trans : blas_no_trans);

      for (int i = 0; i < lenc; i++)
         C[i] *= beta;

      if (order == blas_rowmajor)
         for (int i = 0; i < nlhs; i++)
            usmv( transb, alpha, &A[i * idxa], inca, &C[i], ldC );
      else
         for (int i = 0; i < nlhs; i++)
            usmv( transb, alpha, &A[i * inca], idxa, &C[i * ldC], 1 );
      return 0;
   }
};


typedef TSp_mat<float> FSp_mat;
typedef TSp_mat<double> DSp_mat;
typedef TSp_mat<complex<float> > CSp_mat;
typedef TSp_mat<complex<double> > ZSp_mat;



void table_print();
void print(int A);

}
/* namespace */


namespace NIST_SPBLAS
{

static vector<Sp_mat *> Table;
static unsigned int Table_active_matrices = 0;
int Table_insert(Sp_mat* S);
int Table_remove(unsigned int i);

/*
  finds an empty slot in global sparse marix table, and fills it with
  given entry.  Returns -1 if no spot found, or table is corrupt.
*/
int Table_insert(Sp_mat* S)
{
  if (Table_active_matrices <= Table.size())
  {
    Table.push_back(S);
    Table_active_matrices++;
    return Table.size() - 1;
  }
  else
  {

    /* there is an available slot; find it. */
    for (unsigned int i=0; i<Table.size(); i++)
    {
      if (Table[i] == NULL)
      {
        Table[i] = S;
        Table_active_matrices++;
        return i;
      }
    }
  }

  return -1;
}

/*
  removes an exisiting sparse matrix from global table.  Returns 0, if
  successfull, 1 otherwise.
*/
int Table_remove(unsigned int i)
{
  if (i < Table.size() && Table[i] != NULL)
  {
    Table[i] = NULL;
    Table_active_matrices--;
    return 0;
  }
  else
    return -1;
}



void Sp_mat::print() const
{


  cout << "State : " <<
    (is_void() ? "void" :
     is_new()  ? "new" :
     is_open() ? "open" :
     is_valid() ? "valid" : "unknown") << "\n";

  cout << "M = " <<  num_rows() <<  "  N = " << num_cols() <<
        "  nz = " << num_nonzeros() << "\n";

#define yesno(exp) ( (exp) ? "yes" : "no" )

  cout << "real: "     << yesno(is_real()) << "\n";
  cout << "complex: "  << yesno(is_complex()) << "\n";
  cout << "double "    << yesno(is_double_precision()) << "\n";
  cout << "single "    << yesno(is_single_precision()) << "\n";

  cout << "upper_triangular: " << yesno(is_upper_triangular()) << "\n";
  cout << "lower_triangular: " << yesno(is_lower_triangular()) << "\n";

  cout << "regular:    " << yesno(is_opt_regular()) << "\n";
  cout << "irregular:  " << yesno(is_opt_irregular()) << "\n";
  cout << "block:      " << yesno(is_opt_block()) << "\n";
  cout << "unassembled:" << yesno(is_opt_unassembled()) << "\n";

#undef yesno
}

void table_print()
{
  cout << "Table has " << Table.size() << " element(s). \n";
  for (unsigned int i=0; i< Table.size(); i++)
  {
    if (Table[i] != 0)
    {
      cout << "***** Table[" << i << "]: \n";
      Table[i]->print();
      cout << "\n\n";
    }
  }
}

void print(int A)
{
  cout << "\n";
  Table[A]->print();
  cout << "\n";
}



}
/* namespace NIST_SPBLAS */

using namespace std;
using namespace NIST_SPBLAS;





/* Level 1 */

/* these macros are useful for creating some consistency between the
   various precisions and floating point types.
*/

typedef float    FLOAT;
typedef double   DOUBLE;
typedef complex<float> COMPLEX_FLOAT;
typedef complex<double> COMPLEX_DOUBLE;



typedef float          SPBLAS_FLOAT_IN;
typedef double         SPBLAS_DOUBLE_IN;
typedef const void *   SPBLAS_COMPLEX_FLOAT_IN;
typedef const void *   SPBLAS_COMPLEX_DOUBLE_IN;

typedef float *  SPBLAS_FLOAT_OUT;
typedef double * SPBLAS_DOUBLE_OUT;
typedef void *   SPBLAS_COMPLEX_FLOAT_OUT;
typedef void *   SPBLAS_COMPLEX_DOUBLE_OUT;



typedef float *  SPBLAS_FLOAT_IN_OUT;
typedef double * SPBLAS_DOUBLE_IN_OUT;
typedef void *   SPBLAS_COMPLEX_FLOAT_IN_OUT;
typedef void *   SPBLAS_COMPLEX_DOUBLE_IN_OUT;

typedef const float *  SPBLAS_VECTOR_FLOAT_IN;
typedef const double * SPBLAS_VECTOR_DOUBLE_IN;
typedef const void *   SPBLAS_VECTOR_COMPLEX_FLOAT_IN;
typedef const void *   SPBLAS_VECTOR_COMPLEX_DOUBLE_IN;

typedef float *  SPBLAS_VECTOR_FLOAT_OUT;
typedef double * SPBLAS_VECTOR_DOUBLE_OUT;
typedef void *   SPBLAS_VECTOR_COMPLEX_FLOAT_OUT;
typedef void *   SPBLAS_VECTOR_COMPLEX_DOUBLE_OUT;

typedef float *  SPBLAS_VECTOR_FLOAT_IN_OUT;
typedef double * SPBLAS_VECTOR_DOUBLE_IN_OUT;
typedef void *   SPBLAS_VECTOR_COMPLEX_FLOAT_IN_OUT;
typedef void *   SPBLAS_VECTOR_COMPLEX_DOUBLE_IN_OUT;



#define SPBLAS_TO_FLOAT_IN(x)   x
#define SPBLAS_TO_DOUBLE_IN(x)  x
#define SPBLAS_TO_COMPLEX_FLOAT_IN(x) \
        (* reinterpret_cast<const complex<float> *>(x))
#define SPBLAS_TO_COMPLEX_DOUBLE_IN(x)  \
        (* reinterpret_cast<const complex<double> *>(x))

#define SPBLAS_TO_FLOAT_OUT(x)  x
#define SPBLAS_TO_DOUBLE_OUT(x) x
#define SPBLAS_TO_COMPLEX_FLOAT_OUT(x)  reinterpret_cast<complex<float> *>(x)
#define SPBLAS_TO_COMPLEX_DOUBLE_OUT(x) reinterpret_cast<complex<double> *>(x)

#define SPBLAS_TO_FLOAT_IN_OUT(x)   x
#define SPBLAS_TO_DOUBLE_IN_OUT(x)  x
#define SPBLAS_TO_COMPLEX_FLOAT_IN_OUT(x)  reinterpret_cast<complex<float> *>(x)
#define SPBLAS_TO_COMPLEX_DOUBLE_IN_OUT(x) reinterpret_cast<complex<double>*>(x)


#define SPBLAS_TO_VECTOR_DOUBLE_IN(x)   x
#define SPBLAS_TO_VECTOR_FLOAT_IN(x)  x
#define SPBLAS_TO_VECTOR_COMPLEX_FLOAT_IN(x) \
                          reinterpret_cast<const complex<float>*>(x)
#define SPBLAS_TO_VECTOR_COMPLEX_DOUBLE_IN(x) \
                          reinterpret_cast<const complex<double>*>(x)

#define SPBLAS_TO_VECTOR_DOUBLE_OUT(x)  x
#define SPBLAS_TO_VECTOR_FLOAT_OUT(x)   x
#define SPBLAS_TO_VECTOR_COMPLEX_FLOAT_OUT(x) \
                          reinterpret_cast<complex<float>*>(x)
#define SPBLAS_TO_VECTOR_COMPLEX_DOUBLE_OUT(x) \
                          reinterpret_cast<complex<double>*>(x)

#define SPBLAS_TO_VECTOR_DOUBLE_IN_OUT(x)   x
#define SPBLAS_TO_VECTOR_FLOAT_IN_OUT(x)  x
#define SPBLAS_TO_VECTOR_COMPLEX_FLOAT_IN_OUT(x) \
                          reinterpret_cast<complex<float>*>(x)
#define SPBLAS_TO_VECTOR_COMPLEX_DOUBLE_IN_OUT(x) \
                          reinterpret_cast<complex<double>*>(x)






#define BLAS_FLOAT_NAME(routine_name) BLAS_s##routine_name
#define BLAS_DOUBLE_NAME(routine_name) BLAS_d##routine_name
#define BLAS_COMPLEX_FLOAT_NAME(routine_name) BLAS_c##routine_name
#define BLAS_COMPLEX_DOUBLE_NAME(routine_name) BLAS_z##routine_name


#define TSp_MAT_SET_FLOAT(A) {A->set_single_precision(); A->set_real();}
#define TSp_MAT_SET_DOUBLE(A) {A->set_double_precision(); A->set_real();}
#define TSp_MAT_SET_COMPLEX_FLOAT(A) {A->set_single_precision(); A->set_complex();}
#define TSp_MAT_SET_COMPLEX_DOUBLE(A) {A->set_double_precision(); A->set_complex();}


        /*------------------------------------*/
        /* Non-precision Sparse BLAS routines */
        /*------------------------------------*/

/* -------- */
/*  USSP()  */
/* -------- */
int BLAS_ussp(blas_sparse_matrix A, int pname)
{
    Sp_mat *S = Table[A];


                                /* Note: these are returns, in the case */
                                /* statement, so "break" is not needed.  */
    switch (pname)
    {
        case (blas_zero_base) : S->set_zero_base(); break;
        case (blas_one_base)  : S->set_one_base(); break;

        case (blas_unit_diag) : S->set_unit_diag(); break;
        case (blas_complex)   : S->set_complex(); break;
        case (blas_real)      : S->set_real(); break;

        case (blas_single_precision) : S->set_single_precision(); break;
        case (blas_double_precision) : S->set_double_precision(); break;


        case (blas_lower_triangular) : S->set_lower_triangular(); break;
        case (blas_upper_triangular) : S->set_upper_triangular(); break;

        case (blas_lower_symmetric) : S->set_lower_symmetric(); break;
        case (blas_upper_symmetric) : S->set_upper_symmetric(); break;

        case (blas_lower_hermitian) : S->set_lower_hermitian(); break;
        case (blas_upper_hermitian) : S->set_upper_hermitian(); break;


                                        /* optimizations not used */
        case (blas_regular )        :
        case (blas_irregular)       :
        case (blas_block)           :
        case (blas_unassembled)     :    return 0;

        default:                return -1;  /* invalid property */
    }

    return 0;

}

/* -------- */
/*  USGP()  */
/* -------- */

int BLAS_usgp(blas_sparse_matrix A, int pname)
{
  Sp_mat *S = Table[A];

    switch (pname)
    {
        case (blas_num_rows)          : return S->num_rows();
        case (blas_num_cols)          : return S->num_cols();
        case (blas_num_nonzeros)      : return S->num_nonzeros();

        case (blas_complex)           : return S->is_complex();
        case (blas_real)              : return S->is_real();
        case (blas_single_precision)  : return S->is_single_precision();
        case (blas_double_precision)  : return S->is_double_precision();


        case (blas_lower_triangular) : return S->is_lower_triangular(); break;
        case (blas_upper_triangular) : return S->is_upper_triangular(); break;

        case (blas_general)           : return S->is_general();
        case (blas_symmetric)         : return S->is_symmetric();
        case (blas_hermitian)         : return S->is_hermitian();

        case (blas_zero_base) : return S->is_zero_base();
        case (blas_one_base) : return  S->is_one_base();


        case (blas_rowmajor)          : return S->is_rowmajor();
        case (blas_colmajor)          : return S->is_colmajor();
        case (blas_new_handle)        : return S->is_new();
        case (blas_valid_handle)      : return S->is_valid();
        case (blas_open_handle)       : return S->is_open();
        case (blas_invalid_handle)    : return S->is_void();

        case (blas_regular)           : return S->is_opt_regular();
        case (blas_irregular)         : return S->is_opt_irregular();
        case (blas_block)             : return S->is_opt_block();
        case (blas_unassembled)       : return S->is_opt_unassembled();

        default:                return -1;  /* invalid property */
    }
}

/* -------- */
/*  USDS()  */
/* -------- */
int BLAS_usds(int A)
{
  Sp_mat *S  = Table[A];

  S->destroy();
  Table_remove(A);

  return 0;
}




/* --------------------------- */
/*  Level 1 generic routines   */
/* --------------------------- */

/* dummy routines for real version of usdot to compile. */

inline const double& conj(const double &x)
{
  return x;
}

inline const float& conj(const float &x)
{
  return x;
}


template <class T>
void BLAS_xusdot( enum blas_conj_type conj_flag, int nz,
    const T *x,  const int *index,  const T *y, int incy,
    T *r, enum blas_base_type index_base)
{

  T t(0);

  if (index_base == blas_one_base)
    y -= incy;

  if (conj_flag == blas_no_conj)
  {
    for (int i=0; i<nz; i++)
      t += x[i] * y[index[i]*incy];
  }
  else
    for (int i=0; i<nz; i++)
      t += conj(x[i]) * y[index[i]*incy];


  *r = t;
}


template <class T>
void BLAS_xusaxpy(int nz, T alpha, const T *x,
    const int *index, T *y, int incy,
    enum blas_base_type index_base)
{

  if (index_base == blas_one_base)
    y -= incy;

  for (int i=0; i<nz; i++)
  {
     y[index[i]*incy] +=  (alpha * x[i]);
  }
}

template <class T>
void BLAS_xusga( int nz, const T *y, int incy, T *x, const int *indx,
              enum blas_base_type index_base )
{
  if (index_base == blas_one_base)
    y -= incy;

  for (int i=0; i<nz; i++)
    x[i] = y[indx[i]*incy];

}

template <class T>
void BLAS_xusgz( int nz, T *y, int incy, T *x, const int *indx,
              enum blas_base_type index_base )
{
  if (index_base == blas_one_base)
    y -= incy;

  for (int i=0; i<nz; i++)
  {
    x[i] = y[indx[i]*incy];
    y[indx[i]*incy] = (T) 0.0;
  }

}


template <class T>
void BLAS_xussc(int nz, const T *x, T *y, int incy, const int *index,
    enum blas_base_type index_base)
{
  if (index_base == blas_one_base)
    y -= incy;

  for (int i=0; i<nz; i++)
    y[index[i]*incy] = x[i];

}

/* --------------------------- */
/* Level 2&3 generic precision */
/* --------------------------- */


template <class T>
int BLAS_xuscr_insert_entry(blas_sparse_matrix A,  const T& val, int i, int j)
{
  return ((TSp_mat<T> *)Table[A])->insert_entry(val, i, j);

}

template <class T>
int BLAS_xuscr_insert_entries(blas_sparse_matrix A, int nz, const T* Val,
      const int* I, const int *J)
{
  return ((TSp_mat<T>*) Table[A])->insert_entries(nz, Val, I, J);
}


template <class T>
int BLAS_xuscr_insert_col(blas_sparse_matrix A, int j, int nz, const T* Val,
      const int* indx)
{
  return ((TSp_mat<T>*) Table[A])->insert_col(j, nz, Val, indx);
}

template <class T>
int BLAS_xuscr_insert_row(blas_sparse_matrix A, int i, int nz, const T* Val,
      const int* indx)
{
  return ((TSp_mat<T>*) Table[A])->insert_row(i, nz, Val, indx);
}

template <class T>
int BLAS_xuscr_insert_clique(blas_sparse_matrix A, int k, int l, const T* Val,
      const int row_stride, const int col_stride, const int *indx,
      const int *jndx)
{
  return ((TSp_mat<T>*) Table[A])->insert_clique(k, l, Val, row_stride,
            col_stride, indx, jndx);
}

template <class T>
int BLAS_xuscr_insert_block(blas_sparse_matrix A, const T* Val,
      const int row_stride, const int col_stride, int bi, int bj )
{
  return ((TSp_mat<T>*) Table[A])->insert_block(Val,
        row_stride, col_stride, bi, bj);
}

inline int BLAS_xuscr_end(blas_sparse_matrix A)
{
  return (Table[A])->end_construction();
}


template <class T>
int BLAS_xusmv(enum blas_trans_type transa, const T& alpha,
    blas_sparse_matrix A, const T *x, int incx, T *y, int incy )
{
  TSp_mat<T> *M = (TSp_mat<T> *) Table[A];

  ASSERT_RETURN(M->is_valid(), 1);

  return M->usmv(transa, alpha, x, incx, y, incy);
}


template <class T>
int BLAS_xusmm(enum blas_order_type ordera, enum blas_trans_type transa,
    int nrhs, const T& alpha, blas_sparse_matrix A,
    const T *B, int ldB, T* C, int ldC)
{
  TSp_mat<T> *M = (TSp_mat<T> *) Table[A];

  ASSERT_RETURN(M->is_valid(), 1);

  return M->usmm(ordera, transa, nrhs, alpha, B, ldB, C, ldC);
}


template <class T>
int BLAS_xussv(enum blas_trans_type transa, const T& alpha,
    blas_sparse_matrix A, T *x, int incx)
{
  TSp_mat<T> *M =
        (TSp_mat<T> *) Table[A];

  ASSERT_RETURN(M->is_valid(), 1);

  return M->ussv(transa, alpha, x, incx);
}

template <class T>
int BLAS_xussm(enum blas_order_type orderA, enum blas_trans_type transa,
    int nrhs, const T& alpha, blas_sparse_matrix A, T *C, int ldC)
{
  TSp_mat<T> *M =
        (TSp_mat<T> *) Table[A];

  ASSERT_RETURN(M->is_valid(), 1);

  return M->ussm(orderA, transa, nrhs, alpha, C, ldC);
}

/*   --- end of generic rouintes ---- */


/*********/
/*  ----  double  Level 1 rouintes ----- */
/*********/

 void BLAS_DOUBLE_NAME(usdot)(
    enum blas_conj_type conj_flag,
    int  nz,
    SPBLAS_VECTOR_DOUBLE_IN x,
    const int *index,
    SPBLAS_VECTOR_DOUBLE_IN y,
    int incy,
    SPBLAS_DOUBLE_OUT r,
    enum blas_base_type index_base)
{
   BLAS_xusdot(conj_flag, nz,
          SPBLAS_TO_VECTOR_DOUBLE_IN( x ), index,
          SPBLAS_TO_VECTOR_DOUBLE_IN( y ), incy,
          SPBLAS_TO_DOUBLE_OUT( r ), index_base);
}

 void BLAS_DOUBLE_NAME(usaxpy)(
    int nz,
    SPBLAS_DOUBLE_IN alpha,
    SPBLAS_VECTOR_DOUBLE_IN x,
    const int *index,
    SPBLAS_VECTOR_DOUBLE_IN_OUT y,
    int incy,
    enum blas_base_type index_base)
{
  BLAS_xusaxpy(nz, SPBLAS_TO_DOUBLE_IN( alpha ),
      SPBLAS_TO_VECTOR_DOUBLE_IN( x ), index,
      SPBLAS_TO_VECTOR_DOUBLE_IN_OUT( y ),
      incy, index_base);
}

 void BLAS_DOUBLE_NAME(usga)(
    int nz,
    SPBLAS_VECTOR_DOUBLE_IN y,
    int incy,
    SPBLAS_VECTOR_DOUBLE_IN_OUT x,
    const int *indx,
    enum blas_base_type index_base )
{
  BLAS_xusga( nz, SPBLAS_TO_VECTOR_DOUBLE_IN( y ), incy,
        SPBLAS_TO_VECTOR_DOUBLE_IN_OUT( x ), indx, index_base);

}

 void BLAS_DOUBLE_NAME(usgz)(
    int nz,
    SPBLAS_VECTOR_DOUBLE_IN_OUT y,
    int incy,
    SPBLAS_VECTOR_DOUBLE_OUT x,
    const int *indx,
    enum blas_base_type index_base )
{
  BLAS_xusgz(nz, SPBLAS_TO_DOUBLE_IN_OUT(y ), incy, SPBLAS_TO_DOUBLE_OUT(  x ),
    indx, index_base);
}


 void BLAS_DOUBLE_NAME(ussc)(
    int nz,
    SPBLAS_VECTOR_DOUBLE_IN x,
    SPBLAS_VECTOR_DOUBLE_IN_OUT y,
    int incy,
    const int *index,
    enum blas_base_type index_base)
{
  BLAS_xussc(nz, SPBLAS_TO_VECTOR_DOUBLE_IN( x ),
  SPBLAS_TO_DOUBLE_IN_OUT( y ), incy, index,
  index_base);
}

/*  DOUBLE Level 2/3 creation routines */

int BLAS_DOUBLE_NAME(uscr_begin)(int M, int N)
{
  TSp_mat<DOUBLE> *A = new TSp_mat<DOUBLE>(M, N);
  TSp_MAT_SET_DOUBLE(A);

  return Table_insert(A);
}

blas_sparse_matrix BLAS_DOUBLE_NAME(uscr_block_begin)(
    int Mb, int Nb, int k, int l )
{
  TSp_mat<DOUBLE> *A = new TSp_mat<DOUBLE>(Mb*k, Nb*l);

  TSp_MAT_SET_DOUBLE(A);
  A->set_const_block_parameters(Mb, Nb, k, l);

  return Table_insert(A);
}

blas_sparse_matrix BLAS_DOUBLE_NAME(uscr_variable_block_begin)(
    int Mb, int Nb, const int *k, const int *l )
{
  TSp_mat<DOUBLE> *A = new TSp_mat<DOUBLE>(
                accumulate(k, k+Mb, 0), accumulate(l, l+Nb, 0) );

  TSp_MAT_SET_DOUBLE(A);
  A->set_var_block_parameters(Mb, Nb, k, l);

  return Table_insert(A);

}

/*  DOUBLE Level 2/3 insertion routines */

int BLAS_DOUBLE_NAME(uscr_insert_entry)(
    blas_sparse_matrix A, SPBLAS_DOUBLE_IN val, int i, int j )
{
  return BLAS_xuscr_insert_entry(A, SPBLAS_TO_DOUBLE_IN( val ), i, j);
}

int BLAS_DOUBLE_NAME(uscr_insert_entries)(
    blas_sparse_matrix A, int nz,
    SPBLAS_VECTOR_DOUBLE_IN val,
    const int *indx, const int *jndx )
{
  return BLAS_xuscr_insert_entries(A, nz, SPBLAS_TO_VECTOR_DOUBLE_IN( val ), indx, jndx);
}

int BLAS_DOUBLE_NAME(uscr_insert_col)(
    blas_sparse_matrix A, int j, int nz,
    SPBLAS_VECTOR_DOUBLE_IN val, const int *indx )
{
  return BLAS_xuscr_insert_col(A, j, nz, SPBLAS_TO_VECTOR_DOUBLE_IN( val ), indx);
}

int BLAS_DOUBLE_NAME(uscr_insert_row)(
  blas_sparse_matrix A, int i, int nz,
  SPBLAS_VECTOR_DOUBLE_IN val, const int *indx );

int BLAS_DOUBLE_NAME(uscr_insert_clique)(
    blas_sparse_matrix A,
    const int k,
    const int l,
    SPBLAS_VECTOR_DOUBLE_IN val,
    const int row_stride,
    const int col_stride,
    const int *indx,
    const int *jndx );

int BLAS_DOUBLE_NAME(uscr_insert_block)(
    blas_sparse_matrix A,
    SPBLAS_VECTOR_DOUBLE_IN val,
    int row_stride,
    int col_stride,
    int i, int j )
{
  return BLAS_xuscr_insert_block(
        A, SPBLAS_TO_VECTOR_DOUBLE_IN( val ),
        row_stride, col_stride, i, j);
}

int BLAS_DOUBLE_NAME(uscr_end)(blas_sparse_matrix A)
{
  return BLAS_xuscr_end(A);
}

/*  DOUBLE Level 2/3 computational routines */

 int BLAS_DOUBLE_NAME(usmv)(enum
    blas_trans_type transa,
    SPBLAS_DOUBLE_IN alpha,
    blas_sparse_matrix A,
    SPBLAS_VECTOR_DOUBLE_IN x,
    int incx,
    SPBLAS_VECTOR_DOUBLE_IN_OUT y,
    int incy )
{
  return BLAS_xusmv(
      transa,   SPBLAS_TO_DOUBLE_IN( alpha ), A,
      SPBLAS_TO_VECTOR_DOUBLE_IN( x ), incx,
      SPBLAS_TO_VECTOR_DOUBLE_IN_OUT( y ), incy);
}

int BLAS_DOUBLE_NAME(usmm)(
    enum blas_order_type order,
    enum blas_trans_type transa,
    int nrhs,
    SPBLAS_DOUBLE_IN alpha,
    blas_sparse_matrix A,
    SPBLAS_VECTOR_DOUBLE_IN b,
    int ldb,
    SPBLAS_VECTOR_DOUBLE_IN_OUT c,
    int ldc )
{
  return BLAS_xusmm(
      order, transa, nrhs,
      SPBLAS_TO_DOUBLE_IN( alpha), A,
      SPBLAS_TO_VECTOR_DOUBLE_IN(b), ldb,
      SPBLAS_TO_VECTOR_DOUBLE_IN_OUT( c ), ldc);
}

int BLAS_DOUBLE_NAME(ussv)(
    enum blas_trans_type transa,
    SPBLAS_DOUBLE_IN alpha,
    blas_sparse_matrix A,
    SPBLAS_VECTOR_DOUBLE_IN_OUT x,
    int incx )
{
  return BLAS_xussv( transa,
        SPBLAS_TO_DOUBLE_IN( alpha ), A,
        SPBLAS_TO_VECTOR_DOUBLE_IN_OUT( x ),
        incx);
}


int BLAS_DOUBLE_NAME(ussm)(
    enum blas_order_type order,
    enum blas_trans_type transt,
    int nrhs,
    SPBLAS_DOUBLE_IN alpha,
    blas_sparse_matrix A,
    SPBLAS_VECTOR_DOUBLE_IN_OUT b,
    int ldb )
{
  return BLAS_xussm(order, transt, nrhs,
      SPBLAS_TO_DOUBLE_IN( alpha ), A,
      SPBLAS_TO_VECTOR_DOUBLE_IN_OUT( b ), ldb);
}


/*  ----   end of DOUBLE routines -------  */




 void BLAS_COMPLEX_DOUBLE_NAME(usdot)(
    enum blas_conj_type conj_flag,
    int  nz,
    SPBLAS_VECTOR_COMPLEX_DOUBLE_IN x,
    const int *index,
    SPBLAS_VECTOR_COMPLEX_DOUBLE_IN y,
    int incy,
    SPBLAS_COMPLEX_DOUBLE_OUT r,
    enum blas_base_type index_base)
{
   BLAS_xusdot(conj_flag, nz,
          SPBLAS_TO_VECTOR_COMPLEX_DOUBLE_IN( x ), index,
          SPBLAS_TO_VECTOR_COMPLEX_DOUBLE_IN( y ), incy,
          SPBLAS_TO_COMPLEX_DOUBLE_OUT( r ), index_base);
}

 void BLAS_COMPLEX_DOUBLE_NAME(usaxpy)(
    int nz,
    SPBLAS_COMPLEX_DOUBLE_IN alpha,
    SPBLAS_VECTOR_COMPLEX_DOUBLE_IN x,
    const int *index,
    SPBLAS_VECTOR_COMPLEX_DOUBLE_IN_OUT y,
    int incy,
    enum blas_base_type index_base)
{
  BLAS_xusaxpy(nz, SPBLAS_TO_COMPLEX_DOUBLE_IN( alpha ),
      SPBLAS_TO_VECTOR_COMPLEX_DOUBLE_IN( x ), index,
      SPBLAS_TO_VECTOR_COMPLEX_DOUBLE_IN_OUT( y ),
      incy, index_base);
}

 void BLAS_COMPLEX_DOUBLE_NAME(usga)(
    int nz,
    SPBLAS_VECTOR_COMPLEX_DOUBLE_IN y,
    int incy,
    SPBLAS_VECTOR_COMPLEX_DOUBLE_IN_OUT x,
    const int *indx,
    enum blas_base_type index_base )
{
  BLAS_xusga( nz, SPBLAS_TO_VECTOR_COMPLEX_DOUBLE_IN( y ), incy,
        SPBLAS_TO_VECTOR_COMPLEX_DOUBLE_IN_OUT( x ), indx, index_base);

}

 void BLAS_COMPLEX_DOUBLE_NAME(usgz)(
    int nz,
    SPBLAS_VECTOR_COMPLEX_DOUBLE_IN_OUT y,
    int incy,
    SPBLAS_VECTOR_COMPLEX_DOUBLE_OUT x,
    const int *indx,
    enum blas_base_type index_base )
{
  BLAS_xusgz(nz, SPBLAS_TO_COMPLEX_DOUBLE_IN_OUT(y ), incy, SPBLAS_TO_COMPLEX_DOUBLE_OUT(  x ),
    indx, index_base);
}


 void BLAS_COMPLEX_DOUBLE_NAME(ussc)(
    int nz,
    SPBLAS_VECTOR_COMPLEX_DOUBLE_IN x,
    SPBLAS_VECTOR_COMPLEX_DOUBLE_IN_OUT y,
    int incy,
    const int *index,
    enum blas_base_type index_base)
{
  BLAS_xussc(nz, SPBLAS_TO_VECTOR_COMPLEX_DOUBLE_IN( x ),
  SPBLAS_TO_COMPLEX_DOUBLE_IN_OUT( y ), incy, index,
  index_base);
}

/*  COMPLEX_DOUBLE Level 2/3 creation routines */

int BLAS_COMPLEX_DOUBLE_NAME(uscr_begin)(int M, int N)
{
  TSp_mat<COMPLEX_DOUBLE> *A = new TSp_mat<COMPLEX_DOUBLE>(M, N);
  TSp_MAT_SET_COMPLEX_DOUBLE(A);

  return Table_insert(A);
}

blas_sparse_matrix BLAS_COMPLEX_DOUBLE_NAME(uscr_block_begin)(
    int Mb, int Nb, int k, int l )
{
  TSp_mat<COMPLEX_DOUBLE> *A = new TSp_mat<COMPLEX_DOUBLE>(Mb*k, Nb*l);

  TSp_MAT_SET_COMPLEX_DOUBLE(A);
  A->set_const_block_parameters(Mb, Nb, k, l);

  return Table_insert(A);
}

blas_sparse_matrix BLAS_COMPLEX_DOUBLE_NAME(uscr_variable_block_begin)(
    int Mb, int Nb, const int *k, const int *l )
{
  TSp_mat<COMPLEX_DOUBLE> *A = new TSp_mat<COMPLEX_DOUBLE>(
                accumulate(k, k+Mb, 0), accumulate(l, l+Nb, 0) );

  TSp_MAT_SET_COMPLEX_DOUBLE(A);
  A->set_var_block_parameters(Mb, Nb, k, l);

  return Table_insert(A);

}

/*  COMPLEX_DOUBLE Level 2/3 insertion routines */

int BLAS_COMPLEX_DOUBLE_NAME(uscr_insert_entry)(
    blas_sparse_matrix A, SPBLAS_COMPLEX_DOUBLE_IN val, int i, int j )
{
  return BLAS_xuscr_insert_entry(A, SPBLAS_TO_COMPLEX_DOUBLE_IN( val ), i, j);
}

int BLAS_COMPLEX_DOUBLE_NAME(uscr_insert_entries)(
    blas_sparse_matrix A, int nz,
    SPBLAS_VECTOR_COMPLEX_DOUBLE_IN val,
    const int *indx, const int *jndx )
{
  return BLAS_xuscr_insert_entries(A, nz, SPBLAS_TO_VECTOR_COMPLEX_DOUBLE_IN( val ), indx, jndx);
}

int BLAS_COMPLEX_DOUBLE_NAME(uscr_insert_col)(
    blas_sparse_matrix A, int j, int nz,
    SPBLAS_VECTOR_COMPLEX_DOUBLE_IN val, const int *indx )
{
  return BLAS_xuscr_insert_col(A, j, nz, SPBLAS_TO_VECTOR_COMPLEX_DOUBLE_IN( val ), indx);
}

int BLAS_COMPLEX_DOUBLE_NAME(uscr_insert_row)(
  blas_sparse_matrix A, int i, int nz,
  SPBLAS_VECTOR_COMPLEX_DOUBLE_IN val, const int *indx );

int BLAS_COMPLEX_DOUBLE_NAME(uscr_insert_clique)(
    blas_sparse_matrix A,
    const int k,
    const int l,
    SPBLAS_VECTOR_COMPLEX_DOUBLE_IN val,
    const int row_stride,
    const int col_stride,
    const int *indx,
    const int *jndx );

int BLAS_COMPLEX_DOUBLE_NAME(uscr_insert_block)(
    blas_sparse_matrix A,
    SPBLAS_VECTOR_COMPLEX_DOUBLE_IN val,
    int row_stride,
    int col_stride,
    int i, int j )
{
  return BLAS_xuscr_insert_block(
        A, SPBLAS_TO_VECTOR_COMPLEX_DOUBLE_IN( val ),
        row_stride, col_stride, i, j);
}

int BLAS_COMPLEX_DOUBLE_NAME(uscr_end)(blas_sparse_matrix A)
{
  return BLAS_xuscr_end(A);
}

/*  COMPLEX_DOUBLE Level 2/3 computational routines */

 int BLAS_COMPLEX_DOUBLE_NAME(usmv)(enum
    blas_trans_type transa,
    SPBLAS_COMPLEX_DOUBLE_IN alpha,
    blas_sparse_matrix A,
    SPBLAS_VECTOR_COMPLEX_DOUBLE_IN x,
    int incx,
    SPBLAS_VECTOR_COMPLEX_DOUBLE_IN_OUT y,
    int incy )
{
  return BLAS_xusmv(
      transa,   SPBLAS_TO_COMPLEX_DOUBLE_IN( alpha ), A,
      SPBLAS_TO_VECTOR_COMPLEX_DOUBLE_IN( x ), incx,
      SPBLAS_TO_VECTOR_COMPLEX_DOUBLE_IN_OUT( y ), incy);
}

int BLAS_COMPLEX_DOUBLE_NAME(usmm)(
    enum blas_order_type order,
    enum blas_trans_type transa,
    int nrhs,
    SPBLAS_COMPLEX_DOUBLE_IN alpha,
    blas_sparse_matrix A,
    SPBLAS_VECTOR_COMPLEX_DOUBLE_IN b,
    int ldb,
    SPBLAS_VECTOR_COMPLEX_DOUBLE_IN_OUT c,
    int ldc )
{
  return BLAS_xusmm(
      order, transa, nrhs,
      SPBLAS_TO_COMPLEX_DOUBLE_IN( alpha), A,
      SPBLAS_TO_VECTOR_COMPLEX_DOUBLE_IN(b), ldb,
      SPBLAS_TO_VECTOR_COMPLEX_DOUBLE_IN_OUT( c ), ldc);
}

int BLAS_COMPLEX_DOUBLE_NAME(ussv)(
    enum blas_trans_type transa,
    SPBLAS_COMPLEX_DOUBLE_IN alpha,
    blas_sparse_matrix A,
    SPBLAS_VECTOR_COMPLEX_DOUBLE_IN_OUT x,
    int incx )
{
  return BLAS_xussv( transa,
        SPBLAS_TO_COMPLEX_DOUBLE_IN( alpha ), A,
        SPBLAS_TO_VECTOR_COMPLEX_DOUBLE_IN_OUT( x ),
        incx);
}


int BLAS_COMPLEX_DOUBLE_NAME(ussm)(
    enum blas_order_type order,
    enum blas_trans_type transt,
    int nrhs,
    SPBLAS_COMPLEX_DOUBLE_IN alpha,
    blas_sparse_matrix A,
    SPBLAS_VECTOR_COMPLEX_DOUBLE_IN_OUT b,
    int ldb )
{
  return BLAS_xussm(order, transt, nrhs,
      SPBLAS_TO_COMPLEX_DOUBLE_IN( alpha ), A,
      SPBLAS_TO_VECTOR_COMPLEX_DOUBLE_IN_OUT( b ), ldb);
}




/*  ----   end of COMPLEX_COMPLEX_COMPLEX_DOUBLE routines -------  */


/*********/
/*  ----  double  Level 1 rouintes ----- */
/*********/

 void BLAS_FLOAT_NAME(usdot)(
    enum blas_conj_type conj_flag,
    int  nz,
    SPBLAS_VECTOR_FLOAT_IN x,
    const int *index,
    SPBLAS_VECTOR_FLOAT_IN y,
    int incy,
    SPBLAS_FLOAT_OUT r,
    enum blas_base_type index_base)
{
   BLAS_xusdot(conj_flag, nz,
          SPBLAS_TO_VECTOR_FLOAT_IN( x ), index,
          SPBLAS_TO_VECTOR_FLOAT_IN( y ), incy,
          SPBLAS_TO_FLOAT_OUT( r ), index_base);
}

 void BLAS_FLOAT_NAME(usaxpy)(
    int nz,
    SPBLAS_FLOAT_IN alpha,
    SPBLAS_VECTOR_FLOAT_IN x,
    const int *index,
    SPBLAS_VECTOR_FLOAT_IN_OUT y,
    int incy,
    enum blas_base_type index_base)
{
  BLAS_xusaxpy(nz, SPBLAS_TO_FLOAT_IN( alpha ),
      SPBLAS_TO_VECTOR_FLOAT_IN( x ), index,
      SPBLAS_TO_VECTOR_FLOAT_IN_OUT( y ),
      incy, index_base);
}

 void BLAS_FLOAT_NAME(usga)(
    int nz,
    SPBLAS_VECTOR_FLOAT_IN y,
    int incy,
    SPBLAS_VECTOR_FLOAT_IN_OUT x,
    const int *indx,
    enum blas_base_type index_base )
{
  BLAS_xusga( nz, SPBLAS_TO_VECTOR_FLOAT_IN( y ), incy,
        SPBLAS_TO_VECTOR_FLOAT_IN_OUT( x ), indx, index_base);

}

 void BLAS_FLOAT_NAME(usgz)(
    int nz,
    SPBLAS_VECTOR_FLOAT_IN_OUT y,
    int incy,
    SPBLAS_VECTOR_FLOAT_OUT x,
    const int *indx,
    enum blas_base_type index_base )
{
  BLAS_xusgz(nz, SPBLAS_TO_FLOAT_IN_OUT(y ), incy, SPBLAS_TO_FLOAT_OUT(  x ),
    indx, index_base);
}


 void BLAS_FLOAT_NAME(ussc)(
    int nz,
    SPBLAS_VECTOR_FLOAT_IN x,
    SPBLAS_VECTOR_FLOAT_IN_OUT y,
    int incy,
    const int *index,
    enum blas_base_type index_base)
{
  BLAS_xussc(nz, SPBLAS_TO_VECTOR_FLOAT_IN( x ),
  SPBLAS_TO_FLOAT_IN_OUT( y ), incy, index,
  index_base);
}

/*  FLOAT Level 2/3 creation routines */

int BLAS_FLOAT_NAME(uscr_begin)(int M, int N)
{
  TSp_mat<FLOAT> *A = new TSp_mat<FLOAT>(M, N);
  TSp_MAT_SET_FLOAT(A);

  return Table_insert(A);
}

blas_sparse_matrix BLAS_FLOAT_NAME(uscr_block_begin)(
    int Mb, int Nb, int k, int l )
{
  TSp_mat<FLOAT> *A = new TSp_mat<FLOAT>(Mb*k, Nb*l);

  TSp_MAT_SET_FLOAT(A);
  A->set_const_block_parameters(Mb, Nb, k, l);

  return Table_insert(A);
}

blas_sparse_matrix BLAS_FLOAT_NAME(uscr_variable_block_begin)(
    int Mb, int Nb, const int *k, const int *l )
{
  TSp_mat<FLOAT> *A = new TSp_mat<FLOAT>(
                accumulate(k, k+Mb, 0), accumulate(l, l+Nb, 0) );

  TSp_MAT_SET_FLOAT(A);
  A->set_var_block_parameters(Mb, Nb, k, l);

  return Table_insert(A);

}

/*  FLOAT Level 2/3 insertion routines */

int BLAS_FLOAT_NAME(uscr_insert_entry)(
    blas_sparse_matrix A, SPBLAS_FLOAT_IN val, int i, int j )
{
  return BLAS_xuscr_insert_entry(A, SPBLAS_TO_FLOAT_IN( val ), i, j);
}

int BLAS_FLOAT_NAME(uscr_insert_entries)(
    blas_sparse_matrix A, int nz,
    SPBLAS_VECTOR_FLOAT_IN val,
    const int *indx, const int *jndx )
{
  return BLAS_xuscr_insert_entries(A, nz, SPBLAS_TO_VECTOR_FLOAT_IN( val ), indx, jndx);
}

int BLAS_FLOAT_NAME(uscr_insert_col)(
    blas_sparse_matrix A, int j, int nz,
    SPBLAS_VECTOR_FLOAT_IN val, const int *indx )
{
  return BLAS_xuscr_insert_col(A, j, nz, SPBLAS_TO_VECTOR_FLOAT_IN( val ), indx);
}

int BLAS_FLOAT_NAME(uscr_insert_row)(
  blas_sparse_matrix A, int i, int nz,
  SPBLAS_VECTOR_FLOAT_IN val, const int *indx );

int BLAS_FLOAT_NAME(uscr_insert_clique)(
    blas_sparse_matrix A,
    const int k,
    const int l,
    SPBLAS_VECTOR_FLOAT_IN val,
    const int row_stride,
    const int col_stride,
    const int *indx,
    const int *jndx );

int BLAS_FLOAT_NAME(uscr_insert_block)(
    blas_sparse_matrix A,
    SPBLAS_VECTOR_FLOAT_IN val,
    int row_stride,
    int col_stride,
    int i, int j )
{
  return BLAS_xuscr_insert_block(
        A, SPBLAS_TO_VECTOR_FLOAT_IN( val ),
        row_stride, col_stride, i, j);
}

int BLAS_FLOAT_NAME(uscr_end)(blas_sparse_matrix A)
{
  return BLAS_xuscr_end(A);
}

/*  FLOAT Level 2/3 computational routines */

 int BLAS_FLOAT_NAME(usmv)(enum
    blas_trans_type transa,
    SPBLAS_FLOAT_IN alpha,
    blas_sparse_matrix A,
    SPBLAS_VECTOR_FLOAT_IN x,
    int incx,
    SPBLAS_VECTOR_FLOAT_IN_OUT y,
    int incy )
{
  return BLAS_xusmv(
      transa,   SPBLAS_TO_FLOAT_IN( alpha ), A,
      SPBLAS_TO_VECTOR_FLOAT_IN( x ), incx,
      SPBLAS_TO_VECTOR_FLOAT_IN_OUT( y ), incy);
}

int BLAS_FLOAT_NAME(usmm)(
    enum blas_order_type order,
    enum blas_trans_type transa,
    int nrhs,
    SPBLAS_FLOAT_IN alpha,
    blas_sparse_matrix A,
    SPBLAS_VECTOR_FLOAT_IN b,
    int ldb,
    SPBLAS_VECTOR_FLOAT_IN_OUT c,
    int ldc )
{
  return BLAS_xusmm(
      order, transa, nrhs,
      SPBLAS_TO_FLOAT_IN( alpha), A,
      SPBLAS_TO_VECTOR_FLOAT_IN(b), ldb,
      SPBLAS_TO_VECTOR_FLOAT_IN_OUT( c ), ldc);
}

int BLAS_FLOAT_NAME(ussv)(
    enum blas_trans_type transa,
    SPBLAS_FLOAT_IN alpha,
    blas_sparse_matrix A,
    SPBLAS_VECTOR_FLOAT_IN_OUT x,
    int incx )
{
  return BLAS_xussv( transa,
        SPBLAS_TO_FLOAT_IN( alpha ), A,
        SPBLAS_TO_VECTOR_FLOAT_IN_OUT( x ),
        incx);
}


int BLAS_FLOAT_NAME(ussm)(
    enum blas_order_type order,
    enum blas_trans_type transt,
    int nrhs,
    SPBLAS_FLOAT_IN alpha,
    blas_sparse_matrix A,
    SPBLAS_VECTOR_FLOAT_IN_OUT b,
    int ldb )
{
  return BLAS_xussm(order, transt, nrhs,
      SPBLAS_TO_FLOAT_IN( alpha ), A,
      SPBLAS_TO_VECTOR_FLOAT_IN_OUT( b ), ldb);
}


/*  ----   end of FLOAT routines -------  */




 void BLAS_COMPLEX_FLOAT_NAME(usdot)(
    enum blas_conj_type conj_flag,
    int  nz,
    SPBLAS_VECTOR_COMPLEX_FLOAT_IN x,
    const int *index,
    SPBLAS_VECTOR_COMPLEX_FLOAT_IN y,
    int incy,
    SPBLAS_COMPLEX_FLOAT_OUT r,
    enum blas_base_type index_base)
{
   BLAS_xusdot(conj_flag, nz,
          SPBLAS_TO_VECTOR_COMPLEX_FLOAT_IN( x ), index,
          SPBLAS_TO_VECTOR_COMPLEX_FLOAT_IN( y ), incy,
          SPBLAS_TO_COMPLEX_FLOAT_OUT( r ), index_base);
}

 void BLAS_COMPLEX_FLOAT_NAME(usaxpy)(
    int nz,
    SPBLAS_COMPLEX_FLOAT_IN alpha,
    SPBLAS_VECTOR_COMPLEX_FLOAT_IN x,
    const int *index,
    SPBLAS_VECTOR_COMPLEX_FLOAT_IN_OUT y,
    int incy,
    enum blas_base_type index_base)
{
  BLAS_xusaxpy(nz, SPBLAS_TO_COMPLEX_FLOAT_IN( alpha ),
      SPBLAS_TO_VECTOR_COMPLEX_FLOAT_IN( x ), index,
      SPBLAS_TO_VECTOR_COMPLEX_FLOAT_IN_OUT( y ),
      incy, index_base);
}

 void BLAS_COMPLEX_FLOAT_NAME(usga)(
    int nz,
    SPBLAS_VECTOR_COMPLEX_FLOAT_IN y,
    int incy,
    SPBLAS_VECTOR_COMPLEX_FLOAT_IN_OUT x,
    const int *indx,
    enum blas_base_type index_base )
{
  BLAS_xusga( nz, SPBLAS_TO_VECTOR_COMPLEX_FLOAT_IN( y ), incy,
        SPBLAS_TO_VECTOR_COMPLEX_FLOAT_IN_OUT( x ), indx, index_base);

}

 void BLAS_COMPLEX_FLOAT_NAME(usgz)(
    int nz,
    SPBLAS_VECTOR_COMPLEX_FLOAT_IN_OUT y,
    int incy,
    SPBLAS_VECTOR_COMPLEX_FLOAT_OUT x,
    const int *indx,
    enum blas_base_type index_base )
{
  BLAS_xusgz(nz, SPBLAS_TO_COMPLEX_FLOAT_IN_OUT(y ), incy, SPBLAS_TO_COMPLEX_FLOAT_OUT(  x ),
    indx, index_base);
}


 void BLAS_COMPLEX_FLOAT_NAME(ussc)(
    int nz,
    SPBLAS_VECTOR_COMPLEX_FLOAT_IN x,
    SPBLAS_VECTOR_COMPLEX_FLOAT_IN_OUT y,
    int incy,
    const int *index,
    enum blas_base_type index_base)
{
  BLAS_xussc(nz, SPBLAS_TO_VECTOR_COMPLEX_FLOAT_IN( x ),
  SPBLAS_TO_COMPLEX_FLOAT_IN_OUT( y ), incy, index,
  index_base);
}

/*  COMPLEX_FLOAT Level 2/3 creation routines */

int BLAS_COMPLEX_FLOAT_NAME(uscr_begin)(int M, int N)
{
  TSp_mat<COMPLEX_FLOAT> *A = new TSp_mat<COMPLEX_FLOAT>(M, N);
  TSp_MAT_SET_COMPLEX_FLOAT(A);

  return Table_insert(A);
}

blas_sparse_matrix BLAS_COMPLEX_FLOAT_NAME(uscr_block_begin)(
    int Mb, int Nb, int k, int l )
{
  TSp_mat<COMPLEX_FLOAT> *A = new TSp_mat<COMPLEX_FLOAT>(Mb*k, Nb*l);

  TSp_MAT_SET_COMPLEX_FLOAT(A);
  A->set_const_block_parameters(Mb, Nb, k, l);

  return Table_insert(A);
}

blas_sparse_matrix BLAS_COMPLEX_FLOAT_NAME(uscr_variable_block_begin)(
    int Mb, int Nb, const int *k, const int *l )
{
  TSp_mat<COMPLEX_FLOAT> *A = new TSp_mat<COMPLEX_FLOAT>(
                accumulate(k, k+Mb, 0), accumulate(l, l+Nb, 0) );

  TSp_MAT_SET_COMPLEX_FLOAT(A);
  A->set_var_block_parameters(Mb, Nb, k, l);

  return Table_insert(A);

}

/*  COMPLEX_FLOAT Level 2/3 insertion routines */

int BLAS_COMPLEX_FLOAT_NAME(uscr_insert_entry)(
    blas_sparse_matrix A, SPBLAS_COMPLEX_FLOAT_IN val, int i, int j )
{
  return BLAS_xuscr_insert_entry(A, SPBLAS_TO_COMPLEX_FLOAT_IN( val ), i, j);
}

int BLAS_COMPLEX_FLOAT_NAME(uscr_insert_entries)(
    blas_sparse_matrix A, int nz,
    SPBLAS_VECTOR_COMPLEX_FLOAT_IN val,
    const int *indx, const int *jndx )
{
  return BLAS_xuscr_insert_entries(A, nz, SPBLAS_TO_VECTOR_COMPLEX_FLOAT_IN( val ), indx, jndx);
}

int BLAS_COMPLEX_FLOAT_NAME(uscr_insert_col)(
    blas_sparse_matrix A, int j, int nz,
    SPBLAS_VECTOR_COMPLEX_FLOAT_IN val, const int *indx )
{
  return BLAS_xuscr_insert_col(A, j, nz, SPBLAS_TO_VECTOR_COMPLEX_FLOAT_IN( val ), indx);
}

int BLAS_COMPLEX_FLOAT_NAME(uscr_insert_row)(
  blas_sparse_matrix A, int i, int nz,
  SPBLAS_VECTOR_COMPLEX_FLOAT_IN val, const int *indx );

int BLAS_COMPLEX_FLOAT_NAME(uscr_insert_clique)(
    blas_sparse_matrix A,
    const int k,
    const int l,
    SPBLAS_VECTOR_COMPLEX_FLOAT_IN val,
    const int row_stride,
    const int col_stride,
    const int *indx,
    const int *jndx );

int BLAS_COMPLEX_FLOAT_NAME(uscr_insert_block)(
    blas_sparse_matrix A,
    SPBLAS_VECTOR_COMPLEX_FLOAT_IN val,
    int row_stride,
    int col_stride,
    int i, int j )
{
  return BLAS_xuscr_insert_block(
        A, SPBLAS_TO_VECTOR_COMPLEX_FLOAT_IN( val ),
        row_stride, col_stride, i, j);
}

int BLAS_COMPLEX_FLOAT_NAME(uscr_end)(blas_sparse_matrix A)
{
  return BLAS_xuscr_end(A);
}

/*  COMPLEX_FLOAT Level 2/3 computational routines */

 int BLAS_COMPLEX_FLOAT_NAME(usmv)(enum
    blas_trans_type transa,
    SPBLAS_COMPLEX_FLOAT_IN alpha,
    blas_sparse_matrix A,
    SPBLAS_VECTOR_COMPLEX_FLOAT_IN x,
    int incx,
    SPBLAS_VECTOR_COMPLEX_FLOAT_IN_OUT y,
    int incy )
{
  return BLAS_xusmv(
      transa,   SPBLAS_TO_COMPLEX_FLOAT_IN( alpha ), A,
      SPBLAS_TO_VECTOR_COMPLEX_FLOAT_IN( x ), incx,
      SPBLAS_TO_VECTOR_COMPLEX_FLOAT_IN_OUT( y ), incy);
}

int BLAS_COMPLEX_FLOAT_NAME(usmm)(
    enum blas_order_type order,
    enum blas_trans_type transa,
    int nrhs,
    SPBLAS_COMPLEX_FLOAT_IN alpha,
    blas_sparse_matrix A,
    SPBLAS_VECTOR_COMPLEX_FLOAT_IN b,
    int ldb,
    SPBLAS_VECTOR_COMPLEX_FLOAT_IN_OUT c,
    int ldc )
{
  return BLAS_xusmm(
      order, transa, nrhs,
      SPBLAS_TO_COMPLEX_FLOAT_IN( alpha), A,
      SPBLAS_TO_VECTOR_COMPLEX_FLOAT_IN(b), ldb,
      SPBLAS_TO_VECTOR_COMPLEX_FLOAT_IN_OUT( c ), ldc);
}

int BLAS_COMPLEX_FLOAT_NAME(ussv)(
    enum blas_trans_type transa,
    SPBLAS_COMPLEX_FLOAT_IN alpha,
    blas_sparse_matrix A,
    SPBLAS_VECTOR_COMPLEX_FLOAT_IN_OUT x,
    int incx )
{
  return BLAS_xussv( transa,
        SPBLAS_TO_COMPLEX_FLOAT_IN( alpha ), A,
        SPBLAS_TO_VECTOR_COMPLEX_FLOAT_IN_OUT( x ),
        incx);
}


int BLAS_COMPLEX_FLOAT_NAME(ussm)(
    enum blas_order_type order,
    enum blas_trans_type transt,
    int nrhs,
    SPBLAS_COMPLEX_FLOAT_IN alpha,
    blas_sparse_matrix A,
    SPBLAS_VECTOR_COMPLEX_FLOAT_IN_OUT b,
    int ldb )
{
  return BLAS_xussm(order, transt, nrhs,
      SPBLAS_TO_COMPLEX_FLOAT_IN( alpha ), A,
      SPBLAS_TO_VECTOR_COMPLEX_FLOAT_IN_OUT( b ), ldb);
}




/*  ----   end of COMPLEX_COMPLEX_COMPLEX_FLOAT routines -------  */


/* custom functions */

spmat_t create(int row, int col)
{
   blas_sparse_matrix A = BLAS_suscr_begin(row, col);
   BLAS_suscr_end(A);
   return (spmat_t) A;
}

void destroy(spmat_t A)
{
   BLAS_usds((blas_sparse_matrix) A);
}

template <class T>
inline void xcolmeans(blas_sparse_matrix A, T *means)
{
   TSp_mat<T> *const Ap = ((TSp_mat<T> *) Table[A]);

   memset(means, 0, Ap->num_cols() * sizeof(T));
   Ap->colmeans(means);
}

template <class T>
inline void xcolvars(blas_sparse_matrix A, T *vars)
{
   TSp_mat<T> *const Ap = ((TSp_mat<T> *) Table[A]);

   memset(vars, 0, Ap->num_cols() * sizeof(T));
   Ap->colvars(vars);
}

template <class T>
inline void xrowmeans(blas_sparse_matrix A, T *means)
{
   TSp_mat<T> *const Ap = ((TSp_mat<T> *) Table[A]);

   memset(means, 0, Ap->num_rows() * sizeof(T));
   Ap->rowmeans(means);
}

template <class T>
inline void xrowvars(blas_sparse_matrix A, T *vars)
{
   TSp_mat<T> *const Ap = ((TSp_mat<T> *) Table[A]);

   memset(vars, 0, Ap->num_rows() * sizeof(T));
   Ap->rowvars(vars);
}

template<class T>
inline void xscalecols(blas_sparse_matrix A, T *s)
{
   ((TSp_mat<T>*) Table[A])->scalecols(s);
}

template<class T>
inline void xscalerows(blas_sparse_matrix A, T *s)
{
   ((TSp_mat<T>*) Table[A])->scalerows(s);
}

template<class T>
inline void xsetelement( blas_sparse_matrix A, int row, int col, T val )
{
   ((TSp_mat<T>*) Table[A])->setelement(row, col, val);
}

template<class T>
inline T &xgetelement( blas_sparse_matrix A, int row, int col)
{
   return ((TSp_mat<T>*) Table[A])->getelement(row, col);
}

template<class T>
inline T *xgetelementp( blas_sparse_matrix A, int row, int col)
{
   return ((TSp_mat<T>*) Table[A])->getelementp(row, col);
}


template<class T>
inline void xusgemm(enum blas_side_type sidea, enum blas_trans_type transa, enum blas_trans_type transb,
   int nohs, const T &alpha, blas_sparse_matrix A, const T *B, int ldB, const T &beta, T *C, int ldC)
{
   ((TSp_mat<T>*) Table[A])->usgemm(sidea, transa, transb, nohs, alpha, B, ldB, beta, C, ldC);
}

template<class T>
inline void xusgemma(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb,
   int nrhs, const T &alpha, blas_sparse_matrix A, const T *B, int ldB, const T &beta, T *C, int ldC)
{
   ((TSp_mat<T>*) Table[A])->usgemma(order, transa, transb, nrhs, alpha, B, ldB, beta, C, ldC);
}

template<class T>
inline int xusgemmb(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb,
   int nlhs, const T &alpha, const T *A, int ldA, blas_sparse_matrix B, const T &beta, T *C, int ldC)
{
   return ((TSp_mat<T>*) Table[B])->usgemmb(order, transa, transb, nlhs, alpha, A, ldA, beta, C, ldC);
}

void scolmeans(spmat_t A, float *means)
{
   xcolmeans((blas_sparse_matrix) A, means);
}

void scolvars(spmat_t A, float *vars)
{
   xcolvars((blas_sparse_matrix) A, vars);
}

void srowmeans(spmat_t A, float *means)
{
   xrowmeans((blas_sparse_matrix) A, means);
}

void srowvars(spmat_t A, float *vars)
{
   xrowvars((blas_sparse_matrix) A, vars);
}

void sscalecols(spmat_t A, float *s)
{
   xscalecols((blas_sparse_matrix) A, s);
}

void sscalerows(spmat_t A, float *s)
{
   xscalerows((blas_sparse_matrix) A, s);
}

spmat_t srowsubset(spmat_t A, int first_row, int nrow)
{
  TSp_mat<float> *copy = ((TSp_mat<float>*) Table[(blas_sparse_matrix) A])->rowsubset(first_row, nrow);
  TSp_MAT_SET_FLOAT(copy);
  blas_sparse_matrix copy_handle = Table_insert(copy);
  BLAS_suscr_end(copy_handle);
  return (spmat_t) copy_handle;
}

spmat_t suscr_csr(int m, int n, float *val, int *col, int *ptr)
{
   TSp_mat<float> *A = new TSp_mat<float>(m, n, val, ptr, col);
   TSp_MAT_SET_FLOAT(A);
   blas_sparse_matrix A_handle = Table_insert(A);
   BLAS_suscr_end(A_handle);
   return (spmat_t) A_handle;
}

void ssetelement( spmat_t A, int row, int col, float val )
{
   xsetelement<float>((blas_sparse_matrix) A, row, col, val);
}

void ssetelement( spmat_t A, int idx, float val )
{
   TSp_mat<float> *Ap = ((TSp_mat<float>*) Table[(blas_sparse_matrix) A]);

   xsetelement<float>((blas_sparse_matrix) A, idx / Ap->num_cols(), idx % Ap->num_cols(), val);
}

float &sgetelement( spmat_t A, int row, int col)
{
   return xgetelement<float>((blas_sparse_matrix) A, row, col);
}

float &sgetelement( spmat_t A, int idx )
{
   TSp_mat<float> *Ap = ((TSp_mat<float>*) Table[(blas_sparse_matrix) A]);

   return xgetelement<float>((blas_sparse_matrix) A, idx / Ap->num_cols(), idx % Ap->num_cols());
}

float *sgetelementp( spmat_t A, int row, int col)
{
   return xgetelementp<float>((blas_sparse_matrix) A, row, col);
}

float *sgetelementp( spmat_t A, int idx )
{
   TSp_mat<float> *Ap = ((TSp_mat<float>*) Table[(blas_sparse_matrix) A]);

   return xgetelementp<float>((blas_sparse_matrix) A, idx / Ap->num_cols(), idx % Ap->num_cols());
}

inline enum blas_trans_type trans2nist(char t)
{
   switch (t)
   {
   case 'n':
   case 'N':
      return blas_no_trans;
   case 't':
   case 'T':
      return blas_trans;
   case 'c':
   case 'C':
      return blas_conj_trans;
   default:
      //bool valid_trans_char = false;
      //assert(valid_trans_char);
      return blas_no_trans; /* for compiler satisfaction */
   }
}

inline enum blas_side_type side2nist(char s)
{
   switch (s)
   {
   case 'l':
   case 'L':
      return blas_left_side;
   case 'r':
   case 'R':
      return blas_right_side;
   default:
      //bool valid_side_char = false;
      //assert(valid_side_char);
      return blas_left_side; /* for compiler satisfaction */
   }
}

void susgemm(char sidea, char transa, char transb, int nohs, const float &alpha, spmat_t A,
   const float *B, int ldB, const float &beta, float *C, int ldC)
{
   xusgemm(side2nist(sidea), trans2nist(transa), trans2nist(transb), nohs,
      alpha, (blas_sparse_matrix) A, B, ldB, beta, C, ldC);
}

void susgemma(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb,
   int nrhs, const float &alpha, blas_sparse_matrix A, const float *B, int ldB, const float &beta,
   float *C, int ldC)
{
   xusgemma(order, transa, transb, nrhs, alpha, A, B, ldB, beta, C, ldC);
}

void susgemmb(enum blas_order_type order, enum blas_trans_type transa, enum blas_trans_type transb,
   int nlhs, const float &alpha, const float *A, int ldA, blas_sparse_matrix B, const float &beta,
   float *C, int ldC)
{
   xusgemmb(order, transa, transb, nlhs, alpha, A, ldA, B, beta, C, ldC);
}

bool handle_valid(spmat_t A)
{
   return (A >= 0 && A < Table.size());
}

#else /* ifndef MKL */

/* this part wraps the MKL interface */
#include <mkl_spblas.h>
#include <mkl_trans.h>

#include <bitset>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <algorithm>
#include <iterator>

using std::map;
using std::bitset;
using std::make_pair;
using std::vector;
using std::runtime_error;

#include <iostream>
using std::cout;
using std::cerr;
using std::endl;

static const std::string nimpl("not implemented");
#define not_implemented runtime_error(nimpl)

class MklCsrMatrix
{
protected:
   int m; /* number of rows */
   int n; /* number of cols */

   /* three vectors implement the CSR format */
   vector<float> val; /* non-zero values in rowmajor */
   vector<int>   col; /* col[i] is the column index of val[i] */
   vector<int>   ptr; /* [ptr[i], ptr[i+1]) is the index range for row i, ptr has size m+1,
                       * ptr[0] is always 0, ptr[m] is the number of non-zero values in the matrix */

   static char *mdescr(void)
   {
      static char c[6] = {'g', 'l', 'n', 'c', 'x', 'x'};
      return c;
   }

   static char trans_inv(char t)
   {
      switch (t)
      {
      case 't':
      case 'T':
         return 'n';

      case 'n':
      case 'N':
         return 't';

      default:
         throw runtime_error("transposition param not supported");
         return t;
      }
   }

public:
   MklCsrMatrix(MklCsrMatrix &copy) : m(copy.m), n(copy.n), val(copy.val), col(copy.col), ptr(copy.ptr) {}
   MklCsrMatrix() : m(0), n(0), val(vector<float>()), col(vector<int>()), ptr(vector<int>()) {}
   MklCsrMatrix(int m, int n) : m(m), n(n), val(0), col(0), ptr(m + 1) {}
   MklCsrMatrix(int m, int n, const float *val, const int *col, const int *ptr)
      : m(m), n(n), val(val, val + ptr[m]), col(col, col + ptr[m]), ptr(ptr, ptr + m + 1) {}
   MklCsrMatrix(int ncol,
      vector<float>::const_iterator val_begin, vector<float>::const_iterator val_end,
      vector<int>::const_iterator col_begin, vector<int>::const_iterator col_end,
      vector<int>::const_iterator ptr_begin, vector<int>::const_iterator ptr_end)
      : m(ptr_end - ptr_begin - 1), n(ncol), val(val_begin, val_end), col(col_begin, col_end),
      ptr(ptr_begin, ptr_end)
   {
      const int offset = ptr[0];

      if (offset > 0)
         for (typename vector<int>::iterator p = ptr.begin(); p < ptr.end(); p++)
            *p -= offset;
   }

   int rows() const { return m; }
   int cols() const { return n; }

   void mml(char transa, char transb, int nrhs, float alpha, const float *b, int ldb, float beta,
      float *c, int ldc) const
   {
      /* need to const_cast some args since they are not declared const in the MKL interface */

      if (transb == 'n' || transb == 'N')
      {
         cerr << "simple mkl_scsrmm" << endl;
         cerr << transa << " " << m << " " << nrhs << " " << n << " " << alpha << " " << MklCsrMatrix::mdescr()
            << " " << "val, col, ptr0, ptr1, " << b << " " << ldb << " " << beta
            << " " << c << " " << ldc << endl;
         mkl_scsrmm(&transa, const_cast<int *>(&m), &nrhs, const_cast<int *>(&n), &alpha,
            MklCsrMatrix::mdescr(), const_cast<float *>(&val[0]), const_cast<int *>(&col[0]),
            const_cast<int *>(&ptr[0]), const_cast<int *>(&ptr[1]), const_cast<float *>(b), &ldb,
            &beta, c, &ldc);
         cerr << "done" << endl;
      }
      else if (transb == 't' || transb == 'T')
      {
         const int middle_dim = (transa == 't' || transa == 'T' ? m : n);
         float *tb = new float[middle_dim * nrhs];

         //mkl_somatcopy('r', 't', nrhs, middle_dim, 1.f, const_cast<float *>(b), ldb, tb, nrhs);
         mkl_somatcopy('c', 't', nrhs, middle_dim, 1.f, const_cast<float *>(b), nrhs, tb, ldb);
         mkl_scsrmm(&transa, const_cast<int *>(&m), &nrhs, const_cast<int *>(&n), &alpha,
            MklCsrMatrix::mdescr(), const_cast<float *>(&val[0]), const_cast<int *>(&col[0]),
            const_cast<int *>(&ptr[0]), const_cast<int *>(&ptr[1]), tb, &nrhs, &beta, c, &ldc);

         delete[] tb;
      }
      else
      {
         throw runtime_error("transposition param not supported");
      }
   }

   void mml_vec(char transa, char transb, int nrhs, float alpha, const float *b, int ldb, float beta,
      float *c, int ldc)
   {
      /* Compute A * B = C by A * b = c, where b and c are col-vectors of B and C, respectively.
       * The transposition of A is handled by the underlying routine. In case of A * B^T = C, we
       * need to do the same but with B's row vectors. */

      const int nlhs = (transa == 't' || transa == 'T' ? n : m);
      float *ccol = new float[nlhs];

      if (transb == 'n' || transb == 'N')
      {
         const int middle_dim = (transa == 't' || transa == 'T' ? m : n);
         float *bcol = new float[middle_dim];

         for (int i = 0; i < nrhs; i++)
         {
            for (int j = 0; j < middle_dim; j++)
               bcol[j] = b[j * ldb + i];

            if (beta != 0.f)
               for (int j = 0; j < nlhs; j++)
                  ccol[j] = c[j * ldc + i];

            mkl_scsrmv(&transa, &m, &n, &alpha, MklCsrMatrix::mdescr(),
               &val[0], &col[0], &ptr[0], &ptr[1],
               bcol, &beta, ccol);

            for (int j = 0; j < nlhs; j++)
               c[j * ldc + i] = ccol[j];
         }

         delete[] bcol;
      }
      else if (transb == 't' || transb == 'T')
      {
         for (int i = 0; i < nrhs; i++)
         {
            if (beta != 0.f)
               for (int j = 0; j < nlhs; j++)
                  ccol[j] = c[j * ldc + i];

            mkl_scsrmv(&transa, &m, &n, &alpha, MklCsrMatrix::mdescr(),
               &val[0], &col[0], &ptr[0], &ptr[1],
               const_cast<float *>(&b[i * ldb]), &beta, ccol);

            for (int j = 0; j < nlhs; j++)
               c[j * ldc + i] = ccol[j];
         }
      }
      else
      {
         throw runtime_error("transposition param not supported");
      }

      delete[] ccol;
   }

   void mmr_vec(char transa, char transb, int nlhs, float alpha, const float *b, int ldb, float beta,
      float *c, int ldc)
   {
      /* Compute B * A = C by b * A = c, where b and c are row-vectors of B and C, respectively.
       * This equals A^T * b^T = c^T, where the transposition matters only for A and is performed
       * by the underlying routine. In case of B^T * A = C, we need to do the same but with B's
       * column vectors. */

      transa = trans_inv(transa);

      if (transb == 'n' || transb == 'N')
      {
         for (int i = 0; i < nlhs; i++)
         {
            mkl_scsrmv(&transa, &m, &n, &alpha, MklCsrMatrix::mdescr(),
               &val[0], &col[0], &ptr[0], &ptr[1],
               const_cast<float *>(&b[i * ldb]), &beta, const_cast<float *>(&c[i * ldc]));
         }
      }
      else if (transb == 't' || transb == 'T')
      {
         const int middle_dim = (transa == 't' || transa == 'T' ? m : n);
         float *bcol = new float[middle_dim];

         for (int i = 0; i < nlhs; i++)
         {
            for (int j = 0; j < middle_dim; j++)
               bcol[j] = b[j * ldb + i];

            mkl_scsrmv(&transa, &m, &n, &alpha, MklCsrMatrix::mdescr(),
               &val[0], &col[0], &ptr[0], &ptr[1],
               bcol, &beta, const_cast<float *>(&c[i * ldc]));
         }

         delete[] bcol;
      }
      else
      {
         throw runtime_error("transposition param not supported");
      }
   }

   void mmr_tra(char transa, char transb, int nlhs, float alpha, const float *b, int ldb, float beta,
      float *c, int ldc)
   {
      /* Compute B * A = C by A^T * B^T = C^T, which has the form nrhs x middle_dim x nlhs */
      transa = trans_inv(transa);
      transb = trans_inv(transb);

      float *_b;
      int _ldb;

      if (transb == 't' || transb == 'T')
      {
         const int middle_dim = (transa == 't' || transa == 'T' ? m : n);
         _b = new float[nlhs * middle_dim];
         mkl_somatcopy('r', 't', nlhs, middle_dim, 1.f, const_cast<float *>(b), ldb, _b, nlhs);
         _ldb = nlhs;
      }
      else
      {
         _b = const_cast<float *>(b);
         _ldb = ldb;
      }

      const int nrhs = (transa == 't' || transa == 'T' ? n : m);
      float *_c = new float[nlhs * nrhs];

      if (beta != 0.f)
      {
         mkl_somatcopy('r', 't', nlhs, nrhs, 1.f, c, ldc, _c, nlhs);
      }

      mkl_scsrmm(&transa, &m, &nlhs, &n, &alpha,
         MklCsrMatrix::mdescr(), &val[0], &col[0],
         &ptr[0], &ptr[1], _b, &_ldb, &beta, _c, &nlhs);

      mkl_somatcopy('r', 't', nrhs, nlhs, 1.f, _c, nlhs, c, ldc);

      if (_b != b)
         delete[] _b;

      delete[] _c;
   }

   void colmeans(float *means) const
   {
      memset(means, 0, n * sizeof(float));

      for (int i = 0; i < m; i++)
         for (int j = ptr[i]; j < ptr[i+1]; j++)
            means[col[j]] += val[j];

      for (int j = 0; j < n; j++)
         means[j] /= m;
   }

   void colvars(float *vars) const
   {
      vector<float> means(n);
      vector<int>   nnz(n);

      memset(vars, 0, n * sizeof(float));
      colmeans(&means[0]);

      for (int i = 0; i < m; i++)
         for (int j = ptr[i]; j < ptr[i+1]; j++)
         {
            const float x = val[j] - means[col[j]];
            vars[col[j]] += x * x;
            nnz[col[j]]++;
         }

      for (int j = 0; j < n; j++)
      {
         vars[j] += means[j] * means[j] * (m - nnz[j]);
         vars[j] /= m;
      }
   }

   MklCsrMatrix *rowsubset(int first_row, int nrows) const
   {
      return new MklCsrMatrix(n,
         val.begin() + ptr[first_row], val.begin() + ptr[first_row + nrows],
         col.begin() + ptr[first_row], col.begin() + ptr[first_row + nrows],
         ptr.begin() + first_row, ptr.begin() + first_row + nrows + 1);
   }

   void setelement(int r, int c, float v)
   {
      int j;

      for (j = ptr[r]; j < ptr[r+1] && col[j] < c; j++) /* adjust j to point to A[r,c] */
         ;

      if (v == 0.f) /* set to zero */
      {
         if (j < ptr[r+1] && col[j] == c) /* non-zero entry affected, delete */
         {
            val.erase(val.begin() + j);
            col.erase(col.begin() + j);

            for (typename vector<int>::iterator p = ptr.begin() + r + 1; p < ptr.end(); p++)
               (*p)--;
         }
      }
      else /* set to non-zero */
      {
         if (j < ptr[r+1] && col[j] == c) /* non-zero entry affected, overwrite */
         {
            val[j] = v;
         }
         else /* zero entry affected, create */
         {
            val.insert(val.begin() + j, v);
            col.insert(col.begin() + j, c);

            for (int *p = &ptr[r+1]; p <= &ptr[m]; p++)
               (*p)++;
         }
      }
   }

   /* get pointer to element if allocated, NULL otherwise */
   float *getelementp(int r, int c)
   {
      int j;

      for (j = ptr[r]; j < ptr[r+1] && col[j] < c; j++) /* adjust j to point to A[r,c] */
         ;

      if (j < ptr[r+1] && col[j] == c) /* non-zero entry affected, assign */
         return &val[j];

      return NULL;
   }

   float &getelement(int r, int c)
   {
      int j;

      for (j = ptr[r]; j < ptr[r+1] && col[j] < c; j++) /* adjust j to point to A[r,c] */
         ;

      if (j < ptr[r+1] && col[j] == c) /* non-zero entry affected */
      {
         return val[j];
      }
      else /* zero entry affected, create */
      {
         val.insert(val.begin() + j, 0.f);
         col.insert(col.begin() + j, c);

         for (typename vector<int>::iterator p = ptr.begin() + r + 1; p < ptr.end(); p++)
            (*p)++;

         return val[j];
      }
   }

   void scalecols(const float *s)
   {
      for (int i = 0; i < m; i++)
         for (int j = ptr[i]; j < ptr[i+1]; j++)
            val[j] *= s[col[j]];
   }

   void scalerows(const float *s)
   {
      for (int i = 0; i < m; i++)
         for (int j = ptr[i]; j < ptr[i+1]; j++)
            val[j] *= s[i];
   }

   void print() const /* debug */
   {
      printf("MklCsrMatrix of dim %dx%d", m, n);

      printf("\nval: ");

      for (int i = 0; i < ptr[m]; i++)
         printf("%5.3f ", val[i]);

      printf("\ncol: ");

      for (int i = 0; i < ptr[m]; i++)
         printf("%d ", col[i]);

      printf("\nptr: ");

      for (int i = 0; i < m+1; i++)
         printf("%d ", ptr[i]);

      printf("\n");

      /*for (int i = 0; i < m; i++)
      {
         int idx = -1;

         for (int j = 0; j < (ptr[i+1] - ptr[i]); j++)
         {
            idx = ptr[i] + j;

            for (int k = (j == 0 ? 0 : col[idx-1] + 1); k < col[idx]; k++)
               printf("   .   ");

            printf("%5.3f ", val[idx]);
         }

         for (int k = (idx == -1 ? 0 : col[idx] + 1); k < n; k++)
            printf("   .   ");

         printf("\n");
      }*/
   }
};

/* Store allocated matrices in a map using int keys as handles, the bitset keys keeps track of
 * (de-) allocated keys. */
static map<int, MklCsrMatrix *> table;
static bitset<1024> keys;

/* routines for handle-pointer-conversion and key management */
static inline MklCsrMatrix *getMatPtr(int key)
{
   typename map<int, MklCsrMatrix *>::iterator i = table.find(key);

   return (i == table.end() ? NULL : i->second);
}

static inline MklCsrMatrix *getMat(int key)
{
   MklCsrMatrix *p = getMatPtr(key);

   if (p == NULL)
      throw runtime_error("MKL CSR Matrix key invalid");

   return p;
}

static inline int getKey()
{
   if (keys.count() == keys.size())
      throw runtime_error("key allocation error");

   int i = 0;

   while (keys[i]) /* no need to check range, there is at least one 0-bit */
      i++;

   keys.set(i);
   return i;
}

static inline void putKey(int i)
{
   if (i < 0 || i >= keys.size())
      throw runtime_error("attempt to put back invalid key");

   keys.reset(i);
}

/* column means and variances */
void scolmeans(spmat_t A, float *means)
{
   cerr << "scolmeans" << endl;
   getMat(A)->colmeans(means);
}

void scolvars(spmat_t A, float *vars)
{
   cerr << "scolvars" << endl;
   getMat(A)->colvars(vars);
}

void srowmeans(spmat_t A, float *means)
{
   throw not_implemented;
}

void srowvars(spmat_t A, float *vars)
{
   throw not_implemented;
}

/* scale rows/cols */
void sscalecols(spmat_t A, float *s)
{
   cerr << "sscalecols" << endl;
   getMat(A)->scalecols(s);
}

void sscalerows(spmat_t A, float *s)
{
   cerr << "scalerows" << endl;
   getMat(A)->scalerows(s);
}

/* select row subset */
spmat_t srowsubset(spmat_t A, int first_row, int nrow)
{
   spmat_t B = getKey();

   cerr << "srowsubset(A=" <<  A << ", first_row=" << first_row << ", nrow=" << nrow << ")" << endl;
   table.insert(make_pair(B, getMat(A)->rowsubset(first_row, nrow)));
   return B;
}

/* creation and destruction */
spmat_t suscr_csr(int m, int n, float *val, int *col, int *ptr)
{
   spmat_t A = getKey();

   cerr << "suscr_csr(m=" << m << ", n=" << n << ", val, col, ptr)" << endl;
   table.insert(make_pair(A, new MklCsrMatrix(m, n, val, col, ptr)));
   return A;
}

spmat_t create(int row, int col)
{
   spmat_t A = getKey();

   cerr << "create" << endl;
   table.insert(make_pair(A, new MklCsrMatrix(row, col)));
   return A;
}

void destroy(spmat_t A)
{
   typename map<int, MklCsrMatrix *>::iterator i = table.find(A);

   if (i == table.end())
      throw runtime_error("MKL CSR Matrix key invalid");

   cerr << "destroy(A=" << A << ")" << endl;

   delete i->second;
   putKey(i->first);
   table.erase(i);
}

/* set element (set to zero will delete entry) */
void ssetelement( spmat_t A, int row, int col, float val )
{
   cerr << "sgetelement(A=" << A << ", row=" << row << ", col=" << col << ")" << endl;
   getMat(A)->setelement(row, col, val);
}

void ssetelement( spmat_t A, int idx, float val )
{
   MklCsrMatrix *Ap = getMat(A);

   cerr << "ssetelement(A=" << A << ", idx=" << idx << ")" << endl;
   Ap->setelement(idx / Ap->cols(), idx % Ap->cols(), val);
}

/* get element reference (zero assignment will not delete entry) */
float &sgetelement( spmat_t A, int row, int col)
{
   cerr << "sgetelement(A=" << A << ", row=" << row << ", col=" << col << ")" << endl;
   return getMat(A)->getelement(row, col);
}

float &sgetelement( spmat_t A, int idx )
{
   MklCsrMatrix *Ap = getMat(A);
   cerr << "sgetelement(A=" << A << ", idx=" << idx << ")" << endl;
   return Ap->getelement(idx / Ap->cols(), idx % Ap->cols());
}

float *sgetelementp( spmat_t A, int row, int col )
{
   cerr << "sgetelementp(A=" << A << ", row=" << row << ", col=" << col << ")" << endl;
   return getMat(A)->getelementp(row, col);
}

float *sgetelementp( spmat_t A, int idx )
{
   MklCsrMatrix *Ap = getMat(A);

   cerr << "sgetelementp(A=" << A << ", idx=" << idx << ")" << endl;
   return Ap->getelementp(idx / Ap->cols(), idx % Ap->cols());
}


#include <ctime> /* for clock */

/* sgemm routines with sparse matrix being lhs (A) or rhs (B) of the product */
void susgemm(char sidea, char transa, char transb, int nohs, const float &alpha, spmat_t A,
   const float *B, int ldB, const float &beta, float *C, int ldC)
{
   cerr << "susgemm(sidea=" << sidea << ", transa=" << transa << ", transb=" << transb
      << ", nohs=" << nohs << ", alpha=" << alpha
      << ", A=" << A << " (" << getMat(A)->rows() << "x" << getMat(A)->cols() << ")"
      << ", B=" << B << ", ldB=" << ldB << ", beta=" << beta << ", C=" << C
      << ", ldC=" << ldC << endl;

   switch (sidea)
   {
   case 'l':
   case 'L':
      //getMat(A)->mml(transa, transb, nohs, alpha, B, ldB, beta, C, ldC);
      getMat(A)->mml_vec(transa, transb, nohs, alpha, B, ldB, beta, C, ldC);
      break;

   case 'r':
   case 'R':
   {
      //getMat(A)->mmr_tra(transa, transb, nohs, alpha, B, ldB, beta, C, ldC);
      getMat(A)->mmr_vec(transa, transb, nohs, alpha, B, ldB, beta, C, ldC);
      break;
   }

   default:
      throw runtime_error("unsupported side type");
      break;
   }
}

/* checks whether A is a valid handle */
bool handle_valid(spmat_t A)
{
   return (getMatPtr(A) != NULL);
}

namespace NIST_SPBLAS
{
   void print(spmat_t A)
   {
      getMat(A)->print();
   }
}

#endif /* ifndef MKL */

#if 0
/* test section */
#include <iostream>
#include <cstring>

using namespace std;

void printmat(float *A, int r, int c)
{
   for (int i = 0; i < r; i++)
   {
      for (int j = 0; j < c; j++)
         cout << A[i*c + j] << ", ";
      cout << endl;
   }
}

#include <fstream>
#include <cstdlib>

int main()
{
   /*float val[] = {1.2345, 3.49677, 2.123079, 5.12956, 4.234844, 2.13407685};
   int col[] = {2, 0, 3, 2, 1, 2};
   int ptr[] = {0, 1, 3, 4, 6};

   spmat_t A_mkl = _suscr_csr(4, 4, val, col, ptr);
   spmat_t A_gen = suscr_csr(4, 4, val, col, ptr);

   NIST_SPBLAS::print(A_gen);
   NIST_SPBLAS::_print(A_mkl);

   float B[] = {0.568, 1.153, 5.063, 4.238,
                2.225, 3.853, 4.129, 5.534};

   float C_mkl[8];
   float C_gen[8];

   memset(C_mkl, 0, sizeof(C_mkl));
   memset(C_gen, 0, sizeof(C_gen));*/


   ifstream ifs("../../RFN-doc/t10k-images-idx3-ubyte", ifstream::in | ifstream::binary);

   int v;

   for (int i = 0; i < 16; i++)
   {
      v = ifs.get();
   }

   cout << endl;
   vector<float> val;
   vector<int> col;
   vector<int> ptr;

   ptr.push_back(0);

   for (int i = 0; i < 10000; i++)
   {
      ptr.push_back(ptr[ptr.size()-1]);

      for (int j = 0; j < 784; j++)
      {
         v = ifs.get();

         if (v != 0)
         {
            val.push_back(v/255.f);
            col.push_back(j);
            ptr[ptr.size()-1]++;
         }
      }
   }

   ifs.close();

   spmat_t A = suscr_csr(10000, 784, &val[0], &col[0], &ptr[0]);
   NIST_SPBLAS::print(A);

   float B[28 * 784];
   srand(0);

   for (int i = 0; i < 28*784; i++)
      B[i] = ((float) rand() / RAND_MAX) * 2 - 1;

   float C[10000 * 28];

   susgemm('l', 'n', 't', 28, 1.f, A, B, 784, 0.f, C, 28);

   for (int i = 0; i < 32; i++)
      cout << C[rand() % (10000 * 28)] << ", ";

   cout << endl;

   return 0;

   /*
   //memset(C, 0, sizeof(C));

   _susgemm('r', 'n', 't', 2, .1f, A_mkl, B, 2, 0.f, C_mkl, 4);
   //printmat(B, 2, 4);
   printmat(C_mkl, 2, 4);

   susgemm('r', 'n', 't', 2, .1f, A_gen, B, 2, 0.f, C_gen, 4);
   //printmat(B, 2, 4);
   printmat(C_gen, 2, 4);
   */


   /*NIST_SPBLAS::print(A);
   NIST_SPBLAS::print(srowsubset(A, 0, 4));*/
}
#endif //0
