#include <Rcpp.h>
#include <vector>
#include <cstdio>

#include "librfn.h"
#include "use_R_impl.h"

#include <ctime>

using namespace Rcpp;


RcppExport SEXP R_train_rfn(SEXP Xs, SEXP Ws, SEXP Ps, SEXP ns, SEXP ms, SEXP ks, SEXP n_iters,
   SEXP batch_sizes, SEXP etaWs, SEXP etaPs, SEXP minPs, SEXP h_thresholds, SEXP dropout_rates,
   SEXP input_noise_rates, SEXP l2_weightdecays, SEXP l1_weightdecays, SEXP momentums,
   SEXP noise_types, SEXP apply_relus, SEXP apply_scalings, SEXP apply_newton_updates, SEXP seeds, SEXP gpu_id)
{
   BEGIN_RCPP

   int n = as<int>(ns);
   int m = as<int>(ms);
   int k = as<int>(ks);

   std::vector<float> X = as<std::vector<float> >(Xs);
   std::vector<float> W = as<std::vector<float> >(Ws);
   std::vector<float> P = as<std::vector<float> >(Ps);

   GetRNGstate();

   clock_t t = clock();
   train_rfn(&X[0], &W[0], &P[0], n, m, k, as<int>(n_iters), as<int>(batch_sizes), as<float>(etaWs),
      as<float>(etaPs), as<float>(minPs), as<float>(h_thresholds), as<float>(dropout_rates),
      as<float>(input_noise_rates), as<float>(l2_weightdecays), as<float>(l1_weightdecays),
      as<float>(momentums), as<int>(noise_types), as<int>(apply_relus), as<int>(apply_scalings),
      as<int>(apply_newton_updates), as<int>(seeds), as<int>(gpu_id));
   t = clock() - t;

   PutRNGstate();

   NumericVector W_ret = wrap(W);
   W_ret.attr("dim") = Dimension(m, k); /* conversion from rowmajor 2 colmajor */

   NumericVector P_ret = wrap(P);

   List ret;
   ret["W"] = W_ret;
   ret["P"] = P_ret;
   ret["T"] = wrap<double>(((double) t) / CLOCKS_PER_SEC);
   return ret;

   END_RCPP
}

RcppExport SEXP R_train_rfn_sparse(SEXP Xs, SEXP rowvs, SEXP colvs, SEXP Ws, SEXP Ps, SEXP ns, SEXP ms, SEXP ks, SEXP n_iters,
   SEXP batch_sizes, SEXP etaWs, SEXP etaPs, SEXP minPs, SEXP h_thresholds, SEXP dropout_rates,
   SEXP input_noise_rates, SEXP l2_weightdecays, SEXP l1_weightdecays, SEXP momentums,
   SEXP noise_types, SEXP apply_relus, SEXP apply_scalings, SEXP apply_newton_updates, SEXP seeds, SEXP gpu_id)
{
   BEGIN_RCPP

   int n = as<int>(ns);
   int m = as<int>(ms);
   int k = as<int>(ks);

   std::vector<int> rowv = as<std::vector<int> >(rowvs);
   std::vector<int> colv = as<std::vector<int> >(colvs);

   std::vector<float> X = as<std::vector<float> >(Xs);
   std::vector<float> W = as<std::vector<float> >(Ws);
   std::vector<float> P = as<std::vector<float> >(Ps);

   GetRNGstate();

   clock_t t = clock();
   train_rfn_sparse(&X[0], &colv[0], &rowv[0], &W[0], &P[0], n, m, k, as<int>(n_iters), as<int>(batch_sizes), as<float>(etaWs),
      as<float>(etaPs), as<float>(minPs), as<float>(h_thresholds), as<float>(dropout_rates),
      as<float>(input_noise_rates), as<float>(l2_weightdecays), as<float>(l1_weightdecays),
      as<float>(momentums), as<int>(noise_types), as<int>(apply_relus), as<int>(apply_scalings),
      as<int>(apply_newton_updates), as<int>(seeds), as<int>(gpu_id));
   t = clock() - t;

   PutRNGstate();

   NumericVector W_ret = wrap(W);
   W_ret.attr("dim") = Dimension(m, k);

   NumericVector P_ret = wrap(P);

   List ret;
   ret["W"] = W_ret;
   ret["P"] = P_ret;
   ret["T"] = wrap<double>(((double) t) / CLOCKS_PER_SEC);
   return ret;

   END_RCPP
}


RcppExport SEXP R_calculate_W(SEXP Xs, SEXP Ws, SEXP Ps, SEXP ns, SEXP ms, SEXP ks, SEXP activations,
   SEXP apply_scalings, SEXP h_thresholds, SEXP gpu_id)
{
   BEGIN_RCPP

   int n = as<int>(ns);
   int m = as<int>(ms);
   int k = as<int>(ks);

   std::vector<float> X = as<std::vector<float> >(Xs);
   std::vector<float> W = as<std::vector<float> >(Ws);
   std::vector<float> P = as<std::vector<float> >(Ps);
   std::vector<float> Wout(k*m);

   calculate_W(&X[0], &W[0], &P[0], &Wout[0], n, m, k, as<int>(activations),
      as<int>(apply_scalings), as<float>(h_thresholds), as<int>(gpu_id));

   NumericVector Wout_ret = wrap(Wout);
   Wout_ret.attr("dim") = Dimension(m, k);

   return Wout_ret;

   END_RCPP
}

RcppExport SEXP R_calculate_W_sparse(SEXP Xs, SEXP rowvs, SEXP colvs, SEXP Ws, SEXP Ps, SEXP ns, SEXP ms, SEXP ks, SEXP activations,
   SEXP apply_scalings, SEXP h_thresholds, SEXP gpu_id)
{
   BEGIN_RCPP

   int n = as<int>(ns);
   int m = as<int>(ms);
   int k = as<int>(ks);


   std::vector<int> rowv = as<std::vector<int> >(rowvs);
   std::vector<int> colv = as<std::vector<int> >(colvs);

   std::vector<float> X = as<std::vector<float> >(Xs);
   std::vector<float> W = as<std::vector<float> >(Ws);
   std::vector<float> P = as<std::vector<float> >(Ps);
   std::vector<float> Wout(k*m);

   calculate_W_sparse(&X[0], &colv[0], &rowv[0], &W[0], &P[0], &Wout[0], n, m, k, as<int>(activations),
      as<int>(apply_scalings), as<float>(h_thresholds), as<int>(gpu_id));

   NumericVector Wout_ret = wrap(Wout);
   Wout_ret.attr("dim") = Dimension(m, k);

   return Wout_ret;

   END_RCPP
}


//#include <mkl_spblas.h>
//#include <mkl_trans.h>
using std::cerr;
using std::endl;

RcppExport SEXP somatcopy(SEXP s_arows, SEXP s_acols, SEXP s_A)
{
   BEGIN_RCPP

   char order = 'c';//as<char>(s_order);
   char trans = 't';//as<char>(s_trans);
   size_t arows = as<size_t>(s_arows);
   size_t acols = as<size_t>(s_acols);
   std::vector<float> A = as<std::vector<float> >(s_A);
   size_t lda = arows;
   std::vector<float> B(A.size());
   size_t ldb = acols;

   //mkl_somatcopy(order, trans, arows, acols, 1.f, &A[0], lda, &B[0], ldb);

   NumericVector B_ret = wrap(B);
   B_ret.attr("dim") = Dimension(acols, arows);

   return B_ret;

   END_RCPP
}
