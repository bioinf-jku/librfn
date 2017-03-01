#ifndef LIBRFN_H
#define LIBRFN_H

/*
Copyright © 2015-2017 Thomas Unterthiner
Additional Contributions by Thomas Adler, Balázs Bencze
Licensed under GPL, version 2 or a later (see LICENSE.txt)
*/

#ifdef __cplusplus
extern "C" {
#endif


const int USE_CPU = -2;


/**
 * Trains an RFN network.
 *
 * Note: All arguments are assumed to be in C-order (ie., row-major)
 * and in host (ie., CPU) memory.
 * If necessary, any transfers from and to the GPU will be
 * done internally by the function itself.
 *
 * @param X             [n, m] data matrix, with 1 sample per row
 * @param W             [k, m] weight matrix, expected to be pre-initialized
 * @param P             [m, ] vector, used to store Psi
 * @param n             number of samples
 * @param m             number of input features
 * @param k             number of hidden units
 * @param n_iter        number of iterations the algorithm will run
 * @param learnrate     learnrate
 * @param dropout_rate  the dropout rate for hidden activations
 * @param input_dropout_rate  the dropout rate for input units
 * @param seed          seed for the random number generation
 * @param gpu_id        ID of the GPU that this will run on
 *                      If this is -1 use the GPU with the most free memory
 *                      If this is -2, the CPU is used instead of the GPU
 *
 * @return 0 on success, 1 otherwise. The trained network will be stored
 *         in the W_host and P_host variables.
 */
int train_rfn(const float* X, float* W, float* P, const int n,
              const int m, const int k, const int n_iter, int batch_size,
              const float etaW, const float etaP, const float minP, const float h_threshold,
              const float dropout_rate, const float input_noise_rate,
              const float l2_weightdecay, const float l1_weightdecay,
              const float momentum,
              const int noise_type, const int activation_type, const int apply_scaling,
              const int applyNewtonUpdate, unsigned long seed, int gpu_id);


/**
 * Trains an RFN network.
 * The parameters are the same as in `int train_rfn`, except that X is encoded
 * as a sparse matrix in CSR format.
 *
 * Note: the number of nonzero elements of X should be stored in Xrowptr[n]
 */
int train_rfn_sparse(const float* Xvals, const int* Xcols, const int *Xrowptr,
                     float* W, float* P, const int n,
                     const int m, const int k, const int n_iter, int batch_size,
                     const float etaW, const float etaP, const float minP, const float h_threshold,
                     const float dropout_rate, const float input_noise_rate,
                     const float l2_weightdecay, const float l1_weightdecay,
                     const float momentum,
                     const int noise_type, const int activation_type, const int apply_scaling,
                     const int applyNewtonUpdate, unsigned long seed, int gpu_id);

/**
 * Given a trained RFN, this will calculate the weights that are used to
 * estimate the hidden activations.
 *
 * This needs access to the training data, as the W need to incorporate
 * the scaling that would otherwise be done on the hidden activations.
 * The scaling parameters have to be fitted on the training data's H.
 *
 * Note: All arguments are assumed to be in C-order (ie., row-major)
 * and in host (ie., CPU) memory. Any necessary transfers from and to the GPU
 * will be done internally by the function itself.
 *
 * @param X             [n, m] training data matrix, with 1 sample per row
 * @param W             [k, m] RFN weight matrix
 * @param P             [m] vector, contains Psi
 * @param Wout          [k, m] output weight matrix
 * @param n             number of training samples
 * @param m             number of input features
 * @param k             number of hidden units
 * @param gpu_id        ID of the GPU that this will run on
 *                      If this is -1 use the GPU with the most free memory
 *                      If this is -2, the CPU is used instead of the GPU
 */
void calculate_W(const float* X, const float* W, const float* P, float* Wout,
                 const int n, const int m, const int k,
                 const int activation_type, const int apply_scaling, const float h_threshold,
                 int gpu_id);

/**
 * Given a trained RFN, this will calculate the weights that are used to
 * estimate the hidden activations.
 *
 * The parameters are the same as in `void calculate_W`, except that X is encoded
 * as a sparse matrix in CSR format.
 *
 * Note: the number of nonzero elements of X should be stored in Xrowptr[n]
 */
void calculate_W_sparse(const float* Xvals, const int* Xcols, const int *Xrowptr,
                        const float* W, const float* P, float* Wout,
                        const int n, const int m, const int k,
                        const int activation_type, const int apply_scaling, const float h_threshold,
                        int gpu_id);

#ifdef __cplusplus
}
#endif

#endif /* LIBRFN_H */
