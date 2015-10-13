/*
Copyright © 2015 Thomas Unterthiner
Licensed under GPL, version 2 or a later (see LICENSE.txt)
*/

#ifndef LIBRFN_H
#define LIBRFN_H


#ifdef __cplusplus
extern "C" {
#endif


/**
 * Trains an RFN network on the CPU.
 *
 * Note: All arguments are assumed to be in C-order (ie., row-major)
 * and in host (ie., CPU) memory.
 *
 * @param X_host            [n, m] data matrix, with 1 sample per row
 * @param W_host            [k, m] weight matrix, expected to be initialized
 * @param P_host            [m, ] vector, used to store Psi, expected to be initialized
 * @param n                 number of samples
 * @param m                 number of input features
 * @param k                 number of hidden units
 * @param n_iter            number of iterations the algorithm will run
 * @param batch_size        Size of minibatches (use -1 for full batch training)
 * @param etaW              learnrate for the weights
 * @param etaP              learnrate for Psi
 * @param minP              lower limit for entries in Psi
 * @param h_threshold       number below which hiddens are set to 0 (usually 0)
 * @param dropout_rate      the dropout rate for hidden activations
 * @param input_noise_rate  the noise/dropout rate for input units
 * @param l2_weightdecay    amount of L2 weight decay / L2 penalty term
 * @param l1_weightdecay    amount of L1 weight decay / L1 penalty term
 * @param momentum          momentum term
 * @param noise_type        input noise type:
 *                             0: no noise               1: dropout noise
 *                             2: salt & pepper noise    3: gaussian noise
 * @param seed              seed for the random number generation
 * @param activation_type   activation function for the hidden units:
 *                             0: linear (do nothing)    1: ReLU
 *                             2: Leaky ReLU             3: sigmoid  4: tanh
 * @param apply_scaling     wether to scale the hiddens to unit variance or not
 * @param applyNewtonUpdate Use newton update step instead of gradient step
 * @param seed              seed for the random number generation
 *
 * @return 0 on success, 1 otherwise. The trained network will be stored
 *         in the W and P variables.
 *
 */
int train_cpu(const float* X, float* W, float* P, const int n, const int m,
              const int k, const int n_iter, int batch_size, const float etaW,
              const float etaP, const float minP, const float h_threshold,
              const float dropout_rate, const float input_noise_rate,
              const float l2_weightdecay, const float l1_weightdecay,
              const float momentum,
              const int noise_type, const int apply_relu, const int apply_scaling,
              const int applyNewtonUpdate, unsigned long seed);


/**
 * Trains an RFN network on the GPU.
 *
 * Note: All arguments are assumed to be in C-order (ie., row-major)
 * and in host (ie., CPU) memory. All transfers from and to the GPU will be
 * done internally by the function itself.
 *
 * @param X_host            [n, m] data matrix, with 1 sample per row
 * @param W_host            [k, m] weight matrix, expected to be initialized
 * @param P_host            [m, ] vector, used to store Psi
 * @param n                 number of samples
 * @param m                 number of input features
 * @param k                 number of hidden units
 * @param n_iter            number of iterations the algorithm will run
 * @param batch_size        Size of minibatches (use -1 for full batch training)
 * @param etaW              learnrate for the weights
 * @param etaP              learnrate for Psi
 * @param minP              lower limit for entries in Psi
 * @param h_threshold       number below which hiddens are set to 0 (usually 0)
 * @param dropout_rate      the dropout rate for hidden activations
 * @param input_noise_rate  the noise/dropout rate for input units
 * @param l2_weightdecay    amount of L2 weight decay / L2 penalty term
 * @param l1_weightdecay    amount of L1 weight decay / L1 penalty term
 * @param momentum          momentum term
 * @param noise_type        input noise type:
 *                             0: no noise               1: dropout noise
 *                             2: salt & pepper noise    3: gaussian noise
 * @param seed              seed for the random number generation
 * @param activation_type   activation function for the hidden units:
 *                             0: linear (do nothing)    1: ReLU
 *                             2: Leaky ReLU             3: sigmoid  4: tanh
 * @param apply_scaling     wether to scale the hiddens to unit variance or not
 * @param applyNewtonUpdate Use newton update step instead of gradient step
 * @param seed              seed for the random number generation
 * @param gpu_id            ID of the GPU that this will run on (if this is -1,
 *                          the GPU with the most free memory will be picked)
 *
 * @return 0 on success, 1 otherwise. The trained network will be stored
 *         in the W_host and P_host variables.
 */
int train_gpu(const float* X_host, float* W_host, float* P_host, const int n,
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
 * and in host (ie., CPU) memory.
 *
 * @param X                 [n, m] training data matrix, with 1 sample per row
 * @param W                 [k, m] RFN weight matrix
 * @param P                 [m] vector, contains Psi
 * @param Wout              [k, m] output weight matrix
 * @param n                 number of training samples
 * @param m                 number of input features
 * @param k                 number of hidden units
 * @param activation_type   activation function for the hidden units:
 *                             0: linear (do nothing)    1: ReLU
 *                             2: Leaky ReLU             3: sigmoid  4: tanh
 * @param apply_scaling     wether to scale the hiddens to unit variance or not
 * @param h_threshold       number below which hiddens are set to 0 (usually 0)
 */
void calculate_W_cpu(const float* X, const float* W, const float* P, float* Wout,
                     const int n, const int m, const int k,
                     const int activation_type, const int apply_scaling, const float h_threshold);


#ifndef NOGPU
/**
 * Given a trained RFN, this will calculate the weights that are used to
 * estimate the hidden activations.
 *
 * This needs access to the training data, as the W need to incorporate
 * the scaling that would otherwise be done on the hidden activations.
 * The scaling parameters have to be fitted on the training data's H.
 *
 * Note: All arguments are assumed to be in C-order (ie., row-major)
 * and in host (ie., CPU) memory. All transfers from and to the GPU will be
 * done internally by the function itself.
 *
 * @param X                 [n, m] training data matrix, with 1 sample per row
 * @param W                 [k, m] RFN weight matrix
 * @param P                 [m] vector, contains Psi
 * @param Wout              [k, m] output weight matrix
 * @param n                 number of training samples
 * @param m                 number of input features
 * @param k                 number of hidden units
 * @param activation_type   activation function for the hidden units:
 *                             0: linear (do nothing)    1: ReLU
 *                             2: Leaky ReLU             3: sigmoid  4: tanh
 * @param apply_scaling     wether to scale the hiddens to unit variance or not
 * @param h_threshold       number below which hiddens are set to 0 (usually 0)
 * @param gpu_id            ID of the GPU that this will run on (if this is -1,
 *                          the GPU with the most free memory will be picked)
 */
void calculate_W_gpu(const float* X, const float* W, const float* P, float* Wout,
                     const int n, const int m, const int k,
                     const int activation_type, const int apply_scaling,
                     const float h_threshold, int gpu_id);
#endif

#ifdef __cplusplus
}
#endif

#endif
