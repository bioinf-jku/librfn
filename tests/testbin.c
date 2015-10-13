
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <sys/time.h>
#include "../librfn.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// random in (0, 1]
static double rand_unif(void) {
    return (rand())/(RAND_MAX+1.0);
}
/*
// generates random samples from a 0/1 Gaussian via Box-Mueller
static double rand_normal(void) {
    return sqrt(-2.0*log(rand_unif())) * cos(2.0*M_PI*rand_unif());
}
*/

float time_diff(struct timeval *t2, struct timeval *t1) {
    long int diff = (t2->tv_usec + 1000000 * t2->tv_sec) - (t1->tv_usec + 1000000 * t1->tv_sec);
    return diff / 1000000.0f;
}



int main(int argc, char** argv) {
    int n = 50000;
    int m = 784;
    int k = 2048;
    int n_iter = 10;
    int gpu_id = -1;

    if (argc > 1)
        k = atoi(argv[1]);

    if (argc > 2)
        n_iter = atoi(argv[2]);

    if (argc > 3)
        m = atoi(argv[3]);

    if (argc > 4)
        gpu_id = atoi(argv[4]);


    float* X = (float*) malloc(n*m*sizeof(float));
    float* W = (float*) malloc(n*k*sizeof(float));
    float* P = (float*) malloc(m*sizeof(float));

    for (int i = 0; i < n*m; ++i)
        X[i] = 5.0f* rand_unif() - 0.5;
    for (int i = 0; i < n*k; ++i)
        W[i] = rand_unif() - 0.5;

    struct timeval t0, t1;
    gettimeofday(&t0, 0);
    train_gpu(X, W, P, n, m, k, n_iter, 0.1, 0.1, 1e-2, 0.0, 0.0, 32, gpu_id);
    //train_cpu(X, W, P, n, m, k, n_iter, 0.1, 0.1, 1e-2, 0.0, 0.0, 32);
    gettimeofday(&t1, 0);
    printf("time for rfn: %3.4fs\n", time_diff(&t1, &t0));
    free(X);
    free(W);
    free(P);
    return 0;
}
