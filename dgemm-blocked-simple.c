const char *dgemm_desc = "Simple blocked dgemm.";

#define min(a, b) (((a) < (b))? (a) : (b))

#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

static void do_block(int n, int M, int N, int K, double *A, double *B, double *C) {
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            double cij = C[i + j * n];
            for (int k = 0; k < K; ++k) {
                cij += A[i + k * n] * B[k + j * n];
            }
            C[i + j * n] = cij;
        }
    }
}

void square_dgemm(int n, double *A, double *B, double *C) {
  
  int l1_cache = 32*1024; //bytes
  int size_of_double = 8; //bytes
  double no_of_doubles_in_l1 = l1_cache/size_of_double;
  int BLOCK_SIZE = sqrt(no_of_doubles_in_l1/3); 

    for (int i = 0; i < n; i += BLOCK_SIZE) {
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            for (int k = 0; k < n; k += BLOCK_SIZE) {
                int M = min (BLOCK_SIZE, n - i);
                int N = min (BLOCK_SIZE, n - j);
                int K = min (BLOCK_SIZE, n - k);
                do_block(n, M, N, K, A + i + k*n, B + k + j*n, C + i + j*n);
            }
        }
    }
}
