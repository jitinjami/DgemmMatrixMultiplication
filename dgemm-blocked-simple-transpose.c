const char *dgemm_desc = "Simple-tranposed blocked dgemm.";

#define min(a, b) (((a) < (b))? (a) : (b))

#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

/*
 * This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N.
 */

static void do_block_tranpose(int n, int M, int N, int K, double *A, double *B, double *C) {
    // For each row i of A
    for (int i = 0; i < M; ++i) {
        //For each column j of B
        for (int j = 0; j < N; ++j) {
            // Compute C(i,j)
            double cij = C[i + j * n];
            for (int k = 0; k < K; ++k) {
                cij += A[k + i * n] * B[k + j * n];
            }
            C[i + j * n] = cij;
        }
    }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are n-by-n matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */
void square_dgemm(int n, double *A, double *B, double *C) {
  
  int l1_cache = 32*1024; //bytes
  int size_of_double = 8; //bytes
  double no_of_doubles_in_l1 = l1_cache/size_of_double;
  int BLOCK_SIZE = sqrt(no_of_doubles_in_l1/3); 

  double *A_t = (double *) malloc(sizeof(double) * n * n);

  //Transpose A -> row-major format
  for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
           A_t[j + n * i] = A[i + n * j];
      }
  }


    // For each block-row of A
    for (int i = 0; i < n; i += BLOCK_SIZE) {
        // For each block-column of B
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            // Accumulate block dgemms into block of C
            for (int k = 0; k < n; k += BLOCK_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min (BLOCK_SIZE, n - i);
                int N = min (BLOCK_SIZE, n - j);
                int K = min (BLOCK_SIZE, n - k);
                // Perform individual block dgemm
                do_block_tranpose(n, M, N, K, A_t + k + i*n, B + k + j*n, C + i + j*n);
            }
        }
    }
    free(A_t);
}
