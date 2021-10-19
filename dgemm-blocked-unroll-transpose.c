const char *dgemm_desc = "Unrolled-tranposed blocked dgemm.";

#define min(a, b) (((a) < (b))? (a) : (b))

#include <immintrin.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

static void do_block_unroll_transpose(int n, int dim_A, int dim_B, int dim_C, double *A, double *B, double *C) {

    double cij;
    double ci1j;
    double cij1;
    double ci1j1;

    double ci2j;
    double ci2j1;
    double ci2j2;
    double ci1j2;
    double cij2;



    double r0, c0, r1, c1, r2, c2;
    int i, j, k;
    for (i = 0; i < dim_A / 3 * 3; i += 3) {
        for (j = 0; j < dim_B / 3 * 3; j += 3) {
            cij = C[i + j * n];
            ci1j = C[i + 1 + j * n];
            cij1 = C[i + (j + 1) * n];
            ci1j1 = C[(i + 1) + (j + 1) * n];
            ci2j = C[(i + 2) + (j) * n];
            ci2j1 = C[(i + 2) + (j + 1) * n];
            ci2j2 = C[(i + 2) + (j + 2) * n];
            ci1j2 = C[(i + 1) + (j + 2) * n];
            cij2 = C[(i) + (j + 2) * n];

            for (k = 0; k < dim_C; k += 1) {
                r0 = A[k + i * n];
                c0 = B[k + j * n];
                r1 = A[k + (i + 1) * n];
                c1 = B[k + (j + 1) * n];
                r2 = A[k + (i + 2) * n];
                c2 = B[k + (j + 2) * n]; 
                cij += r0 * c0;
                ci1j += r1 * c0;
                cij1 += r0 * c1;
                ci1j1 += r1 * c1;
                ci2j += r2 * c0;
                ci2j1 += r2 * c1;
                ci2j2 += r2 * c2;
                ci1j2 += r1 * c2;
                cij2 += r0 * c2;
            }


            C[i + j * n] = cij;
            C[i + 1 + j * n] = ci1j;
            C[i + (j + 1) * n] = cij1;
            C[(i + 1) + (j + 1) * n] = ci1j1;

            C[(i + 2) + (j) * n] = ci2j;
            C[(i + 2) + (j + 1) * n] = ci2j1;
            C[(i + 2) + (j + 2) * n] = ci2j2;
            C[(i + 1) + (j + 2) * n] = ci1j2;
            C[(i) + (j + 2) * n] = cij2;

        }
        for (j = dim_B / 3 * 3; j < dim_B; ++j) {
            cij = C[i + j * n];
            ci1j = C[i + 1 + j * n];
            ci2j = C[i + 2 + j * n];
            for (k = 0; k < dim_C; ++k) {
                cij += A[k + i * n] * B[k + j * n];
                ci1j += A[k + (i + 1) * n] * B[k + j * n];
                ci2j += A[k + (i + 2) * n] * B[k + j * n];
            }
            C[i + j * n] = cij;
            C[i + 1 + j * n] = ci1j;
            C[i + 2 + j * n] = ci2j;
        }

    }
    for (i = dim_A / 3 * 3; i < dim_A; ++i) {
        for (j = 0; j < dim_B; ++j) {
            cij = C[i + j * n];
            for (k = 0; k < dim_C; ++k) {
                cij += A[k + i * n] * B[k + j * n];
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

  double *A_t = (double *) malloc(sizeof(double) * n * n);

  for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
           A_t[j + n * i] = A[i + n * j];
      }
  }

    for (int i = 0; i < n; i += BLOCK_SIZE) {
        for (int j = 0; j < n; j += BLOCK_SIZE) {
            for (int k = 0; k < n; k += BLOCK_SIZE) {
                int dim_A = min (BLOCK_SIZE, n - i);
                int dim_B = min (BLOCK_SIZE, n - j);
                int dim_C = min (BLOCK_SIZE, n - k);
                do_block_unroll_transpose(n, dim_A, dim_B, dim_C, A_t + k + i * n, B + k + j * n, C + i + j * n);
            }
        }
    }
    free(A_t);
}
