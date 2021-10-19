const char *dgemm_desc = "Simple blocked dgemm.";

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




static void do_block(int lda, int M, int N, int K, double *A, double *B, double *C) {
    // For each row i of A
    for (int i = 0; i < M; ++i) {
        //For each column j of B
        for (int j = 0; j < N; ++j) {
            // Compute C(i,j)
            double cij = C[i + j * lda];
            for (int k = 0; k < K; ++k) {
                cij += A[i + k * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }
}

static void do_block_unroll(int lda, int M, int N, int K, double *A, double *B, double *C) {

    double cij;
    double ci1j;
    double cij1;
    double ci1j1;

    double ci2j;
    double ci2j1;
    double ci2j2;
    double ci1j2;
    double cij2;



    double t0, t1, t2, t3, t4, t5, t6, t7;
    int i, j, k;
    // For each row i of A
    for (i = 0; i < M / 3 * 3; i += 3) {
        //For each column j of B
        for (j = 0; j < N / 3 * 3; j += 3) {
            // Compute C(i,j)
            cij = C[i + j * lda];
            // Compute C(i+1,j)
            ci1j = C[(i + 1) + j * lda];
            // Compute C(i,j+1)
            cij1 = C[i + (j + 1) * lda];
            // Compute C(i+1,j+1)
            ci1j1 = C[(i + 1) + (j + 1) * lda];


            //Compute C(i+2, j)
            ci2j = C[(i + 2) + (j) * lda];
            //Compute C(i+2, j+1)
            ci2j1 = C[(i + 2) + (j + 1) * lda];
            //Compute C(i+2, j+2)
            ci2j2 = C[(i + 2) + (j + 2) * lda];
            //Compute C(i+1, j+2)
            ci1j2 = C[(i + 1) + (j + 2) * lda];
            //Compute C(i, j+2)
            cij2 = C[(i) + (j + 2) * lda];

            for (k = 0; k < K; k += 1) {
                //1st Row
                t0 = A[i + k * lda];
                //1st Col
                t1 = B[k + j * lda];
                //2nd Row
                t2 = A[(i + 1) + k * lda];
                //2nd Col
                t3 = B[k + (j + 1) * lda];
                //3rd Row
                t4 = A[(i + 2)+ k * lda];
                //3rd Col
                t5 = B[k + (j + 2) * lda]; 


                cij += t0 * t1;
                ci1j += t2 * t1;
                cij1 += t0 * t3;
                ci1j1 += t2 * t3;

                ci2j += t4 * t1;
                ci2j1 += t4 * t3;
                ci2j2 += t5 * t4;
                ci1j2 += t2 * t5;
                cij2 += t0 * t5;

            }


            C[i + j * lda] = cij;
            C[i + 1 + j * lda] = ci1j;
            C[i + (j + 1) * lda] = cij1;
            C[(i + 1) + (j + 1) * lda] = ci1j1;

            C[(i + 2) + (j) * lda] = ci2j;
            C[(i + 2) + (j + 1) * lda] = ci2j1;
            C[(i + 2) + (j + 2) * lda] = ci2j2;
            C[(i + 1) + (j + 2) * lda] = ci1j2;
            C[(i) + (j + 2) * lda] = cij2;

        }
//        The odd col of matrix B, this should only have ONE iteration!
        for (j = N / 3 * 3; j < N; ++j) {
            cij = C[i + j * lda];
            ci1j = C[i + 1 + j * lda];
            ci2j = C[i + 2 + j * lda];
            for (k = 0; k < K; ++k) {
                cij += A[i + k * lda] * B[k + j * lda];
                ci1j += A[(i + 1) + k * lda] * B[k + j * lda];
                ci2j += A[(i + 2) + k * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
            C[i + 1 + j * lda] = ci1j;
            C[i + 2 + j * lda] = ci2j;

        }

    }

    //The odd row of matrix A, this should only have ONE iteration!
    for (i = M / 3 * 3; i < M; ++i) {
        //For each column j of B
        for (j = 0; j < N; ++j) {
            cij = C[i + j * lda];
            for (k = 0; k < K; ++k) {
                cij += A[i + k * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }
}


static void do_block_unroll_transpose(int lda, int M, int N, int K, double *A, double *B, double *C) {

    double cij;
    double ci1j;
    double cij1;
    double ci1j1;

    double ci2j;
    double ci2j1;
    double ci2j2;
    double ci1j2;
    double cij2;



    double t0, t1, t2, t3, t4, t5, t6, t7;
    int i, j, k;
    // For each row i of A
    for (i = 0; i < M / 3 * 3; i += 3) {
        //For each column j of B
        for (j = 0; j < N / 3 * 3; j += 3) {
            // Compute C(i,j)
            cij = C[i + j * lda];
            // Compute C(i+1,j)
            ci1j = C[i + 1 + j * lda];
            // Compute C(i,j+1)
            cij1 = C[i + (j + 1) * lda];
            // Compute C(i+1,j+1)
            ci1j1 = C[(i + 1) + (j + 1) * lda];


            //Compute C(i+2, j)
            ci2j = C[(i + 2) + (j) * lda];
            //Compute C(i+2, j+1)
            ci2j1 = C[(i + 2) + (j + 1) * lda];
            //Compute C(i+2, j+2)
            ci2j2 = C[(i + 2) + (j + 2) * lda];
            //Compute C(i+1, j+2)
            ci1j2 = C[(i + 1) + (j + 2) * lda];
            //Compute C(i, j+2)
            cij2 = C[(i) + (j + 2) * lda];

            for (k = 0; k < K; k += 1) {
                //1st Row
                t0 = A[k + i * lda];
                //1st Col
                t1 = B[k + j * lda];
                //2nd Row
                t2 = A[k + (i + 1) * lda];
                //2nd Col
                t3 = B[k + (j + 1) * lda];
                //3rd Row
                t4 = A[k + (i + 2) * lda];
                //3rd Col
                t5 = B[k + (j + 2) * lda]; 


                cij += t0 * t1;
                ci1j += t2 * t1;
                cij1 += t0 * t3;
                ci1j1 += t2 * t3;

                ci2j += t4 * t1;
                ci2j1 += t4 * t3;
                ci2j2 += t5 * t4;
                ci1j2 += t2 * t5;
                cij2 += t0 * t5;

            }


            C[i + j * lda] = cij;
            C[i + 1 + j * lda] = ci1j;
            C[i + (j + 1) * lda] = cij1;
            C[(i + 1) + (j + 1) * lda] = ci1j1;

            C[(i + 2) + (j) * lda] = ci2j;
            C[(i + 2) + (j + 1) * lda] = ci2j1;
            C[(i + 2) + (j + 2) * lda] = ci2j2;
            C[(i + 1) + (j + 2) * lda] = ci1j2;
            C[(i) + (j + 2) * lda] = cij2;

        }
//        The odd col of matrix B, this should only have ONE iteration!
        for (j = N / 3 * 3; j < N; ++j) {
            cij = C[i + j * lda];
            ci1j = C[i + 1 + j * lda];
            ci2j = C[i + 2 + j * lda];
            for (k = 0; k < K; ++k) {
                cij += A[k + i * lda] * B[k + j * lda];
                ci1j += A[k + (i + 1) * lda] * B[k + j * lda];
                ci2j += A[k + (i + 2) * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
            C[i + 1 + j * lda] = ci1j;
            C[i + 2 + j * lda] = ci2j;

        }

    }

    //The odd row of matrix A, this should only have ONE iteration!
    for (i = M / 3 * 3; i < M; ++i) {
        //For each column j of B
        for (j = 0; j < N; ++j) {
            cij = C[i + j * lda];
            for (k = 0; k < K; ++k) {
                cij += A[k + i * lda] * B[k + j * lda];
            }
            C[i + j * lda] = cij;
        }
    }

}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */
void square_dgemm(int lda, double *A, double *B, double *C) {
  
  int l1_cache = 32*1024; //bytes
  int size_of_double = 8; //bytes
  double no_of_doubles_in_l1 = l1_cache/size_of_double;
  int BLOCK_SIZE = sqrt(no_of_doubles_in_l1/3); 

  double *A_t = (double *) malloc(sizeof(double) * lda * lda);

  //Transpose A -> row-major format
  for (int i = 0; i < lda; i++) {
      for (int j = 0; j < lda; j++) {
           A_t[j + lda * i] = A[i + lda * j];
      }
  }


    // For each block-row of A
    for (int i = 0; i < lda; i += BLOCK_SIZE) {
        // For each block-column of B
        for (int j = 0; j < lda; j += BLOCK_SIZE) {
            // Accumulate block dgemms into block of C
            for (int k = 0; k < lda; k += BLOCK_SIZE) {
                // Correct block dimensions if block "goes off edge of" the matrix
                int M = min (BLOCK_SIZE, lda - i);
                int N = min (BLOCK_SIZE, lda - j);
                int K = min (BLOCK_SIZE, lda - k);
                // Perform individual block dgemm
                //do_block(lda, M, N, K, A + i + k*lda, B + k + j*lda, C + i + j*lda);
                //do_block_unroll(lda, M, N, K, A + i + k * lda, B + k + j*lda, C + i + j*lda);
                //do_block_unroll_transpose(lda, M, N, K, A_t + k + i * lda, B + k + j * lda, C + i + j * lda);
            }
        }
    }
    free(A_t);
}