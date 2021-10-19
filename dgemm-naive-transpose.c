/* 
    Please include compiler name below (you may also include any other modules you would like to be loaded)

COMPILER= gnu

    Please include All compiler flags and libraries as you want them run. You can simply copy this over from the Makefile's first few lines
 
CC = cc
OPT = -O3
CFLAGS = -Wall -std=gnu99 $(OPT)
MKLROOT = /opt/intel/composer_xe_2013.1.117/mkl
LDLIBS = -lrt -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_lp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group -lpthread -lm

*/
#include <stdlib.h>
const char* dgemm_desc = "Naive, three-loop dgemm tranpose.";

void square_dgemm (int n, double* A, double* B, double* C)
{
  double *A_t = (double *) malloc(sizeof(double) * n * n);

  for (int i = 0; i < n; i++) {
      for (int j = 0; j < n; j++) {
           A_t[j + n * i] = A[i + n * j];
      }
  }
  for (int i = 0; i < n; ++i)
    for (int j = 0; j < n; ++j) 
    {
      double cij = C[i+j*n];
      for( int k = 0; k < n; k++ ){
        cij += A_t[k+i*n] * B[k+j*n];
      }
      C[i+j*n] = cij;
    }
}
