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

const char* dgemm_desc = "Naive, three-loop dgemm.";
#define min(a,b) (((a)<(b))?(a):(b))

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format.
 * On exit, A and B maintain their input values. */    
void matmultiply (int n, int dim_A, int dim_B, int dim_C, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < dim_A; ++i){
    /* For each column j of B */
    for (int j = 0; j < dim_B; ++j) 
    {
      /* Compute C(i,j) */
      double cij = C[i + j * n];
      for( int k = 0; k < dim_C; ++k ){
        cij += A[i + k * n] * B[k + j * n];
      }
      C[i + j * n] = cij;
    }
  }
}

void matmultiply_register(int n, int dim_A, int dim_B, int dim_C, double* A, double* B, double* C)
{
  /* For each row i of A */
  for (int i = 0; i < dim_A; i +=2){
    /* For each column j of B */
    for (int j = 0; j < dim_B; j +=2) 
    {
      /* Compute C(i,j) */
      double cij = C[i + j * n];
      // Compute C(i+1,j)
      double ci1j = C[(i + 1) + j * n];
      // Compute C(i,j+1)
      double cij1 = C[i + (j + 1) * n];
      // Compute C(i+1,j+1)
      double ci1j1 = C[(i + 1) + (j + 1) * n];
      for( int k = 0; k < dim_C; ++k ){
        cij += A[i + k * n] * B[k + j * n];
        ci1j += A[(i + 1) + k * n] * B[k + j * n];
        cij1 += A[i + k * n] * B[k + (j+1) * n];
        ci1j1 += A[(i+1) + k * n] * B[k + (j+1) * n];
      }
      C[i + j * n] = cij;
      C[(i + 1) + j * n] = ci1j;
      C[i + (j + 1) * n] = cij1;
      C[(i + 1) + (j + 1) * n] = ci1j1;
    }
  }
}
void square_dgemm (int n, double* A, double* B, double* C)
{
  int l1_cache = 32*1024; //bytes
  int size_of_double = 8; //bytes
  double no_of_doubles_in_l1 = l1_cache/size_of_double;
  int stride = sqrt(no_of_doubles_in_l1/3); 
  /* For each block-row of A */ 
  for (int i = 0; i < n; i += stride)
    /* For each block-column of B */
    for (int j = 0; j < n; j += stride)
      /* Accumulate block dgemms into block of C */
      for (int k = 0; k < n; k += stride)
      {
        /* Correct block dimensions if block "goes off edge of" the matrix */
        int dim_A = min (stride, n-i);
        int dim_B = min (stride, n-j);
        int dim_C = min (stride, n-k);

        /* Perform individual block dgemm */
        matmultiply_register(n, dim_A, dim_B, dim_C, A + i + k*n, B + k + j*n, C + i + j*n);
      }
}
