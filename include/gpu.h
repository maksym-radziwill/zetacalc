#include <cuComplex.h>
#include <cuda_runtime.h>

void cudaKernel(struct precomputation_table *,
		cuDoubleComplex *,
		int,
		cudaStream_t,
		int);

#define cudaCheckErrors(msg) \
  do { \
    cudaError_t __err = cudaGetLastError(); \
    if (__err != cudaSuccess) { \
      fprintf(stderr, "Fatal error: %s (%s at %s:%d)\n", \
	      msg, cudaGetErrorString(__err), \
	      __FILE__, __LINE__); \
      fprintf(stderr, "*** FAILED - ABORTING\n"); \
      exit(1); \
    } \
  } while (0)

#define THREAD_PER_BLOCK 32
#define A_BUF_SIZE 8
#define B_BUF_SIZE 8

struct precomputation_table {
  union {
    int number_of_log_terms : 4;
    int number_of_sqrt_terms : 4;
  } size;
  int block_size;
  double a[A_BUF_SIZE]; 
  double b[B_BUF_SIZE];
};


