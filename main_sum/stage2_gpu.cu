#include "config.h"
#if HAVE_CUDA
#include <stdio.h>
#include <stdlib.h>
#include "gpu.h"

static texture <int2, 1, cudaReadModeElementType> k_power_texture;

#define TEXTURE_CUT_OFF 3200

static __inline__ __device__ double fetch_double(int2 p){
    return __hiloint2double(p.y, p.x);
}


__global__ void kernel
	   (struct precomputation_table * dev_precompute, 
	   cuDoubleComplex * dev_current_terms, int size){

     /* We want each thread to give rise to one value of current_terms */

     int bid = blockIdx.x;
     int tid = threadIdx.x;

     /* Check if we include or not bid == size */

     if(bid >= size) return;

     // fetch these values with thread 0
     // and then broadcast them using 

     int block_size;
     int number_of_log_terms;
     int number_of_sqrt_terms;

     if(threadIdx.x == 0){
     	 block_size = dev_precompute[bid].block_size;
         number_of_log_terms = dev_precompute[bid].number_of_log_terms;
         number_of_sqrt_terms = dev_precompute[bid].number_of_sqrt_terms;
     }

     __syncthreads();

     block_size = __shfl(block_size, 0);
     number_of_log_terms = __shfl(number_of_log_terms, 0);
     number_of_sqrt_terms = __shfl(number_of_sqrt_terms, 0); 

     if(number_of_sqrt_terms == 0 || number_of_log_terms == 0 || block_size == 0){
     			     dev_current_terms[bid] = make_cuDoubleComplex(0.0,0.0); 
			     return;			     
     }

     double a[A_BUF_SIZE]; 
     double b[B_BUF_SIZE];
     
     #pragma unroll 6
     for(int i = 1; i <= number_of_log_terms; i++)	
     	     a[i] = dev_precompute[bid].a[i];
	     
     for(int i = 1; i <= number_of_sqrt_terms; i++)
     	     b[i] = dev_precompute[bid].b[i];

     __shared__ cuDoubleComplex dev_S [2*THREAD_PER_BLOCK];
     dev_S[tid] = make_cuDoubleComplex(0.0,0.0); 
     dev_S[2*THREAD_PER_BLOCK - 1 - tid] = make_cuDoubleComplex(0.0,0.0); 

 /* Now we break down onto all threads in the block */

    double k_power = 1;
    double x = 0;
    double y = 0;
    double s,c,e;



      if(block_size <= TEXTURE_CUT_OFF){

     #pragma unroll 100
     for(unsigned int k = tid; k < block_size; k += blockDim.x){

	    x = 0;
	    y = 0;
	    
	    #pragma unroll 6
	    for(int l = 1; l <= number_of_log_terms; l++) {
		  y = __fma_rn(a[l], fetch_double(tex1Dfetch(k_power_texture, 8*k + l-1)), y); 

		  if(l <= number_of_sqrt_terms)
		     x = __fma_rn(b[l], fetch_double(tex1Dfetch(k_power_texture, 8*k + l - 1)), x); 
            }

	    sincos(y, &s, &c);
	    e = exp(x); 

   	    dev_S[tid] = cuCadd(dev_S[tid],
	    	      make_cuDoubleComplex(__dmul_rn(e,c),__dmul_rn(e,s)));

      }		  
      }else{
        for(unsigned int k = tid; k < block_size; k += blockDim.x){

            k_power = 1;
	    x = 0;
	    y = 0;
	    
	    #pragma unroll 6
	    for(int l = 1; l <= number_of_log_terms; l++) {
	          k_power = __dmul_rn (k_power, k); 
		  y = __fma_rn(a[l], k_power, y); 

		  if(l <= number_of_sqrt_terms)
		     x = __fma_rn(b[l], k_power, x); 
            }

	    sincos(y, &s, &c);
	    e = exp(x); 

   	    dev_S[tid] = cuCadd(dev_S[tid],
	    	      make_cuDoubleComplex(__dmul_rn(e,c),__dmul_rn(e,s)));

      }

      }


      __syncthreads();

	// This is the proper parallel way to do things 
	// but the simplistic addition below works just as well

	// For the commented version to work blockDim.x needs to be a power of two 


	       dev_S[tid] = cuCadd(dev_S[tid], dev_S[tid + 16]);
	       __syncthreads();
	       dev_S[tid] = cuCadd(dev_S[tid], dev_S[tid + 8]);
	       __syncthreads();
	       dev_S[tid] = cuCadd(dev_S[tid], dev_S[tid + 4]);
	       __syncthreads();
	       dev_S[tid] = cuCadd(dev_S[tid], dev_S[tid + 2]);
	       __syncthreads();
	       dev_S[tid] = cuCadd(dev_S[tid], dev_S[tid + 1]);
	       __syncthreads();


/*
      for(unsigned int s = blockDim.x/2; s > 0; s >>= 1){ 
   	  dev_S[tid] = cuCadd(dev_S[tid],dev_S[tid + s]); 
      }
*/


/*
      if(tid == 0){
      	     for(int i = 1; i < blockDim.x; i++)
	     	     dev_S[0] = cuCadd(dev_S[0],dev_S[i]); 
             
             // dev_current_terms[bid] = dev_S[0]; 
      }


//      __syncthreads();

*/

      /* Now we want to add all the local dev_S together */
      if(tid == 0)
            dev_current_terms[bid] = dev_S[0];             

	    return;

}

void allocateTexture(){

     double * k_power_table_host = (double *) malloc(8*TEXTURE_CUT_OFF*sizeof(double)); 
     double * k_power_table_device; 

  k_power_texture.normalized = false;  // don't use normalized values                                           
  k_power_texture.addressMode[0] = cudaAddressModeClamp; // don't wrap around indices                           
  k_power_texture.addressMode[1] = cudaAddressModeClamp;

     
     cudaMalloc((void **) &k_power_table_device, 8*TEXTURE_CUT_OFF*sizeof(double));	
     for(int k = 0; k < TEXTURE_CUT_OFF;k++){
     	     for(int l = 1; l <= 8; l++){
	     	     k_power_table_host[8*k + l - 1] = (double) pow(k,l); 
	     }
     }

     cudaMemcpy(k_power_table_device, 
     			k_power_table_host, 8*TEXTURE_CUT_OFF*sizeof(double), 
     				      cudaMemcpyHostToDevice);

     cudaBindTexture(NULL, k_power_texture, k_power_table_device, 8*TEXTURE_CUT_OFF*sizeof(double));

}

void cudaKernel
     (struct precomputation_table * dev_precompute, 
     cuDoubleComplex * dev_current_terms, int size,
     cudaStream_t stream, int id_gpu){

   int thread_per_block = THREAD_PER_BLOCK; 

   kernel<<<size,	
     	thread_per_block,
	0,
	stream>>>(dev_precompute, dev_current_terms, size);

}
#endif