


/*
 * Functions for computing the "chunk"
 *
 *      sum_{n=v}^{v + N - 1} exp(i*y*log(n)) / sqrt(n)
 * 
 * for y = t, t+delta, t+2*delta, ... , t+(M-1)*delta.
 * These "stage ma2" functions use taylor expansions on the summands
 * to achieve significant speedup as compared to individually
 * computing each summand to full precision. This method is only
 * efficient when n is >> t^(1/4). 
 *
 */


#include <queue>
#include <fstream>
#include "theta_sums.h"
#include "main_sum.h"
#include "log.h"

#include "config.h"
#if HAVE_CUDA
#include "gpu.h"
#endif

#include "quad-mpfr.h"
#include <quadmath.h>
#include <time.h>
#include "quad-imp.h"

using namespace std;



unsigned int stage_2_block_size(Double v, Double t) {
  //
  // For a given v and t, we determine a good block size to use
  // in stage 2 via the formula
  //
  //      block_size = min{ 5000, min( v / t^0.25 , v / 500^2) }
  //
  // The numbers 5000 and 500 in the formula are somwhat arbitrary and
  // can be adjusted.
  
  unsigned int block_size = (unsigned int)( min(v * pow(t, -.25), v/(500.0 * 500.0)  ) );
  if(block_size > 5*1024) return 5*1024;
  
  return block_size;
}

#if HAVE_CUDA
inline __attribute__((always_inline)) void precompute_a_b(mpz_t v, 
		    unsigned int *K, 
		    mpfr_t t, 
		    Double epsilon, 
		    struct precomputation_table * precompute) {

  
  
  //
  // A function to compute the block
  //
  //      sum_{n = v}^{v + K - 1} exp(i*t*log n) / sqrt(n)
  //
  // where K is calculated by calling stage_2_block_size.
  //

  
  
  precompute->size.number_of_log_terms = 0; 
  precompute->size.number_of_sqrt_terms = 0; 
  precompute->block_size = 0; 
  
  if(*K == 0 || *K == 1) return;
      
  Double vv = mpz_get_d(v);
  Double tt = mpfr_get_d(t, GMP_RNDN);
    
  // compute an appropriate block size 
  unsigned int block_size = min(stage_2_block_size(vv, tt), *K);

  precompute->block_size = block_size; 

  if(block_size < 2) {
    cout << "Error: in stage 2, computed a block size that is too small " << endl;
    cout << "Refusing to continue, and returning NAN, without setting K." << endl;
    return;
  }

  *K = block_size;

  //__float128 x_f = block_size / v_f; 
  Double x = block_size/vv;

  // The following code to estimate the number of terms that we need in the taylor expansion 
  // is useful (and fast) because it doesn't take any logarithms (instead we just grab the 
  // exponent of the relevant numbers to estimate the log base 2). 
  int logepsilon = fastlog2(epsilon);
  int logtt = fastlog2(tt);
  int logx = fastlog2(x);

  // estimate the number of terms needed in the taylor expansions of 
  // t * log(1 + k / v) and -0.5 * log(1 + k / v)
  int number_of_log_terms = (logepsilon - logtt)/logx;
  int number_of_sqrt_terms = logepsilon/logx;

  if(number_of_sqrt_terms > 8 || number_of_log_terms > 8)
    cout << "Warning : The size of A_TABLE in include/gpu.h needs to be increased beyond 8!" << endl;
 
  
  precompute->size.number_of_sqrt_terms = number_of_sqrt_terms;
  precompute->size.number_of_log_terms = number_of_log_terms;
  
  // arrays that will hold auxilliary coefficents in the Taylor expansions
  // of t * log(1 + k / v) and -0.5 * log(1 + k / v)
  
  // We want to accurately compute the quantities t/v mod pi. To do so
  // it should be enough to use log2(t) - log2(v) + 53 bits.
  int vsize = mpz_sizeinbase(v, 2);
  int precision = mpfr_get_exp(t) - vsize + 53;       

  
    mpfr_t z, mp_v_power, twopi_l, z1, twopi;

  if(precision >= 2*53){
    mpfr_init2(z, precision);               // We are going to change the
    mpfr_init2(mp_v_power, precision);      // precision on these variables,
    mpfr_init2(twopi_l, precision);         // so we can't use MPFR_DECL_INIT.
    mpfr_init2(z1, 53); 
    //    MPFR_DECL_INIT(z1, 53);
    mpfr_init2(twopi, precision);
    mpfr_const_pi(twopi, GMP_RNDN);
    mpfr_mul_si(twopi, twopi, 2, GMP_RNDN);
  }

  int precision0 = precision; 
  
  MPFR_DECL_INIT(one_over_v, precision);

  mpfr_set_z(one_over_v, v, GMP_RNDN);
  __float128 v_f = mpfr_get_float128(one_over_v); 
  __float128 one_over_v_f = 0; 

  // Not clear if any gain from this
  
  if(precision >= 2*53){
    mpfr_ui_div(one_over_v, 1, one_over_v, GMP_RNDN);
    one_over_v_f = mpfr_get_float128(one_over_v); 
  }else{
    one_over_v_f = 1/v_f; 
  }

  int sign = 1;
  double one_over_vv = 1.0/vv;
  double v_power = one_over_vv;

  if(precision >= 2*53){
    mpfr_set(mp_v_power, one_over_v, GMP_RNDN); // mp_v_power = 1 / v
    mpfr_set(twopi_l, twopi, GMP_RNDN); // two_pi_l = two_pi
  }
  
  // Code to compute auxilliary coefficients in the taylor expansion
  // of t * log(1 + k / v) mod 2_pi. The mpfr precision is adjusted on 
  // each iteration to improve efficiency.
  //
  
  __float128 mp_v_power_f = 0, twopi_l_f = 0, twopi_f = 0, t_f = 0, z_f = 0, z1_f = 0; 

  if(precision >= 53){
    //mp_v_power_f = mpfr_get_float128(one_over_v); 
    
    // Hardcoded value of 2pi
    ieee854_float128 f;
    f.words64.high = 4612128158286889681;
    f.words64.low = -8905435550453399104;
    twopi_l_f = f.value;            
    twopi_f = twopi_l_f;
    
    // This is a weird bug but we need
    // to first transfer t to a local variable
    
    MPFR_DECL_INIT(t0, 2*53);
    mpfr_set(t0, t, GMP_RNDN); 
    t_f = mpfr_get_float128(t0); 

    z_f = 0; 
    z1_f = 0;
    mp_v_power_f = one_over_v_f; 
  }
  
  for(int l = 1; l <= number_of_log_terms; l++) {
    
    if(precision >= 2*53){

      mpfr_set_prec(z, precision);
      mpfr_prec_round(mp_v_power, precision, GMP_RNDN);
      mpfr_prec_round(twopi_l, precision, GMP_RNDN);
      mpfr_mul(z, t, mp_v_power, GMP_RNDN); // z = t / v^l  
      mpfr_fmod(z1, z, twopi_l, GMP_RNDN); // z1 = t / v^l mod 2_pi_l
            
      precompute->a[l] = sign * mpfr_get_d(z1, GMP_RNDN)/l; // a[l] = (-1)^(l+1) * z1 / l
      mpfr_mul(mp_v_power, mp_v_power, one_over_v, GMP_RNDN); // mp_v_power = mp_v_power / v
      mpfr_add(twopi_l, twopi_l, twopi, GMP_RNDN); // twopi_l = twopi_l + twopi
      
      precision -= vsize;
      
      
      if(precision < 2*53){
	mp_v_power_f = mpfr_get_float128(mp_v_power); 
	twopi_l_f = mpfr_get_float128(twopi_l); 
      }
      
      if(precision < 53){
	v_power = (double) mpfr_get_d(mp_v_power, GMP_RNDN); 
      }
      
    }else{

      if(precision >= 53){ 
	z_f = t_f * mp_v_power_f;
	z1_f = fmodq(z_f, twopi_l_f); 	  
	z1_f = sign*z1_f / l; 
	double p = (double) z1_f; 
	precompute->a[l] = p; 
	mp_v_power_f = mp_v_power_f * one_over_v_f; 
	twopi_l_f = twopi_l_f + twopi_f;
	
	precision -= vsize; 
	
	if(precision < 53)
	  v_power = (double) mp_v_power_f; 
	
      }else{
	precompute->a[l] = sign*tt*v_power / l; 
	v_power = v_power * one_over_vv; 
      } 
    }

    sign = -sign;

  }
  
  // code to compute auxilliary coefficients in the taylor expansion of -0.5 * log(1 + k / v)
  // there is no need for mpfr here (because we're not doing and mod-ing by 2_pi)
  //

  Double s = 1;
  for(int l = 1; l <= number_of_sqrt_terms; l++) {
    s = s * (-1.0/vv); // s = (-1)^l / v^l 
    precompute->b[l] = .5 * s/l; // b[l] = 0.5 * (-1)^l / (l * v^l)
  }
  
  if(precision0 >= 2*53){
    mpfr_clear(mp_v_power);
    mpfr_clear(z);
    mpfr_clear(z1);
    mpfr_clear(twopi);
    mpfr_clear(twopi_l);
  }

  return;

}

int estimate_size(mpz_t v0, mpfr_t t, unsigned int N){
  mpz_t v;
  mpz_init(v);
  unsigned int N0 = N; 
  unsigned int K = N0; 
  double tt = mpfr_get_d(t, GMP_RNDN); 
  int size = 0; 
  
  mpz_set(v, v0);
  
  while(N0 > 0){
    double vv = mpz_get_d(v);

    if(K > 1)
      K = min(stage_2_block_size(vv, tt), K);

    size = size + 1; 
    
    N0 = N0 - K;
    mpz_add_ui(v,v,K);
    K = N0; 
  }

  return size; 
}

void reallocate(struct precomputation_table ** dev_precompute,
		cuDoubleComplex ** dev_current_terms,
		struct precomputation_table ** precompute,
		cuDoubleComplex ** host_current_terms_pre,
		Complex ** host_current_terms, int estimate){
  
    cudaFree(*dev_precompute);
    cudaFree(*dev_current_terms);
    cudaCheckErrors("cudaFree device"); 
    
    cudaFreeHost(*precompute);
    cudaFreeHost(*host_current_terms_pre);
    cudaFreeHost(*host_current_terms);
    cudaCheckErrors("cudaFree host"); 

    cudaMalloc((void **) dev_precompute, estimate*sizeof(struct precomputation_table));
    cudaMalloc((void **) dev_current_terms, estimate*sizeof(cuDoubleComplex));
    cudaCheckErrors("cudaMalloc device");                                                              
    
    cudaMallocHost((void **) precompute, estimate*sizeof(struct precomputation_table));   
    cudaMallocHost((void **) host_current_terms_pre, estimate*sizeof(cuDoubleComplex));   
    cudaMallocHost((void **) host_current_terms, estimate*sizeof(Complex));           
    cudaCheckErrors("cudaMalloc host");                                                                

    
}

extern void allocateTexture(); 

void allocate(struct precomputation_table ** dev_precompute,
	      cuDoubleComplex ** dev_current_terms,
	      struct precomputation_table ** precompute,
	      cuDoubleComplex ** host_current_terms_pre,
	      Complex ** host_current_terms, int estimate){


  allocateTexture();
  cudaCheckErrors("texture"); 


  cudaMalloc((void **) dev_precompute, estimate*sizeof(struct precomputation_table));
  cudaMalloc((void **) dev_current_terms, estimate*sizeof(cuDoubleComplex));
  cudaCheckErrors("cudaMalloc device");                                                              
  
  cudaMallocHost((void **) precompute, estimate*sizeof(struct precomputation_table));   
  cudaMallocHost((void **) host_current_terms_pre, estimate*sizeof(cuDoubleComplex));   
  cudaMallocHost((void **) host_current_terms, estimate*sizeof(Complex));           
  cudaCheckErrors("cudaMalloc host");                                                                


}
		     
int zeta_block_stage2(mpz_t v0,
		      unsigned int N,
		      mpfr_t t,
		      Double delta,
		      int M, Complex * S,
		      struct precomputation_table ** dev_precompute,
		      cuDoubleComplex ** dev_current_terms,
		      struct precomputation_table ** precompute,
		      cuDoubleComplex ** host_current_terms_pre,
		      Complex ** host_current_terms,
		      int initial_size, cudaStream_t stream, int id_gpu,
		      pthread_mutex_t * mutex_lock, int allocated_size) {
  
  //
  // A function to compute the the "chunk"
  //
  //      sum_{n=v0}^{v0 + N - 1} exp(it log n) / sqrt(n)
  //
  // in blocks of length K (this is done by calling zeta_block_stage2_basic repeatedly, which 
  // computes an appropriate K on each call)
  //
  
  // if the sum is empty return 0 

  if(N == 0) return 0;
  
  mpz_t v;
  mpz_init(v);
  
  mpz_set(v, v0);
  unsigned int K = N; 
  
  unsigned int N0 = N; 
  unsigned int K0 = K; 
  
  /* Get an (tight) estimate of how much memory we will need */


    

  int estimate = estimate_size(v,t,N); 
  
  /* If needed allocate/reallocate memory -- always
     use the same global pointer so that we don't need
     to constantly free/malloc */

  if(allocated_size == 0){
    allocate(dev_precompute, 
	     dev_current_terms, 
	     precompute, 
	     host_current_terms_pre, 
	     host_current_terms, estimate); 
    cudaStreamSynchronize(stream); // for textures
  }

  if(((allocated_size < estimate) || (allocated_size > 2*estimate)) 
     && allocated_size > 0)
    reallocate(dev_precompute, 
	       dev_current_terms, 
	       precompute, 
	       host_current_terms_pre, 
	       host_current_terms, estimate); 


  /* Precomputing the taylor expansions a & b */
  
  int j = 0; 
  

  while(N > 0){  
    precompute_a_b(v, &K, t, STAGE2_PRECISION, *precompute + j); 
    j = j + 1; 
    
    N = N - K;
    mpz_add_ui(v,v,K);
    K = N; 
  }
  
  if(j != estimate) cout << "Something is seriously messed up\n"; 

  /* Copying the table onto the device, 
     launching the kernel and copying back */

  cudaMemcpyAsync(*dev_precompute, 
		  *precompute, 
		  estimate*sizeof(struct precomputation_table),
		  cudaMemcpyHostToDevice, 
		  stream);
  cudaCheckErrors("memcpyAsync to device\n");
      
  cudaKernel(*dev_precompute, *dev_current_terms, estimate, stream, id_gpu); 
  cudaCheckErrors("cudaKernel launch"); 
    
  cudaMemcpyAsync(*host_current_terms_pre, 
		  *dev_current_terms, 
		  estimate*sizeof(cuDoubleComplex), 
		  cudaMemcpyDeviceToHost, 
		  stream);
  cudaCheckErrors("memcpyAsync from device\n"); 
    
  cudaStreamSynchronize(stream);
  cudaCheckErrors("cudaStream synchronize\n"); 

  /* Converting cuDoubleComplex to Complex */

  for(int i = 0; i < estimate; i++){      
    (*host_current_terms)[i] = 
      Complex(cuCreal((*host_current_terms_pre)[i]),
	      cuCimag((*host_current_terms_pre)[i]));
  }

  
  /* Returning to the original variables */
  
  K = K0; 
  N = N0; 
  mpz_set(v, v0);
  
  /* Executing the original loop */
    
  int i = 0; 
  
  while(N > 0){
    /* the current term is located in host_current_terms[i] */ 
    Complex current_term = (*host_current_terms)[i]; 
    /* But the host_current_term needs additional processing 
       that we were unwilling to handle on the GPU */
    double vv = mpz_get_d(v);
    current_term /= sqrt(vv); // We could handle this one on the GPU
    current_term *= exp_itlogn(v); 
    i += 1; 
      
    if(K == 0) current_term = 0.0;
      
    // if sum consists of 1 term, return value directly
    if(K == 1) current_term = exp_itlogn(v)/sqrt(mpz_get_d(v));
    
    double tt = mpfr_get_d(t, GMP_RNDN);
    
    // compute an appropriate block size 
    unsigned int block_size = min(stage_2_block_size(vv, tt), K);
    K = block_size; 
    
    Complex multiplier = exp(I*delta*log(mpz_get_d(v))); 
    
    if(current_term != current_term){
      cout << "Caught NaN\n";
      exit(0);
    }
    
    for(int l = 0; l < M; l++) {
      S[l] += current_term;
      current_term *= multiplier;
    }
    
    N = N - K;
    mpz_add_ui(v, v, K);
    K = N;
    
  }

  mpz_clear(v); 

  return estimate;     
}
#endif
