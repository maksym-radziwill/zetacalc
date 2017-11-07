/*
 * 
 * A description of main_sum.cc: 
 *
 * 1. zetacalc is first asked to compute a partial zeta sum. This is done by feeding zetacalc an input file, 
 * or from the commandline. 
 *
 * 2. The function partial_zeta_sum is then called. It decides which stage (i.e. method) to use for 
 * which piece of the partial sum; here:
 * 
 *   stage 1 is direct evaluation using exp_itlogn and mpfr
 *   stage 2 is direct evaluation using taylor expansions and mpfr
 *   stage 3 uses the theta algorithm
 *      
 * 3. partial_zeta_sum_stage (templated function) handles the pieces corresponding to stages 1, 2, and 3
 * (notice, almost always, only one piece is not empty). This is done by
 *
 *      a. calling zeta_sum_thread to create threads to compute "chunks" 
 *      b. creating sum_data_t structs (templated), which keep track of the next available "chunk", 
 *         and handle thread locking/unlocking & "reporting".
 *
 * 4. In turn, zeta_sum_thread calls zeta_block_stage1, or zeta_block_stage2, or zeta_block_stage3. These functions 
 * grab the next available "chunk" in the partial sum (by calling sum_data_t->next_block), they call
 * the function compute_taylor_coefficients, then compute the "chunk" in smaller blocks using the appropriate 
 * stage (i.e. method).
 * 
 */

/* Currently serialization is sometimes correct and sometimes incorrect
   It's not clear what this behaviour is due to -- my guess is that it has
   to do with an untimely handling of SIGHUP and this signal should be catched
   by the program and then exited gracefully */
/* No but loading the same file gives rise to different answers -- this doesn't
   quite make sense */

/* For serialization should also check if files with larger N parameter are present and
   if yes then just load those */

#include "config.h"

#if HAVE_MPI
#include <mpi.h>
#endif

#include <thread>
#include <iostream>
#include <random>
#include <sys/types.h>
#include "theta_sums.h"
#include "main_sum.h"
#include "log.h"

#if HAVE_CUDA
#include "gpu.h"
#endif
#include <fstream>
#include <csignal>

#include <unistd.h>
#include <sys/syscall.h>
#define gettid() syscall(SYS_gettid)

using namespace std;

typedef std::mt19937 RNG;

void stage_2_start(mpz_t v, mpfr_t t) {
  //
  // Computes the starting point in the main sum for stage 2 and 
  // stores it in v. The starting point is computed according to 
  // the formula:
  //
  //      v =  max{ floor( 3 * t^{1/4} ) , 1100000 } 
  //
  // With this choice, the block size in stage 2 will be at least 3.
  // The appearance of 1100000 in the formula is to ensure that the
  // Taylor expansions used in stage 2 don't get too long (which they
  // can if v is too small).
  //

  /* Could also introduce a stage1.5 where we do the GPU computation on
     the CPU but maybe that is... stupid ? */
  
  stage_3_start(v,t);
  mpz_div_ui(v, v,500000);

  if(mpz_cmp_ui(v,1100000) < 0) mpz_set_ui(v, 1100000); 

  return; 
  
  
  mpfr_t x;
  mpfr_init2(x, mpfr_get_prec(t));
  
  mpfr_root(x, t, 4, GMP_RNDN); // x = t^(1/4)
  mpfr_mul_ui(x, x, 3u, GMP_RNDN); // x = 3 * t^(1/4)
  mpfr_get_z(v, x, GMP_RNDN); // v = floor(3 * t^(1/4)
  

  mpfr_set(x, t, GMP_RNDN); 
  mpfr_log10(x, x, GMP_RNDN);  

  // Adjust the starting point of stage2 so that we can augment the block_size
  // in stage2 without running out of memory

  int multiplier = (int) mpfr_get_d(x, GMP_RNDN); 
  multiplier -= 29; 
  // the -6 is somewhat arbitrary
  if(multiplier < -6){
    if(mpz_cmp_ui(v, 1100000u) < 0) mpz_set_ui(v, 1100000u); 
  }else{
    if(multiplier < 1) multiplier = 0; 
    multiplier = pow(10, multiplier); 
    if(mpz_cmp_ui(v, multiplier*11000000u) < 0) mpz_set_ui(v, multiplier*11000000u);
  }
  mpfr_clear(x);


}


void stage_3_start(mpz_t v, mpfr_t t) {
  //
  // A function to compute the starting point in the main sum for the stage 3 and 
  // store it in v. The starting point is computed according to the formula:
  //
  //      v = 1200 * t^(1/3)
  //
  // With this choice, the block size in stage 3 will be at least 1200.
  //
  
  MPFR_DECL_INIT(x, mpfr_get_prec(t));
  
  mpfr_cbrt(x, t, GMP_RNDN); // x = t^(1/3)
  
  /* WARNING : Previously the multiplier below was 1200 not 34*1200 */
  /* We adjusted it to 34*1200 so that t = 10^30 is completely in the 
     range of stage2 sums */

  mpfr_mul_ui(x, x, 100*1200u, GMP_RNDN);  // x = stage_3_start * t^(1/3)

  /* Should carefully decide if the threshold is 10^14 or 10^15 ? */
  
  if(mpfr_cmp_d(x,5*1000000000000000) > 0){
    mpfr_set_d(x, 5*1000000000000000, GMP_RNDN); 
  }

  mpfr_get_z(v, x, GMP_RNDN); // v = floor( stage_3_start * t^(1/3) )
  
}

void draw_bar(int s, int p){
  cout << fixed;
  cout.precision(1); 
  int pp = 4*p/100;
  if (p < 0) return; 
  cout << "\rStage " << s << " : [";
  for(int i = 0; i < pp; i++){
    cout << "#"; 
  }
  for(int i = pp; i < 40; i++){
    cout << " "; 
  }
  cout << "] " << (double) p/10.0 << " %     ";
  
  cout.flush();
}


int estimate_size2(mpz_t v0, mpfr_t t, unsigned int N){
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

static int closing = 0;

void signalHandler( int signum ) {
  cout << "\rSaving state and exiting...                          ";
  cout.flush();
  closing = 1;
#if HAVE_MPI
  MPI_Bcast(&closing, 1, MPI_INT, 0, MPI_COMM_WORLD); 
#endif
}

string prepare_string(string filename, int gpus, int number_of_gpu_threads,
		      int cpus, int number_of_cpu_threads, int total_cores,
		      int total_gpus, mpz_t start1, mpz_t len1){
  return
    filename + "," 
    + to_string(gpus) + "," 
    + to_string(number_of_gpu_threads) + ","
    + to_string(cpus) + "," 
    + to_string(number_of_cpu_threads) + ","
    + to_string(total_cores) + "," 
    + to_string(total_gpus) + "," 
    + mpz_get_str(NULL, 10, start1) + "," 
    + mpz_get_str(NULL, 10, len1);
  
}


void serialize_local(string filename, Complex * S, int M){
  ofstream out;
  out.open(filename,std::ofstream::out);
  out.precision(2000);
  for(int i = 0; i < M; i++){
    out << S[i] << endl; 
  }
  out.flush();
  out.close();
}

void deserialize_local(string filename, Complex * S, int M){
    ifstream in;
    in.open(filename, std::ifstream::in);
    in.precision(2000);
    for(int i = 0; i < M; i++)
      in >> S[i]; 
    in.close();
}

bool is_file_exist(const char *fileName)
{
  std::ifstream infile(fileName);
  return infile.good();
}



template<int stage> struct sum_data_t {
  // 
  // A struct to help keep track of threads. It contains the function next_block, 
  // which increments the starting point of the next available "chunk", and the function 
  // report (to extract the result of the "chunk" computation)
  //
  
  int M; // # of gird points where partial sum will be computed
  complex<double> * S; // storage space for the values of the partial sum at t, t + delta, ..., t + (M-1)*delta
  pthread_mutex_t * report_mutex; // to handle thread locking/unlocking in preparation for a block computation
  pthread_mutex_t * next_mutex; // to manage threads 
  
  unsigned int stage2_max_size;
  
  mpz_t next; // the starting point of the next available block
  mpz_t end; // the end point of the partial sum (which consists of several blocks)
  double length; // the length (= # of terms) of the next available block
  mpfr_t t; // the height t where the block is computed
  double tt; // a double version of t for double-precision arithmetic
  double delta; // to compute the block at t, t + delta, t + 2*delta, ... , t + (M-1) * delta
  double epsilon; // a precision parameter that is passed to the theta algorithm 
  
  // to help with debugging 
  int percent_finished;

  string filename; 
  
  /* GPU Data -- each thread get its own CUDA stream */
#if HAVE_CUDA
  cudaStream_t * cudaStreams;  // Place holder for stream data
#endif
  int number_of_gpu_threads; 
  int number_of_cpu_threads; 
  int numStreams; // number of streams -> set to number_of_threads
  int gpus; // number of GPUS  
  
  // a struct constructor: sets variable values and initializes a thread
  sum_data_t(mpz_t start, mpz_t _length, mpfr_t _t, double _delta , int _M, complex<double> * _S, double _epsilon, int num_cpu_th, int num_gpu_th, int num_gpus, string file) {
    filename = file;
    
    mpz_init(next);
    mpz_init(end);
    mpfr_init2(t, mpfr_get_prec(_t));
        
    mpz_set(next, start); // next = start
    mpz_add(end, start, _length); // end = start + _length
    mpfr_set(t, _t, GMP_RNDN);  
    tt = mpfr_get_d(t, GMP_RNDN); 
    length = mpz_get_d(_length);
    
    delta = _delta;
    M = _M;
    S = _S;
    
    stage2_max_size = 1000000;
    
    epsilon = _epsilon;
    report_mutex = new pthread_mutex_t;
    next_mutex = new pthread_mutex_t;
    pthread_mutex_init(report_mutex, NULL);
    pthread_mutex_init(next_mutex, NULL);
    percent_finished = -1;
    
    /* New GPU data initialization */
    gpus = num_gpus; 
    number_of_gpu_threads = num_gpu_th; 
    number_of_cpu_threads = num_cpu_th;
#if HAVE_CUDA
    cudaStreams = new cudaStream_t [number_of_gpu_threads];
#endif
    /* Initializing cudaStreams now with cudaCreateStream would create
       a bug if we are using multiple GPU's -- the GPU device has to be
       set first */
  }
  
  // a struct destructor
  ~sum_data_t() {
    pthread_mutex_destroy(report_mutex);
    pthread_mutex_destroy(next_mutex);
    delete report_mutex;
    delete next_mutex;

    // Problem?
#if HAVE_CUDA
    delete cudaStreams;
#endif
    mpz_clear(next);
    mpz_clear(end);
    mpfr_clear(t);
  }

  void serialize(string file){

    /* Still need to save t and also
       create some kind of serialization global file
       where we will co-ordinate that the same value of t
       is being looked at , and that the number of threads is the same
       and the number of cpu's and gpu's */ 

    ofstream out;
    out.open(file,std::ofstream::out);
    out.precision(2000);
    out << M << endl
	<< length << endl
	<< tt << endl 
        << delta << endl
	<< epsilon << endl
	<< percent_finished << endl
	<< number_of_cpu_threads << endl
	<< number_of_gpu_threads << endl
	<< gpus << endl
        << next << endl
	<< end << endl
	<< stage2_max_size << endl;
    out.flush();
    out.close();
    
    FILE * fd = fopen(file.c_str(), "a");
    mpfr_out_str(fd, 10, 0, t, GMP_RNDD);
    fflush(fd);
    fclose(fd);

  }

  void deserialize(string file){
    string t_str; 
    ifstream in;
    in.open(file, std::ifstream::in);
    in.precision(2000);
    in >> M >> length >> tt >> delta >> epsilon 
       >> percent_finished >> number_of_cpu_threads 
       >> number_of_gpu_threads >> gpus;
    in >> next >> end >> stage2_max_size;
    in >> t_str; 
    in.close(); 
    mpfr_set_str(t, t_str.c_str(), 10, GMP_RNDN); 
  }  
  
  // locks thread, computes and returns block_size and stores the starting point of next 
  // available "chunk", then unlocks thread 
  unsigned long next_block(mpz_t start) {
    pthread_mutex_lock(next_mutex);
    unsigned long max_block_size;
    

    if(stage == 1) { max_block_size = 10000;}
    /* We increase the stage2 block_size as it augments GPU throughput
       at the later stages of the computation, say once we are 1/4 done */
    else if(stage == 2) {
      // This is a bit hacky and is done to transfer bigger chunks of memory
      // at once later in the process

      // If we are tight on memory augment 10000000
      // However the choice below should suffice as long as there is at least 4G
      // of RAM

      //      if(stage2_max_size != 16000000)
      //	if(mpz_cmp_ui(start, 10000000) > 0)
      //	  stage2_max_size = 16000000;
      
      max_block_size = 2*16000000;

      //      max_block_size = stage2_max_size; 
      
    } // this might be a problem memory problem
    else if(stage == 3) { max_block_size = 1000000;}
    else {
      cout << "this segment of the code should never be reached, exiting" << endl;
      exit(-1);
    }
    
    unsigned int block_size;
    mpz_sub(start, end, next); /* start = end - next (notice this is
				  an abuse of notation since now start
				  denotes the length of the remaining
				  partial sum */
    
    if(mpz_cmp_ui(start, max_block_size) < 0) { /* if start < max_block_size,
						   set block size = start */
      block_size = mpz_get_ui(start);
    }
    else {
      block_size = max_block_size;
    }
    
    double remainder = mpz_get_d(start);
    int current_percent_finished = 1000 * (1.0 - remainder/length);
    if(percent_finished != current_percent_finished)
      draw_bar(stage, current_percent_finished);       
    
    percent_finished = current_percent_finished;
    
    mpz_set(start, next);               /* start = next (start is now the
					   beginning point of the remainder
					   of the partial sum) */
    
    mpz_add_ui(next, next, block_size); // next = next + block_size
    
    
    pthread_mutex_unlock(next_mutex);
    return block_size;
  }
  
  // once a thread is done, it "reports" its output to be added to the array S 
  // that was created with the struct 
  void report(complex<double> * S2) {
    pthread_mutex_lock(report_mutex);
    
    for(int m = 0; m < M; m++) {
      S[m] += S2[m];
    }
    
    pthread_mutex_unlock(report_mutex);
  }
};

// This function is declared here and defined at the end of the file
template<int stage> void * partial_zeta_sum_stage(void * data); 

/* Also need to add case when total_gpus = 0 */

Complex partial_zeta_sum(mpz_t start, mpz_t length, mpfr_t t, Double & delta, int &M, Complex * S, int Kmin, int number_of_cpu_threads, int number_of_gpu_threads, int gpus, string filename) {
  
  signal(SIGINT, signalHandler);
  signal(SIGTERM,signalHandler);
  
  //
  // This function computes the partial sum:
  //
  //          \sum_{n=start}^{start + length - 1} exp(i*y*log n)/sqrt(n)
  //
  // for y = t,t+delta,t+2*delta,...,t+(M-1)*delta, then stores the resulting 
  // values in S. It also returns the value of the partial sum at y = t
  //

  mpz_t n2, n3, end, N1, N2, N3;
  mpz_init(n2);
  mpz_init(n3);
  mpz_init(end);
  mpz_init(N1);
  mpz_init(N2);
  mpz_init(N3);
  
  stage_2_start(n2, t); // n2 is the starting point of stage_2 

  stage_3_start(n3, t); // n3 is the starting point of stage_3
    
  // stage 1 contribution to the partial sum comes from start <= n < n2,
  // stage 2 from n2 <= n < n3, and stage 3 from n3 <= n < end. The "if 
  // statements" below reset n2 & n3 appropriately to mark the start & 
  // end points of stages 2 & 3.
  //

  /* Init MPI and adjust according to capabilities */

  int cpus = number_of_cpu_threads; // thread::hardware_concurrency();

#if HAVE_MPI


  int process_id;
  int max_proc;
  MPI_Comm_size(MPI_COMM_WORLD, &max_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

  //  printf("Process id %d out of %d\n", process_id, max_proc); 

  int has_cpu[max_proc]; 
  int has_gpu[max_proc];
  for(int i = 0; i < max_proc; i++){
    has_gpu[i] = 0;
    has_cpu[i] = 0; 
  }
  
  has_gpu[process_id] = gpus;
  has_cpu[process_id] = cpus; 
  
  int has_gpu_global[max_proc]; 
  int has_cpu_global[max_proc]; 
  
  MPI_Reduce(&has_gpu, &has_gpu_global, max_proc, MPI_INT, MPI_SUM, 0,
	     MPI_COMM_WORLD);

  MPI_Reduce(&has_cpu, &has_cpu_global, max_proc, MPI_INT, MPI_SUM, 0,
	     MPI_COMM_WORLD); 
  
  /* The code here assumes one MPI process per node */

  int total_gpus = 0; 
  int total_cores = 0; 
  
  MPI_Reduce(&gpus, &total_gpus, 1, MPI_INT, MPI_SUM, 0,
  	     MPI_COMM_WORLD);
  MPI_Reduce(&cpus, &total_cores, 1, MPI_INT, MPI_SUM, 0, 
  	     MPI_COMM_WORLD); 

  MPI_Bcast(&total_gpus, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD); 
  MPI_Bcast(&total_cores, 1, MPI_INT, 0, MPI_COMM_WORLD); 
  MPI_Barrier(MPI_COMM_WORLD); 
  MPI_Bcast(&has_gpu_global, max_proc, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD); 
  MPI_Bcast(&has_cpu_global, max_proc, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD); 
  
  if(process_id == 0){
    for(int i = 0; i < max_proc; i++){
      printf("host %d has %d gpus and %d cpus\n", i, has_gpu_global[i], has_cpu_global[i]);
    }
    printf("total number of cpu %d\ntotal numbers of gpus %d\n", total_cores, total_gpus); 
    
    if(total_gpus == 0){
      printf("There are no GPU's on this cluster -- stage2 code will not be able to run\n"
	     "and the results will be incorrect. Aborting\n"); 
      MPI_Finalize(); 
      exit(0); 
    }
  }
  
  
  int offset_gpu = 0; 
  for(int i = 0; i < process_id; i++){
    offset_gpu += has_gpu_global[i]; 
  }

  int offset_cpu = 0;
  for(int i = 0; i < process_id; i++){
    offset_cpu += has_cpu_global[i]; 
  }
#else
  int offset_cpu = 0; 
  int offset_gpu = 0; 
  int process_id = 0; 
  int total_gpus = gpus; 
  int total_cores = cpus; 
#endif

  // Here we would like to dissect into intervals that are perhaps 2*total_gpus shorter
  // and also 2*total_cpus shorter and iterate this construction twice
#if HAVE_MPI
  double ResultReS[M];
  double ResultImS[M];
  
  for(int i = 0; i < M; i++){
    ResultReS [i] = 0;
    ResultImS [i] = 0; 
  }
#endif
  
  mpz_t len0;
  mpz_t start0;
  mpz_t len1;
  mpz_t start1;

  mpz_init(len0); 
  mpz_init(start0);
  mpz_init(len1);
  mpz_init(start1);

  // we carry out a precomputation, which is needed in conjunction with a 
  // modification of Fenynman's algorithm for computing exp(i*t*log(n)); see 
  // log.cc  

  create_exp_itlogn_table(t);


  
  /* If tiny value stupid to use MPI and also leads to incorrect results */
  /*
  mpz_add(end, start, length);
  if(mpz_cmp_ui(end, 1000000000) < 0){
    number_of_cpu_threads = 1; 
    total_cores = 1;
  }
  */


  
  /* Bugs if small values and MPI used heavily... because we subtract */
  
  
  mpz_sub(len0, n2, start);
  mpz_mod_ui(len0, len0, total_cores);
  mpz_sub(n2, n2, len0);
  
    
  
  mpz_sub(len0, n3, n2); // len0 = lenth of stage2
  mpz_mod_ui(len0, len0, total_gpus); // len0 = length of stage2 % total_gpus
  mpz_sub(n3, n3, len0); // n2 = start of stage2 + ((length of stage2) % total_gpus)
  
  
  mpz_add(end, start, length); // end = start+length, is the endpoint of the partial sum 
  // (so the last term in the sum is n = end - 1)

  
  
  // Do a similar adjustement for stage3
  
  if(mpz_cmp(end,n3) < 0) 
    mpz_set(n3, end); // if end < n3, set n3 = end
  if(mpz_cmp(start,n3) > 0) 
    mpz_set(n3, start); // if start > n3, set n3 = start
  if(mpz_cmp(end,n2) < 0) 
    mpz_set(n2, end); // if end < n2, set n2 = end
  if(mpz_cmp(start,n2) > 0) 
    mpz_set(n2, start); // if start > n2, set n2 = start

      
  mpz_sub(N1, n2, start); // N1 = n20 - start, is the length of stage 1 sum
  mpz_sub(N2, n3, n2); // N2 = n3 - n2, is the length of stage 2 sum
  mpz_sub(N3, end, n3); // N3 = end - n3, is the length of stage 3 sum

  if(process_id == 0){
    cout << "Stage 1 range = [" << start << "," << n2 << "]\n";

    if(mpz_cmp(n2,n3) == 0)
      cout << "Stage 2 range = empty\n";
    else
      cout << "Stage 2 range = (" << n2 << "," << n3 << "]\n";

    if(mpz_cmp(n3,end) == 0)
      cout << "Stage 3 range = empty\n";
    else
      cout << "Stage 3 range = [" << n3 << "," << end << "]\n";
  }
    
  
  Complex * S1 = new Complex [M]; 
  Complex * S2 = new Complex [M]; 
  Complex * S3 = new Complex [M]; 

  pthread_t threads[3];
  
  // with the current layout there are no advantages to using thread
  // but the code has been written in such a way so that
  // the snippets below can be rearranged to start stage 2 and 3
  // in paralell -- this can be advantageous on some architectures
  // since the use of the gpu and cpu is somewhat independent
  // in particular this should be advantageous if the number of cores
  // exceeds the number 4 times the number of gpus ... 

  /* Would make sense to actually do this computation on process_id == 1 say */

    mpz_set(len0, N1);    
    mpz_div_ui(len0, len0, total_cores); // len = length_of_stage2 / (tot * total_gpus)           
    
    // So this covers almost everything except for the remainder...
    // but we can add the remainder to stage1 sums... which we did.
    // so now this dissection should be perfect
    
    mpz_set(start0, len0);  // start0 = length_of_stage2 / (tot * total_gpus)
    
    mpz_mul_ui(start0, start0, offset_cpu );
    // start0 = (k * total_gpus + offset_gpu) * length_of_stage2 / (tot * total gpus)
    
    mpz_add(start0, start0, start);
    // start0 = n2 + (k * total_gpus + offset_gpu) * length_stage_2 / total_gpus
    
    mpz_mul_ui(len0, len0, cpus);
    // len = gpus * length_of_stage2 / (tot * total_gpus)    
    

    string filename1 = prepare_string(filename, gpus, number_of_gpu_threads,
				      cpus, number_of_cpu_threads, total_cores,
				      total_gpus, start0, len0); 
    
    sum_data_t<1> sum1(start0, len0,
		       t, delta, M, S1, STAGE2_PRECISION, number_of_cpu_threads, 
		       number_of_gpu_threads, gpus, filename1);
    
    
    // compute the stages 1, 2, & 3 sums
    pthread_create(&threads[0], NULL, partial_zeta_sum_stage<1>, (void *) &sum1);    
    pthread_join(threads[0], NULL);
    cout.precision(6);
    if(!closing)
      cout << "\rStage 1 : Host " << process_id << " Done. Sum was: " << (sum1.S)[0] << "                           " << endl;
    
    
  if(closing){
#if HAVE_MPI
    MPI_Finalize();
#endif 
    exit(0); 
  }

  double ReS[M];
  double ImS[M]; 
  for(int j = 0; j < M; j++){
    ReS[j] = (sum1.S)[j].real();
    ImS[j] = (sum1.S)[j].imag(); 
  }

  int tot = 1; 
  
  //  if(total_gpus > gpus)
  //  tot = number_of_gpu_threads; 

  for(int k = 0; k < tot; k++){
  
    // Split stage2 according to how many GPU's
    
    if(gpus > 0){
      mpz_set(len0, N2);    
      mpz_div_ui(len0, len0, tot*total_gpus); // len = length_of_stage2 / (tot * total_gpus)           
      
      // So this covers almost everything except for the remainder...
      // but we can add the remainder to stage1 sums... which we did.
      // so now this dissection should be perfect
      
      mpz_set(start0, len0);  // start0 = length_of_stage2 / (tot * total_gpus)
    
      mpz_mul_ui(start0, start0, k*total_gpus + offset_gpu );
      // start0 = (k * total_gpus + offset_gpu) * length_of_stage2 / (tot * total gpus)
      
      mpz_add(start0, start0, n2);
      // start0 = n2 + (k * total_gpus + offset_gpu) * length_stage_2 / total_gpus
      
      mpz_mul_ui(len0, len0, gpus);
      // len = gpus * length_of_stage2 / (tot * total_gpus)    
      
    }
    
    string filename2 =  prepare_string(filename, gpus, number_of_gpu_threads,
				       cpus, number_of_cpu_threads, total_cores,
				       total_gpus, start0, len0); 
    
    sum_data_t<2> sum2(start0, len0,
		       t, delta, M, S2, STAGE2_PRECISION, number_of_cpu_threads, 
		       number_of_gpu_threads, gpus, filename2);  
    
#if HAVE_CUDA
    if(gpus > 0){
      pthread_create(&threads[1], NULL, partial_zeta_sum_stage<2>, (void *) &sum2);    
      pthread_join(threads[1], NULL);
      cout.precision(6);
      if(!closing){	
	cout << "\rStage 2 : Host " << process_id << " Done";
	if(tot > 1){
	  cout << " (" << k << "/" << tot << "). ";
	}else{
	  cout << ". "; 
	}
	cout << " Sum was: " << (sum2.S)[0] << "                                    " << endl;
      }
    }
#endif
    
    if(closing){
#if HAVE_MPI
      MPI_Finalize(); 
#endif
      exit(0); 
    }
    
    for(int l = 0; l < M; l++) {
      if(gpus > 0){
	ReS[l] += (sum2.S)[l].real();
	ImS[l] += (sum2.S)[l].imag();
      }
    }     
  }
  
  
  
  // TODO: The dissection here needs to be improved
  // As the one here induces small losses of precision
  
  /* Preparation for stage3 */
  
  mpz_sub(len1, end, n3); // len = length of stage3
  mpz_div_ui(len1, len1, total_cores ); // len = length_of_stage3 / total_cpus
  
  mpz_set(start1, len1);  // start1 = length_of_stage3 / total_cpus
  mpz_mul_ui(start1, start1, offset_cpu ); // start1 = offset_cpu * length_of_stage3 / total gpus
  mpz_add(start1, start1, n3); // start1 = n3 + offset_gpu * length_stage3 / total_cpus
  
  mpz_mul_ui(len1, len1, cpus); // len = cpus * length_of_stage3 / total_cpus

  string filename3 = prepare_string(filename, gpus, number_of_gpu_threads,
				    cpus, number_of_cpu_threads, total_cores,
				    total_gpus, start1, len1); 
    
  sum_data_t<3> sum3(start1, len1,
		     t, delta, M, S3, STAGE3_PRECISION, number_of_cpu_threads, 
		     number_of_gpu_threads, gpus, filename3);
  
  pthread_create(&threads[2], NULL, partial_zeta_sum_stage<3>, (void *) &sum3); 
  
  /* There is a race condition created by the threads spawned by the threads of partial_zeta_sum */  
  
  pthread_join(threads[2], NULL);  
  cout.precision(6);
  if(!closing){
    cout << "\rStage 3 : Host " << process_id << " Done. Sum was: " << (sum3.S)[0] << "                           " << endl;
  }
  
  if(closing){
#if HAVE_MPI
    MPI_Finalize(); 
#endif
    exit(0); 
  }
  
  // Store the contributions of stages 1, 2 & 3 in S, which is the array passed to partial_zeta_sum.
  // We first initialize S to zero, then add up the contributions of S1, S2, & S3
  
  for(int l = 0; l < M; l++) {
    ReS[l] += (sum3.S)[l].real();
    ImS[l] += (sum3.S)[l].imag();
  }
  
#if HAVE_MPI  
  MPI_Reduce(&ReS, &ResultReS, M, MPI_DOUBLE, MPI_SUM, 0,
	     MPI_COMM_WORLD);
  MPI_Reduce(&ImS, &ResultImS, M, MPI_DOUBLE, MPI_SUM, 0,
	     MPI_COMM_WORLD);
  
  MPI_Barrier(MPI_COMM_WORLD); 

  if(process_id == 0){
    for(int i = 0; i < M; i++){
      S[i] = Complex (ResultReS[i], ResultImS[i]); 
    }
    
  }else{
    MPI_Finalize();
    exit(0); 
  }
#else
  for(int i = 0; i < M; i++)
    S[i] = Complex (ReS[i], ImS[i]); 
#endif
  
  if(closing){ 
#if HAVE_MPI
    MPI_Finalize(); 
#endif
    exit(0); }
  
#if HAVE_MPI
  MPI_Finalize();
#endif
  
  mpz_clear(start0);
  mpz_clear(start1);
  mpz_clear(len0);
  mpz_clear(len1); 

  mpz_clear(n2);
  mpz_clear(n3);
  mpz_clear(end);
  mpz_clear(N1);
  mpz_clear(N2);
  mpz_clear(N3);
  
  delete [] S1;
  delete [] S2;
  delete [] S3;
  
  
  return S[0]; // return the value of the partial sum at t
}

struct threadinfo {
  void * data;
  int num; 
};

template<int stage> void * zeta_sum_thread(void * ti) {
  
  //
  // creates a sum_data_t struct, computes auxilliary coefficients in the linear 
  // combination of quadratic sums that results from taylor-expanding the terms in 
  // the main sum, and starts a computation of the partial sum (in "chunks")
  //

  struct threadinfo * t = (struct threadinfo *) ti; 
  sum_data_t<stage> * sum_data = (sum_data_t<stage>*) t->data;

#if HAVE_MPI
  int world_rank; 
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

#else
  int world_rank = 0;
#endif

  // Assign GPU per thread 

  int tid = t->num; 
#if HAVE_CUDA
  if(stage == 2) cudaSetDevice(tid % sum_data->gpus); 
#endif
 

  mpz_t v;
  mpz_init(v);
  
  // returns size of currently available block, stores its starting point in v, and 
  // increments "next" in sum_data to "next + length"
  
  // Fetch a stream per thread and initialize it -- it is important
  // to do this after the thread has been assigned to a GPU -- otherwise
  // we will each stream will be writting to an un-assigned memory address
  // on the wrong GPU
#if HAVE_CUDA  
  int gpus = sum_data->gpus; 
  int id_gpu = tid % gpus; 
  cudaStream_t stream = sum_data->cudaStreams[tid % sum_data->number_of_gpu_threads]; 
  if(stage == 2){ 
    cudaStreamCreate(&stream); 
    cudaCheckErrors("couldn't create CUDA streams\n"); 
  }
#endif

  // array where thread output will be stored
  complex<double> * S = new complex<double>[sum_data->M];
  
  // initializes S[0], S[1], ... , S[M-1] to zero
  for(int l = 0; l < sum_data->M; l++) S[l] = 0;
  
  // array where the auxilliary coefficients in the linear combination of quadratic 
  // sums is stored (however, it doesn't seem this needs to be called each time since
  // it only depends on t ...)
  Complex Z[30];
  if(stage == 3)
    compute_taylor_coefficients(sum_data->t, Z);
  
  /* Initialize Host & CUDA buffer */
  
  /* Stage 2 buffers */
#if HAVE_CUDA
  struct precomputation_table * dev_precompute;
  cuDoubleComplex * dev_current_terms;
  struct precomputation_table * precompute;
  cuDoubleComplex * host_current_terms_pre;
  Complex * host_current_terms;
#endif

  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);
  
  
  // compute the partial sum in "chunks" 
  //  int serialize = 1; 
  string filename = sum_data->filename 
    + ",file,"
    + to_string(tid) + ","
    + processor_name + ","
    + to_string(stage); 
    //    + to_string(tid) + "," 
    //  + to_string(process_id) + "," 
    // + to_string(stage);  // Adding the tid here would be a mistake

  
  /* The problem here is that the threads are sharing the sum_data structure 
     thus it shouldn't be written to separate files as it creates a race condition
     on the other hand the array S is private and should be stored and loaded 
     separately */
  
#if HAVE_CUDA
  long long size = 0;
#endif

  if(is_file_exist(filename.c_str())){
    deserialize_local(filename, S, sum_data->M);
  }else{
    serialize_local(filename, S, sum_data->M);
  }
    
  unsigned long length = sum_data->next_block(v);
  
  while(length != 0) {
    
    /* Here as we iterate through the loop we could periodically save all the
       data associated to sum_data , this should allow one to resume the program
       if it is interrupted */ 
    
    if(stage==1) 
      zeta_block_stage1(v, length, sum_data->t, sum_data->delta, sum_data->M, S);
#if HAVE_CUDA
    if(stage==2)
      size = zeta_block_stage2(v,
			       length,
			       sum_data->t,
			       sum_data->delta,
			       sum_data->M,
			       S,
			       &dev_precompute,
			       &dev_current_terms,
			       &precompute,
			       &host_current_terms_pre,
			       &host_current_terms,
			       0, // Not needed anymore
			       stream,
			       id_gpu,
			       sum_data->next_mutex,
			       size);
#endif
    if(stage==3) 
      zeta_block_stage3(v, length, sum_data->t, Z, sum_data->delta, sum_data->M, S);

        
    //    if(size < 3200){
    //   pthread_mutex_lock(sum_data->next_mutex); 
    //  sum_data->stage2_max_size = 16000000; 
    //  pthread_mutex_unlock(sum_data->next_mutex);
    //   }

    /* This is admitedly a bit hacky */
    /* This is done so that size never falls below 3200, after which GPU transfers become
       more expensive */

    // serialize_local(filename, S, sum_data->M); 

    
    if(closing){
      // Do we need to lock the mutex when serializing?
      pthread_mutex_lock(sum_data->next_mutex); 
      serialize_local(filename, S, sum_data->M);
      pthread_mutex_unlock(sum_data->next_mutex); 
      pthread_exit(NULL); 
    }    

    length = sum_data->next_block(v);

      
  }

  /* Free CUDA Buffers */
#if HAVE_CUDA  
  if(stage == 2 && size > 0 ){        
    cudaFree(dev_precompute);
    cudaFree(dev_current_terms);
    cudaCheckErrors("cudaFree device"); 
    
    cudaFreeHost(precompute);
    cudaFreeHost(host_current_terms_pre);
    cudaFreeHost(host_current_terms);
    cudaCheckErrors("cudaFree host"); 
    
    cudaStreamDestroy(stream);
    cudaCheckErrors("error destroying stream"); 

  }

#endif
  
  pthread_mutex_lock(sum_data->next_mutex);
  // Do we need to lock the mutex when serializing ?
  serialize_local(filename, S, sum_data->M); 
  sum_data->report(S);  
  sum_data->serialize(sum_data->filename + ",global," + to_string(stage)); 
  pthread_mutex_unlock(sum_data->next_mutex);

  mpz_clear(v);
  
  pthread_exit(NULL);
}

#if HAVE_CUDA
extern void allocateTexture();     
#endif

template<int stage> void * partial_zeta_sum_stage(void * data){
  
  // 
  // Computes the partial sum:
  //
  //      \sum_{n = start}^{start + length - 1} exp(i*t*log(n)) / sqrt(n)
  //
  // at y = t, t+delta, t+2*delta,...,t+(M-1)*delta, and stores the result 
  // in S[0], S[1], ... , S[M-1]. The stage number determines which method 
  // to use for computing the partial main sum.
  //
  // The function attempts to compute the sum to within epsilon
  // using the given number of threads. It also returns S[0], which
  // is the value of the partial sum at t.
  //

  sum_data_t<stage> * sum_ptr = (sum_data_t<stage>*) data; 

  mpz_t len;
  mpz_init(len);
  mpz_set_d(len, sum_ptr->length); 
  sum_data_t<stage> sum (sum_ptr->next, len, sum_ptr->t, sum_ptr->delta,
			 sum_ptr->M, sum_ptr->S, sum_ptr->epsilon, sum_ptr->number_of_cpu_threads,
			 sum_ptr->number_of_gpu_threads, 
			 sum_ptr->gpus, sum_ptr->filename); 
  
  for(int l = 0; l < sum.M; l++) (sum.S)[l] = 0;
  
  string filename = sum.filename + ",global," + to_string(stage);
  
  if(is_file_exist(filename.c_str())){    
    sum.deserialize(filename);
  }else{
    sum.serialize(filename);
  }
  
  int number_of_cpu_threads = sum.number_of_cpu_threads;
  
  int number_of_threads = 0; 
  
  if(stage == 1 || stage == 3)
    number_of_threads = number_of_cpu_threads; 
#if HAVE_CUDA
  int number_of_gpu_threads = sum.number_of_gpu_threads; 
  if(stage == 2)
    number_of_threads = number_of_gpu_threads; 
#endif
  pthread_t threads[number_of_threads];
  
  struct threadinfo ti [number_of_threads]; 
  
  for(int n = 0; n < number_of_threads; ++n) {
    ti[n].num = n;
    ti[n].data = (void *) &sum; 
    pthread_create(&threads[n], NULL, zeta_sum_thread<stage>, (void *) &ti[n]);
  }
  
  for(int n = 0; n < number_of_threads; ++n) {
    pthread_join(threads[n], NULL);
  }
  
  if(closing){
    pthread_mutex_lock(sum.next_mutex); 
    (sum.serialize)(filename);
    pthread_mutex_unlock(sum.next_mutex);  
  }
  
  return NULL; 

}
