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
#include "md5.h"

using namespace std;

typedef std::mt19937 RNG;

static int closing = 0; 
static int end_display = 0;
static int thread_off = 0;

bool is_file_exist(string fileName)
{
  std::ifstream infile(fileName);
  return infile.good();
}



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
  mpz_div_ui(v, v,100000); // One can play with this
  // There is a trade-off : a longer stage1 means that stage2 is better
  // spread out among hosts in stage2 when using MPI

  if(mpz_cmp_ui(v,1100000) < 0) mpz_set_ui(v, 1100000); 

  return; 

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

void signalHandler( int signum ) {
  closing = 1;
  //MPI_Bcast(&closing, 1, MPI_INT, 0, MPI_COMM_WORLD); 
  // MPI_Barrier(MPI_COMM_WORLD);
  cout << "\rSaving state and exiting...                          "; 
  cout.flush();
}

void write_stream(std::ofstream & out, Complex * S, int M){
  for(int i = 0; i < M; i++)
    out.write(reinterpret_cast<char*>(&S[i]), sizeof(Complex));  
}

void read_stream(std::ifstream & in, Complex * S, int M){
  for(int i = 0; i < M; i++){
    in.read(reinterpret_cast<char*>(&S[i]), sizeof(Complex));
  }
}

void write(const std::string& file_name, Complex * S, int M){
  std::ofstream out;
  out.open(file_name, std::ios::binary);
  for(int i = 0; i < M; i++)
    out.write(reinterpret_cast<char*>(&S[i]), sizeof(Complex));
  out.close();
}

void read(const std::string& file_name, Complex * S, int M){
  std::ifstream in;
  in.open(file_name, std::ios::binary);
  for(int i = 0; i < M; i++){
    in.read(reinterpret_cast<char*>(&S[i]), sizeof(Complex));
  }
  in.close();
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
  int percent_finished = 0;

  /* GPU Data -- each thread get its own CUDA stream */
#if HAVE_CUDA
  cudaStream_t * cudaStreams;  // Place holder for stream data
#endif

  int number_of_gpu_threads; 
  int number_of_cpu_threads; 
  int numStreams; // number of streams -> set to number_of_threads
  int gpus; // number of GPUS  

  int world = 0; 
  int rank;
  int * stats; 
  
#if HAVE_MPI  
  MPI_Request r; 
#endif
  
  // a struct constructor: sets variable values and initializes a thread
  sum_data_t(mpz_t start, mpz_t _length, mpfr_t _t, double _delta , int _M, complex<double> * _S, double _epsilon, int num_cpu_th, int num_gpu_th, int num_gpus) {
  
#if HAVE_MPI
    MPI_Comm_size(MPI_COMM_WORLD, &world); 
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
#endif
    
    stats = (int * ) malloc((world + 1)*sizeof(int)); 
    for(int i = 0; i < world + 1; i++) stats[i] = 0; 
    
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
    percent_finished = 0;
    
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
  

  ~sum_data_t() {
    pthread_mutex_destroy(report_mutex);
    pthread_mutex_destroy(next_mutex);
    delete report_mutex;
    delete next_mutex;

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
    out.precision(30);
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

  void record_output(string filename){
    write(filename, S, M);
  }

  void load_output(string filename){
    read(filename, S, M);
  }
  
  string hash(int tid, int rank, string str2){
    string test;
    std::ostringstream out;
    out.precision(30);
    out << str2 << endl;
    out << tid << endl;
    out << M << endl
	<< length << endl
	<< tt << endl 
        << delta << endl
	<< epsilon << endl
      //   	<< percent_finished << endl
	<< number_of_cpu_threads << endl
	<< number_of_gpu_threads << endl
	<< gpus << endl
      //      	<< next << endl
      	<< end << endl
	<< stage << endl;
    char* str = NULL;
    mpfr_exp_t e;
    str = mpfr_get_str (NULL, &e, 10, 0, t, MPFR_RNDN);
    out << str << endl << e << endl;
    out << rank;
    return ".data" + md5(out.str());    
  }
  
  void deserialize(string file){
    string t_str; 
    ifstream in;
    in.open(file, std::ifstream::in);
    in.precision(30);
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

    mpz_set(start, next);               /* start = next (start is now the
					   beginning point of the remainder
					   of the partial sum) */
    
    mpz_add_ui(next, next, block_size); // next = next + block_size
    

    percent_finished = (int) (1000*(1.0 - remainder/length)); 
    
    /* Basically need to run this in some independent loop */ 

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

struct threadinfo {
  void * data;
  int num;
  Complex * S;
};

// This function is declared here and defined at the end of the file
template<int stage> void * partial_zeta_sum_stage(void * data); 


struct cluster_info{
  int process_id;
  int max_proc;
  int * has_cpu; 
  int * has_gpu;
  int * has_cpu_global;
  int * has_gpu_global;
  int total_gpus;
  int total_cpus;
  int total_cores;
};

struct cluster_info get_cluster_info(int gpus, int cpus){
  
  struct cluster_info cinfo; 
#if HAVE_MPI  
  MPI_Comm_size(MPI_COMM_WORLD, &cinfo.max_proc);
  MPI_Comm_rank(MPI_COMM_WORLD, &cinfo.process_id);
#else
  cinfo.max_proc = 0;
  cinfo.process_id = 0; 
#endif
  
  cinfo.has_cpu = (int *) malloc((cinfo.max_proc + 1)*sizeof(int)); 
  cinfo.has_gpu = (int *) malloc((cinfo.max_proc + 1)*sizeof(int)); 
  
  for(int i = 0; i < cinfo.max_proc; i++){
    cinfo.has_gpu[i] = 0;
    cinfo.has_cpu[i] = 0; 
  }
  
  cinfo.has_gpu[cinfo.process_id] = gpus;
  cinfo.has_cpu[cinfo.process_id] = cpus; 

  cinfo.has_gpu_global = (int *) malloc((cinfo.max_proc + 1)*sizeof(int));
  cinfo.has_cpu_global = (int *) malloc((cinfo.max_proc + 1)*sizeof(int)); 

#if HAVE_MPI
  MPI_Reduce(&cinfo.has_gpu[0], &cinfo.has_gpu_global[0], cinfo.max_proc, MPI_INT, MPI_SUM, 0,
	     MPI_COMM_WORLD);

  MPI_Reduce(&cinfo.has_cpu[0], &cinfo.has_cpu_global[0], cinfo.max_proc, MPI_INT, MPI_SUM, 0,
	     MPI_COMM_WORLD); 
    
  MPI_Reduce(&cinfo.has_gpu[cinfo.process_id], &cinfo.total_gpus, 1, MPI_INT, MPI_SUM, 0,
  	     MPI_COMM_WORLD);
  MPI_Reduce(&cinfo.has_cpu[cinfo.process_id], &cinfo.total_cores, 1, MPI_INT, MPI_SUM, 0, 
  	     MPI_COMM_WORLD); 

  MPI_Bcast(&cinfo.total_gpus, 1, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD); 
  MPI_Bcast(&cinfo.total_cores, 1, MPI_INT, 0, MPI_COMM_WORLD); 
  MPI_Barrier(MPI_COMM_WORLD); 
  MPI_Bcast(&cinfo.has_gpu_global[0], cinfo.max_proc, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD); 
  MPI_Bcast(&cinfo.has_cpu_global[0], cinfo.max_proc, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Barrier(MPI_COMM_WORLD); 
#else
  cinfo.has_gpu[0] = gpus;
  cinfo.has_cpu[0] = cpus; 
  cinfo.has_gpu_global[0] = gpus;
  cinfo.has_cpu_global[0] = cpus; 
  cinfo.total_gpus = gpus;
  cinfo.total_cores = cpus; 
#endif

  
  return cinfo; 
  
}

void print_cluster_info(struct cluster_info c){
  if(c.process_id == 0){ 
    for(int i = 0; i < c.max_proc; i++){
      printf("host %d has %d gpus and %d cpus\n", i, c.has_gpu_global[i], c.has_cpu_global[i]);
    }
    printf("total number of cpu %d\ntotal numbers of gpus %d\n", c.total_cores, c.total_gpus); 
  }
}

void check_for_gpus(struct cluster_info c){
  if(c.process_id == 0){    
    if(c.total_gpus == 0){
      printf("There are no GPU's on this cluster -- stage2 code will not be able to run\n"
	     "and the results will be incorrect. Aborting\n"); 
#if HAVE_MPI
      MPI_Finalize();
#endif
      exit(0); 
    }
  }
}

struct stage_len {
  mpz_t n2;
  mpz_t n3;
  mpz_t N1;
  mpz_t N2;
  mpz_t N3; 
  mpz_t end; 
};

void free_stage_len(struct stage_len s){
  mpz_clear(s.n2);
  mpz_clear(s.n3);
  mpz_clear(s.end);
  mpz_clear(s.N1);
  mpz_clear(s.N2);
  mpz_clear(s.N3);
}

int compute_offset_cpu(struct cluster_info c){
  int offset = 0;
  for(int i = 0; i < c.process_id; i++) offset += c.has_cpu_global[i];
  return offset;
}

int compute_offset_gpu(struct cluster_info c){
  int offset = 0;
  for(int i = 0; i < c.process_id; i++) offset += c.has_gpu_global[i];
  return offset; 
}

struct stage_len compute_stage_len(struct cluster_info c, mpz_t start, mpfr_t t, mpz_t length){
  struct stage_len s; 

  mpz_t len0, len1, start0, start1; 
  mpz_init(len0); mpz_init(start0);
  mpz_init(len1); mpz_init(start1);
  
  mpz_init(s.n2); mpz_init(s.n3); mpz_init(s.end);
  mpz_init(s.N1); mpz_init(s.N2); mpz_init(s.N3);
  
  stage_2_start(s.n2, t); // n2 is the starting point of stage_2 
  stage_3_start(s.n3, t); // n3 is the starting point of stage_3


  /* These might be quite irrelevant */
  mpz_sub(len0, s.n2, start);
  mpz_mod_ui(len0, len0, c.total_cores);
  mpz_sub(s.n2, s.n2, len0);
  
  mpz_sub(len0, s.n3, s.n2); // len0 = lenth of stage2
  mpz_mod_ui(len0, len0, c.total_gpus); // len0 = length of stage2 % total_gpus
  mpz_sub(s.n3, s.n3, len0); // n2 = start of stage2 + ((length of stage2) % total_gpus)
  
  mpz_add(s.end, start, length); // end = start+length, is the endpoint of the partial sum 
  // (so the last term in the sum is n = end - 1)
  /* End of possibly irrelevant things */
  
  if(mpz_cmp(s.end,s.n3) < 0) 
    mpz_set(s.n3, s.end); // if end < n3, set n3 = end
  if(mpz_cmp(start,s.n3) > 0) 
    mpz_set(s.n3, start); // if start > n3, set n3 = start
  if(mpz_cmp(s.end,s.n2) < 0) 
    mpz_set(s.n2, s.end); // if end < n2, set n2 = end
  if(mpz_cmp(start,s.n2) > 0) 
    mpz_set(s.n2, start); // if start > n2, set n2 = start

  mpz_sub(s.N1, s.n2, start); // N1 = n20 - start, is the length of stage 1 sum
  mpz_sub(s.N2, s.n3, s.n2); // N2 = n3 - n2, is the length of stage 2 sum
  mpz_sub(s.N3, s.end, s.n3); // N3 = end - n3, is the length of stage 3 sum

  mpz_clear(len0); mpz_clear(len1); mpz_clear(start0); mpz_clear(start1); 
  
  return s;
  
}

void print_stage_info(struct stage_len s, struct cluster_info c,mpz_t start){
  if(c.process_id == 0){
    cout << "Stage 1 range = [" << start << "," << s.n2 << "]\n";
    
    if(mpz_cmp(s.n2,s.n3) == 0)
      cout << "Stage 2 range = empty\n";
    else
      cout << "Stage 2 range = (" << s.n2 << "," << s.n3 << "]\n";
    
    if(mpz_cmp(s.n3,s.end) == 0)
      cout << "Stage 3 range = empty\n";
    else
      cout << "Stage 3 range = [" << s.n3 << "," << s.end << "]\n";
  }
}

struct start_length {
  mpz_t start0;
  mpz_t len0; 
};

void free_start_length(struct start_length p){
  mpz_clear(p.start0);
  mpz_clear(p.len0); 
}

struct start_length compute_start_length(int total, int offset, mpz_t start, mpz_t N1, int cpus){
  
  struct start_length s;
  mpz_init(s.start0);
  mpz_init(s.len0);
  
  mpz_set(s.len0, N1);    
  mpz_div_ui(s.len0, s.len0, total); // len = length_of_stage2 / (tot * total_gpus)           
    
    // So this covers almost everything except for the remainder...
    // but we can add the remainder to stage1 sums... which we did.
    // so now this dissection should be perfect
    
  mpz_set(s.start0, s.len0);  // start0 = length_of_stage2 / (tot * total_gpus)
    
  mpz_mul_ui(s.start0, s.start0, offset);
  // start0 = (k * total_gpus + offset_gpu) * length_of_stage2 / (tot * total gpus)
    
  mpz_add(s.start0, s.start0, start);
  // start0 = n2 + (k * total_gpus + offset_gpu) * length_stage_2 / total_gpus
    
  mpz_mul_ui(s.len0, s.len0, cpus);
  // len = gpus * length_of_stage2 / (tot * total_gpus)    

  return s;
}

/* Also need to add case when total_gpus = 0 */

struct shared_thread_data {
  string filename;
  int gpus;
  int cpus;
  int number_of_gpu_threads;
  int number_of_cpu_threads; 
  int M; 
  double delta;
};

struct shared_thread_data
init_shared_thread_data(string filename, int gpus,
			int number_of_gpu_threads,
			int number_of_cpu_threads,
			int M, int cpus, double delta){

  struct shared_thread_data th; 
  th.filename = filename;
  th.gpus = gpus;
  th.number_of_gpu_threads = number_of_gpu_threads;
  th.number_of_cpu_threads = number_of_cpu_threads;
  th.M = M;
  th.cpus = cpus;
  th.delta = delta; 
  
  return th;
}


template<int stage> Complex * submit_thread(struct shared_thread_data th,
					    struct cluster_info c,
					    struct start_length p,
					    mpfr_t t, int precision){

  pthread_t thread;

  Complex * S1 = new Complex [th.M]; 
  Complex * S = (Complex *) malloc(sizeof(Complex)*th.M);

  if(mpz_cmp_ui(p.len0,0) == 0){
    for(int i = 0; i < th.M; i++) S[i] = 0;
    return S; 
  }
  
  sum_data_t<stage> sum1
    (p.start0, p.len0, t, th.delta, th.M, S1, precision, th.number_of_cpu_threads,
     th.number_of_gpu_threads, th.gpus);

  //  partial_zeta_sum_stage<stage>((void *) &sum1); 
  pthread_create(&thread, NULL, partial_zeta_sum_stage<stage>, (void *) &sum1);
  pthread_join(thread, NULL); 

  for(int i = 0; i < th.M; i++){
    S[i] = sum1.S[i]; 
  }
  
  delete [] S1;

  if(sum1.rank == 0) cout << endl;
  free_start_length(p); 

  return S; 
}

/* Need to review special cases : 
 * - No MPI
 * - gpus = 0
 */
 

Complex partial_zeta_sum(mpz_t start, mpz_t length, mpfr_t t, Double & delta, int &M, Complex * S, int Kmin, int number_of_cpu_threads, int number_of_gpu_threads, pthread_t thread, int gpus, string filename) {

  //
  // This function computes the partial sum:
  //
  //          \sum_{n=start}^{start + length - 1} exp(i*y*log n)/sqrt(n)
  //
  // for y = t,t+delta,t+2*delta,...,t+(M-1)*delta, then stores the resulting 
  // values in S. It also returns the value of the partial sum at y = t
  //

  /* Init MPI and adjust according to capabilities */
  
  int cpus = number_of_cpu_threads; // thread::hardware_concurrency();
  
  struct cluster_info c = get_cluster_info(gpus, cpus); 
  print_cluster_info(c);
  check_for_gpus(c); 
  
  signal(SIGINT, signalHandler);
  signal(SIGTERM,signalHandler);

  struct stage_len s = compute_stage_len(c,start,t,length); 
  print_stage_info(s,c,start); 
  
  // we carry out a precomputation, which is needed in conjunction with a 
  // modification of Fenynman's algorithm for computing exp(i*t*log(n)); see 
  // log.cc
  
  create_exp_itlogn_table(t);
      
  struct shared_thread_data th =
    init_shared_thread_data(filename, gpus,
			    number_of_gpu_threads,
			    number_of_cpu_threads,
			    M, cpus, delta); 

  /* Stage 1 */
  
  /* Further partitions the stage length in s according to how many clusters we have */
  struct start_length p = compute_start_length(c.total_cores, compute_offset_cpu(c),
					       start, s.N1, cpus);

  /* Submit the job -- The resulting output is an array of size M */
  Complex * sum1 = submit_thread<1>(th,c,p,t,STAGE2_PRECISION);

  /* We are assuming here that we always have gpus > 0 
     if not something else should be done */

#if HAVE_MPI
  MPI_Barrier(MPI_COMM_WORLD); 
#endif

  /* Stage 2 */
  
  p = compute_start_length(c.total_gpus, compute_offset_gpu(c),
			   s.n2, s.N2, gpus);
  Complex * sum2 = submit_thread<2>(th,c,p,t,STAGE2_PRECISION);

#if HAVE_MPI
  MPI_Barrier(MPI_COMM_WORLD); 
#endif

  /* Stage 3 */

  p = compute_start_length(c.total_cores, compute_offset_cpu(c),
			   s.n3, s.N3, cpus);  
  Complex * sum3 = submit_thread<3>(th,c,p,t,STAGE3_PRECISION);

#if HAVE_MPI
  MPI_Barrier(MPI_COMM_WORLD); 
#endif

  /* Compute final sum */

  double ReS[M];
  double ImS[M];
  
  for(int i = 0; i < M ; i++){
    ReS[i] = (sum1)[i].real() + (sum2)[i].real() + (sum3)[i].real();
    ImS[i] = (sum1)[i].imag() + (sum2)[i].imag() + (sum3)[i].imag();
  }  

#if HAVE_MPI  
  double ResultReS[M];
  double ResultImS[M];
  
  for(int i = 0; i < M; i++){
    ResultReS [i] = 0;
    ResultImS [i] = 0; 
  }

  MPI_Barrier(MPI_COMM_WORLD);
  
  MPI_Reduce(&ReS, &ResultReS, M, MPI_DOUBLE, MPI_SUM, 0,
	     MPI_COMM_WORLD);
  MPI_Reduce(&ImS, &ResultImS, M, MPI_DOUBLE, MPI_SUM, 0,
	     MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
#endif

  /* This is just to fix a small "bug" when there is only one host
     as we don't want to print the output of partial computations */
  
  if(closing == 1 && c.max_proc == 1){
    exit(0); 
  }
  
#if HAVE_MPI
  closing = 1; 

  //  usleep(100000);  /* A bit hacky but whatever */
  
  MPI_Finalize();

  if(c.process_id == 0){
    for(int i = 0; i < M; i++) S[i] = Complex (ResultReS[i], ResultImS[i]);
  }
  else
    exit(0);
  
#else
  
  for(int i = 0; i < M; i++) S[i] = Complex (ReS[i], ImS[i]); 
  
#endif
  
  free_stage_len(s); 
  
  return S[0]; // return the value of the partial sum at t
}

static pthread_barrier_t barr[3]; 

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
  // we will each stream will be w itting to an un-assigned memory address
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

  Complex * S = (Complex *) t->S; 

  /* ! WARNING ! THis is where the bug was coming from ... */

  // initializes S[0], S[1], ... , S[M-1] to zero
  //  if(!is_file_exist((sum_data->hash)(tid,0,"S")))
  //   for(int l = 0; l < sum_data->M; l++) S[l] = 0;
  
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

#if HAVE_MPI
  int rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
#endif

  
#if HAVE_CUDA
  long long size = 0;
#endif

  unsigned long length = sum_data->next_block(v);

  while(length != 0) {
    /* Here as we iterate through the loop we could periodically save all the
       data associated to sum_data , this should allow one to resume the program
       if it is interrupted */ 
    
    if(stage==1)
      zeta_block_stage1(v, length, sum_data->t, sum_data->delta, sum_data->M, S);

#if HAVE_CUDA

    if(stage==2){

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

    }
#endif
    
    if(stage==3) 
      zeta_block_stage3(v, length, sum_data->t, Z, sum_data->delta, sum_data->M, S);
    
    if(closing){
      break;
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

  sum_data->report(S);

  pthread_mutex_unlock(sum_data->next_mutex);
  
  int rc = pthread_barrier_wait(&barr[stage]);
  if(rc != 0 && rc != PTHREAD_BARRIER_SERIAL_THREAD)
    printf("WARNING : Could not wait on barrier\n");

  mpz_clear(v); 
  
  pthread_exit(NULL);
}

#if HAVE_CUDA
extern void allocateTexture();     
#endif

template<int stage> void * display_thread_main(void * data){
  sum_data_t<stage> * sum_data = (sum_data_t<stage>*) data;

  int buf; 
  
  pthread_setcanceltype(PTHREAD_CANCEL_ASYNCHRONOUS, &buf); 

  while(thread_off != 1){

    usleep(10000);
    
#if HAVE_MPI
    //   MPI_Barrier(MPI_COMM_WORLD);
    MPI_Gather(&(sum_data->percent_finished), 1, MPI_INT, sum_data->stats, 1,
		MPI_INT,0, MPI_COMM_WORLD);
#else
    sum_data->stats[0] = sum_data->percent_finished;
#endif
    
    if(sum_data->rank == 0){      
      cout.precision(1);
      cout << fixed;
      cout << "\rS" << stage << ": "; 
      for(int i = 0; i < sum_data->world; i++)	
	cout << "H" << i << " (" << ((double) sum_data->stats[i]) / 10 << ") " ;
      cout << "                   "; 
      cout.flush();
    }
  }

  
  
  //  MPI_Allgather(&(sum_data->percent_finished), 1, MPI_INT, sum_data->stats, 1,
  //		MPI_INT, MPI_COMM_WORLD);
  
  //  MPI_Barrier(MPI_COMM_WORLD);
  
  return NULL;
}

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

#if HAVE_MPI
  int rank;
  MPI_Comm_rank (MPI_COMM_WORLD, &rank);
#else
  int rank = 0;
#endif

  sum_data_t<stage> * sum_ptr = (sum_data_t<stage>*) data;

  //  if(is_file_exist(sum_ptr->hash(0,rank,"finished"))){
  //  (*sum_ptr).load_output(sum_ptr->hash(0,rank,"finished"));
  //  return NULL;
  //}
  
  mpz_t len;
  mpz_init(len);
  mpz_set_d(len, sum_ptr->length);
  
  sum_data_t<stage> sum(sum_ptr->next, len, sum_ptr->t, sum_ptr->delta,
			sum_ptr->M, sum_ptr->S, sum_ptr->epsilon, sum_ptr->number_of_cpu_threads,
			sum_ptr->number_of_gpu_threads, 
			sum_ptr->gpus); 

  // Second zero should be replaced by processor id ? */

  if(is_file_exist(sum_ptr->hash(0,rank,"sum")))
    sum.deserialize(sum_ptr->hash(0,rank,"sum"));
  else
    for(int l = 0; l < sum.M; l++) (sum.S)[l] = 0;
  
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
  
  struct threadinfo * ti = (struct threadinfo *) malloc(number_of_threads*sizeof(struct threadinfo)); 
  
  if(pthread_barrier_init(&barr[stage], NULL, number_of_threads))
    printf("WARNING : Could not create a barrier\n");

  /* Open file with serialized data */
  
  ifstream in;
  if(is_file_exist(sum_ptr->hash(0,rank,"S")))
    in.open(sum_ptr->hash(0,rank,"S"), std::ios::binary);

  for(int n = 0; n < number_of_threads; ++n) {
    ti[n].num = n;
    ti[n].data = (void *) &sum;
    ti[n].S = (Complex *) malloc(sum_ptr->M*sizeof(Complex));

    /* Decide if use already computed input */
    if(is_file_exist(sum_ptr->hash(0,rank,"S")))
      read_stream(in, ti[n].S, sum_ptr->M);
    else
      for(int i = 0; i < sum_ptr->M; i++) ti[n].S[i] = 0;
  }
  
  in.close();

  pthread_t display_thread;
  pthread_create(&display_thread, NULL, display_thread_main<stage>, (void *) &sum); 
  pthread_detach(display_thread); 
  
  for(int n = 0; n < number_of_threads; ++n)
    pthread_create(&threads[n], NULL, zeta_sum_thread<stage>, (void *) &ti[n]);

  for(int n = 0; n < number_of_threads; ++n)
    pthread_join(threads[n], NULL);
  
  //  thread_off = 1; 

  /* Record output */

  /* Could it be that the problem is here?? */
  
  //  if(mpz_cmp(sum.next, sum.end) == 0){
  //  sum.record_output(sum_ptr->hash(0,rank,"finished")); 
  //    remove((sum_ptr->hash(0,rank,"S")).c_str());
  // remove((sum_ptr->hash(0,rank,"sum")).c_str());
  //}else{
  {
    sum.serialize(sum_ptr->hash(0,rank,"sum"));
    ofstream out;
    out.open(sum_ptr->hash(0,rank,"S"),std::ios::binary);
    for(int n = 0; n < number_of_threads; n++)
      write_stream(out, ti[n].S, sum_ptr->M);    
    out.close();
  }

  MPI_Barrier(MPI_COMM_WORLD);

  pthread_cancel(display_thread);
  
  return NULL; 

}
