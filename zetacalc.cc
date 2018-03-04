/* Still a bug for very small values when using Mpirun */
/* Still a small worry about the A_BUF size */ 

#include "theta_sums.h"
#include "main_sum.h"
#include "config.h"

#if HAVE_MPI
#include <mpi.h>
#endif

#include <getopt.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <string>
#include <cstring>
#include <thread>
#include <unistd.h>

#if HAVE_CUDA
#include "gpu.h"
#endif

#include "dirent.h"

using namespace std;

string output_filename = "";

void usage(char * name) {
  const char * text =
    " t [--delta delta] [--N N]\n"
    " [--number_of_gpu_threads thread_gpu] [--Kmin k]\n"
    " [--number_of_cpu_threads thread_cpu] [--output outputfile]\n"
    " [--start start] [--length len] [--checkRH]\n\n"
    " The first option t is the ordinate and is required\n\n"
    " N          = how many consecutive points to compute\n"
    " delta      = spacing between the points\n"
    " thread_cpu = number of cpu threads to use in the computation\n"
    " thread_gpu = number of gpu threads to use in the computation\n"
    " out        = name of the file to which output the results\n"
    " start      = if specified evaluate a Dirichlet polynomial\n"
    "              starting at integer start. If length not specified\n"
    "              then end point as in Riemann Siegel formula\n"
    " length     = if specified evaluate the Dirichlet polynomial\n"
    "              starting at start and of length length. If start\n"
    "              not specified then start at 1\n"
    " checkRH    = computes the data needed to check the Riemann hypothesis\n"
    "              from t to t + 1\n\n"
    " ------------------------------------------------------------\n"
    " Standard output : The values of the Z-function evaluated at \n"
    " well-spaced points. If either start or length option is specified\n"
    " then the returned value are the complex numbers corresponding to\n"
    " the evaluation of the Dirichlet polynomial\n"
    " ------------------------------------------------------------\n\n"
    " The program supports basic serialization . Thus it can be interrupted\n"
    " with SIGKILL or SIGHUP and will resume where it started if it is run again\n"
    " with the same initial parameters and on the same hosts\n"
    " The serialization files are kept in files starting with .file*\n\n"
    " The program supports MPI and needs to be run with mpirun to see the full\n"
    " effects. It also supports computations on multi-GPU systems\n"; 
  
  cout << "Usage: " << name << text;

  exit(0); 
  
}

string serialization_filename(int N, double delta, mpfr_t t){
  mpfr_exp_t r; 
  char * t_str = mpfr_get_str(NULL, &r, 10, 50, t, MPFR_RNDN); 
  string filename2 = ".file," + to_string(N) + "," 
    + to_string(delta) + "," + t_str + "," + to_string(r); 
  free(t_str); 
  return filename2; 
}

#if HAVE_CUDA
int cudaCheck(char * name){
  
  int gpus; 
  if(cudaGetDeviceCount(&gpus)){
    return 0;
  }
  //  cudaCheckErrors("no GPU's detected"); 
  
  for(int i = 0; i < gpus; i++){
    struct cudaDeviceProp props; 
    cudaGetDeviceProperties(&props, i);
    
    if(props.totalGlobalMem < ((unsigned int) 2*1024*1024*1024 - 1)){
      fprintf(stderr,
	      "Insufficient memory (%ld) on %s to run %s\n",
	      props.totalGlobalMem, props.name, name); 
      exit(-1); 
    }
    
    if(props.maxGridSize[0] < (int) ((unsigned int) 2*1024*1024*1024 - 1)){
      fprintf(stderr,
	      "Too few blocks per grid (%d) -- likely %s is an old device\n",
	      props.maxGridSize[0], props.name); 
      exit(-1); 
    }
  }
  
  return gpus; 
}
#else
int cudaCheck(char * name){
  printf("--- Not compiled with GPU support ---\n");
  return 0;
}
#endif

bool is_digits(const std::string &str)
{
  return str.find_first_not_of("0123456789.") == std::string::npos;
}

int main(int argc, char * argv[]) {
  
  // initial setup of default arguments
  
  int Kmin = 0;
  
  double delta = .01;
  int N = 100;
  
  mpfr_t t;
  mpfr_init2(t, 250);
  mpfr_set_str(t, "1000", 10, GMP_RNDN);
  
  mpz_t start;
  mpz_t length;
  
  mpz_init(start);
  mpz_init(length);
  
  mpz_set_str(start, "1", 10);

  int Zset = 1; 
  int startset = 0;
  int lengthset = 0; 
  int gpus = cudaCheck(argv[0]);
  int cpus = thread::hardware_concurrency();
  
  int number_of_cpu_threads = cpus; 
  int number_of_gpu_threads = 0; 
  
  if(gpus > 0){
    if(cpus >= gpus)
      number_of_gpu_threads = cpus; 
    else{
      number_of_gpu_threads = max(gpus*(cpus/gpus),1); 
      if(number_of_gpu_threads < 2*gpus) number_of_gpu_threads = 2*gpus;
      if(number_of_gpu_threads > 4*gpus) number_of_gpu_threads = 4*gpus;
    }
  }

  int checkRH = 0; 
  
  int required = 0; 
   
  while (1) {
    enum {KMIN = 2, T, DELTA, FILENAME, NUMTHREADS, N_OPTION, OUTPUT, HELP, NUM_GPU_THREADS, NUM_CPU_THREADS, START, LENGTH, CHECKRH};
    
    static struct option options[] = 
      {
	{"Kmin", required_argument, 0, KMIN},
	{"t", required_argument, 0, T},
	{"delta", required_argument, 0, DELTA},
	{"number_of_cpu_threads", required_argument, 0, NUM_CPU_THREADS},
	{"number_of_gpu_threads", required_argument, 0, NUM_GPU_THREADS},
	{"N", required_argument, 0, N_OPTION},
	{"output", required_argument, 0, OUTPUT},
	{"help", required_argument, 0, HELP},
	{"start", required_argument, 0, START},
	{"length", required_argument, 0, LENGTH},
	{"checkRH", 0, 0, CHECKRH},
	{0, 0, 0, 0}
      };
    
    int option_index = 0;
    int c = getopt_long(argc, argv, "", options, &option_index);

    if(argc >= 2){
      if(is_digits(argv[1])){
	mpfr_set_str(t, argv[1], 10, GMP_RNDN);
	required = 1; 
      }
    }else{
      usage(argv[0]);
    }
    
    if (c == -1) break;
    
    if(option_index == 0 && optarg == NULL){
      usage(argv[0]); 
      return -1; 
    }
        
    switch(options[option_index].val) {
    case KMIN:
      Kmin = atoi(optarg);
      break;
    case T:
      mpfr_set_str(t, optarg, 10, GMP_RNDN);
      required = 1; 
      break;
    case CHECKRH:
      checkRH = 1;
      break;
    case START:
      mpz_set_str(start, optarg, 10);
      Zset = 0;
      startset = 1; 
      break;
    case LENGTH:
      Zset = 0; 
      lengthset = 1;
      mpz_set_str(length, optarg, 10);
      if(!startset)
	mpz_set_ui(start, 1);
      break;
    case DELTA:
      delta = atof(optarg);
      break;
    case NUM_CPU_THREADS:
      number_of_cpu_threads = atoi(optarg);
      break;
    case NUM_GPU_THREADS:
      number_of_gpu_threads = atoi(optarg); 
      break;
    case N_OPTION:
      N = atoi(optarg);
      break;
    case OUTPUT:
      output_filename = optarg;
      break;
    case HELP:
      usage(argv[0]);
      break;
    }
  }

  if(!required){
    cout << "The first argument needs to be the t ordinate\n"; 
    usage(argv[0]); 
  }
  
  // Set-up serializing

  if(checkRH){
    
    mpfr_t temp;
    mpfr_t temp_pi;
    mpfr_init(temp_pi); 
    mpfr_init(temp);
    
    mpfr_log(temp, t, GMP_RNDN);
    mpfr_ui_div(temp, 1, temp, GMP_RNDN);
    mpfr_const_pi(temp_pi, GMP_RNDN);
    //mpfr_pow_d(temp, temp, 1.5, GMP_RNDN); 
    mpfr_mul(temp, temp, temp_pi, GMP_RNDN); 
    mpfr_mul_ui(temp, temp, 2, GMP_RNDN);
    mpfr_sqrt(temp, temp, GMP_RNDN);
    mpfr_pow_ui(temp, temp, 3, GMP_RNDN);
    //mpfr_mul_ui(temp, temp, temp_pi, GMP_RNDN);
    // mpfr_mul(temp, temp, temp, GMP_RNDN);
    //    mpfr_d_pow(temp, 1.5, temp,  GMP_RNDN); 
   
    
    delta = mpfr_get_d(temp, GMP_RNDN); 
    cout << "Setting delta = " << delta << endl;
    
    N = (int) pow(1/delta, 1.5);
    cout << "Setting N = " << N << endl; 
   
    mpfr_clear(temp_pi);
    mpfr_clear(temp); 

  }

  
  string filename2 = serialization_filename(N, delta, t);  
  
  complex<double> * S = new Complex[N];

  
  // Set-up the initial variables
  if(Zset || (!lengthset)){
    mpfr_t z;
    mpfr_init2(z, 250);
    mpfr_const_pi(z, GMP_RNDN); // z = pi
    mpfr_mul_2ui(z, z, 1, GMP_RNDN); // z = 2 pi
    mpfr_div(z, t, z, GMP_RNDN); // z = t / (2 pi)
    mpfr_sqrt(z, z, GMP_RNDN); // z = sqrt(t / 2 pi)
    mpfr_get_z(length, z, GMP_RNDD);  // length = floor (sqrt (t / 2pi))
    
    // MPI additional dissection
    
    // that is going to be the endpoint, really, or the endpoint + 1,
    // so we adjust the length if we are not starting at the endpoint.
    
    mpz_sub(length, length, start); 
    mpz_add_ui(length, length, 1u);
    
    mpfr_clear(z);

    if(mpz_cmp(start, length) > 0){
      cout << "length not specified and start greater than default length. Aborting" << endl;
      exit(0); 
    }
    
  }
  
#if HAVE_MPI
  
  int process_id = 0; 
  
  cout << "\rStarting MPI ... ";
  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &process_id);
  
  if(process_id == 0)
    cout << "done\n"; 
#endif
  
  if(process_id == 0)
    cout << number_of_cpu_threads << " CPU threads per host, " 
	 << number_of_gpu_threads << " GPU threads per host" << endl;

  // Delete the pthread from partial_zeta_sum
  
  pthread_t thread; 

  partial_zeta_sum(start, length, t, delta, N, S, Kmin, 
		   number_of_cpu_threads, number_of_gpu_threads, thread, gpus, filename2);

  int delta_prec = 2 + max((int) (log(1/delta) / log(10)),0); 
  
  cout << fixed;

  
  
  ofstream output_file;
  
  if(output_filename != ""){
    output_file.open(output_filename.c_str());
  }
  
  ostream & outFile = (output_filename != "" ? output_file : std::cout);
  
  outFile.precision(3);
  
  mpfr_exp_t r; 
  char * t_str = mpfr_get_str(NULL, &r, 10, 50, t, MPFR_RNDN); 
  
  outFile << "t = ";
  for(int i = 0; i < r; i++) outFile << t_str[i]; 
  outFile << "."; 
  outFile << t_str[r] << endl; 
  outFile << "N = " << N << " , delta = " << delta << "\n"; 
  
  free(t_str);

  
  if(Zset){
    compute_Z_from_rs_sum(t, delta, N, S, S);
    
    for(int n = 0; n < N; n++) {
      
      outFile.precision(delta_prec); 
      outFile << "Z(t + " 
	      << delta * n
	      << ") = ";
      outFile.precision(17); 
      outFile << S[n].real() << endl;  
    }
    
  }else{
    
    cout << "D = Dirichlet polynomial starting at "
	 << start << " and of length " << length << endl; 
    
    for(int n = 0; n < N; n++) {
      
      outFile.precision(delta_prec); 
      outFile << "D(t + " 
	      << delta * n
	      << ") = ";
      outFile.precision(17); 
      outFile << S[n] << endl;  
    }
    
  }
  
  
  delete [] S;

  //  MPI_Finalize();
  
  return 0;
}
