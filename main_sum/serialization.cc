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
  out.precision(30);
  for(int i = 0; i < M; i++){
    out << S[i] << endl; 
  }
  out.flush();
  out.close();
}

void deserialize_local(string filename, Complex * S, int M){
    ifstream in;
    in.open(filename, std::ifstream::in);
    in.precision(30);
    for(int i = 0; i < M; i++)
      in >> S[i]; 
    in.close();
}

bool is_file_exist(const char *fileName)
{
  std::ifstream infile(fileName);
  return infile.good();
}

