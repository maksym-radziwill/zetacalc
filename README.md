This program computes the Riemann zeta function on the critical line. It uses
the GPU to perform the calculations related to the Riemann Siegel formula. It
is based on the code of Bober and Hiary and ports what they call the stage2
computation to the GPU. Here is a list of improvements compared to the code
of Bober and Hiary:

   * Support for NVIDIA GPU's
   * Support for multi-GPU architectures
   * Added native support for MPI
   * MPI works in heterogeneous environments with mixed GPU and CPU systems
   * Support for serialization (program can be interrupted and resumed later
     on a MPI cluster or a single machine).
   * Minor improvements in performance of CPU code of Bober and Hiary
   * Cleaned up user interface and command line options

=== PERFORMANCE ===

The rough improvements in performance should be of the order of 1 GPU = 80 CPU
assuming that both the GPU and CPU are top of the line. Important command line
options for performance are number_of_cpu_threads and number_of_gpu_threads.
The number of cpu threads is important for GPU performance and should be roughly
the number of cores in the system. The number of GPU threads should at least be
a multiple of 2 of the number of GPU, but possibly much higher (up to a multiple
of 4) depending on the strength of the GPU. The exact parameter should be adjusted
to maximize GPU utilization while the program is ran, as shown by the nvidia-smi
command.

=== CORRECTNESS ===

The code has been checked and benchmarked in the range 10^{6} < t < 10^{29}. It
might malfunction for very small t ran on a cluster with many machines. The
behavior and performance has not been benchmarked in the range t = 10^{30} and
higher. The precision appears to be 5 decimal points off from the results of Bober
and Hiary which is insignificant for most applications (such as checking the
Riemann Hypothesis or finding large values of the Riemann zeta-function). 

=== TESTING ===

A script called script_aws.sh has been included to allow quick deployement on
Amazon ec. 2 GPU clusters. 