#include <stdio.h>
 
double compute_capability(cudaDeviceProp devProp){
       if(devProp.major == 9999) return 9999; 
         			
       return ((double) devProp.major + (((double) devProp.minor) / 10));
}
 
int main()
{
    int devCount;
    cudaGetDeviceCount(&devCount);

    if(devCount == 0) exit(1); 

    double cc = 10000;  // Initial compute capability -- needs to be greater than 9999
    	      		// as 9999 is the compute capability of a virtual device

			

    for (int i = 0; i < devCount; ++i)
    {
	cudaDeviceProp devProp;
	cudaGetDeviceProperties(&devProp, i);
	double cx = compute_capability(devProp);
	if(cx < cc) cc = cx;
	if(cc == 9999) exit(1); // No cuda device available 
    }
    
    int major = (int) cc; 
    int minor = (int) ((double) (((cc - (double) major)*10)));

    printf("%d%d\n", major, minor); 
    exit(0); 
}