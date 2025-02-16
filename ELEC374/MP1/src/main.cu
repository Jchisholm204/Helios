#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>

int getCudaCoresPerSM(int major, int minor) {
    // Approximate CUDA cores per SM based on architecture
    if (major == 2) return 32;  // Fermi
    if (major == 3) return 192; // Kepler
    if (major == 5) return 128; // Maxwell
    if (major == 6) return (minor == 1) ? 128 : 64; // Pascal
    if (major == 7) return 64;  // Volta and Turing
    if (major == 8) return 128; // Ampere
    if (major == 9) return 128; // Hopper
    return 0; // Unknown
}

void getDevProperties(void){
    int dc = 0;
    cudaGetDeviceCount(&dc);
    printf("There are %d CUDA Devices in this system\n", dc);
    cudaDeviceProp dp;
    cudaGetDeviceProperties(&dp, 0);

    printf("Device: %s\n", dp.name);
    printf("Core Clock: %.1f Mhz\n", (float)dp.clockRate/1000.0);
    printf("Stream Processors: %d\n", dp.multiProcessorCount);
    printf("Cores: %d\n", dp.multiProcessorCount*getCudaCoresPerSM(dp.major, dp.minor));
    printf("Warp Size: %d\n", dp.warpSize);
    printf("Global Memory: %0.2f GB\n", (float)dp.totalGlobalMem/1000000000);
    printf("Constant Memory: %0.2f KB\n", (float)dp.totalConstMem/1000);
    printf("Shared Memory per Block: %0.2f KB\n", (float)dp.sharedMemPerBlock/1000);
    printf("Registers per Block: %d\n", dp.regsPerBlock);
    printf("Threads per Block: %d\n", dp.maxThreadsPerBlock);
    printf("Maximum Block Dimensions: %d x %d x %d\n", dp.maxThreadsDim[0], dp.maxThreadsDim[1], dp.maxThreadsDim[2]);
    printf("Maximum Grid Size: %d x %d x %d\n", dp.maxGridSize[0], dp.maxGridSize[1], dp.maxGridSize[2]);

}


int main() {

    cudaError_t cudaStatus;
    getDevProperties();
    
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
