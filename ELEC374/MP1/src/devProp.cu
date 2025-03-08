/**
 * @file devProp.cu
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief 
 * @version 0.1
 * @date 2025-03-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#include "devProp.hpp"
#include <stdio.h>

static int getCudaCoresPerSM(int major, int minor) {
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
    int n_dev = 0;
    cudaGetDeviceCount(&n_dev);
    printf("There are %d CUDA Devices in this system\n", n_dev);
    for (int i = 0; i < n_dev; i++){
        cudaDeviceProp dp;
        cudaGetDeviceProperties(&dp, 0);

        // Print Device Name
        printf("=======%*s%*s=======\n",20+((int)strlen(dp.name))/2,dp.name,20-(int)strlen(dp.name)/2,"");
        // Device Core Clock (kHz/1000=MHz)
        printf("Core Clock: %.1f MHz\n", (float)dp.clockRate / 1000.0);
        // Stream Processors
        printf("Stream Processors: %d\n", dp.multiProcessorCount);
        // Cores - From the Generational Specifications
        printf("Cores: %d\n", dp.multiProcessorCount * getCudaCoresPerSM(dp.major, dp.minor));
        // Warp Size
        printf("Warp Size: %d\n", dp.warpSize);
        // Global Memory (with Byte to GB conversion)
        printf("Global Memory: %0.2f GB\n", (float)dp.totalGlobalMem / 1000000000);
        // Constant Memory (with Byte to KB conversion)
        printf("Constant Memory: %0.2f KB\n", (float)dp.totalConstMem / 1000);
        // Shared Block Memory (with Byte to KB conversion)
        printf("Shared Memory per Block: %0.2f KB\n", (float)dp.sharedMemPerBlock / 1000);
        // Max Registers Per Block
        printf("Registers per Block: %d\n", dp.regsPerBlock);
        // Max Threads per Block
        printf("Threads per Block: %d\n", dp.maxThreadsPerBlock);
        // Max Block Dimensions (Threads in a Block)
        printf("Maximum Block Dimensions: %d x %d x %d\n", dp.maxThreadsDim[0], dp.maxThreadsDim[1], dp.maxThreadsDim[2]);
        // Maximum Grid Size (Blocks in a Grid)
        printf("Maximum Grid Size: %d x %d x %d\n", dp.maxGridSize[0], dp.maxGridSize[1], dp.maxGridSize[2]);
        printf("=======================================================\n");
    }
}

