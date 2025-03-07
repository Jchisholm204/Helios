#include "cuda_runtime.h"
#include "device_launch_parameters.h"
 
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
 
#include <chrono>

#define BLOCK_WIDTH     16
#define MATRIX_WIDTH    4096
 
__global__ void MatrixMulKernel(float* M, float* N, float* P, int width)
{
    int Row = blockIdx.y * blockDim.y + threadIdx.y;
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
 
    if (Row < width && Col < width) {
        float Pval = 0;
        for (int k = 0; k < width; k++) Pval += M[Row * width + k] * N[k * width + Col];
        P[Row * width + Col] = Pval;
    }
}
 
int correct_output(float* host, float* device, int width)
{
    const float tolerance = 1e-5;                     // Small tolerance for floating-point comparisons
    for (int i = 0; i < width * width; i++) {
        if (fabs(host[i] - device[i]) > tolerance)    // Compare with tolerance
            return 0;
    }
    return 1;
}
 
 
int getCoresPerSM(int major) {
    switch (major) {
    case 3: return 192;     // Kepler
    case 5: return 128;     // Maxwell
    case 6: return 64;      // Pascal (6.0) or 128 (6.1, 6.2)
    case 7: return 64;      // Volta and Turing
    case 8: return 128;     // Ampere
    case 9: return 128;     // Hopper
    default: return -1;     // Unknown
    }
}
 
void randMatGen(float* M, int width) {
    for (int i = 0; i < width * width; i++) {
        M[i] = rand();
    }
}
 
void matrixMul(float* M, float* N, float* P, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            float Pval = 0;
            for (int k = 0; k < width; k++) {
                Pval += M[i * width + k] * N[k * width + j];
            }
            P[i * width + j] = Pval;
        }
    }
}
 
void printMatrix(float* M, int width) {
    printf("{ ");
 
    for (int i = 0; i < width; i++) {
        printf("\n  ");
        for (int j = 0; j < width; j++) {
            printf(" %f ", M[i*width + j]);
        }
    }
    printf("\n}\n");
 
}
 
 
void lukesCode(void){
    // Gets information about devices connected
    int nd;
    cudaGetDeviceCount(&nd);
    printf("Numer of Devices: %d\n\n", nd);
    for (int d = 0; d < nd; d++)
    {
        cudaDeviceProp dp;
        cudaGetDeviceProperties(&dp, d);
        int major = dp.major;
        int coresPerSM = getCoresPerSM(major);
 
        printf("======================== Device %d ========================\n", nd);
        printf("GPU type: %s\n", dp.name);
        printf("Clock rate: %d\n", dp.clockRate);
        printf("Number of streaming multiprocessors: %d\n", dp.multiProcessorCount);
        printf("Number of cores: %d\n", dp.multiProcessorCount * coresPerSM);
        printf("Warp size: %d\n", dp.warpSize);
        printf("Amount of global memory: %lld\n", (long long)dp.totalGlobalMem);
        printf("Amount of contant memory: %d\n", dp.totalConstMem);
        printf("Amount of shared memory per block: %d\n", dp.sharedMemPerBlock);
        printf("Number of registers per block: %d\n", dp.regsPerBlock);
        printf("Max threads per block: %d\n", dp.maxThreadsPerBlock);
        printf("Max dimension of block: %d\n", dp.maxThreadsDim);
        printf("Max dimension of grid: %d x %d x %d\n", dp.maxGridSize[0], dp.maxGridSize[1], dp.maxGridSize[2]);
        printf("===========================================================\n");
 
    }
 
 
    int nbytes = MATRIX_WIDTH * MATRIX_WIDTH * sizeof(float);
 
    // allocate host memory
    float* h_M = (float*)malloc(nbytes);
    float* h_N = (float*)malloc(nbytes);
    float* h_P = (float*)malloc(nbytes);    
    memset(h_P, 0, nbytes);
 
    randMatGen(h_M, MATRIX_WIDTH);
    randMatGen(h_N, MATRIX_WIDTH);
 
 
    // allocate device memory
    float* d_M, * d_N, * d_P;
 
    cudaMalloc((void**)&d_M, nbytes);
    cudaMalloc((void**)&d_N, nbytes);
    cudaMalloc((void**)&d_P, nbytes);
    //cudaMemset(d_P, 0, nbytes);
 
 
    // set dimensions
    int NumBlocks = MATRIX_WIDTH / BLOCK_WIDTH;
    //int NumBlocks = (MATRIX_WIDTH + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
    if (MATRIX_WIDTH % BLOCK_WIDTH) NumBlocks++;
    dim3 dimGrid = dim3(NumBlocks, NumBlocks);
    dim3 dimBlock = dim3(BLOCK_WIDTH, BLOCK_WIDTH);
 
    // create cuda event handles
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
 
    cudaEvent_t h_to_d_start, h_to_d_stop, d_to_h_start, d_to_h_stop;
    cudaEventCreate(&h_to_d_start);
    cudaEventCreate(&h_to_d_stop);
    cudaEventCreate(&d_to_h_start);
    cudaEventCreate(&d_to_h_stop);
 
    cudaDeviceSynchronize();
    float gpu_time = 0.0f;
    float h_to_d_time = 0.0f;
    float d_to_h_time = 0.0f;
 
    // asynchronously issue work to the GPU (all to stream 0)
    // Host to Device Data transfer
    cudaEventRecord(h_to_d_start, 0);
    cudaMemcpyAsync(d_M, h_M, nbytes, cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(d_N, h_N, nbytes, cudaMemcpyHostToDevice, 0);
    //cudaMemcpyAsync(d_P, h_P, nbytes, cudaMemcpyHostToDevice, 0);
    cudaEventRecord(h_to_d_stop, 0);
    cudaEventSynchronize(h_to_d_stop);
 
 
    // Call on kernal
    cudaEventRecord(start, 0);
    //MatrixMulKernel<<<dimGrid, dimBlock, 0, 0 >>>(d_M, d_N, d_P, MATRIX_WIDTH);
    MatrixMulKernel <<<1, 1, 0, 0 >>> (d_M, d_N, d_P, MATRIX_WIDTH);
 
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
 
 
    // Device to Host Data ransfer
    cudaEventRecord(d_to_h_start, 0);
    cudaMemcpyAsync(h_P, d_P, nbytes, cudaMemcpyDeviceToHost, 0);
    cudaEventRecord(d_to_h_stop, 0);
    cudaEventSynchronize(d_to_h_stop);
 
 
    float* P = (float*)malloc(nbytes);
    auto start_cpu = std::chrono::high_resolution_clock::now();
    matrixMul(h_M, h_N, P, MATRIX_WIDTH);
    auto stop_cpu = std::chrono::high_resolution_clock::now();
 
    std::chrono::duration<float, std::milli> cpu_time = stop_cpu - start_cpu;
 
    cudaEventElapsedTime(&gpu_time, start, stop); // time difference between start and stop
    cudaEventElapsedTime(&h_to_d_time, h_to_d_start, h_to_d_stop); // time difference between start and stop
    cudaEventElapsedTime(&d_to_h_time, d_to_h_start, d_to_h_stop); // time difference between start and stop
 
    // print the GPU times
    printf("\nTime spent executing by the GPU: %.4f\n", gpu_time);
    printf("\nTime spent executing by the CPU: %.4f\n", cpu_time.count());
 
    printf("\nTime spent transfering data from host to device: %.4f\n", h_to_d_time);
    printf("\nTime spent transfering data from device to host: %.4f\n", d_to_h_time);
 
    // check the output for correctness
    bool wasCorrect = correct_output(P, h_P, MATRIX_WIDTH);
 
    if (wasCorrect) printf("Test PASSED");
    else            printf("Test FAILED");
 
 
    // release resources
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaEventDestroy(d_to_h_start);
    cudaEventDestroy(d_to_h_stop);
    cudaEventDestroy(h_to_d_start);
    cudaEventDestroy(h_to_d_stop);
    free(h_M);
    free(h_N);
    free(h_P);
    free(P);
 
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
 
    cudaDeviceReset();
}