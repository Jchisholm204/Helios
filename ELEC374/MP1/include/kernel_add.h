#ifndef _KERNEL_ADD_H_
#define _KERNEL_ADD_H_

#include <stdio.h>

static __global__ void __noinline__ cudaAddKernel(float *A, float *B, float *C, size_t n){
    int index = blockIdx.x*blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for(int i = index; i < n; i+=stride)
        C[i] = A[i] + B[i];
}

static void cudaAdd(void){
    int N = 1<<20;
    float *A, *B, *C;
    cudaMallocManaged(&A, N*sizeof(float));
    cudaMallocManaged(&B, N*sizeof(float));
    cudaMallocManaged(&C, N*sizeof(float));
    for(int i = 0; i < N; i++){
        A[i] = 1.0f;
        B[i] = 2.0f;
    }

    int blockSize = 256;
    // Allocate one thread per add
    int numBlocks = (N + blockSize - 1)/blockSize;
    printf("Launching %d blocks each with %d threads\n", numBlocks, blockSize);
    printf("%d threads total\n", numBlocks*blockSize);
    printf("Each thread will add %0.1f elements\n", (float)N/((float)numBlocks*(float)blockSize));

    cudaAddKernel<<<numBlocks, blockSize>>>(A, B, C, N);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA kernel launch failed: %s\n", cudaGetErrorString(err));
    }
    printf("Waiting for Sync\n");
    cudaDeviceSynchronize();

    printf("Total Error: ");
    float err_count = 0;
    for(int i = 0; i < N; i++){
        err_count += A[i] + B[i] - C[i];
    }
    printf("%0.2f\n", err_count);
    
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
}

#endif
