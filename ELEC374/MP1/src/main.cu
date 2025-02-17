#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>

#include "info.h"
#include "kernel_add.h"

typedef float* mat_t;
mat_t M_host, N_host, P_host, P_ref;
mat_t M_dev, N_dev, P_dev;
#define MAT_N (256)
#define MAT_SIZE (sizeof(float)*MAT_N*MAT_N)
#define MAT(row, col) (MAT_N*row + col)

#define FREE(p) if(!p){ free(p); p = NULL; }

bool inRange(float val, float trg, float rng){
    if (val > (trg+rng)) return false;
    if (val < (trg-rng)) return false;
    return true;
}

int MAT_compare(mat_t A, mat_t B, float rng){
    int err_count = 0;
    for(int i = 0; i < MAT_N; i++){
        for(int j = 0; j < MAT_N; j++){
            err_count += (1-inRange(A[MAT(i,j)], B[MAT(i,j)], rng));
        }
    }
    return err_count;
}

void MAT_print(mat_t Mat, size_t n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%5.2f ", Mat[MAT(i, j)]);
        }
        printf("\n");
    }
}

void MAT_fillRand(mat_t Mat, int max_i, float div){
    srand(time(NULL));
    for(int i = 0; i < MAT_N; i++){
        for(int j = 0; j < MAT_N; j++){
            Mat[MAT(i, j)] = (float)(rand() % max_i)/div;
        }
    }
}

void MAT_fill(mat_t Mat, float val){
    for(int i = 0; i < MAT_N; i++){
        for(int j = 0; j < MAT_N; j++){
            Mat[MAT(i, j)] = val;
        }
    }
}

void cpuMatMul(mat_t P, mat_t M, mat_t N){
    for(int i = 0; i < MAT_N; i++){
        for(int j = 0; j < MAT_N; j++){
            double sum = 0;
            for(int k = 0; k < MAT_N; k++)
                sum += M[MAT(i, k)]*N[MAT(k, j)];
            P[MAT(i, j)] = sum;
        }
    }
}

__global__ void __noinline__ gpuMatMul(mat_t P, mat_t M, mat_t N){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;

    row = row ;
    col = col ;
    for(int i = row; i < MAT_N; i+=gridDim.x){
        for(int j = col; j < MAT_N; j+=gridDim.y){
            float pVal = 0;
            for(int k = 0; k < MAT_N; k++){
                pVal += M[MAT(i, k)]*N[MAT(k, j)];
            }
            P[MAT(i, j)] = pVal;
        }
    }
}

int main() {
    cudaError_t cudaStatus;
    getDevProperties();

    // cudaAdd();

    // Allocate Host side memories
    M_host = (mat_t)malloc(MAT_SIZE);
    N_host = (mat_t)malloc(MAT_SIZE);
    P_host = (mat_t)malloc(MAT_SIZE);
    P_ref  = (mat_t)malloc(MAT_SIZE);

    // Fill the matricies with data
    MAT_fillRand(M_host, 10, 1.4);
    MAT_fillRand(N_host, 11, 1.8);
    MAT_fill(P_host, 0);

    // Allocate Device Side Memory
    cudaMalloc(&M_dev, MAT_SIZE);
    cudaMalloc(&N_dev, MAT_SIZE);
    cudaMalloc(&P_dev, MAT_SIZE);
    cudaDeviceSynchronize();

    // Copy the data from the host to the device
    cudaMemcpy(M_dev, M_host, MAT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(N_dev, N_host, MAT_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(P_dev, P_host, MAT_SIZE, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    // Run the Multiplication Kernel
    int n_threads = 32;
    int n_blocks = MAT_N/n_threads;
    dim3 dimGrid(n_blocks, n_blocks, 1);
    dim3 dimBlock(n_threads, n_threads, 1);

    gpuMatMul<<<dimGrid, dimBlock>>>(P_dev, M_dev, N_dev);

    cudaDeviceSynchronize();
    
    // Copy the result back to the host
    cudaMemcpy(P_host, P_dev, MAT_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(M_host, M_dev, MAT_SIZE, cudaMemcpyDeviceToHost);
    cudaMemcpy(N_host, N_dev, MAT_SIZE, cudaMemcpyDeviceToHost);

    // Compute the reference matrix
    cpuMatMul(P_ref, M_host, N_host);
    // printf("Matrix M:\n");
    // MAT_print(M_host, MAT_N);
    // printf("Matrix N:\n");
    // MAT_print(N_host, MAT_N);
    // printf("Matrix CPU Multiplication Result:\n");
    // MAT_print(P_ref, MAT_N);
    // printf("Matrix GPU Multiplication Result:\n");
    // MAT_print(P_host, MAT_N);

    printf("Detected %d Errors\n", MAT_compare(P_host, P_ref, 0.1));

    FREE(M_host);
    FREE(N_host);
    FREE(P_host);
    FREE(P_ref);
    cudaFree(M_dev);
    cudaFree(N_dev);
    cudaFree(P_dev);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
