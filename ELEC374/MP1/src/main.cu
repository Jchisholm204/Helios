#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <iostream>
#include <time.h>

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

void MAT_fillRand(mat_t Mat, size_t N, int max_i, float div){
    srand(time(NULL));
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
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

void cpuMatMul(mat_t P, mat_t M, mat_t N, size_t size){
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            double sum = 0;
            for(int k = 0; k < size; k++)
                sum += M[MAT(i, k)]*N[MAT(k, j)];
            P[MAT(i, j)] = sum;
        }
    }
}

__global__ void __noinline__ gpuMatMul(mat_t P, mat_t M, mat_t N, size_t size){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    row = row % size;
    col = col % size;
    for(int i = row; i < size; i+=gridDim.x){
        for(int j = col; j < size; j+=gridDim.y){
            float pVal = 0;
            for(int k = 0; k < size; k++){
                pVal += M[MAT(i, k)]*N[MAT(k, j)];
            }
            P[MAT(i, j)] = pVal;
        }
    }
}

__global__ void __noinline__ gpuMatMulX(mat_t P, mat_t M, mat_t N, size_t size){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    row = row % size;
    col = col % size;
    float pVal = 0;
    for (int k = 0; k < size; k++){
        pVal += M[MAT(row, k)] * N[MAT(k, col)];
    }
    P[MAT(row, col)] = pVal;
}

#define TILE_WIDTH 8
__global__ void __noinline__ gpuMatMulT(mat_t P, mat_t M, mat_t N, size_t Width){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH]; // Shared memory for sub-matrix of M
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH]; // Shared memory for sub-matrix of N

    // Thread and block indices
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    // Identify the row and column of the P element
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    // Loop over tiles
    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        // Load tiles into shared memory
        Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];

        __syncthreads(); // Ensure all threads load data before proceeding

        // Perform matrix multiplication for the tile
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads(); // Synchronize before loading next tile
    }

    // Write result to global memory
    P[Row * Width + Col] = Pvalue;
}


mat_t runtest(size_t size, int g_size, int b_size){
    // Allocate Host side memories
    size_t mat_size = size*size*sizeof(float);
    M_host = (mat_t)malloc(mat_size);
    N_host = (mat_t)malloc(mat_size);
    P_host = (mat_t)malloc(mat_size);

    // Fill the matricies with data
    MAT_fillRand(M_host, size, 10, 1.4);
    MAT_fillRand(N_host, size, 11, 1.8);
    MAT_fill(P_host, 0);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();


    // Allocate Device Side Memory
    cudaMalloc(&M_dev, mat_size);
    cudaMalloc(&N_dev, mat_size);
    cudaMalloc(&P_dev, mat_size);

    cudaDeviceSynchronize();

    // Copy the data from the host to the device
    cudaMemcpy(M_dev, M_host, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(N_dev, N_host, mat_size, cudaMemcpyHostToDevice);
    cudaMemcpy(P_dev, P_host, mat_size, cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();

    cudaEventRecord(start, 0);
    // Run the Multiplication Kernel
    dim3 dimGrid(g_size, g_size, 1);
    dim3 dimBlock(b_size, b_size, 1);

    gpuMatMulX<<<dimGrid, dimBlock>>>(P_dev, M_dev, N_dev, size);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float memcpy_time = 0;
    cudaEventElapsedTime(&memcpy_time, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    printf("GPU for %ld took %0.2f ms\n", size, memcpy_time);

    
    // Copy the result back to the host
    cudaMemcpy(P_host, P_dev, mat_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(M_host, M_dev, mat_size, cudaMemcpyDeviceToHost);
    cudaMemcpy(N_host, N_dev, mat_size, cudaMemcpyDeviceToHost);


    FREE(M_host);
    FREE(N_host);
    cudaFree(M_dev);
    cudaFree(N_dev);
    cudaFree(P_dev);
    return P_host;
}

int main() {
    cudaError_t cudaStatus;
    getDevProperties();
    int batch = 0;
    for(int b = 2; b < 33; b=b<<1){
        printf("Batch %d\n", b);
        for (int i = 256; i < 4098; i = i << 1){
            mat_t P = runtest(i, (i + 1) / b, b);
            FREE(P);
        }
    }

    // Compute the reference matrix
    
    // double startT = (float)clock();
    // cpuMatMul(P_ref, M_host, N_host, MAT_N);
    // double endT = (float)clock();
    // printf("CPU for %d took %0.2f ms\n", MAT_N, endT-startT);

    // Output Matrix (Debug)
    // printf("Matrix M:\n");
    // MAT_print(M_host, MAT_N);
    // printf("Matrix N:\n");
    // MAT_print(N_host, MAT_N);
    // printf("Matrix CPU Multiplication Result:\n");
    // MAT_print(P_ref, MAT_N);
    // printf("Matrix GPU Multiplication Result:\n");
    // MAT_print(P_host, MAT_N);

    // printf("Detected %d Errors\n", MAT_compare(P_host, P_ref, 0.1));

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
