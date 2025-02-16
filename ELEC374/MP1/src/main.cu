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

float *M, *N, *P;
#define MAT_SIZE (1 << 20)
#define MAT(row, col) (MAT_SIZE*row + col)

#define FREE(p) if(!p){ free(p); p = NULL; }

void MAT_print(float *Mat, size_t n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%5.1f ", Mat[MAT(i, j)]);
        }
        printf("\n");
    }
}

void MAT_fill(float *Mat, size_t n, int max_i, float div){
    srand(time(NULL));
    for(int i = 0; i < MAT_SIZE; i++){
        for(int j = 0; j < MAT_SIZE; j++){
            Mat[MAT(i, j)] = (float)(rand() % max_i)/div;
        }
    }
}

__global__ void MatrixMulKernel(float *M, float *N, float *P, int width){
    int row = blockIdx.y *blockDim.y + threadIdx.y;
    int col = blockIdx.x *blockDim.x + threadIdx.x;
    if(row < width && col < width){
        float Pvalue = 0;
        for(int k = 0; k < width; ++k)
            Pvalue += M[row*width + k] * N[k*width + col];
        P[row*width + col] = Pvalue;
    }
}

__global__
void add(float *A, float *B, float *C, size_t n){
    for(int i = 0; i < n; i++)
        C[i] = A[i] + B[i];
}

int main() {

    cudaError_t cudaStatus;
    getDevProperties();

    // printf("Allocating Host Matricies\n");
    // M = (float*)malloc(sizeof(float)*MAT_SIZE*MAT_SIZE);
    // N = (float*)malloc(sizeof(float)*MAT_SIZE*MAT_SIZE);
    // P = (float*)malloc(sizeof(float)*MAT_SIZE*MAT_SIZE);
    // if(!M || !N || !P){
    //     fprintf(stderr, "Host Matrix Allocation Failure");
    //     FREE(M);
    //     FREE(N);
    //     FREE(P);
    //     exit(-1);
    // }
    // Populate the Matricies with random values
    // MAT_fill(M, MAT_SIZE, 1000, 18.321);
    // MAT_fill(N, MAT_SIZE, 1000, 21.123);
    // Print out sample Matricies
    // printf("Matrix M:\n");
    // MAT_print(M, 10);
    //
    // printf("Matrix N:\n");
    // MAT_print(N, 10);

    float *A, *B, *C;
    // A = new float[MAT_SIZE];
    // B = new float[MAT_SIZE];
    // C = new float[MAT_SIZE];
    cudaMallocManaged(&A, MAT_SIZE*sizeof(float));
    cudaMallocManaged(&B, MAT_SIZE*sizeof(float));
    cudaMallocManaged(&C, MAT_SIZE*sizeof(float));
    for(int i = 0; i < MAT_SIZE; i++){
        A[i] = i;
        B[i] = i*10;
    }
    // printf("Array A:\n");
    // for(int i = 0; i < MAT_SIZE; i++){
    //     printf("%5.1f ", A[i]);
    // }
    // printf("\n");
    //
    // printf("Array B:\n");
    // for(int i = 0; i < MAT_SIZE; i++){
    //     printf("%5.1f ", B[i]);
    // }
    printf("\n");
    
    add<<<1, 1>>>(A, B, C, MAT_SIZE);

    cudaDeviceSynchronize();

    printf("Result:\n");
    for(int i = 0; i < MAT_SIZE; i++){
        // printf("%5.1f ", C[i]);
        if(A[i] + B[i] != C[i])
            printf("ADD ERR %5.1f ", C[i]);
    }
    printf("\n");
    
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    // MatrixMulKernel<<<1, 1>>>(M, N, P, MAT_SIZE);

    // printf("Matrix P:\n");
    // MAT_print(P, 10);

    printf("Freeing Host Matricies\n");
    FREE(M);
    FREE(N);
    FREE(P);
    
    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}
