/**
 * @file kMatMul.cu
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief 
 * @version 0.1
 * @date 2025-03-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#include "kMatMul.hpp"


__global__ void __noinline__ kMatMul_ssm(mat_t P, mat_t M, mat_t N, size_t size){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    row = row > size ? size : row;
    col = col > size ? size : col;
    for(int i = row; i < size; i+=gridDim.x){
        for(int j = col; j < size; j+=gridDim.y){
            float pVal = 0;
            for(int k = 0; k < size; k++){
                pVal += M[MAT(size, i, k)]*N[MAT(size, k, j)];
            }
            P[MAT(size, i, j)] = pVal;
        }
    }
}

__global__ void __noinline__ kMatMul_msm(mat_t P, mat_t M, mat_t N, size_t size){
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    // row = row > size ? size : row;
    // col = col > size ? size : col;
    if (row < size && col < size){
        float pVal = 0;
        for (int k = 0; k < size; k++){
            pVal += M[MAT(size, row, k)] * N[MAT(size, k, col)];
        }
        P[MAT(size, row, col)] = pVal;
    }
}

#define TILE_WIDTH 32
__global__ void kMatMul_tiled(mat_t P, mat_t M, mat_t N, size_t Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH + 1]; // Avoid bank conflicts

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0.0f;

    for (int ph = 0; ph < (Width + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        if (Row < Width && (ph * TILE_WIDTH + tx) < Width)
            Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        else
            Mds[ty][tx] = 0.0f; // Avoid out-of-bounds memory access

        if (Col < Width && (ph * TILE_WIDTH + ty) < Width)
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        else
            Nds[ty][tx] = 0.0f;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }

        __syncthreads();
    }

    if (Row < Width && Col < Width)
        P[Row * Width + Col] = Pvalue;
}
