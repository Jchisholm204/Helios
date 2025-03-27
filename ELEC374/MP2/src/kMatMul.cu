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
    __shared__ float s_M[TILE_WIDTH][TILE_WIDTH + 1];
    __shared__ float s_N[TILE_WIDTH][TILE_WIDTH + 1];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float temp = 0.0f;
    for(int i = 0; i < (Width + TILE_WIDTH-1)/TILE_WIDTH; i++){
        int t_x = i*TILE_WIDTH + tx;
        int t_y = i*TILE_WIDTH + ty;
        // if(t_y + Col < Width)
            s_M[ty][tx] = M[MAT(Width, Row, t_x)];
        // if(t_x + Row < Width)
            s_N[ty][tx] = N[MAT(Width, t_y, Col)];

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            temp += s_M[ty][k] * s_N[k][tx];
        }

        __syncthreads();
    }

        P[MAT(Width, Row, Col)] = temp;
}



