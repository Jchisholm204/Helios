#include <stdio.h>
#include <iostream>
#include <time.h>
#include <chrono>
#include <cpuMatMul.hpp>

bool inRange(float val, float trg, float rng){
    if (val > (trg+rng)) return false;
    if (val < (trg-rng)) return false;
    return true;
}

int MAT_compare(mat_t A, mat_t B, size_t N, float rng){
    int err_count = 0;
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            err_count += (1-inRange(A[MAT(N, i,j)], B[MAT(N, i,j)], rng));
        }
    }
    return err_count;
}

void MAT_print(mat_t Mat, size_t n){
    for(int i = 0; i < n; i++){
        for(int j = 0; j < n; j++){
            printf("%5.2f ", Mat[MAT(n, i, j)]);
        }
        printf("\n");
    }
}

void MAT_fillRand(mat_t Mat, size_t N, int max_i, float div){
    srand(time(NULL));
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            Mat[MAT(N, i, j)] = (float)(rand() % max_i)/div;
        }
    }
}

void MAT_fill(mat_t Mat, size_t N, float val){
    for(int i = 0; i < N; i++){
        for(int j = 0; j < N; j++){
            Mat[MAT(N, i, j)] = val;
        }
    }
}

void cpuMatMul(mat_t P, mat_t M, mat_t N, size_t size){
    for(int i = 0; i < size; i++){
        for(int j = 0; j < size; j++){
            double sum = 0;
            for(int k = 0; k < size; k++)
                sum += M[MAT(size, i, k)]*N[MAT(size, k, j)];
            P[MAT(size, i, j)] = sum;
        }
    }
}
