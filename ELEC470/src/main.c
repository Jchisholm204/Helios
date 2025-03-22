/**
 * @file main.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief 
 * @version 2.1
 * @date 2024-05-22
 * 
 * @copyright Copyright (c) 2024
 * 
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <pthread.h>
#include <time.h>

#define BENCH_INIT clock_t bench_start_time = clock()
#define BENCH_START bench_start_time = clock()
#define BENCH_END   (float)(1000.0*(clock()-bench_start_time)/CLOCKS_PER_SEC)

#define N_THREADS 1

struct thread_ini {
    float *A, *B;
    float res;
    size_t n;
};

float dot_product(float *A, float *B, size_t n){
    // NULL Check
    if(!A || !B) return 0.0;
    // Compute Result
    float res = 0.0;
    for(size_t i = 0; i < n; i++){
        res += A[i] * B[i];
    }
    return res;
}

float dot_product_omp(float *A, float *B, size_t n){
    // NULL Check
    if(!A || !B) return 0.0;
    // Compute Result
    float res = 0.0;
    omp_set_num_threads(N_THREADS);
// #pragma omp parallel
//     if(omp_get_thread_num() == 0)
//         printf("OMP Threads = %d\n", omp_get_num_threads());
#pragma omp parallel for reduction(+:res)
    for(size_t i = 0; i < n; i++){
        res += A[i] * B[i];
    }
    return res;
}

void *pthread_dp(void *pParam){
    struct thread_ini *arg = (struct thread_ini*)pParam;
    arg->res =  dot_product(arg->A, arg->B, arg->n);
    return arg;
}

float dot_product_pthread(float *A, float *B, size_t n){
    pthread_t t_hndl[N_THREADS];
    struct thread_ini t_args[N_THREADS];
    int per_thread = n/N_THREADS;
    for(int i = 0; i < N_THREADS; i++){
        t_args[i].A = &A[i*per_thread];
        t_args[i].B = &B[i*per_thread];
        t_args[i].n = per_thread;
        pthread_create(&t_hndl[i], NULL, pthread_dp, &t_args[i]);
    }
    float res = 0;
    for(int i = 0; i < N_THREADS; i++){
        pthread_join(t_hndl[i], (void**)&t_args[i]);
        res += t_args[i].res;
    }
    return res;
}

// Compute n vector X nxn matrix
float *mdot(float **M, float *V, size_t n){
    float *R = malloc(n*sizeof(float));
    for(int i = 0; i < n; i++){
        R[i] = dot_product(M[i], V, n);
    }
    return R;
}

// Compute n vector X nxn matrix
float *mdot_pll(float **M, float *V, size_t n){
    float *R = malloc(n*sizeof(float));
    for(int i = 0; i < n; i++){
        R[i] = dot_product_omp(M[i], V, n);
    }
    return R;
}

// Compute n vector X nxn matrix
float *mdot_epc(float **M, float *V, size_t n){
    float *R = malloc(n*sizeof(float));
#pragma omp parallel for
    for(int i = 0; i < n; i++){
        R[i] = dot_product(M[i], V, n);
    }
    return R;
}

void print_vec(float* V, size_t n){
    for(int i = 0; i < n; i++)
        printf("%0.2f ", V[i]);
    printf("\n");
}

void compare_vec(float *A, float *B, size_t n){
    for(size_t i = 0; i < n; i++){
        if(A[i] != B[i]){
            printf("FAIL\n");
            return;
        }
    }
    printf("PASS\n");
}

int main(int argc, char **argv){
    BENCH_INIT;
    size_t n = 16;
    // Allocate Vectors
    float *A = malloc(n*sizeof(float));
    float *B = malloc(n*sizeof(float));
    // Create matrix
    float **M = malloc(n*sizeof(float*));
    for(int i = 0; i < n; i++){
        M[i] = malloc(n*sizeof(float));
        for(int j = 0; j < n; j++){
            M[i][j] = 1;
        }
    }

    // Fill Vectors
    for(size_t i = 0; i < n; i++){
        A[i] = 2;
        B[i] = 1;
    }

    BENCH_START;
    float omp = dot_product_omp(A, B, n);
    printf("OMP Result = %0.2f\n", omp);
    printf("OMP Time = %0.2f\n", BENCH_END);
    BENCH_START;
    float st = dot_product(A, B, n);
    printf("ST  Result = %0.2f\n", st);
    printf("ST  Time = %0.2f\n", BENCH_END);
    BENCH_START;
    float pth = dot_product_pthread(A, B, n);
    printf("PTH Result = %0.2f\n", pth);
    printf("PTH Time = %0.2f\n", BENCH_END);
    BENCH_START;
    float *Vref = mdot(M, A, n);
    printf("MDOT Result:\n");
    printf("MDOT Time = %0.2f\n", BENCH_END);
    // print_vec(V, n);
    BENCH_START;
    float *V = mdot_pll(M, A, n);
    printf("PLL Result:\n");
    printf("PLL Time = %0.2f\n", BENCH_END);
    // print_vec(V, n);
    compare_vec(V, Vref, n);
    free(V);
    BENCH_START;
    V = mdot_epc(M, A, n);
    printf("EPC Result:\n");
    printf("EPC Time = %0.2f\n", BENCH_END);
    // print_vec(V, n);
    compare_vec(V, Vref, n);
    free(V);

    // Free Memory
    for(int i = 0; i < n; i++)
        free(M[i]);
    free(Vref);
    free(M);
    free(A);
    free(B);
    return 0;
}
