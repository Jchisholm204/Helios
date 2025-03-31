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

#define BENCH_INIT double bench_start_time = omp_get_wtime()
#define BENCH_START bench_start_time = omp_get_wtime()
#define BENCH_END   (float)(1000.0*(omp_get_wtime()-bench_start_time))

#define N_THREADS 4
// #define WORK_MAT (1<<12)
// #define WORK_VEC (1<<24)
#define N_TRIALS 10

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
    if(!A || !B) return 0.0;
    float res = 0.0;
    omp_set_num_threads(N_THREADS);
    #pragma omp parallel for schedule(static, n / N_THREADS) reduction(+:res)
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

extern int mpi_main(int argc, char** argv);

int run_test(size_t WORK_VEC, size_t WORK_MAT){
    BENCH_INIT;
    // Allocate Vectors
    float *A = malloc(WORK_VEC*sizeof(float));
    float *B = malloc(WORK_VEC*sizeof(float));
    // Create matrix
    // float **M = malloc(WORK_MAT*sizeof(float*));
    // for(int i = 0; i < WORK_MAT; i++){
    //     M[i] = malloc(WORK_MAT*sizeof(float));
    //     for(int j = 0; j < WORK_MAT; j++){
    //         M[i][j] = 1;
    //     }
    // }

    // Fill Vectors
    for(size_t i = 0; i < WORK_VEC; i++){
        A[i] = 2;
        B[i] = 2;
    }


    // printf("Work Mat = %d\n", WORK_MAT);
    // printf("Work Vec = %ld\n", WORK_VEC);
    printf("Starting Tests\n");

    float time = 0;
    BENCH_START;
    for(int i = 0; i < N_TRIALS; i++){
        float st = dot_product(A, B, WORK_VEC);
    }
    time = BENCH_END/N_TRIALS;
    // printf("ST  Result = %0.2f\n", dot_product(A, B, WORK_VEC));
    printf("ST  Time = %0.4f\n", time);
    time = 0;
    BENCH_START;
    for(int i = 0; i < N_TRIALS; i++){
        float omp = dot_product_omp(A, B, WORK_VEC);
    }
    time = BENCH_END/N_TRIALS;
    // printf("OMP Result = %0.2f\n", dot_product_omp(A, B, n));
    printf("OMP Time = %0.4f\n", time);
    time = 0;
    BENCH_START;
    for(int i = 0; i < N_TRIALS; i++){
        float pth = dot_product_pthread(A, B, WORK_VEC);
    }
    time = BENCH_END/N_TRIALS;
    // printf("PTH Result = %0.2f\n", dot_product_pthread(A, B, n));
    printf("PTH Time = %0.4f\n", time);


    // time = 0;
    // for(int i = 0; i < N_TRIALS; i++){
    //     BENCH_START;
    //     float *Vref = mdot(M, A, WORK_MAT);
    //     time += BENCH_END;
    //     free(Vref);
    // }
    // // printf("MDOT Result:\n");
    // printf("MDOT Time = %0.2f\n", time/N_TRIALS);
    // // print_vec(V, n);
    // time = 0;
    // for(int i = 0; i < N_TRIALS; i++){
    //     BENCH_START;
    //     float *V = mdot_pll(M, A, WORK_MAT);
    //     time += BENCH_END;
    //     free(V);
    // }
    // // printf("PLL Result:\n");
    // printf("PLL Time = %0.2f\n", time/N_TRIALS);
    // // print_vec(V, n);
    // // compare_vec(V, Vref, WORK_MAT);
    // time = 0;
    // for(int i = 0; i < N_TRIALS; i++){
    //     BENCH_START;
    //     float *V = mdot_epc(M, A, WORK_MAT);
    //     time += BENCH_END;
    //     free(V);
    // }
    // printf("EPC Result:\n");
    // printf("EPC Time = %0.2f\n", time/N_TRIALS);
    // print_vec(V, n);
    // compare_vec(V, Vref, WORK_MAT);

    // Free Memory
    // for(int i = 0; i < WORK_MAT; i++)
    //     free(M[i]);
    // free(M);
    free(A);
    free(B);
    return 0;
}


int main(int argc, char **argv){
    // return mpi_main(argc, argv);
    printf("N Threads = %d\n", N_THREADS);
    printf("N Trials = %d\n", N_TRIALS);
    for(int i = 20; i < 27; i++){
        printf("Running Test 2^%d\n", i);
        run_test(1<<i, 1<<i);
    }
    return 0;
}
