/**
 * @file main.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief MPI Vector/Matrix Dot Product Code
 * @version 2.1
 * @date 2024-05-30
 * 
 * @copyright Copyright (c) 2025
 * 
 * When I wrote this I had a singular thought... How creative can I make this code?
 * As a result, I wrote this code as though MPI uses zero copy transfers and assumed
 *  that network delay is an imaginary concept. 
 *      Please ignore my error log, 
 *      - Jacob Chisholm
 * 
 */

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <mpi.h>


#define WORK_MAT ((1ULL<<15))
#define WORK_VEC ((1ULL<<15))
// #define WORK_MAT (16)
// #define WORK_VEC (16)
#define N_TRIALS 1

int procid, nprocs;

extern void print_vec(float* V, size_t n);

float dot_product_mpi(float *A, float *B, size_t n){
    // NULL Check
    if(!A || !B) return 0.0;
    // Compute Result
    float res = 0.0;
    MPI_Scatter(A, n/nprocs, MPI_FLOAT, A, n/nprocs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    MPI_Scatter(B, n/nprocs, MPI_FLOAT, B, n/nprocs, MPI_FLOAT, 0, MPI_COMM_WORLD);
    for(size_t i = 0; i < n/nprocs; i++){
        res += A[i] * B[i];
    }
    // printf("proc %d preres = %0.2f\n", procid, res);
    MPI_Allreduce(&res, &res, 1, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    return res;
}

float * mdot_mpi(float *M, float *V, size_t n){
    // I'm going to need this right?
    float *T = malloc(n*sizeof(float));
    // Result Matrix, allocate to procs, return to all procs 
    float *R = malloc(n*sizeof(float));
    for(int i =0; i < n; i++)
        R[i] = 0;
    // Broadcast V to all arrays
    MPI_Bcast(V, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    // Block iterate over the array
    for(int i = 0; i < n; i+=nprocs){
        MPI_Scatter(&M[i*n], n, MPI_FLOAT, T, n, MPI_FLOAT, 0, MPI_COMM_WORLD);
        for(int j = 0; j < n; j++)
            R[i+procid] += T[j]*V[j];
        for(int p = 0; p < nprocs; p++)
            MPI_Bcast(&R[i+p], 1, MPI_FLOAT, p, MPI_COMM_WORLD);
    }
    free(T);
    MPI_Barrier(MPI_COMM_WORLD);
    // if(procid == 0)
    //     print_vec(R, n);
    return R;
}

int mpi_main(int argc, char** argv){
    // Init MPI
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    // Allocate Vectors
    float *A = malloc(WORK_VEC*sizeof(float));
    float *B = malloc(WORK_VEC*sizeof(float));
    // Create matrix
    float *M = malloc(WORK_MAT*WORK_MAT*sizeof(float));
    if(procid == 0){
        // Fill Vectors
        for(size_t i = 0; i < WORK_VEC; i++){
            A[i] = i;
            B[i] = 1;
        }
        for(int i = 0; i < WORK_MAT*WORK_MAT; i++){
            M[i] = 1;
        }

        printf("N Threads = %d\n", nprocs);
        printf("N Trials = %d\n", N_TRIALS);
        printf("Work Mat = %lld\n", WORK_MAT);
        printf("Work Vec = %lld\n", WORK_VEC);
        printf("Starting Tests\n");
    }

    double time = MPI_Wtime();
    float res;
    for(int i =0; i < N_TRIALS; i++){
        free(mdot_mpi(M, B, WORK_VEC));
        MPI_Barrier(MPI_COMM_WORLD);
    }
    time  = (MPI_Wtime() - time)/N_TRIALS;
    if(procid == 0){
        printf("MPI Time = %0.4f\n", time);
        printf("MPI RES = %0.4f\n", res);
    }

    // Free Memory
    free(M);
    free(A);
    free(B);
    MPI_Finalize();
    int flag;
FINALIZE:
    MPI_Finalized(&flag);
    if(flag == 0) goto FINALIZE;
    return 0;
}


