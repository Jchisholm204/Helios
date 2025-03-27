/**
 * @file main.cu
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief 
 * @version 0.1
 * @date 2025-03-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

#include "main.hpp"
#include "devProp.hpp"
#include "kMatMul.hpp"
#include "cpuMatMul.hpp"

#define N_TESTS 1
#define N_MAT 4096
#define N_MAT_MIN 256
// #define N_MAT_MAX 1024
#define N_MAT_MAX 4097
// #define N_MAT_MIN 8
// #define N_MAT_MAX 64
#define BLOCK_WIDTH 32
 
int calc_test(testParams_t *tests, int n_tests, int ident){
    testParams_t average, variation;
    average.mat_n = n_tests;
    average.t_mem_alloc = 0;
    average.t_mem_host_to_device = 0;
    average.t_mem_device_to_host = 0;
    average.t_cpu_compute = 0;
    average.t_gpu_compute = 0;
    average.test_success = 1;
    for(int i = 1; i < n_tests; i++){
        average.t_mem_alloc += tests[i].t_mem_alloc;
        average.t_mem_host_to_device += tests[i].t_mem_host_to_device;
        average.t_mem_device_to_host += tests[i].t_mem_device_to_host;
        average.t_cpu_compute += tests[i].t_cpu_compute;
        average.t_gpu_compute += tests[i].t_gpu_compute;
        average.test_success &= tests[i].test_success;
    }
    average.t_mem_alloc /= n_tests;
    average.t_mem_host_to_device /= n_tests;
    average.t_mem_device_to_host /= n_tests;
    average.t_cpu_compute /= n_tests;
    average.t_gpu_compute /= n_tests;
    printf("Average:\n");
    print_test(average, ident);

    // Calculate Variation
    variation.mat_n = n_tests;
    variation.t_mem_alloc = 0;
    variation.t_mem_host_to_device = 0;
    variation.t_mem_device_to_host = 0;
    variation.t_cpu_compute = 0;
    variation.t_gpu_compute = 0;
    variation.test_success = 1;
    for(int i = 0; i < n_tests; i++){
        variation.t_mem_alloc += pow(tests[i].t_mem_alloc - average.t_mem_alloc, 2);
        variation.t_mem_host_to_device += pow(tests[i].t_mem_host_to_device - average.t_mem_host_to_device, 2);
        variation.t_mem_device_to_host += pow(tests[i].t_mem_device_to_host - average.t_mem_device_to_host, 2);
        variation.t_cpu_compute += pow(tests[i].t_cpu_compute - average.t_cpu_compute, 2);
        variation.t_gpu_compute += pow(tests[i].t_gpu_compute - average.t_gpu_compute, 2);
    }

    variation.t_mem_alloc /= n_tests;
    variation.t_mem_host_to_device /= n_tests;
    variation.t_mem_device_to_host /= n_tests;
    variation.t_cpu_compute /= n_tests;
    variation.t_gpu_compute /= n_tests;

    variation.t_mem_alloc = sqrt(variation.t_mem_alloc);
    variation.t_mem_host_to_device = sqrt(variation.t_mem_host_to_device);
    variation.t_mem_device_to_host = sqrt(variation.t_mem_device_to_host);
    variation.t_cpu_compute = sqrt(variation.t_cpu_compute);
    variation.t_gpu_compute = sqrt(variation.t_gpu_compute);
    variation.test_success = average.test_success;
    printf("Variance:\n");
    print_test(variation, ident);
    return 0;
}
 
int run_tests(size_t n_mat, size_t n_tests){
    printf("Running %d tests with matrix dim %d\n", n_tests, n_mat);
    testParams_t *tests = (testParams_t*)malloc(sizeof(testParams_t)*n_tests);
    for(int i = 0; i < n_tests; i++){
        tests[i].kernel = eKernel_ssm;
        tests[i].dim_block = 1;
        tests[i].dim_grid = 1;
        tests[i].mat_n = n_mat;
        run_test_single(tests[i]);
        // print_test(tests[i], i);
    }
    calc_test(tests, n_tests, n_mat);
    FREE(tests);
    return 0;
}
 
int run_tests_mt(size_t n_mat, size_t n_tests, int block_width){
    printf("Running %d tests with matrix dim %d block width: %d\n", n_tests, n_mat, block_width);
    testParams_t *tests = (testParams_t*)malloc(sizeof(testParams_t)*n_tests);
    for(int i = 0; i < n_tests; i++){
        tests[i].kernel = eKernel_tiled;
        tests[i].dim_block = block_width;
        tests[i].dim_grid = (n_mat)/block_width;
        tests[i].mat_n = n_mat;
        run_test_single(tests[i]);
        // print_test(tests[i], i);
    }
    calc_test(tests, n_tests, n_mat);
    FREE(tests);
    
    return 0;
}
 
void run_test(void){
    testParams_t tests[N_TESTS];
    printf("Running %d tests with matrix dim %d\n", N_TESTS, N_MAT);
    for(int i = 0; i < N_TESTS; i++){
        tests[i].kernel = eKernel_tiled;
        tests[i].dim_block = BLOCK_WIDTH;
        tests[i].dim_grid = (N_MAT+BLOCK_WIDTH-1)/BLOCK_WIDTH;
        tests[i].mat_n = N_MAT;
        run_test_single(tests[i]);
        print_test(tests[i], i);
    }
    testParams_t average;
    average.mat_n = N_TESTS;
    average.t_mem_alloc = 0;
    average.t_mem_host_to_device = 0;
    average.t_mem_device_to_host = 0;
    average.t_cpu_compute = 0;
    average.t_gpu_compute = 0;
    average.test_success = 1;
    for(int i = 0; i < N_TESTS; i++){
        average.t_mem_alloc += tests[i].t_mem_alloc;
        average.t_mem_host_to_device += tests[i].t_mem_host_to_device;
        average.t_mem_device_to_host += tests[i].t_mem_device_to_host;
        average.t_cpu_compute += tests[i].t_cpu_compute;
        average.t_gpu_compute += tests[i].t_gpu_compute;
        average.test_success &= tests[i].test_success;
    }
    average.t_mem_alloc /= N_TESTS;
    average.t_mem_host_to_device /= N_TESTS;
    average.t_mem_device_to_host /= N_TESTS;
    average.t_cpu_compute /= N_TESTS;
    average.t_gpu_compute /= N_TESTS;
    print_test(average);

}
 
int main(int argc, char** argv) {
    getDevProperties();
    run_test();
    return 0;
    // for(int i = 256; i < 1025; i = i*2){
    for(int i = N_MAT_MIN; i < (N_MAT_MAX+1); i = i*2){
        run_tests_mt(i, N_TESTS, BLOCK_WIDTH);
        printf("============= TEST %d COMPLETED =============\n", i);
    }
    printf("All Tests Complete\n");
    return 0;
}
