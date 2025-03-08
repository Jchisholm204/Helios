/**
 * @file test.cu
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief Test Script to Benchmark and Verify GPU Mat Mul Kernel
 * @version 0.1
 * @date 2025-03-07
 * 
 * @copyright Copyright (c) 2025
 * 
 */

#include <chrono>
#include "main.hpp"
#include "cpuMatMul.hpp"
#include "kMatMul.hpp"

void run_test_single(testParams_t &params){
    size_t n_mat = params.mat_n;
    // printf("Running test with matrix size:: %ld\n", n_mat);

    // Allocate the Host Side Memory
    mat_t host_N = (mat_t)malloc(MAT_SIZE(n_mat));
    mat_t host_M = (mat_t)malloc(MAT_SIZE(n_mat));
    // Result from GPU
    mat_t host_P_gpu = (mat_t)malloc(MAT_SIZE(n_mat));
    // Result from CPU
    mat_t host_P_cpu = (mat_t)malloc(MAT_SIZE(n_mat));
    if(!host_M || !host_N || !host_P_cpu || !host_P_gpu){
        fprintf(stderr, "Failed to Allocate one or more host side Array - Test Aborted");
        FREE(host_N);
        FREE(host_M);
        FREE(host_P_cpu);
        FREE(host_P_gpu);
        params.test_success = false;
        return;
    }
    // Fill the host arrays with data (Uses cRand)
    MAT_fillRand(host_N, n_mat, 111, 9.96);
    MAT_fillRand(host_M, n_mat, 142, 12.23);
    // Zero out the host side results
    MAT_fill(host_P_cpu, n_mat, 0);
    MAT_fill(host_P_gpu, n_mat, 0);

    // Create Events for benchmarking transfers
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaDeviceSynchronize();

    // Allocate the Device side Memory
    mat_t dev_N, dev_M, dev_P;
    // Start recording the memory allocation time
    cudaEventRecord(start, 0);
    cudaMalloc(&dev_N, MAT_SIZE(n_mat));
    cudaMalloc(&dev_M, MAT_SIZE(n_mat));
    cudaMalloc(&dev_P, MAT_SIZE(n_mat));
    // Benchmark Allocation time
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&params.t_mem_alloc, start, stop);
    // Restart the CUDA Events
    cudaEventDestroy(start);
    cudaEventCreate(&start);
    cudaEventDestroy(stop);
    cudaEventCreate(&stop);

    // Check that memory was allocated
    if(!dev_N || !dev_M || !dev_P){
        fprintf(stderr, "Failed to allocate Device Side Memory - Test Aborted");
        if(!dev_N)
            cudaFree(dev_N);
        if(!dev_M)
            cudaFree(dev_M);
        if(!dev_P)
            cudaFree(dev_P);
        FREE(host_N);
        FREE(host_M);
        FREE(host_P_cpu);
        FREE(host_P_gpu);
        params.test_success = false;
        return;
    }

    // Start Memory Event
    cudaEventRecord(start, 0);
    // Transfer Memory from Host to Device
    cudaMemcpyAsync(dev_N, host_N, MAT_SIZE(n_mat), cudaMemcpyHostToDevice, 0);
    cudaMemcpyAsync(dev_M, host_M, MAT_SIZE(n_mat), cudaMemcpyHostToDevice, 0);
    // Benchmark the Memory
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&params.t_mem_host_to_device, start, stop);
    // Restart the CUDA Events
    cudaEventDestroy(start);
    cudaEventCreate(&start);
    cudaEventDestroy(stop);
    cudaEventCreate(&stop);

    // Start Kernel Event
    cudaEventRecord(start, 0);
    // Launch the Kernel
    switch(params.kernel){
        default:
        case eKernel_ssm:
            kMatMul_ssm<<<params.dim_grid, params.dim_block>>>(dev_P, dev_M, dev_N, n_mat);
            break;
        case eKernel_msm:
            kMatMul_msm<<<params.dim_grid, params.dim_block>>>(dev_P, dev_M, dev_N, n_mat);
            break;
        case eKernel_tiled:
            kMatMul_tiled<<<params.dim_grid, params.dim_block>>>(dev_P, dev_M, dev_N, n_mat);
            break;
    };
    // Benchmark the Memory
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&params.t_gpu_compute, start, stop);
    // Restart the CUDA Events
    cudaEventDestroy(start);
    cudaEventCreate(&start);
    cudaEventDestroy(stop);
    cudaEventCreate(&stop);

    // Start Memory Event
    cudaEventRecord(start, 0);
    cudaMemcpyAsync(host_P_gpu, dev_P, MAT_SIZE(n_mat), cudaMemcpyDeviceToHost, 0);
    // Benchmark the Memory
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&params.t_mem_device_to_host, start, stop);
    // Restart the CUDA Events
    cudaEventDestroy(start);
    cudaEventCreate(&start);
    cudaEventDestroy(stop);
    cudaEventCreate(&stop);

    // Start the CPU Matrix Multiplication
    auto cpu_start = std::chrono::high_resolution_clock::now();
    cpuMatMul(host_P_cpu, host_M, host_N, n_mat);
    auto cpu_end = std::chrono::high_resolution_clock::now();
    auto cpu_time = std::chrono::duration_cast<std::chrono::milliseconds>(cpu_end - cpu_start);
    params.t_cpu_compute = cpu_time.count();

    // Verify the result was produced correctly
    params.test_success = MAT_compare(host_P_gpu, host_P_cpu, n_mat, 2) == 0;


    // Optionally, debug print the results
    // MAT_print(host_P_cpu, n_mat);
    // MAT_print(host_P_gpu, n_mat);

    // Free Memory
    FREE(host_N);
    FREE(host_M);
    FREE(host_P_cpu);
    FREE(host_P_gpu);
    cudaFree(dev_N);
    cudaFree(dev_M);
    cudaFree(dev_P);
    return;
}


void print_test(testParams_t &result, int tn){
    if(tn == -1) tn = result.mat_n;
    printf("======= Results for Test %d =======\n", tn);
    printf("Time to Allocate Device Memory = %0.2f\n", result.t_mem_alloc);
    printf("Host to Device Transfer Time = %0.2f\n", result.t_mem_host_to_device);
    printf("GPU Compute Time = %0.2f\n", result.t_gpu_compute);
    printf("CPU Compute Time = %0.2f\n", result.t_cpu_compute);
    printf("Device To Host Transfer Time = %0.2f\n", result.t_mem_device_to_host);
    if(result.test_success)
        printf("TEST PASSED\n");
    else
        printf("TEST FAILURE\n");
    printf("=====================================\n");
}