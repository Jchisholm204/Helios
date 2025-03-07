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

 
int main(int argc, char** argv) {
    getDevProperties();
    return 0;
}
