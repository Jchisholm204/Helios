#include <iostream>
#include <math.h>
#include <time.h>
#include <chrono>

void add(int n, float *x, float *y){
    for(int i = 0; i < n; i++){
        y[i] = x[i] + y[i];
    }
}

int main(void){
    int N = 1 << 20;

    float *x = new float[N];
    float *y = new float[N];


    for(int i = 0; i < N; i++){
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    auto t_start = std::chrono::high_resolution_clock::now();

    // Add elements on the CPU
    add(N, x, y);

    auto t_end = std::chrono::high_resolution_clock::now();

    float maxErr = 0.0f;
    for(int i = 0; i < N; i++)
        maxErr = fmax(maxErr, fabs(y[i]-3.0f));
    std::cout << "Max Error: " << maxErr << std::endl;

    std::cout << "Time: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t_end-t_start).count()/1.0e6f << " ms" << std::endl;

    delete [] x;
    delete [] y;

    return 0;
}
