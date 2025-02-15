#include <matplot/matplot.h>
#include <iostream>
#include <random>
#include <queue>
#include <vector>

static constexpr float service_rate = 0.75;
static constexpr int n_trials = 9;
static constexpr float arrival_rate[n_trials] = {0.2, 0.4, 0.5, 0.6, 0.65, 0.7, 0.72, 0.74, 0.745};
static constexpr int iterations = 10e6;

int main() {
    using namespace matplot;
    // Setup random generation
    std::random_device rd;
    std::mt19937 gen(rd());

    for(int i = 0; i < n_trials; i++){

        // Setup recorded values
        double avg_wait = 0, avg_len;
        std::vector<int> wait_time, queue_length;
        std::queue<int>  queue;

        // Setup random generation
        std::geometric_distribution<int> service(service_rate);
        float lambda = arrival_rate[i];
        std::geometric_distribution<int> arrival(lambda);

        // Run the simulation
        for(int t = 0; t < iterations; t++){
            // Handle packets being added to the queue
            if(arrival(gen) == 0){
                queue.push(t);
            }

            // Handle Departure
            if(!queue.empty() && service(gen) == 0){
                int t_entry = queue.front();
                queue.pop();
                // Push the wait time onto the recorded wait times
                wait_time.push_back(t - t_entry);
                // Contribute to the average wait time
                avg_wait += (double)(t-t_entry);
            }
            queue_length.push_back(queue.size());
            avg_len += (double)queue.size();
        }
        avg_len /= iterations;
        avg_wait /= wait_time.size();
        printf("Simulation %d: arrival rate = %.2f\n", i, lambda);
        printf("Average Measured Queue Size: %0.2f\n", avg_len);
        printf("Average Measured Queue Wait: %0.2f tu\n", avg_wait);
        printf("Computed Queue Wait:         %0.2f tu\n", avg_len/lambda);
    }
    return 0;
}

