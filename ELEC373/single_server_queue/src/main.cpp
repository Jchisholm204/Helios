#include <matplot/matplot.h>
#include <iostream>
#include <random>
#include <queue>
#include <vector>

static constexpr float service_rate = 0.75;
static constexpr int n_trials = 9;
static constexpr float arrival_rate[n_trials] = {0.2, 0.4, 0.5, 0.6, 0.65, 0.7, 0.72, 0.74, 0.745};
static constexpr int iterations = 10e6;

void run_sim(std::vector<float> &avg_len, std::vector<float> &avg_wait){
    // Setup random generation
    std::random_device rd;
    std::mt19937 gen(rd());

    // std::vector<float> avg_wait, avg_len;

    for(int i = 0; i < n_trials; i++){

        // Setup recorded values
        double trial_avg_wait = 0, trial_avg_len;
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
                trial_avg_wait += (double)(t-t_entry);
            }
            queue_length.push_back(queue.size());
            trial_avg_len += (double)queue.size();
        }
        trial_avg_len /= iterations;
        trial_avg_wait /= wait_time.size();
        avg_len.push_back(trial_avg_len);
        avg_wait.push_back(trial_avg_wait);
        printf("Simulation %d: arrival rate = %.2f\n", i, lambda);
        printf("Average Measured Queue Size: %0.2f\n", trial_avg_len);
        printf("Average Measured Queue Wait: %0.2f tu\n", trial_avg_wait);
        printf("Computed Queue Wait:         %0.2f tu\n", trial_avg_len/lambda);
    }
    printf("Finished Simulations\n");

}

void plot_delay_arrival_rate(std::vector<float> avg_wait, std::vector<float> avg_len){
    using namespace matplot;

    // Plot the queue delay against arrival rates
    std::vector<float> arrival_rate_vec(arrival_rate, arrival_rate + n_trials);
    figure();
    plot(arrival_rate_vec, avg_wait, "-o")->line_width(2).color("b");
    xlabel("Arrival Rate (λ)");
    ylabel("Average Queue Wait (W)");
    title("Queue Delay vs. Arrival Rate");
    grid(true);
    show();

}

int main() {
    std::vector<float> avg_wait, avg_len;
    run_sim(avg_len, avg_wait);
    plot_delay_arrival_rate(avg_wait, avg_len);

    return 0;
}

