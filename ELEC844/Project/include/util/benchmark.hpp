/**
 * @file benchmark.hpp
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-11-18
 * @modified Last Modified: 2025-11-18
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _BENCHMARK_HPP_
#define _BENCHMARK_HPP_

#include "nlohmann/json.hpp"

#include <chrono>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

using json = nlohmann::json;

class Benchmark;

class WallTimer {
  public:
    WallTimer(std::string name);

    void start(void);
    void stop(void);
    double get_elapsed(void) const;

    const std::string& get_name(void) const { return name; }

    const std::vector<std::pair<size_t, double>>& get_history(void) const { return history; }

  private:
    friend Benchmark;
    void reset(void);
    size_t iteration;
    std::vector<std::pair<size_t, double>> history;
    std::string name;
    using clock = std::chrono::steady_clock;
    clock::time_point t_start;
    clock::time_point t_end;
    bool running = false;
};

class Benchmark {
  public:
    Benchmark(std::string fname, std::vector<std::pair<std::string, std::string>> params = {});

    void start_benchmark();
    void stop_benchmark();

    std::shared_ptr<WallTimer> new_timer(std::string name);

    void export_json(void) const;
    std::ostringstream export_csv() const;

  private:
    std::vector<std::pair<std::string, std::string>> params;
    std::vector<std::shared_ptr<WallTimer>> timers;
    size_t iteration_count = 0;
    std::string fname;
};

#endif
