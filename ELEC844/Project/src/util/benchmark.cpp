/**
 * @file benchmark.cpp
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-11-18
 * @modified Last Modified: 2025-11-18
 *
 * @copyright Copyright (c) 2025
 */

#include "util/benchmark.hpp"

#include <iostream>
#include <sstream>

//
// ---------------- WallTimer ----------------
//

WallTimer::WallTimer(std::string name) : name(std::move(name)) {
    this->iteration = 0;
}

void WallTimer::start(void) {
    running = true;
    t_start = clock::now();
}

void WallTimer::stop(void) {
    if (running) {
        t_end = clock::now();
        running = false;
        this->history.push_back({iteration, this->get_elapsed()});
    }
}

void WallTimer::reset(void) {
    running = false;
    t_start = t_end;
}

double WallTimer::get_elapsed(void) const {
    return std::chrono::duration<double>(t_end - t_start).count()*1000;
}

//
// ---------------- Benchmark ----------------
//

Benchmark::Benchmark(std::string fname,
                     std::vector<std::pair<std::string, std::string>> params) {
    this->params = params;
    this->params.push_back({"name", fname});
    this->fname = fname;
}

void Benchmark::start_benchmark() {
    for (auto timer : timers) {
        timer->reset();
        timer->iteration = iteration_count;
    }
}

void Benchmark::stop_benchmark() {
    for (auto timer : timers) {
        timer->stop();
    }
    iteration_count++;
}

std::shared_ptr<WallTimer> Benchmark::new_timer(std::string name) {
    std::shared_ptr<WallTimer> new_t = std::make_shared<WallTimer>(name);
    this->timers.push_back(new_t);
    return new_t;
}

void Benchmark::export_json(void) const {
    json root;

    for (const auto & param : params){
        root[param.first] = param.second;
    }

    for (const auto& timer : timers) {
        json& branch = root[timer->name];
        for (const auto& record : timer->history) {
            branch[record.first] = record.second;
        }
    }

    std::ofstream f(fname);
    std::cout << root.dump(4);
}

std::ostringstream Benchmark::export_csv() const {
    std::ostringstream out;

    for (const auto& param : params) {
        out << "#" << param.first << ": " << param.second << std::endl;
    }

    out << "#" << std::endl;

    // optional: CSV header
    out << "timer_name,key,value\n";

    for (const auto& timer : timers) {
        for (const auto& record : timer->history) {
            // write CSV row
            out << timer->name << "," << record.first << "," << record.second
                << "\n";
        }
    }
    return out;
}

std::vector<std::pair<std::string, double>> Benchmark::export_mean(void){
    std::vector<std::pair<std::string, double>> avg;
    for (const auto& timer : timers) {
        double sum = 0;
        for (const auto& record : timer->history) {
            sum += record.second;
        }
        avg.push_back({timer->name, sum/timer->history.size()});
    }
    return avg;
}

