/**
 * @file softmax.hpp
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief Simple Softmax Function
 * @version 0.1
 * @date Created: 2025-11-18
 * @modified Last Modified: 2025-11-18
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _SOFTMAX_HPP_
#define _SOFTMAX_HPP_
#include <vector>
#include <cmath>

static inline std::vector<double> softmax(const std::vector<double>& input) {
    std::vector<double> out(input.size());
    double sum_exp = 0.0;
    for(size_t i = 0; i < input.size(); i++){
        out[i] = std::exp(input[i]);
        sum_exp += out[i];
    }
    for(size_t i = 0; i < input.size(); i++){
        out[i] = out[i]/sum_exp;
    }
    return out;
}

#endif
