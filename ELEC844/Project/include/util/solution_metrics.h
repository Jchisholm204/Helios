/**
 * @file solution_metrics.h
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-12-12
 * @modified Last Modified: 2025-12-12
 *
 * @copyright Copyright (c) 2025
 */

#ifndef _SOLUION_METRICS_H_
#define _SOLUION_METRICS_H_
#include <stddef.h>
#include "statespace/statespace.h"

struct solution_metrics {
    double time;
    double length;
    double optimal_length;
    double quality;
    size_t n_collision_checks;
    state_t *path;
    size_t n_path;
};

#endif
