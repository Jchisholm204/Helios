/**
 * @file evaluate.cpp
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief
 * @version 0.1
 * @date Created: 2025-12-13
 * @modified Last Modified: 2025-12-13
 *
 * @copyright Copyright (c) 2025
 */

#include "ompl/bit.hpp"
#include "ompl/fmt.hpp"
#include "ompl/ompl.hpp"

#include <stdio.h>

#define N_TESTS 100
#define STRLN 200

void ompl_evaluate(double solve_time) {
    char fname[STRLN];
    int sec = (int) solve_time;
    snprintf(fname, STRLN, "./tests/bit_%dD_%dS.csv", STATESPACE_DIMS, sec);
    FILE *fp = fopen(fname, "w");
    if (!fp) {
        fprintf(stderr, "Failed to open output file..\n");
        return;
    }
    fprintf(fp, "time (first),length (first),optimal length "
                "(first),quality(first),collision checks (first),");
    fprintf(fp, "time (best),length (best),optimal length "
                "(best),quality(best),collision checks (best),");
    fprintf(fp, "time (final),length (final),optimal length "
                "(final),quality(final),collision checks (final)\n");

    for (size_t test_i = 0; test_i < N_TESTS; test_i++) {
        printf("Running Test %ld..\n", test_i);
        // struct ompl_planner *planner = ompl_init_fmt(5000);
        struct ompl_planner *planner = ompl_init_bit();
        ompl_solve(planner, solve_time);
        // Grab all of the metrics from the run
        struct ompl_metrics *metrics = &planner->metrics;
        struct solution_metrics *m_first = &metrics->first;
        struct solution_metrics *m_best = &metrics->best;
        struct solution_metrics *m_final = &metrics->final;
        // Print out the results of this run
        printf("Solved Test %ld in %2.4f ms:\n", test_i, m_final->time);
        printf(" First Solution: len=%2.3f quality=%1.4f time=%2.2f\n",
               m_first->length, m_first->quality, m_first->time);
        printf(" Best Solution: len=%2.3f quality=%1.4f time=%2.2f\n",
               m_best->length, m_best->quality, m_best->time);
        printf(" Final Solution: len=%2.3f quality=%1.4f time=%2.2f\n",
               m_final->length, m_final->quality, m_final->time);
        // Print the results of the tests out to the csv file
        // Time, Length, Optimal Len, Quality, Collisions
        fprintf(fp, "%f,%f,%f,%f,%ld,", m_first->time, m_first->length,
                m_first->optimal_length, m_first->quality,
                m_first->n_collision_checks);
        fprintf(fp, "%f,%f,%f,%f,%ld,", m_best->time, m_best->length,
                m_best->optimal_length, m_best->quality,
                m_best->n_collision_checks);
        fprintf(fp, "%f,%f,%f,%f,%ld\n", m_final->time, m_final->length,
                m_final->optimal_length, m_final->quality,
                m_final->n_collision_checks);
        ompl_free(&planner);
    }
    printf("Finished all tests..\n");
    fclose(fp);
    printf("File Closed.\n");
}
