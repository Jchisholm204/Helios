/**
 * @file main.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief ELEC 844 Final Project
 * @version 0.2
 * @date Created: 2025-11-15
 * @modified Last Modified: 2025-12-09
 *
 * @copyright Copyright (c) 2025
 */

#include "main.h"

#include <mpi.h>
#include <stdio.h>
#include <unistd.h>

int main(int argc, char **argv) {

    MPI_Init(&argc, &argv);
    int procid = -1;
    MPI_Comm_rank(MPI_COMM_WORLD, &procid);

    int pid = getpid();

    printf("%d -> %d\n", procid, pid);

    MPI_Barrier(MPI_COMM_WORLD);

    for (int i = 0; i < 200; i++) {
        printf("Process %d Iter %d\n", procid, i);
        sleep(1);
        MPI_Barrier(MPI_COMM_WORLD);
    }

    MPI_Finalize();
}
