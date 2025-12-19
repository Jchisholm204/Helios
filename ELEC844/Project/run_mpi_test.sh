#!/usr/bin/env bash

NP=(1 2 4 8 16 32 64)

module load mpi


for P in "${NP[@]}"; do
    mpirun -n $P ./build/844A1
done
