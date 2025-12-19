#!/usr/bin/env bash

NP=(1 2 4 8 16 32 64)


for P in "${NP[@]}"; do
    mpirun -n $P ./build/844A1
done
