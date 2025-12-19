#!/usr/bin/env bash

# TEST_DIMS=(2 4 6 8 10)
# NP=(1 2 4 8 16 32)
TEST_DIMS=(2 4)
NP=(1 2)

module load mpi

for D in "${TEST_DIMS[@]}"; do
    echo "Compiling For D=${D}"
    if [ -d "./build" ]; then
        rm -r "./build"
    fi
	cmake \
		-DCMAKE_C_COMPILER=gcc\
		-DCMAKE_CXX_COMPILER=g++\
		-B./build \
		-DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
		-DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
		-G "Unix Makefiles" \
        -DSTATESPACE_DIMS=$D \

    make build

    for P in "${NP[@]}"; do
        mpirun -n $P ./build/844A1

    done
done
