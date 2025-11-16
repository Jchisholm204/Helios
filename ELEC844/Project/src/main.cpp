/**
 * @file main.c
 * @author Jacob Chisholm (https://Jchisholm204.github.io)
 * @brief ELEC 844 Final Project
 * @version 0.1
 * @date Created: 2025-11-15
 * @modified Last Modified: 2025-11-15
 *
 * @copyright Copyright (c) 2025
 */


#include "main.h"

#include <stdio.h>
#include <iostream>

int main(int argc, char** argv) {
    printf("ELEC 844 Final Project\nargs:\n");
    for(int i = 0; i < argc; i++){
        printf(" %d) %s\n", i, argv[i]);
    }
    std::cout << "Running OMPL Test" << std::endl;
    return ompl_test(argc, argv);
}
