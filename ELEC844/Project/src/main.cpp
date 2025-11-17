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
#include "display/display.hpp"

#include <stdio.h>
#include <iostream>

int main(int argc, char** argv) {
    printf("ELEC 844 Final Project\nargs:\n");
    for(int i = 0; i < argc; i++){
        printf(" %d) %s\n", i, argv[i]);
    }
    Display d(100, 100);
    std::vector<std::pair<float, float>> points = {{5, 5}, {10, 10}, {20, 20}, {0, 0}};
    while(!d.poll_quit()){
        d.clear();
        d.label("ELEC 844 Project - FMT* - Jacob Chisholm");
        d.draw_grid(10, 10);
        d.draw_points(points, {255, 0, 255});
        d.draw_points({{44, 66}, {22, 33}}, {0, 0, 255});
        d.render();
    }

    return 0;
    // std::cout << "Running OMPL Test" << std::endl;
    // return ompl_test(argc, argv);
}
