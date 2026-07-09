cmake_minimum_required(VERSION 3.21)

project(standard_project_template VERSION 0.1)

# Language Options
enable_language(C)
set(CMAKE_C_STANDARD 11)
# Ensure ISO C
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

# Add additional include directories for standard libraries
include_directories(SYSTEM ${CMAKE_C_COMPILER_INCLUDE_DIRS})

# Set additional compiler flags (e.g., include paths for standard libraries)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -isystem ${CMAKE_C_COMPILER_INCLUDE_DIRS}")

# Enable export of compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(GENERIC_COMPILE_OPTIONS
    -Wall
    -Wextra
    -Wpedantic
    -pedantic
    # Include Debug Symbols
    -g
    # Color Logs
    -DLOG_USE_COLOR
    # Optimization
    -O2
)

# Libraries
include(${CMAKE_SOURCE_DIR}/cmake/FindUCX.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/FindMPI.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/FindThreads.cmake)
include(${CMAKE_SOURCE_DIR}/cmake/FindPMIx.cmake)


set(COMMON_LIBRARIES
    m
    pthread
)

add_subdirectory(common)
add_subdirectory(server)
add_subdirectory(client)
