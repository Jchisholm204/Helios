cmake_minimum_required(VERSION 3.21)

project(cc)
set(PROJECT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

# set(CMAKE_C_COMPILER mpicc ${CMAKE_C_FLAGS} ${CMAKE_LDFLAGS})

set(EXECUTABLE ${CMAKE_PROJECT_NAME})
enable_language(C CXX)
set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

# Add additional include directories for standard libraries
include_directories(SYSTEM ${CMAKE_C_COMPILER_INCLUDE_DIRS})

# Set additional compiler flags (e.g., include paths for standard libraries)
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -isystem ${CMAKE_C_COMPILER_INCLUDE_DIRS}")

# Enable export of compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


set(INCLUDE_DIRECTORIES
    ${PROJECT_DIR}/include
)

set(GENERIC_COMPILE_OPTIONS
    # -Wall
    # -Wextra
    # -Wpedantic
    # -pedantic
    # -fno-common
    )

file(GLOB_RECURSE PROJECT_SOURCES FOLLOW_SYMLINKS
    ${PROJECT_DIR}/src/*.c
    ${PROJECT_DIR}/src/*.cpp)


# Append to this list
include_directories(
    # Needed for linking protobuf-c on alliancecan systems
    ~/.local/include
    # Include Directories for MPI on RHEL Systems
    /usr/include/openmpi-x86_64
)

# Append to this list
link_directories(
    # Needed for linking protobuf-c on alliancecan systems
    ~/.local/lib
    # Library Directories for MPI on RHEL Systems
    /usr/lib/
    /usr/lib64/openmpi/lib
)

add_executable(${EXECUTABLE}
    ${PROJECT_SOURCES}
)

target_include_directories(${EXECUTABLE} PRIVATE
    ${INCLUDE_DIRECTORIES})

target_compile_options(${EXECUTABLE} PRIVATE
    ${CPU_PARAMETERS}
    ${GENERIC_COMPILE_OPTIONS}
    # -Wdouble-promotion
    # -Wformat
    # -Wshadow
    # -Wconversion
    # -fstack-usage
    -g
    # -fmessage-length=0
    # -ffunction-sections
    # -fdata-sections
    # --specs=nano.specs
    #$<$<CONFIG:Debug>:-Os -g3>
    #$<$<CONFIG:Release>:-Os -g0>

    # Maximum Optimization
    # -O3
    -O3
)


target_link_libraries(${EXECUTABLE}
    -lm
    pthread
    # -lc
    # ssh
    # protobuf
    # pthread
    # protobuf-c
    # -lgcc
    # -lnosys
    # -lrdimon
)
