# Server Project Makefile
set(TARG server_project)

file(GLOB_RECURSE SERVER_SOURCES FOLLOW_SYMLINKS 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_executable(${TARG} 
    ${SERVER_SOURCES}
)

target_include_directories(${TARG} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_options(${TARG} PRIVATE ${GENERIC_COMPILE_OPTIONS})


target_link_libraries(${TARG} PRIVATE
    ${COMMON_LIBRARIES}
    ${CMAKE_PROJECT_NAME}::ucx
    ${CMAKE_PROJECT_NAME}::pmix
    # Link to the common project
    common_project
)
