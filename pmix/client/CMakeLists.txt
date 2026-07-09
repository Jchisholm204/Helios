# Client Library Project Makefile
set(CLIENT_TARG client_project)

file(GLOB_RECURSE CLIENT_SOURCES FOLLOW_SYMLINKS 
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp
)

add_executable(${CLIENT_TARG}
    ${CLIENT_SOURCES}
)

target_include_directories(${CLIENT_TARG} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)

target_compile_options(${CLIENT_TARG} PRIVATE ${GENERIC_COMPILE_OPTIONS})

target_link_libraries(${CLIENT_TARG} PRIVATE 
    common_project
    ${CMAKE_PROJECT_NAME}::ucx
    ${COMMON_LIBRARIES}
)
