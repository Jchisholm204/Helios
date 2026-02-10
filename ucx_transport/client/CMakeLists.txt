set(TARG ucx_client)

file(GLOB_RECURSE CLIENT_SOURCES FOLLOW_SYMLINKS 
    ${CMAKE_CURRENT_SOURCE_DIR}/*.c
    ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
)

add_executable(${TARG} ${CLIENT_SOURCES})

target_include_directories(${TARG} PRIVATE ${INCLUDE_DIRECTORIES})

target_compile_options(${TARG} PRIVATE ${GENERIC_COMPILE_OPTIONS})

target_link_libraries(${TARG} PRIVATE ${COMMON_LIBRARIES})
