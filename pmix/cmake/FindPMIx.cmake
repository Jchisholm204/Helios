# FindUCX.cmake

find_package(pmix)

if(NOT pmix_FOUND)
    message(WARNING "Failed to find PMIx")
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(PMIX REQUIRED IMPORTED_TARGET pmix)
    add_library(${CMAKE_PROJECT_NAME}::pmix INTERFACE IMPORTED GLOBAL)
    target_link_libraries(${CMAKE_PROJECT_NAME}::pmix INTERFACE 
        PkgConfig::PMIX
    )
    message(STATUS "Found PMIx though PkgConfig")
else()
    message(STATUS "Found PMIx though CMake")
    add_library(${CMAKE_PROJECT_NAME}::pmix INTERFACE IMPORTED GLOBAL)
    target_link_libraries(${CMAKE_PROJECT_NAME}::pmix INTERFACE 
        pmix::pmix
    )

    set(UCX_FOUND TRUE CACHE INTERNAL TRUE)
endif()

