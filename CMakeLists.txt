cmake_minimum_required(VERSION 3.10)
project(memory_accessors_tensor LANGUAGES CXX)

option(USE_MKL "Enable optional MKL sparse backend hooks when available" OFF)

add_executable(ttm_bench
    src/main.cpp
)

target_include_directories(ttm_bench PRIVATE src)

if (USE_MKL)
    find_path(MKL_INCLUDE_DIR mkl.h)
    if (MKL_INCLUDE_DIR)
        target_include_directories(ttm_bench PRIVATE ${MKL_INCLUDE_DIR})
        target_compile_definitions(ttm_bench PRIVATE MEMORY_ACCESSORS_MKL_HEADERS_AVAILABLE=1)
        message(STATUS "MKL headers found: ${MKL_INCLUDE_DIR}")
    else()
        message(WARNING "USE_MKL=ON, but mkl.h was not found. Building with custom backend only.")
    endif()
    target_compile_definitions(ttm_bench PRIVATE MEMORY_ACCESSORS_USE_MKL_REQUESTED=1)
endif()

set_target_properties(ttm_bench PROPERTIES
    CXX_STANDARD 11
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

find_package(Threads REQUIRED)
target_link_libraries(ttm_bench PRIVATE Threads::Threads)

if (MSVC)
    target_compile_options(ttm_bench PRIVATE /O2 /W4)
else()
    target_compile_options(ttm_bench PRIVATE -std=c++11 -O3 -march=native -Wall -Wextra -Wpedantic)
endif()
