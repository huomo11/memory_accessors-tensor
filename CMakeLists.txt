cmake_minimum_required(VERSION 3.15)

project(sparse_dense_ttm_bench LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(ttm_bench
    src/main.cpp
)

if(MSVC)
    target_compile_options(ttm_bench PRIVATE /O2 /permissive-)
else()
    target_compile_options(ttm_bench PRIVATE -O3 -march=native -Wall -Wextra -Wpedantic)
endif()
