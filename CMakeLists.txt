cmake_minimum_required(VERSION 3.10)
project(memory_accessors_tensor LANGUAGES CXX)

add_executable(ttm_bench
    src/main.cpp
)

target_include_directories(ttm_bench PRIVATE src)

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
