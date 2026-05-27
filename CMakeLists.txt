cmake_minimum_required(VERSION 3.10)
project(memory_accessors_tensor LANGUAGES CXX)

option(USE_MKL "Enable optional MKL sparse backend hooks when available" OFF)
set(MKL_ROOT "" CACHE PATH "Path to Intel oneAPI MKL root")

add_executable(ttm_bench
    src/main.cpp
)

target_include_directories(ttm_bench PRIVATE src)

if (USE_MKL)
    set(_MKL_ROOT_CANDIDATES)
    if (MKL_ROOT)
        list(APPEND _MKL_ROOT_CANDIDATES "${MKL_ROOT}")
    endif()
    if (DEFINED ENV{MKLROOT})
        list(APPEND _MKL_ROOT_CANDIDATES "$ENV{MKLROOT}")
    endif()

    find_path(MKL_INCLUDE_DIR
        NAMES mkl.h
        PATHS ${_MKL_ROOT_CANDIDATES}
        PATH_SUFFIXES include
        NO_DEFAULT_PATH
    )
    find_library(MKL_RT_LIBRARY
        NAMES mkl_rt libmkl_rt.so
        PATHS ${_MKL_ROOT_CANDIDATES}
        PATH_SUFFIXES lib/intel64 lib
        NO_DEFAULT_PATH
    )

    if (NOT MKL_INCLUDE_DIR)
        find_path(MKL_INCLUDE_DIR mkl.h)
    endif()
    if (NOT MKL_RT_LIBRARY)
        find_library(MKL_RT_LIBRARY NAMES mkl_rt libmkl_rt.so)
    endif()

    if (MKL_INCLUDE_DIR AND MKL_RT_LIBRARY)
        target_include_directories(ttm_bench PRIVATE ${MKL_INCLUDE_DIR})
        target_compile_definitions(ttm_bench PRIVATE USE_MKL_BACKEND=1)
        target_link_libraries(ttm_bench PRIVATE ${MKL_RT_LIBRARY})
        if (UNIX)
            target_link_libraries(ttm_bench PRIVATE pthread dl m)
        endif()
        message(STATUS "MKL backend enabled")
        message(STATUS "MKL include: ${MKL_INCLUDE_DIR}")
        message(STATUS "MKL runtime: ${MKL_RT_LIBRARY}")
    else()
        message(WARNING "USE_MKL=ON, but MKL headers/runtime were not found. Building with custom backend only.")
    endif()
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
