cmake_minimum_required(VERSION 3.16)

project(memory_accessors_tensor_profile LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

if(NOT MKLROOT AND DEFINED ENV{MKLROOT})
  set(MKLROOT "$ENV{MKLROOT}" CACHE PATH "Intel MKL root")
endif()

if(NOT MKLROOT)
  message(FATAL_ERROR "MKLROOT is required. Configure with -DMKLROOT=$MKLROOT or export MKLROOT.")
endif()

find_path(MKL_INCLUDE_DIR
  NAMES mkl.h
  PATHS "${MKLROOT}/include"
  NO_DEFAULT_PATH)

find_library(MKL_INTEL_LP64
  NAMES mkl_intel_lp64
  PATHS "${MKLROOT}/lib/intel64" "${MKLROOT}/lib"
  NO_DEFAULT_PATH)
find_library(MKL_INTEL_THREAD
  NAMES mkl_intel_thread
  PATHS "${MKLROOT}/lib/intel64" "${MKLROOT}/lib"
  NO_DEFAULT_PATH)
find_library(MKL_CORE
  NAMES mkl_core
  PATHS "${MKLROOT}/lib/intel64" "${MKLROOT}/lib"
  NO_DEFAULT_PATH)
find_library(MKL_IOMP5
  NAMES iomp5 libiomp5
  PATHS "${MKLROOT}/../compiler/latest/lib/intel64_lin"
        "${MKLROOT}/../../compiler/latest/lib/intel64_lin"
        "${MKLROOT}/lib/intel64" "${MKLROOT}/lib"
  NO_DEFAULT_PATH)

if(NOT MKL_INCLUDE_DIR OR NOT MKL_INTEL_LP64 OR NOT MKL_INTEL_THREAD OR NOT MKL_CORE OR NOT MKL_IOMP5)
  message(FATAL_ERROR "Could not find required Intel MKL headers/libraries under MKLROOT=${MKLROOT}")
endif()

add_executable(ttm_profile src/main.cpp)
target_include_directories(ttm_profile PRIVATE "${MKL_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(ttm_profile PRIVATE
  "${MKL_INTEL_LP64}"
  "${MKL_INTEL_THREAD}"
  "${MKL_CORE}"
  "${MKL_IOMP5}"
  pthread
  m
  dl)
target_compile_options(ttm_profile PRIVATE -Wall -Wextra -Wpedantic)
