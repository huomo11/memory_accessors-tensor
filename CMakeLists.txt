cmake_minimum_required(VERSION 3.16)

project(memory_accessors_tensor_profile LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

if(NOT DEFINED MKLROOT OR "${MKLROOT}" STREQUAL "")
  if(DEFINED ENV{MKLROOT} AND NOT "$ENV{MKLROOT}" STREQUAL "")
    set(MKLROOT "$ENV{MKLROOT}" CACHE PATH "Intel MKL root")
  endif()
endif()

if(NOT DEFINED MKLROOT OR "${MKLROOT}" STREQUAL "")
  message(FATAL_ERROR "MKLROOT is required. Configure with -DMKLROOT=/opt/intel/oneapi/mkl/latest or export MKLROOT.")
endif()

set(MKL_INCLUDE_DIR "${MKLROOT}/include")
set(MKL_MKL_H "${MKL_INCLUDE_DIR}/mkl.h")
set(MKL_SPBLAS_H "${MKL_INCLUDE_DIR}/mkl_spblas.h")

if(NOT EXISTS "${MKL_MKL_H}")
  message(FATAL_ERROR "MKL header missing: ${MKL_MKL_H}")
endif()

if(NOT EXISTS "${MKL_SPBLAS_H}")
  message(FATAL_ERROR "MKL header missing: ${MKL_SPBLAS_H}")
endif()

find_library(MKL_LIBRARY
  NAMES mkl_rt libmkl_rt.so
  PATHS "${MKLROOT}/lib/intel64" "${MKLROOT}/lib"
  NO_DEFAULT_PATH)

if(NOT MKL_LIBRARY)
  message(FATAL_ERROR
    "MKL runtime library missing. Expected libmkl_rt.so in "
    "${MKLROOT}/lib/intel64 or ${MKLROOT}/lib")
endif()

get_filename_component(MKL_LIBRARY_DIR "${MKL_LIBRARY}" DIRECTORY)

message(STATUS "MKLROOT = ${MKLROOT}")
message(STATUS "MKL_INCLUDE_DIR = ${MKL_INCLUDE_DIR}")
message(STATUS "MKL_LIBRARY = ${MKL_LIBRARY}")

add_executable(ttm_profile src/main.cpp)
target_include_directories(ttm_profile PRIVATE "${MKL_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/src")
target_link_libraries(ttm_profile PRIVATE
  "${MKL_LIBRARY}"
  pthread
  dl
  m)
set_target_properties(ttm_profile PROPERTIES
  BUILD_RPATH "${MKL_LIBRARY_DIR}")
target_compile_options(ttm_profile PRIVATE -Wall -Wextra -Wpedantic)
