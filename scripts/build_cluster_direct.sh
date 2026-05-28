#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -z "${MKLROOT:-}" ]]; then
  echo "MKLROOT is required, for example: export MKLROOT=/opt/intel/oneapi/mkl/latest" >&2
  exit 1
fi

MKL_INCLUDE_DIR="${MKLROOT}/include"
MKL_LIBRARY=""

if [[ -f "${MKLROOT}/lib/intel64/libmkl_rt.so" ]]; then
  MKL_LIBRARY="${MKLROOT}/lib/intel64/libmkl_rt.so"
elif [[ -f "${MKLROOT}/lib/libmkl_rt.so" ]]; then
  MKL_LIBRARY="${MKLROOT}/lib/libmkl_rt.so"
else
  echo "MKL runtime library missing: expected libmkl_rt.so in ${MKLROOT}/lib/intel64 or ${MKLROOT}/lib" >&2
  exit 1
fi

if [[ ! -f "${MKL_INCLUDE_DIR}/mkl.h" ]]; then
  echo "MKL header missing: ${MKL_INCLUDE_DIR}/mkl.h" >&2
  exit 1
fi

if [[ ! -f "${MKL_INCLUDE_DIR}/mkl_spblas.h" ]]; then
  echo "MKL header missing: ${MKL_INCLUDE_DIR}/mkl_spblas.h" >&2
  exit 1
fi

MKL_LIBRARY_DIR="$(dirname "${MKL_LIBRARY}")"
CXX_COMPILER="${CXX:-icpx}"

mkdir -p build

"${CXX_COMPILER}" -std=c++14 -O2 -g -Wall -Wextra -Wpedantic \
  -Isrc -I"${MKL_INCLUDE_DIR}" \
  src/main.cpp \
  "${MKL_LIBRARY}" \
  -Wl,-rpath,"${MKL_LIBRARY_DIR}" \
  -lpthread -ldl -lm \
  -o build/ttm_profile

echo "Built build/ttm_profile"
