#pragma once

#include <stdexcept>
#include <string>
#include <vector>

#include <mkl.h>
#include <mkl_spblas.h>

#include "csr_builder.hpp"
#include "timer.hpp"

struct MklCsrHandle {
  sparse_matrix_t handle = nullptr;

  ~MklCsrHandle() {
    if (handle != nullptr) {
      mkl_sparse_destroy(handle);
    }
  }

  MklCsrHandle() = default;
  MklCsrHandle(const MklCsrHandle&) = delete;
  MklCsrHandle& operator=(const MklCsrHandle&) = delete;
};

struct MklSetupTiming {
  double create_ms = 0.0;
  double optimize_ms = 0.0;
};

inline void check_sparse_status(sparse_status_t status, const char* what) {
  if (status != SPARSE_STATUS_SUCCESS) {
    throw std::runtime_error(std::string(what) + " failed with sparse_status_t=" +
                             std::to_string(static_cast<int>(status)));
  }
}

inline void set_thread_count(int threads) {
  mkl_set_num_threads(threads);
}

inline MklCsrHandle create_mkl_csr(const CSRMatrix& csr, double* create_ms) {
  MklCsrHandle matrix;
  Timer timer;
  check_sparse_status(
      mkl_sparse_d_create_csr(&matrix.handle, SPARSE_INDEX_BASE_ZERO, csr.rows,
                              csr.cols,
                              const_cast<int*>(csr.row_ptr.data()),
                              const_cast<int*>(csr.row_ptr.data() + 1),
                              const_cast<int*>(csr.col_idx.data()),
                              const_cast<double*>(csr.values.data())),
      "mkl_sparse_d_create_csr");
  *create_ms = timer.elapsed_ms();
  return matrix;
}

inline double optimize_mkl_csr(const MklCsrHandle& matrix) {
  Timer timer;
  check_sparse_status(mkl_sparse_optimize(matrix.handle), "mkl_sparse_optimize");
  return timer.elapsed_ms();
}

inline double run_mkl_spmm(const MklCsrHandle& matrix, int rank,
                           const std::vector<double>& factor,
                           std::vector<double>& output) {
  matrix_descr descr;
  descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  descr.mode = SPARSE_FILL_MODE_FULL;
  descr.diag = SPARSE_DIAG_NON_UNIT;

  Timer timer;
  check_sparse_status(
      mkl_sparse_d_mm(SPARSE_OPERATION_NON_TRANSPOSE, 1.0, matrix.handle, descr,
                      SPARSE_LAYOUT_ROW_MAJOR, factor.data(), rank, rank, 0.0,
                      output.data(), rank),
      "mkl_sparse_d_mm");
  return timer.elapsed_ms();
}
