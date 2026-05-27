#ifndef SPMM_BACKEND_HPP
#define SPMM_BACKEND_HPP

#include "csr_matrix.hpp"
#include "dense_matrix.hpp"

#include <stddef.h>
#include <stdint.h>
#include <iostream>
#include <string>
#include <vector>

#if defined(USE_MKL_BACKEND)
#include <mkl.h>
#endif

enum SpMMBackend {
    BACKEND_CUSTOM = 0,
    BACKEND_MKL = 1
};

inline const char* spmm_backend_name(SpMMBackend backend) {
    if (backend == BACKEND_MKL) {
        return "mkl";
    }
    return "custom";
}

inline SpMMBackend choose_default_backend() {
#if defined(USE_MKL_BACKEND)
    return BACKEND_MKL;
#else
    return BACKEND_CUSTOM;
#endif
}

inline void custom_csr_spmm_fp64(const CsrMatrix& a,
                                 const DenseMatrix<double>& f,
                                 size_t rank,
                                 std::vector<double>* y,
                                 bool accumulate) {
    if (!accumulate) {
        y->assign((size_t)a.nrows * rank, 0.0);
    } else if (y->empty()) {
        y->assign((size_t)a.nrows * rank, 0.0);
    }

    uint32_t row;
    for (row = 0; row < a.nrows; ++row) {
        size_t p;
        for (p = a.row_ptr[row]; p < a.row_ptr[(size_t)row + 1]; ++p) {
            uint32_t col = a.col_idx[p];
            double v = a.values[p];
            size_t r;
            for (r = 0; r < rank; ++r) {
                (*y)[(size_t)row * rank + r] += v * f((size_t)col, r);
            }
        }
    }
}

inline void custom_csr_spmm_fp32(const CsrMatrixF& a,
                                 const DenseMatrix<float>& f,
                                 size_t rank,
                                 std::vector<float>* y,
                                 bool accumulate) {
    if (!accumulate) {
        y->assign((size_t)a.nrows * rank, 0.0f);
    } else if (y->empty()) {
        y->assign((size_t)a.nrows * rank, 0.0f);
    }

    uint32_t row;
    for (row = 0; row < a.nrows; ++row) {
        size_t p;
        for (p = a.row_ptr[row]; p < a.row_ptr[(size_t)row + 1]; ++p) {
            uint32_t col = a.col_idx[p];
            float v = a.values[p];
            size_t r;
            for (r = 0; r < rank; ++r) {
                (*y)[(size_t)row * rank + r] += v * f((size_t)col, r);
            }
        }
    }
}

#if defined(USE_MKL_BACKEND)
inline bool convert_csr_indices_to_int(const CsrMatrix& a,
                                       std::vector<MKL_INT>* rows_start,
                                       std::vector<MKL_INT>* rows_end,
                                       std::vector<MKL_INT>* col_idx) {
    rows_start->assign(a.nrows, 0);
    rows_end->assign(a.nrows, 0);
    col_idx->assign(a.col_idx.size(), 0);

    uint32_t row;
    for (row = 0; row < a.nrows; ++row) {
        (*rows_start)[row] = (MKL_INT)a.row_ptr[row];
        (*rows_end)[row] = (MKL_INT)a.row_ptr[(size_t)row + 1];
    }
    size_t p;
    for (p = 0; p < a.col_idx.size(); ++p) {
        (*col_idx)[p] = (MKL_INT)a.col_idx[p];
    }
    return true;
}

inline bool convert_csr_indices_to_int(const CsrMatrixF& a,
                                       std::vector<MKL_INT>* rows_start,
                                       std::vector<MKL_INT>* rows_end,
                                       std::vector<MKL_INT>* col_idx) {
    rows_start->assign(a.nrows, 0);
    rows_end->assign(a.nrows, 0);
    col_idx->assign(a.col_idx.size(), 0);

    uint32_t row;
    for (row = 0; row < a.nrows; ++row) {
        (*rows_start)[row] = (MKL_INT)a.row_ptr[row];
        (*rows_end)[row] = (MKL_INT)a.row_ptr[(size_t)row + 1];
    }
    size_t p;
    for (p = 0; p < a.col_idx.size(); ++p) {
        (*col_idx)[p] = (MKL_INT)a.col_idx[p];
    }
    return true;
}

inline void mkl_csr_spmm_fp64(const CsrMatrix& a,
                              const DenseMatrix<double>& f,
                              size_t rank,
                              std::vector<double>* y,
                              bool accumulate) {
    if (!accumulate) {
        y->assign((size_t)a.nrows * rank, 0.0);
    } else if (y->empty()) {
        y->assign((size_t)a.nrows * rank, 0.0);
    }

    std::vector<MKL_INT> rows_start;
    std::vector<MKL_INT> rows_end;
    std::vector<MKL_INT> col_idx;
    convert_csr_indices_to_int(a, &rows_start, &rows_end, &col_idx);

    sparse_matrix_t handle = 0;
    sparse_status_t status = mkl_sparse_d_create_csr(
        &handle,
        SPARSE_INDEX_BASE_ZERO,
        (MKL_INT)a.nrows,
        (MKL_INT)a.ncols,
        rows_start.empty() ? (MKL_INT*)0 : &rows_start[0],
        rows_end.empty() ? (MKL_INT*)0 : &rows_end[0],
        col_idx.empty() ? (MKL_INT*)0 : &col_idx[0],
        a.values.empty() ? (double*)0 : (double*)&a.values[0]);

    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "mkl_sparse_d_create_csr failed, falling back to custom backend\n";
        custom_csr_spmm_fp64(a, f, rank, y, accumulate);
        return;
    }

    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr.mode = SPARSE_FILL_MODE_FULL;
    descr.diag = SPARSE_DIAG_NON_UNIT;

    double alpha = 1.0;
    double beta = accumulate ? 1.0 : 0.0;
    status = mkl_sparse_d_mm(
        SPARSE_OPERATION_NON_TRANSPOSE,
        alpha,
        handle,
        descr,
        SPARSE_LAYOUT_ROW_MAJOR,
        f.data(),
        (MKL_INT)rank,
        (MKL_INT)rank,
        beta,
        y->empty() ? (double*)0 : &((*y)[0]),
        (MKL_INT)rank);

    sparse_status_t destroy_status = mkl_sparse_destroy(handle);
    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "mkl_sparse_d_mm failed, falling back to custom backend\n";
        custom_csr_spmm_fp64(a, f, rank, y, accumulate);
    }
    if (destroy_status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "mkl_sparse_destroy failed\n";
    }
}

inline void mkl_csr_spmm_fp32(const CsrMatrixF& a,
                              const DenseMatrix<float>& f,
                              size_t rank,
                              std::vector<float>* y,
                              bool accumulate) {
    if (!accumulate) {
        y->assign((size_t)a.nrows * rank, 0.0f);
    } else if (y->empty()) {
        y->assign((size_t)a.nrows * rank, 0.0f);
    }

    std::vector<MKL_INT> rows_start;
    std::vector<MKL_INT> rows_end;
    std::vector<MKL_INT> col_idx;
    convert_csr_indices_to_int(a, &rows_start, &rows_end, &col_idx);

    sparse_matrix_t handle = 0;
    sparse_status_t status = mkl_sparse_s_create_csr(
        &handle,
        SPARSE_INDEX_BASE_ZERO,
        (MKL_INT)a.nrows,
        (MKL_INT)a.ncols,
        rows_start.empty() ? (MKL_INT*)0 : &rows_start[0],
        rows_end.empty() ? (MKL_INT*)0 : &rows_end[0],
        col_idx.empty() ? (MKL_INT*)0 : &col_idx[0],
        a.values.empty() ? (float*)0 : (float*)&a.values[0]);

    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "mkl_sparse_s_create_csr failed, falling back to custom backend\n";
        custom_csr_spmm_fp32(a, f, rank, y, accumulate);
        return;
    }

    matrix_descr descr;
    descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    descr.mode = SPARSE_FILL_MODE_FULL;
    descr.diag = SPARSE_DIAG_NON_UNIT;

    float alpha = 1.0f;
    float beta = accumulate ? 1.0f : 0.0f;
    status = mkl_sparse_s_mm(
        SPARSE_OPERATION_NON_TRANSPOSE,
        alpha,
        handle,
        descr,
        SPARSE_LAYOUT_ROW_MAJOR,
        f.data(),
        (MKL_INT)rank,
        (MKL_INT)rank,
        beta,
        y->empty() ? (float*)0 : &((*y)[0]),
        (MKL_INT)rank);

    sparse_status_t destroy_status = mkl_sparse_destroy(handle);
    if (status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "mkl_sparse_s_mm failed, falling back to custom backend\n";
        custom_csr_spmm_fp32(a, f, rank, y, accumulate);
    }
    if (destroy_status != SPARSE_STATUS_SUCCESS) {
        std::cerr << "mkl_sparse_destroy failed\n";
    }
}
#endif

inline void backend_csr_spmm_fp64(SpMMBackend backend,
                                  const CsrMatrix& a,
                                  const DenseMatrix<double>& f,
                                  size_t rank,
                                  std::vector<double>* y,
                                  bool accumulate) {
#if defined(USE_MKL_BACKEND)
    if (backend == BACKEND_MKL) {
        mkl_csr_spmm_fp64(a, f, rank, y, accumulate);
        return;
    }
#else
    (void)backend;
#endif
    custom_csr_spmm_fp64(a, f, rank, y, accumulate);
}

inline void backend_csr_spmm_fp32(SpMMBackend backend,
                                  const CsrMatrixF& a,
                                  const DenseMatrix<float>& f,
                                  size_t rank,
                                  std::vector<float>* y,
                                  bool accumulate) {
#if defined(USE_MKL_BACKEND)
    if (backend == BACKEND_MKL) {
        mkl_csr_spmm_fp32(a, f, rank, y, accumulate);
        return;
    }
#else
    (void)backend;
#endif
    custom_csr_spmm_fp32(a, f, rank, y, accumulate);
}

#endif
