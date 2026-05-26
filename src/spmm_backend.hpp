#ifndef SPMM_BACKEND_HPP
#define SPMM_BACKEND_HPP

#include "csr_matrix.hpp"
#include "dense_matrix.hpp"

#include <stddef.h>
#include <string>
#include <vector>

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
    return BACKEND_CUSTOM;
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

inline void backend_csr_spmm_fp64(SpMMBackend backend,
                                  const CsrMatrix& a,
                                  const DenseMatrix<double>& f,
                                  size_t rank,
                                  std::vector<double>* y,
                                  bool accumulate) {
    (void)backend;
    custom_csr_spmm_fp64(a, f, rank, y, accumulate);
}

#endif
