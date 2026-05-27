#ifndef CSR_MATRIX_HPP
#define CSR_MATRIX_HPP

#include "sparse_tensor.hpp"

#include <algorithm>
#include <stdint.h>
#include <stddef.h>
#include <vector>

struct CsrBuildEntry {
    uint32_t row;
    uint32_t col;
    uint32_t source_index;

    CsrBuildEntry() : row(0), col(0), source_index(0) {}
    CsrBuildEntry(uint32_t r, uint32_t c, uint32_t s)
        : row(r), col(c), source_index(s) {}
};

struct CsrMatrix {
    uint32_t nrows;
    uint32_t ncols;
    std::vector<size_t> row_ptr;
    std::vector<uint32_t> col_idx;
    std::vector<double> values;

    CsrMatrix() : nrows(0), ncols(0) {}
    CsrMatrix(uint32_t rows, uint32_t cols) : nrows(rows), ncols(cols) {
        row_ptr.assign((size_t)rows + 1, 0);
    }

    size_t nnz() const { return values.size(); }
};

struct CsrMatrixF {
    uint32_t nrows;
    uint32_t ncols;
    std::vector<size_t> row_ptr;
    std::vector<uint32_t> col_idx;
    std::vector<float> values;

    CsrMatrixF() : nrows(0), ncols(0) {}
    CsrMatrixF(uint32_t rows, uint32_t cols) : nrows(rows), ncols(cols) {
        row_ptr.assign((size_t)rows + 1, 0);
    }

    size_t nnz() const { return values.size(); }
};

inline uint32_t csr_unfold_rows(const SparseTensorCOO& x, int mode) {
    if (mode == 0) {
        return x.dim1 * x.dim2;
    }
    if (mode == 1) {
        return x.dim0 * x.dim2;
    }
    return x.dim0 * x.dim1;
}

inline uint32_t csr_unfold_cols(const SparseTensorCOO& x, int mode) {
    if (mode == 0) {
        return x.dim0;
    }
    if (mode == 1) {
        return x.dim1;
    }
    return x.dim2;
}

inline void tensor_entry_to_csr(const SparseEntry& e, const SparseTensorCOO& x,
                                int mode, uint32_t* row, uint32_t* col) {
    if (mode == 0) {
        *row = e.j * x.dim2 + e.k;
        *col = e.i;
    } else if (mode == 1) {
        *row = e.i * x.dim2 + e.k;
        *col = e.j;
    } else {
        *row = e.i * x.dim1 + e.j;
        *col = e.k;
    }
}

inline bool csr_build_entry_less(const CsrBuildEntry& a, const CsrBuildEntry& b) {
    if (a.row != b.row) {
        return a.row < b.row;
    }
    return a.col < b.col;
}

inline void build_csr_entries(const SparseTensorCOO& x, int mode,
                              std::vector<CsrBuildEntry>* entries) {
    entries->clear();
    entries->reserve(x.entries.size());
    size_t n;
    for (n = 0; n < x.entries.size(); ++n) {
        uint32_t row = 0;
        uint32_t col = 0;
        tensor_entry_to_csr(x.entries[n], x, mode, &row, &col);
        entries->push_back(CsrBuildEntry(row, col, (uint32_t)n));
    }
    std::sort(entries->begin(), entries->end(), csr_build_entry_less);
}

inline CsrMatrix build_csr_fp64_from_tensor(const SparseTensorCOO& x, int mode) {
    CsrMatrix csr(csr_unfold_rows(x, mode), csr_unfold_cols(x, mode));
    std::vector<CsrBuildEntry> entries;
    build_csr_entries(x, mode, &entries);

    size_t n;
    for (n = 0; n < entries.size(); ++n) {
        csr.row_ptr[(size_t)entries[n].row + 1] += 1;
    }
    for (n = 1; n < csr.row_ptr.size(); ++n) {
        csr.row_ptr[n] += csr.row_ptr[n - 1];
    }

    csr.col_idx.assign(entries.size(), 0);
    csr.values.assign(entries.size(), 0.0);
    for (n = 0; n < entries.size(); ++n) {
        csr.col_idx[n] = entries[n].col;
        csr.values[n] = x.entries[entries[n].source_index].value;
    }
    return csr;
}

inline CsrMatrixF build_csr_fp32_from_tensor(const SparseTensorCOO& x, int mode) {
    CsrMatrixF csr(csr_unfold_rows(x, mode), csr_unfold_cols(x, mode));
    std::vector<CsrBuildEntry> entries;
    build_csr_entries(x, mode, &entries);

    size_t n;
    for (n = 0; n < entries.size(); ++n) {
        csr.row_ptr[(size_t)entries[n].row + 1] += 1;
    }
    for (n = 1; n < csr.row_ptr.size(); ++n) {
        csr.row_ptr[n] += csr.row_ptr[n - 1];
    }

    csr.col_idx.assign(entries.size(), 0);
    csr.values.assign(entries.size(), 0.0f);
    for (n = 0; n < entries.size(); ++n) {
        csr.col_idx[n] = entries[n].col;
        csr.values[n] = (float)x.entries[entries[n].source_index].value;
    }
    return csr;
}

#endif
