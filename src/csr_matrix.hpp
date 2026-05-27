#ifndef CSR_MATRIX_HPP
#define CSR_MATRIX_HPP

#include "sp_ttm.hpp"

#include <stdint.h>
#include <stddef.h>
#include <vector>

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

inline CsrMatrix build_csr_from_prepared(const PreparedTTM& prep,
                                         const std::vector<double>& source_values) {
    CsrMatrix csr(prep.output_rows, prep.factor_rows);
    csr.row_ptr = prep.row_starts;
    csr.col_idx.assign(prep.entries.size(), 0);
    csr.values.assign(prep.entries.size(), 0.0);

    size_t p;
    for (p = 0; p < prep.entries.size(); ++p) {
        const PreparedEntry& e = prep.entries[p];
        csr.col_idx[p] = e.factor_row;
        csr.values[p] = source_values[e.source_index];
    }
    return csr;
}

inline CsrMatrixF build_csr_float_from_prepared(const PreparedTTM& prep,
                                                const std::vector<double>& source_values) {
    CsrMatrixF csr(prep.output_rows, prep.factor_rows);
    csr.row_ptr = prep.row_starts;
    csr.col_idx.assign(prep.entries.size(), 0);
    csr.values.assign(prep.entries.size(), 0.0f);

    size_t p;
    for (p = 0; p < prep.entries.size(); ++p) {
        const PreparedEntry& e = prep.entries[p];
        csr.col_idx[p] = e.factor_row;
        csr.values[p] = (float)source_values[e.source_index];
    }
    return csr;
}

inline CsrMatrix build_csr_column_tile(const CsrMatrix& csr,
                                       uint32_t col_begin,
                                       uint32_t col_end) {
    CsrMatrix tile(csr.nrows, col_end - col_begin);
    size_t row;
    for (row = 0; row < (size_t)csr.nrows; ++row) {
        size_t p;
        for (p = csr.row_ptr[row]; p < csr.row_ptr[row + 1]; ++p) {
            uint32_t col = csr.col_idx[p];
            if (col >= col_begin && col < col_end) {
                tile.row_ptr[row + 1] += 1;
            }
        }
    }
    for (row = 1; row < tile.row_ptr.size(); ++row) {
        tile.row_ptr[row] += tile.row_ptr[row - 1];
    }

    tile.col_idx.assign(tile.row_ptr[(size_t)csr.nrows], 0);
    tile.values.assign(tile.row_ptr[(size_t)csr.nrows], 0.0);
    std::vector<size_t> cursor = tile.row_ptr;
    for (row = 0; row < (size_t)csr.nrows; ++row) {
        size_t p;
        for (p = csr.row_ptr[row]; p < csr.row_ptr[row + 1]; ++p) {
            uint32_t col = csr.col_idx[p];
            if (col >= col_begin && col < col_end) {
                size_t out_pos = cursor[row];
                tile.col_idx[out_pos] = col - col_begin;
                tile.values[out_pos] = csr.values[p];
                cursor[row] += 1;
            }
        }
    }
    return tile;
}

inline void build_csr_factor_tiles(const CsrMatrix& csr,
                                   uint32_t tile_rows,
                                   std::vector<CsrMatrix>* tiles) {
    uint32_t rows = tile_rows == 0 ? 64 : tile_rows;
    uint32_t tile_count = (csr.ncols + rows - 1) / rows;
    tiles->clear();
    tiles->reserve(tile_count);
    uint32_t tile;
    for (tile = 0; tile < tile_count; ++tile) {
        uint32_t col_begin = tile * rows;
        uint32_t col_end = col_begin + rows;
        if (col_end > csr.ncols) {
            col_end = csr.ncols;
        }
        tiles->push_back(build_csr_column_tile(csr, col_begin, col_end));
    }
}

#endif
