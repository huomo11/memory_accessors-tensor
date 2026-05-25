#ifndef SP_TTM_HPP
#define SP_TTM_HPP

#include "dense_matrix.hpp"
#include "sparse_tensor.hpp"

#include <algorithm>
#include <stdint.h>
#include <stddef.h>
#include <thread>
#include <vector>

struct PreparedEntry {
    uint32_t output_row;
    uint32_t factor_row;
    uint32_t source_index;

    PreparedEntry() : output_row(0), factor_row(0), source_index(0) {}
    PreparedEntry(uint32_t out, uint32_t factor, uint32_t src)
        : output_row(out), factor_row(factor), source_index(src) {}
};

struct PreparedTTM {
    int mode;
    uint32_t output_rows;
    uint32_t factor_rows;
    std::vector<PreparedEntry> entries;
    std::vector<size_t> row_starts;

    PreparedTTM() : mode(0), output_rows(0), factor_rows(0) {}
};

inline uint32_t ttm_output_rows(const SparseTensorCOO& x, int mode) {
    if (mode == 0) {
        return x.dim1 * x.dim2;
    }
    if (mode == 1) {
        return x.dim0 * x.dim2;
    }
    return x.dim0 * x.dim1;
}

inline uint32_t ttm_factor_rows(const SparseTensorCOO& x, int mode) {
    if (mode == 0) {
        return x.dim0;
    }
    if (mode == 1) {
        return x.dim1;
    }
    return x.dim2;
}

inline void map_entry_for_mode(const SparseEntry& e, const SparseTensorCOO& x, int mode,
                               uint32_t* output_row, uint32_t* factor_row) {
    if (mode == 0) {
        *output_row = e.j * x.dim2 + e.k;
        *factor_row = e.i;
    } else if (mode == 1) {
        *output_row = e.i * x.dim2 + e.k;
        *factor_row = e.j;
    } else {
        *output_row = e.i * x.dim1 + e.j;
        *factor_row = e.k;
    }
}

inline bool prepared_entry_less(const PreparedEntry& a, const PreparedEntry& b) {
    if (a.output_row != b.output_row) {
        return a.output_row < b.output_row;
    }
    return a.factor_row < b.factor_row;
}

inline PreparedTTM prepare_ttm(const SparseTensorCOO& x, int mode) {
    PreparedTTM p;
    p.mode = mode;
    p.output_rows = ttm_output_rows(x, mode);
    p.factor_rows = ttm_factor_rows(x, mode);
    p.entries.reserve(x.entries.size());

    size_t n;
    for (n = 0; n < x.entries.size(); ++n) {
        uint32_t out = 0;
        uint32_t factor = 0;
        map_entry_for_mode(x.entries[n], x, mode, &out, &factor);
        p.entries.push_back(PreparedEntry(out, factor, (uint32_t)n));
    }

    std::sort(p.entries.begin(), p.entries.end(), prepared_entry_less);

    p.row_starts.assign((size_t)p.output_rows + 1, 0);
    for (n = 0; n < p.entries.size(); ++n) {
        p.row_starts[(size_t)p.entries[n].output_row + 1] += 1;
    }
    for (n = 1; n < p.row_starts.size(); ++n) {
        p.row_starts[n] += p.row_starts[n - 1];
    }
    return p;
}

template <typename ValueT, typename FactorT, typename OutputT>
void compute_rows_worker(const PreparedTTM* p,
                         const ValueT* values,
                         const FactorT* factor,
                         size_t rank,
                         OutputT* output,
                         uint32_t row_begin,
                         uint32_t row_end) {
    uint32_t row;
    for (row = row_begin; row < row_end; ++row) {
        size_t start = p->row_starts[row];
        size_t end = p->row_starts[(size_t)row + 1];
        size_t pos;
        for (pos = start; pos < end; ++pos) {
            const PreparedEntry& e = p->entries[pos];
            OutputT v = (OutputT)values[e.source_index];
            size_t r;
            for (r = 0; r < rank; ++r) {
                output[(size_t)row * rank + r] +=
                    v * (OutputT)factor[(size_t)e.factor_row * rank + r];
            }
        }
    }
}

template <typename ValueT, typename FactorT, typename OutputT>
void compute_ttm_threaded(const PreparedTTM& p,
                          const std::vector<ValueT>& values,
                          const DenseMatrix<FactorT>& factor,
                          size_t rank,
                          int thread_count,
                          std::vector<OutputT>* output) {
    output->assign((size_t)p.output_rows * rank, (OutputT)0);
    if (thread_count < 1) {
        thread_count = 1;
    }
    if ((uint32_t)thread_count > p.output_rows && p.output_rows > 0) {
        thread_count = (int)p.output_rows;
    }

    std::vector<std::thread> threads;
    uint32_t t;
    for (t = 0; t < (uint32_t)thread_count; ++t) {
        uint32_t begin = (uint32_t)(((uint64_t)p.output_rows * t) / (uint32_t)thread_count);
        uint32_t end = (uint32_t)(((uint64_t)p.output_rows * (t + 1)) / (uint32_t)thread_count);
        threads.push_back(std::thread(compute_rows_worker<ValueT, FactorT, OutputT>,
                                      &p,
                                      values.empty() ? (const ValueT*)0 : &values[0],
                                      factor.data(),
                                      rank,
                                      output->empty() ? (OutputT*)0 : &((*output)[0]),
                                      begin,
                                      end));
    }

    for (t = 0; t < threads.size(); ++t) {
        threads[t].join();
    }
}

template <typename T>
void copy_as_double(const std::vector<T>& src, std::vector<double>* dst) {
    dst->assign(src.size(), 0.0);
    size_t i;
    for (i = 0; i < src.size(); ++i) {
        (*dst)[i] = (double)src[i];
    }
}

#endif
