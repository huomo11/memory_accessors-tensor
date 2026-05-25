#ifndef SPARSE_TENSOR_HPP
#define SPARSE_TENSOR_HPP

#include <stdint.h>
#include <cstdlib>
#include <random>
#include <vector>

struct SparseEntry {
    uint32_t i;
    uint32_t j;
    uint32_t k;
    double value;

    SparseEntry() : i(0), j(0), k(0), value(0.0) {}
    SparseEntry(uint32_t ii, uint32_t jj, uint32_t kk, double vv)
        : i(ii), j(jj), k(kk), value(vv) {}
};

struct SparseTensorCOO {
    uint32_t dim0;
    uint32_t dim1;
    uint32_t dim2;
    std::vector<SparseEntry> entries;

    SparseTensorCOO() : dim0(0), dim1(0), dim2(0) {}
    SparseTensorCOO(uint32_t d0, uint32_t d1, uint32_t d2)
        : dim0(d0), dim1(d1), dim2(d2) {}

    size_t nnz() const { return entries.size(); }
};

inline SparseTensorCOO generate_random_tensor(uint32_t d0, uint32_t d1, uint32_t d2,
                                              size_t nnz, uint64_t seed) {
    SparseTensorCOO x(d0, d1, d2);
    x.entries.reserve(nnz);

    std::mt19937_64 rng(seed);
    std::uniform_int_distribution<unsigned long> dist0(0, d0 - 1);
    std::uniform_int_distribution<unsigned long> dist1(0, d1 - 1);
    std::uniform_int_distribution<unsigned long> dist2(0, d2 - 1);
    std::uniform_real_distribution<double> val_dist(-1.0, 1.0);

    size_t n;
    for (n = 0; n < nnz; ++n) {
        x.entries.push_back(SparseEntry((uint32_t)dist0(rng),
                                        (uint32_t)dist1(rng),
                                        (uint32_t)dist2(rng),
                                        val_dist(rng)));
    }

    return x;
}

#endif
