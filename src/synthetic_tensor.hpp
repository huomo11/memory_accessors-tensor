#pragma once

#include <cstdint>
#include <random>
#include <vector>

struct SparseTensorCOO {
  int dim0 = 0;
  int dim1 = 0;
  int dim2 = 0;
  std::vector<int> i;
  std::vector<int> j;
  std::vector<int> k;
  std::vector<double> values;
};

inline SparseTensorCOO make_synthetic_tensor(int dim0, int dim1, int dim2,
                                             std::int64_t nnz,
                                             std::uint64_t seed) {
  SparseTensorCOO tensor;
  tensor.dim0 = dim0;
  tensor.dim1 = dim1;
  tensor.dim2 = dim2;
  tensor.i.resize(static_cast<std::size_t>(nnz));
  tensor.j.resize(static_cast<std::size_t>(nnz));
  tensor.k.resize(static_cast<std::size_t>(nnz));
  tensor.values.resize(static_cast<std::size_t>(nnz));

  std::mt19937_64 rng(seed);
  std::uniform_int_distribution<int> dist_i(0, dim0 - 1);
  std::uniform_int_distribution<int> dist_j(0, dim1 - 1);
  std::uniform_int_distribution<int> dist_k(0, dim2 - 1);
  std::uniform_real_distribution<double> dist_v(-1.0, 1.0);

  for (std::int64_t n = 0; n < nnz; ++n) {
    const auto idx = static_cast<std::size_t>(n);
    tensor.i[idx] = dist_i(rng);
    tensor.j[idx] = dist_j(rng);
    tensor.k[idx] = dist_k(rng);
    tensor.values[idx] = dist_v(rng);
  }
  return tensor;
}
