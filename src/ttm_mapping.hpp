#pragma once

#include <cstdint>
#include <stdexcept>
#include <vector>

#include "synthetic_tensor.hpp"

struct MappedEntry {
  int output_row = 0;
  int factor_row = 0;
  double value = 0.0;
};

struct MappingShape {
  int output_rows = 0;
  int factor_rows = 0;
};

inline MappingShape mapping_shape(const SparseTensorCOO& tensor, int mode) {
  switch (mode) {
    case 0:
      return {tensor.dim1 * tensor.dim2, tensor.dim0};
    case 1:
      return {tensor.dim0 * tensor.dim2, tensor.dim1};
    case 2:
      return {tensor.dim0 * tensor.dim1, tensor.dim2};
    default:
      throw std::invalid_argument("mode must be 0, 1, or 2");
  }
}

inline std::vector<MappedEntry> map_ttm_entries(const SparseTensorCOO& tensor,
                                                int mode) {
  const auto nnz = tensor.values.size();
  std::vector<MappedEntry> entries(nnz);
  for (std::size_t n = 0; n < nnz; ++n) {
    const int i = tensor.i[n];
    const int j = tensor.j[n];
    const int k = tensor.k[n];
    if (mode == 0) {
      entries[n] = {j * tensor.dim2 + k, i, tensor.values[n]};
    } else if (mode == 1) {
      entries[n] = {i * tensor.dim2 + k, j, tensor.values[n]};
    } else if (mode == 2) {
      entries[n] = {i * tensor.dim1 + j, k, tensor.values[n]};
    } else {
      throw std::invalid_argument("mode must be 0, 1, or 2");
    }
  }
  return entries;
}
