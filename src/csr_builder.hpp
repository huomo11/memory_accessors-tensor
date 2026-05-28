#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

#include "ttm_mapping.hpp"

struct CSRMatrix {
  int rows = 0;
  int cols = 0;
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<double> values;
};

inline CSRMatrix build_csr(std::vector<MappedEntry> entries, int rows, int cols) {
  std::stable_sort(entries.begin(), entries.end(),
                   [](const MappedEntry& a, const MappedEntry& b) {
                     if (a.output_row != b.output_row) {
                       return a.output_row < b.output_row;
                     }
                     return a.factor_row < b.factor_row;
                   });

  CSRMatrix csr;
  csr.rows = rows;
  csr.cols = cols;
  csr.row_ptr.assign(static_cast<std::size_t>(rows) + 1, 0);
  csr.col_idx.resize(entries.size());
  csr.values.resize(entries.size());

  for (const auto& entry : entries) {
    ++csr.row_ptr[static_cast<std::size_t>(entry.output_row) + 1];
  }
  for (int r = 0; r < rows; ++r) {
    csr.row_ptr[static_cast<std::size_t>(r) + 1] +=
        csr.row_ptr[static_cast<std::size_t>(r)];
  }

  for (std::size_t n = 0; n < entries.size(); ++n) {
    csr.col_idx[n] = entries[n].factor_row;
    csr.values[n] = entries[n].value;
  }
  return csr;
}
