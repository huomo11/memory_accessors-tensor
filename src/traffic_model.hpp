#pragma once

#include <cstdint>

struct TrafficModel {
  std::uint64_t index_bytes = 0;
  std::uint64_t value_bytes = 0;
  std::uint64_t factor_matrix_bytes = 0;
  std::uint64_t output_matrix_bytes = 0;
  std::uint64_t factor_logical_bytes = 0;
  std::uint64_t y_write_min_bytes = 0;
};

inline TrafficModel estimate_traffic(int output_rows, int factor_rows,
                                     std::int64_t nnz, int rank) {
  TrafficModel t;
  t.index_bytes = static_cast<std::uint64_t>(nnz) * sizeof(int) +
                  static_cast<std::uint64_t>(output_rows + 1) * sizeof(int);
  t.value_bytes = static_cast<std::uint64_t>(nnz) * sizeof(double);
  t.factor_matrix_bytes =
      static_cast<std::uint64_t>(factor_rows) * rank * sizeof(double);
  t.output_matrix_bytes =
      static_cast<std::uint64_t>(output_rows) * rank * sizeof(double);
  t.factor_logical_bytes =
      static_cast<std::uint64_t>(nnz) * rank * sizeof(double);
  t.y_write_min_bytes = t.output_matrix_bytes;
  return t;
}
