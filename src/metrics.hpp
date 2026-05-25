#ifndef METRICS_HPP
#define METRICS_HPP

#include <stdint.h>
#include <stddef.h>
#include <cmath>
#include <vector>

struct MetricsRow {
    double total_ms;
    double format_prepare_ms;
    double upcast_prepare_ms;
    double compute_ms;
    double kernel_ms;
    uint64_t index_storage_bytes;
    uint64_t value_storage_bytes;
    uint64_t factor_storage_bytes;
    uint64_t output_storage_bytes;
    uint64_t index_logical_read_bytes;
    uint64_t value_logical_read_bytes;
    uint64_t factor_logical_read_bytes;
    uint64_t factor_compute_logical_read_bytes;
    uint64_t tile_workspace_bytes;
    uint64_t output_logical_write_bytes;
    double rel_error;
    int rank;
    uint64_t nnz;
    int mode;
    int thread_count;
    uint64_t seed;
    const char* variant;
    uint32_t dim0;
    uint32_t dim1;
    uint32_t dim2;
    uint32_t tile_rows;
    int repeat;

    MetricsRow()
        : total_ms(0.0), format_prepare_ms(0.0), upcast_prepare_ms(0.0),
          compute_ms(0.0), kernel_ms(0.0), index_storage_bytes(0), value_storage_bytes(0),
          factor_storage_bytes(0), output_storage_bytes(0),
          index_logical_read_bytes(0), value_logical_read_bytes(0),
          factor_logical_read_bytes(0), factor_compute_logical_read_bytes(0),
          tile_workspace_bytes(0), output_logical_write_bytes(0),
          rel_error(0.0), rank(0), nnz(0), mode(0), thread_count(1),
          seed(0), variant(""), dim0(0), dim1(0), dim2(0), tile_rows(64),
          repeat(0) {}
};

inline double relative_frobenius_error(const std::vector<double>& y,
                                       const std::vector<double>& baseline) {
    double num = 0.0;
    double den = 0.0;
    size_t i;
    for (i = 0; i < y.size() && i < baseline.size(); ++i) {
        double diff = y[i] - baseline[i];
        num += diff * diff;
        den += baseline[i] * baseline[i];
    }
    if (den == 0.0) {
        return num == 0.0 ? 0.0 : std::sqrt(num);
    }
    return std::sqrt(num / den);
}

#endif
