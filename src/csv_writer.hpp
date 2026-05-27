#ifndef CSV_WRITER_HPP
#define CSV_WRITER_HPP

#include "metrics.hpp"

#include <fstream>
#include <iomanip>
#include <string>

class CsvWriter {
public:
    explicit CsvWriter(const std::string& path) : out_(path.c_str()) {
        out_ << "total_ms,"
             << "format_prepare_ms,"
             << "upcast_prepare_ms,"
             << "compute_ms,"
             << "kernel_ms,"
             << "index_storage_bytes,"
             << "value_storage_bytes,"
             << "factor_storage_bytes,"
             << "output_storage_bytes,"
             << "index_logical_read_bytes,"
             << "value_logical_read_bytes,"
             << "factor_logical_read_bytes,"
             << "factor_compute_logical_read_bytes,"
             << "output_logical_write_bytes,"
             << "rel_error,"
             << "rank,"
             << "nnz,"
             << "mode,"
             << "thread_count,"
             << "seed,"
             << "variant,"
             << "dim0,"
             << "dim1,"
             << "dim2,"
             << "repeat,"
             << "backend,"
             << "csr_nrows,"
             << "csr_ncols,"
             << "csr_nnz\n";
    }

    bool good() const { return out_.good(); }

    void write(const MetricsRow& r) {
        out_ << std::setprecision(12)
             << r.total_ms << ","
             << r.format_prepare_ms << ","
             << r.upcast_prepare_ms << ","
             << r.compute_ms << ","
             << r.kernel_ms << ","
             << r.index_storage_bytes << ","
             << r.value_storage_bytes << ","
             << r.factor_storage_bytes << ","
             << r.output_storage_bytes << ","
             << r.index_logical_read_bytes << ","
             << r.value_logical_read_bytes << ","
             << r.factor_logical_read_bytes << ","
             << r.factor_compute_logical_read_bytes << ","
             << r.output_logical_write_bytes << ","
             << r.rel_error << ","
             << r.rank << ","
             << r.nnz << ","
             << r.mode << ","
             << r.thread_count << ","
             << r.seed << ","
             << r.variant << ","
             << r.dim0 << ","
             << r.dim1 << ","
             << r.dim2 << ","
             << r.repeat << ","
             << r.backend << ","
             << r.csr_nrows << ","
             << r.csr_ncols << ","
             << r.csr_nnz << "\n";
    }

private:
    std::ofstream out_;
};

#endif
