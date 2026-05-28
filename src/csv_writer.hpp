#pragma once

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

struct CsvRow {
  std::string variant;
  int mode = 0;
  int rank = 0;
  std::int64_t nnz = 0;
  int threads = 0;
  int repeat = 0;
  int dim0 = 0;
  int dim1 = 0;
  int dim2 = 0;
  int output_rows = 0;
  int factor_rows = 0;
  std::int64_t csr_nnz = 0;
  double mapping_ms = 0.0;
  double csr_build_ms = 0.0;
  double mkl_create_ms = 0.0;
  double mkl_optimize_ms = 0.0;
  double factor_init_ms = 0.0;
  double output_init_ms = 0.0;
  double mkl_spmm_ms = 0.0;
  double checksum_ms = 0.0;
  double total_ms = 0.0;
  double mkl_spmm_pct = 0.0;
  double prepare_pct = 0.0;
  std::uint64_t index_bytes = 0;
  std::uint64_t value_bytes = 0;
  std::uint64_t factor_matrix_bytes = 0;
  std::uint64_t output_matrix_bytes = 0;
  std::uint64_t factor_logical_bytes = 0;
  std::uint64_t y_write_min_bytes = 0;
  double checksum = 0.0;
};

class CsvWriter {
 public:
  explicit CsvWriter(const std::string& path) {
    const std::filesystem::path p(path);
    if (p.has_parent_path()) {
      std::filesystem::create_directories(p.parent_path());
    }
    out_.open(path, std::ios::out | std::ios::trunc);
    if (!out_) {
      throw std::runtime_error("failed to open CSV output: " + path);
    }
    write_header();
  }

  void write(const CsvRow& r) {
    out_ << r.variant << ',' << r.mode << ',' << r.rank << ',' << r.nnz << ','
         << r.threads << ',' << r.repeat << ',' << r.dim0 << ',' << r.dim1
         << ',' << r.dim2 << ',' << r.output_rows << ',' << r.factor_rows
         << ',' << r.csr_nnz << ',' << fp(r.mapping_ms) << ','
         << fp(r.csr_build_ms) << ',' << fp(r.mkl_create_ms) << ','
         << fp(r.mkl_optimize_ms) << ',' << fp(r.factor_init_ms) << ','
         << fp(r.output_init_ms) << ',' << fp(r.mkl_spmm_ms) << ','
         << fp(r.checksum_ms) << ',' << fp(r.total_ms) << ','
         << fp(r.mkl_spmm_pct) << ',' << fp(r.prepare_pct) << ','
         << r.index_bytes << ',' << r.value_bytes << ','
         << r.factor_matrix_bytes << ',' << r.output_matrix_bytes << ','
         << r.factor_logical_bytes << ',' << r.y_write_min_bytes << ','
         << std::setprecision(17) << r.checksum << '\n';
  }

 private:
  static std::string fp(double value) {
    std::ostringstream ss;
    ss << std::fixed << std::setprecision(6) << value;
    return ss.str();
  }

  void write_header() {
    out_ << "variant,mode,rank,nnz,threads,repeat,dim0,dim1,dim2,"
            "output_rows,factor_rows,csr_nnz,mapping_ms,csr_build_ms,"
            "mkl_create_ms,mkl_optimize_ms,factor_init_ms,output_init_ms,"
            "mkl_spmm_ms,checksum_ms,total_ms,mkl_spmm_pct,prepare_pct,"
            "index_bytes,value_bytes,factor_matrix_bytes,output_matrix_bytes,"
            "factor_logical_bytes,y_write_min_bytes,checksum\n";
  }

  std::ofstream out_;
};
