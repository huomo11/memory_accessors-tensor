#include <cstdint>
#include <exception>
#include <iostream>
#include <random>
#include <utility>
#include <vector>

#include "args.hpp"
#include "checksum.hpp"
#include "csr_builder.hpp"
#include "csv_writer.hpp"
#include "mkl_spmm_profile.hpp"
#include "synthetic_tensor.hpp"
#include "timer.hpp"
#include "traffic_model.hpp"
#include "ttm_mapping.hpp"

namespace {

std::vector<double> make_factor(int rows, int rank, std::uint64_t seed) {
  std::vector<double> factor(static_cast<std::size_t>(rows) * rank);
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<double> dist(-1.0, 1.0);
  for (double& v : factor) {
    v = dist(rng);
  }
  return factor;
}

CsvRow profile_once(const SparseTensorCOO& tensor, const Args& args, int mode,
                    int rank, int repeat) {
  CsvRow row;
  row.variant = "mkl_fp64_profile";
  row.mode = mode;
  row.rank = rank;
  row.nnz = args.nnz;
  row.threads = args.threads;
  row.repeat = repeat;
  row.dim0 = tensor.dim0;
  row.dim1 = tensor.dim1;
  row.dim2 = tensor.dim2;

  Timer total;

  Timer timer;
  auto mapped = map_ttm_entries(tensor, mode);
  row.mapping_ms = timer.elapsed_ms();

  const auto shape = mapping_shape(tensor, mode);
  row.output_rows = shape.output_rows;
  row.factor_rows = shape.factor_rows;

  timer.reset();
  auto csr = build_csr(std::move(mapped), shape.output_rows, shape.factor_rows);
  row.csr_build_ms = timer.elapsed_ms();
  row.csr_nnz = static_cast<std::int64_t>(csr.values.size());

  double create_ms = 0.0;
  auto mkl_csr = create_mkl_csr(csr, &create_ms);
  row.mkl_create_ms = create_ms;

  row.mkl_optimize_ms = optimize_mkl_csr(mkl_csr);

  timer.reset();
  auto factor = make_factor(shape.factor_rows, rank,
                            args.seed + 1000003ULL * static_cast<std::uint64_t>(rank) +
                                97ULL * static_cast<std::uint64_t>(mode));
  row.factor_init_ms = timer.elapsed_ms();

  timer.reset();
  std::vector<double> output(static_cast<std::size_t>(shape.output_rows) * rank,
                             0.0);
  row.output_init_ms = timer.elapsed_ms();

  row.mkl_spmm_ms = run_mkl_spmm(mkl_csr, rank, factor, output);

  timer.reset();
  row.checksum = checksum_dense(output);
  row.checksum_ms = timer.elapsed_ms();

  row.total_ms = total.elapsed_ms();
  const double prepare_ms = row.mapping_ms + row.csr_build_ms + row.mkl_create_ms +
                            row.mkl_optimize_ms + row.factor_init_ms +
                            row.output_init_ms;
  if (row.total_ms > 0.0) {
    row.prepare_pct = prepare_ms / row.total_ms;
    row.mkl_spmm_pct = row.mkl_spmm_ms / row.total_ms;
  }

  const auto traffic =
      estimate_traffic(shape.output_rows, shape.factor_rows, row.csr_nnz, rank);
  row.index_bytes = traffic.index_bytes;
  row.value_bytes = traffic.value_bytes;
  row.factor_matrix_bytes = traffic.factor_matrix_bytes;
  row.output_matrix_bytes = traffic.output_matrix_bytes;
  row.factor_logical_bytes = traffic.factor_logical_bytes;
  row.y_write_min_bytes = traffic.y_write_min_bytes;

  return row;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Args args = parse_args(argc, argv);
    set_thread_count(args.threads);

    std::cerr << "Generating synthetic tensor: dims=" << args.dim0 << ','
              << args.dim1 << ',' << args.dim2 << " nnz=" << args.nnz
              << " seed=" << args.seed << '\n';
    const auto tensor =
        make_synthetic_tensor(args.dim0, args.dim1, args.dim2, args.nnz, args.seed);

    CsvWriter csv(args.output);
    for (const int mode : args.modes) {
      for (const int rank : args.ranks) {
        std::cerr << "warmup mode=" << mode << " rank=" << rank << '\n';
        (void)profile_once(tensor, args, mode, rank, -1);
        for (int repeat = 0; repeat < args.repeats; ++repeat) {
          const auto row = profile_once(tensor, args, mode, rank, repeat);
          csv.write(row);
          std::cerr << "mode=" << mode << " rank=" << rank
                    << " repeat=" << repeat << " total_ms=" << row.total_ms
                    << " mkl_spmm_ms=" << row.mkl_spmm_ms
                    << " checksum=" << row.checksum << '\n';
        }
      }
    }
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "error: " << e.what() << '\n';
    return 1;
  }
}
