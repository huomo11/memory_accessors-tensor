#include "csv_writer.hpp"
#include "dense_matrix.hpp"
#include "metrics.hpp"
#include "precision_policy.hpp"
#include "sparse_tensor.hpp"
#include "sp_ttm.hpp"
#include "timer.hpp"

#include <errno.h>
#include <stdint.h>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>
#include <sstream>
#include <string>
#include <vector>

struct Options {
    std::string output;
    uint32_t dim0;
    uint32_t dim1;
    uint32_t dim2;
    std::vector<uint64_t> nnzs;
    std::vector<int> ranks;
    std::vector<int> modes;
    std::vector<int> thread_counts;
    int active_threads;
    int repeats;
    uint64_t seed;
    uint32_t tile_rows;

    Options()
        : output("results/smoke.csv"), dim0(16), dim1(12), dim2(10),
          active_threads(1), repeats(1), seed(42), tile_rows(64) {}
};

static bool parse_long_checked(const char* text, long* value) {
    char* end = 0;
    errno = 0;
    long v = std::strtol(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    *value = v;
    return true;
}

static bool parse_ull_checked(const char* text, uint64_t* value) {
    char* end = 0;
    errno = 0;
    unsigned long long v = std::strtoull(text, &end, 10);
    if (errno != 0 || end == text || *end != '\0') {
        return false;
    }
    *value = (uint64_t)v;
    return true;
}

static std::vector<std::string> split_commas(const std::string& s) {
    std::vector<std::string> out;
    std::string cur;
    size_t i;
    for (i = 0; i < s.size(); ++i) {
        if (s[i] == ',') {
            out.push_back(cur);
            cur.clear();
        } else {
            cur.push_back(s[i]);
        }
    }
    out.push_back(cur);
    return out;
}

static bool parse_int_list(const std::string& text, std::vector<int>* out) {
    std::vector<std::string> parts = split_commas(text);
    out->clear();
    size_t i;
    for (i = 0; i < parts.size(); ++i) {
        long v = 0;
        if (!parse_long_checked(parts[i].c_str(), &v)) {
            return false;
        }
        out->push_back((int)v);
    }
    return true;
}

static bool parse_ull_list(const std::string& text, std::vector<uint64_t>* out) {
    std::vector<std::string> parts = split_commas(text);
    out->clear();
    size_t i;
    for (i = 0; i < parts.size(); ++i) {
        uint64_t v = 0;
        if (!parse_ull_checked(parts[i].c_str(), &v)) {
            return false;
        }
        out->push_back(v);
    }
    return true;
}

static bool parse_dims(const std::string& text, uint32_t* d0, uint32_t* d1, uint32_t* d2) {
    std::vector<std::string> parts = split_commas(text);
    if (parts.size() != 3) {
        return false;
    }
    long vals[3];
    int i;
    for (i = 0; i < 3; ++i) {
        if (!parse_long_checked(parts[(size_t)i].c_str(), &vals[i]) || vals[i] <= 0) {
            return false;
        }
    }
    *d0 = (uint32_t)vals[0];
    *d1 = (uint32_t)vals[1];
    *d2 = (uint32_t)vals[2];
    return true;
}

static void usage() {
    std::cerr << "Usage: ttm_bench --output out.csv --dims I,J,K --nnz N "
              << "--ranks r1,r2 --modes 0,1,2 --threads T --repeats R "
              << "--seed S --tile-rows T\n";
}

static bool parse_args(int argc, char** argv, Options* opt) {
    int i = 1;
    while (i < argc) {
        if (i + 1 >= argc) {
            return false;
        }
        std::string key(argv[i]);
        std::string val(argv[i + 1]);
        if (key == "--output") {
            opt->output = val;
        } else if (key == "--dims") {
            if (!parse_dims(val, &opt->dim0, &opt->dim1, &opt->dim2)) {
                return false;
            }
        } else if (key == "--nnz") {
            if (!parse_ull_list(val, &opt->nnzs)) {
                return false;
            }
        } else if (key == "--ranks") {
            if (!parse_int_list(val, &opt->ranks)) {
                return false;
            }
        } else if (key == "--modes") {
            if (!parse_int_list(val, &opt->modes)) {
                return false;
            }
        } else if (key == "--threads") {
            if (!parse_int_list(val, &opt->thread_counts)) {
                return false;
            }
            size_t ti;
            for (ti = 0; ti < opt->thread_counts.size(); ++ti) {
                if (opt->thread_counts[ti] < 1) {
                    return false;
                }
            }
        } else if (key == "--repeats") {
            long tmp = 0;
            if (!parse_long_checked(val.c_str(), &tmp) || tmp < 1) {
                return false;
            }
            opt->repeats = (int)tmp;
        } else if (key == "--seed") {
            if (!parse_ull_checked(val.c_str(), &opt->seed)) {
                return false;
            }
        } else if (key == "--tile-rows") {
            long tmp = 0;
            if (!parse_long_checked(val.c_str(), &tmp) || tmp < 1) {
                return false;
            }
            opt->tile_rows = (uint32_t)tmp;
        } else {
            return false;
        }
        i += 2;
    }
    if (opt->ranks.empty()) {
        opt->ranks.push_back(4);
    }
    if (opt->modes.empty()) {
        opt->modes.push_back(0);
    }
    if (opt->nnzs.empty()) {
        opt->nnzs.push_back(1000);
    }
    if (opt->thread_counts.empty()) {
        opt->thread_counts.push_back(1);
    }
    return true;
}

static DenseMatrix<double> generate_factor_double(size_t rows, size_t cols, uint64_t seed) {
    DenseMatrix<double> f(rows, cols);
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    size_t i;
    for (i = 0; i < f.size(); ++i) {
        f.values()[i] = dist(rng);
    }
    return f;
}

template <typename OutT, typename InT>
static std::vector<OutT> convert_values(const std::vector<InT>& in) {
    std::vector<OutT> out(in.size());
    size_t i;
    for (i = 0; i < in.size(); ++i) {
        out[i] = (OutT)in[i];
    }
    return out;
}

template <typename OutT, typename InT>
static DenseMatrix<OutT> convert_factor(const DenseMatrix<InT>& in) {
    DenseMatrix<OutT> out(in.rows(), in.cols());
    size_t i;
    for (i = 0; i < in.size(); ++i) {
        out.values()[i] = (OutT)in.values()[i];
    }
    return out;
}

static std::vector<double> tensor_values_double(const SparseTensorCOO& x) {
    std::vector<double> vals(x.entries.size());
    size_t i;
    for (i = 0; i < x.entries.size(); ++i) {
        vals[i] = x.entries[i].value;
    }
    return vals;
}

static MetricsRow make_metric_base(const Options& opt, const PreparedTTM& prep,
                                   VariantKind variant, int rank, int mode,
                                   int repeat, double format_ms) {
    MetricsRow row;
    uint64_t nnz = (uint64_t)prep.entries.size();
    uint64_t out_elems = (uint64_t)prep.output_rows * (uint64_t)rank;
    uint64_t factor_elems = (uint64_t)prep.factor_rows * (uint64_t)rank;
    size_t value_size = variant_value_storage_size(variant);
    size_t factor_size = variant_factor_storage_size(variant);
    size_t factor_compute_size = variant_factor_compute_read_size(variant);
    size_t output_size = variant_output_storage_size(variant);

    row.format_prepare_ms = format_ms;
    row.index_storage_bytes = nnz * 3ULL * (uint64_t)sizeof(uint32_t);
    row.value_storage_bytes = nnz * (uint64_t)value_size;
    row.factor_storage_bytes = factor_elems * (uint64_t)factor_size;
    row.output_storage_bytes = out_elems * (uint64_t)output_size;
    row.index_logical_read_bytes = nnz * 3ULL * (uint64_t)sizeof(uint32_t);
    row.value_logical_read_bytes = nnz * (uint64_t)value_size;
    row.factor_logical_read_bytes = nnz * (uint64_t)rank * (uint64_t)factor_size;
    row.factor_compute_logical_read_bytes =
        nnz * (uint64_t)rank * (uint64_t)factor_compute_size;
    row.tile_workspace_bytes = 0;
    if (variant == VAR_FACTOR_FP32_BLOCKED_COMPUTE_FP64) {
        uint64_t rows = (uint64_t)opt.tile_rows;
        if (rows > (uint64_t)prep.factor_rows) {
            rows = (uint64_t)prep.factor_rows;
        }
        row.tile_workspace_bytes = rows * (uint64_t)rank * (uint64_t)sizeof(double);
    }
    {
        uint64_t streamed = nnz * (uint64_t)rank * (uint64_t)output_size;
        row.output_logical_write_bytes = row.output_storage_bytes > streamed
            ? row.output_storage_bytes : streamed;
    }
    row.rank = rank;
    row.nnz = nnz;
    row.mode = mode;
    row.thread_count = opt.active_threads;
    row.seed = opt.seed;
    row.variant = variant_name(variant);
    row.dim0 = opt.dim0;
    row.dim1 = opt.dim1;
    row.dim2 = opt.dim2;
    row.tile_rows = opt.tile_rows;
    row.repeat = repeat;
    return row;
}

static MetricsRow run_variant(const Options& opt,
                              const PreparedTTM& prep,
                              double format_ms,
                              const std::vector<double>& values64,
                              const DenseMatrix<double>& factor64,
                              VariantKind variant,
                              int rank,
                              int mode,
                              int repeat,
                              const std::vector<double>& baseline) {
    MetricsRow row = make_metric_base(opt, prep, variant, rank, mode, repeat, format_ms);
    Timer timer;
    std::vector<double> out64;

    if (variant == VAR_FACTOR_FP64_COMPUTE_FP64) {
        row.upcast_prepare_ms = 0.0;
        timer.reset();
        compute_ttm_threaded<double, double, double>(prep, values64, factor64,
                                                     (size_t)rank, opt.active_threads, &out64);
        row.compute_ms = timer.elapsed_ms();
    } else if (variant == VAR_FACTOR_FP32_COMPUTE_FP64) {
        DenseMatrix<float> factor32 = convert_factor<float>(factor64);
        timer.reset();
        DenseMatrix<double> factor_workspace = convert_factor<double>(factor32);
        row.upcast_prepare_ms = timer.elapsed_ms();
        timer.reset();
        compute_ttm_threaded<double, double, double>(prep, values64, factor_workspace,
                                                     (size_t)rank, opt.active_threads, &out64);
        row.compute_ms = timer.elapsed_ms();
    } else if (variant == VAR_FACTOR_FP32_ONFLY_COMPUTE_FP64) {
        DenseMatrix<float> factor32 = convert_factor<float>(factor64);
        row.upcast_prepare_ms = 0.0;
        timer.reset();
        compute_ttm_threaded<double, float, double>(prep, values64, factor32,
                                                    (size_t)rank, opt.active_threads, &out64);
        row.compute_ms = timer.elapsed_ms();
    } else if (variant == VAR_FACTOR_FP32_BLOCKED_COMPUTE_FP64) {
        DenseMatrix<float> factor32 = convert_factor<float>(factor64);
        row.thread_count = 1;
        compute_ttm_blocked_fp32_fp64(prep, values64, factor32, (size_t)rank,
                                      &out64, &row.upcast_prepare_ms, &row.compute_ms);
    } else if (variant == VAR_FACTOR_FP32_COMPUTE_FP32) {
        DenseMatrix<float> factor32 = convert_factor<float>(factor64);
        std::vector<float> values32 = convert_values<float>(values64);
        std::vector<float> outf;
        row.upcast_prepare_ms = 0.0;
        timer.reset();
        compute_ttm_threaded<float, float, float>(prep, values32, factor32,
                                                  (size_t)rank, opt.active_threads, &outf);
        row.compute_ms = timer.elapsed_ms();
        copy_as_double(outf, &out64);
    } else if (variant == VAR_VALUE_FP32_FACTOR_FP64_COMPUTE_FP64) {
        std::vector<float> values32 = convert_values<float>(values64);
        timer.reset();
        std::vector<double> value_workspace = convert_values<double>(values32);
        row.upcast_prepare_ms = timer.elapsed_ms();
        timer.reset();
        compute_ttm_threaded<double, double, double>(prep, value_workspace, factor64,
                                                     (size_t)rank, opt.active_threads, &out64);
        row.compute_ms = timer.elapsed_ms();
    }

    row.kernel_ms = row.upcast_prepare_ms + row.compute_ms;
    row.total_ms = row.format_prepare_ms + row.kernel_ms;
    if (variant == VAR_FACTOR_FP64_COMPUTE_FP64) {
        row.rel_error = 0.0;
    } else {
        row.rel_error = relative_frobenius_error(out64, baseline);
    }
    return row;
}

int main(int argc, char** argv) {
    Options opt;
    if (!parse_args(argc, argv, &opt)) {
        usage();
        return 2;
    }

    CsvWriter csv(opt.output);
    if (!csv.good()) {
        std::cerr << "Failed to open output CSV: " << opt.output << "\n";
        return 1;
    }

    size_t ni;
    for (ni = 0; ni < opt.nnzs.size(); ++ni) {
        SparseTensorCOO x = generate_random_tensor(opt.dim0, opt.dim1, opt.dim2,
                                                   (size_t)opt.nnzs[ni], opt.seed + (uint64_t)ni);
        std::vector<double> values64 = tensor_values_double(x);

        size_t mi;
        for (mi = 0; mi < opt.modes.size(); ++mi) {
            int mode = opt.modes[mi];
            if (mode < 0 || mode > 2) {
                std::cerr << "Invalid mode: " << mode << "\n";
                return 2;
            }

            Timer prep_timer;
            PreparedTTM prep = prepare_ttm(x, mode, opt.tile_rows);
            double format_ms = prep_timer.elapsed_ms();

            size_t ri;
            for (ri = 0; ri < opt.ranks.size(); ++ri) {
                int rank = opt.ranks[ri];
                if (rank <= 0) {
                    std::cerr << "Invalid rank: " << rank << "\n";
                    return 2;
                }
                DenseMatrix<double> factor64 =
                    generate_factor_double((size_t)prep.factor_rows, (size_t)rank,
                                           opt.seed + (uint64_t)mode * 1000003ULL + (uint64_t)rank);

                size_t ti;
                for (ti = 0; ti < opt.thread_counts.size(); ++ti) {
                    opt.active_threads = opt.thread_counts[ti];

                    int rep;
                    for (rep = 0; rep < opt.repeats; ++rep) {
                        std::vector<double> baseline;
                        Timer compute_timer;
                        compute_ttm_threaded<double, double, double>(prep, values64, factor64,
                                                                     (size_t)rank, opt.active_threads, &baseline);
                        double baseline_compute_ms = compute_timer.elapsed_ms();

                        MetricsRow base = make_metric_base(opt, prep, VAR_FACTOR_FP64_COMPUTE_FP64,
                                                           rank, mode, rep, format_ms);
                        base.compute_ms = baseline_compute_ms;
                        base.upcast_prepare_ms = 0.0;
                        base.kernel_ms = base.upcast_prepare_ms + base.compute_ms;
                        base.total_ms = base.format_prepare_ms + base.kernel_ms;
                        base.rel_error = 0.0;
                        csv.write(base);

                        VariantKind vars[5];
                        vars[0] = VAR_FACTOR_FP32_COMPUTE_FP64;
                        vars[1] = VAR_FACTOR_FP32_ONFLY_COMPUTE_FP64;
                        vars[2] = VAR_FACTOR_FP32_BLOCKED_COMPUTE_FP64;
                        vars[3] = VAR_FACTOR_FP32_COMPUTE_FP32;
                        vars[4] = VAR_VALUE_FP32_FACTOR_FP64_COMPUTE_FP64;
                        int vi;
                        for (vi = 0; vi < 5; ++vi) {
                            MetricsRow row = run_variant(opt, prep, format_ms, values64, factor64,
                                                         vars[vi], rank, mode, rep, baseline);
                            csv.write(row);
                        }
                    }
                }
            }
        }
    }

    return 0;
}
