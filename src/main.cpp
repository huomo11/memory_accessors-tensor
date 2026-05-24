#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <type_traits>
#include <vector>

namespace {

using Clock = std::chrono::steady_clock;

struct Args {
    std::size_t dim0 = 128;
    std::size_t dim1 = 128;
    std::size_t dim2 = 128;
    std::size_t nnz = 200000;
    std::vector<int> modes{0, 1, 2};
    std::vector<std::size_t> ranks{16, 32, 64};
    int repeats = 3;
    int threads = static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    std::uint64_t seed = 42;
    std::string output = "results.csv";
};

struct TensorCOO {
    std::vector<std::uint32_t> i0;
    std::vector<std::uint32_t> i1;
    std::vector<std::uint32_t> i2;
    std::vector<double> values64;
    std::vector<float> values32;
};

struct PreparedEntry {
    std::uint64_t row = 0;
    std::uint32_t k = 0;
    std::size_t nz = 0;
};

struct Prepared {
    std::vector<PreparedEntry> entries;
    std::vector<std::size_t> row_starts;
    std::vector<std::uint64_t> rows;
    std::size_t row_count = 0;
    double format_prepare_ms = 0.0;
};

struct Metrics {
    double total_ms = 0.0;
    double format_prepare_ms = 0.0;
    double upcast_prepare_ms = 0.0;
    double compute_ms = 0.0;
    std::size_t index_storage_bytes = 0;
    std::size_t value_storage_bytes = 0;
    std::size_t factor_storage_bytes = 0;
    std::size_t output_storage_bytes = 0;
    std::size_t index_logical_read_bytes = 0;
    std::size_t value_logical_read_bytes = 0;
    std::size_t factor_logical_read_bytes = 0;
    std::size_t output_logical_write_bytes = 0;
    double rel_error = 0.0;
};

enum class FactorStorage {
    Fp64,
    Fp32,
};

enum class ValueStorage {
    Fp64,
    Fp32,
};

enum class ComputeType {
    Fp64,
    Fp32,
};

struct Variant {
    std::string name;
    FactorStorage factor_storage;
    ValueStorage value_storage;
    ComputeType compute_type;
};

double ms_since(const Clock::time_point start, const Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - start).count();
}

std::vector<std::string> split(const std::string& s, const char delim) {
    std::vector<std::string> out;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        if (!item.empty()) {
            out.push_back(item);
        }
    }
    return out;
}

std::vector<std::size_t> parse_size_list(const std::string& text) {
    std::vector<std::size_t> out;
    for (const auto& part : split(text, ',')) {
        out.push_back(static_cast<std::size_t>(std::stoull(part)));
    }
    if (out.empty()) {
        throw std::runtime_error("empty integer list: " + text);
    }
    return out;
}

static int parse_int_arg(const std::string& s, const std::string& name) {
    char* end = nullptr;
    errno = 0;
    const long value = std::strtol(s.c_str(), &end, 10);

    if (errno != 0 || end == s.c_str() || *end != '\0') {
        throw std::runtime_error("Invalid integer for " + name + ": " + s);
    }
    if (value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
        throw std::runtime_error("Integer out of range for " + name + ": " + s);
    }
    return static_cast<int>(value);
}

std::vector<int> parse_int_list(const std::string& text, const std::string& name) {
    std::vector<int> out;
    for (const auto& part : split(text, ',')) {
        out.push_back(parse_int_arg(part, name));
    }
    if (out.empty()) {
        throw std::runtime_error("empty integer list: " + text);
    }
    return out;
}

Args parse_args(int argc, char** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto need_value = [&](const std::string& option) -> std::string {
            if (i + 1 >= argc) {
                throw std::runtime_error("missing value for " + option);
            }
            return argv[++i];
        };

        if (key == "--dims") {
            const auto dims = parse_size_list(need_value(key));
            if (dims.size() != 3) {
                throw std::runtime_error("--dims expects exactly three comma-separated dimensions");
            }
            args.dim0 = dims[0];
            args.dim1 = dims[1];
            args.dim2 = dims[2];
        } else if (key == "--nnz") {
            args.nnz = static_cast<std::size_t>(std::stoull(need_value(key)));
        } else if (key == "--ranks") {
            args.ranks = parse_size_list(need_value(key));
        } else if (key == "--modes") {
            args.modes = parse_int_list(need_value(key), key);
        } else if (key == "--threads") {
            args.threads = std::max(1, parse_int_arg(need_value(key), key));
        } else if (key == "--repeats") {
            args.repeats = std::max(1, parse_int_arg(need_value(key), key));
        } else if (key == "--seed") {
            args.seed = static_cast<std::uint64_t>(std::stoull(need_value(key)));
        } else if (key == "--output") {
            args.output = need_value(key);
        } else if (key == "--help" || key == "-h") {
            std::cout
                << "Usage: ttm_bench [options]\n"
                << "  --dims D0,D1,D2       Tensor dimensions, default 128,128,128\n"
                << "  --nnz N               Number of COO nonzeros, default 200000\n"
                << "  --ranks R0,R1,...     Factor ranks, default 16,32,64\n"
                << "  --modes M0,M1,...     Modes to test, default 0,1,2\n"
                << "  --threads T           Worker threads, default hardware_concurrency\n"
                << "  --repeats N           Repeats per case, default 3\n"
                << "  --seed S              RNG seed, default 42\n"
                << "  --output PATH         CSV output path, default results.csv\n";
            std::exit(0);
        } else {
            throw std::runtime_error("unknown option: " + key);
        }
    }

    if (args.dim0 == 0 || args.dim1 == 0 || args.dim2 == 0) {
        throw std::runtime_error("all dimensions must be positive");
    }
    if (args.nnz == 0) {
        throw std::runtime_error("--nnz must be positive");
    }
    for (const int mode : args.modes) {
        if (mode < 0 || mode > 2) {
            throw std::runtime_error("--modes values must be 0, 1, or 2");
        }
    }
    for (const std::size_t rank : args.ranks) {
        if (rank == 0) {
            throw std::runtime_error("--ranks values must be positive");
        }
    }
    return args;
}

std::size_t mode_dim(const Args& args, const int mode) {
    if (mode == 0) {
        return args.dim0;
    }
    if (mode == 1) {
        return args.dim1;
    }
    return args.dim2;
}

std::size_t output_rows_for_mode(const Args& args, const int mode) {
    if (mode == 0) {
        return args.dim1 * args.dim2;
    }
    if (mode == 1) {
        return args.dim0 * args.dim2;
    }
    return args.dim0 * args.dim1;
}

TensorCOO generate_tensor(const Args& args) {
    TensorCOO tensor;
    tensor.i0.resize(args.nnz);
    tensor.i1.resize(args.nnz);
    tensor.i2.resize(args.nnz);
    tensor.values64.resize(args.nnz);
    tensor.values32.resize(args.nnz);

    std::mt19937_64 rng(args.seed);
    std::uniform_int_distribution<std::uint32_t> d0(0, static_cast<std::uint32_t>(args.dim0 - 1));
    std::uniform_int_distribution<std::uint32_t> d1(0, static_cast<std::uint32_t>(args.dim1 - 1));
    std::uniform_int_distribution<std::uint32_t> d2(0, static_cast<std::uint32_t>(args.dim2 - 1));
    std::uniform_real_distribution<double> value_dist(-1.0, 1.0);

    for (std::size_t nz = 0; nz < args.nnz; ++nz) {
        tensor.i0[nz] = d0(rng);
        tensor.i1[nz] = d1(rng);
        tensor.i2[nz] = d2(rng);
        tensor.values64[nz] = value_dist(rng);
        tensor.values32[nz] = static_cast<float>(tensor.values64[nz]);
    }
    return tensor;
}

std::vector<double> generate_factor64(const std::size_t dim, const std::size_t rank, const std::uint64_t seed) {
    std::vector<double> factor(dim * rank);
    std::mt19937_64 rng(seed);
    std::normal_distribution<double> dist(0.0, 1.0 / std::sqrt(static_cast<double>(rank)));
    for (double& x : factor) {
        x = dist(rng);
    }
    return factor;
}

std::vector<float> to_float(const std::vector<double>& input) {
    std::vector<float> output(input.size());
    std::transform(input.begin(), input.end(), output.begin(), [](const double x) {
        return static_cast<float>(x);
    });
    return output;
}

std::uint64_t output_row(const TensorCOO& tensor, const Args& args, const int mode, const std::size_t nz) {
    if (mode == 0) {
        return static_cast<std::uint64_t>(tensor.i1[nz]) * args.dim2 + tensor.i2[nz];
    }
    if (mode == 1) {
        return static_cast<std::uint64_t>(tensor.i0[nz]) * args.dim2 + tensor.i2[nz];
    }
    return static_cast<std::uint64_t>(tensor.i0[nz]) * args.dim1 + tensor.i1[nz];
}

std::uint32_t mode_index(const TensorCOO& tensor, const int mode, const std::size_t nz) {
    if (mode == 0) {
        return tensor.i0[nz];
    }
    if (mode == 1) {
        return tensor.i1[nz];
    }
    return tensor.i2[nz];
}

Prepared prepare_entries(const TensorCOO& tensor, const Args& args, const int mode) {
    const auto start = Clock::now();

    Prepared prepared;
    prepared.entries.resize(tensor.values64.size());
    prepared.row_count = output_rows_for_mode(args, mode);

    for (std::size_t nz = 0; nz < tensor.values64.size(); ++nz) {
        prepared.entries[nz] = PreparedEntry{output_row(tensor, args, mode, nz), mode_index(tensor, mode, nz), nz};
    }

    std::sort(prepared.entries.begin(), prepared.entries.end(), [](const PreparedEntry& a, const PreparedEntry& b) {
        if (a.row != b.row) {
            return a.row < b.row;
        }
        return a.k < b.k;
    });

    prepared.rows.reserve(prepared.entries.size());
    prepared.row_starts.push_back(0);
    for (std::size_t i = 0; i < prepared.entries.size(); ++i) {
        if (i == 0 || prepared.entries[i].row != prepared.entries[i - 1].row) {
            if (i != 0) {
                prepared.row_starts.push_back(i);
            }
            prepared.rows.push_back(prepared.entries[i].row);
        }
    }
    prepared.row_starts.push_back(prepared.entries.size());

    prepared.format_prepare_ms = ms_since(start, Clock::now());
    return prepared;
}

std::vector<std::size_t> partition_rows(const std::size_t rows, const int threads) {
    const std::size_t worker_count = std::min<std::size_t>(std::max(1, threads), std::max<std::size_t>(1, rows));
    std::vector<std::size_t> cuts(worker_count + 1, 0);
    for (std::size_t t = 0; t <= worker_count; ++t) {
        cuts[t] = (rows * t) / worker_count;
    }
    return cuts;
}

template <typename ValueT, typename FactorT, typename AccumT>
void compute_rows(
    const ValueT* values,
    const Prepared& prepared,
    const FactorT* factor,
    const std::vector<std::size_t>& cuts,
    const std::size_t rank,
    std::vector<AccumT>& output) {

    auto worker = [&](const std::size_t begin_row_idx, const std::size_t end_row_idx) {
        for (std::size_t row_idx = begin_row_idx; row_idx < end_row_idx; ++row_idx) {
            const std::uint64_t row = prepared.rows[row_idx];
            AccumT* out = output.data() + static_cast<std::size_t>(row) * rank;
            const std::size_t begin = prepared.row_starts[row_idx];
            const std::size_t end = prepared.row_starts[row_idx + 1];

            for (std::size_t p = begin; p < end; ++p) {
                const PreparedEntry& entry = prepared.entries[p];
                const AccumT value = static_cast<AccumT>(values[entry.nz]);
                const FactorT* factor_row = factor + static_cast<std::size_t>(entry.k) * rank;
                for (std::size_t r = 0; r < rank; ++r) {
                    out[r] += value * static_cast<AccumT>(factor_row[r]);
                }
            }
        }
    };

    std::vector<std::thread> pool;
    pool.reserve(cuts.size() - 1);
    for (std::size_t t = 0; t + 1 < cuts.size(); ++t) {
        pool.emplace_back(worker, cuts[t], cuts[t + 1]);
    }
    for (auto& th : pool) {
        th.join();
    }
}

template <typename ValueT, typename FactorT, typename AccumT>
double compute_once(
    const ValueT* values,
    const Prepared& prepared,
    const FactorT* factor,
    const std::size_t rank,
    const int threads,
    std::vector<AccumT>& output) {

    std::fill(output.begin(), output.end(), AccumT{0});
    const std::vector<std::size_t> cuts = partition_rows(prepared.rows.size(), threads);
    const auto start = Clock::now();
    compute_rows<ValueT, FactorT, AccumT>(values, prepared, factor, cuts, rank, output);
    return ms_since(start, Clock::now());
}

template <typename SourceT>
double upcast_to_double(const std::vector<SourceT>& source, std::vector<double>& workspace) {
    const auto start = Clock::now();
    workspace.resize(source.size());
    std::transform(source.begin(), source.end(), workspace.begin(), [](const SourceT x) {
        return static_cast<double>(x);
    });
    return ms_since(start, Clock::now());
}

double relative_error(const std::vector<double>& reference, const std::vector<double>& candidate) {
    long double diff2 = 0.0;
    long double ref2 = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const long double d = static_cast<long double>(candidate[i]) - reference[i];
        diff2 += d * d;
        ref2 += static_cast<long double>(reference[i]) * reference[i];
    }
    if (ref2 == 0.0L) {
        return diff2 == 0.0L ? 0.0 : std::numeric_limits<double>::infinity();
    }
    return static_cast<double>(std::sqrt(diff2 / ref2));
}

double relative_error_float_output(const std::vector<double>& reference, const std::vector<float>& candidate) {
    long double diff2 = 0.0;
    long double ref2 = 0.0;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const long double d = static_cast<long double>(candidate[i]) - reference[i];
        diff2 += d * d;
        ref2 += static_cast<long double>(reference[i]) * reference[i];
    }
    if (ref2 == 0.0L) {
        return diff2 == 0.0L ? 0.0 : std::numeric_limits<double>::infinity();
    }
    return static_cast<double>(std::sqrt(diff2 / ref2));
}

Metrics run_variant(
    const Variant& variant,
    const TensorCOO& tensor,
    const Prepared& prepared,
    const Args& args,
    const int mode,
    const std::size_t rank,
    const std::vector<double>& factor64,
    const std::vector<float>& factor32,
    const std::vector<double>& reference) {

    const std::size_t output_size = prepared.row_count * rank;
    const std::size_t factor_dim = mode_dim(args, mode);
    const std::size_t value_type_bytes = variant.value_storage == ValueStorage::Fp64 ? sizeof(double) : sizeof(float);
    const std::size_t factor_type_bytes = variant.factor_storage == FactorStorage::Fp64 ? sizeof(double) : sizeof(float);
    const std::size_t output_type_bytes = variant.compute_type == ComputeType::Fp64 ? sizeof(double) : sizeof(float);

    Metrics m;
    m.format_prepare_ms = prepared.format_prepare_ms;
    m.index_storage_bytes = args.nnz * 3 * sizeof(std::uint32_t);
    m.value_storage_bytes = args.nnz * value_type_bytes;
    m.factor_storage_bytes = factor_dim * rank * factor_type_bytes;
    m.output_storage_bytes = output_size * output_type_bytes;
    m.index_logical_read_bytes = args.nnz * 3 * sizeof(std::uint32_t);
    m.value_logical_read_bytes = args.nnz * value_type_bytes;
    m.factor_logical_read_bytes = args.nnz * rank * factor_type_bytes;
    m.output_logical_write_bytes = std::max(m.output_storage_bytes, args.nnz * rank * output_type_bytes);

    if (variant.compute_type == ComputeType::Fp32) {
        std::vector<float> output(output_size);
        m.compute_ms = compute_once<double, float, float>(tensor.values64.data(), prepared, factor32.data(), rank, args.threads, output);
        m.rel_error = relative_error_float_output(reference, output);
    } else if (variant.value_storage == ValueStorage::Fp32) {
        std::vector<double> value_workspace;
        m.upcast_prepare_ms += upcast_to_double(tensor.values32, value_workspace);
        std::vector<double> output(output_size);
        m.compute_ms = compute_once<double, double, double>(value_workspace.data(), prepared, factor64.data(), rank, args.threads, output);
        m.rel_error = relative_error(reference, output);
    } else if (variant.factor_storage == FactorStorage::Fp32) {
        std::vector<double> factor_workspace;
        m.upcast_prepare_ms += upcast_to_double(factor32, factor_workspace);
        std::vector<double> output(output_size);
        m.compute_ms = compute_once<double, double, double>(tensor.values64.data(), prepared, factor_workspace.data(), rank, args.threads, output);
        m.rel_error = relative_error(reference, output);
    } else {
        std::vector<double> output(output_size);
        m.compute_ms = compute_once<double, double, double>(tensor.values64.data(), prepared, factor64.data(), rank, args.threads, output);
        m.rel_error = relative_error(reference, output);
    }

    m.total_ms = m.format_prepare_ms + m.upcast_prepare_ms + m.compute_ms;
    return m;
}

std::vector<double> compute_reference(
    const TensorCOO& tensor,
    const Prepared& prepared,
    const std::vector<double>& factor64,
    const std::size_t rank,
    const int threads) {

    std::vector<double> reference(prepared.row_count * rank);
    compute_once<double, double, double>(tensor.values64.data(), prepared, factor64.data(), rank, threads, reference);
    return reference;
}

void write_csv_header(std::ofstream& out) {
    out << "total_ms,format_prepare_ms,upcast_prepare_ms,compute_ms,"
        << "index_storage_bytes,value_storage_bytes,factor_storage_bytes,output_storage_bytes,"
        << "index_logical_read_bytes,value_logical_read_bytes,factor_logical_read_bytes,output_logical_write_bytes,"
        << "rel_error,rank,nnz,mode,thread_count,seed,variant,dim0,dim1,dim2,repeat\n";
}

void write_csv_row(
    std::ofstream& out,
    const Variant& variant,
    const Metrics& m,
    const Args& args,
    const std::size_t rank,
    const int mode,
    const int repeat) {

    out << std::setprecision(9) << m.total_ms << ','
        << m.format_prepare_ms << ','
        << m.upcast_prepare_ms << ','
        << m.compute_ms << ','
        << m.index_storage_bytes << ','
        << m.value_storage_bytes << ','
        << m.factor_storage_bytes << ','
        << m.output_storage_bytes << ','
        << m.index_logical_read_bytes << ','
        << m.value_logical_read_bytes << ','
        << m.factor_logical_read_bytes << ','
        << m.output_logical_write_bytes << ','
        << std::scientific << m.rel_error << std::defaultfloat << ','
        << rank << ','
        << args.nnz << ','
        << mode << ','
        << args.threads << ','
        << args.seed << ','
        << variant.name << ','
        << args.dim0 << ','
        << args.dim1 << ','
        << args.dim2 << ','
        << repeat << '\n';
}

} // namespace

int main(int argc, char** argv) {
    try {
        const Args args = parse_args(argc, argv);
        const TensorCOO tensor = generate_tensor(args);
        const std::vector<Variant> variants{
            {"factor_fp64_compute_fp64", FactorStorage::Fp64, ValueStorage::Fp64, ComputeType::Fp64},
            {"factor_fp32_compute_fp64", FactorStorage::Fp32, ValueStorage::Fp64, ComputeType::Fp64},
            {"factor_fp32_compute_fp32", FactorStorage::Fp32, ValueStorage::Fp64, ComputeType::Fp32},
            {"value_fp32_factor_fp64_compute_fp64", FactorStorage::Fp64, ValueStorage::Fp32, ComputeType::Fp64},
        };

        std::ofstream out(args.output);
        if (!out) {
            throw std::runtime_error("failed to open output file: " + args.output);
        }
        write_csv_header(out);

        for (const int mode : args.modes) {
            const Prepared prepared = prepare_entries(tensor, args, mode);
            for (const std::size_t rank : args.ranks) {
                const std::vector<double> factor64 = generate_factor64(mode_dim(args, mode), rank, args.seed + 1009 * (mode + 1) + rank);
                const std::vector<float> factor32 = to_float(factor64);

                for (int repeat = 0; repeat < args.repeats; ++repeat) {
                    const std::vector<double> reference = compute_reference(tensor, prepared, factor64, rank, args.threads);
                    for (const Variant& variant : variants) {
                        Metrics m = run_variant(variant, tensor, prepared, args, mode, rank, factor64, factor32, reference);
                        write_csv_row(out, variant, m, args, rank, mode, repeat);
                    }
                }
            }
        }

        std::cout << "wrote " << args.output << '\n';
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << '\n';
        return 1;
    }
}
