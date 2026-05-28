#pragma once

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct Args {
  std::string output = "results/mkl_fp64_profile.csv";
  int dim0 = 256;
  int dim1 = 256;
  int dim2 = 256;
  std::int64_t nnz = 1000000;
  std::vector<int> ranks{64, 128, 256};
  std::vector<int> modes{0, 1, 2};
  int threads = 1;
  int repeats = 5;
  std::uint64_t seed = 1;
};

inline std::vector<int> parse_int_list(const std::string& text) {
  std::vector<int> values;
  std::stringstream ss(text);
  std::string item;
  while (std::getline(ss, item, ',')) {
    if (!item.empty()) {
      values.push_back(std::stoi(item));
    }
  }
  return values;
}

inline Args parse_args(int argc, char** argv) {
  Args args;
  for (int n = 1; n < argc; ++n) {
    const std::string key(argv[n]);
    auto require_value = [&](const char* name) -> std::string {
      if (n + 1 >= argc) {
        throw std::invalid_argument(std::string("missing value for ") + name);
      }
      return argv[++n];
    };

    if (key == "--output") {
      args.output = require_value("--output");
    } else if (key == "--dims") {
      const auto dims = parse_int_list(require_value("--dims"));
      if (dims.size() != 3) {
        throw std::invalid_argument("--dims expects I,J,K");
      }
      args.dim0 = dims[0];
      args.dim1 = dims[1];
      args.dim2 = dims[2];
    } else if (key == "--nnz") {
      args.nnz = std::stoll(require_value("--nnz"));
    } else if (key == "--ranks") {
      args.ranks = parse_int_list(require_value("--ranks"));
    } else if (key == "--modes") {
      args.modes = parse_int_list(require_value("--modes"));
    } else if (key == "--threads") {
      args.threads = std::stoi(require_value("--threads"));
    } else if (key == "--repeats") {
      args.repeats = std::stoi(require_value("--repeats"));
    } else if (key == "--seed") {
      args.seed = static_cast<std::uint64_t>(std::stoull(require_value("--seed")));
    } else if (key == "--help" || key == "-h") {
      throw std::invalid_argument(
          "usage: ttm_profile --output path --dims I,J,K --nnz N "
          "--ranks r1,r2,r3 --modes 0,1,2 --threads T --repeats R --seed S");
    } else {
      throw std::invalid_argument("unknown argument: " + key);
    }
  }

  if (args.dim0 <= 0 || args.dim1 <= 0 || args.dim2 <= 0 || args.nnz < 0 ||
      args.threads <= 0 || args.repeats <= 0 || args.ranks.empty() ||
      args.modes.empty()) {
    throw std::invalid_argument("invalid non-positive dimensions, nnz, threads, repeats, ranks, or modes");
  }
  for (const int mode : args.modes) {
    if (mode < 0 || mode > 2) {
      throw std::invalid_argument("mode must be 0, 1, or 2");
    }
  }
  for (const int rank : args.ranks) {
    if (rank <= 0) {
      throw std::invalid_argument("rank must be positive");
    }
  }
  return args;
}
