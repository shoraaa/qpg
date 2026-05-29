#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

struct SampleResult {
  std::vector<int32_t> choices;
  std::vector<float> energies;
  std::vector<int32_t> trace_starts;
  std::vector<int32_t> trace_sources;
  std::vector<int32_t> trace_chosen_edges;
};

static float transition_log_weight(float pheromone, float heuristic, float prior,
                                   float alpha, float beta, float gamma) {
  constexpr float eps = 1e-12f;
  return alpha * std::log(std::max(pheromone, eps)) +
         beta * std::log(std::max(heuristic, eps)) + gamma * prior;
}

static float residual_heuristic(int32_t target, const std::vector<int32_t> &counts,
                                const float *weights, const float *lengths,
                                int32_t n_bio_nodes, int32_t end_index) {
  if (target == end_index) {
    float remaining = 0.0f;
    for (int32_t i = 0; i < n_bio_nodes; ++i) {
      remaining += std::max(0.0f, weights[i] - static_cast<float>(counts[i]));
    }
    return remaining <= 0.0f ? 1.0f : 0.01f;
  }
  int32_t node_index = target / 2;
  if (node_index < 0 || node_index >= n_bio_nodes) {
    return 0.01f;
  }
  float residual = std::max(0.0f, weights[node_index] - static_cast<float>(counts[node_index]));
  return 0.01f + residual * (1.0f + std::sqrt(1.0f + lengths[node_index]) / 10.0f);
}

static float path_energy(const std::vector<int32_t> &choices,
                         const float *q_data, int32_t q_rows,
                         float offset, int32_t states_per_time) {
  float energy = offset;
  for (int32_t t = 0; t < static_cast<int32_t>(choices.size()); ++t) {
    int32_t row = t * states_per_time + choices[t];
    if (row < 0 || row >= q_rows) {
      throw std::runtime_error("active QUBO row out of bounds");
    }
    for (int32_t s = 0; s < static_cast<int32_t>(choices.size()); ++s) {
      int32_t col = s * states_per_time + choices[s];
      energy += q_data[row * q_rows + col];
    }
  }
  return energy;
}

py::dict sample_batch(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> offsets_in,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> targets_in,
    py::array_t<float, py::array::c_style | py::array::forcecast> pheromone_in,
    py::array_t<float, py::array::c_style | py::array::forcecast> heuristic_in,
    py::array_t<float, py::array::c_style | py::array::forcecast> prior_in,
    py::array_t<float, py::array::c_style | py::array::forcecast> weights_in,
    py::array_t<float, py::array::c_style | py::array::forcecast> lengths_in,
    py::array_t<float, py::array::c_style | py::array::forcecast> q_in,
    float offset, int32_t states_per_time, int32_t horizon, int32_t n_ants,
    int32_t start_source, int32_t end_index, float alpha, float beta, float gamma,
    uint64_t seed, bool parallel_traced) {

  auto offsets = offsets_in.unchecked<1>();
  auto targets = targets_in.unchecked<1>();
  auto pheromone = pheromone_in.unchecked<1>();
  auto heuristic = heuristic_in.unchecked<1>();
  auto prior = prior_in.unchecked<1>();
  auto weights = weights_in.unchecked<1>();
  auto lengths = lengths_in.unchecked<1>();
  auto q = q_in.unchecked<2>();

  const int32_t n_sources = static_cast<int32_t>(offsets.shape(0)) - 1;
  const int32_t n_edges = static_cast<int32_t>(targets.shape(0));
  if (n_sources <= 0 || n_edges <= 0) {
    throw std::runtime_error("empty ACO edge graph");
  }
  if (pheromone.shape(0) != n_edges || heuristic.shape(0) != n_edges ||
      prior.shape(0) != n_edges) {
    throw std::runtime_error("edge array length mismatch");
  }
  if (q.shape(0) != q.shape(1)) {
    throw std::runtime_error("Q must be square");
  }
  if (weights.shape(0) != lengths.shape(0)) {
    throw std::runtime_error("weights/lengths length mismatch");
  }
  if (start_source < 0 || start_source >= n_sources) {
    throw std::runtime_error("start_source out of bounds");
  }
  if (end_index < 0 || end_index >= n_sources) {
    throw std::runtime_error("end_index out of bounds");
  }

  SampleResult result;
  result.choices.resize(static_cast<size_t>(n_ants) * horizon);
  result.energies.resize(n_ants);
  result.trace_starts.resize(static_cast<size_t>(n_ants) + 1);
  result.trace_sources.resize(static_cast<size_t>(n_ants) * horizon);
  result.trace_chosen_edges.resize(static_cast<size_t>(n_ants) * horizon);
  for (int32_t ant = 0; ant <= n_ants; ++ant) {
    result.trace_starts[ant] = ant * horizon;
  }

  auto sample_ant = [&](int32_t ant, std::mt19937_64 &rng) {
    std::vector<float> log_weights;
    std::vector<float> probs;
    std::vector<int32_t> choices;
    std::vector<int32_t> counts(static_cast<size_t>(weights.shape(0)));
    choices.reserve(horizon);
    choices.clear();
    std::fill(counts.begin(), counts.end(), 0);
    int32_t source = start_source;
    for (int32_t depth = 0; depth < horizon; ++depth) {
      int32_t begin = offsets(source);
      int32_t end = offsets(source + 1);
      if (begin < 0 || end < begin || end > n_edges) {
        throw std::runtime_error("invalid edge offsets");
      }
      if (begin == end) {
        throw std::runtime_error("source has no outgoing edges");
      }

      log_weights.clear();
      probs.clear();
      float max_log = -std::numeric_limits<float>::infinity();
      for (int32_t edge = begin; edge < end; ++edge) {
        float dyn_heuristic = residual_heuristic(
            targets(edge), counts, weights.data(0), lengths.data(0),
            static_cast<int32_t>(weights.shape(0)), end_index);
        float lw = transition_log_weight(pheromone(edge), dyn_heuristic,
                                         prior(edge), alpha, beta, gamma);
        log_weights.push_back(lw);
        max_log = std::max(max_log, lw);
      }
      float total = 0.0f;
      for (float lw : log_weights) {
        float p = std::exp(lw - max_log);
        probs.push_back(p);
        total += p;
      }
      float r = std::generate_canonical<float, 24>(rng) * total;
      float acc = 0.0f;
      int32_t local_pick = static_cast<int32_t>(probs.size()) - 1;
      for (int32_t i = 0; i < static_cast<int32_t>(probs.size()); ++i) {
        acc += probs[i];
        if (acc >= r) {
          local_pick = i;
          break;
        }
      }
      int32_t edge_id = begin + local_pick;
      int32_t target = targets(edge_id);

      const size_t trace_index = static_cast<size_t>(ant) * horizon + depth;
      result.trace_sources[trace_index] = source;
      result.trace_chosen_edges[trace_index] = edge_id;
      choices.push_back(target);
      result.choices[static_cast<size_t>(ant) * horizon + depth] = target;
      if (target != end_index) {
        int32_t node_index = target / 2;
        if (node_index >= 0 && node_index < static_cast<int32_t>(counts.size())) {
          counts[node_index] += 1;
        }
      }
      source = target;
    }
    result.energies[ant] =
        path_energy(choices, q.data(0, 0), static_cast<int32_t>(q.shape(0)),
                    offset, states_per_time);
  };

  if (parallel_traced) {
    py::gil_scoped_release release;
#pragma omp parallel for schedule(static)
    for (int32_t ant = 0; ant < n_ants; ++ant) {
      std::mt19937_64 rng(seed + 0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(ant + 1));
      sample_ant(ant, rng);
    }
  } else {
    py::gil_scoped_release release;
    std::mt19937_64 rng(seed);
    for (int32_t ant = 0; ant < n_ants; ++ant) {
      sample_ant(ant, rng);
    }
  }

  py::array_t<int32_t> choices_out({n_ants, horizon});
  std::copy(result.choices.begin(), result.choices.end(),
            choices_out.mutable_data());
  py::array_t<float> energies_out(n_ants);
  std::copy(result.energies.begin(), result.energies.end(),
            energies_out.mutable_data());
  py::array_t<int32_t> starts_out(result.trace_starts.size());
  std::copy(result.trace_starts.begin(), result.trace_starts.end(),
            starts_out.mutable_data());
  py::array_t<int32_t> sources_out(result.trace_sources.size());
  std::copy(result.trace_sources.begin(), result.trace_sources.end(),
            sources_out.mutable_data());
  py::array_t<int32_t> chosen_edges_out(result.trace_chosen_edges.size());
  std::copy(result.trace_chosen_edges.begin(), result.trace_chosen_edges.end(),
            chosen_edges_out.mutable_data());

  py::dict out;
  out["choices"] = choices_out;
  out["energies"] = energies_out;
  out["trace_starts"] = starts_out;
  out["trace_sources"] = sources_out;
  out["trace_chosen_edges"] = chosen_edges_out;
  return out;
}

py::dict sample_batch_traces(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> offsets_in,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> targets_in,
    py::array_t<float, py::array::c_style | py::array::forcecast> pheromone_in,
    py::array_t<float, py::array::c_style | py::array::forcecast> heuristic_in,
    py::array_t<float, py::array::c_style | py::array::forcecast> prior_in,
    py::array_t<float, py::array::c_style | py::array::forcecast> weights_in,
    py::array_t<float, py::array::c_style | py::array::forcecast> lengths_in,
    int32_t horizon, int32_t n_ants, int32_t start_source, int32_t end_index,
    float alpha, float beta, float gamma, uint64_t seed, bool parallel_traced) {

  auto offsets = offsets_in.unchecked<1>();
  auto targets = targets_in.unchecked<1>();
  auto pheromone = pheromone_in.unchecked<1>();
  auto heuristic = heuristic_in.unchecked<1>();
  auto prior = prior_in.unchecked<1>();
  auto weights = weights_in.unchecked<1>();
  auto lengths = lengths_in.unchecked<1>();

  const int32_t n_sources = static_cast<int32_t>(offsets.shape(0)) - 1;
  const int32_t n_edges = static_cast<int32_t>(targets.shape(0));
  if (n_sources <= 0 || n_edges <= 0) {
    throw std::runtime_error("empty ACO edge graph");
  }
  if (pheromone.shape(0) != n_edges || heuristic.shape(0) != n_edges ||
      prior.shape(0) != n_edges) {
    throw std::runtime_error("edge array length mismatch");
  }
  if (weights.shape(0) != lengths.shape(0)) {
    throw std::runtime_error("weights/lengths length mismatch");
  }
  if (start_source < 0 || start_source >= n_sources) {
    throw std::runtime_error("start_source out of bounds");
  }
  if (end_index < 0 || end_index >= n_sources) {
    throw std::runtime_error("end_index out of bounds");
  }

  SampleResult result;
  result.choices.resize(static_cast<size_t>(n_ants) * horizon);
  result.trace_starts.resize(static_cast<size_t>(n_ants) + 1);
  result.trace_sources.resize(static_cast<size_t>(n_ants) * horizon);
  result.trace_chosen_edges.resize(static_cast<size_t>(n_ants) * horizon);
  for (int32_t ant = 0; ant <= n_ants; ++ant) {
    result.trace_starts[ant] = ant * horizon;
  }

  auto sample_ant = [&](int32_t ant, std::mt19937_64 &rng) {
    std::vector<float> log_weights;
    std::vector<float> probs;
    std::vector<int32_t> counts(static_cast<size_t>(weights.shape(0)));
    std::fill(counts.begin(), counts.end(), 0);
    int32_t source = start_source;
    for (int32_t depth = 0; depth < horizon; ++depth) {
      int32_t begin = offsets(source);
      int32_t end = offsets(source + 1);
      if (begin < 0 || end < begin || end > n_edges) {
        throw std::runtime_error("invalid edge offsets");
      }
      if (begin == end) {
        throw std::runtime_error("source has no outgoing edges");
      }

      log_weights.clear();
      probs.clear();
      float max_log = -std::numeric_limits<float>::infinity();
      for (int32_t edge = begin; edge < end; ++edge) {
        float dyn_heuristic = residual_heuristic(
            targets(edge), counts, weights.data(0), lengths.data(0),
            static_cast<int32_t>(weights.shape(0)), end_index);
        float lw = transition_log_weight(pheromone(edge), dyn_heuristic,
                                         prior(edge), alpha, beta, gamma);
        log_weights.push_back(lw);
        max_log = std::max(max_log, lw);
      }
      float total = 0.0f;
      for (float lw : log_weights) {
        float p = std::exp(lw - max_log);
        probs.push_back(p);
        total += p;
      }
      float r = std::generate_canonical<float, 24>(rng) * total;
      float acc = 0.0f;
      int32_t local_pick = static_cast<int32_t>(probs.size()) - 1;
      for (int32_t i = 0; i < static_cast<int32_t>(probs.size()); ++i) {
        acc += probs[i];
        if (acc >= r) {
          local_pick = i;
          break;
        }
      }
      int32_t edge_id = begin + local_pick;
      int32_t target = targets(edge_id);

      const size_t trace_index = static_cast<size_t>(ant) * horizon + depth;
      result.trace_sources[trace_index] = source;
      result.trace_chosen_edges[trace_index] = edge_id;
      result.choices[trace_index] = target;
      if (target != end_index) {
        int32_t node_index = target / 2;
        if (node_index >= 0 && node_index < static_cast<int32_t>(counts.size())) {
          counts[node_index] += 1;
        }
      }
      source = target;
    }
  };

  if (parallel_traced) {
    py::gil_scoped_release release;
#pragma omp parallel for schedule(static)
    for (int32_t ant = 0; ant < n_ants; ++ant) {
      std::mt19937_64 rng(seed + 0x9e3779b97f4a7c15ULL * static_cast<uint64_t>(ant + 1));
      sample_ant(ant, rng);
    }
  } else {
    py::gil_scoped_release release;
    std::mt19937_64 rng(seed);
    for (int32_t ant = 0; ant < n_ants; ++ant) {
      sample_ant(ant, rng);
    }
  }

  py::array_t<int32_t> choices_out({n_ants, horizon});
  std::copy(result.choices.begin(), result.choices.end(), choices_out.mutable_data());
  py::array_t<int32_t> starts_out(result.trace_starts.size());
  std::copy(result.trace_starts.begin(), result.trace_starts.end(), starts_out.mutable_data());
  py::array_t<int32_t> sources_out(result.trace_sources.size());
  std::copy(result.trace_sources.begin(), result.trace_sources.end(), sources_out.mutable_data());
  py::array_t<int32_t> chosen_edges_out(result.trace_chosen_edges.size());
  std::copy(result.trace_chosen_edges.begin(), result.trace_chosen_edges.end(), chosen_edges_out.mutable_data());

  py::dict out;
  out["choices"] = choices_out;
  out["trace_starts"] = starts_out;
  out["trace_sources"] = sources_out;
  out["trace_chosen_edges"] = chosen_edges_out;
  return out;
}

PYBIND11_MODULE(qpg_aco_cpp, m) {
  m.doc() = "QPG DyNACO-style ACO sampler";
  m.def("set_num_threads", [](int n_threads) {
    if (n_threads <= 0) {
      throw std::runtime_error("n_threads must be > 0");
    }
    omp_set_num_threads(n_threads);
  }, py::arg("n_threads"), "Set OpenMP thread count");
  m.def("get_max_threads", []() { return omp_get_max_threads(); });
  m.def("get_num_procs", []() { return omp_get_num_procs(); });
  m.def("set_dynamic", [](bool enabled) { omp_set_dynamic(enabled ? 1 : 0); },
        py::arg("enabled"), "Enable or disable OpenMP dynamic teams");
  m.def("get_dynamic", []() { return omp_get_dynamic() != 0; });
  m.def("sample_batch", &sample_batch, py::arg("offsets"), py::arg("targets"),
        py::arg("pheromone"), py::arg("heuristic"), py::arg("prior"),
        py::arg("weights"), py::arg("lengths"), py::arg("Q"),
        py::arg("offset"), py::arg("states_per_time"), py::arg("horizon"),
        py::arg("n_ants"), py::arg("start_source"), py::arg("end_index"),
        py::arg("alpha"), py::arg("beta"), py::arg("gamma"), py::arg("seed"),
        py::arg("parallel_traced") = true);
  m.def("sample_batch_traces", &sample_batch_traces,
        py::arg("offsets"), py::arg("targets"), py::arg("pheromone"),
        py::arg("heuristic"), py::arg("prior"), py::arg("weights"),
        py::arg("lengths"), py::arg("horizon"), py::arg("n_ants"),
        py::arg("start_source"), py::arg("end_index"), py::arg("alpha"),
        py::arg("beta"), py::arg("gamma"), py::arg("seed"),
        py::arg("parallel_traced") = true);
}
