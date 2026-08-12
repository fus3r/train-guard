// train-guard replay kernel: a native port of trainguard.policy.PolicyEngine
// plus the time-weighted aggregation used by trainguard.simulation.
//
// The kernel exists for one purpose: evaluating many candidate policies
// against one recorded observation trace much faster than the Python
// reference, while producing bit-identical IEEE-754 doubles. The Python
// side owns all parsing, validation and reporting; this program receives
// pre-validated numbers and returns aggregates.
//
// Exactness contract with the Python reference:
//   - every float crosses the process boundary as a C99 hexadecimal
//     literal (Python float.hex() in, printf "%a" out), so no decimal
//     rounding is involved in either direction;
//   - every arithmetic expression mirrors the Python operation order,
//     and the build disables FMA contraction, so each intermediate
//     double is identical to the CPython result;
//   - durations are derived from integer epoch microseconds exactly as
//     datetime.timedelta.total_seconds does: (t2 - t1) / 1e6.
//
// Protocol (line-oriented ASCII on stdin/stdout, LF newlines):
//   in : TGK 1 <policies> <observations> <emit_actions> <hot_ref> <low_ref>
//        P <run_on_battery> <floor> <battery_band> <ac_band> <t_gentle>
//          <t_pause> <t_resume> <charge_cool_until> <t_charge_gentle>
//        O <t_us> <source> <percent|-> <temperature|-> <charging>
//   out: TGK 1 OK
//        R <full_s> <gentle_s> <stop_s> <run_s> <hot_degc_s> <low_batt_s>
//          <action_transitions> <decision_transitions> <fnv1a64>
//        A <one digit per observation>            (only when emit_actions)
//        END
//
// Exact bounded sensitivity uses a separate backwards-compatible protocol:
//   in : TGS 1 <policies> <observations> <hot_ref> <low_ref> <temp_hw> <charge_hw>
//        followed by the same P and O rows
//   out: TGS 1 OK
//        S <min_run> <max_run> <min_hot> <max_hot> <min_low> <max_low>
//          <stable_s> <ambiguous_s> <stable_samples> <ambiguous_samples>
//        END
// Sources: 0=ac 1=battery 2=no_battery. Bands and actions: 0=full
// 1=gentle 2=stop. A change to any of this is a protocol version bump.
//
// Usage: train-guard-kernel [threads]. The optional argument spreads
// *whole policies* over worker threads. Each policy's accumulation stays
// one sequential dependency chain evaluated by exactly one thread, rows
// land in a preallocated slot per policy index, and printing happens
// after every join, so the output is byte-identical for every thread
// count. Nothing about the wire protocol changes.

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <clocale>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <limits>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int kProtocolVersion = 1;

enum Action : std::uint8_t { kFull = 0, kGentle = 1, kStop = 2 };

// Mirrors trainguard.model.DecisionReason declaration order.
enum Reason : std::uint8_t {
  kThermalCooldown = 0,
  kThermalLimit = 1,
  kBatteryDisabled = 2,
  kBatteryFloor = 3,
  kBatteryPolicy = 4,
  kWarmCharging = 5,
  kWarmAc = 6,
  kNoBattery = 7,
  kAcPolicy = 8,
};

enum Source : std::uint8_t { kAc = 0, kBattery = 1, kNoBattery2 = 2 };

struct Policy {
  bool run_on_battery;
  double battery_floor_pct;
  std::uint8_t battery_band;
  std::uint8_t ac_band;
  double temp_gentle_c;
  double temp_pause_c;
  double temp_resume_c;
  double charge_cool_until_pct;
  double temp_charge_gentle_c;
};

struct Obs {
  std::int64_t t_us;
  std::uint8_t source;
  bool has_percent;
  double percent;
  bool has_temperature;
  double temperature;
  bool charging;
  double duration_s;  // 0.0 for the final observation
};

struct Decision {
  std::uint8_t action;
  std::uint8_t reason;
};

// Exact port of trainguard.policy.PolicyEngine.decide. `cooling` is the
// per-replay hysteresis state and starts false.
inline Decision decide(const Policy& p, const Obs& o, bool& cooling) {
  if (cooling) {
    if (o.has_temperature && o.temperature <= p.temp_resume_c) {
      cooling = false;
    } else {
      return {kStop, kThermalCooldown};
    }
  }
  if (o.has_temperature && o.temperature >= p.temp_pause_c) {
    cooling = true;
    return {kStop, kThermalLimit};
  }
  if (o.source == kBattery) {
    if (!p.run_on_battery) {
      return {kStop, kBatteryDisabled};
    }
    if (o.has_percent && o.percent <= p.battery_floor_pct) {
      return {kStop, kBatteryFloor};
    }
    return {p.battery_band, kBatteryPolicy};
  }
  if (o.charging && o.has_percent && o.percent < p.charge_cool_until_pct &&
      o.has_temperature && o.temperature >= p.temp_charge_gentle_c) {
    return {kGentle, kWarmCharging};
  }
  if (o.has_temperature && o.temperature >= p.temp_gentle_c) {
    return {kGentle, kWarmAc};
  }
  if (o.source == kNoBattery2) {
    return {p.ac_band, kNoBattery};
  }
  return {p.ac_band, kAcPolicy};
}

struct Row {
  double action_seconds[3];
  double run_seconds;
  double hot_degc_seconds;
  double low_battery_seconds;
  std::int64_t action_transitions;
  std::int64_t decision_transitions;
  std::uint64_t checksum;
};

struct ObjectiveEnvelope {
  bool reachable;
  double minimum_run;
  double maximum_run;
  double minimum_hot;
  double maximum_hot;
  double minimum_low;
  double maximum_low;
};

struct SensitivityRow {
  double minimum_run;
  double maximum_run;
  double minimum_hot;
  double maximum_hot;
  double minimum_low;
  double maximum_low;
  double stable_seconds;
  double ambiguous_seconds;
  std::int64_t stable_samples;
  std::int64_t ambiguous_samples;
};

// This function is the exactness contract: the same operations in the
// same order as trainguard.sweep._python_rows. It is shared by the
// sequential and threaded paths so they cannot diverge; a policy's whole
// accumulation always runs on exactly one thread.
Row evaluate(const Policy& policy, const std::vector<Obs>& observations, double hot_ref,
             double low_ref, std::string* actions_out) {
  Row row{};
  row.checksum = 0xcbf29ce484222325ULL;
  bool cooling = false;
  int previous_action = -1;
  int previous_reason = -1;
  for (std::size_t index = 0; index < observations.size(); ++index) {
    const Obs& obs = observations[index];
    const Decision decision = decide(policy, obs, cooling);
    const double duration = obs.duration_s;
    row.action_seconds[decision.action] += duration;
    if (decision.action != kStop) {
      row.run_seconds += duration;
      if (obs.has_temperature) {
        const double excess = obs.temperature - hot_ref;
        if (excess > 0.0) {
          row.hot_degc_seconds += excess * duration;
        }
      }
      if (obs.source == kBattery && obs.has_percent && obs.percent <= low_ref) {
        row.low_battery_seconds += duration;
      }
    }
    if (previous_action >= 0 && decision.action != previous_action) {
      ++row.action_transitions;
    }
    if (previous_action >= 0 &&
        (decision.action != previous_action || decision.reason != previous_reason)) {
      ++row.decision_transitions;
    }
    previous_action = decision.action;
    previous_reason = decision.reason;
    row.checksum = (row.checksum ^ decision.action) * 0x100000001b3ULL;
    row.checksum = (row.checksum ^ decision.reason) * 0x100000001b3ULL;
    if (actions_out != nullptr) {
      (*actions_out)[index] = static_cast<char>('0' + decision.action);
    }
  }
  return row;
}

std::vector<double> representatives(double value, double half_width,
                                    const std::vector<double>& thresholds, double lower,
                                    double upper) {
  const double low = std::max(lower, value - half_width);
  const double high = std::min(upper, value + half_width);
  std::vector<double> points{low, high};
  points.reserve(2 + thresholds.size() * 3);
  for (const double threshold : thresholds) {
    if (low <= threshold && threshold <= high) {
      points.push_back(threshold);
    }
    const double below = std::nextafter(threshold, -std::numeric_limits<double>::infinity());
    const double above = std::nextafter(threshold, std::numeric_limits<double>::infinity());
    if (low <= below && below <= high) {
      points.push_back(below);
    }
    if (low <= above && above <= high) {
      points.push_back(above);
    }
  }
  std::sort(points.begin(), points.end());
  points.erase(std::unique(points.begin(), points.end()), points.end());
  return points;
}

void update_envelope(ObjectiveEnvelope& target, const ObjectiveEnvelope& source,
                     const Decision& decision, double duration, double hot_rate,
                     double low_rate) {
  const double run_increment = decision.action == kStop ? 0.0 : duration;
  const double minimum_run = source.minimum_run + run_increment;
  const double maximum_run = source.maximum_run + run_increment;
  const double minimum_hot = source.minimum_hot + hot_rate * duration;
  const double maximum_hot = source.maximum_hot + hot_rate * duration;
  const double minimum_low = source.minimum_low + low_rate * duration;
  const double maximum_low = source.maximum_low + low_rate * duration;
  if (!target.reachable) {
    target = {true, minimum_run, maximum_run, minimum_hot, maximum_hot, minimum_low,
              maximum_low};
    return;
  }
  target.minimum_run = std::min(target.minimum_run, minimum_run);
  target.maximum_run = std::max(target.maximum_run, maximum_run);
  target.minimum_hot = std::min(target.minimum_hot, minimum_hot);
  target.maximum_hot = std::max(target.maximum_hot, maximum_hot);
  target.minimum_low = std::min(target.minimum_low, minimum_low);
  target.maximum_low = std::max(target.maximum_low, maximum_low);
}

SensitivityRow evaluate_sensitivity(const Policy& policy, const std::vector<Obs>& observations,
                                    double hot_ref, double low_ref, double temperature_half_width,
                                    double charge_half_width) {
  ObjectiveEnvelope reachable[2]{};
  reachable[0] = {true, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
  SensitivityRow result{};

  const std::vector<double> temperature_thresholds{
      policy.temp_resume_c, policy.temp_pause_c, policy.temp_charge_gentle_c,
      policy.temp_gentle_c, hot_ref};
  const std::vector<double> percent_thresholds{
      policy.battery_floor_pct, policy.charge_cool_until_pct, low_ref};

  for (const Obs& recorded : observations) {
    const std::vector<double> temperatures =
        recorded.has_temperature
            ? representatives(recorded.temperature, temperature_half_width,
                              temperature_thresholds, -100.0, 200.0)
            : std::vector<double>{0.0};
    const std::vector<double> percentages =
        recorded.has_percent
            ? representatives(recorded.percent, charge_half_width, percent_thresholds, 0.0, 100.0)
            : std::vector<double>{0.0};

    ObjectiveEnvelope next[2]{};
    unsigned action_mask = 0;
    for (int cooling_index = 0; cooling_index < 2; ++cooling_index) {
      if (!reachable[cooling_index].reachable) {
        continue;
      }
      for (const double temperature : temperatures) {
        for (const double percent : percentages) {
          Obs perturbed = recorded;
          if (recorded.has_temperature) {
            perturbed.temperature = temperature;
          }
          if (recorded.has_percent) {
            perturbed.percent = percent;
          }
          bool cooling = cooling_index != 0;
          const Decision decision = decide(policy, perturbed, cooling);
          action_mask |= 1U << decision.action;
          const bool running = decision.action != kStop;
          double hot_rate = 0.0;
          if (running && perturbed.has_temperature) {
            const double excess = perturbed.temperature - hot_ref;
            if (excess > 0.0) {
              hot_rate = excess;
            }
          }
          const double low_rate =
              running && perturbed.source == kBattery && perturbed.has_percent &&
                      perturbed.percent <= low_ref
                  ? 1.0
                  : 0.0;
          update_envelope(next[cooling ? 1 : 0], reachable[cooling_index], decision,
                          recorded.duration_s, hot_rate, low_rate);
        }
      }
    }

    if (action_mask != 0 && (action_mask & (action_mask - 1U)) == 0) {
      result.stable_seconds += recorded.duration_s;
      ++result.stable_samples;
    } else {
      result.ambiguous_seconds += recorded.duration_s;
      ++result.ambiguous_samples;
    }
    reachable[0] = next[0];
    reachable[1] = next[1];
  }

  bool first = true;
  for (const ObjectiveEnvelope& envelope : reachable) {
    if (!envelope.reachable) {
      continue;
    }
    if (first) {
      result.minimum_run = envelope.minimum_run;
      result.maximum_run = envelope.maximum_run;
      result.minimum_hot = envelope.minimum_hot;
      result.maximum_hot = envelope.maximum_hot;
      result.minimum_low = envelope.minimum_low;
      result.maximum_low = envelope.maximum_low;
      first = false;
    } else {
      result.minimum_run = std::min(result.minimum_run, envelope.minimum_run);
      result.maximum_run = std::max(result.maximum_run, envelope.maximum_run);
      result.minimum_hot = std::min(result.minimum_hot, envelope.minimum_hot);
      result.maximum_hot = std::max(result.maximum_hot, envelope.maximum_hot);
      result.minimum_low = std::min(result.minimum_low, envelope.minimum_low);
      result.maximum_low = std::max(result.maximum_low, envelope.maximum_low);
    }
  }
  return result;
}

void print_row(const Row& row) {
  std::printf("R %a %a %a %a %a %a %" PRId64 " %" PRId64 " %016" PRIx64 "\n",
              row.action_seconds[kFull], row.action_seconds[kGentle], row.action_seconds[kStop],
              row.run_seconds, row.hot_degc_seconds, row.low_battery_seconds,
              row.action_transitions, row.decision_transitions, row.checksum);
}

void print_sensitivity_row(const SensitivityRow& row) {
  std::printf("S %a %a %a %a %a %a %a %a %" PRId64 " %" PRId64 "\n", row.minimum_run,
              row.maximum_run, row.minimum_hot, row.maximum_hot, row.minimum_low,
              row.maximum_low, row.stable_seconds, row.ambiguous_seconds, row.stable_samples,
              row.ambiguous_samples);
}

[[noreturn]] void fail(const char* message) {
  std::fprintf(stderr, "train-guard-kernel: %s\n", message);
  std::exit(2);
}

bool next_token(std::string& token) {
  token.clear();
  int c = std::getchar();
  while (c == ' ' || c == '\n' || c == '\r' || c == '\t') {
    c = std::getchar();
  }
  if (c == EOF) {
    return false;
  }
  while (c != EOF && c != ' ' && c != '\n' && c != '\r' && c != '\t') {
    token.push_back(static_cast<char>(c));
    c = std::getchar();
  }
  return true;
}

std::string require_token(const char* what) {
  std::string token;
  if (!next_token(token)) {
    std::string message = "unexpected end of input while reading ";
    message += what;
    fail(message.c_str());
  }
  return token;
}

std::int64_t parse_int(const std::string& token, const char* what) {
  errno = 0;
  char* end = nullptr;
  const long long value = std::strtoll(token.c_str(), &end, 10);
  if (errno != 0 || end == token.c_str() || *end != '\0') {
    std::string message = "invalid integer for ";
    message += what;
    fail(message.c_str());
  }
  return static_cast<std::int64_t>(value);
}

double parse_double(const std::string& token, const char* what) {
  errno = 0;
  char* end = nullptr;
  const double value = std::strtod(token.c_str(), &end);
  if (errno != 0 || end == token.c_str() || *end != '\0') {
    std::string message = "invalid number for ";
    message += what;
    fail(message.c_str());
  }
  return value;
}

bool parse_flag(const std::string& token, const char* what) {
  if (token == "0") {
    return false;
  }
  if (token == "1") {
    return true;
  }
  std::string message = "invalid flag for ";
  message += what;
  fail(message.c_str());
}

std::uint8_t parse_band(const std::string& token, const char* what) {
  if (token == "0") {
    return kFull;
  }
  if (token == "1") {
    return kGentle;
  }
  std::string message = "invalid band for ";
  message += what;
  fail(message.c_str());
}

}  // namespace

int main(int argc, char** argv) {
  std::setlocale(LC_ALL, "C");

  long requested_threads = 1;
  if (argc > 2) {
    fail("usage: train-guard-kernel [threads]");
  }
  if (argc == 2) {
    errno = 0;
    char* end = nullptr;
    requested_threads = std::strtol(argv[1], &end, 10);
    if (errno != 0 || end == argv[1] || *end != '\0' || requested_threads < 1 ||
        requested_threads > 4096) {
      fail("threads must be an integer between 1 and 4096");
    }
  }

  const std::string magic = require_token("magic");
  const bool sensitivity_mode = magic == "TGS";
  if (magic != "TGK" && !sensitivity_mode) {
    fail("input does not start with the TGK or TGS magic token");
  }
  if (parse_int(require_token("protocol version"), "protocol version") != kProtocolVersion) {
    std::fprintf(stderr, "train-guard-kernel: unsupported protocol version\n");
    return 3;
  }
  const std::int64_t policy_count = parse_int(require_token("policy count"), "policy count");
  const std::int64_t obs_count = parse_int(require_token("observation count"), "observation count");
  const bool emit_actions =
      sensitivity_mode ? false : parse_flag(require_token("emit_actions"), "emit_actions");
  const double hot_ref = parse_double(require_token("hot_ref"), "hot_ref");
  const double low_ref = parse_double(require_token("low_ref"), "low_ref");
  const double temperature_half_width =
      sensitivity_mode ? parse_double(require_token("temperature_half_width"),
                                      "temperature_half_width")
                       : 0.0;
  const double charge_half_width =
      sensitivity_mode
          ? parse_double(require_token("charge_half_width"), "charge_half_width")
          : 0.0;
  if (policy_count < 1 || obs_count < 1) {
    fail("policy and observation counts must be at least 1");
  }
  if (!std::isfinite(hot_ref) || !std::isfinite(low_ref) ||
      !std::isfinite(temperature_half_width) || !std::isfinite(charge_half_width) ||
      temperature_half_width < 0.0 || charge_half_width < 0.0) {
    fail("references must be finite and uncertainty half-widths finite and non-negative");
  }
  if (sensitivity_mode &&
      (hot_ref < -100.0 || hot_ref > 200.0 || low_ref < 0.0 || low_ref > 100.0 ||
       temperature_half_width > 300.0 || charge_half_width > 100.0)) {
    fail("sensitivity references or half-widths are outside the supported domain");
  }

  std::vector<Policy> policies;
  policies.reserve(static_cast<std::size_t>(policy_count));
  for (std::int64_t index = 0; index < policy_count; ++index) {
    if (require_token("policy row") != "P") {
      fail("expected a P row");
    }
    Policy policy;
    policy.run_on_battery = parse_flag(require_token("run_on_battery"), "run_on_battery");
    policy.battery_floor_pct = parse_double(require_token("battery_floor_pct"), "battery_floor_pct");
    policy.battery_band = parse_band(require_token("battery_band"), "battery_band");
    policy.ac_band = parse_band(require_token("ac_band"), "ac_band");
    policy.temp_gentle_c = parse_double(require_token("temp_gentle_c"), "temp_gentle_c");
    policy.temp_pause_c = parse_double(require_token("temp_pause_c"), "temp_pause_c");
    policy.temp_resume_c = parse_double(require_token("temp_resume_c"), "temp_resume_c");
    policy.charge_cool_until_pct =
        parse_double(require_token("charge_cool_until_pct"), "charge_cool_until_pct");
    policy.temp_charge_gentle_c =
        parse_double(require_token("temp_charge_gentle_c"), "temp_charge_gentle_c");
    policies.push_back(policy);
  }

  std::vector<Obs> observations;
  observations.reserve(static_cast<std::size_t>(obs_count));
  for (std::int64_t index = 0; index < obs_count; ++index) {
    if (require_token("observation row") != "O") {
      fail("expected an O row");
    }
    Obs obs;
    obs.t_us = parse_int(require_token("t_us"), "t_us");
    const std::int64_t source = parse_int(require_token("source"), "source");
    if (source < 0 || source > 2) {
      fail("source must be 0, 1 or 2");
    }
    obs.source = static_cast<std::uint8_t>(source);
    const std::string percent = require_token("percent");
    obs.has_percent = percent != "-";
    obs.percent = obs.has_percent ? parse_double(percent, "percent") : 0.0;
    const std::string temperature = require_token("temperature");
    obs.has_temperature = temperature != "-";
    obs.temperature = obs.has_temperature ? parse_double(temperature, "temperature") : 0.0;
    obs.charging = parse_flag(require_token("charging"), "charging");
    obs.duration_s = 0.0;
    if (index > 0) {
      const Obs& previous = observations.back();
      if (obs.t_us < previous.t_us) {
        fail("observations must be ordered by t_us");
      }
      observations[static_cast<std::size_t>(index - 1)].duration_s =
          static_cast<double>(obs.t_us - previous.t_us) / 1e6;
    }
    observations.push_back(obs);
  }
  std::string trailing;
  if (next_token(trailing)) {
    fail("unexpected trailing input");
  }

  std::printf("%s %d OK\n", sensitivity_mode ? "TGS" : "TGK", kProtocolVersion);

  // emit_actions responses carry one byte per observation per policy, so
  // that path streams sequentially instead of buffering every string.
  std::size_t worker_count = static_cast<std::size_t>(requested_threads);
  if (emit_actions) {
    worker_count = 1;
  }
  worker_count = std::min(worker_count, policies.size());

  if (sensitivity_mode && worker_count <= 1) {
    for (const Policy& policy : policies) {
      print_sensitivity_row(evaluate_sensitivity(policy, observations, hot_ref, low_ref,
                                                 temperature_half_width, charge_half_width));
    }
  } else if (sensitivity_mode) {
    std::vector<SensitivityRow> rows(policies.size());
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    const std::size_t chunk = (policies.size() + worker_count - 1) / worker_count;
    try {
      for (std::size_t worker = 0; worker < worker_count; ++worker) {
        const std::size_t begin = worker * chunk;
        const std::size_t end = std::min(begin + chunk, policies.size());
        if (begin >= end) {
          break;
        }
        workers.emplace_back([&policies, &observations, &rows, hot_ref, low_ref,
                              temperature_half_width, charge_half_width, begin, end] {
          for (std::size_t index = begin; index < end; ++index) {
            rows[index] = evaluate_sensitivity(policies[index], observations, hot_ref, low_ref,
                                               temperature_half_width, charge_half_width);
          }
        });
      }
    } catch (...) {
      for (std::thread& worker : workers) {
        worker.join();
      }
      fail("could not start worker threads");
    }
    for (std::thread& worker : workers) {
      worker.join();
    }
    for (const SensitivityRow& row : rows) {
      print_sensitivity_row(row);
    }
  } else if (worker_count <= 1) {
    std::string actions;
    if (emit_actions) {
      actions.resize(static_cast<std::size_t>(obs_count));
    }
    for (const Policy& policy : policies) {
      const Row row =
          evaluate(policy, observations, hot_ref, low_ref, emit_actions ? &actions : nullptr);
      print_row(row);
      if (emit_actions) {
        std::printf("A %s\n", actions.c_str());
      }
    }
  } else {
    std::vector<Row> rows(policies.size());
    std::vector<std::thread> workers;
    workers.reserve(worker_count);
    const std::size_t chunk = (policies.size() + worker_count - 1) / worker_count;
    try {
      for (std::size_t worker = 0; worker < worker_count; ++worker) {
        const std::size_t begin = worker * chunk;
        const std::size_t end = std::min(begin + chunk, policies.size());
        if (begin >= end) {
          break;
        }
        workers.emplace_back([&policies, &observations, &rows, hot_ref, low_ref, begin, end] {
          for (std::size_t index = begin; index < end; ++index) {
            rows[index] = evaluate(policies[index], observations, hot_ref, low_ref, nullptr);
          }
        });
      }
    } catch (...) {
      for (std::thread& worker : workers) {
        worker.join();
      }
      fail("could not start worker threads");
    }
    for (std::thread& worker : workers) {
      worker.join();
    }
    for (const Row& row : rows) {
      print_row(row);
    }
  }
  std::printf("END\n");
  if (std::fflush(stdout) != 0) {
    fail("failed to flush results");
  }
  return 0;
}
