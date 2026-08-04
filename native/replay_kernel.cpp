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

void print_row(const Row& row) {
  std::printf("R %a %a %a %a %a %a %" PRId64 " %" PRId64 " %016" PRIx64 "\n",
              row.action_seconds[kFull], row.action_seconds[kGentle], row.action_seconds[kStop],
              row.run_seconds, row.hot_degc_seconds, row.low_battery_seconds,
              row.action_transitions, row.decision_transitions, row.checksum);
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

  if (require_token("magic") != "TGK") {
    fail("input does not start with the TGK magic token");
  }
  if (parse_int(require_token("protocol version"), "protocol version") != kProtocolVersion) {
    std::fprintf(stderr, "train-guard-kernel: unsupported protocol version\n");
    return 3;
  }
  const std::int64_t policy_count = parse_int(require_token("policy count"), "policy count");
  const std::int64_t obs_count = parse_int(require_token("observation count"), "observation count");
  const bool emit_actions = parse_flag(require_token("emit_actions"), "emit_actions");
  const double hot_ref = parse_double(require_token("hot_ref"), "hot_ref");
  const double low_ref = parse_double(require_token("low_ref"), "low_ref");
  if (policy_count < 1 || obs_count < 1) {
    fail("policy and observation counts must be at least 1");
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

  std::printf("TGK %d OK\n", kProtocolVersion);

  // emit_actions responses carry one byte per observation per policy, so
  // that path streams sequentially instead of buffering every string.
  std::size_t worker_count = static_cast<std::size_t>(requested_threads);
  if (emit_actions) {
    worker_count = 1;
  }
  worker_count = std::min(worker_count, policies.size());

  if (worker_count <= 1) {
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
