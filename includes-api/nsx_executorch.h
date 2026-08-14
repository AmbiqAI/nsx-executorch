#pragma once

#include <cstddef>
#include <cstdint>

namespace nsx::executorch {

struct Buffer {
  void* data;
  std::size_t size;
};

enum class Stage : std::uint8_t {
  kOk = 0,
  kInvalidArgument,
  kLoadProgram,
  kInspectMethod,
  kMemoryPlan,
  kLoadMethod,
  kInput,
  kExecute,
  kOutput,
  kProfilingUnavailable,
};

enum class OperatorKind : std::uint8_t {
  kKernel = 0,
  kDelegate,
};

struct OperatorEvent {
  OperatorKind kind;
  std::int32_t chain_index;
  std::uint32_t instruction_index;
};

using BeginOperatorCallback = std::uint32_t (*)(
    void* user_data, const OperatorEvent& event);
using EndOperatorCallback = void (*)(void* user_data, std::uint32_t handle);

// Optional per-operator hooks backed by ExecuTorch's EventTracer. Both
// callbacks must be supplied. The begin/end callbacks bracket each operator,
// making them suitable for resetting and sampling target PMU counters without
// exposing ExecuTorch types to callers.
struct ProfilingCallbacks {
  void* user_data;
  BeginOperatorCallback begin_operator;
  EndOperatorCallback end_operator;
};

struct RunResult {
  Stage stage;
  std::uint32_t executorch_error;
  std::size_t input_count;
  std::size_t output_count;
  std::size_t planned_bytes_required;
  // DWT cycles spent in Method::execute(). This excludes program/method load,
  // input/output copies, and caller-side reporting. With profiling callbacks
  // installed it includes their instrumentation overhead.
  std::uint32_t execution_cycles;

  constexpr bool ok() const { return stage == Stage::kOk; }
};

// Loads the first method in a baked .pte, copies tensor inputs into its
// memory-planned arena, executes once, and copies tensor outputs out. The
// runtime performs no heap allocation: every arena and I/O buffer is owned by
// the caller and must remain valid for the duration of this call.
RunResult run_once(
    const void* program_data,
    std::size_t program_size,
    Buffer method_arena,
    Buffer planned_arena,
    Buffer temporary_arena,
    const Buffer* inputs,
    std::size_t input_count,
    Buffer* outputs,
    std::size_t output_count);

// Equivalent to run_once(), with optional per-operator callbacks. Build the
// nsx-executorch module with NSX_EXECUTORCH_ENABLE_PROFILING=ON before passing
// non-null callbacks.
RunResult run_once_profiled(
    const void* program_data,
    std::size_t program_size,
    Buffer method_arena,
    Buffer planned_arena,
    Buffer temporary_arena,
    const Buffer* inputs,
    std::size_t input_count,
    Buffer* outputs,
    std::size_t output_count,
    const ProfilingCallbacks* profiling);

const char* stage_name(Stage stage);

}  // namespace nsx::executorch
