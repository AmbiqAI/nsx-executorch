#include "nsx_executorch.h"

#include <cstring>

#include "am_mcu_apollo.h"
#include "nsx_core.h"
#include <executorch/extension/data_loader/buffer_data_loader.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/event_tracer.h>
#include <executorch/runtime/core/hierarchical_allocator.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/executor/memory_manager.h>
#include <executorch/runtime/executor/method.h>
#include <executorch/runtime/executor/program.h>
#include <executorch/runtime/platform/platform.h>
#include <executorch/runtime/platform/runtime.h>

namespace {

using executorch::runtime::Error;

std::uint32_t error_code(Error error) {
  return static_cast<std::uint32_t>(error);
}

nsx::executorch::RunResult fail(
    nsx::executorch::Stage stage,
    Error error = Error::Ok,
    std::size_t input_count = 0,
    std::size_t output_count = 0,
    std::size_t planned_bytes_required = 0,
    std::uint32_t execution_cycles = 0) {
  return {stage, error_code(error), input_count, output_count,
          planned_bytes_required, execution_cycles};
}

class OperatorEventTracer final : public executorch::runtime::EventTracer {
 public:
  explicit OperatorEventTracer(
      const nsx::executorch::ProfilingCallbacks* callbacks)
      : callbacks_(callbacks) {}

  void create_event_block(const char*) override {}

  executorch::runtime::EventTracerEntry start_profiling(
      const char* name,
      executorch::runtime::ChainID chain_id =
          executorch::runtime::kUnsetChainId,
      executorch::runtime::DebugHandle debug_handle =
          executorch::runtime::kUnsetDebugHandle) override {
    executorch::runtime::EventTracerEntry entry{};
    entry.event_id = -1;
    if (callbacks_ == nullptr || callbacks_->begin_operator == nullptr ||
        callbacks_->end_operator == nullptr) {
      return entry;
    }

    nsx::executorch::OperatorKind kind;
    if (std::strcmp(name, "OPERATOR_CALL") == 0) {
      kind = nsx::executorch::OperatorKind::kKernel;
    } else if (std::strcmp(name, "DELEGATE_CALL") == 0) {
      kind = nsx::executorch::OperatorKind::kDelegate;
    } else {
      return entry;
    }

    if (chain_id == executorch::runtime::kUnsetChainId) {
      chain_id = current_chain_id();
    }
    if (debug_handle == executorch::runtime::kUnsetDebugHandle) {
      debug_handle = current_debug_handle();
    }
    const nsx::executorch::OperatorEvent event = {
        kind, static_cast<std::int32_t>(chain_id),
        static_cast<std::uint32_t>(debug_handle)};
    entry.event_id = callbacks_->begin_operator(callbacks_->user_data, event);
    entry.chain_id = chain_id;
    entry.debug_handle = debug_handle;
    return entry;
  }

  void end_profiling(executorch::runtime::EventTracerEntry entry) override {
    if (callbacks_ != nullptr && callbacks_->end_operator != nullptr &&
        entry.event_id >= 0) {
      callbacks_->end_operator(
          callbacks_->user_data, static_cast<std::uint32_t>(entry.event_id));
    }
  }

  void track_allocation(executorch::runtime::AllocatorID, std::size_t) override {}
  executorch::runtime::AllocatorID track_allocator(const char*) override {
    return 0;
  }
  executorch::runtime::EventTracerEntry start_profiling_delegate(
      const char*, executorch::runtime::DelegateDebugIntId) override {
    return {};
  }
  void end_profiling_delegate(
      executorch::runtime::EventTracerEntry, const void*, std::size_t) override {}
  void log_profiling_delegate(
      const char*,
      executorch::runtime::DelegateDebugIntId,
      et_timestamp_t,
      et_timestamp_t,
      const void*,
      std::size_t) override {}
  executorch::runtime::Result<bool> log_evalue(
      const executorch::runtime::EValue&,
      executorch::runtime::LoggedEValueType) override {
    return true;
  }
  executorch::runtime::Result<bool> log_intermediate_output_delegate(
      const char*,
      executorch::runtime::DelegateDebugIntId,
      const executorch::aten::Tensor&) override {
    return true;
  }
  executorch::runtime::Result<bool> log_intermediate_output_delegate(
      const char*,
      executorch::runtime::DelegateDebugIntId,
      executorch::aten::ArrayRef<executorch::aten::Tensor>) override {
    return true;
  }
  executorch::runtime::Result<bool> log_intermediate_output_delegate(
      const char*, executorch::runtime::DelegateDebugIntId, const int&) override {
    return true;
  }
  executorch::runtime::Result<bool> log_intermediate_output_delegate(
      const char*, executorch::runtime::DelegateDebugIntId, const bool&) override {
    return true;
  }
  executorch::runtime::Result<bool> log_intermediate_output_delegate(
      const char*, executorch::runtime::DelegateDebugIntId, const double&) override {
    return true;
  }
  void set_delegation_intermediate_output_filter(
      executorch::runtime::EventTracerFilterBase*) override {}

 private:
  const nsx::executorch::ProfilingCallbacks* callbacks_;
};

}  // namespace

extern "C" void et_pal_init(void) {
  // Cortex-M55 implements DWT CYCCNT as an alias of the PMU cycle counter.
  // Enable the SoC debug clock and the architectural PMU explicitly so cycle
  // measurement never depends on a debugger having configured them first.
  (void)am_hal_debug_enable();
  CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
  ARM_PMU_Enable();
  ARM_PMU_CYCCNT_Reset();
  ARM_PMU_CNTR_Enable(PMU_CNTENSET_CCNTR_ENABLE_Msk);
  DWT->CYCCNT = 0;
  DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
  __DSB();
  __ISB();
}

extern "C" void et_pal_abort(void) {
  __disable_irq();
  while (true) {
    __BKPT(0);
  }
}

extern "C" et_timestamp_t et_pal_current_ticks(void) {
  return DWT->CYCCNT;
}

extern "C" et_tick_ratio_t et_pal_ticks_to_ns_multiplier(void) {
  extern std::uint32_t SystemCoreClock;
  return {1000000000ULL, SystemCoreClock == 0 ? 1ULL : SystemCoreClock};
}

extern "C" void et_pal_emit_log_message(
    et_timestamp_t,
    et_pal_log_level_t,
    const char*,
    const char*,
    std::size_t,
    const char*,
    std::size_t) {}

extern "C" void* et_pal_allocate(std::size_t) { return nullptr; }
extern "C" void et_pal_free(void*) {}

namespace nsx::executorch {

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
    const ProfilingCallbacks* profiling) {
  using ::executorch::extension::BufferDataLoader;
  using ::executorch::runtime::HierarchicalAllocator;
  using ::executorch::runtime::MemoryAllocator;
  using ::executorch::runtime::MemoryManager;
  using ::executorch::runtime::Method;
  using ::executorch::runtime::Program;
  using ::executorch::runtime::Span;

  if (program_data == nullptr || program_size == 0 || method_arena.data == nullptr ||
      method_arena.size == 0 || planned_arena.data == nullptr ||
      temporary_arena.data == nullptr || (input_count != 0 && inputs == nullptr) ||
      (output_count != 0 && outputs == nullptr)) {
    return fail(Stage::kInvalidArgument);
  }
  if (profiling != nullptr &&
      ((profiling->begin_operator == nullptr) !=
       (profiling->end_operator == nullptr))) {
    return fail(Stage::kInvalidArgument);
  }
#if !defined(ET_EVENT_TRACER_ENABLED)
  if (profiling != nullptr && profiling->begin_operator != nullptr) {
    return fail(Stage::kProfilingUnavailable, Error::NotSupported);
  }
#endif

  ::executorch::runtime::runtime_init();
  BufferDataLoader loader(program_data, program_size);
  auto program_result = Program::load(&loader);
  if (!program_result.ok()) {
    return fail(Stage::kLoadProgram, program_result.error());
  }
  Program program = std::move(*program_result);

  auto method_name_result = program.get_method_name(0);
  if (!method_name_result.ok()) {
    return fail(Stage::kInspectMethod, method_name_result.error());
  }
  auto meta_result = program.method_meta(*method_name_result);
  if (!meta_result.ok()) {
    return fail(Stage::kInspectMethod, meta_result.error());
  }
  const auto meta = *meta_result;
  if (meta.num_memory_planned_buffers() != 1) {
    return fail(Stage::kMemoryPlan, Error::NotSupported);
  }
  auto planned_size_result = meta.memory_planned_buffer_size(0);
  if (!planned_size_result.ok()) {
    return fail(Stage::kMemoryPlan, planned_size_result.error());
  }
  const std::size_t planned_bytes =
      static_cast<std::size_t>(*planned_size_result);
  if (planned_arena.size < planned_bytes) {
    return fail(Stage::kMemoryPlan, Error::MemoryAllocationFailed, 0, 0,
                planned_bytes);
  }

  MemoryAllocator method_allocator(method_arena.size,
                                   static_cast<std::uint8_t*>(method_arena.data));
  MemoryAllocator temporary_allocator(
      temporary_arena.size, static_cast<std::uint8_t*>(temporary_arena.data));
  Span<std::uint8_t> planned_span(
      static_cast<std::uint8_t*>(planned_arena.data), planned_arena.size);
  Span<std::uint8_t> planned_spans[] = {planned_span};
  HierarchicalAllocator planned_memory(
      Span<Span<std::uint8_t>>(planned_spans, 1));
  MemoryManager memory_manager(
      &method_allocator, &planned_memory, &temporary_allocator);

  OperatorEventTracer event_tracer(profiling);
  auto method_result = program.load_method(
      *method_name_result,
      &memory_manager,
      (profiling != nullptr && profiling->begin_operator != nullptr)
          ? &event_tracer
          : nullptr);
  if (!method_result.ok()) {
    return fail(Stage::kLoadMethod, method_result.error(), 0, 0, planned_bytes);
  }
  Method method = std::move(*method_result);
  if (method.inputs_size() != input_count || method.outputs_size() != output_count) {
    return fail(Stage::kInvalidArgument, Error::InvalidArgument,
                method.inputs_size(), method.outputs_size(), planned_bytes);
  }

  for (std::size_t index = 0; index < input_count; ++index) {
    auto& value = method.mutable_input(index);
    if (!value.isTensor()) {
      return fail(Stage::kInput, Error::InvalidType, input_count, output_count,
                  planned_bytes);
    }
    auto tensor = value.toTensor();
    if (inputs[index].data == nullptr || inputs[index].size < tensor.nbytes() ||
        tensor.mutable_data_ptr() == nullptr) {
      return fail(Stage::kInput, Error::InvalidArgument, input_count,
                  output_count, planned_bytes);
    }
    std::memcpy(tensor.mutable_data_ptr(), inputs[index].data, tensor.nbytes());
  }

  const std::uint32_t execute_start = DWT->CYCCNT;
  const Error execute_error = method.execute();
  const std::uint32_t execution_cycles = DWT->CYCCNT - execute_start;
  if (execute_error != Error::Ok) {
    return fail(Stage::kExecute, execute_error, input_count, output_count,
                planned_bytes, execution_cycles);
  }

  for (std::size_t index = 0; index < output_count; ++index) {
    const auto& value = method.get_output(index);
    if (!value.isTensor()) {
      return fail(Stage::kOutput, Error::InvalidType, input_count, output_count,
                  planned_bytes);
    }
    const auto tensor = value.toTensor();
    if (outputs[index].data == nullptr || outputs[index].size < tensor.nbytes()) {
      outputs[index].size = tensor.nbytes();
      return fail(Stage::kOutput, Error::InvalidArgument, input_count,
                  output_count, planned_bytes);
    }
    std::memcpy(outputs[index].data, tensor.const_data_ptr(), tensor.nbytes());
    outputs[index].size = tensor.nbytes();
  }
  return {Stage::kOk, error_code(Error::Ok), input_count, output_count,
          planned_bytes, execution_cycles};
}

RunResult run_once(
    const void* program_data,
    std::size_t program_size,
    Buffer method_arena,
    Buffer planned_arena,
    Buffer temporary_arena,
    const Buffer* inputs,
    std::size_t input_count,
    Buffer* outputs,
    std::size_t output_count) {
  return run_once_profiled(
      program_data, program_size, method_arena, planned_arena, temporary_arena,
      inputs, input_count, outputs, output_count, nullptr);
}

const char* stage_name(Stage stage) {
  switch (stage) {
    case Stage::kOk: return "ok";
    case Stage::kInvalidArgument: return "invalid_argument";
    case Stage::kLoadProgram: return "load_program";
    case Stage::kInspectMethod: return "inspect_method";
    case Stage::kMemoryPlan: return "memory_plan";
    case Stage::kLoadMethod: return "load_method";
    case Stage::kInput: return "input";
    case Stage::kExecute: return "execute";
    case Stage::kOutput: return "output";
    case Stage::kProfilingUnavailable: return "profiling_unavailable";
  }
  return "unknown";
}

}  // namespace nsx::executorch
