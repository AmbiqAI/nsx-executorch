/*
 * SPDX-FileCopyrightText: 2026 Ambiq
 *
 * SPDX-License-Identifier: Apache-2.0
 *
 * HeliaAotBackend: runs partitions that helia-torch delegated to heliaAOT.
 *
 * The compiled code is not in the PTE. heliaAOT generated one NSX module per
 * partition, which the firmware links; the module's generated registration
 * file adds it to the registry in includes-api/nsx_helia_aot.h. The delegate
 * blob (aot/nsx_cortex_m/helia_aot/blob.py) names the module and records the
 * delegate's tensor contract, checked at init and at every execute.
 *
 * execute() copies each input tensor into the module's input buffer, runs the
 * module, and copies its outputs back. The copies are the delegate's tensor
 * sizes (e.g. 490 + 12 bytes for KWS).
 */

#include "nsx_helia_aot.h"

#include <cstdint>
#include <cstring>

#include <executorch/runtime/backend/interface.h>
#include <executorch/runtime/core/error.h>
#include <executorch/runtime/core/evalue.h>
#include <executorch/runtime/core/memory_allocator.h>
#include <executorch/runtime/platform/log.h>

namespace {

using executorch::aten::ScalarType;
using executorch::aten::Tensor;
using executorch::runtime::ArrayRef;
using executorch::runtime::Backend;
using executorch::runtime::BackendExecutionContext;
using executorch::runtime::BackendInitContext;
using executorch::runtime::BackendInterface;
using executorch::runtime::CompileSpec;
using executorch::runtime::DelegateHandle;
using executorch::runtime::Error;
using executorch::runtime::EValue;
using executorch::runtime::FreeableBuffer;
using executorch::runtime::MemoryAllocator;
using executorch::runtime::Result;
using executorch::runtime::Span;

constexpr char kBackendId[] = "HeliaAotBackend";
constexpr char kMagic[4] = {'H', 'A', 'O', 'T'};
constexpr std::uint16_t kBlobVersion = 1;
constexpr std::size_t kMaxModuleName = 63;

struct TensorContract {
  std::uint8_t scalar_type;
  std::uint32_t nbytes;
};

struct Handle {
  const nsx_helia_aot_model_t* model;
  std::uint32_t num_inputs;
  std::uint32_t num_outputs;
  TensorContract* tensors;  // inputs then outputs
};

std::uint16_t read_u16(const std::uint8_t* p) {
  return static_cast<std::uint16_t>(p[0] | (p[1] << 8));
}

std::uint32_t read_u32(const std::uint8_t* p) {
  return static_cast<std::uint32_t>(p[0]) | (static_cast<std::uint32_t>(p[1]) << 8) |
         (static_cast<std::uint32_t>(p[2]) << 16) | (static_cast<std::uint32_t>(p[3]) << 24);
}

class HeliaAotBackend final : public BackendInterface {
 public:
  bool is_available() const override { return true; }

  Result<DelegateHandle*> init(
      BackendInitContext& context,
      FreeableBuffer* processed,
      ArrayRef<CompileSpec> /*compile_specs*/) const override {
    const auto* blob = static_cast<const std::uint8_t*>(processed->data());
    const std::size_t size = processed->size();
    if (size < 16 || std::memcmp(blob, kMagic, sizeof(kMagic)) != 0 ||
        read_u16(blob + 4) != kBlobVersion) {
      ET_LOG(Error, "HeliaAotBackend: not a version %u delegate blob", kBlobVersion);
      return Error::DelegateInvalidCompatibility;
    }
    const std::uint16_t name_len = read_u16(blob + 6);
    std::size_t offset = 8 + name_len;
    offset += (4 - offset % 4) % 4;
    if (name_len == 0 || name_len > kMaxModuleName || offset + 8 > size) {
      ET_LOG(Error, "HeliaAotBackend: malformed delegate blob");
      return Error::InvalidProgram;
    }
    char name[kMaxModuleName + 1];
    std::memcpy(name, blob + 8, name_len);
    name[name_len] = '\0';
    const std::uint32_t num_inputs = read_u32(blob + offset);
    const std::uint32_t num_outputs = read_u32(blob + offset + 4);
    offset += 8;
    const std::uint32_t num_tensors = num_inputs + num_outputs;
    if (offset + 8u * num_tensors > size) {
      ET_LOG(Error, "HeliaAotBackend: truncated tensor contract");
      return Error::InvalidProgram;
    }

    const nsx_helia_aot_model_t* model = nsx_helia_aot_find(name);
    if (model == nullptr) {
      ET_LOG(Error,
             "HeliaAotBackend: module %s is not linked; add the heliaAOT module helia-torch "
             "generated for this PTE to the firmware",
             name);
      return Error::NotFound;
    }
    if (model->num_inputs != num_inputs || model->num_outputs != num_outputs) {
      ET_LOG(Error, "HeliaAotBackend: module %s has %u/%u inputs/outputs, the PTE expects %u/%u", name,
             static_cast<unsigned>(model->num_inputs), static_cast<unsigned>(model->num_outputs),
             static_cast<unsigned>(num_inputs), static_cast<unsigned>(num_outputs));
      return Error::DelegateInvalidCompatibility;
    }

    MemoryAllocator* allocator = context.get_runtime_allocator();
    Handle* handle = allocator->allocateInstance<Handle>();
    TensorContract* tensors = allocator->allocateList<TensorContract>(num_tensors > 0 ? num_tensors : 1);
    if (handle == nullptr || tensors == nullptr) {
      return Error::MemoryAllocationFailed;
    }
    for (std::uint32_t i = 0; i < num_tensors; ++i) {
      tensors[i].scalar_type = blob[offset + 8 * i];
      tensors[i].nbytes = read_u32(blob + offset + 8 * i + 4);
    }
    handle->model = model;
    handle->num_inputs = num_inputs;
    handle->num_outputs = num_outputs;
    handle->tensors = tensors;
    processed->Free();  // everything needed now lives in the handle

    if (model->init() != 0) {
      ET_LOG(Error, "HeliaAotBackend: %s init failed", name);
      return Error::DelegateInvalidHandle;
    }
    return handle;
  }

  Error execute(
      BackendExecutionContext& /*context*/,
      DelegateHandle* opaque,
      Span<EValue*> args) const override {
    auto* handle = static_cast<Handle*>(opaque);
    const std::uint32_t num_inputs = handle->num_inputs;
    const std::uint32_t num_outputs = handle->num_outputs;
    if (args.size() != num_inputs + num_outputs) {
      ET_LOG(Error, "HeliaAotBackend: %u arguments, expected %u", static_cast<unsigned>(args.size()),
             static_cast<unsigned>(num_inputs + num_outputs));
      return Error::InvalidArgument;
    }
    for (std::uint32_t i = 0; i < num_inputs + num_outputs; ++i) {
      if (!args[i]->isTensor()) {
        return Error::InvalidArgument;
      }
      const Tensor& t = args[i]->toTensor();
      const TensorContract& c = handle->tensors[i];
      if (static_cast<std::uint8_t>(t.scalar_type()) != c.scalar_type || t.nbytes() != c.nbytes) {
        ET_LOG(Error, "HeliaAotBackend: %s tensor %u is %u bytes of type %d, expected %u bytes of type %u",
               i < num_inputs ? "input" : "output", static_cast<unsigned>(i < num_inputs ? i : i - num_inputs),
               static_cast<unsigned>(t.nbytes()), static_cast<int>(t.scalar_type()),
               static_cast<unsigned>(c.nbytes), static_cast<unsigned>(c.scalar_type));
        return Error::InvalidArgument;
      }
    }
    const nsx_helia_aot_model_t* model = handle->model;
    for (std::uint32_t i = 0; i < num_inputs; ++i) {
      void* dst = model->input(i);
      if (dst == nullptr) {
        return Error::Internal;
      }
      std::memcpy(dst, args[i]->toTensor().const_data_ptr(), handle->tensors[i].nbytes);
    }
    if (model->run() != 0) {
      ET_LOG(Error, "HeliaAotBackend: %s run failed", model->name);
      return Error::Internal;
    }
    for (std::uint32_t o = 0; o < num_outputs; ++o) {
      const void* src = model->output(o);
      if (src == nullptr) {
        return Error::Internal;
      }
      std::memcpy(args[num_inputs + o]->toTensor().mutable_data_ptr(), src,
                  handle->tensors[num_inputs + o].nbytes);
    }
    return Error::Ok;
  }
};

HeliaAotBackend g_backend;
bool g_registered = false;

}  // namespace

namespace nsx::executorch::detail {

// Called from run_once(): explicit registration keeps the backend linked, which a static
// initializer in a static library would not guarantee.
void register_helia_aot_backend() {
  if (g_registered) {
    return;
  }
  const Backend backend{kBackendId, &g_backend};
  if (::executorch::runtime::register_backend(backend) == Error::Ok) {
    g_registered = true;
  }
}

}  // namespace nsx::executorch::detail
