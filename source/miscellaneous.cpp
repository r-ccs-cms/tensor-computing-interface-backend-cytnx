#include "tci/miscellaneous.h"

#include <cytnx.hpp>

namespace tci {

  // Context management for CytnxContextHandle
  template <> void create_context(CytnxContextHandle& ctx) {
    // Use GPU 0 if available, otherwise fall back to CPU
    if (cytnx::Device.Ngpus > 0) {
      ctx.set_value(cytnx::Device.cuda);
    } else {
      ctx.set_value(cytnx::Device.cpu);
    }
  }

  void create_context(CytnxContextHandle& ctx, int gpu_id) {
    if (cytnx::Device.Ngpus == 0) {
      throw std::runtime_error("No CUDA devices available");
    }
    if (gpu_id < 0) {
      throw std::out_of_range("GPU ID must be non-negative");
    }
    if (gpu_id >= cytnx::Device.Ngpus) {
      throw std::out_of_range("GPU ID exceeds available devices (available: "
                              + std::to_string(cytnx::Device.Ngpus) + ")");
    }
    ctx.set_value(cytnx::Device.cuda + gpu_id);
  }

  template <> void destroy_context(CytnxContextHandle& ctx) {
    // No specific cleanup needed
  }

  // All cytnx::Tensor specializations removed
  // CytnxTensor<ElemT> implementations are in cytnx_typed_tensor_impl.h

}  // namespace tci
