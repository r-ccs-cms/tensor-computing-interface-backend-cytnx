#include "tci/miscellaneous.h"

#include <cytnx.hpp>

namespace tci {

  // Context management for CytnxContextHandle
  template <> void create_context(CytnxContextHandle& ctx) {
    // Initialize Cytnx device context (device ID)
    ctx.set_value(cytnx::Device.cpu);  // Default to CPU (-1)
  }

  template <> void destroy_context(CytnxContextHandle& ctx) {
    // No specific cleanup needed for Cytnx device context
    // Cytnx handles this automatically
  }

  // All cytnx::Tensor specializations removed
  // CytnxTensor<ElemT> implementations are in cytnx_typed_tensor_impl.h

}  // namespace tci
