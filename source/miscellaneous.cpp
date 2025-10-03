#include "tci/miscellaneous.h"

#include <cytnx.hpp>

namespace tci {

  // Context management for CytnxContextHandle
  template <> void create_context(CytnxContextHandle& ctx) {
    ctx.set_value(cytnx::Device.cpu);
  }

  template <> void destroy_context(CytnxContextHandle& ctx) {
    // No specific cleanup needed
  }

  // All cytnx::Tensor specializations removed
  // CytnxTensor<ElemT> implementations are in cytnx_typed_tensor_impl.h

}  // namespace tci
