#include "tci/miscellaneous.h"

#include <cmath>
#include <cytnx.hpp>
#include <functional>
#include <iostream>
#include <sstream>

#include "tci/cytnx_tensor_traits.h"
#include "tci/debugging.h"

namespace tci {

  // Template specializations for miscellaneous functions using Cytnx

  // In-place context creation
  template <> void create_context(CytnxContextHandle& ctx) {
    // Initialize Cytnx device context (device ID)
    ctx.set_value(cytnx::Device.cpu);  // Default to CPU (-1)
  }

  template <> void destroy_context(CytnxContextHandle& ctx) {
    // No specific cleanup needed for Cytnx device context
    // Cytnx handles this automatically
  }

  template <> void show(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a) {
    TCI_TIME_FUNCTION_WITH_INFO(
        "tci::show", "shape=[" + std::to_string(a.shape().size())
                         + " dims], dtype=" + std::to_string(static_cast<int>(a.dtype())));

    // Print tensor using Cytnx's built-in print functionality
    std::cout << a << std::endl;
  }

  // to_container implementation moved to header for generic template support

  // eq implementation moved to header for template visibility

  // Convert between same tensor types (device transfer for cytnx::Tensor)
  template <> void convert(context_handle_t<cytnx::Tensor>& ctx1, const cytnx::Tensor& t1,
                           context_handle_t<cytnx::Tensor>& ctx2, cytnx::Tensor& t2) {
    // Create tensor info string
    std::ostringstream info;
    auto shape = t1.shape();
    info << "shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i > 0) info << ",";
      info << shape[i];
    }
    info << "], ctx1=" << ctx1 << " -> ctx2=" << ctx2;

    TCI_TIME_FUNCTION_WITH_INFO("tci::convert", info.str());

    // For same tensor types, perform deep copy
    // If contexts differ, this could involve device transfer (CPU <-> GPU)

    if (ctx1 == ctx2) {
      // Same context - simple copy
      t2 = t1.clone();
    } else {
      // Different contexts - handle device transfer
      // Clone first, then move to target device if needed
      t2 = t1.clone();

      // If ctx2 specifies a different device, move tensor there
      if (ctx2 >= 0) {  // GPU device
        t2 = t2.to(ctx2);
      } else if (ctx2 == cytnx::Device.cpu) {  // CPU device
        t2 = t2.to(cytnx::Device.cpu);
      }
      // For other device types, Cytnx will handle appropriately
    }
  }

}  // namespace tci