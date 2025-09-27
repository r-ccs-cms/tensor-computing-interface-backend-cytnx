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

  // Out-of-place context creation
  template <> int create_context() {
    // Return default CPU device context for Cytnx
    return cytnx::Device.cpu;  // Default to CPU (-1)
  }

  // In-place context creation
  template <> void create_context(int& ctx) {
    // Initialize Cytnx device context (device ID)
    ctx = cytnx::Device.cpu;  // Default to CPU (-1)
  }

  template <> void destroy_context(int& ctx) {
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

  template <> bool eq(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a,
                      const cytnx::Tensor& b, const elem_t<cytnx::Tensor> epsilon) {
    // Create tensor info string
    std::ostringstream info;
    auto shape = a.shape();
    info << "tensors shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i > 0) info << ",";
      info << shape[i];
    }
    info << "], epsilon=" << std::abs(epsilon);

    TCI_TIME_FUNCTION_WITH_INFO("tci::eq", info.str());

    // Check tensor equality within tolerance
    if (a.shape() != b.shape()) {
      return false;
    }

    // Calculate the difference between tensors
    cytnx::Tensor diff = a - b;

    // Calculate the Frobenius norm of the difference
    auto norm_result = cytnx::linalg::Norm(diff);

    // Extract the scalar value with explicit cast (same as in norm function)
    double diff_norm;
    if (norm_result.dtype() == cytnx::Type.Double) {
      diff_norm = static_cast<double>(norm_result.at({0}).real());
    } else if (norm_result.dtype() == cytnx::Type.ComplexDouble) {
      diff_norm = static_cast<double>(norm_result.at({0}).real());  // Norm should always be real
    } else {
      // Convert to double first
      auto converted = norm_result.astype(cytnx::Type.Double);
      diff_norm = static_cast<double>(converted.at({0}).real());
    }

    // Compare with epsilon tolerance
    double eps_magnitude = std::abs(epsilon);
    return diff_norm <= eps_magnitude;
  }

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