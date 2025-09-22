#include "tci/miscellaneous.h"
#include "tci/cytnx_tensor_traits.h"
#include <cytnx.hpp>
#include <iostream>
#include <cmath>

namespace tci {

// Template specializations for miscellaneous functions using Cytnx

// Out-of-place context creation
template <>
int create_context() {
    // Return default CPU device context for Cytnx
    return cytnx::Device.cpu;  // Default to CPU (-1)
}

// In-place context creation
template <>
void create_context(int &ctx) {
    // Initialize Cytnx device context (device ID)
    ctx = cytnx::Device.cpu;  // Default to CPU (-1)
}

template <>
void destroy_context(int &ctx) {
    // No specific cleanup needed for Cytnx device context
    // Cytnx handles this automatically
}

template <>
void show(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &a
) {
    // Print tensor using Cytnx's built-in print functionality
    std::cout << a << std::endl;
}

template <>
bool eq(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &a,
    const cytnx::Tensor &b,
    const elem_t<cytnx::Tensor> epsilon
) {
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
        diff_norm = static_cast<double>(norm_result.at({0}).real()); // Norm should always be real
    } else {
        // Convert to double first
        auto converted = norm_result.astype(cytnx::Type.Double);
        diff_norm = static_cast<double>(converted.at({0}).real());
    }

    // Compare with epsilon tolerance
    double eps_magnitude = std::abs(epsilon);
    return diff_norm <= eps_magnitude;
}

} // namespace tci