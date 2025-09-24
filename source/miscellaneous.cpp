#include "tci/miscellaneous.h"
#include "tci/cytnx_tensor_traits.h"
#include "tci/debugging.h"
#include <cytnx.hpp>
#include <iostream>
#include <cmath>
#include <functional>
#include <sstream>

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
    TCI_TIME_FUNCTION_WITH_INFO("tci::show",
        "shape=[" + std::to_string(a.shape().size()) + " dims], dtype=" + std::to_string(static_cast<int>(a.dtype())));

    // Print tensor using Cytnx's built-in print functionality
    std::cout << a << std::endl;
}

// Generic template for to_container - specialized for cytnx::Tensor
template <typename RandomIt, typename Func>
void to_container(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &a,
    RandomIt first,
    Func &&coors2idx
) {
    // Create tensor info string
    std::ostringstream info;
    auto shape = a.shape();
    info << "shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) info << ",";
        info << shape[i];
    }
    info << "], elements=" << a.storage_size();

    TCI_TIME_FUNCTION_WITH_INFO("tci::to_container", info.str());

    // Get tensor shape
    auto tensor_shape = a.shape();
    std::vector<cytnx::cytnx_uint64> shape_vec;
    shape_vec.reserve(tensor_shape.size());
    for (const auto& dim : tensor_shape) {
        shape_vec.push_back(static_cast<cytnx::cytnx_uint64>(dim));
    }

    // Generate all possible coordinates and copy elements
    std::function<void(std::vector<cytnx::cytnx_uint64>&, size_t)> visit_coords =
        [&](std::vector<cytnx::cytnx_uint64>& coords, size_t depth) {
            if (depth == shape_vec.size()) {
                // We have a complete coordinate - get the element and store it
                elem_coors_t<cytnx::Tensor> tci_coords;
                tci_coords.reserve(coords.size());
                for (const auto& coord : coords) {
                    tci_coords.push_back(static_cast<elem_coor_t<cytnx::Tensor>>(coord));
                }

                // Get element from tensor
                auto cytnx_scalar = a.at(coords);

                // Convert to elem_t<cytnx::Tensor> (complex128)
                elem_t<cytnx::Tensor> elem_value;
                if (a.dtype() == cytnx::Type.ComplexDouble) {
                    double real_part = static_cast<double>(cytnx_scalar.real());
                    double imag_part = static_cast<double>(cytnx_scalar.imag());
                    elem_value = elem_t<cytnx::Tensor>(real_part, imag_part);
                } else if (a.dtype() == cytnx::Type.Double) {
                    double real_part = static_cast<double>(cytnx_scalar.real());
                    elem_value = elem_t<cytnx::Tensor>(real_part, 0.0);
                } else {
                    double real_part = static_cast<double>(cytnx_scalar.real());
                    elem_value = elem_t<cytnx::Tensor>(real_part, 0.0);
                }

                // Use coors2idx to determine storage location and store element
                auto index = coors2idx(tci_coords);
                *(first + index) = elem_value;
                return;
            }

            // Recurse through all values for current dimension
            for (cytnx::cytnx_uint64 i = 0; i < shape_vec[depth]; ++i) {
                coords[depth] = i;
                visit_coords(coords, depth + 1);
            }
        };

    if (!shape_vec.empty()) {
        std::vector<cytnx::cytnx_uint64> coords(shape_vec.size());
        visit_coords(coords, 0);
    }
}

template <>
bool eq(
    context_handle_t<cytnx::Tensor> &ctx,
    const cytnx::Tensor &a,
    const cytnx::Tensor &b,
    const elem_t<cytnx::Tensor> epsilon
) {
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

template <>
void convert(
    context_handle_t<cytnx::Tensor> &ctx1,
    const cytnx::Tensor &t1,
    context_handle_t<cytnx::Tensor> &ctx2,
    cytnx::Tensor &t2
) {
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

} // namespace tci