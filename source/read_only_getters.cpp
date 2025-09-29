#include "tci/read_only_getters.h"

#include <cytnx.hpp>

#include "tci/cytnx_tensor_traits.h"

namespace tci {

  // Template specializations for Cytnx::Tensor

  template <>
  rank_t<cytnx::Tensor> rank(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a) {
    return static_cast<rank_t<cytnx::Tensor>>(a.shape().size());
  }

  template <>
  shape_t<cytnx::Tensor> shape(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a) {
    auto cytnx_shape = a.shape();
    shape_t<cytnx::Tensor> result;
    result.reserve(cytnx_shape.size());
    for (const auto& dim : cytnx_shape) {
      result.push_back(static_cast<bond_dim_t<cytnx::Tensor>>(dim));
    }
    return result;
  }

  template <>
  ten_size_t<cytnx::Tensor> size(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a) {
    auto cytnx_shape = a.shape();
    ten_size_t<cytnx::Tensor> total_size = 1;
    for (const auto& dim : cytnx_shape) {
      total_size *= static_cast<ten_size_t<cytnx::Tensor>>(dim);
    }
    return total_size;
  }

  template <> std::size_t size_bytes(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a) {
    // Calculate based on element type and total size
    auto total_elements = size(ctx, a);

    // Get the actual element size based on dtype
    std::size_t element_size;
    if (a.dtype() == cytnx::Type.ComplexDouble) {
      element_size = sizeof(cytnx::cytnx_complex128);
    } else if (a.dtype() == cytnx::Type.ComplexFloat) {
      element_size = sizeof(cytnx::cytnx_complex64);
    } else if (a.dtype() == cytnx::Type.Double) {
      element_size = sizeof(cytnx::cytnx_double);
    } else if (a.dtype() == cytnx::Type.Float) {
      element_size = sizeof(cytnx::cytnx_float);
    } else if (a.dtype() == cytnx::Type.Int64) {
      element_size = sizeof(cytnx::cytnx_int64);
    } else if (a.dtype() == cytnx::Type.Int32) {
      element_size = sizeof(cytnx::cytnx_int32);
    } else if (a.dtype() == cytnx::Type.Int16) {
      element_size = sizeof(cytnx::cytnx_int16);
    } else if (a.dtype() == cytnx::Type.Uint64) {
      element_size = sizeof(cytnx::cytnx_uint64);
    } else if (a.dtype() == cytnx::Type.Uint32) {
      element_size = sizeof(cytnx::cytnx_uint32);
    } else if (a.dtype() == cytnx::Type.Uint16) {
      element_size = sizeof(cytnx::cytnx_uint16);
    } else {
      // Default to complex128 for unknown types
      element_size = sizeof(cytnx::cytnx_complex128);
    }

    return total_elements * element_size;
  }

  template <> void get_elem(context_handle_t<cytnx::Tensor>& ctx, const cytnx::Tensor& a,
                            const elem_coors_t<cytnx::Tensor>& coors, elem_t<cytnx::Tensor>& elem) {
    // Convert coordinates to Cytnx format
    std::vector<cytnx::cytnx_uint64> cytnx_coors;
    cytnx_coors.reserve(coors.size());
    for (const auto& coord : coors) {
      cytnx_coors.push_back(static_cast<cytnx::cytnx_uint64>(coord));
    }

    // Store element based on tensor's actual dtype to preserve type information
    switch (a.dtype()) {
      case cytnx::Type.Double:
        elem = a.at<cytnx::cytnx_double>(cytnx_coors);
        break;
      case cytnx::Type.Float:
        elem = a.at<cytnx::cytnx_float>(cytnx_coors);
        break;
      case cytnx::Type.ComplexDouble:
        elem = a.at<cytnx::cytnx_complex128>(cytnx_coors);
        break;
      case cytnx::Type.ComplexFloat:
        elem = a.at<cytnx::cytnx_complex64>(cytnx_coors);
        break;
      default:
        throw std::runtime_error("Unsupported tensor element type in get_elem");
    }
  }

  template <> elem_t<cytnx::Tensor> get_elem(context_handle_t<cytnx::Tensor>& ctx,
                                             const cytnx::Tensor& a,
                                             const elem_coors_t<cytnx::Tensor>& coors) {
    elem_t<cytnx::Tensor> elem;
    get_elem(ctx, a, coors, elem);
    return elem;
  }

}  // namespace tci