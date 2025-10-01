#include "tci/read_only_getters.h"

#include <cytnx.hpp>

#include "tci/cytnx_tensor_traits.h"
#include "tci/cytnx_typed_tensor.h"

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

  // size and size_bytes implementations moved to header for template visibility

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

  // CytnxTensor<ElemT> template implementations are header-only
  // See include/tci/cytnx_typed_tensor_impl.h

}  // namespace tci