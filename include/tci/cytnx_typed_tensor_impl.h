#pragma once

#include "tci/cytnx_typed_tensor.h"
#include "tci/cytnx_tensor_traits.h"
#include "tci/tensor_traits.h"
#include <cytnx.hpp>
#include <vector>

namespace tci {

  // Helper: Convert ElemT to cytnx::Type
  namespace detail {
    template <typename ElemT>
    constexpr unsigned int elem_to_cytnx_type() {
      if constexpr (std::is_same_v<ElemT, cytnx::cytnx_double>) {
        return cytnx::Type.Double;
      } else if constexpr (std::is_same_v<ElemT, cytnx::cytnx_float>) {
        return cytnx::Type.Float;
      } else if constexpr (std::is_same_v<ElemT, cytnx::cytnx_complex128>) {
        return cytnx::Type.ComplexDouble;
      } else if constexpr (std::is_same_v<ElemT, cytnx::cytnx_complex64>) {
        return cytnx::Type.ComplexFloat;
      } else {
        static_assert(sizeof(ElemT) == 0, "Unsupported element type");
      }
    }
  }  // namespace detail

  // Template function declarations (non-specialized)
  // These are separate from the main TCI header declarations to avoid conflicts

  // Construction/Destruction functions for CytnxTensor<ElemT>
  template <typename ElemT>
  void allocate(context_handle_t<CytnxTensor<ElemT>>& ctx,
                const shape_t<CytnxTensor<ElemT>>& shape,
                CytnxTensor<ElemT>& a);

  template <typename ElemT>
  CytnxTensor<ElemT> allocate(context_handle_t<CytnxTensor<ElemT>>& ctx,
                               const shape_t<CytnxTensor<ElemT>>& shape);

  // Read-only getter functions for CytnxTensor<ElemT>
  template <typename ElemT>
  void get_elem(context_handle_t<CytnxTensor<ElemT>>& ctx,
                const CytnxTensor<ElemT>& a,
                const elem_coors_t<CytnxTensor<ElemT>>& coors,
                elem_t<CytnxTensor<ElemT>>& elem);

  template <typename ElemT>
  elem_t<CytnxTensor<ElemT>> get_elem(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                       const CytnxTensor<ElemT>& a,
                                       const elem_coors_t<CytnxTensor<ElemT>>& coors);

  // Tensor manipulation functions for CytnxTensor<ElemT>
  template <typename ElemT, typename Func>
  void for_each(context_handle_t<CytnxTensor<ElemT>>& ctx,
                CytnxTensor<ElemT>& inout,
                Func&& f);

  template <typename ElemT, typename Func>
  void for_each(context_handle_t<CytnxTensor<ElemT>>& ctx,
                const CytnxTensor<ElemT>& in,
                Func&& f);

  // Template implementations

  template <typename ElemT>
  void allocate(context_handle_t<CytnxTensor<ElemT>>& ctx,
                const shape_t<CytnxTensor<ElemT>>& shape,
                CytnxTensor<ElemT>& a) {
    std::vector<cytnx::cytnx_uint64> cytnx_shape;
    cytnx_shape.reserve(shape.size());
    for (const auto& dim : shape) {
      cytnx_shape.push_back(static_cast<cytnx::cytnx_uint64>(dim));
    }
    a.backend = cytnx::Tensor(cytnx_shape, detail::elem_to_cytnx_type<ElemT>(), ctx);
  }

  template <typename ElemT>
  CytnxTensor<ElemT> allocate(context_handle_t<CytnxTensor<ElemT>>& ctx,
                               const shape_t<CytnxTensor<ElemT>>& shape) {
    CytnxTensor<ElemT> result;
    allocate(ctx, shape, result);
    return result;
  }

  template <typename ElemT>
  void get_elem(context_handle_t<CytnxTensor<ElemT>>& ctx,
                const CytnxTensor<ElemT>& a,
                const elem_coors_t<CytnxTensor<ElemT>>& coors,
                elem_t<CytnxTensor<ElemT>>& elem) {
    std::vector<cytnx::cytnx_uint64> cytnx_coors;
    cytnx_coors.reserve(coors.size());
    for (const auto& coord : coors) {
      cytnx_coors.push_back(static_cast<cytnx::cytnx_uint64>(coord));
    }
    elem = a.backend.template at<ElemT>(cytnx_coors);
  }

  template <typename ElemT>
  elem_t<CytnxTensor<ElemT>> get_elem(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                       const CytnxTensor<ElemT>& a,
                                       const elem_coors_t<CytnxTensor<ElemT>>& coors) {
    elem_t<CytnxTensor<ElemT>> elem;
    get_elem(ctx, a, coors, elem);
    return elem;
  }

  // for_each implementation for CytnxTensor<ElemT> (mutable version)
  template <typename ElemT, typename Func>
  void for_each(context_handle_t<CytnxTensor<ElemT>>& ctx,
                CytnxTensor<ElemT>& inout,
                Func&& f) {
    auto total_size = static_cast<cytnx::cytnx_uint64>(inout.backend.storage().size());

    // Direct access to underlying storage for performance
    auto* data = inout.backend.storage().template data<ElemT>();

    for (cytnx::cytnx_uint64 i = 0; i < total_size; ++i) {
      f(data[i]);
    }
  }

  // for_each implementation for CytnxTensor<ElemT> (const version)
  template <typename ElemT, typename Func>
  void for_each(context_handle_t<CytnxTensor<ElemT>>& ctx,
                const CytnxTensor<ElemT>& in,
                Func&& f) {
    auto total_size = static_cast<cytnx::cytnx_uint64>(in.backend.storage().size());

    // Direct access to underlying storage for performance
    const auto* data = in.backend.storage().template data<ElemT>();

    for (cytnx::cytnx_uint64 i = 0; i < total_size; ++i) {
      f(data[i]);
    }
  }

  // Additional functions for CytnxTensor<ElemT>

  template <typename ElemT>
  void zeros(context_handle_t<CytnxTensor<ElemT>>& ctx,
             const shape_t<CytnxTensor<ElemT>>& shape,
             CytnxTensor<ElemT>& a) {
    allocate(ctx, shape, a);
    a.backend.storage().set_zeros();
  }

  template <typename ElemT>
  void eye(context_handle_t<CytnxTensor<ElemT>>& ctx,
           bond_dim_t<CytnxTensor<ElemT>> dim,
           CytnxTensor<ElemT>& a) {
    a.backend = cytnx::eye(dim, detail::elem_to_cytnx_type<ElemT>(), ctx);
  }

  template <typename ElemT>
  void fill(context_handle_t<CytnxTensor<ElemT>>& ctx,
            const shape_t<CytnxTensor<ElemT>>& shape,
            elem_t<CytnxTensor<ElemT>> value,
            CytnxTensor<ElemT>& a) {
    allocate(ctx, shape, a);
    auto total_size = static_cast<cytnx::cytnx_uint64>(a.backend.storage().size());
    auto* data = a.backend.storage().template data<ElemT>();
    for (cytnx::cytnx_uint64 i = 0; i < total_size; ++i) {
      data[i] = value;
    }
  }

  template <typename ElemT>
  void set_elem(context_handle_t<CytnxTensor<ElemT>>& ctx,
                CytnxTensor<ElemT>& a,
                const elem_coors_t<CytnxTensor<ElemT>>& coors,
                elem_t<CytnxTensor<ElemT>> elem) {
    std::vector<cytnx::cytnx_uint64> cytnx_coors;
    cytnx_coors.reserve(coors.size());
    for (const auto& coord : coors) {
      cytnx_coors.push_back(static_cast<cytnx::cytnx_uint64>(coord));
    }
    a.backend.template at<ElemT>(cytnx_coors) = elem;
  }

  template <typename ElemT>
  shape_t<CytnxTensor<ElemT>> shape(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                     const CytnxTensor<ElemT>& a) {
    auto cytnx_shape = a.backend.shape();
    shape_t<CytnxTensor<ElemT>> result;
    result.reserve(cytnx_shape.size());
    for (const auto& dim : cytnx_shape) {
      result.push_back(static_cast<bond_dim_t<CytnxTensor<ElemT>>>(dim));
    }
    return result;
  }

}  // namespace tci