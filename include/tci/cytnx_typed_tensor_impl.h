#pragma once

#include "tci/cytnx_typed_tensor.h"
#include "tci/cytnx_tensor_traits.h"
#include "tci/tensor_traits.h"
#include "tci/construction_destruction.h"
#include "tci/tensor_linear_algebra.h"
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

  template <typename ElemT>
  rank_t<CytnxTensor<ElemT>> rank(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                   const CytnxTensor<ElemT>& a) {
    return static_cast<rank_t<CytnxTensor<ElemT>>>(a.backend.shape().size());
  }

  template <typename ElemT>
  ten_size_t<CytnxTensor<ElemT>> size(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                       const CytnxTensor<ElemT>& a) {
    return static_cast<ten_size_t<CytnxTensor<ElemT>>>(a.backend.storage().size());
  }

  template <typename ElemT>
  ten_size_t<CytnxTensor<ElemT>> size_bytes(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                             const CytnxTensor<ElemT>& a) {
    // Calculate total bytes using dtype element size
    auto dtype = a.backend.dtype();
    std::size_t elem_size = 0;
    if (dtype == cytnx::Type.Float) elem_size = sizeof(float);
    else if (dtype == cytnx::Type.Double) elem_size = sizeof(double);
    else if (dtype == cytnx::Type.ComplexFloat) elem_size = sizeof(cytnx::cytnx_complex64);
    else if (dtype == cytnx::Type.ComplexDouble) elem_size = sizeof(cytnx::cytnx_complex128);
    else if (dtype == cytnx::Type.Int64) elem_size = sizeof(cytnx::cytnx_int64);
    else if (dtype == cytnx::Type.Uint64) elem_size = sizeof(cytnx::cytnx_uint64);
    else if (dtype == cytnx::Type.Int32) elem_size = sizeof(cytnx::cytnx_int32);
    else if (dtype == cytnx::Type.Uint32) elem_size = sizeof(cytnx::cytnx_uint32);
    return a.backend.storage().size() * elem_size;
  }

  template <typename ElemT, typename RandNumGen>
  void random(context_handle_t<CytnxTensor<ElemT>>& ctx,
              const shape_t<CytnxTensor<ElemT>>& shape,
              RandNumGen&& gen,
              CytnxTensor<ElemT>& a) {
    allocate(ctx, shape, a);
    auto total_size = static_cast<cytnx::cytnx_uint64>(a.backend.storage().size());
    auto* data = a.backend.storage().template data<ElemT>();

    // gen() should return elem_t<CytnxTensor<ElemT>> (i.e., ElemT)
    // For real types: gen() returns cytnx_double, cytnx_float, etc.
    // For complex types: gen() returns cytnx_complex128, cytnx_complex64, etc.
    for (cytnx::cytnx_uint64 i = 0; i < total_size; ++i) {
      data[i] = gen();
    }
  }

  template <typename ElemT>
  void show(context_handle_t<CytnxTensor<ElemT>>& ctx,
            const CytnxTensor<ElemT>& a) {
    std::cout << a.backend << std::endl;
  }

  // Copy operation
  template <typename ElemT>
  void copy(context_handle_t<CytnxTensor<ElemT>>& ctx,
            const CytnxTensor<ElemT>& orig,
            CytnxTensor<ElemT>& dist) {
    dist.backend = orig.backend.clone();
  }

  template <typename ElemT>
  CytnxTensor<ElemT> copy(context_handle_t<CytnxTensor<ElemT>>& ctx,
                          const CytnxTensor<ElemT>& orig) {
    CytnxTensor<ElemT> result;
    copy(ctx, orig, result);
    return result;
  }

  // Clear operation
  template <typename ElemT>
  void clear(context_handle_t<CytnxTensor<ElemT>>& ctx,
             CytnxTensor<ElemT>& a) {
    // Create empty tensor
    a.backend = cytnx::Tensor();
  }

  // Reshape operation
  template <typename ElemT>
  void reshape(context_handle_t<CytnxTensor<ElemT>>& ctx,
               CytnxTensor<ElemT>& inout,
               const shape_t<CytnxTensor<ElemT>>& new_shape) {
    std::vector<cytnx::cytnx_int64> cytnx_shape;
    cytnx_shape.reserve(new_shape.size());
    for (const auto& dim : new_shape) {
      cytnx_shape.push_back(static_cast<cytnx::cytnx_int64>(dim));
    }
    inout.backend = inout.backend.reshape(cytnx_shape);
  }

  template <typename ElemT>
  void reshape(context_handle_t<CytnxTensor<ElemT>>& ctx,
               const CytnxTensor<ElemT>& in,
               const shape_t<CytnxTensor<ElemT>>& new_shape,
               CytnxTensor<ElemT>& out) {
    std::vector<cytnx::cytnx_int64> cytnx_shape;
    cytnx_shape.reserve(new_shape.size());
    for (const auto& dim : new_shape) {
      cytnx_shape.push_back(static_cast<cytnx::cytnx_int64>(dim));
    }
    out.backend = in.backend.reshape(cytnx_shape);
  }

  // Transpose operation
  template <typename ElemT>
  void transpose(context_handle_t<CytnxTensor<ElemT>>& ctx,
                 CytnxTensor<ElemT>& inout,
                 const std::vector<bond_idx_t<CytnxTensor<ElemT>>>& new_order) {
    std::vector<cytnx::cytnx_uint64> cytnx_order;
    cytnx_order.reserve(new_order.size());
    for (const auto& idx : new_order) {
      cytnx_order.push_back(static_cast<cytnx::cytnx_uint64>(idx));
    }
    inout.backend = inout.backend.permute(cytnx_order);
  }

  template <typename ElemT>
  void transpose(context_handle_t<CytnxTensor<ElemT>>& ctx,
                 const CytnxTensor<ElemT>& in,
                 const std::vector<bond_idx_t<CytnxTensor<ElemT>>>& new_order,
                 CytnxTensor<ElemT>& out) {
    std::vector<cytnx::cytnx_uint64> cytnx_order;
    cytnx_order.reserve(new_order.size());
    for (const auto& idx : new_order) {
      cytnx_order.push_back(static_cast<cytnx::cytnx_uint64>(idx));
    }
    out.backend = in.backend.permute(cytnx_order);
  }

  // Complex conjugate
  template <typename ElemT>
  void cplx_conj(context_handle_t<CytnxTensor<ElemT>>& ctx,
                 CytnxTensor<ElemT>& inout) {
    if constexpr (std::is_same_v<ElemT, cytnx::cytnx_complex128> ||
                  std::is_same_v<ElemT, cytnx::cytnx_complex64>) {
      inout.backend = inout.backend.Conj();
    }
    // For real types, do nothing
  }

  template <typename ElemT>
  void cplx_conj(context_handle_t<CytnxTensor<ElemT>>& ctx,
                 const CytnxTensor<ElemT>& in,
                 CytnxTensor<ElemT>& out) {
    if constexpr (std::is_same_v<ElemT, cytnx::cytnx_complex128> ||
                  std::is_same_v<ElemT, cytnx::cytnx_complex64>) {
      out.backend = in.backend.Conj();
    } else {
      out.backend = in.backend.clone();
    }
  }

  // Real part extraction
  template <typename ElemT>
  void real(context_handle_t<CytnxTensor<ElemT>>& ctx,
            const CytnxTensor<ElemT>& in,
            real_ten_t<CytnxTensor<ElemT>>& out) {
    if constexpr (std::is_same_v<ElemT, cytnx::cytnx_complex128> ||
                  std::is_same_v<ElemT, cytnx::cytnx_complex64>) {
      // Clone first since real() is not const
      auto temp = in.backend.clone();
      out.backend = temp.real();
    } else {
      // For real tensors, just copy
      out.backend = in.backend.clone();
    }
  }

  template <typename ElemT>
  real_ten_t<CytnxTensor<ElemT>> real(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                      const CytnxTensor<ElemT>& in) {
    real_ten_t<CytnxTensor<ElemT>> result;
    real(ctx, in, result);
    return result;
  }

  // Imaginary part extraction
  template <typename ElemT>
  void imag(context_handle_t<CytnxTensor<ElemT>>& ctx,
            const CytnxTensor<ElemT>& in,
            real_ten_t<CytnxTensor<ElemT>>& out) {
    if constexpr (std::is_same_v<ElemT, cytnx::cytnx_complex128> ||
                  std::is_same_v<ElemT, cytnx::cytnx_complex64>) {
      // Clone first since imag() is not const
      auto temp = in.backend.clone();
      out.backend = temp.imag();
    } else {
      // For real tensors, return zeros
      allocate(ctx, shape(ctx, in), out);
      out.backend.storage().set_zeros();
    }
  }

  template <typename ElemT>
  real_ten_t<CytnxTensor<ElemT>> imag(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                      const CytnxTensor<ElemT>& in) {
    real_ten_t<CytnxTensor<ElemT>> result;
    imag(ctx, in, result);
    return result;
  }

  // Convert real tensor to complex tensor
  template <typename ElemT>
  void to_cplx(context_handle_t<CytnxTensor<ElemT>>& ctx,
               const CytnxTensor<ElemT>& in,
               cplx_ten_t<CytnxTensor<ElemT>>& out) {
    if (in.backend.dtype() == cytnx::Type.ComplexDouble ||
        in.backend.dtype() == cytnx::Type.ComplexFloat) {
      // Already complex, just copy
      out.backend = in.backend.clone();
    } else {
      // Convert real to complex
      out.backend = in.backend.astype(cytnx::Type.ComplexDouble);
    }
  }

  // Norm calculation
  template <typename ElemT>
  real_t<CytnxTensor<ElemT>> norm(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                  const CytnxTensor<ElemT>& a) {
    return a.backend.Norm().template item<real_t<CytnxTensor<ElemT>>>();
  }

  // Normalize
  template <typename ElemT>
  real_t<CytnxTensor<ElemT>> normalize(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                       CytnxTensor<ElemT>& inout) {
    auto n = norm(ctx, inout);
    if (n > 0) {
      inout.backend = inout.backend / n;
    }
    return n;
  }

  template <typename ElemT>
  real_t<CytnxTensor<ElemT>> normalize(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                       const CytnxTensor<ElemT>& in,
                                       CytnxTensor<ElemT>& out) {
    auto n = norm(ctx, in);
    if (n > 0) {
      out.backend = in.backend / n;
    } else {
      out.backend = in.backend.clone();
    }
    return n;
  }

  // Scale
  template <typename ElemT>
  void scale(context_handle_t<CytnxTensor<ElemT>>& ctx,
             CytnxTensor<ElemT>& inout,
             const elem_t<CytnxTensor<ElemT>> s) {
    inout.backend = inout.backend * s;
  }

  template <typename ElemT>
  void scale(context_handle_t<CytnxTensor<ElemT>>& ctx,
             const CytnxTensor<ElemT>& in,
             const elem_t<CytnxTensor<ElemT>> s,
             CytnxTensor<ElemT>& out) {
    out.backend = in.backend * s;
  }

  // Diag - extract diagonal or create diagonal matrix
  template <typename ElemT>
  void diag(context_handle_t<CytnxTensor<ElemT>>& ctx,
            CytnxTensor<ElemT>& inout) {
    auto r = inout.backend.shape().size();
    if (r == 1) {
      // Create diagonal matrix from vector
      auto dim = static_cast<cytnx::cytnx_uint64>(inout.backend.shape()[0]);
      auto result = cytnx::zeros({dim, dim},
                                  detail::elem_to_cytnx_type<ElemT>(),
                                  ctx);

      auto* data = inout.backend.storage().template data<ElemT>();
      auto* result_data = result.storage().template data<ElemT>();

      for (cytnx::cytnx_uint64 i = 0; i < dim; ++i) {
        result_data[i * dim + i] = data[i];
      }

      inout.backend = result;
    } else if (r == 2) {
      // Extract diagonal from matrix
      auto dim = static_cast<cytnx::cytnx_uint64>(
          std::min(inout.backend.shape()[0], inout.backend.shape()[1]));
      auto result = cytnx::zeros({dim},
                                  detail::elem_to_cytnx_type<ElemT>(),
                                  ctx);

      auto rows = inout.backend.shape()[0];
      auto* in_data = inout.backend.storage().template data<ElemT>();
      auto* result_data = result.storage().template data<ElemT>();

      for (cytnx::cytnx_uint64 i = 0; i < dim; ++i) {
        result_data[i] = in_data[i * rows + i];
      }

      inout.backend = result;
    }
  }

  template <typename ElemT>
  void diag(context_handle_t<CytnxTensor<ElemT>>& ctx,
            const CytnxTensor<ElemT>& in,
            CytnxTensor<ElemT>& out) {
    out.backend = in.backend.clone();
    diag(ctx, out);
  }

  // Trace - partial trace over specified bond pairs
  template <typename ElemT>
  void trace(context_handle_t<CytnxTensor<ElemT>>& ctx,
             CytnxTensor<ElemT>& inout,
             const bond_idx_pairs_t<CytnxTensor<ElemT>>& bdidx_pairs) {
    cytnx::Tensor result = inout.backend;

    // Create index mapping to track axis renumbering after each trace
    std::vector<cytnx::cytnx_int64> axis_map(result.shape().size());
    std::iota(axis_map.begin(), axis_map.end(), 0);

    // Sort pairs by maximum index in descending order
    auto sorted_pairs = bdidx_pairs;
    std::sort(sorted_pairs.begin(), sorted_pairs.end(),
              [](const auto& a, const auto& b) {
                return std::max(a.first, a.second) > std::max(b.first, b.second);
              });

    for (const auto& [orig_idx1, orig_idx2] : sorted_pairs) {
      // Find current positions of these axes
      auto it1 = std::find(axis_map.begin(), axis_map.end(), orig_idx1);
      auto it2 = std::find(axis_map.begin(), axis_map.end(), orig_idx2);

      if (it1 == axis_map.end() || it2 == axis_map.end()) {
        throw std::invalid_argument("trace: axis index not found in mapping");
      }

      cytnx::cytnx_uint64 curr_idx1 = std::distance(axis_map.begin(), it1);
      cytnx::cytnx_uint64 curr_idx2 = std::distance(axis_map.begin(), it2);

      // Debug output
      if (std::getenv("TCI_VERBOSE")) {
        std::cerr << "trace: orig(" << orig_idx1 << "," << orig_idx2 << ") -> curr("
                  << curr_idx1 << "," << curr_idx2 << ") rank=" << result.shape().size() << std::endl;
      }

      // Perform trace
      result = result.Trace(curr_idx1, curr_idx2);

      // Update axis mapping: remove the two traced axes
      axis_map.erase(it1 < it2 ? it1 : it2);
      axis_map.erase(it1 < it2 ? (it2 - 1) : it1);
    }

    inout.backend = result;
  }

  template <typename ElemT>
  void trace(context_handle_t<CytnxTensor<ElemT>>& ctx,
             const CytnxTensor<ElemT>& in,
             const bond_idx_pairs_t<CytnxTensor<ElemT>>& bdidx_pairs,
             CytnxTensor<ElemT>& out) {
    out.backend = in.backend.clone();
    trace(ctx, out, bdidx_pairs);
  }

  // Contract - tensor contraction following Einstein summation
  // Restored from b7ecb2a9^ (correct implementation using cytnx::linalg::Tensordot)
  template <typename ElemT>
  void contract(context_handle_t<CytnxTensor<ElemT>>& ctx,
                const CytnxTensor<ElemT>& a,
                const std::vector<bond_label_t<CytnxTensor<ElemT>>>& bd_labs_a,
                const CytnxTensor<ElemT>& b,
                const std::vector<bond_label_t<CytnxTensor<ElemT>>>& bd_labs_b,
                CytnxTensor<ElemT>& c,
                const std::vector<bond_label_t<CytnxTensor<ElemT>>>& bd_labs_c) {
    (void)ctx;

    const auto rank_a = a.backend.shape().size();
    const auto rank_b = b.backend.shape().size();

    bool treat_as_label_mode
        = (bd_labs_a.size() == rank_a) && (bd_labs_b.size() == rank_b);

    const auto in_range = [](size_t rank, const std::vector<bond_label_t<CytnxTensor<ElemT>>>& axes_list) {
      for (auto axis : axes_list) {
        if (axis < 0 || static_cast<size_t>(axis) >= rank) {
          return false;
        }
      }
      return true;
    };

    if (!in_range(rank_a, bd_labs_a) || !in_range(rank_b, bd_labs_b)) {
      treat_as_label_mode = true;
    }

    if (bd_labs_a.size() > rank_a || bd_labs_b.size() > rank_b) {
      treat_as_label_mode = true;
    }

    if (treat_as_label_mode) {
      detail::NCONAnalysis<CytnxTensor<ElemT>> analysis(bd_labs_a, bd_labs_b, bd_labs_c);

      if (analysis.contract_axes_a.empty() && analysis.contract_axes_b.empty()) {
        // Outer product case
        auto flatten = [](const cytnx::Tensor& tensor, bool row_vector) {
          cytnx::Tensor flat = tensor.clone();
          cytnx::cytnx_uint64 total
              = std::accumulate(tensor.shape().begin(), tensor.shape().end(),
                                static_cast<cytnx::cytnx_uint64>(1),
                                std::multiplies<cytnx::cytnx_uint64>());
          if (row_vector) {
            flat.reshape_({1, static_cast<cytnx::cytnx_int64>(total)});
          } else {
            flat.reshape_({static_cast<cytnx::cytnx_int64>(total), 1});
          }
          return flat;
        };

        auto a_flat = flatten(a.backend, false);
        auto b_flat = flatten(b.backend, true);
        cytnx::Tensor outer_matrix = cytnx::linalg::Matmul(a_flat, b_flat);

        std::vector<cytnx::cytnx_uint64> target_shape;
        auto a_shape = a.backend.shape();
        auto b_shape = b.backend.shape();
        target_shape.insert(target_shape.end(), a_shape.begin(), a_shape.end());
        target_shape.insert(target_shape.end(), b_shape.begin(), b_shape.end());

        cytnx::Tensor result = outer_matrix.reshape(target_shape);
        if (!analysis.output_permutation.empty()) {
          std::vector<cytnx::cytnx_uint64> valid_perm;
          for (auto idx : analysis.output_permutation) {
            if (idx < static_cast<cytnx::cytnx_uint64>(result.shape().size())) {
              valid_perm.push_back(idx);
            }
          }
          if (valid_perm.size() == result.shape().size()) {
            result = result.permute(valid_perm);
          }
        }
        c.backend = std::move(result);
        return;
      }

      if (analysis.contract_axes_a.empty() || analysis.contract_axes_b.empty()) {
        throw std::invalid_argument("contract: no valid contraction axes found");
      }

      try {
        cytnx::Tensor result = cytnx::linalg::Tensordot(a.backend, b.backend,
                                                        analysis.contract_axes_a,
                                                        analysis.contract_axes_b);
        if (!analysis.output_permutation.empty()) {
          std::vector<cytnx::cytnx_uint64> valid_perm;
          for (auto idx : analysis.output_permutation) {
            if (idx < static_cast<cytnx::cytnx_uint64>(result.shape().size())) {
              valid_perm.push_back(idx);
            }
          }
          if (valid_perm.size() == result.shape().size()) {
            result = result.permute(valid_perm);
          }
        }
        c.backend = std::move(result);
      } catch (const std::exception& e) {
        throw std::runtime_error(std::string("contract: Tensordot failed - ") + e.what());
      }
      return;
    }

    // Axis mode
    auto convert_axes = [](size_t rank,
                           const std::vector<bond_label_t<CytnxTensor<ElemT>>>& axes_list,
                           const char* which) {
      std::vector<cytnx::cytnx_uint64> axes;
      axes.reserve(axes_list.size());
      std::vector<bool> seen(rank, false);
      for (auto axis : axes_list) {
        if (axis < 0 || static_cast<size_t>(axis) >= rank) {
          std::ostringstream oss;
          oss << "contract: axis index out of range for " << which << " (rank=" << rank
              << ", requested=" << axis << ")";
          oss << " | axes list=";
          for (auto v : axes_list) oss << v << ' ';
          throw std::out_of_range(oss.str());
        }
        auto idx = static_cast<size_t>(axis);
        if (seen[idx]) {
          throw std::invalid_argument("contract: duplicate axis index detected");
        }
        seen[idx] = true;
        axes.push_back(static_cast<cytnx::cytnx_uint64>(idx));
      }
      return axes;
    };

    auto contract_axes_a = convert_axes(rank_a, bd_labs_a, "first tensor");
    auto contract_axes_b = convert_axes(rank_b, bd_labs_b, "second tensor");

    if (contract_axes_a.size() != contract_axes_b.size()) {
      throw std::invalid_argument("contract: axis lists must have equal length");
    }

    auto collect_free_axes = [](size_t rank,
                                const std::vector<cytnx::cytnx_uint64>& contracted) {
      std::vector<bool> used(rank, false);
      for (auto idx : contracted) used[idx] = true;
      std::vector<cytnx::cytnx_uint64> free_axes;
      free_axes.reserve(rank - contracted.size());
      for (size_t i = 0; i < rank; ++i) {
        if (!used[i]) free_axes.push_back(static_cast<cytnx::cytnx_uint64>(i));
      }
      return free_axes;
    };

    const auto free_axes_a = collect_free_axes(rank_a, contract_axes_a);
    const auto free_axes_b = collect_free_axes(rank_b, contract_axes_b);
    const size_t total_free = free_axes_a.size() + free_axes_b.size();

    std::vector<cytnx::cytnx_uint64> permute_order;

    if (!bd_labs_c.empty()) {
      if (bd_labs_c.size() != total_free) {
        throw std::invalid_argument("contract: output axis specification mismatch");
      }

      std::vector<bool> seen(total_free, false);
      permute_order.reserve(total_free);
      for (auto axis : bd_labs_c) {
        if (axis < 0 || static_cast<size_t>(axis) >= total_free || seen[axis]) {
          throw std::invalid_argument("contract: invalid output axis permutation");
        }
        seen[axis] = true;
        permute_order.push_back(static_cast<cytnx::cytnx_uint64>(axis));
      }
    }

    try {
      cytnx::Tensor result
          = cytnx::linalg::Tensordot(a.backend, b.backend, contract_axes_a, contract_axes_b);

      if (!permute_order.empty()) {
        result = result.permute(permute_order);
      }

      c.backend = std::move(result);
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string("contract: Tensordot failed - ") + e.what());
    }
  }

  // Linear combination
  template <typename ElemT>
  void linear_combine(context_handle_t<CytnxTensor<ElemT>>& ctx,
                      const std::vector<CytnxTensor<ElemT>>& ins,
                      CytnxTensor<ElemT>& out) {
    if (ins.empty()) {
      return;
    }

    out.backend = ins[0].backend.clone();
    for (size_t i = 1; i < ins.size(); ++i) {
      out.backend = out.backend + ins[i].backend;
    }
  }

  template <typename ElemT>
  void linear_combine(context_handle_t<CytnxTensor<ElemT>>& ctx,
                      const std::vector<CytnxTensor<ElemT>>& ins,
                      const std::vector<elem_t<CytnxTensor<ElemT>>>& coefs,
                      CytnxTensor<ElemT>& out) {
    if (ins.empty() || ins.size() != coefs.size()) {
      return;
    }

    out.backend = ins[0].backend * coefs[0];
    for (size_t i = 1; i < ins.size(); ++i) {
      out.backend = out.backend + ins[i].backend * coefs[i];
    }
  }

  // SVD (full)
  template <typename ElemT>
  void svd(context_handle_t<CytnxTensor<ElemT>>& ctx,
           const CytnxTensor<ElemT>& a,
           const rank_t<CytnxTensor<ElemT>> num_of_bds_as_row,
           CytnxTensor<ElemT>& u,
           real_ten_t<CytnxTensor<ElemT>>& s_diag,
           CytnxTensor<ElemT>& v_dag) {
    // Get shape and compute matrix dimensions
    auto a_shape = shape(ctx, a);
    cytnx::cytnx_uint64 left_dim = 1;
    for (rank_t<CytnxTensor<ElemT>> i = 0; i < num_of_bds_as_row; ++i) {
      left_dim *= a_shape[i];
    }
    cytnx::cytnx_uint64 right_dim = 1;
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      right_dim *= a_shape[i];
    }

    // Reshape to matrix
    auto a_reshaped = a.backend.reshape({static_cast<cytnx::cytnx_int64>(left_dim),
                                          static_cast<cytnx::cytnx_int64>(right_dim)});

    // Perform full SVD
    auto svd_result = cytnx::linalg::Svd(a_reshaped, true);  // Return U, S, Vt

    if (svd_result.size() < 3) {
      throw std::runtime_error("svd: unexpected result size from Svd");
    }

    // Extract S, U, Vt (order: S, U, V†)
    auto& s_backend = svd_result[0];
    auto& u_backend = svd_result[1];
    auto& vt_backend = svd_result[2];

    bond_dim_t<CytnxTensor<ElemT>> bond_dim = s_backend.shape()[0];

    // Extract real singular values
    cytnx::Tensor s_real = s_backend.dtype() == cytnx::Type.Double ? s_backend : s_backend.real();

    // Reshape U
    shape_t<CytnxTensor<ElemT>> u_shape;
    for (rank_t<CytnxTensor<ElemT>> i = 0; i < num_of_bds_as_row; ++i) {
      u_shape.push_back(a_shape[i]);
    }
    u_shape.push_back(bond_dim);
    std::vector<cytnx::cytnx_int64> u_cytnx_shape;
    for (auto dim : u_shape) {
      u_cytnx_shape.push_back(static_cast<cytnx::cytnx_int64>(dim));
    }
    u.backend = u_backend.reshape(u_cytnx_shape);

    // Set S (as real tensor)
    s_diag.backend = s_real;

    // Reshape V†
    shape_t<CytnxTensor<ElemT>> v_shape;
    v_shape.push_back(bond_dim);
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      v_shape.push_back(a_shape[i]);
    }
    std::vector<cytnx::cytnx_int64> v_cytnx_shape;
    for (auto dim : v_shape) {
      v_cytnx_shape.push_back(static_cast<cytnx::cytnx_int64>(dim));
    }
    v_dag.backend = vt_backend.reshape(v_cytnx_shape);
  }

  // QR decomposition
  template <typename ElemT>
  void qr(context_handle_t<CytnxTensor<ElemT>>& ctx,
          const CytnxTensor<ElemT>& a,
          const rank_t<CytnxTensor<ElemT>> num_of_bds_as_row,
          CytnxTensor<ElemT>& q,
          CytnxTensor<ElemT>& r) {
    // Get shape and compute matrix dimensions
    auto a_shape = shape(ctx, a);
    cytnx::cytnx_uint64 left_dim = 1;
    for (rank_t<CytnxTensor<ElemT>> i = 0; i < num_of_bds_as_row; ++i) {
      left_dim *= a_shape[i];
    }
    cytnx::cytnx_uint64 right_dim = 1;
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      right_dim *= a_shape[i];
    }

    // Reshape to matrix
    auto a_reshaped = a.backend.reshape({static_cast<cytnx::cytnx_int64>(left_dim),
                                          static_cast<cytnx::cytnx_int64>(right_dim)});

    // Perform QR
    auto qr_result = cytnx::linalg::Qr(a_reshaped);

    if (qr_result.size() < 2) {
      throw std::runtime_error("qr: unexpected result size from Qr");
    }

    auto& q_backend = qr_result[0];
    auto& r_backend = qr_result[1];

    auto bond_dim = q_backend.shape()[1];

    // Reshape Q
    shape_t<CytnxTensor<ElemT>> q_shape;
    for (rank_t<CytnxTensor<ElemT>> i = 0; i < num_of_bds_as_row; ++i) {
      q_shape.push_back(a_shape[i]);
    }
    q_shape.push_back(bond_dim);
    std::vector<cytnx::cytnx_int64> q_cytnx_shape;
    for (auto dim : q_shape) {
      q_cytnx_shape.push_back(static_cast<cytnx::cytnx_int64>(dim));
    }
    q.backend = q_backend.reshape(q_cytnx_shape);

    // Reshape R
    shape_t<CytnxTensor<ElemT>> r_shape;
    r_shape.push_back(bond_dim);
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      r_shape.push_back(a_shape[i]);
    }
    std::vector<cytnx::cytnx_int64> r_cytnx_shape;
    for (auto dim : r_shape) {
      r_cytnx_shape.push_back(static_cast<cytnx::cytnx_int64>(dim));
    }
    r.backend = r_backend.reshape(r_cytnx_shape);
  }

  // LQ decomposition
  template <typename ElemT>
  void lq(context_handle_t<CytnxTensor<ElemT>>& ctx,
          const CytnxTensor<ElemT>& a,
          const rank_t<CytnxTensor<ElemT>> num_of_bds_as_row,
          CytnxTensor<ElemT>& l,
          CytnxTensor<ElemT>& q) {
    // LQ = (Q†L†)† where Q†L† is QR of A†
    // Transpose and do QR, then transpose results back

    auto a_shape = shape(ctx, a);
    cytnx::cytnx_uint64 left_dim = 1;
    for (rank_t<CytnxTensor<ElemT>> i = 0; i < num_of_bds_as_row; ++i) {
      left_dim *= a_shape[i];
    }
    cytnx::cytnx_uint64 right_dim = 1;
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      right_dim *= a_shape[i];
    }

    // Reshape and transpose
    auto a_reshaped = a.backend.reshape({static_cast<cytnx::cytnx_int64>(left_dim),
                                          static_cast<cytnx::cytnx_int64>(right_dim)});
    auto a_t = a_reshaped.permute({1, 0});

    // Perform QR on transposed
    auto qr_result = cytnx::linalg::Qr(a_t);

    if (qr_result.size() < 2) {
      throw std::runtime_error("lq: unexpected result size from Qr");
    }

    auto& q_backend = qr_result[0];
    auto& r_backend = qr_result[1];

    // L = R†, Q = Q†
    auto l_backend = r_backend.permute({1, 0});
    auto q_backend_final = q_backend.permute({1, 0});

    auto bond_dim = l_backend.shape()[1];

    // Reshape L
    shape_t<CytnxTensor<ElemT>> l_shape;
    for (rank_t<CytnxTensor<ElemT>> i = 0; i < num_of_bds_as_row; ++i) {
      l_shape.push_back(a_shape[i]);
    }
    l_shape.push_back(bond_dim);
    std::vector<cytnx::cytnx_int64> l_cytnx_shape;
    for (auto dim : l_shape) {
      l_cytnx_shape.push_back(static_cast<cytnx::cytnx_int64>(dim));
    }
    l.backend = l_backend.reshape(l_cytnx_shape);

    // Reshape Q
    shape_t<CytnxTensor<ElemT>> q_shape;
    q_shape.push_back(bond_dim);
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      q_shape.push_back(a_shape[i]);
    }
    std::vector<cytnx::cytnx_int64> q_cytnx_shape;
    for (auto dim : q_shape) {
      q_cytnx_shape.push_back(static_cast<cytnx::cytnx_int64>(dim));
    }
    q.backend = q_backend_final.reshape(q_cytnx_shape);
  }

  template <typename ElemT>
  void trunc_svd(context_handle_t<CytnxTensor<ElemT>>& ctx,
                 const CytnxTensor<ElemT>& a,
                 const rank_t<CytnxTensor<ElemT>> num_of_bds_as_row,
                 CytnxTensor<ElemT>& u,
                 real_ten_t<CytnxTensor<ElemT>>& s_diag,
                 CytnxTensor<ElemT>& v_dag,
                 real_t<CytnxTensor<ElemT>>& trunc_err,
                 const bond_dim_t<CytnxTensor<ElemT>> chi_max,
                 const real_t<CytnxTensor<ElemT>> s_min) {
    // Get shape and compute matrix dimensions
    auto a_shape = shape(ctx, a);
    cytnx::cytnx_uint64 left_dim = 1;
    for (rank_t<CytnxTensor<ElemT>> i = 0; i < num_of_bds_as_row; ++i) {
      left_dim *= a_shape[i];
    }
    cytnx::cytnx_uint64 right_dim = 1;
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      right_dim *= a_shape[i];
    }

    // Reshape to matrix
    auto a_reshaped = a.backend.reshape({static_cast<cytnx::cytnx_int64>(left_dim),
                                          static_cast<cytnx::cytnx_int64>(right_dim)});

    // Perform SVD
    // Parameters: tensor, chi_max, s_min, is_UvT, return_err, mindim
    // Note: return_err=0 to avoid ASAN container-overflow in Cytnx's Svd_truncate
    auto svd_result = cytnx::linalg::Svd_truncate(a_reshaped, chi_max, s_min, true, 0, 1);

    if (svd_result.size() < 3) {
      throw std::runtime_error("trunc_svd: unexpected result size from Svd_truncate");
    }

    // Extract S, U, Vt (order: S, U, V†)
    auto& s_backend = svd_result[0];
    auto& u_backend = svd_result[1];
    auto& vt_backend = svd_result[2];

    // Calculate truncation error manually from minimum singular value
    bond_dim_t<CytnxTensor<ElemT>> bond_dim = s_backend.shape()[0];
    if (bond_dim > 0) {
      if (s_backend.dtype() == cytnx::Type.Double) {
        auto* s_data = s_backend.template ptr_as<double>();
        trunc_err = s_data[bond_dim - 1];
      } else if (s_backend.dtype() == cytnx::Type.Float) {
        auto* s_data = s_backend.template ptr_as<float>();
        trunc_err = static_cast<double>(s_data[bond_dim - 1]);
      } else {
        trunc_err = 0.0;
      }
    } else {
      trunc_err = 0.0;
    }

    // Extract real singular values (SVD of complex tensors may return complex-typed tensor)
    cytnx::Tensor s_real = s_backend.dtype() == cytnx::Type.Double ? s_backend : s_backend.real();

    // Reshape U
    shape_t<CytnxTensor<ElemT>> u_shape;
    for (rank_t<CytnxTensor<ElemT>> i = 0; i < num_of_bds_as_row; ++i) {
      u_shape.push_back(a_shape[i]);
    }
    u_shape.push_back(bond_dim);
    std::vector<cytnx::cytnx_int64> u_cytnx_shape;
    for (auto dim : u_shape) {
      u_cytnx_shape.push_back(static_cast<cytnx::cytnx_int64>(dim));
    }
    u.backend = u_backend.reshape(u_cytnx_shape);

    // Set S (as real tensor)
    s_diag.backend = s_real;

    // Reshape V†
    shape_t<CytnxTensor<ElemT>> v_shape;
    v_shape.push_back(bond_dim);
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      v_shape.push_back(a_shape[i]);
    }
    std::vector<cytnx::cytnx_int64> v_cytnx_shape;
    for (auto dim : v_shape) {
      v_cytnx_shape.push_back(static_cast<cytnx::cytnx_int64>(dim));
    }
    v_dag.backend = vt_backend.reshape(v_cytnx_shape);
  }

  // Eigenvalue decomposition - eigvals (general matrix eigenvalues)
  template <typename ElemT>
  void eigvals(context_handle_t<CytnxTensor<ElemT>>& ctx,
               const CytnxTensor<ElemT>& a,
               const rank_t<CytnxTensor<ElemT>> num_of_bds_as_row,
               cplx_ten_t<CytnxTensor<ElemT>>& w_diag) {
    auto a_shape = shape(ctx, a);

    cytnx::cytnx_uint64 row_dim = 1;
    cytnx::cytnx_uint64 col_dim = 1;

    for (rank_t<CytnxTensor<ElemT>> i = 0; i < num_of_bds_as_row && i < a_shape.size(); ++i) {
      row_dim *= a_shape[i];
    }
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      col_dim *= a_shape[i];
    }

    if (row_dim != col_dim) {
      throw std::invalid_argument("eigvals: matrix must be square");
    }

    cytnx::Tensor matrix = a.backend.clone();
    matrix.reshape_({static_cast<cytnx::cytnx_int64>(row_dim),
                     static_cast<cytnx::cytnx_int64>(col_dim)});

    auto eig_result = cytnx::linalg::Eig(matrix);
    w_diag.backend = eig_result[0];

    if (w_diag.backend.shape().size() != 1) {
      w_diag.backend.reshape_({static_cast<cytnx::cytnx_int64>(row_dim)});
    }
    if (w_diag.backend.dtype() != cytnx::Type.ComplexDouble) {
      w_diag.backend = w_diag.backend.astype(cytnx::Type.ComplexDouble);
    }
  }

  // Eigenvalue decomposition - eigvalsh (hermitian matrix eigenvalues)
  template <typename ElemT>
  void eigvalsh(context_handle_t<CytnxTensor<ElemT>>& ctx,
                const CytnxTensor<ElemT>& a,
                const rank_t<CytnxTensor<ElemT>> num_of_bds_as_row,
                real_ten_t<CytnxTensor<ElemT>>& w_diag) {
    auto a_shape = shape(ctx, a);

    cytnx::cytnx_uint64 row_dim = 1;
    cytnx::cytnx_uint64 col_dim = 1;

    for (rank_t<CytnxTensor<ElemT>> i = 0; i < num_of_bds_as_row && i < a_shape.size(); ++i) {
      row_dim *= a_shape[i];
    }
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      col_dim *= a_shape[i];
    }

    if (row_dim != col_dim) {
      throw std::invalid_argument("eigvalsh: matrix must be square");
    }

    cytnx::Tensor matrix = a.backend.clone();
    matrix.reshape_({static_cast<cytnx::cytnx_int64>(row_dim),
                     static_cast<cytnx::cytnx_int64>(col_dim)});

    auto eigh_result = cytnx::linalg::Eigh(matrix);
    w_diag.backend = eigh_result[0];

    if (w_diag.backend.shape().size() != 1) {
      w_diag.backend.reshape_({static_cast<cytnx::cytnx_int64>(row_dim)});
    }
  }

  // Eigenvalue decomposition - eig (general matrix eigenvalues and eigenvectors)
  template <typename ElemT>
  void eig(context_handle_t<CytnxTensor<ElemT>>& ctx,
           const CytnxTensor<ElemT>& a,
           const rank_t<CytnxTensor<ElemT>> num_of_bds_as_row,
           cplx_ten_t<CytnxTensor<ElemT>>& w_diag,
           cplx_ten_t<CytnxTensor<ElemT>>& v) {
    auto a_shape = shape(ctx, a);

    cytnx::cytnx_uint64 row_dim = 1;
    cytnx::cytnx_uint64 col_dim = 1;

    for (rank_t<CytnxTensor<ElemT>> i = 0; i < num_of_bds_as_row && i < a_shape.size(); ++i) {
      row_dim *= a_shape[i];
    }
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      col_dim *= a_shape[i];
    }

    if (row_dim != col_dim) {
      throw std::invalid_argument("eig: matrix must be square");
    }

    cytnx::Tensor matrix = a.backend.clone();
    matrix.reshape_({static_cast<cytnx::cytnx_int64>(row_dim),
                     static_cast<cytnx::cytnx_int64>(col_dim)});

    auto eig_result = cytnx::linalg::Eig(matrix);
    w_diag.backend = eig_result[0];
    v.backend = eig_result[1];

    if (w_diag.backend.shape().size() != 1) {
      w_diag.backend.reshape_({static_cast<cytnx::cytnx_int64>(row_dim)});
    }
    if (w_diag.backend.dtype() != cytnx::Type.ComplexDouble) {
      w_diag.backend = w_diag.backend.astype(cytnx::Type.ComplexDouble);
    }

    if (v.backend.shape().size() != 2) {
      v.backend.reshape_({static_cast<cytnx::cytnx_int64>(row_dim),
                          static_cast<cytnx::cytnx_int64>(row_dim)});
    }
    if (v.backend.dtype() != cytnx::Type.ComplexDouble) {
      v.backend = v.backend.astype(cytnx::Type.ComplexDouble);
    }
  }

  // Eigenvalue decomposition - eigh (hermitian matrix eigenvalues and eigenvectors)
  template <typename ElemT>
  void eigh(context_handle_t<CytnxTensor<ElemT>>& ctx,
            const CytnxTensor<ElemT>& a,
            const rank_t<CytnxTensor<ElemT>> num_of_bds_as_row,
            real_ten_t<CytnxTensor<ElemT>>& w_diag,
            CytnxTensor<ElemT>& v) {
    auto a_shape = shape(ctx, a);

    cytnx::cytnx_uint64 row_dim = 1;
    cytnx::cytnx_uint64 col_dim = 1;

    for (rank_t<CytnxTensor<ElemT>> i = 0; i < num_of_bds_as_row && i < a_shape.size(); ++i) {
      row_dim *= a_shape[i];
    }
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      col_dim *= a_shape[i];
    }

    if (row_dim != col_dim) {
      throw std::invalid_argument("eigh: matrix must be square");
    }

    cytnx::Tensor matrix = a.backend.clone();
    matrix.reshape_({static_cast<cytnx::cytnx_int64>(row_dim),
                     static_cast<cytnx::cytnx_int64>(col_dim)});

    auto eigh_result = cytnx::linalg::Eigh(matrix);
    w_diag.backend = eigh_result[0];
    v.backend = eigh_result[1];

    if (w_diag.backend.shape().size() != 1) {
      w_diag.backend.reshape_({static_cast<cytnx::cytnx_int64>(row_dim)});
    }

    if (v.backend.shape().size() != 2) {
      v.backend.reshape_({static_cast<cytnx::cytnx_int64>(row_dim),
                          static_cast<cytnx::cytnx_int64>(row_dim)});
    }
  }

  // Tensor equality check with epsilon tolerance
  template <typename ElemT>
  bool eq(context_handle_t<CytnxTensor<ElemT>>& ctx,
          const CytnxTensor<ElemT>& a,
          const CytnxTensor<ElemT>& b,
          const elem_t<CytnxTensor<ElemT>> epsilon) {
    (void)ctx;
    // Check shape first
    if (a.backend.shape() != b.backend.shape()) {
      return false;
    }

    // Calculate the difference between tensors
    cytnx::Tensor diff = a.backend - b.backend;

    // Calculate the Frobenius norm of the difference
    auto norm_result = cytnx::linalg::Norm(diff);

    // Extract the scalar value
    double diff_norm;
    if (norm_result.dtype() == cytnx::Type.Double) {
      diff_norm = static_cast<double>(norm_result.at({0}).real());
    } else if (norm_result.dtype() == cytnx::Type.ComplexDouble) {
      diff_norm = static_cast<double>(norm_result.at({0}).real());
    } else {
      auto converted = norm_result.astype(cytnx::Type.Double);
      diff_norm = static_cast<double>(converted.at({0}).real());
    }

    // Compare with epsilon tolerance
    double eps_magnitude = std::abs(epsilon);
    return diff_norm <= eps_magnitude;
  }

  // assign_from_container - create tensor from container
  template <typename ElemT, typename RandomIt, typename Func>
  void assign_from_container(context_handle_t<CytnxTensor<ElemT>>& ctx,
                             const shape_t<CytnxTensor<ElemT>>& shape,
                             RandomIt init_elems_begin,
                             Func&& coors2idx,
                             CytnxTensor<ElemT>& a) {
    // Allocate tensor with the specified shape
    allocate(ctx, shape, a);

    // Generate all coordinate combinations and assign values
    std::function<void(elem_coors_t<CytnxTensor<ElemT>>, std::size_t)> assign_recursive;
    assign_recursive = [&](elem_coors_t<CytnxTensor<ElemT>> current_coords, std::size_t dim) {
      if (dim == shape.size()) {
        // Base case: all dimensions set, assign the element
        auto index = std::invoke(coors2idx, current_coords);
        auto value = *(init_elems_begin + index);

        // Convert value to ElemT
        ElemT elem_val;
        if constexpr (std::is_same_v<ElemT, decltype(value)>) {
          elem_val = value;
        } else if constexpr (std::is_arithmetic_v<decltype(value)>) {
          elem_val = static_cast<ElemT>(value);
        } else if constexpr (std::is_same_v<ElemT, cytnx::cytnx_complex128> &&
                             std::is_same_v<decltype(value), std::complex<double>>) {
          elem_val = cytnx::cytnx_complex128(value.real(), value.imag());
        } else if constexpr (std::is_same_v<ElemT, cytnx::cytnx_complex64> &&
                             std::is_same_v<decltype(value), std::complex<float>>) {
          elem_val = cytnx::cytnx_complex64(value.real(), value.imag());
        } else {
          elem_val = static_cast<ElemT>(value);
        }

        set_elem(ctx, a, current_coords, elem_val);
      } else {
        // Recursive case: iterate through current dimension
        for (bond_dim_t<CytnxTensor<ElemT>> i = 0; i < shape[dim]; ++i) {
          current_coords.push_back(i);
          assign_recursive(current_coords, dim + 1);
          current_coords.pop_back();
        }
      }
    };

    assign_recursive({}, 0);
  }

  // to_container - copy tensor elements to container
  template <typename ElemT, typename RandomIt, typename Func>
  void to_container(context_handle_t<CytnxTensor<ElemT>>& ctx,
                    const CytnxTensor<ElemT>& a,
                    RandomIt first,
                    Func&& coors2idx) {
    const auto ten_shape = shape(ctx, a);
    const auto total_size = size(ctx, a);

    // Create coordinate vector for iteration
    elem_coors_t<CytnxTensor<ElemT>> coors(ten_shape.size(), 0);

    for (size_t flat_idx = 0; flat_idx < total_size; ++flat_idx) {
      // Get element at current coordinates
      auto elem = get_elem(ctx, a, coors);

      // Use lambda to convert coordinates to container index
      auto container_idx = std::invoke(coors2idx, coors);

      // Store element in container with type conversion
      using ContainerValueType = typename std::iterator_traits<RandomIt>::value_type;

      // Convert ElemT to container value type
      if constexpr (std::is_same_v<ElemT, ContainerValueType>) {
        // Direct assignment
        *(first + container_idx) = elem;
      } else if constexpr (std::is_arithmetic_v<ContainerValueType>) {
        // Convert complex to real by taking real part
        if constexpr (std::is_same_v<ElemT, cytnx::cytnx_complex128> ||
                      std::is_same_v<ElemT, cytnx::cytnx_complex64>) {
          *(first + container_idx) = static_cast<ContainerValueType>(elem.real());
        } else {
          *(first + container_idx) = static_cast<ContainerValueType>(elem);
        }
      } else if constexpr (std::is_same_v<ContainerValueType, std::complex<double>> &&
                           (std::is_same_v<ElemT, cytnx::cytnx_complex128> ||
                            std::is_same_v<ElemT, cytnx::cytnx_complex64>)) {
        // Convert cytnx complex to std::complex
        *(first + container_idx) = std::complex<double>(elem.real(), elem.imag());
      } else if constexpr (std::is_same_v<ContainerValueType, std::complex<float>> &&
                           (std::is_same_v<ElemT, cytnx::cytnx_complex128> ||
                            std::is_same_v<ElemT, cytnx::cytnx_complex64>)) {
        // Convert cytnx complex to std::complex<float>
        *(first + container_idx) = std::complex<float>(
            static_cast<float>(elem.real()),
            static_cast<float>(elem.imag()));
      } else {
        // Fallback: static_cast
        *(first + container_idx) = static_cast<ContainerValueType>(elem);
      }

      // Advance to next coordinate (row-major order)
      for (int dim = static_cast<int>(coors.size()) - 1; dim >= 0; --dim) {
        if (++coors[dim] < ten_shape[dim]) {
          break;
        }
        coors[dim] = 0;
      }
    }
  }

  // Tensor Manipulation functions - independent implementations needed

  // expand, extract_sub, replace_sub helper functions
  // Restored from git show b7ecb2a9^:source/tensor_manipulation.cpp
  namespace {
    void copy_original_data_recursive(const cytnx::Tensor& src, cytnx::Tensor& dst, std::size_t dim,
                                      std::vector<cytnx::cytnx_uint64> current_coords,
                                      const std::vector<cytnx::cytnx_uint64>& original_shape) {
      if (dim == original_shape.size()) {
        // Base case: copy element
        auto src_elem = src.at(current_coords);
        dst.at(current_coords) = src_elem;
        return;
      }

      for (cytnx::cytnx_uint64 i = 0; i < original_shape[dim]; ++i) {
        current_coords.push_back(i);
        copy_original_data_recursive(src, dst, dim + 1, current_coords, original_shape);
        current_coords.pop_back();
      }
    }

    void extract_elements_recursive(
        const cytnx::Tensor& src, cytnx::Tensor& dst, std::size_t dim,
        std::vector<cytnx::cytnx_uint64> src_coords, std::vector<cytnx::cytnx_uint64> dst_coords,
        const List<Pair<cytnx::cytnx_uint64, cytnx::cytnx_uint64>>& coor_pairs) {
      if (dim == coor_pairs.size()) {
        // Base case: copy element
        auto elem = src.at(src_coords);
        dst.at(dst_coords) = elem;
        return;
      }

      auto [start, end] = coor_pairs[dim];
      for (cytnx::cytnx_uint64 i = start; i < end; ++i) {
        src_coords.push_back(i);
        dst_coords.push_back(i - start);
        extract_elements_recursive(src, dst, dim + 1, src_coords, dst_coords, coor_pairs);
        src_coords.pop_back();
        dst_coords.pop_back();
      }
    }

    void replace_elements_recursive(cytnx::Tensor& main_tensor, const cytnx::Tensor& sub_tensor,
                                    std::size_t dim, const std::vector<cytnx::cytnx_uint64>& begin_pt,
                                    std::vector<cytnx::cytnx_uint64>& sub_coords,
                                    const std::vector<cytnx::cytnx_uint64>& sub_shape) {
      if (dim == sub_shape.size()) {
        // Base case: copy element from sub to main
        std::vector<cytnx::cytnx_uint64> main_coords;
        for (std::size_t i = 0; i < begin_pt.size(); ++i) {
          main_coords.push_back(begin_pt[i] + sub_coords[i]);
        }
        auto elem = sub_tensor.at(sub_coords);
        main_tensor.at(main_coords) = elem;
        return;
      }

      for (cytnx::cytnx_uint64 i = 0; i < sub_shape[dim]; ++i) {
        sub_coords[dim] = i;
        replace_elements_recursive(main_tensor, sub_tensor, dim + 1, begin_pt, sub_coords,
                                   sub_shape);
      }
    }
  }  // anonymous namespace

  template <typename ElemT>
  void expand(context_handle_t<CytnxTensor<ElemT>>& ctx,
              CytnxTensor<ElemT>& inout,
              const Map<bond_idx_t<CytnxTensor<ElemT>>, bond_dim_t<CytnxTensor<ElemT>>>& bond_idx_increment_map) {
    auto original_shape = inout.backend.shape();
    std::vector<cytnx::cytnx_uint64> new_shape(original_shape.begin(), original_shape.end());

    // Apply increments to shape
    for (const auto& [bond_idx, increment] : bond_idx_increment_map) {
      if (bond_idx >= new_shape.size()) {
        throw std::invalid_argument("Bond index out of range");
      }
      new_shape[bond_idx] += increment;
    }

    // Create new tensor with expanded shape, initialized to zero
    cytnx::Tensor expanded = cytnx::zeros(new_shape, inout.backend.dtype(), ctx);

    // Copy original data to the beginning of each expanded dimension
    auto original_coords = original_shape;
    copy_original_data_recursive(inout.backend, expanded, 0, {}, original_coords);

    inout.backend = std::move(expanded);
  }

  template <typename ElemT>
  void expand(context_handle_t<CytnxTensor<ElemT>>& ctx,
              const CytnxTensor<ElemT>& in,
              const Map<bond_idx_t<CytnxTensor<ElemT>>, bond_dim_t<CytnxTensor<ElemT>>>& bond_idx_increment_map,
              CytnxTensor<ElemT>& out) {
    out = in;
    expand(ctx, out, bond_idx_increment_map);
  }

  // shrink
  // Restored from git show b7ecb2a9^:source/tensor_manipulation.cpp
  template <typename ElemT>
  void shrink(context_handle_t<CytnxTensor<ElemT>>& ctx,
              CytnxTensor<ElemT>& inout,
              const bond_idx_elem_coor_pair_map<CytnxTensor<ElemT>>& bd_idx_el_coor_pair_map) {
    auto original_shape = inout.backend.shape();

    // Build coordinate pairs list from the map
    List<Pair<elem_coor_t<CytnxTensor<ElemT>>, elem_coor_t<CytnxTensor<ElemT>>>> coor_pairs;
    coor_pairs.resize(original_shape.size());

    // Initialize with full ranges
    for (std::size_t i = 0; i < original_shape.size(); ++i) {
      coor_pairs[i] = {0, original_shape[i]};
    }

    // Apply shrinking ranges from the map
    for (const auto& [bond_idx, coor_pair] : bd_idx_el_coor_pair_map) {
      if (bond_idx >= original_shape.size()) {
        throw std::invalid_argument("Bond index out of range");
      }
      coor_pairs[bond_idx] = coor_pair;
    }

    // Use extract_sub to perform the shrinking
    extract_sub(ctx, inout, coor_pairs);
  }

  template <typename ElemT>
  void shrink(context_handle_t<CytnxTensor<ElemT>>& ctx,
              const CytnxTensor<ElemT>& in,
              const bond_idx_elem_coor_pair_map<CytnxTensor<ElemT>>& bd_idx_el_coor_pair_map,
              CytnxTensor<ElemT>& out) {
    out = in;
    shrink(ctx, out, bd_idx_el_coor_pair_map);
  }

  // extract_sub
  // Restored from git show b7ecb2a9^:source/tensor_manipulation.cpp
  template <typename ElemT>
  void extract_sub(context_handle_t<CytnxTensor<ElemT>>& ctx,
                   CytnxTensor<ElemT>& inout,
                   const List<Pair<elem_coor_t<CytnxTensor<ElemT>>, elem_coor_t<CytnxTensor<ElemT>>>>& coor_pairs) {
    auto original_shape = inout.backend.shape();

    if (coor_pairs.size() != original_shape.size()) {
      throw std::invalid_argument("Number of coordinate pairs must match tensor rank");
    }

    // Calculate new shape
    std::vector<cytnx::cytnx_uint64> new_shape;
    for (std::size_t i = 0; i < coor_pairs.size(); ++i) {
      auto [start, end] = coor_pairs[i];
      if (start >= end || end > original_shape[i]) {
        throw std::invalid_argument("Invalid coordinate range");
      }
      new_shape.push_back(end - start);
    }

    // Create result tensor
    cytnx::Tensor result = cytnx::zeros(new_shape, inout.backend.dtype(), ctx);

    // Extract sub-tensor by copying elements
    extract_elements_recursive(inout.backend, result, 0, {}, {}, coor_pairs);

    inout.backend = std::move(result);
  }

  template <typename ElemT>
  void extract_sub(context_handle_t<CytnxTensor<ElemT>>& ctx,
                   const CytnxTensor<ElemT>& in,
                   const List<Pair<elem_coor_t<CytnxTensor<ElemT>>, elem_coor_t<CytnxTensor<ElemT>>>>& coor_pairs,
                   CytnxTensor<ElemT>& out) {
    out = in;
    extract_sub(ctx, out, coor_pairs);
  }

  // replace_sub
  // Restored from git show b7ecb2a9^:source/tensor_manipulation.cpp
  template <typename ElemT>
  void replace_sub(context_handle_t<CytnxTensor<ElemT>>& ctx,
                   CytnxTensor<ElemT>& inout,
                   const CytnxTensor<ElemT>& sub,
                   const elem_coors_t<CytnxTensor<ElemT>>& begin_pt) {
    auto main_shape = inout.backend.shape();
    auto sub_shape = sub.backend.shape();

    if (begin_pt.size() != main_shape.size() || sub_shape.size() != main_shape.size()) {
      throw std::invalid_argument("Dimension mismatch");
    }

    // Check bounds
    for (std::size_t i = 0; i < begin_pt.size(); ++i) {
      if (begin_pt[i] + sub_shape[i] > main_shape[i]) {
        throw std::invalid_argument("Sub-tensor exceeds bounds");
      }
    }

    // Replace elements recursively
    std::vector<cytnx::cytnx_uint64> sub_coords(sub_shape.size(), 0);
    replace_elements_recursive(inout.backend, sub.backend, 0, begin_pt, sub_coords, sub_shape);
  }

  template <typename ElemT>
  void replace_sub(context_handle_t<CytnxTensor<ElemT>>& ctx,
                   const CytnxTensor<ElemT>& in,
                   const CytnxTensor<ElemT>& sub,
                   const elem_coors_t<CytnxTensor<ElemT>>& begin_pt,
                   CytnxTensor<ElemT>& out) {
    out = in;
    replace_sub(ctx, out, sub, begin_pt);
  }

  // concatenate
  // Restored from git show b7ecb2a9^:source/tensor_manipulation.cpp
  template <typename ElemT>
  void concatenate(context_handle_t<CytnxTensor<ElemT>>& ctx,
                   const List<CytnxTensor<ElemT>>& ins,
                   const bond_idx_t<CytnxTensor<ElemT>> axis,
                   CytnxTensor<ElemT>& out) {
    if (ins.empty()) {
      throw std::invalid_argument("Cannot concatenate empty list of tensors");
    }

    const auto& first = ins[0].backend;
    auto first_shape = first.shape();

    if (axis >= first_shape.size()) {
      throw std::invalid_argument("axis exceeds tensor rank");
    }

    // Check all tensors have compatible shapes
    size_t total_concat_dim = 0;
    for (size_t i = 0; i < ins.size(); ++i) {
      const auto& tensor = ins[i].backend;
      auto shape = tensor.shape();

      if (shape.size() != first_shape.size()) {
        throw std::invalid_argument("All tensors must have the same rank");
      }

      for (size_t j = 0; j < shape.size(); ++j) {
        if (j != axis && shape[j] != first_shape[j]) {
          throw std::invalid_argument("All tensors must have the same shape except along concat dimension");
        }
      }

      total_concat_dim += shape[axis];
    }

    // Calculate output shape
    auto out_shape = first_shape;
    out_shape[axis] = total_concat_dim;

    // Create output tensor
    out.backend = cytnx::zeros(out_shape, first.dtype(), first.device());

    // Copy data from each input tensor to appropriate slice of output
    size_t offset = 0;
    for (const auto& tensor : ins) {
      auto tensor_shape = tensor.backend.shape();

      // Create coordinate vectors for slicing
      std::vector<cytnx::Accessor> accessors(out_shape.size());
      for (size_t i = 0; i < out_shape.size(); ++i) {
        if (i == axis) {
          accessors[i] = cytnx::Accessor::range(offset, offset + tensor_shape[i]);
        } else {
          accessors[i] = cytnx::Accessor::all();
        }
      }

      // Assign the tensor to the appropriate slice
      out.backend.set(accessors, tensor.backend);
      offset += tensor_shape[axis];
    }
  }

  // stack
  // Restored from git show b7ecb2a9^:source/tensor_manipulation.cpp
  template <typename ElemT>
  void stack(context_handle_t<CytnxTensor<ElemT>>& ctx,
             const List<CytnxTensor<ElemT>>& ins,
             const bond_idx_t<CytnxTensor<ElemT>> axis,
             CytnxTensor<ElemT>& out) {
    if (ins.empty()) {
      throw std::invalid_argument("Cannot stack empty list of tensors");
    }

    // Implement stacking by creating a new dimension at axis
    // First verify all tensors have the same shape
    const auto& first_shape = ins[0].backend.shape();
    for (size_t i = 1; i < ins.size(); ++i) {
      if (ins[i].backend.shape() != first_shape) {
        throw std::invalid_argument("All tensors must have the same shape for stacking");
      }
    }

    // Create new shape with additional dimension for stacking
    std::vector<cytnx::cytnx_uint64> new_shape;
    for (size_t i = 0; i < first_shape.size(); ++i) {
      if (i == static_cast<size_t>(axis)) {
        new_shape.push_back(ins.size());  // Number of tensors to stack
      }
      new_shape.push_back(first_shape[i]);
    }
    // Handle case where axis is at the end
    if (static_cast<size_t>(axis) >= first_shape.size()) {
      new_shape.push_back(ins.size());
    }

    // Create output tensor
    out.backend = cytnx::zeros(new_shape, ins[0].backend.dtype(), ins[0].backend.device());

    // Copy data from each tensor
    for (size_t tensor_idx = 0; tensor_idx < ins.size(); ++tensor_idx) {
      // Create index for where to place this tensor in the stacked result
      std::vector<cytnx::Accessor> accessors;

      for (size_t i = 0; i < new_shape.size(); ++i) {
        if (i == static_cast<size_t>(axis)) {
          accessors.push_back(cytnx::Accessor(static_cast<cytnx::cytnx_int64>(tensor_idx)));
        } else {
          accessors.push_back(cytnx::Accessor::all());
        }
      }

      // Copy the tensor data to the appropriate slice using set method
      out.backend.set(accessors, ins[tensor_idx].backend);
    }
  }

  // ========================================================================
  // for_each_with_coors implementation for CytnxTensor<ElemT>
  // ========================================================================

  namespace detail {
    // Helper function for for_each_with_coors (mutable version)
    template <typename ElemT, typename Func>
    void for_each_recursive_typed(CytnxTensor<ElemT>& tensor, Func&& f, std::size_t dim,
                                  std::vector<cytnx::cytnx_uint64>& coords,
                                  const std::vector<cytnx::cytnx_uint64>& shape) {
      if (dim == shape.size()) {
        // Base case: apply function to element at current coordinates
        auto& elem = tensor.backend.template at<ElemT>(coords);
        // Convert coords to elem_coors_t<CytnxTensor<ElemT>>
        elem_coors_t<CytnxTensor<ElemT>> tci_coords;
        tci_coords.reserve(coords.size());
        for (const auto& coord : coords) {
          tci_coords.push_back(static_cast<elem_coor_t<CytnxTensor<ElemT>>>(coord));
        }
        std::invoke(f, elem, tci_coords);
      } else {
        // Recursive case: iterate through current dimension
        for (cytnx::cytnx_uint64 i = 0; i < shape[dim]; ++i) {
          coords.push_back(i);
          for_each_recursive_typed(tensor, std::forward<Func>(f), dim + 1, coords, shape);
          coords.pop_back();
        }
      }
    }

    // Helper function for for_each_with_coors (const version)
    template <typename ElemT, typename Func>
    void for_each_recursive_const_typed(const CytnxTensor<ElemT>& tensor, Func&& f, std::size_t dim,
                                        std::vector<cytnx::cytnx_uint64>& coords,
                                        const std::vector<cytnx::cytnx_uint64>& shape) {
      if (dim == shape.size()) {
        // Base case: apply function to element at current coordinates
        const auto& elem = tensor.backend.template at<ElemT>(coords);
        // Convert coords to elem_coors_t<CytnxTensor<ElemT>>
        elem_coors_t<CytnxTensor<ElemT>> tci_coords;
        tci_coords.reserve(coords.size());
        for (const auto& coord : coords) {
          tci_coords.push_back(static_cast<elem_coor_t<CytnxTensor<ElemT>>>(coord));
        }
        std::invoke(f, elem, tci_coords);
      } else {
        // Recursive case: iterate through current dimension
        for (cytnx::cytnx_uint64 i = 0; i < shape[dim]; ++i) {
          coords.push_back(i);
          for_each_recursive_const_typed(tensor, std::forward<Func>(f), dim + 1, coords, shape);
          coords.pop_back();
        }
      }
    }
  }  // namespace detail

  // for_each_with_coors for CytnxTensor<ElemT> (mutable version)
  template <typename ElemT, typename Func>
  void for_each_with_coors(context_handle_t<CytnxTensor<ElemT>>& ctx, CytnxTensor<ElemT>& inout, Func&& f) {
    auto shape = inout.backend.shape();
    std::vector<cytnx::cytnx_uint64> coords;
    coords.reserve(shape.size());

    detail::for_each_recursive_typed(inout, std::forward<Func>(f), 0, coords, shape);
  }

  // for_each_with_coors for CytnxTensor<ElemT> (const version)
  template <typename ElemT, typename Func>
  void for_each_with_coors(context_handle_t<CytnxTensor<ElemT>>& ctx, const CytnxTensor<ElemT>& in, Func&& f) {
    auto shape = in.backend.shape();
    std::vector<cytnx::cytnx_uint64> coords;
    coords.reserve(shape.size());

    detail::for_each_recursive_const_typed(in, std::forward<Func>(f), 0, coords, shape);
  }

  // move - move tensor contents (in-place)
  template <typename ElemT>
  void move(context_handle_t<CytnxTensor<ElemT>>& ctx,
            CytnxTensor<ElemT>& from,
            CytnxTensor<ElemT>& to) {
    to.backend = std::move(from.backend);
  }

  // move - move tensor contents (out-of-place)
  template <typename ElemT>
  CytnxTensor<ElemT> move(context_handle_t<CytnxTensor<ElemT>>& ctx,
                          CytnxTensor<ElemT>& from) {
    CytnxTensor<ElemT> result;
    move(ctx, from, result);
    return result;
  }

  // to_cplx - convert to complex tensor (out-of-place)
  template <typename ElemT>
  cplx_ten_t<CytnxTensor<ElemT>> to_cplx(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                         const CytnxTensor<ElemT>& in) {
    cplx_ten_t<CytnxTensor<ElemT>> result;
    to_cplx(ctx, in, result);
    return result;
  }

  // contract - tensor contraction (string version)
  template <typename ElemT>
  void contract(context_handle_t<CytnxTensor<ElemT>>& ctx,
                const CytnxTensor<ElemT>& a,
                const std::string_view bd_labs_str_a,
                const CytnxTensor<ElemT>& b,
                const std::string_view bd_labs_str_b,
                CytnxTensor<ElemT>& c,
                const std::string_view bd_labs_str_c) {
    List<bond_label_t<CytnxTensor<ElemT>>> bd_labs_a, bd_labs_b, bd_labs_c;
    for (char ch : bd_labs_str_a) {
      bd_labs_a.push_back(static_cast<bond_label_t<CytnxTensor<ElemT>>>(ch));
    }
    for (char ch : bd_labs_str_b) {
      bd_labs_b.push_back(static_cast<bond_label_t<CytnxTensor<ElemT>>>(ch));
    }
    for (char ch : bd_labs_str_c) {
      bd_labs_c.push_back(static_cast<bond_label_t<CytnxTensor<ElemT>>>(ch));
    }
    contract(ctx, a, bd_labs_a, b, bd_labs_b, c, bd_labs_c);
  }

  // Explicit specializations for zeros (out-of-place) for all supported element types
  template <>
  inline CytnxTensor<cytnx::cytnx_double> zeros<CytnxTensor<cytnx::cytnx_double>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_double>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_double>>& shape) {
    CytnxTensor<cytnx::cytnx_double> result;
    zeros(ctx, shape, result);
    return result;
  }

  template <>
  inline CytnxTensor<cytnx::cytnx_float> zeros<CytnxTensor<cytnx::cytnx_float>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_float>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_float>>& shape) {
    CytnxTensor<cytnx::cytnx_float> result;
    zeros(ctx, shape, result);
    return result;
  }

  template <>
  inline CytnxTensor<cytnx::cytnx_complex128> zeros<CytnxTensor<cytnx::cytnx_complex128>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_complex128>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_complex128>>& shape) {
    CytnxTensor<cytnx::cytnx_complex128> result;
    zeros(ctx, shape, result);
    return result;
  }

  template <>
  inline CytnxTensor<cytnx::cytnx_complex64> zeros<CytnxTensor<cytnx::cytnx_complex64>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_complex64>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_complex64>>& shape) {
    CytnxTensor<cytnx::cytnx_complex64> result;
    zeros(ctx, shape, result);
    return result;
  }

  // Explicit specializations for eye (out-of-place) for all supported element types
  template <>
  inline CytnxTensor<cytnx::cytnx_double> eye<CytnxTensor<cytnx::cytnx_double>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_double>>& ctx,
      const bond_dim_t<CytnxTensor<cytnx::cytnx_double>> N) {
    CytnxTensor<cytnx::cytnx_double> result;
    eye(ctx, N, result);
    return result;
  }

  template <>
  inline CytnxTensor<cytnx::cytnx_float> eye<CytnxTensor<cytnx::cytnx_float>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_float>>& ctx,
      const bond_dim_t<CytnxTensor<cytnx::cytnx_float>> N) {
    CytnxTensor<cytnx::cytnx_float> result;
    eye(ctx, N, result);
    return result;
  }

  template <>
  inline CytnxTensor<cytnx::cytnx_complex128> eye<CytnxTensor<cytnx::cytnx_complex128>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_complex128>>& ctx,
      const bond_dim_t<CytnxTensor<cytnx::cytnx_complex128>> N) {
    CytnxTensor<cytnx::cytnx_complex128> result;
    eye(ctx, N, result);
    return result;
  }

  template <>
  inline CytnxTensor<cytnx::cytnx_complex64> eye<CytnxTensor<cytnx::cytnx_complex64>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_complex64>>& ctx,
      const bond_dim_t<CytnxTensor<cytnx::cytnx_complex64>> N) {
    CytnxTensor<cytnx::cytnx_complex64> result;
    eye(ctx, N, result);
    return result;
  }
}  // namespace tci