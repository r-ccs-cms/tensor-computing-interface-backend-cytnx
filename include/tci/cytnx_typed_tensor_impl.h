#pragma once

#include "tci/cytnx_typed_tensor.h"
#include "tci/cytnx_tensor_traits.h"
#include "tci/tensor_traits.h"
#include "tci/tensor_linear_algebra_impl.h"
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
    return tci::size_bytes(ctx, a.backend);
  }

  template <typename ElemT, typename RandNumGen>
  void random(context_handle_t<CytnxTensor<ElemT>>& ctx,
              const shape_t<CytnxTensor<ElemT>>& shape,
              RandNumGen&& gen,
              CytnxTensor<ElemT>& a) {
    allocate(ctx, shape, a);
    auto total_size = static_cast<cytnx::cytnx_uint64>(a.backend.storage().size());
    auto* data = a.backend.storage().template data<ElemT>();

    std::uniform_real_distribution<real_t<CytnxTensor<ElemT>>> dist(0.0, 1.0);
    for (cytnx::cytnx_uint64 i = 0; i < total_size; ++i) {
      if constexpr (std::is_same_v<ElemT, cytnx::cytnx_complex128> ||
                    std::is_same_v<ElemT, cytnx::cytnx_complex64>) {
        using RealType = typename ElemT::value_type;
        data[i] = ElemT(dist(gen), dist(gen));
      } else {
        data[i] = dist(gen);
      }
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
    cytnx::Tensor out_backend;
    tci::to_cplx(ctx, in.backend, out_backend);
    out.backend = out_backend;
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
  template <typename ElemT>
  void contract(context_handle_t<CytnxTensor<ElemT>>& ctx,
                const CytnxTensor<ElemT>& a,
                const std::vector<bond_label_t<CytnxTensor<ElemT>>>& bd_labs_a,
                const CytnxTensor<ElemT>& b,
                const std::vector<bond_label_t<CytnxTensor<ElemT>>>& bd_labs_b,
                CytnxTensor<ElemT>& c,
                const std::vector<bond_label_t<CytnxTensor<ElemT>>>& bd_labs_c) {
    context_handle_t<cytnx::Tensor> backend_ctx = ctx;
    tci::contract(backend_ctx, a.backend, bd_labs_a, b.backend, bd_labs_b, c.backend, bd_labs_c);
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

  // SVD (full) - Frontend (CytnxTensor) thin adapter delegating to backend
  template <typename ElemT>
  void svd(context_handle_t<CytnxTensor<ElemT>>& ctx,
           const CytnxTensor<ElemT>& a,
           const rank_t<CytnxTensor<ElemT>> num_of_bds_as_row,
           CytnxTensor<ElemT>& u,
           real_ten_t<CytnxTensor<ElemT>>& s_diag,
           CytnxTensor<ElemT>& v_dag) {
    // Delegate to backend (cytnx::Tensor) implementation
    context_handle_t<cytnx::Tensor> backend_ctx = ctx;
    tci::svd(backend_ctx, a.backend, num_of_bds_as_row,
             u.backend, s_diag.backend, v_dag.backend);
  }

  // QR decomposition - Frontend (CytnxTensor) thin adapter delegating to backend
  template <typename ElemT>
  void qr(context_handle_t<CytnxTensor<ElemT>>& ctx,
          const CytnxTensor<ElemT>& a,
          const rank_t<CytnxTensor<ElemT>> num_of_bds_as_row,
          CytnxTensor<ElemT>& q,
          CytnxTensor<ElemT>& r) {
    // Delegate to backend (cytnx::Tensor) implementation
    context_handle_t<cytnx::Tensor> backend_ctx = ctx;
    tci::qr(backend_ctx, a.backend, num_of_bds_as_row,
            q.backend, r.backend);
  }

  // LQ decomposition - Frontend (CytnxTensor) thin adapter delegating to backend
  template <typename ElemT>
  void lq(context_handle_t<CytnxTensor<ElemT>>& ctx,
          const CytnxTensor<ElemT>& a,
          const rank_t<CytnxTensor<ElemT>> num_of_bds_as_row,
          CytnxTensor<ElemT>& l,
          CytnxTensor<ElemT>& q) {
    // Delegate to backend (cytnx::Tensor) implementation
    context_handle_t<cytnx::Tensor> backend_ctx = ctx;
    tci::lq(backend_ctx, a.backend, num_of_bds_as_row,
            l.backend, q.backend);
  }

  // Frontend (CytnxTensor) - thin adapter delegating to backend (Backend Unification Pattern)
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
    // Delegate to backend (cytnx::Tensor) implementation
    context_handle_t<cytnx::Tensor> backend_ctx = ctx;
    tci::trunc_svd(backend_ctx, a.backend, num_of_bds_as_row,
                   u.backend, s_diag.backend, v_dag.backend,
                   trunc_err, chi_max, s_min);
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
    context_handle_t<cytnx::Tensor> backend_ctx = ctx;
    elem_t<cytnx::Tensor> epsilon_variant = epsilon;
    return tci::eq(backend_ctx, a.backend, b.backend, epsilon_variant);
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

  // Tensor Manipulation functions - delegate to cytnx::Tensor backend

  // expand
  template <typename ElemT>
  void expand(context_handle_t<CytnxTensor<ElemT>>& ctx,
              CytnxTensor<ElemT>& inout,
              const Map<bond_idx_t<CytnxTensor<ElemT>>, bond_dim_t<CytnxTensor<ElemT>>>& bond_idx_increment_map) {
    context_handle_t<cytnx::Tensor> backend_ctx = ctx;
    tci::expand(backend_ctx, inout.backend, bond_idx_increment_map);
  }

  template <typename ElemT>
  void expand(context_handle_t<CytnxTensor<ElemT>>& ctx,
              const CytnxTensor<ElemT>& in,
              const Map<bond_idx_t<CytnxTensor<ElemT>>, bond_dim_t<CytnxTensor<ElemT>>>& bond_idx_increment_map,
              CytnxTensor<ElemT>& out) {
    context_handle_t<cytnx::Tensor> backend_ctx = ctx;
    tci::expand(backend_ctx, in.backend, bond_idx_increment_map, out.backend);
  }

  // shrink
  template <typename ElemT>
  void shrink(context_handle_t<CytnxTensor<ElemT>>& ctx,
              CytnxTensor<ElemT>& inout,
              const bond_idx_elem_coor_pair_map<CytnxTensor<ElemT>>& bd_idx_el_coor_pair_map) {
    context_handle_t<cytnx::Tensor> backend_ctx = ctx;
    tci::shrink(backend_ctx, inout.backend, bd_idx_el_coor_pair_map);
  }

  template <typename ElemT>
  void shrink(context_handle_t<CytnxTensor<ElemT>>& ctx,
              const CytnxTensor<ElemT>& in,
              const bond_idx_elem_coor_pair_map<CytnxTensor<ElemT>>& bd_idx_el_coor_pair_map,
              CytnxTensor<ElemT>& out) {
    context_handle_t<cytnx::Tensor> backend_ctx = ctx;
    tci::shrink(backend_ctx, in.backend, bd_idx_el_coor_pair_map, out.backend);
  }

  // extract_sub
  template <typename ElemT>
  void extract_sub(context_handle_t<CytnxTensor<ElemT>>& ctx,
                   CytnxTensor<ElemT>& inout,
                   const List<Pair<elem_coor_t<CytnxTensor<ElemT>>, elem_coor_t<CytnxTensor<ElemT>>>>& coor_pairs) {
    context_handle_t<cytnx::Tensor> backend_ctx = ctx;
    tci::extract_sub(backend_ctx, inout.backend, coor_pairs);
  }

  template <typename ElemT>
  void extract_sub(context_handle_t<CytnxTensor<ElemT>>& ctx,
                   const CytnxTensor<ElemT>& in,
                   const List<Pair<elem_coor_t<CytnxTensor<ElemT>>, elem_coor_t<CytnxTensor<ElemT>>>>& coor_pairs,
                   CytnxTensor<ElemT>& out) {
    context_handle_t<cytnx::Tensor> backend_ctx = ctx;
    tci::extract_sub(backend_ctx, in.backend, coor_pairs, out.backend);
  }

  // replace_sub
  template <typename ElemT>
  void replace_sub(context_handle_t<CytnxTensor<ElemT>>& ctx,
                   CytnxTensor<ElemT>& inout,
                   const CytnxTensor<ElemT>& sub,
                   const elem_coors_t<CytnxTensor<ElemT>>& begin_pt) {
    context_handle_t<cytnx::Tensor> backend_ctx = ctx;
    tci::replace_sub(backend_ctx, inout.backend, sub.backend, begin_pt);
  }

  template <typename ElemT>
  void replace_sub(context_handle_t<CytnxTensor<ElemT>>& ctx,
                   const CytnxTensor<ElemT>& in,
                   const CytnxTensor<ElemT>& sub,
                   const elem_coors_t<CytnxTensor<ElemT>>& begin_pt,
                   CytnxTensor<ElemT>& out) {
    context_handle_t<cytnx::Tensor> backend_ctx = ctx;
    tci::replace_sub(backend_ctx, in.backend, sub.backend, begin_pt, out.backend);
  }

  // concatenate
  template <typename ElemT>
  void concatenate(context_handle_t<CytnxTensor<ElemT>>& ctx,
                   const List<CytnxTensor<ElemT>>& ins,
                   const bond_idx_t<CytnxTensor<ElemT>> axis,
                   CytnxTensor<ElemT>& out) {
    // Convert CytnxTensor list to cytnx::Tensor list
    List<cytnx::Tensor> backend_tensors;
    for (const auto& tensor : ins) {
      backend_tensors.push_back(tensor.backend);
    }

    context_handle_t<cytnx::Tensor> backend_ctx = ctx;
    tci::concatenate(backend_ctx, backend_tensors, axis, out.backend);
  }

  // stack
  template <typename ElemT>
  void stack(context_handle_t<CytnxTensor<ElemT>>& ctx,
             const List<CytnxTensor<ElemT>>& ins,
             const bond_idx_t<CytnxTensor<ElemT>> axis,
             CytnxTensor<ElemT>& out) {
    // Convert CytnxTensor list to cytnx::Tensor list
    List<cytnx::Tensor> backend_tensors;
    for (const auto& tensor : ins) {
      backend_tensors.push_back(tensor.backend);
    }

    context_handle_t<cytnx::Tensor> backend_ctx = ctx;
    tci::stack(backend_ctx, backend_tensors, axis, out.backend);
  }

}  // namespace tci