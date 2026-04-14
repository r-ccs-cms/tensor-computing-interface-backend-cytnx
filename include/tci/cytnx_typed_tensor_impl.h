#pragma once

#include <cytnx.hpp>
#include <limits>
#include <vector>

#include "tci/construction_destruction.h"
#include "tci/cytnx_tensor_traits.h"
#include "tci/cytnx_typed_tensor.h"
#include "tci/tensor_linear_algebra.h"
#include "tci/tensor_traits.h"

namespace tci {

  // Helper: Convert elem_t<TenT> to cytnx::Type
  namespace detail {
    template <typename ElemT> constexpr unsigned int elem_to_cytnx_type() {
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

  // Read-only getter functions for TenT
  template <typename TenT> void get_elem(context_handle_t<TenT>& ctx, const TenT& a,
                                         const elem_coors_t<TenT>& coors, elem_t<TenT>& elem);

  template <typename TenT> elem_t<TenT> get_elem(context_handle_t<TenT>& ctx, const TenT& a,
                                                 const elem_coors_t<TenT>& coors);

  // Tensor manipulation functions for TenT
  template <typename TenT, typename Func>
  void for_each(context_handle_t<TenT>& ctx, TenT& inout, Func&& f);

  template <typename TenT, typename Func>
  void for_each(context_handle_t<TenT>& ctx, const TenT& in, Func&& f);

  // Template implementations

  template <typename TenT> elem_t<TenT> get_elem(context_handle_t<TenT>& ctx, const TenT& a,
                                                 const elem_coors_t<TenT>& coors) {
    std::vector<cytnx::cytnx_uint64> cytnx_coors;
    cytnx_coors.reserve(coors.size());
    for (const auto& coord : coors) {
      cytnx_coors.push_back(static_cast<cytnx::cytnx_uint64>(coord));
    }
    return a.backend.template at<elem_t<TenT>>(cytnx_coors);
  }

  // Void overload (reserved for future GPU support)
  template <typename TenT> void get_elem(context_handle_t<TenT>& ctx, const TenT& a,
                                         const elem_coors_t<TenT>& coors, elem_t<TenT>& elem) {
    // Runtime warning (emitted once per program execution)
    static bool warned = false;
    if (!warned) {
      // Check suppression flag
      const char* suppress = std::getenv("TCI_SUPPRESS_FUTURE_API_WARNING");

      // Show warning by default, unless explicitly suppressed
      if (!suppress || std::atoi(suppress) == 0) {
        std::cerr
            << "[TCI Warning] tci::get_elem void overload is reserved for future GPU support.\n"
            << "  This call is currently forwarded to the return-value version.\n"
            << "  Recommended migration: auto elem = tci::get_elem(ctx, a, coors);\n"
            << "  To suppress this warning: export TCI_SUPPRESS_FUTURE_API_WARNING=1\n";
      }
      warned = true;
    }

    // Forward to current API
    elem = get_elem(ctx, a, coors);
  }

  // for_each implementation for TenT (mutable version)
  template <typename TenT, typename Func>
  void for_each(context_handle_t<TenT>& ctx, TenT& inout, Func&& f) {
    auto total_size = static_cast<cytnx::cytnx_uint64>(inout.backend.storage().size());

    // Direct access to underlying storage for performance
    auto* data = inout.backend.storage().template data<elem_t<TenT>>();

    for (cytnx::cytnx_uint64 i = 0; i < total_size; ++i) {
      f(data[i]);
    }
  }

  // for_each implementation for TenT (const version)
  template <typename TenT, typename Func>
  void for_each(context_handle_t<TenT>& ctx, const TenT& in, Func&& f) {
    auto total_size = static_cast<cytnx::cytnx_uint64>(in.backend.storage().size());

    // Direct access to underlying storage for performance
    const auto* data = in.backend.storage().template data<elem_t<TenT>>();

    for (cytnx::cytnx_uint64 i = 0; i < total_size; ++i) {
      f(data[i]);
    }
  }

  template <typename TenT> void set_elem(context_handle_t<TenT>& ctx, TenT& a,
                                         const elem_coors_t<TenT>& coors, elem_t<TenT> elem) {
    std::vector<cytnx::cytnx_uint64> cytnx_coors;
    cytnx_coors.reserve(coors.size());
    for (const auto& coord : coors) {
      cytnx_coors.push_back(static_cast<cytnx::cytnx_uint64>(coord));
    }
    a.backend.template at<elem_t<TenT>>(cytnx_coors) = elem;
  }

  template <typename TenT> shape_t<TenT> shape(context_handle_t<TenT>& ctx, const TenT& a) {
    auto cytnx_shape = a.backend.shape();
    shape_t<TenT> result;
    result.reserve(cytnx_shape.size());
    for (const auto& dim : cytnx_shape) {
      result.push_back(static_cast<bond_dim_t<TenT>>(dim));
    }
    return result;
  }

  template <typename TenT> order_t<TenT> order(context_handle_t<TenT>& ctx, const TenT& a) {
    return static_cast<order_t<TenT>>(a.backend.shape().size());
  }

  template <typename TenT> ten_size_t<TenT> size(context_handle_t<TenT>& ctx, const TenT& a) {
    return static_cast<ten_size_t<TenT>>(a.backend.storage().size());
  }

  template <typename TenT> std::size_t size_bytes(context_handle_t<TenT>& ctx, const TenT& a) {
    // Calculate total bytes using dtype element size
    auto dtype = a.backend.dtype();
    std::size_t elem_size = 0;
    if (dtype == cytnx::Type.Float)
      elem_size = sizeof(float);
    else if (dtype == cytnx::Type.Double)
      elem_size = sizeof(double);
    else if (dtype == cytnx::Type.ComplexFloat)
      elem_size = sizeof(cytnx::cytnx_complex64);
    else if (dtype == cytnx::Type.ComplexDouble)
      elem_size = sizeof(cytnx::cytnx_complex128);
    else if (dtype == cytnx::Type.Int64)
      elem_size = sizeof(cytnx::cytnx_int64);
    else if (dtype == cytnx::Type.Uint64)
      elem_size = sizeof(cytnx::cytnx_uint64);
    else if (dtype == cytnx::Type.Int32)
      elem_size = sizeof(cytnx::cytnx_int32);
    else if (dtype == cytnx::Type.Uint32)
      elem_size = sizeof(cytnx::cytnx_uint32);
    return a.backend.storage().size() * elem_size;
  }

  template <typename TenT> void show(context_handle_t<TenT>& ctx, const TenT& a) {
    std::cout << a.backend << std::endl;
  }

  // Copy operation
  template <typename TenT> TenT copy(context_handle_t<TenT>& ctx, const TenT& orig) {
    TenT result;
    result.backend = orig.backend.clone();
    return result;
  }

  // Clear operation
  template <typename TenT> void clear(context_handle_t<TenT>& ctx, TenT& a) {
    // Create empty tensor
    a.backend = cytnx::Tensor();
  }

  // Reshape operation
  template <typename TenT>
  void reshape(context_handle_t<TenT>& ctx, TenT& inout, const shape_t<TenT>& new_shape) {
    std::vector<cytnx::cytnx_int64> cytnx_shape;
    cytnx_shape.reserve(new_shape.size());
    for (const auto& dim : new_shape) {
      cytnx_shape.push_back(static_cast<cytnx::cytnx_int64>(dim));
    }
    inout.backend = inout.backend.reshape(cytnx_shape);
  }

  template <typename TenT> void reshape(context_handle_t<TenT>& ctx, const TenT& in,
                                        const shape_t<TenT>& new_shape, TenT& out) {
    std::vector<cytnx::cytnx_int64> cytnx_shape;
    cytnx_shape.reserve(new_shape.size());
    for (const auto& dim : new_shape) {
      cytnx_shape.push_back(static_cast<cytnx::cytnx_int64>(dim));
    }
    out.backend = in.backend.reshape(cytnx_shape);
  }

  // Transpose operation
  template <typename TenT> void transpose(context_handle_t<TenT>& ctx, TenT& inout,
                                          const std::vector<bond_idx_t<TenT>>& new_order) {
    std::vector<cytnx::cytnx_uint64> cytnx_order;
    cytnx_order.reserve(new_order.size());
    for (const auto& idx : new_order) {
      cytnx_order.push_back(static_cast<cytnx::cytnx_uint64>(idx));
    }
    inout.backend = inout.backend.permute(cytnx_order);
  }

  template <typename TenT> void transpose(context_handle_t<TenT>& ctx, const TenT& in,
                                          const std::vector<bond_idx_t<TenT>>& new_order,
                                          TenT& out) {
    std::vector<cytnx::cytnx_uint64> cytnx_order;
    cytnx_order.reserve(new_order.size());
    for (const auto& idx : new_order) {
      cytnx_order.push_back(static_cast<cytnx::cytnx_uint64>(idx));
    }
    out.backend = in.backend.permute(cytnx_order);
  }

  // Complex conjugate
  template <typename TenT> void cplx_conj(context_handle_t<TenT>& ctx, TenT& inout) {
    if constexpr (std::is_same_v<elem_t<TenT>, cytnx::cytnx_complex128>
                  || std::is_same_v<elem_t<TenT>, cytnx::cytnx_complex64>) {
      inout.backend = inout.backend.Conj();
    }
    // For real types, do nothing
  }

  template <typename TenT> void cplx_conj(context_handle_t<TenT>& ctx, const TenT& in, TenT& out) {
    if constexpr (std::is_same_v<elem_t<TenT>, cytnx::cytnx_complex128>
                  || std::is_same_v<elem_t<TenT>, cytnx::cytnx_complex64>) {
      out.backend = in.backend.Conj();
    } else {
      out.backend = in.backend.clone();
    }
  }

  // Real part extraction
  template <typename TenT>
  void real(context_handle_t<TenT>& ctx, const TenT& in, real_ten_t<TenT>& out) {
    if constexpr (std::is_same_v<elem_t<TenT>, cytnx::cytnx_complex128>
                  || std::is_same_v<elem_t<TenT>, cytnx::cytnx_complex64>) {
      // Clone first since real() is not const
      auto temp = in.backend.clone();
      out.backend = temp.real();
    } else {
      // For real tensors, just copy
      out.backend = in.backend.clone();
    }
  }

  template <typename TenT> real_ten_t<TenT> real(context_handle_t<TenT>& ctx, const TenT& in) {
    real_ten_t<TenT> result;
    real(ctx, in, result);
    return result;
  }

  // Imaginary part extraction
  template <typename TenT>
  void imag(context_handle_t<TenT>& ctx, const TenT& in, real_ten_t<TenT>& out) {
    if constexpr (std::is_same_v<elem_t<TenT>, cytnx::cytnx_complex128>
                  || std::is_same_v<elem_t<TenT>, cytnx::cytnx_complex64>) {
      // Clone first since imag() is not const
      auto temp = in.backend.clone();
      out.backend = temp.imag();
    } else {
      // For real tensors, return zeros
      out = allocate<real_ten_t<TenT>>(ctx, shape(ctx, in));
      out.backend.storage().set_zeros();
    }
  }

  template <typename TenT> real_ten_t<TenT> imag(context_handle_t<TenT>& ctx, const TenT& in) {
    real_ten_t<TenT> result;
    imag(ctx, in, result);
    return result;
  }

  // Convert real tensor to complex tensor
  template <typename TenT>
  void to_cplx(context_handle_t<TenT>& ctx, const TenT& in, cplx_ten_t<TenT>& out) {
    if (in.backend.dtype() == cytnx::Type.ComplexDouble
        || in.backend.dtype() == cytnx::Type.ComplexFloat) {
      // Already complex, just copy
      out.backend = in.backend.clone();
    } else {
      // Convert real to complex
      out.backend = in.backend.astype(cytnx::Type.ComplexDouble);
    }
  }

  // Norm calculation
  template <typename TenT> real_t<TenT> norm(context_handle_t<TenT>& ctx, const TenT& a) {
    return a.backend.Norm().template item<real_t<TenT>>();
  }

  // Normalize
  template <typename TenT> real_t<TenT> normalize(context_handle_t<TenT>& ctx, TenT& inout) {
    auto n = norm(ctx, inout);
    if (n > 0) {
      inout.backend = inout.backend / n;
    }
    return n;
  }

  template <typename TenT>
  real_t<TenT> normalize(context_handle_t<TenT>& ctx, const TenT& in, TenT& out) {
    auto n = norm(ctx, in);
    if (n > 0) {
      out.backend = in.backend / n;
    } else {
      out.backend = in.backend.clone();
    }
    return n;
  }

  // Scale
  template <typename TenT>
  void scale(context_handle_t<TenT>& ctx, TenT& inout, const elem_t<TenT> s) {
    inout.backend = inout.backend * s;
  }

  template <typename TenT>
  void scale(context_handle_t<TenT>& ctx, const TenT& in, const elem_t<TenT> s, TenT& out) {
    out.backend = in.backend * s;
  }

  // Diag - extract diagonal or create diagonal matrix
  template <typename TenT> void diag(context_handle_t<TenT>& ctx, TenT& inout) {
    auto r = inout.backend.shape().size();
    if (r == 1) {
      // Create diagonal matrix from vector
      auto dim = static_cast<cytnx::cytnx_uint64>(inout.backend.shape()[0]);
      auto result = cytnx::zeros({dim, dim}, detail::elem_to_cytnx_type<elem_t<TenT>>(), ctx);

      auto* data = inout.backend.storage().template data<elem_t<TenT>>();
      auto* result_data = result.storage().template data<elem_t<TenT>>();

      for (cytnx::cytnx_uint64 i = 0; i < dim; ++i) {
        result_data[i * dim + i] = data[i];
      }

      inout.backend = result;
    } else if (r == 2) {
      // Extract diagonal from matrix
      auto dim = static_cast<cytnx::cytnx_uint64>(
          std::min(inout.backend.shape()[0], inout.backend.shape()[1]));
      auto result = cytnx::zeros({dim}, detail::elem_to_cytnx_type<elem_t<TenT>>(), ctx);

      auto rows = inout.backend.shape()[0];
      auto* in_data = inout.backend.storage().template data<elem_t<TenT>>();
      auto* result_data = result.storage().template data<elem_t<TenT>>();

      for (cytnx::cytnx_uint64 i = 0; i < dim; ++i) {
        result_data[i] = in_data[i * rows + i];
      }

      inout.backend = result;
    }
  }

  template <typename TenT> void diag(context_handle_t<TenT>& ctx, const TenT& in, TenT& out) {
    out.backend = in.backend.clone();
    diag(ctx, out);
  }

  // Trace - partial trace over specified bond pairs
  template <typename TenT>
  void trace(context_handle_t<TenT>& ctx, TenT& inout, const bond_idx_pairs_t<TenT>& bdidx_pairs) {
    cytnx::Tensor result = inout.backend;

    // Create index mapping to track axis renumbering after each trace
    std::vector<cytnx::cytnx_int64> axis_map(result.shape().size());
    std::iota(axis_map.begin(), axis_map.end(), 0);

    // Sort pairs by maximum index in descending order
    auto sorted_pairs = bdidx_pairs;
    std::sort(sorted_pairs.begin(), sorted_pairs.end(), [](const auto& a, const auto& b) {
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
        std::cerr << "trace: orig(" << orig_idx1 << "," << orig_idx2 << ") -> curr(" << curr_idx1
                  << "," << curr_idx2 << ") rank=" << result.shape().size() << std::endl;
      }

      // Perform trace
      result = result.Trace(curr_idx1, curr_idx2);

      // Update axis mapping: remove the two traced axes
      axis_map.erase(it1 < it2 ? it1 : it2);
      axis_map.erase(it1 < it2 ? (it2 - 1) : it1);
    }

    inout.backend = result;
  }

  template <typename TenT> void trace(context_handle_t<TenT>& ctx, const TenT& in,
                                      const bond_idx_pairs_t<TenT>& bdidx_pairs, TenT& out) {
    out.backend = in.backend.clone();
    trace(ctx, out, bdidx_pairs);
  }

  // Contract - tensor contraction following Einstein summation
  // Restored from b7ecb2a9^ (correct implementation using cytnx::linalg::Tensordot)
  template <typename TenT> void contract(context_handle_t<TenT>& ctx, const TenT& a,
                                         const std::vector<bond_label_t<TenT>>& bd_labs_a,
                                         const TenT& b,
                                         const std::vector<bond_label_t<TenT>>& bd_labs_b, TenT& c,
                                         const std::vector<bond_label_t<TenT>>& bd_labs_c) {
    (void)ctx;

    const auto rank_a = a.backend.shape().size();
    const auto rank_b = b.backend.shape().size();

    bool treat_as_label_mode = (bd_labs_a.size() == rank_a) && (bd_labs_b.size() == rank_b);

    const auto in_range = [](size_t rank, const std::vector<bond_label_t<TenT>>& axes_list) {
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
      detail::NCONAnalysis<TenT> analysis(bd_labs_a, bd_labs_b, bd_labs_c);

      if (analysis.contract_axes_a.empty() && analysis.contract_axes_b.empty()) {
        // Outer product case
        auto flatten = [](const cytnx::Tensor& tensor, bool row_vector) {
          cytnx::Tensor flat = tensor.clone();
          cytnx::cytnx_uint64 total = std::accumulate(tensor.shape().begin(), tensor.shape().end(),
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
        cytnx::Tensor result = cytnx::linalg::Tensordot(
            a.backend, b.backend, analysis.contract_axes_a, analysis.contract_axes_b);
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
    auto convert_axes
        = [](size_t rank, const std::vector<bond_label_t<TenT>>& axes_list, const char* which) {
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

    auto collect_free_axes = [](size_t rank, const std::vector<cytnx::cytnx_uint64>& contracted) {
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

  // Linear combination (out-of-place)
  template <typename TenT>
  TenT linear_combine(context_handle_t<TenT>& ctx, const std::vector<TenT>& ins) {
    TenT out;
    if (ins.empty()) {
      return out;
    }

    out.backend = ins[0].backend.clone();
    for (size_t i = 1; i < ins.size(); ++i) {
      out.backend = out.backend + ins[i].backend;
    }
    return out;
  }

  template <typename TenT> TenT linear_combine(context_handle_t<TenT>& ctx,
                                               const std::vector<TenT>& ins,
                                               const std::vector<elem_t<TenT>>& coefs) {
    TenT out;
    if (ins.empty() || ins.size() != coefs.size()) {
      return out;
    }

    out.backend = ins[0].backend * coefs[0];
    for (size_t i = 1; i < ins.size(); ++i) {
      out.backend = out.backend + ins[i].backend * coefs[i];
    }
    return out;
  }

  // SVD (full)
  template <typename TenT> void svd(context_handle_t<TenT>& ctx, const TenT& a,
                                    const order_t<TenT> num_of_bds_as_row, TenT& u,
                                    real_ten_t<TenT>& s_diag, TenT& v_dag) {
    // Get shape and compute matrix dimensions
    auto a_shape = shape(ctx, a);
    cytnx::cytnx_uint64 left_dim = 1;
    for (order_t<TenT> i = 0; i < num_of_bds_as_row; ++i) {
      left_dim *= a_shape[i];
    }
    cytnx::cytnx_uint64 right_dim = 1;
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      right_dim *= a_shape[i];
    }

    // Reshape to matrix
    auto a_reshaped = a.backend.reshape(
        {static_cast<cytnx::cytnx_int64>(left_dim), static_cast<cytnx::cytnx_int64>(right_dim)});

    // Perform full SVD
    auto svd_result = cytnx::linalg::Svd(a_reshaped, true);  // Return U, S, Vt

    if (svd_result.size() < 3) {
      throw std::runtime_error("svd: unexpected result size from Svd");
    }

    // Extract S, U, Vt (order: S, U, V†)
    auto& s_backend = svd_result[0];
    auto& u_backend = svd_result[1];
    auto& vt_backend = svd_result[2];

    bond_dim_t<TenT> bond_dim = s_backend.shape()[0];

    // Extract real singular values
    cytnx::Tensor s_real = s_backend.dtype() == cytnx::Type.Double ? s_backend : s_backend.real();

    // Reshape U
    shape_t<TenT> u_shape;
    for (order_t<TenT> i = 0; i < num_of_bds_as_row; ++i) {
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
    shape_t<TenT> v_shape;
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
  template <typename TenT> void qr(context_handle_t<TenT>& ctx, const TenT& a,
                                   const order_t<TenT> num_of_bds_as_row, TenT& q, TenT& r) {
    // Get shape and compute matrix dimensions
    auto a_shape = shape(ctx, a);
    cytnx::cytnx_uint64 left_dim = 1;
    for (order_t<TenT> i = 0; i < num_of_bds_as_row; ++i) {
      left_dim *= a_shape[i];
    }
    cytnx::cytnx_uint64 right_dim = 1;
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      right_dim *= a_shape[i];
    }

    // Reshape to matrix
    auto a_reshaped = a.backend.reshape(
        {static_cast<cytnx::cytnx_int64>(left_dim), static_cast<cytnx::cytnx_int64>(right_dim)});

    // Perform QR
    auto qr_result = cytnx::linalg::Qr(a_reshaped);

    if (qr_result.size() < 2) {
      throw std::runtime_error("qr: unexpected result size from Qr");
    }

    auto& q_backend = qr_result[0];
    auto& r_backend = qr_result[1];

    auto bond_dim = q_backend.shape()[1];

    // Reshape Q
    shape_t<TenT> q_shape;
    for (order_t<TenT> i = 0; i < num_of_bds_as_row; ++i) {
      q_shape.push_back(a_shape[i]);
    }
    q_shape.push_back(bond_dim);
    std::vector<cytnx::cytnx_int64> q_cytnx_shape;
    for (auto dim : q_shape) {
      q_cytnx_shape.push_back(static_cast<cytnx::cytnx_int64>(dim));
    }
    q.backend = q_backend.reshape(q_cytnx_shape);

    // Reshape R
    shape_t<TenT> r_shape;
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
  template <typename TenT> void lq(context_handle_t<TenT>& ctx, const TenT& a,
                                   const order_t<TenT> num_of_bds_as_row, TenT& l, TenT& q) {
    // LQ = (Q†L†)† where Q†L† is QR of A†
    // Transpose and do QR, then transpose results back

    auto a_shape = shape(ctx, a);
    cytnx::cytnx_uint64 left_dim = 1;
    for (order_t<TenT> i = 0; i < num_of_bds_as_row; ++i) {
      left_dim *= a_shape[i];
    }
    cytnx::cytnx_uint64 right_dim = 1;
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      right_dim *= a_shape[i];
    }

    // Reshape and transpose
    auto a_reshaped = a.backend.reshape(
        {static_cast<cytnx::cytnx_int64>(left_dim), static_cast<cytnx::cytnx_int64>(right_dim)});
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
    shape_t<TenT> l_shape;
    for (order_t<TenT> i = 0; i < num_of_bds_as_row; ++i) {
      l_shape.push_back(a_shape[i]);
    }
    l_shape.push_back(bond_dim);
    std::vector<cytnx::cytnx_int64> l_cytnx_shape;
    for (auto dim : l_shape) {
      l_cytnx_shape.push_back(static_cast<cytnx::cytnx_int64>(dim));
    }
    l.backend = l_backend.reshape(l_cytnx_shape);

    // Reshape Q
    shape_t<TenT> q_shape;
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

  // Truncated SVD - overload (2): chi_max, s_min
  template <typename TenT>
  void trunc_svd(context_handle_t<TenT>& ctx, const TenT& a, const order_t<TenT> num_of_bds_as_row,
                 TenT& u, real_ten_t<TenT>& s_diag, TenT& v_dag, real_t<TenT>& trunc_err,
                 const bond_dim_t<TenT> chi_max, const real_t<TenT> s_min) {
    // Call full version with chi_min=1, target_trunc_err=0
    constexpr bond_dim_t<TenT> chi_min = 1;
    constexpr real_t<TenT> target_trunc_err = 0.0;
    trunc_svd(ctx, a, num_of_bds_as_row, u, s_diag, v_dag, trunc_err, chi_min, chi_max,
              target_trunc_err, s_min);
  }

  // Truncated SVD - overload (1): target_trunc_err, s_min
  template <typename TenT>
  void trunc_svd(context_handle_t<TenT>& ctx, const TenT& a, const order_t<TenT> num_of_bds_as_row,
                 TenT& u, real_ten_t<TenT>& s_diag, TenT& v_dag, real_t<TenT>& trunc_err,
                 const real_t<TenT> target_trunc_err, const real_t<TenT> s_min) {
    // Call full version with chi_min=1, chi_max=∞ (represented by a very large value)
    constexpr bond_dim_t<TenT> chi_min = 1;
    constexpr bond_dim_t<TenT> chi_max = std::numeric_limits<bond_dim_t<TenT>>::max();
    trunc_svd(ctx, a, num_of_bds_as_row, u, s_diag, v_dag, trunc_err, chi_min, chi_max,
              target_trunc_err, s_min);
  }

  // Truncated SVD - overload (3): full control
  template <typename TenT>
  void trunc_svd(context_handle_t<TenT>& ctx, const TenT& a, const order_t<TenT> num_of_bds_as_row,
                 TenT& u, real_ten_t<TenT>& s_diag, TenT& v_dag, real_t<TenT>& trunc_err,
                 const bond_dim_t<TenT> chi_min, const bond_dim_t<TenT> chi_max,
                 const real_t<TenT> target_trunc_err, const real_t<TenT> s_min) {
    // Get shape and compute matrix dimensions
    auto a_shape = shape(ctx, a);
    cytnx::cytnx_uint64 left_dim = 1;
    for (order_t<TenT> i = 0; i < num_of_bds_as_row; ++i) {
      left_dim *= a_shape[i];
    }
    cytnx::cytnx_uint64 right_dim = 1;
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      right_dim *= a_shape[i];
    }

    // Reshape to matrix
    auto a_reshaped = a.backend.reshape(
        {static_cast<cytnx::cytnx_int64>(left_dim), static_cast<cytnx::cytnx_int64>(right_dim)});

    // Compute ||A||_F^2 before truncation for trunc_err calculation.
    // ||A||_F^2 = sum(s_i^2) for all pre-truncation singular values.
    double frobenius_sq = cytnx::linalg::Norm(a_reshaped).template item<double>();
    frobenius_sq *= frobenius_sq;

    // Perform SVD with chi_max constraint
    // Cytnx's Svd_truncate uses LAPACK zgesdd (divide-and-conquer) which
    // can fail on ill-conditioned matrices.  Fall back to Gesvd_truncate
    // (zgesvd, QR iteration) on convergence failure.
    std::vector<cytnx::Tensor> svd_result;
    bool used_fallback = false;
    try {
      svd_result = cytnx::linalg::Svd_truncate(a_reshaped, chi_max, s_min, true, 0, 1);
    } catch (...) {
      used_fallback = true;
      svd_result = cytnx::linalg::Gesvd_truncate(a_reshaped, chi_max, s_min, true, true, 0, 1);
    }

    if (svd_result.size() < 3) {
      throw std::runtime_error(
          std::string("trunc_svd: unexpected result size from ") +
          (used_fallback ? "Gesvd_truncate" : "Svd_truncate"));
    }

    // Extract S, U, Vt (order: S, U, V†)
    auto s_backend = svd_result[0];
    auto u_backend = svd_result[1];
    auto vt_backend = svd_result[2];

    // Apply target_trunc_err: find the largest index where s[i] > target_trunc_err
    bond_dim_t<TenT> bond_dim = s_backend.shape()[0];
    bond_dim_t<TenT> new_bond_dim = bond_dim;

    if (target_trunc_err > 0.0 && bond_dim > chi_min) {
      // Find truncation point based on target_trunc_err
      if (s_backend.dtype() == cytnx::Type.Double) {
        auto* s_data = s_backend.template ptr_as<double>();
        for (bond_dim_t<TenT> i = chi_min; i < bond_dim; ++i) {
          if (s_data[i] <= target_trunc_err) {
            new_bond_dim = i;
            break;
          }
        }
      } else if (s_backend.dtype() == cytnx::Type.Float) {
        auto* s_data = s_backend.template ptr_as<float>();
        for (bond_dim_t<TenT> i = chi_min; i < bond_dim; ++i) {
          if (s_data[i] <= static_cast<float>(target_trunc_err)) {
            new_bond_dim = i;
            break;
          }
        }
      }
      // Ensure we keep at least chi_min dimensions
      new_bond_dim = std::max(new_bond_dim, chi_min);
    }

    // Apply chi_min constraint
    new_bond_dim = std::max(new_bond_dim, chi_min);

    // Truncate tensors if needed
    if (new_bond_dim < bond_dim) {
      // Truncate S
      s_backend = s_backend.get({cytnx::Accessor::range(0, new_bond_dim)});
      // Truncate U
      u_backend = u_backend.get({cytnx::Accessor::all(), cytnx::Accessor::range(0, new_bond_dim)});
      // Truncate Vt
      vt_backend
          = vt_backend.get({cytnx::Accessor::range(0, new_bond_dim), cytnx::Accessor::all()});
      bond_dim = new_bond_dim;
    }

    // Calculate truncation error per TCI spec:
    // epsilon = sum(s_discarded^2) / sum(s_all^2)
    //         = (||A||_F^2 - sum(s_kept^2)) / ||A||_F^2
    if (frobenius_sq > 0.0 && bond_dim > 0) {
      double kept_s2 = 0.0;
      if (s_backend.dtype() == cytnx::Type.Double) {
        auto* s_data = s_backend.template ptr_as<double>();
        for (bond_dim_t<TenT> i = 0; i < bond_dim; ++i) {
          kept_s2 += static_cast<double>(s_data[i]) * static_cast<double>(s_data[i]);
        }
      } else if (s_backend.dtype() == cytnx::Type.Float) {
        auto* s_data = s_backend.template ptr_as<float>();
        for (bond_dim_t<TenT> i = 0; i < bond_dim; ++i) {
          kept_s2 += static_cast<double>(s_data[i]) * static_cast<double>(s_data[i]);
        }
      }
      trunc_err = (frobenius_sq - kept_s2) / frobenius_sq;
      // Clamp to [0, 1] to guard against floating-point roundoff
      if (trunc_err < 0.0) trunc_err = 0.0;
    } else {
      trunc_err = 0.0;
    }

    // Extract real singular values
    cytnx::Tensor s_real = s_backend.dtype() == cytnx::Type.Double ? s_backend : s_backend.real();

    // Reshape U
    shape_t<TenT> u_shape;
    for (order_t<TenT> i = 0; i < num_of_bds_as_row; ++i) {
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
    shape_t<TenT> v_shape;
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
  template <typename TenT> void eigvals(context_handle_t<TenT>& ctx, const TenT& a,
                                        const order_t<TenT> num_of_bds_as_row,
                                        cplx_ten_t<TenT>& w_diag) {
    auto a_shape = shape(ctx, a);

    cytnx::cytnx_uint64 row_dim = 1;
    cytnx::cytnx_uint64 col_dim = 1;

    for (order_t<TenT> i = 0; i < num_of_bds_as_row && i < a_shape.size(); ++i) {
      row_dim *= a_shape[i];
    }
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      col_dim *= a_shape[i];
    }

    if (row_dim != col_dim) {
      throw std::invalid_argument("eigvals: matrix must be square");
    }

    cytnx::Tensor matrix = a.backend.clone();
    matrix.reshape_(
        {static_cast<cytnx::cytnx_int64>(row_dim), static_cast<cytnx::cytnx_int64>(col_dim)});

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
  template <typename TenT> void eigvalsh(context_handle_t<TenT>& ctx, const TenT& a,
                                         const order_t<TenT> num_of_bds_as_row,
                                         real_ten_t<TenT>& w_diag) {
    auto a_shape = shape(ctx, a);

    cytnx::cytnx_uint64 row_dim = 1;
    cytnx::cytnx_uint64 col_dim = 1;

    for (order_t<TenT> i = 0; i < num_of_bds_as_row && i < a_shape.size(); ++i) {
      row_dim *= a_shape[i];
    }
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      col_dim *= a_shape[i];
    }

    if (row_dim != col_dim) {
      throw std::invalid_argument("eigvalsh: matrix must be square");
    }

    cytnx::Tensor matrix = a.backend.clone();
    matrix.reshape_(
        {static_cast<cytnx::cytnx_int64>(row_dim), static_cast<cytnx::cytnx_int64>(col_dim)});

    auto eigh_result = cytnx::linalg::Eigh(matrix);
    w_diag.backend = eigh_result[0];

    if (w_diag.backend.shape().size() != 1) {
      w_diag.backend.reshape_({static_cast<cytnx::cytnx_int64>(row_dim)});
    }
  }

  // Eigenvalue decomposition - eig (general matrix eigenvalues and eigenvectors)
  template <typename TenT> void eig(context_handle_t<TenT>& ctx, const TenT& a,
                                    const order_t<TenT> num_of_bds_as_row, cplx_ten_t<TenT>& w_diag,
                                    cplx_ten_t<TenT>& v) {
    auto a_shape = shape(ctx, a);

    cytnx::cytnx_uint64 row_dim = 1;
    cytnx::cytnx_uint64 col_dim = 1;

    for (order_t<TenT> i = 0; i < num_of_bds_as_row && i < a_shape.size(); ++i) {
      row_dim *= a_shape[i];
    }
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      col_dim *= a_shape[i];
    }

    if (row_dim != col_dim) {
      throw std::invalid_argument("eig: matrix must be square");
    }

    cytnx::Tensor matrix = a.backend.clone();
    matrix.reshape_(
        {static_cast<cytnx::cytnx_int64>(row_dim), static_cast<cytnx::cytnx_int64>(col_dim)});

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
      v.backend.reshape_(
          {static_cast<cytnx::cytnx_int64>(row_dim), static_cast<cytnx::cytnx_int64>(row_dim)});
    }
    if (v.backend.dtype() != cytnx::Type.ComplexDouble) {
      v.backend = v.backend.astype(cytnx::Type.ComplexDouble);
    }
  }

  // Eigenvalue decomposition - eigh (hermitian matrix eigenvalues and eigenvectors)
  template <typename TenT> void eigh(context_handle_t<TenT>& ctx, const TenT& a,
                                     const order_t<TenT> num_of_bds_as_row,
                                     real_ten_t<TenT>& w_diag, TenT& v) {
    auto a_shape = shape(ctx, a);

    cytnx::cytnx_uint64 row_dim = 1;
    cytnx::cytnx_uint64 col_dim = 1;

    for (order_t<TenT> i = 0; i < num_of_bds_as_row && i < a_shape.size(); ++i) {
      row_dim *= a_shape[i];
    }
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      col_dim *= a_shape[i];
    }

    if (row_dim != col_dim) {
      throw std::invalid_argument("eigh: matrix must be square");
    }

    cytnx::Tensor matrix = a.backend.clone();
    matrix.reshape_(
        {static_cast<cytnx::cytnx_int64>(row_dim), static_cast<cytnx::cytnx_int64>(col_dim)});

    auto eigh_result = cytnx::linalg::Eigh(matrix);
    w_diag.backend = eigh_result[0];
    v.backend = eigh_result[1];

    if (w_diag.backend.shape().size() != 1) {
      w_diag.backend.reshape_({static_cast<cytnx::cytnx_int64>(row_dim)});
    }

    if (v.backend.shape().size() != 2) {
      v.backend.reshape_(
          {static_cast<cytnx::cytnx_int64>(row_dim), static_cast<cytnx::cytnx_int64>(row_dim)});
    }
  }

  // Check if two tensors are close within tolerance
  template <typename TenT> bool close(context_handle_t<TenT>& ctx, const TenT& a, const TenT& b,
                                      const real_t<TenT> epsilon) {
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
                                    std::size_t dim,
                                    const std::vector<cytnx::cytnx_uint64>& begin_pt,
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

  template <typename TenT>
  void expand(context_handle_t<TenT>& ctx, TenT& inout,
              const Map<bond_idx_t<TenT>, bond_dim_t<TenT>>& bond_idx_increment_map) {
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

  template <typename TenT>
  void expand(context_handle_t<TenT>& ctx, const TenT& in,
              const Map<bond_idx_t<TenT>, bond_dim_t<TenT>>& bond_idx_increment_map, TenT& out) {
    out = in;
    expand(ctx, out, bond_idx_increment_map);
  }

  // shrink
  // Restored from git show b7ecb2a9^:source/tensor_manipulation.cpp
  template <typename TenT>
  void shrink(context_handle_t<TenT>& ctx, TenT& inout,
              const bond_idx_elem_coor_pair_map<TenT>& bd_idx_el_coor_pair_map) {
    auto original_shape = inout.backend.shape();

    // Build coordinate pairs list from the map
    List<Pair<elem_coor_t<TenT>, elem_coor_t<TenT>>> coor_pairs;
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

  template <typename TenT>
  void shrink(context_handle_t<TenT>& ctx, const TenT& in,
              const bond_idx_elem_coor_pair_map<TenT>& bd_idx_el_coor_pair_map, TenT& out) {
    out = in;
    shrink(ctx, out, bd_idx_el_coor_pair_map);
  }

  // extract_sub
  // Restored from git show b7ecb2a9^:source/tensor_manipulation.cpp
  template <typename TenT>
  void extract_sub(context_handle_t<TenT>& ctx, TenT& inout,
                   const List<Pair<elem_coor_t<TenT>, elem_coor_t<TenT>>>& coor_pairs) {
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

  template <typename TenT>
  void extract_sub(context_handle_t<TenT>& ctx, const TenT& in,
                   const List<Pair<elem_coor_t<TenT>, elem_coor_t<TenT>>>& coor_pairs, TenT& out) {
    out = in;
    extract_sub(ctx, out, coor_pairs);
  }

  // replace_sub
  // Restored from git show b7ecb2a9^:source/tensor_manipulation.cpp
  template <typename TenT> void replace_sub(context_handle_t<TenT>& ctx, TenT& inout,
                                            const TenT& sub, const elem_coors_t<TenT>& begin_pt) {
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

  template <typename TenT> void replace_sub(context_handle_t<TenT>& ctx, const TenT& in,
                                            const TenT& sub, const elem_coors_t<TenT>& begin_pt,
                                            TenT& out) {
    out.backend = in.backend.clone();
    replace_sub(ctx, out, sub, begin_pt);
  }

  // concatenate
  // Restored from git show b7ecb2a9^:source/tensor_manipulation.cpp
  template <typename TenT> void concatenate(context_handle_t<TenT>& ctx, const List<TenT>& ins,
                                            const bond_idx_t<TenT> axis, TenT& out) {
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
          throw std::invalid_argument(
              "All tensors must have the same shape except along concat dimension");
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
  template <typename TenT> void stack(context_handle_t<TenT>& ctx, const List<TenT>& ins,
                                      const bond_idx_t<TenT> axis, TenT& out) {
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
  // for_each_with_coors implementation for TenT
  // ========================================================================

  namespace detail {
    // Helper function for for_each_with_coors (mutable version)
    template <typename TenT, typename Func>
    void for_each_recursive_typed(TenT& tensor, Func&& f, std::size_t dim,
                                  std::vector<cytnx::cytnx_uint64>& coords,
                                  const std::vector<cytnx::cytnx_uint64>& shape) {
      if (dim == shape.size()) {
        // Base case: apply function to element at current coordinates
        auto& elem = tensor.backend.template at<elem_t<TenT>>(coords);
        // Convert coords to elem_coors_t<TenT>
        elem_coors_t<TenT> tci_coords;
        tci_coords.reserve(coords.size());
        for (const auto& coord : coords) {
          tci_coords.push_back(static_cast<elem_coor_t<TenT>>(coord));
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
    template <typename TenT, typename Func>
    void for_each_recursive_const_typed(const TenT& tensor, Func&& f, std::size_t dim,
                                        std::vector<cytnx::cytnx_uint64>& coords,
                                        const std::vector<cytnx::cytnx_uint64>& shape) {
      if (dim == shape.size()) {
        // Base case: apply function to element at current coordinates
        const auto& elem = tensor.backend.template at<elem_t<TenT>>(coords);
        // Convert coords to elem_coors_t<TenT>
        elem_coors_t<TenT> tci_coords;
        tci_coords.reserve(coords.size());
        for (const auto& coord : coords) {
          tci_coords.push_back(static_cast<elem_coor_t<TenT>>(coord));
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

  // for_each_with_coors for TenT (mutable version)
  template <typename TenT, typename Func>
  void for_each_with_coors(context_handle_t<TenT>& ctx, TenT& inout, Func&& f) {
    auto shape = inout.backend.shape();
    std::vector<cytnx::cytnx_uint64> coords;
    coords.reserve(shape.size());

    detail::for_each_recursive_typed(inout, std::forward<Func>(f), 0, coords, shape);
  }

  // for_each_with_coors for TenT (const version)
  template <typename TenT, typename Func>
  void for_each_with_coors(context_handle_t<TenT>& ctx, const TenT& in, Func&& f) {
    auto shape = in.backend.shape();
    std::vector<cytnx::cytnx_uint64> coords;
    coords.reserve(shape.size());

    detail::for_each_recursive_const_typed(in, std::forward<Func>(f), 0, coords, shape);
  }

  // move - move tensor contents (out-of-place), invalidates source
  template <typename TenT> TenT move(context_handle_t<TenT>& ctx, TenT& from) {
    TenT result;
    result.backend = std::move(from.backend);
    from = TenT();
    return result;
  }

  // to_cplx - convert to complex tensor (out-of-place)
  template <typename TenT> cplx_ten_t<TenT> to_cplx(context_handle_t<TenT>& ctx, const TenT& in) {
    cplx_ten_t<TenT> result;
    to_cplx(ctx, in, result);
    return result;
  }

  // contract - tensor contraction (string version)
  template <typename TenT> void contract(context_handle_t<TenT>& ctx, const TenT& a,
                                         const std::string_view bd_labs_str_a, const TenT& b,
                                         const std::string_view bd_labs_str_b, TenT& c,
                                         const std::string_view bd_labs_str_c) {
    List<bond_label_t<TenT>> bd_labs_a, bd_labs_b, bd_labs_c;
    for (char ch : bd_labs_str_a) {
      bd_labs_a.push_back(static_cast<bond_label_t<TenT>>(ch));
    }
    for (char ch : bd_labs_str_b) {
      bd_labs_b.push_back(static_cast<bond_label_t<TenT>>(ch));
    }
    for (char ch : bd_labs_str_c) {
      bd_labs_c.push_back(static_cast<bond_label_t<TenT>>(ch));
    }
    contract(ctx, a, bd_labs_a, b, bd_labs_b, c, bd_labs_c);
  }

  // Forward declarations for allocate (out-of-place) specializations
  template <> CytnxTensor<cytnx::cytnx_double> allocate<CytnxTensor<cytnx::cytnx_double>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_double>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_double>>& shape);
  template <> CytnxTensor<cytnx::cytnx_float> allocate<CytnxTensor<cytnx::cytnx_float>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_float>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_float>>& shape);
  template <> CytnxTensor<cytnx::cytnx_complex128> allocate<CytnxTensor<cytnx::cytnx_complex128>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_complex128>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_complex128>>& shape);
  template <> CytnxTensor<cytnx::cytnx_complex64> allocate<CytnxTensor<cytnx::cytnx_complex64>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_complex64>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_complex64>>& shape);

  // Forward declarations for fill (out-of-place) specializations
  template <> CytnxTensor<cytnx::cytnx_double> fill<CytnxTensor<cytnx::cytnx_double>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_double>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_double>>& shape,
      elem_t<CytnxTensor<cytnx::cytnx_double>> value);
  template <> CytnxTensor<cytnx::cytnx_float> fill<CytnxTensor<cytnx::cytnx_float>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_float>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_float>>& shape,
      elem_t<CytnxTensor<cytnx::cytnx_float>> value);
  template <> CytnxTensor<cytnx::cytnx_complex128> fill<CytnxTensor<cytnx::cytnx_complex128>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_complex128>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_complex128>>& shape,
      elem_t<CytnxTensor<cytnx::cytnx_complex128>> value);
  template <> CytnxTensor<cytnx::cytnx_complex64> fill<CytnxTensor<cytnx::cytnx_complex64>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_complex64>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_complex64>>& shape,
      elem_t<CytnxTensor<cytnx::cytnx_complex64>> value);

  // Explicit specializations for zeros (out-of-place) for all supported element types
  template <> inline CytnxTensor<cytnx::cytnx_double> zeros<CytnxTensor<cytnx::cytnx_double>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_double>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_double>>& shape) {
    return fill<CytnxTensor<cytnx::cytnx_double>>(ctx, shape, 0.0);
  }

  template <> inline CytnxTensor<cytnx::cytnx_float> zeros<CytnxTensor<cytnx::cytnx_float>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_float>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_float>>& shape) {
    return fill<CytnxTensor<cytnx::cytnx_float>>(ctx, shape, 0.0f);
  }

  template <>
  inline CytnxTensor<cytnx::cytnx_complex128> zeros<CytnxTensor<cytnx::cytnx_complex128>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_complex128>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_complex128>>& shape) {
    return fill<CytnxTensor<cytnx::cytnx_complex128>>(ctx, shape, std::complex<double>(0.0, 0.0));
  }

  template <> inline CytnxTensor<cytnx::cytnx_complex64> zeros<CytnxTensor<cytnx::cytnx_complex64>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_complex64>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_complex64>>& shape) {
    return fill<CytnxTensor<cytnx::cytnx_complex64>>(ctx, shape, std::complex<float>(0.0f, 0.0f));
  }

  // Explicit specializations for fill (out-of-place) for all supported element types
  template <> inline CytnxTensor<cytnx::cytnx_double> fill<CytnxTensor<cytnx::cytnx_double>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_double>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_double>>& shape,
      elem_t<CytnxTensor<cytnx::cytnx_double>> value) {
    auto result = allocate<CytnxTensor<cytnx::cytnx_double>>(ctx, shape);
    auto total_size = static_cast<cytnx::cytnx_uint64>(result.backend.storage().size());
    auto* data = result.backend.storage().template data<cytnx::cytnx_double>();
    for (cytnx::cytnx_uint64 i = 0; i < total_size; ++i) {
      data[i] = value;
    }
    return result;
  }

  template <> inline CytnxTensor<cytnx::cytnx_float> fill<CytnxTensor<cytnx::cytnx_float>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_float>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_float>>& shape,
      elem_t<CytnxTensor<cytnx::cytnx_float>> value) {
    auto result = allocate<CytnxTensor<cytnx::cytnx_float>>(ctx, shape);
    auto total_size = static_cast<cytnx::cytnx_uint64>(result.backend.storage().size());
    auto* data = result.backend.storage().template data<cytnx::cytnx_float>();
    for (cytnx::cytnx_uint64 i = 0; i < total_size; ++i) {
      data[i] = value;
    }
    return result;
  }

  template <>
  inline CytnxTensor<cytnx::cytnx_complex128> fill<CytnxTensor<cytnx::cytnx_complex128>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_complex128>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_complex128>>& shape,
      elem_t<CytnxTensor<cytnx::cytnx_complex128>> value) {
    auto result = allocate<CytnxTensor<cytnx::cytnx_complex128>>(ctx, shape);
    auto total_size = static_cast<cytnx::cytnx_uint64>(result.backend.storage().size());
    auto* data = result.backend.storage().template data<cytnx::cytnx_complex128>();
    for (cytnx::cytnx_uint64 i = 0; i < total_size; ++i) {
      data[i] = value;
    }
    return result;
  }

  template <> inline CytnxTensor<cytnx::cytnx_complex64> fill<CytnxTensor<cytnx::cytnx_complex64>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_complex64>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_complex64>>& shape,
      elem_t<CytnxTensor<cytnx::cytnx_complex64>> value) {
    auto result = allocate<CytnxTensor<cytnx::cytnx_complex64>>(ctx, shape);
    auto total_size = static_cast<cytnx::cytnx_uint64>(result.backend.storage().size());
    auto* data = result.backend.storage().template data<cytnx::cytnx_complex64>();
    for (cytnx::cytnx_uint64 i = 0; i < total_size; ++i) {
      data[i] = value;
    }
    return result;
  }

  // Explicit specializations for allocate (out-of-place) - implementations
  template <> inline CytnxTensor<cytnx::cytnx_double> allocate<CytnxTensor<cytnx::cytnx_double>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_double>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_double>>& shape) {
    CytnxTensor<cytnx::cytnx_double> result;
    std::vector<cytnx::cytnx_uint64> cytnx_shape;
    cytnx_shape.reserve(shape.size());
    for (const auto& dim : shape) {
      cytnx_shape.push_back(static_cast<cytnx::cytnx_uint64>(dim));
    }
    result.backend
        = cytnx::Tensor(cytnx_shape, detail::elem_to_cytnx_type<cytnx::cytnx_double>(), ctx);
    return result;
  }

  template <> inline CytnxTensor<cytnx::cytnx_float> allocate<CytnxTensor<cytnx::cytnx_float>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_float>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_float>>& shape) {
    CytnxTensor<cytnx::cytnx_float> result;
    std::vector<cytnx::cytnx_uint64> cytnx_shape;
    cytnx_shape.reserve(shape.size());
    for (const auto& dim : shape) {
      cytnx_shape.push_back(static_cast<cytnx::cytnx_uint64>(dim));
    }
    result.backend
        = cytnx::Tensor(cytnx_shape, detail::elem_to_cytnx_type<cytnx::cytnx_float>(), ctx);
    return result;
  }

  template <>
  inline CytnxTensor<cytnx::cytnx_complex128> allocate<CytnxTensor<cytnx::cytnx_complex128>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_complex128>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_complex128>>& shape) {
    CytnxTensor<cytnx::cytnx_complex128> result;
    std::vector<cytnx::cytnx_uint64> cytnx_shape;
    cytnx_shape.reserve(shape.size());
    for (const auto& dim : shape) {
      cytnx_shape.push_back(static_cast<cytnx::cytnx_uint64>(dim));
    }
    result.backend
        = cytnx::Tensor(cytnx_shape, detail::elem_to_cytnx_type<cytnx::cytnx_complex128>(), ctx);
    return result;
  }

  template <>
  inline CytnxTensor<cytnx::cytnx_complex64> allocate<CytnxTensor<cytnx::cytnx_complex64>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_complex64>>& ctx,
      const shape_t<CytnxTensor<cytnx::cytnx_complex64>>& shape) {
    CytnxTensor<cytnx::cytnx_complex64> result;
    std::vector<cytnx::cytnx_uint64> cytnx_shape;
    cytnx_shape.reserve(shape.size());
    for (const auto& dim : shape) {
      cytnx_shape.push_back(static_cast<cytnx::cytnx_uint64>(dim));
    }
    result.backend
        = cytnx::Tensor(cytnx_shape, detail::elem_to_cytnx_type<cytnx::cytnx_complex64>(), ctx);
    return result;
  }

  // Explicit specializations for eye (out-of-place) for all supported element types
  template <> inline CytnxTensor<cytnx::cytnx_double> eye<CytnxTensor<cytnx::cytnx_double>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_double>>& ctx,
      const bond_dim_t<CytnxTensor<cytnx::cytnx_double>> N) {
    CytnxTensor<cytnx::cytnx_double> result;
    result.backend = cytnx::eye(N, detail::elem_to_cytnx_type<cytnx::cytnx_double>(), ctx);
    return result;
  }

  template <> inline CytnxTensor<cytnx::cytnx_float> eye<CytnxTensor<cytnx::cytnx_float>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_float>>& ctx,
      const bond_dim_t<CytnxTensor<cytnx::cytnx_float>> N) {
    CytnxTensor<cytnx::cytnx_float> result;
    result.backend = cytnx::eye(N, detail::elem_to_cytnx_type<cytnx::cytnx_float>(), ctx);
    return result;
  }

  template <> inline CytnxTensor<cytnx::cytnx_complex128> eye<CytnxTensor<cytnx::cytnx_complex128>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_complex128>>& ctx,
      const bond_dim_t<CytnxTensor<cytnx::cytnx_complex128>> N) {
    CytnxTensor<cytnx::cytnx_complex128> result;
    result.backend = cytnx::eye(N, detail::elem_to_cytnx_type<cytnx::cytnx_complex128>(), ctx);
    return result;
  }

  template <> inline CytnxTensor<cytnx::cytnx_complex64> eye<CytnxTensor<cytnx::cytnx_complex64>>(
      context_handle_t<CytnxTensor<cytnx::cytnx_complex64>>& ctx,
      const bond_dim_t<CytnxTensor<cytnx::cytnx_complex64>> N) {
    CytnxTensor<cytnx::cytnx_complex64> result;
    result.backend = cytnx::eye(N, detail::elem_to_cytnx_type<cytnx::cytnx_complex64>(), ctx);
    return result;
  }

  // ===================================================================
  // Linear Algebra Operations
  // ===================================================================

  // inverse - matrix inverse (in-place)
  template <typename TenT>
  void inverse(context_handle_t<TenT>& ctx, TenT& inout, const order_t<TenT> num_of_bds_as_row) {
    auto a_shape = shape(ctx, inout);

    cytnx::cytnx_uint64 row_dim = 1;
    cytnx::cytnx_uint64 col_dim = 1;

    for (order_t<TenT> i = 0; i < num_of_bds_as_row && i < a_shape.size(); ++i) {
      row_dim *= a_shape[i];
    }
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      col_dim *= a_shape[i];
    }

    if (row_dim != col_dim || row_dim == 0) {
      throw std::invalid_argument("inverse: matrix must be square");
    }

    cytnx::Tensor reshaped = inout.backend.reshape(
        {static_cast<cytnx::cytnx_int64>(row_dim), static_cast<cytnx::cytnx_int64>(col_dim)});

    cytnx::Tensor result;
    try {
      result = cytnx::linalg::InvM(reshaped);
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string("inverse: failed to compute matrix inverse - ")
                               + e.what());
    }

    std::vector<cytnx::cytnx_int64> original_shape;
    original_shape.reserve(a_shape.size());
    for (auto dim : a_shape) {
      original_shape.push_back(static_cast<cytnx::cytnx_int64>(dim));
    }

    inout.backend = result.reshape(original_shape);
  }

  // inverse - matrix inverse (out-of-place)
  template <typename TenT> void inverse(context_handle_t<TenT>& ctx, const TenT& in,
                                        const order_t<TenT> num_of_bds_as_row, TenT& out) {
    auto a_shape = shape(ctx, in);

    cytnx::cytnx_uint64 row_dim = 1;
    cytnx::cytnx_uint64 col_dim = 1;

    for (order_t<TenT> i = 0; i < num_of_bds_as_row && i < a_shape.size(); ++i) {
      row_dim *= a_shape[i];
    }
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      col_dim *= a_shape[i];
    }

    if (row_dim != col_dim || row_dim == 0) {
      throw std::invalid_argument("inverse: matrix must be square");
    }

    cytnx::Tensor reshaped = in.backend.reshape(
        {static_cast<cytnx::cytnx_int64>(row_dim), static_cast<cytnx::cytnx_int64>(col_dim)});

    cytnx::Tensor result;
    try {
      result = cytnx::linalg::InvM(reshaped);
    } catch (const std::exception& e) {
      throw std::runtime_error(std::string("inverse: failed to compute matrix inverse - ")
                               + e.what());
    }

    std::vector<cytnx::cytnx_int64> original_shape;
    original_shape.reserve(a_shape.size());
    for (auto dim : a_shape) {
      original_shape.push_back(static_cast<cytnx::cytnx_int64>(dim));
    }

    out.backend = result.reshape(original_shape);
  }

  // exp - matrix exponential (in-place)
  template <typename TenT>
  void exp(context_handle_t<TenT>& ctx, TenT& inout, const order_t<TenT> num_of_bds_as_row) {
    auto a_shape = shape(ctx, inout);

    cytnx::cytnx_uint64 row_dim = 1;
    cytnx::cytnx_uint64 col_dim = 1;

    for (order_t<TenT> i = 0; i < num_of_bds_as_row && i < a_shape.size(); ++i) {
      row_dim *= a_shape[i];
    }
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      col_dim *= a_shape[i];
    }

    if (row_dim != col_dim) {
      throw std::invalid_argument("exp: matrix must be square");
    }

    // Reshape to 2D matrix
    cytnx::Tensor reshaped = inout.backend.reshape(
        {static_cast<cytnx::cytnx_int64>(row_dim), static_cast<cytnx::cytnx_int64>(col_dim)});

    auto original_dtype = reshaped.dtype();

    // Check if matrix is anti-Hermitian: H = -H^dagger
    cytnx::Tensor H_dag = cytnx::linalg::Conj(reshaped).permute({1, 0});
    cytnx::Tensor diff = reshaped + H_dag;

    // Read norm with correct type
    bool is_float
        = (original_dtype == cytnx::Type.Float || original_dtype == cytnx::Type.ComplexFloat);
    double norm_diff = is_float ? cytnx::linalg::Norm(diff).item<float>()
                                : cytnx::linalg::Norm(diff).item<double>();
    double norm_H = is_float ? cytnx::linalg::Norm(reshaped).item<float>()
                             : cytnx::linalg::Norm(reshaped).item<double>();

    cytnx::Tensor result;
    if (norm_diff < 1e-12 * (norm_H + 1e-14)) {
      // Anti-Hermitian: H = -H^dagger, so iH is Hermitian
      // exp(H) = exp(-i * iH) = ExpH(iH, -i)
      if (is_float) {
        cytnx::Tensor iH = reshaped * cytnx::cytnx_complex64(0.0f, 1.0f);
        result = cytnx::linalg::ExpH(iH, cytnx::cytnx_complex64(0.0f, -1.0f));
        // ExpH may promote to complex128, convert back if needed
        if (result.dtype() != original_dtype) {
          result = result.astype(original_dtype);
        }
      } else {
        cytnx::Tensor iH = reshaped * cytnx::cytnx_complex128(0.0, 1.0);
        result = cytnx::linalg::ExpH(iH, cytnx::cytnx_complex128(0.0, -1.0));
      }
    } else {
      // General case
      result = cytnx::linalg::ExpM(reshaped);
    }

    // Reshape back to original shape
    std::vector<cytnx::cytnx_int64> original_shape;
    for (auto s : a_shape) {
      original_shape.push_back(static_cast<cytnx::cytnx_int64>(s));
    }
    inout.backend = result.reshape(original_shape);
  }

  // exp - matrix exponential (out-of-place)
  template <typename TenT> void exp(context_handle_t<TenT>& ctx, const TenT& in,
                                    const order_t<TenT> num_of_bds_as_row, TenT& out) {
    auto a_shape = shape(ctx, in);

    cytnx::cytnx_uint64 row_dim = 1;
    cytnx::cytnx_uint64 col_dim = 1;

    for (order_t<TenT> i = 0; i < num_of_bds_as_row && i < a_shape.size(); ++i) {
      row_dim *= a_shape[i];
    }
    for (size_t i = num_of_bds_as_row; i < a_shape.size(); ++i) {
      col_dim *= a_shape[i];
    }

    if (row_dim != col_dim) {
      throw std::invalid_argument("exp: matrix must be square");
    }

    // Reshape to 2D matrix
    cytnx::Tensor reshaped = in.backend.reshape(
        {static_cast<cytnx::cytnx_int64>(row_dim), static_cast<cytnx::cytnx_int64>(col_dim)});

    auto original_dtype = reshaped.dtype();

    // Check if matrix is anti-Hermitian: H = -H^dagger
    cytnx::Tensor H_dag = cytnx::linalg::Conj(reshaped).permute({1, 0});
    cytnx::Tensor diff = reshaped + H_dag;

    // Read norm with correct type
    bool is_float
        = (original_dtype == cytnx::Type.Float || original_dtype == cytnx::Type.ComplexFloat);
    double norm_diff = is_float ? cytnx::linalg::Norm(diff).item<float>()
                                : cytnx::linalg::Norm(diff).item<double>();
    double norm_H = is_float ? cytnx::linalg::Norm(reshaped).item<float>()
                             : cytnx::linalg::Norm(reshaped).item<double>();

    cytnx::Tensor result;
    if (norm_diff < 1e-12 * (norm_H + 1e-14)) {
      // Anti-Hermitian: H = -H^dagger, so iH is Hermitian
      // exp(H) = exp(-i * iH) = ExpH(iH, -i)
      if (is_float) {
        cytnx::Tensor iH = reshaped * cytnx::cytnx_complex64(0.0f, 1.0f);
        result = cytnx::linalg::ExpH(iH, cytnx::cytnx_complex64(0.0f, -1.0f));
        // ExpH may promote to complex128, convert back if needed
        if (result.dtype() != original_dtype) {
          result = result.astype(original_dtype);
        }
      } else {
        cytnx::Tensor iH = reshaped * cytnx::cytnx_complex128(0.0, 1.0);
        result = cytnx::linalg::ExpH(iH, cytnx::cytnx_complex128(0.0, -1.0));
      }
    } else {
      // General case
      result = cytnx::linalg::ExpM(reshaped);
    }

    // Reshape back to original shape
    std::vector<cytnx::cytnx_int64> original_shape;
    for (auto s : a_shape) {
      original_shape.push_back(static_cast<cytnx::cytnx_int64>(s));
    }
    out.backend = result.reshape(original_shape);
  }

  // ===================================================================
  // Context Management
  // ===================================================================

  // Context management for TenT
  template <typename TenT> void create_context(context_handle_t<TenT>& ctx) {
    // Use GPU 0 if available, otherwise fall back to CPU
    if (cytnx::Device.Ngpus > 0) {
      ctx = cytnx::Device.cuda;
    } else {
      ctx = cytnx::Device.cpu;
    }
  }

  template <typename TenT> void destroy_context(context_handle_t<TenT>& ctx) {
    // No-op for Cytnx
  }

}  // namespace tci
