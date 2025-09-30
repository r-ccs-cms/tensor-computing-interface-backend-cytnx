#pragma once

#include <complex>
#include <variant>
#include <cytnx.hpp>

#include "tci/tensor_traits.h"
#include "tci/cytnx_typed_tensor.h"

namespace tci {

  /**
   * @brief Specialization of tensor_traits for Cytnx::Tensor
   *
   * This specialization provides the type mappings between TCI's generic interface
   * and Cytnx's specific tensor implementation.
   */
  template <> struct tensor_traits<cytnx::Tensor> {
    using ten_t = cytnx::Tensor;
    using rank_t = cytnx::cytnx_uint64;
    using shape_t = List<cytnx::cytnx_uint64>;
    using bond_dim_t = cytnx::cytnx_uint64;
    using bond_idx_t = cytnx::cytnx_uint64;
    using bond_label_t = cytnx::cytnx_int64;
    using ten_size_t = cytnx::cytnx_uint64;
    using elem_t = std::variant<
        cytnx::cytnx_double,
        cytnx::cytnx_float,
        cytnx::cytnx_complex128,
        cytnx::cytnx_complex64
    >;
    using elem_coor_t = cytnx::cytnx_uint64;
    using elem_coors_t = List<cytnx::cytnx_uint64>;
    using real_t = cytnx::cytnx_double;
    using real_ten_t = cytnx::Tensor;  // Cytnx tensors can hold different types
    using cplx_t = cytnx::cytnx_complex128;
    using cplx_ten_t = cytnx::Tensor;  // Cytnx tensors can hold different types
    using context_handle_t
        = int;  // Cytnx uses device ID for context (Device.cpu=-1, Device.cuda=0+gpu_id)
  };

  /**
   * @brief Specialization of tensor_traits for const cytnx::Tensor
   *
   * This specialization provides the type mappings for const versions of Cytnx tensors,
   * which is needed for read-only operations like const for_each.
   */
  template <> struct tensor_traits<const cytnx::Tensor> {
    using ten_t = const cytnx::Tensor;
    using rank_t = cytnx::cytnx_uint64;
    using shape_t = List<cytnx::cytnx_uint64>;
    using bond_dim_t = cytnx::cytnx_uint64;
    using bond_idx_t = cytnx::cytnx_uint64;
    using bond_label_t = cytnx::cytnx_int64;
    using ten_size_t = cytnx::cytnx_uint64;
    using elem_t = std::variant<
        cytnx::cytnx_double,
        cytnx::cytnx_float,
        cytnx::cytnx_complex128,
        cytnx::cytnx_complex64
    >;
    using elem_coor_t = cytnx::cytnx_uint64;
    using elem_coors_t = List<cytnx::cytnx_uint64>;
    using real_t = cytnx::cytnx_double;
    using real_ten_t = const cytnx::Tensor;
    using cplx_t = cytnx::cytnx_complex128;
    using cplx_ten_t = const cytnx::Tensor;
    using context_handle_t = int;
  };

  // Note: Cytnx::Tensor is a dynamically typed tensor that can hold different
  // element types at runtime. The traits above provide the most general types,
  // and specific element types are handled through runtime type checking
  // in the implementation functions.

  /**
   * @brief Specialization of tensor_traits for CytnxTensor<ElemT>
   *
   * This specialization provides compile-time type information for typed Cytnx tensors.
   * Unlike cytnx::Tensor, CytnxTensor<ElemT> has a fixed element type determined at
   * compile time, making it compatible with TCI's type system expectations.
   *
   * @tparam ElemT Element type (cytnx::cytnx_double, cytnx::cytnx_complex128, etc.)
   */
  template <typename ElemT>
  struct tensor_traits<CytnxTensor<ElemT>> {
    using ten_t = CytnxTensor<ElemT>;
    using rank_t = cytnx::cytnx_uint64;
    using shape_t = List<cytnx::cytnx_uint64>;
    using bond_dim_t = cytnx::cytnx_uint64;
    using bond_idx_t = cytnx::cytnx_uint64;
    using bond_label_t = cytnx::cytnx_int64;
    using ten_size_t = cytnx::cytnx_uint64;

    // Element type is fixed at compile time (TCI spec compliant)
    using elem_t = ElemT;

    using elem_coor_t = cytnx::cytnx_uint64;
    using elem_coors_t = List<cytnx::cytnx_uint64>;

    // Derive real_t from elem_t
    using real_t = std::conditional_t<
        std::is_same_v<ElemT, cytnx::cytnx_complex128>, cytnx::cytnx_double,
        std::conditional_t<std::is_same_v<ElemT, cytnx::cytnx_complex64>, cytnx::cytnx_float,
        ElemT>>;

    using real_ten_t = CytnxTensor<real_t>;

    // Derive cplx_t from real_t
    using cplx_t = std::conditional_t<
        std::is_same_v<real_t, cytnx::cytnx_double>, cytnx::cytnx_complex128,
        cytnx::cytnx_complex64>;

    using cplx_ten_t = CytnxTensor<cplx_t>;

    using context_handle_t = int;
  };

}  // namespace tci