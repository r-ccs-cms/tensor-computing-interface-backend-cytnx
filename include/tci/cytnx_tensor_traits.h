#pragma once

#include <complex>
#include <cytnx.hpp>

#include "tci/tensor_traits.h"

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
    using elem_t = cytnx::cytnx_complex128;  // Default to complex for generality
    using elem_coor_t = cytnx::cytnx_uint64;
    using elem_coors_t = List<cytnx::cytnx_uint64>;
    using real_t = cytnx::cytnx_double;
    using real_ten_t = cytnx::Tensor;  // Cytnx tensors can hold different types
    using cplx_t = cytnx::cytnx_complex128;
    using cplx_ten_t = cytnx::Tensor;  // Cytnx tensors can hold different types
    using context_handle_t
        = int;  // Cytnx uses device ID for context (Device.cpu=-1, Device.cuda=0+gpu_id)
  };

  // Note: Cytnx::Tensor is a dynamically typed tensor that can hold different
  // element types at runtime. The traits above provide the most general types,
  // and specific element types are handled through runtime type checking
  // in the implementation functions.

}  // namespace tci