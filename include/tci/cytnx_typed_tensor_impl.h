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

  template <typename ElemT>
  ten_size_t<CytnxTensor<ElemT>> size(context_handle_t<CytnxTensor<ElemT>>& ctx,
                                       const CytnxTensor<ElemT>& a) {
    return static_cast<ten_size_t<CytnxTensor<ElemT>>>(a.backend.storage().size());
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
    // Parameters: tensor, chi_max, s_min, is_UvT, mindim, return_err
    auto svd_result = cytnx::linalg::Svd_truncate(a_reshaped, chi_max, s_min, true, 1, 1);

    if (svd_result.size() < 4) {
      throw std::runtime_error("trunc_svd: unexpected result size from Svd_truncate");
    }

    // Extract S, U, Vt, error (order: S, U, V†, err)
    auto& s_backend = svd_result[0];
    auto& u_backend = svd_result[1];
    auto& vt_backend = svd_result[2];
    auto& err_tensor = svd_result[3];

    // Extract truncation error from result
    bond_dim_t<CytnxTensor<ElemT>> bond_dim = s_backend.shape()[0];
    if (err_tensor.dtype() == cytnx::Type.Double) {
      trunc_err = err_tensor.template item<double>();
    } else if (err_tensor.dtype() == cytnx::Type.Float) {
      trunc_err = static_cast<double>(err_tensor.template item<float>());
    } else if (err_tensor.dtype() == cytnx::Type.ComplexDouble) {
      trunc_err = std::real(err_tensor.template item<cytnx::cytnx_complex128>());
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

}  // namespace tci