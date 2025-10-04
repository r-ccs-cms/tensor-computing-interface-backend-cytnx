#pragma once

#include <cytnx.hpp>

namespace tci {

  /**
   * @brief Typed wrapper for cytnx::Tensor to provide compile-time element type information
   *
   * @tparam ElemT Element type (cytnx::cytnx_double, cytnx::cytnx_complex128, etc.)
   *
   * This wrapper enables compile-time type safety for Cytnx tensors, which are
   * dynamically typed at runtime. It provides a TenT type that can be used with
   * tci::tensor_traits, similar to other statically-typed tensor backends.
   *
   * Usage:
   * @code
   * using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
   * using Elem = tci::tensor_traits<Tensor>::elem_t;  // = std::complex<double>
   *
   * Tensor t;
   * tci::allocate(ctx, {3, 3}, t);
   * Elem elem = tci::get_elem(ctx, t, {0, 0});
   * auto result = elem / 2.0;  // Standard C++ operators work
   * @endcode
   */
  template <typename ElemT> struct CytnxTensor {
    /// The underlying Cytnx tensor backend
    cytnx::Tensor backend;

    /// Compile-time element type information
    using elem_type = ElemT;

    /// Default constructor
    CytnxTensor() = default;

    /// Construct from existing cytnx::Tensor
    explicit CytnxTensor(cytnx::Tensor t) : backend(std::move(t)) {}

    /// Copy constructor
    CytnxTensor(const CytnxTensor&) = default;

    /// Move constructor
    CytnxTensor(CytnxTensor&&) = default;

    /// Copy assignment
    CytnxTensor& operator=(const CytnxTensor&) = default;

    /// Move assignment
    CytnxTensor& operator=(CytnxTensor&&) = default;

    /// Destructor
    ~CytnxTensor() = default;
  };

}  // namespace tci