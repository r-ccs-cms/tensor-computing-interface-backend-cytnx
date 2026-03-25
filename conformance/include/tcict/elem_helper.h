#pragma once

#include <tci/tensor_traits.h>
#include <type_traits>

namespace tcict {

/// Construct an element value from real and imaginary parts.
/// For real tensor types, the imaginary part is ignored.
/// Backends with non-standard element types (e.g. cuDoubleComplex) should specialize this.
template <typename TenT>
tci::elem_t<TenT> make_elem(double re, double im = 0.0) {
  using elem_type = tci::elem_t<TenT>;
  using cplx_type = tci::cplx_t<TenT>;
  if constexpr (std::is_same_v<elem_type, cplx_type>) {
    return elem_type(re, im);
  } else {
    return static_cast<elem_type>(re);
  }
}

}  // namespace tcict
