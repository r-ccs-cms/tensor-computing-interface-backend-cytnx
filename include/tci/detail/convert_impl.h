#pragma once

#include <cytnx.hpp>
#include <sstream>

#include "tci/cytnx_tensor_traits.h"
#include "tci/cytnx_typed_tensor.h"
#include "tci/debugging.h"

namespace tci {

  namespace detail {
    // Helper function to get Cytnx type ID from C++ type
    template <typename ElemT> constexpr unsigned int get_cytnx_type_id() {
      if constexpr (std::is_same_v<ElemT, cytnx::cytnx_double>) {
        return cytnx::Type.Double;
      } else if constexpr (std::is_same_v<ElemT, cytnx::cytnx_float>) {
        return cytnx::Type.Float;
      } else if constexpr (std::is_same_v<ElemT, cytnx::cytnx_complex128>) {
        return cytnx::Type.ComplexDouble;
      } else if constexpr (std::is_same_v<ElemT, cytnx::cytnx_complex64>) {
        return cytnx::Type.ComplexFloat;
      } else if constexpr (std::is_same_v<ElemT, cytnx::cytnx_int64>) {
        return cytnx::Type.Int64;
      } else if constexpr (std::is_same_v<ElemT, cytnx::cytnx_uint64>) {
        return cytnx::Type.Uint64;
      } else if constexpr (std::is_same_v<ElemT, cytnx::cytnx_int32>) {
        return cytnx::Type.Int32;
      } else if constexpr (std::is_same_v<ElemT, cytnx::cytnx_uint32>) {
        return cytnx::Type.Uint32;
      } else if constexpr (std::is_same_v<ElemT, cytnx::cytnx_int16>) {
        return cytnx::Type.Int16;
      } else if constexpr (std::is_same_v<ElemT, cytnx::cytnx_uint16>) {
        return cytnx::Type.Uint16;
      } else if constexpr (std::is_same_v<ElemT, cytnx::cytnx_bool>) {
        return cytnx::Type.Bool;
      } else {
        static_assert(sizeof(ElemT) == 0, "Unsupported element type for Cytnx conversion");
        return cytnx::Type.Void;
      }
    }
  }  // namespace detail

  /**
   * @brief Convert between different CytnxTensor<ElemT> specializations
   *
   * Primary use case: Element type casting between different types
   * (e.g., CytnxTensor<double> -> CytnxTensor<std::complex<double>>)
   *
   * @tparam ElemT1 Source element type
   * @tparam ElemT2 Target element type
   * @param ctx1 Source context (device)
   * @param t1 Source tensor
   * @param ctx2 Target context (device)
   * @param t2 Target tensor (output)
   */
  template <typename ElemT1, typename ElemT2>
  void convert(context_handle_t<CytnxTensor<ElemT1>>& ctx1, const CytnxTensor<ElemT1>& t1,
               context_handle_t<CytnxTensor<ElemT2>>& ctx2, CytnxTensor<ElemT2>& t2) {
    // Handle empty tensor case
    if (t1.backend.dtype() == cytnx::Type.Void || t1.backend.shape().size() == 0) {
      t2.backend = cytnx::Tensor();
      return;
    }

    // Create tensor info string for profiling
    std::ostringstream info;
    auto shape = t1.backend.shape();
    info << "shape=[";
    for (size_t i = 0; i < shape.size(); ++i) {
      if (i > 0) info << ",";
      info << shape[i];
    }
    info << "], ElemT1=" << cytnx::Type_class::getname(detail::get_cytnx_type_id<ElemT1>())
         << " -> ElemT2=" << cytnx::Type_class::getname(detail::get_cytnx_type_id<ElemT2>())
         << ", ctx1=" << ctx1 << " -> ctx2=" << ctx2;

    TCI_TIME_FUNCTION_WITH_INFO("tci::convert", info.str());

    // If same type, clone to ensure deep copy
    cytnx::Tensor converted;
    if constexpr (std::is_same_v<ElemT1, ElemT2>) {
      converted = t1.backend.clone();
    } else {
      // Convert element type using Cytnx's astype
      unsigned int target_type = detail::get_cytnx_type_id<ElemT2>();
      converted = t1.backend.astype(target_type);
    }

    // Handle device transfer if contexts differ
    if (ctx1 != ctx2) {
      if (ctx2 >= 0) {  // GPU device
        converted = converted.to(ctx2);
      } else if (ctx2 == cytnx::Device.cpu) {  // CPU device
        converted = converted.to(cytnx::Device.cpu);
      }
    }

    t2.backend = std::move(converted);
  }

}  // namespace tci