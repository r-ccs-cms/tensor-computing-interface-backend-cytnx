#pragma once

#include <variant>
#include <complex>
#include <cmath>
#include <cytnx.hpp>

#include "tci/tensor_traits.h"

namespace tci {

    // Forward declaration for elem_t
    template<typename TenT>
    using elem_t = typename tensor_traits<TenT>::elem_t;

    /**
     * @brief Extract the real part of a variant element
     */
    inline double real(const std::variant<cytnx::cytnx_double, cytnx::cytnx_float,
                                         cytnx::cytnx_complex128, cytnx::cytnx_complex64>& element) {
        return std::visit([](const auto& value) -> double {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, cytnx::cytnx_double> ||
                         std::is_same_v<T, cytnx::cytnx_float>) {
                // Real types - return as double
                return static_cast<double>(value);
            } else {
                // Complex types - return real part
                return static_cast<double>(value.real());
            }
        }, element);
    }

    /**
     * @brief Extract the imaginary part of a variant element
     */
    inline double imag(const std::variant<cytnx::cytnx_double, cytnx::cytnx_float,
                                          cytnx::cytnx_complex128, cytnx::cytnx_complex64>& element) {
        return std::visit([](const auto& value) -> double {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, cytnx::cytnx_double> ||
                         std::is_same_v<T, cytnx::cytnx_float>) {
                // Real types - imaginary part is zero
                return 0.0;
            } else {
                // Complex types - return imaginary part
                return static_cast<double>(value.imag());
            }
        }, element);
    }

    /**
     * @brief Calculate the absolute value of a variant element
     */
    inline double abs(const std::variant<cytnx::cytnx_double, cytnx::cytnx_float,
                                         cytnx::cytnx_complex128, cytnx::cytnx_complex64>& element) {
        return std::visit([](const auto& value) -> double {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, cytnx::cytnx_double> ||
                         std::is_same_v<T, cytnx::cytnx_float>) {
                // Real types - use std::abs
                return static_cast<double>(std::abs(value));
            } else {
                // Complex types - use std::abs for complex
                return std::abs(value);
            }
        }, element);
    }

    /**
     * @brief Convert variant element to cytnx::cytnx_complex128 for compatibility
     */
    inline cytnx::cytnx_complex128 to_complex128(const std::variant<cytnx::cytnx_double, cytnx::cytnx_float,
                                                                    cytnx::cytnx_complex128, cytnx::cytnx_complex64>& element) {
        return std::visit([](const auto& value) -> cytnx::cytnx_complex128 {
            using T = std::decay_t<decltype(value)>;
            if constexpr (std::is_same_v<T, cytnx::cytnx_double> ||
                         std::is_same_v<T, cytnx::cytnx_float>) {
                // Real types - imaginary part is zero
                return cytnx::cytnx_complex128(static_cast<double>(value), 0.0);
            } else if constexpr (std::is_same_v<T, cytnx::cytnx_complex128>) {
                // Already complex128
                return value;
            } else {
                // cytnx_complex64 - convert to complex128
                return cytnx::cytnx_complex128(
                    static_cast<double>(value.real()),
                    static_cast<double>(value.imag())
                );
            }
        }, element);
    }

    // Type alias for easier reference
    using cytnx_variant_t = std::variant<cytnx::cytnx_double, cytnx::cytnx_float,
                                        cytnx::cytnx_complex128, cytnx::cytnx_complex64>;


} // namespace tci