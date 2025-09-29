#include <doctest/doctest.h>
#include "tci/tci.h"
#include <complex>

TEST_CASE("TCI std::variant elem_t Implementation Tests") {
    tci::context_handle_t<cytnx::Tensor> ctx;
    tci::create_context(ctx);

    SUBCASE("Real double tensor operations") {
        // Create a double tensor
        cytnx::Tensor tensor = cytnx::zeros({2, 2}, cytnx::Type.Double);

        // Test setting a real value using TCI interface
        tci::elem_t<cytnx::Tensor> real_val = cytnx::cytnx_double(3.14);
        CHECK_NOTHROW(tci::set_elem(ctx, tensor, {0, 0}, real_val));

        // Verify tensor dtype is preserved
        CHECK(tensor.dtype() == cytnx::Type.Double);

        // Test getting the value back using TCI interface
        auto retrieved = tci::get_elem(ctx, tensor, {0, 0});
        CHECK(std::abs(tci::real(retrieved) - 3.14) < 1e-10);
        CHECK(std::abs(tci::imag(retrieved)) < 1e-10);  // Should be zero for real tensor
    }

    SUBCASE("Real float tensor operations") {
        // Create a float tensor
        cytnx::Tensor tensor = cytnx::zeros({2, 2}, cytnx::Type.Float);

        // Test setting a float value
        tci::elem_t<cytnx::Tensor> float_val = cytnx::cytnx_float(2.5f);
        CHECK_NOTHROW(tci::set_elem(ctx, tensor, {1, 1}, float_val));

        // Verify tensor dtype is preserved
        CHECK(tensor.dtype() == cytnx::Type.Float);

        // Test retrieval
        auto retrieved = tci::get_elem(ctx, tensor, {1, 1});
        CHECK(std::abs(tci::real(retrieved) - 2.5) < 1e-6);
        CHECK(std::abs(tci::imag(retrieved)) < 1e-10);
    }

    SUBCASE("Complex double tensor operations") {
        // Create a complex double tensor
        cytnx::Tensor tensor = cytnx::zeros({2, 2}, cytnx::Type.ComplexDouble);

        // Test setting a complex value
        tci::elem_t<cytnx::Tensor> complex_val = cytnx::cytnx_complex128(1.5, 2.5);
        CHECK_NOTHROW(tci::set_elem(ctx, tensor, {0, 1}, complex_val));

        // Verify tensor dtype is preserved
        CHECK(tensor.dtype() == cytnx::Type.ComplexDouble);

        // Test retrieval
        auto retrieved = tci::get_elem(ctx, tensor, {0, 1});
        CHECK(std::abs(tci::real(retrieved) - 1.5) < 1e-10);
        CHECK(std::abs(tci::imag(retrieved) - 2.5) < 1e-10);
    }

    SUBCASE("Complex float tensor operations") {
        // Create a complex float tensor
        cytnx::Tensor tensor = cytnx::zeros({2, 2}, cytnx::Type.ComplexFloat);

        // Test setting a complex float value
        tci::elem_t<cytnx::Tensor> complex_val = cytnx::cytnx_complex64(0.75f, 1.25f);
        CHECK_NOTHROW(tci::set_elem(ctx, tensor, {1, 0}, complex_val));

        // Verify tensor dtype is preserved
        CHECK(tensor.dtype() == cytnx::Type.ComplexFloat);

        // Test retrieval
        auto retrieved = tci::get_elem(ctx, tensor, {1, 0});
        CHECK(std::abs(tci::real(retrieved) - 0.75) < 1e-6);
        CHECK(std::abs(tci::imag(retrieved) - 1.25) < 1e-6);
    }

    SUBCASE("Cross-type compatibility - Complex input to real tensor") {
        // Create a real double tensor
        cytnx::Tensor tensor = cytnx::zeros({2, 2}, cytnx::Type.Double);

        // Set complex value (imaginary part should be ignored)
        tci::elem_t<cytnx::Tensor> complex_input = cytnx::cytnx_complex128(4.0, 3.0);
        CHECK_NOTHROW(tci::set_elem(ctx, tensor, {0, 0}, complex_input));

        // Only real part should be stored
        auto retrieved = tci::get_elem(ctx, tensor, {0, 0});
        CHECK(std::abs(tci::real(retrieved) - 4.0) < 1e-10);
        CHECK(std::abs(tci::imag(retrieved)) < 1e-10);  // Should be zero
    }

    SUBCASE("Cross-type compatibility - Real input to complex tensor") {
        // Create a complex double tensor
        cytnx::Tensor tensor = cytnx::zeros({2, 2}, cytnx::Type.ComplexDouble);

        // Set real value (imaginary part should be zero)
        tci::elem_t<cytnx::Tensor> real_input = cytnx::cytnx_double(5.5);
        CHECK_NOTHROW(tci::set_elem(ctx, tensor, {1, 1}, real_input));

        // Real part stored, imaginary part should be zero
        auto retrieved = tci::get_elem(ctx, tensor, {1, 1});
        CHECK(std::abs(tci::real(retrieved) - 5.5) < 1e-10);
        CHECK(std::abs(tci::imag(retrieved)) < 1e-10);
    }

    SUBCASE("TCI helper functions test") {
        // Test helper functions with different variant types
        tci::elem_t<cytnx::Tensor> real_val = cytnx::cytnx_double(3.0);
        tci::elem_t<cytnx::Tensor> complex_val = cytnx::cytnx_complex128(4.0, 3.0);

        // Test real() function
        CHECK(std::abs(tci::real(real_val) - 3.0) < 1e-10);
        CHECK(std::abs(tci::real(complex_val) - 4.0) < 1e-10);

        // Test imag() function
        CHECK(std::abs(tci::imag(real_val)) < 1e-10);
        CHECK(std::abs(tci::imag(complex_val) - 3.0) < 1e-10);

        // Test abs() function
        CHECK(std::abs(tci::abs(real_val) - 3.0) < 1e-10);
        CHECK(std::abs(tci::abs(complex_val) - 5.0) < 1e-10);  // sqrt(4^2 + 3^2) = 5
    }

    SUBCASE("Type promotion and precision") {
        // Test float to double promotion
        cytnx::Tensor double_tensor = cytnx::zeros({1, 1}, cytnx::Type.Double);
        tci::elem_t<cytnx::Tensor> float_input = cytnx::cytnx_float(1.5f);

        CHECK_NOTHROW(tci::set_elem(ctx, double_tensor, {0, 0}, float_input));
        auto retrieved = tci::get_elem(ctx, double_tensor, {0, 0});
        CHECK(std::abs(tci::real(retrieved) - 1.5) < 1e-6);

        // Test complex64 to complex128 promotion
        cytnx::Tensor complex128_tensor = cytnx::zeros({1, 1}, cytnx::Type.ComplexDouble);
        tci::elem_t<cytnx::Tensor> complex64_input = cytnx::cytnx_complex64(2.0f, 1.5f);

        CHECK_NOTHROW(tci::set_elem(ctx, complex128_tensor, {0, 0}, complex64_input));
        auto retrieved_complex = tci::get_elem(ctx, complex128_tensor, {0, 0});
        CHECK(std::abs(tci::real(retrieved_complex) - 2.0) < 1e-6);
        CHECK(std::abs(tci::imag(retrieved_complex) - 1.5) < 1e-6);
    }

    tci::destroy_context(ctx);
}

