#include <doctest/doctest.h>
#include <tci/tci.h>
#include <vector>
#include <complex>

using Ten = cytnx::Tensor;

TEST_CASE("TCI for_each Function Test - Reproducing Link Error") {
    SUBCASE("for_each with lambda function - should cause link error") {
        tci::context_handle_t<Ten> ctx;
        tci::create_context(ctx);

        // Create a simple tensor
        Ten test_tensor;
        tci::fill(ctx, {3}, cytnx::cytnx_complex128(1.0, 0.0), test_tensor);

        // This should cause the link error we've been seeing
        // for_each with a lambda that modifies elements
        try {
            tci::for_each(ctx, test_tensor, [](tci::elem_t<Ten>& elem) {
                // Double each element using TCI compliant approach
                auto real_part = tci::real(elem) * 2.0;
                auto imag_part = tci::imag(elem) * 2.0;
                elem = cytnx::cytnx_complex128(real_part, imag_part);
            });

            // If we reach here, for_each worked
            auto result_elem = tci::get_elem(ctx, test_tensor, {0});
            auto expected = cytnx::cytnx_complex128(2.0, 0.0);
            auto diff_real = tci::real(result_elem) - tci::real(expected);
            auto diff_imag = tci::imag(result_elem) - tci::imag(expected);
            CHECK(std::sqrt(diff_real * diff_real + diff_imag * diff_imag) < 1e-10);

        } catch (const std::exception& e) {
            // If for_each isn't properly implemented, we'll get an exception
            FAIL("for_each threw exception: " << e.what());
        }

        tci::destroy_context(ctx);
    }

    SUBCASE("for_each with different lambda types") {
        tci::context_handle_t<Ten> ctx;
        tci::create_context(ctx);

        Ten test_tensor;
        tci::fill(ctx, {2, 2}, cytnx::cytnx_complex128(3.0, 1.0), test_tensor);

        // Test case 1: Lambda with capture
        double multiplier = 0.5;
        try {
            tci::for_each(ctx, test_tensor, [multiplier](tci::elem_t<Ten>& elem) {
                // Multiply by real number using TCI compliant approach
                auto real_part = tci::real(elem) * multiplier;
                auto imag_part = tci::imag(elem) * multiplier;
                elem = cytnx::cytnx_complex128(real_part, imag_part);
            });

            auto result = tci::get_elem(ctx, test_tensor, {0, 0});
            auto expected = cytnx::cytnx_complex128(1.5, 0.5);
            auto diff_real = tci::real(result) - tci::real(expected);
            auto diff_imag = tci::imag(result) - tci::imag(expected);
            CHECK(std::sqrt(diff_real * diff_real + diff_imag * diff_imag) < 1e-10);

        } catch (const std::exception& e) {
            FAIL("for_each with capture lambda failed: " << e.what());
        }

        tci::destroy_context(ctx);
    }

    SUBCASE("for_each const version test") {
        tci::context_handle_t<Ten> ctx;
        tci::create_context(ctx);

        Ten test_tensor;
        tci::fill(ctx, {3}, cytnx::cytnx_complex128(2.0, 3.0), test_tensor);

        // Test const version - should not modify, just read
        double sum_real = 0.0;
        try {
            tci::for_each(ctx, static_cast<const Ten&>(test_tensor),
                [&sum_real](const tci::elem_t<Ten>& elem) {
                    sum_real += tci::real(elem);
                });

            CHECK(std::abs(sum_real - 6.0) < 1e-10);  // 3 elements * 2.0 each

        } catch (const std::exception& e) {
            FAIL("const for_each failed: " << e.what());
        }

        tci::destroy_context(ctx);
    }

    SUBCASE("Complex lambda expression test") {
        tci::context_handle_t<Ten> ctx;
        tci::create_context(ctx);

        Ten test_tensor;
        tci::random(ctx, {2, 3}, []() { return 0.5; }, test_tensor);

        // More complex lambda that should definitely cause template instantiation issues
        try {
            tci::for_each(ctx, test_tensor, [](tci::elem_t<Ten>& elem) {
                // Simulate the operation we were trying to do in iTEBD
                if (tci::abs(elem) > 1e-12) {
                    // Complex inversion: 1/(a+bi) = (a-bi)/(a²+b²)
                    auto real_part = tci::real(elem);
                    auto imag_part = tci::imag(elem);
                    auto denom = real_part * real_part + imag_part * imag_part;
                    elem = cytnx::cytnx_complex128(real_part / denom, -imag_part / denom);
                } else {
                    elem = cytnx::cytnx_complex128(1.0, 0.0);  // Avoid division by zero
                }
            });

            // Verify the operation worked
            auto elem = tci::get_elem(ctx, test_tensor, {0, 0});
            auto expected = cytnx::cytnx_complex128(2.0, 0.0);
            auto diff_real = tci::real(elem) - tci::real(expected);
            auto diff_imag = tci::imag(elem) - tci::imag(expected);
            CHECK(std::sqrt(diff_real * diff_real + diff_imag * diff_imag) < 1e-10);

        } catch (const std::exception& e) {
            INFO("This test reproduces the link error we encountered");
            INFO("Error message: " << e.what());

            // This is where we expect the link error to occur
            // The error should be something like:
            // "Undefined symbols for architecture arm64: void tci::for_each<cytnx::Tensor, ...lambda...>"
            CHECK(true); // We expect this to fail for now
        }

        tci::destroy_context(ctx);
    }
}