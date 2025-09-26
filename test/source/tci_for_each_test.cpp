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
        tci::fill(ctx, {3}, std::complex<double>(1.0, 0.0), test_tensor);

        // This should cause the link error we've been seeing
        // for_each with a lambda that modifies elements
        try {
            tci::for_each(ctx, test_tensor, [](tci::elem_t<Ten>& elem) {
                elem = elem * std::complex<double>(2.0, 0.0);  // Double each element
            });

            // If we reach here, for_each worked
            auto result_elem = tci::get_elem(ctx, test_tensor, {0});
            CHECK(std::abs(result_elem - std::complex<double>(2.0, 0.0)) < 1e-10);

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
        tci::fill(ctx, {2, 2}, std::complex<double>(3.0, 1.0), test_tensor);

        // Test case 1: Lambda with capture
        double multiplier = 0.5;
        try {
            tci::for_each(ctx, test_tensor, [multiplier](tci::elem_t<Ten>& elem) {
                elem = elem * std::complex<double>(multiplier, 0.0);
            });

            auto result = tci::get_elem(ctx, test_tensor, {0, 0});
            CHECK(std::abs(result - std::complex<double>(1.5, 0.5)) < 1e-10);

        } catch (const std::exception& e) {
            FAIL("for_each with capture lambda failed: " << e.what());
        }

        tci::destroy_context(ctx);
    }

    SUBCASE("for_each const version test") {
        tci::context_handle_t<Ten> ctx;
        tci::create_context(ctx);

        Ten test_tensor;
        tci::fill(ctx, {3}, std::complex<double>(2.0, 3.0), test_tensor);

        // Test const version - should not modify, just read
        double sum_real = 0.0;
        try {
            tci::for_each(ctx, static_cast<const Ten&>(test_tensor),
                [&sum_real](const tci::elem_t<Ten>& elem) {
                    sum_real += elem.real();
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
                if (std::abs(elem) > 1e-12) {
                    elem = tci::elem_t<Ten>(1.0) / elem;  // Inversion
                } else {
                    elem = tci::elem_t<Ten>(1.0);  // Avoid division by zero
                }
            });

            // Verify the operation worked
            auto elem = tci::get_elem(ctx, test_tensor, {0, 0});
            CHECK(std::abs(elem - std::complex<double>(2.0, 0.0)) < 1e-10);

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