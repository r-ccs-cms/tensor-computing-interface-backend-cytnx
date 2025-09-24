#include <doctest/doctest.h>
#include <tci/tci.h>
#include <vector>
#include <complex>
#include <cstdlib>

TEST_CASE("Miscellaneous Functions") {
    using Ten = cytnx::Tensor;
    using namespace tci;

    // Setup context
    auto ctx = create_context<context_handle_t<Ten>>();

    SUBCASE("tci::to_container") {
        // Create a 2x3 tensor with known values
        Ten a = zeros<Ten>(ctx, {2, 3});

        // Set some values
        // Note: We need to set values using Cytnx API directly for this test
        a = cytnx::arange(0, 6, 1).reshape({2, 3}).astype(cytnx::Type.ComplexDouble);

        // Create container to store results
        std::vector<std::complex<double>> container(6);

        // Define row-major coordinate to index mapping
        auto row_major_map = [](const elem_coors_t<Ten>& coors) -> std::ptrdiff_t {
            return coors[0] * 3 + coors[1];  // 2x3 tensor, row-major
        };

        // Copy tensor to container
        to_container(ctx, a, container.begin(), row_major_map);

        // Verify values
        for (int i = 0; i < 6; ++i) {
            CHECK(container[i].real() == doctest::Approx(static_cast<double>(i)));
            CHECK(container[i].imag() == doctest::Approx(0.0));
        }
    }

    SUBCASE("tci::show") {
        // Create a simple 2x2 identity matrix
        Ten a = eye<Ten>(ctx, 2);

        // This should print to stdout - we can't easily test output
        // but we can ensure it doesn't crash
        CHECK_NOTHROW(show(ctx, a));
    }

    SUBCASE("tci::eq") {
        Ten a = eye<Ten>(ctx, 2);
        Ten b = eye<Ten>(ctx, 2);
        Ten c = zeros<Ten>(ctx, {2, 2});

        // Test equality with small epsilon
        CHECK(eq(ctx, a, b, std::complex<double>(1e-10, 0)));

        // Test inequality
        CHECK_FALSE(eq(ctx, a, c, std::complex<double>(1e-10, 0)));

        // Test with larger epsilon (should be equal to zero within tolerance)
        CHECK(eq(ctx, c, zeros<Ten>(ctx, {2, 2}), std::complex<double>(1e-6, 0)));
    }

    SUBCASE("tci::convert") {
        Ten a = eye<Ten>(ctx, 3);
        Ten b;

        auto ctx2 = create_context<context_handle_t<Ten>>();

        // Test conversion (same type, possibly different context)
        CHECK_NOTHROW(convert(ctx, a, ctx2, b));

        // Verify the conversion worked
        CHECK(eq(ctx, a, b, std::complex<double>(1e-10, 0)));
    }

    // Cleanup
    destroy_context(ctx);
}

TEST_CASE("TCI_VERBOSE Environment Variable Support") {
    using namespace tci::debug;

    SUBCASE("Default verbose level (no environment variable)") {
        // Reset cached value by creating new process or test in isolation
        // For this test, we assume TCI_VERBOSE is not set initially
        CHECK(get_verbose_level() >= 0);
        CHECK(get_verbose_level() <= 2);
    }

    SUBCASE("Verbose level functions") {
        // Test the basic functionality (actual values depend on environment)
        int level = get_verbose_level();

        CHECK(is_verbose(0) == (level >= 0));
        CHECK(is_verbose(1) == (level >= 1));
        CHECK(is_verbose(2) == (level >= 2));
        CHECK(is_verbose(3) == false);  // Level 3 doesn't exist, should always be false
    }

    SUBCASE("Timer functionality") {
        // Test that timer doesn't crash
        CHECK_NOTHROW({
            Timer t("test_timer");
            // Timer will print timing info on destruction if verbose >= 2
        });
    }

    SUBCASE("Function entry logging") {
        // Test that function entry logging doesn't crash
        CHECK_NOTHROW(print_function_entry("test_function"));
        CHECK_NOTHROW(print_function_entry("test_function", "additional_info"));
    }
}

TEST_CASE("Integration with TCI_VERBOSE") {
    using Ten = cytnx::Tensor;
    using namespace tci;

    // Setup context
    auto ctx = create_context<context_handle_t<Ten>>();

    SUBCASE("Verbose instrumentation in miscellaneous functions") {
        // These tests verify that functions with TCI_VERBOSE instrumentation
        // don't crash and work correctly regardless of verbose level

        Ten a = eye<Ten>(ctx, 2);

        // Test show with instrumentation
        CHECK_NOTHROW(show(ctx, a));

        // Test eq with instrumentation
        Ten b = eye<Ten>(ctx, 2);
        CHECK(eq(ctx, a, b, std::complex<double>(1e-10, 0)));

        // Test convert with instrumentation
        Ten c;
        auto ctx2 = create_context<context_handle_t<Ten>>();
        CHECK_NOTHROW(convert(ctx, a, ctx2, c));

        // Test to_container with instrumentation
        std::vector<std::complex<double>> container(4);
        auto row_major_map = [](const elem_coors_t<Ten>& coors) -> std::ptrdiff_t {
            return coors[0] * 2 + coors[1];  // 2x2 tensor, row-major
        };
        CHECK_NOTHROW(to_container(ctx, a, container.begin(), row_major_map));

        destroy_context(ctx2);
    }

    // Cleanup
    destroy_context(ctx);
}