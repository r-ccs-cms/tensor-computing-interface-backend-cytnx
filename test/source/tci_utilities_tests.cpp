#include <doctest/doctest.h>
#include <tci/tci.h>

#include <complex>
#include <cstdlib>
#include <vector>

TEST_CASE("Miscellaneous Functions") {
  using Ten = tci::CytnxTensor<cytnx::cytnx_complex128>;
  using namespace tci;

  // Setup context
  context_handle_t<Ten> ctx;
  create_context(ctx);

  SUBCASE("tci::to_container") {
    // Create a 2x3 tensor with known values
    Ten a = zeros<Ten>(ctx, {2, 3});

    for_each_with_coors(ctx, a, [](elem_t<Ten>& elem, const elem_coors_t<Ten>& coors) {
      elem = elem_t<Ten>(coors[0] * 3 + coors[1], 0.0);
    });

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

  SUBCASE("tci::convert - same context (copy behavior)") {
    Ten a = eye<Ten>(ctx, 3);
    Ten b;

    // Test conversion with same context (should behave like copy)
    CHECK_NOTHROW(convert(ctx, a, ctx, b));

    // Verify the conversion worked - data should be identical
    CHECK(eq(ctx, a, b, std::complex<double>(1e-15, 0)));

    // Verify shapes match
    CHECK(shape(ctx, a) == shape(ctx, b));
    CHECK(rank(ctx, a) == rank(ctx, b));

    // Verify independence (modify original, copy should be unaffected)
    Ten original_b = copy(ctx, b);
    set_elem(ctx, a, {0, 0}, std::complex<double>(999.0, 0.0));
    CHECK(eq(ctx, b, original_b, std::complex<double>(1e-15, 0)));
  }

  SUBCASE("tci::convert - different contexts") {
    Ten a = eye<Ten>(ctx, 3);
    Ten b;

    context_handle_t<Ten> ctx2;
    create_context(ctx2);

    // Test conversion between different contexts
    CHECK_NOTHROW(convert(ctx, a, ctx2, b));

    // Verify the conversion worked
    CHECK(eq(ctx, a, b, std::complex<double>(1e-10, 0)));

    // Verify shapes match
    CHECK(shape(ctx, a) == shape(ctx, b));
    CHECK(rank(ctx, a) == rank(ctx, b));

    destroy_context(ctx2);
  }

  SUBCASE("tci::convert - preserve data integrity") {
    Ten a;
    zeros(ctx, {2, 3}, a);

    // Set known values
    set_elem(ctx, a, {0, 0}, std::complex<double>(1.23, 4.56));
    set_elem(ctx, a, {1, 2}, std::complex<double>(-7.89, 0.12));

    Ten b;
    context_handle_t<Ten> ctx2;
    create_context(ctx2);

    // Convert
    convert(ctx, a, ctx2, b);

    // Verify specific elements preserved
    auto val1 = get_elem(ctx, b, {0, 0});
    CHECK(std::abs(val1.real() - 1.23) < 1e-14);
    CHECK(std::abs(val1.imag() - 4.56) < 1e-14);

    auto val2 = get_elem(ctx, b, {1, 2});
    CHECK(std::abs(val2.real() - (-7.89)) < 1e-14);
    CHECK(std::abs(val2.imag() - 0.12) < 1e-14);

    destroy_context(ctx2);
  }

  SUBCASE("tci::convert - empty tensor") {
    Ten a, b;
    context_handle_t<Ten> ctx2;
    create_context(ctx2);

    // Convert empty tensor
    CHECK_NOTHROW(convert(ctx, a, ctx2, b));

    // Both should have same characteristics
    CHECK(shape(ctx, a) == shape(ctx, b));
    CHECK(rank(ctx, a) == rank(ctx, b));

    destroy_context(ctx2);
  }

  SUBCASE("tci::convert - large tensor") {
    Ten a, b;
    zeros(ctx, {10, 10, 5}, a);

    // Set corner elements
    set_elem(ctx, a, {0, 0, 0}, std::complex<double>(1.0, 0.0));
    set_elem(ctx, a, {9, 9, 4}, std::complex<double>(0.0, 1.0));

    context_handle_t<Ten> ctx2;
    create_context(ctx2);
    convert(ctx, a, ctx2, b);

    // Verify corner elements preserved
    auto corner1 = get_elem(ctx, b, {0, 0, 0});
    CHECK(std::abs(corner1.real() - 1.0) < 1e-14);
    CHECK(std::abs(corner1.imag() - 0.0) < 1e-14);

    auto corner2 = get_elem(ctx, b, {9, 9, 4});
    CHECK(std::abs(corner2.real() - 0.0) < 1e-14);
    CHECK(std::abs(corner2.imag() - 1.0) < 1e-14);

    destroy_context(ctx2);
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
  using Ten = tci::CytnxTensor<cytnx::cytnx_complex128>;
  using namespace tci;

  // Setup context
  context_handle_t<Ten> ctx;
  create_context(ctx);

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
    context_handle_t<Ten> ctx2;
    create_context(ctx2);
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

TEST_CASE("Version Information") {
  using Ten = tci::CytnxTensor<cytnx::cytnx_double>;
  using namespace tci;

  SUBCASE("tci::version") {
    std::string ver = version<Ten>();

    // Check that version string is not empty
    CHECK(!ver.empty());

    // Check that it matches expected format (contains "数字.数字" pattern)
    // Example valid formats: "1.0", "1.0-rc.1", "2.1.3"
    bool has_version_pattern = (ver.find_first_of("0123456789") != std::string::npos) &&
                                (ver.find('.') != std::string::npos);
    CHECK(has_version_pattern);
  }
}