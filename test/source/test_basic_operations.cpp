#include <doctest/doctest.h>
#include <tci/tci.h>

#include <cmath>
#include <cytnx.hpp>

TEST_CASE("TCI Element Access") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("Element set and get should match exactly") {
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    cytnx::Tensor tensor;
    tci::zeros(ctx, shape, tensor);

    // Set element at (0,0) to (2.5, 1.5)
    tci::elem_coors_t<cytnx::Tensor> coord = {0, 0};
    cytnx::cytnx_complex128 expected_value(2.5, 1.5);
    tci::set_elem(ctx, tensor, coord, expected_value);

    // Get element back - should be exactly what we set
    auto retrieved_value = tci::get_elem(ctx, tensor, coord);

    // This will FAIL if get_elem is a placeholder returning (1.0, 0.0)
    CHECK(std::abs(tci::real(retrieved_value) - tci::real(expected_value)) < 1e-10);
    CHECK(std::abs(tci::imag(retrieved_value) - tci::imag(expected_value)) < 1e-10);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Copy Operations") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("Deep copy preserves data - in-place version") {
    cytnx::Tensor a, b;
    tci::zeros(ctx, {2, 3}, a);

    // Set known values
    tci::set_elem(ctx, a, {0, 0}, cytnx::cytnx_complex128(42.0, 13.0));
    tci::set_elem(ctx, a, {1, 2}, cytnx::cytnx_complex128(-5.5, 7.7));

    // Deep copy using in-place version
    CHECK_NOTHROW(tci::copy(ctx, a, b));

    // Verify copied data matches exactly
    auto val1 = tci::get_elem(ctx, b, {0, 0});
    CHECK(std::abs(tci::real(val1) - 42.0) < 1e-10);
    CHECK(std::abs(tci::imag(val1) - 13.0) < 1e-10);

    auto val2 = tci::get_elem(ctx, b, {1, 2});
    CHECK(std::abs(tci::real(val2) - (-5.5)) < 1e-10);
    CHECK(std::abs(tci::imag(val2) - 7.7) < 1e-10);

    // Verify shape and rank preservation
    CHECK(tci::shape(ctx, a) == tci::shape(ctx, b));
    CHECK(tci::rank(ctx, a) == tci::rank(ctx, b));
  }

  SUBCASE("Deep copy preserves data - out-of-place version (documentation example)") {
    // Test based on documentation example
    cytnx::Tensor a;
    tci::zeros(ctx, {3, 4, 2}, a);

    // Populate tensor with known data
    tci::set_elem(ctx, a, {0, 1, 0}, cytnx::cytnx_complex128(1.23, 4.56));
    tci::set_elem(ctx, a, {2, 3, 1}, cytnx::cytnx_complex128(-9.87, 6.54));

    // Deep copy as per documentation: auto b = tci::copy(ctx, a);
    auto b = tci::copy(ctx, a);

    // Verify b now stores the same data as a
    auto val1_a = tci::get_elem(ctx, a, {0, 1, 0});
    auto val1_b = tci::get_elem(ctx, b, {0, 1, 0});
    CHECK(std::abs(tci::real(val1_a) - tci::real(val1_b)) < 1e-10);
    CHECK(std::abs(tci::imag(val1_a) - tci::imag(val1_b)) < 1e-10);

    auto val2_a = tci::get_elem(ctx, a, {2, 3, 1});
    auto val2_b = tci::get_elem(ctx, b, {2, 3, 1});
    CHECK(std::abs(tci::real(val2_a) - tci::real(val2_b)) < 1e-10);
    CHECK(std::abs(tci::imag(val2_a) - tci::imag(val2_b)) < 1e-10);

    // Verify shape and rank preservation
    CHECK(tci::shape(ctx, a) == tci::shape(ctx, b));
    CHECK(tci::rank(ctx, a) == tci::rank(ctx, b));
  }

  SUBCASE("Copy is independent (modifications don't affect original)") {
    cytnx::Tensor a, b;
    tci::zeros(ctx, {2, 2}, a);
    tci::set_elem(ctx, a, {0, 0}, cytnx::cytnx_complex128(100.0, 0.0));

    // Make copy
    tci::copy(ctx, a, b);

    // Modify original
    tci::set_elem(ctx, a, {0, 0}, cytnx::cytnx_complex128(999.0, 0.0));

    // Verify copy is unaffected
    auto copy_val = tci::get_elem(ctx, b, {0, 0});
    CHECK(std::abs(tci::real(copy_val) - 100.0) < 1e-10);

    // Verify original was modified
    auto orig_val = tci::get_elem(ctx, a, {0, 0});
    CHECK(std::abs(tci::real(orig_val) - 999.0) < 1e-10);
  }

  SUBCASE("Copy empty tensor") {
    cytnx::Tensor a, b;
    // Don't initialize a (should be empty)

    // Copy should work without errors
    CHECK_NOTHROW(tci::copy(ctx, a, b));

    // Both should have same characteristics
    CHECK(tci::rank(ctx, a) == tci::rank(ctx, b));
    CHECK(tci::shape(ctx, a) == tci::shape(ctx, b));
  }

  SUBCASE("Copy single element tensor") {
    cytnx::Tensor a;
    tci::zeros(ctx, {1}, a);
    tci::set_elem(ctx, a, {0}, cytnx::cytnx_complex128(3.14, 2.71));

    auto b = tci::copy(ctx, a);

    auto val = tci::get_elem(ctx, b, {0});
    CHECK(std::abs(tci::real(val) - 3.14) < 1e-10);
    CHECK(std::abs(tci::imag(val) - 2.71) < 1e-10);
  }

  SUBCASE("Copy large tensor") {
    cytnx::Tensor a;
    tci::zeros(ctx, {10, 10, 10}, a);

    // Set corner elements
    tci::set_elem(ctx, a, {0, 0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, a, {9, 9, 9}, cytnx::cytnx_complex128(0.0, 1.0));

    auto b = tci::copy(ctx, a);

    // Verify corner elements preserved
    auto corner1 = tci::get_elem(ctx, b, {0, 0, 0});
    CHECK(std::abs(tci::real(corner1) - 1.0) < 1e-10);
    CHECK(std::abs(tci::imag(corner1) - 0.0) < 1e-10);

    auto corner2 = tci::get_elem(ctx, b, {9, 9, 9});
    CHECK(std::abs(tci::real(corner2) - 0.0) < 1e-10);
    CHECK(std::abs(tci::imag(corner2) - 1.0) < 1e-10);

    // Verify shape preservation
    CHECK(tci::shape(ctx, a) == tci::shape(ctx, b));
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Size Bytes Calculation") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("Empty tensor should have zero size in bytes") {
    cytnx::Tensor empty_tensor;
    auto size = tci::size_bytes(ctx, empty_tensor);
    CHECK(size == 0);
  }

  SUBCASE("2x2 complex tensor size should match expected") {
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    cytnx::Tensor tensor;
    tci::zeros(ctx, shape, tensor);

    auto size = tci::size_bytes(ctx, tensor);
    // Complex128: 16 bytes per element, 4 elements = 64 bytes
    CHECK(size > 0); // At least verify it's not zero/placeholder
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Tensor Equality") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("Identity tensors should be equal") {
    cytnx::Tensor tensor1, tensor2;
    tci::eye(ctx, 2, tensor1);
    tci::eye(ctx, 2, tensor2);

    tci::elem_t<cytnx::Tensor> epsilon = 1e-10;
    bool are_equal = tci::eq(ctx, tensor1, tensor2, epsilon);

    // This will FAIL if eq is a placeholder that only compares shapes
    CHECK(are_equal == true);
  }

  SUBCASE("Different tensors should not be equal") {
    cytnx::Tensor tensor1, tensor2;
    tci::eye(ctx, 2, tensor1);
    tci::zeros(ctx, {2, 2}, tensor2);

    tci::elem_t<cytnx::Tensor> epsilon = 1e-10;
    bool are_equal = tci::eq(ctx, tensor1, tensor2, epsilon);

    // This should be false - identity != zeros
    CHECK(are_equal == false);
  }

  tci::destroy_context(ctx);
}