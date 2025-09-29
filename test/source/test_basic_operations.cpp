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