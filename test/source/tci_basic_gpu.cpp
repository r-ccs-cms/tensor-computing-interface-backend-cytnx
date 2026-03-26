#include <doctest/doctest.h>
#include <tci/tci.h>

#include <cytnx.hpp>

TEST_CASE("TCI Context Management") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;

  SUBCASE("Create and destroy context") {
    CHECK_NOTHROW(tci::create_context(ctx));
    CHECK_NOTHROW(tci::destroy_context(ctx));
  }

  SUBCASE("Create GPU context") {
    if (cytnx::Device.Ngpus > 0) {
      CHECK_NOTHROW(tci::create_context(ctx, 0));  // GPU 0
      CHECK(ctx.get() == cytnx::Device.cuda + 0);
      CHECK_NOTHROW(tci::destroy_context(ctx));
    }
  }

  SUBCASE("Create context with different GPU IDs") {
    if (cytnx::Device.Ngpus > 1) {
      CHECK_NOTHROW(tci::create_context(ctx, 1));  // GPU 1
      CHECK(ctx.get() == cytnx::Device.cuda + 1);
      CHECK_NOTHROW(tci::destroy_context(ctx));
    }
  }

  SUBCASE("GPU context error handling") {
    if (cytnx::Device.Ngpus == 0) {
      // No GPU available - should throw
      CHECK_THROWS_AS(tci::create_context(ctx, 0), std::runtime_error);
    } else {
      // GPU available - test out of range
      CHECK_THROWS_AS(tci::create_context(ctx, cytnx::Device.Ngpus), std::out_of_range);
      CHECK_THROWS_AS(tci::create_context(ctx, -1), std::out_of_range);
    }
  }
}

TEST_CASE("TCI GPU Tensor Operations") {
  // Skip test if no GPU available
  if (cytnx::Device.Ngpus == 0) {
    MESSAGE("Skipping GPU tests - no CUDA devices available");
    return;
  }

  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_double>> ctx;
  tci::create_context(ctx, 0);  // GPU 0

  SUBCASE("Create tensor on GPU") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_double>> shape = {2, 3};
    tci::CytnxTensor<cytnx::cytnx_double> tensor;

    CHECK_NOTHROW(tci::zeros(ctx, shape, tensor));
    CHECK(tci::order(ctx, tensor) == 2);
    CHECK(tci::shape(ctx, tensor) == shape);
  }

  SUBCASE("GPU tensor element operations") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_double>> shape = {2, 2};
    tci::CytnxTensor<cytnx::cytnx_double> tensor;
    tci::zeros(ctx, shape, tensor);

    // Set and get element on GPU
    tci::elem_coors_t<tci::CytnxTensor<cytnx::cytnx_double>> coord = {0, 0};
    CHECK_NOTHROW(tci::set_elem(ctx, tensor, coord, 42.0));

    auto elem = tci::get_elem(ctx, tensor, coord);
    CHECK(std::abs(elem - 42.0) < 1e-10);
  }

  tci::destroy_context(ctx);
}
