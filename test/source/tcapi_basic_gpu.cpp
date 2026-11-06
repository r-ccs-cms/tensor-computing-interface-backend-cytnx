#include <doctest/doctest.h>
#include <tcapi/tcapi.h>

#include <cytnx.hpp>

TEST_CASE("TCAPI Context Management") {
  tcapi::context_handle_t<tcapi::CytnxTensor<cytnx::cytnx_complex128>> ctx;

  SUBCASE("Create and destroy context") {
    CHECK_NOTHROW(tcapi::create_context(ctx));
    CHECK_NOTHROW(tcapi::destroy_context(ctx));
  }

  SUBCASE("Create GPU context") {
    if (cytnx::Device.Ngpus > 0) {
      CHECK_NOTHROW(tcapi::create_context(ctx, 0));  // GPU 0
      CHECK(ctx.get() == cytnx::Device.cuda + 0);
      CHECK_NOTHROW(tcapi::destroy_context(ctx));
    }
  }

  SUBCASE("Create context with different GPU IDs") {
    if (cytnx::Device.Ngpus > 1) {
      CHECK_NOTHROW(tcapi::create_context(ctx, 1));  // GPU 1
      CHECK(ctx.get() == cytnx::Device.cuda + 1);
      CHECK_NOTHROW(tcapi::destroy_context(ctx));
    }
  }

  SUBCASE("GPU context error handling") {
    if (cytnx::Device.Ngpus == 0) {
      // No GPU available - should throw
      CHECK_THROWS_AS(tcapi::create_context(ctx, 0), std::runtime_error);
    } else {
      // GPU available - test out of range
      CHECK_THROWS_AS(tcapi::create_context(ctx, cytnx::Device.Ngpus), std::out_of_range);
      CHECK_THROWS_AS(tcapi::create_context(ctx, -1), std::out_of_range);
    }
  }
}

TEST_CASE("TCAPI GPU Tensor Operations") {
  // Skip test if no GPU available
  if (cytnx::Device.Ngpus == 0) {
    MESSAGE("Skipping GPU tests - no CUDA devices available");
    return;
  }

  tcapi::context_handle_t<tcapi::CytnxTensor<cytnx::cytnx_double>> ctx;
  tcapi::create_context(ctx, 0);  // GPU 0

  SUBCASE("Create tensor on GPU") {
    tcapi::shape_t<tcapi::CytnxTensor<cytnx::cytnx_double>> shape = {2, 3};
    tcapi::CytnxTensor<cytnx::cytnx_double> tensor;

    CHECK_NOTHROW(tcapi::zeros(ctx, shape, tensor));
    CHECK(tcapi::order(ctx, tensor) == 2);
    CHECK(tcapi::shape(ctx, tensor) == shape);
  }

  SUBCASE("GPU tensor element operations") {
    tcapi::shape_t<tcapi::CytnxTensor<cytnx::cytnx_double>> shape = {2, 2};
    tcapi::CytnxTensor<cytnx::cytnx_double> tensor;
    tcapi::zeros(ctx, shape, tensor);

    // Set and get element on GPU
    tcapi::elem_coors_t<tcapi::CytnxTensor<cytnx::cytnx_double>> coord = {0, 0};
    CHECK_NOTHROW(tcapi::set_elem(ctx, tensor, coord, 42.0));

    auto elem = tcapi::get_elem(ctx, tensor, coord);
    CHECK(std::abs(elem - 42.0) < 1e-10);
  }

  tcapi::destroy_context(ctx);
}
