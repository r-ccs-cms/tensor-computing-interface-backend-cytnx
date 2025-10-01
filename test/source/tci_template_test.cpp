#include <doctest/doctest.h>
#include <tci/tci.h>
#include <vector>
#include <complex>

using Ten = cytnx::Tensor;

TEST_CASE("Template Function const Type Issues") {
  using namespace tci;

  // Setup context
  auto ctx = create_context<context_handle_t<Ten>>();

  SUBCASE("to_container with const tensor") {
    // Create a 2x2 tensor
    Ten a = eye<Ten>(ctx, 2);

    // Cast to const to test const type traits
    const Ten& const_a = a;

    // Create container to store results
    std::vector<std::complex<double>> container(4);

    // Define row-major coordinate to index mapping
    auto row_major_map = [](const elem_coors_t<Ten>& coors) -> std::ptrdiff_t {
      return coors[0] * 2 + coors[1];  // 2x2 tensor, row-major
    };

    // This should work with const tensor
    CHECK_NOTHROW(to_container(ctx, const_a, container.begin(), row_major_map));

    // Verify values (identity matrix)
    CHECK(container[0].real() == doctest::Approx(1.0));  // (0,0)
    CHECK(container[1].real() == doctest::Approx(0.0));  // (0,1)
    CHECK(container[2].real() == doctest::Approx(0.0));  // (1,0)
    CHECK(container[3].real() == doctest::Approx(1.0));  // (1,1)
  }

  SUBCASE("random function with const type traits") {
    // Test if random function can handle const context scenarios
    shape_t<Ten> shape = {2, 2};

    Ten a;
    CHECK_NOTHROW(random(ctx, shape, []() { return 0.5; }, a));

    // Verify the tensor was created with correct shape
    auto result_shape = tci::shape(ctx, a);
    CHECK(result_shape[0] == 2);
    CHECK(result_shape[1] == 2);
  }

  SUBCASE("elem_coors_t with const tensor type") {
    // Test type alias resolution with const tensor
    shape_t<Ten> shape = {3};

    Ten a;
    fill(ctx, shape, std::complex<double>(1.0, 0.0), a);
    const Ten& const_a = a;

    // This creates elem_coors_t for const tensor implicitly
    elem_coors_t<Ten> coords = {0};

    // Should work with const tensor
    auto elem = get_elem(ctx, const_a, coords);
    CHECK(std::abs(tci::real(elem) - 1.0) < 1e-10);
  }

  destroy_context(ctx);
}

TEST_CASE("CytnxTensor Element Type Conversion") {
  using namespace tci;

  // Setup context
  auto ctx = create_context<context_handle_t<CytnxTensor<cytnx::cytnx_double>>>();

  SUBCASE("double to complex<double> conversion") {
    // Create a double tensor
    CytnxTensor<cytnx::cytnx_double> real_tensor;
    shape_t<CytnxTensor<cytnx::cytnx_double>> shape = {2, 2};
    zeros(ctx, shape, real_tensor);

    // Set some values
    set_elem(ctx, real_tensor, {0, 0}, 1.5);
    set_elem(ctx, real_tensor, {1, 1}, 2.5);

    // Convert to complex tensor
    auto ctx_cplx = create_context<context_handle_t<CytnxTensor<cytnx::cytnx_complex128>>>();
    CytnxTensor<cytnx::cytnx_complex128> cplx_tensor;

    CHECK_NOTHROW(convert(ctx, real_tensor, ctx_cplx, cplx_tensor));

    // Verify shape is preserved
    auto real_shape = tci::shape(ctx, real_tensor);
    auto cplx_shape = tci::shape(ctx_cplx, cplx_tensor);
    CHECK(real_shape == cplx_shape);

    // Verify values are preserved
    auto val00 = get_elem(ctx_cplx, cplx_tensor, {0, 0});
    auto val11 = get_elem(ctx_cplx, cplx_tensor, {1, 1});
    CHECK(val00.real() == doctest::Approx(1.5));
    CHECK(val00.imag() == doctest::Approx(0.0));
    CHECK(val11.real() == doctest::Approx(2.5));
    CHECK(val11.imag() == doctest::Approx(0.0));

    destroy_context(ctx_cplx);
  }

  SUBCASE("float to double conversion") {
    // Create a float tensor
    CytnxTensor<cytnx::cytnx_float> float_tensor;
    shape_t<CytnxTensor<cytnx::cytnx_float>> shape = {3, 3};
    eye(ctx, 3, float_tensor);

    // Convert to double tensor
    auto ctx_double = create_context<context_handle_t<CytnxTensor<cytnx::cytnx_double>>>();
    CytnxTensor<cytnx::cytnx_double> double_tensor;

    CHECK_NOTHROW(convert(ctx, float_tensor, ctx_double, double_tensor));

    // Verify identity matrix structure
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        auto val = get_elem(ctx_double, double_tensor, {i, j});
        double expected = (i == j) ? 1.0 : 0.0;
        CHECK(val == doctest::Approx(expected));
      }
    }

    destroy_context(ctx_double);
  }

  SUBCASE("complex<float> to complex<double> conversion") {
    // Create a complex float tensor
    CytnxTensor<cytnx::cytnx_complex64> cplx_float_tensor;
    shape_t<CytnxTensor<cytnx::cytnx_complex64>> shape = {2, 3};
    zeros(ctx, shape, cplx_float_tensor);

    // Set complex values
    set_elem(ctx, cplx_float_tensor, {0, 0}, cytnx::cytnx_complex64(1.0f, 2.0f));
    set_elem(ctx, cplx_float_tensor, {1, 2}, cytnx::cytnx_complex64(3.0f, 4.0f));

    // Convert to complex double tensor
    auto ctx_cplx_double = create_context<context_handle_t<CytnxTensor<cytnx::cytnx_complex128>>>();
    CytnxTensor<cytnx::cytnx_complex128> cplx_double_tensor;

    CHECK_NOTHROW(convert(ctx, cplx_float_tensor, ctx_cplx_double, cplx_double_tensor));

    // Verify complex values are preserved
    auto val00 = get_elem(ctx_cplx_double, cplx_double_tensor, {0, 0});
    auto val12 = get_elem(ctx_cplx_double, cplx_double_tensor, {1, 2});
    CHECK(val00.real() == doctest::Approx(1.0));
    CHECK(val00.imag() == doctest::Approx(2.0));
    CHECK(val12.real() == doctest::Approx(3.0));
    CHECK(val12.imag() == doctest::Approx(4.0));

    destroy_context(ctx_cplx_double);
  }

  SUBCASE("same type conversion (should work as copy)") {
    // Create a double tensor
    CytnxTensor<cytnx::cytnx_double> tensor1;
    shape_t<CytnxTensor<cytnx::cytnx_double>> shape = {2, 2};
    fill(ctx, shape, 3.14, tensor1);

    // Convert to same type
    auto ctx2 = create_context<context_handle_t<CytnxTensor<cytnx::cytnx_double>>>();
    CytnxTensor<cytnx::cytnx_double> tensor2;

    CHECK_NOTHROW(convert(ctx, tensor1, ctx2, tensor2));

    // Verify copy
    auto val = get_elem(ctx2, tensor2, {0, 0});
    CHECK(val == doctest::Approx(3.14));

    destroy_context(ctx2);
  }

  destroy_context(ctx);
}