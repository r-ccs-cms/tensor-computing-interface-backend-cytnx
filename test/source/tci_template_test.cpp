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
    CHECK(std::abs(elem.real() - 1.0) < 1e-10);
  }

  destroy_context(ctx);
}