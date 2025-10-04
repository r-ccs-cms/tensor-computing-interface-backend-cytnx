#include <doctest/doctest.h>
#include <tci/tci.h>

#include <cytnx.hpp>
#include <functional>

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
    CHECK(tci::rank(ctx, tensor) == 2);
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

TEST_CASE("TCI Tensor Creation") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  SUBCASE("Create zero tensor") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 3};
    tci::CytnxTensor<cytnx::cytnx_complex128> tensor;

    CHECK_NOTHROW(tci::zeros(ctx, shape, tensor));
    CHECK(tci::rank(ctx, tensor) == 2);

    auto result_shape = tci::shape(ctx, tensor);
    CHECK(result_shape.size() == 2);
    CHECK(result_shape[0] == 2);
    CHECK(result_shape[1] == 3);
  }

  SUBCASE("Create identity matrix") {
    tci::CytnxTensor<cytnx::cytnx_complex128> identity;

    CHECK_NOTHROW(tci::eye(ctx, 3, identity));
    CHECK(tci::rank(ctx, identity) == 2);

    auto result_shape = tci::shape(ctx, identity);
    CHECK(result_shape.size() == 2);
    CHECK(result_shape[0] == 3);
    CHECK(result_shape[1] == 3);
  }

  SUBCASE("Create random tensor (in-place)") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 3};
    tci::CytnxTensor<cytnx::cytnx_complex128> tensor;
    std::size_t counter = 0;

    // gen() should return elem_t<TenT> (cytnx_complex128 in this case)
    auto gen = [&]() -> cytnx::cytnx_complex128 {
      double val = static_cast<double>(counter++);
      return cytnx::cytnx_complex128(val, val + 0.5);
    };

    CHECK_NOTHROW(tci::random(ctx, shape, gen, tensor));
    CHECK(counter == 6);  // 6 complex elements = 6 gen() calls

    auto result_shape = tci::shape(ctx, tensor);
    CHECK(result_shape == shape);

    auto elem_00 = tci::get_elem(ctx, tensor, {0, 0});
    CHECK(std::abs(tci::real(elem_00) - 0.0) < 1e-10);
    CHECK(std::abs(tci::imag(elem_00) - 0.5) < 1e-10);

    auto elem_01 = tci::get_elem(ctx, tensor, {0, 1});
    CHECK(std::abs(tci::real(elem_01) - 1.0) < 1e-10);
    CHECK(std::abs(tci::imag(elem_01) - 1.5) < 1e-10);

    auto elem_12 = tci::get_elem(ctx, tensor, {1, 2});
    CHECK(std::abs(tci::real(elem_12) - 5.0) < 1e-10);
    CHECK(std::abs(tci::imag(elem_12) - 5.5) < 1e-10);
  }

  SUBCASE("Create random tensor (out-of-place)") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 2};
    std::size_t counter = 0;

    // gen() should return elem_t<TenT> (cytnx_complex128 in this case)
    auto gen = [&]() -> cytnx::cytnx_complex128 {
      double val = static_cast<double>(counter++);
      return cytnx::cytnx_complex128(val, val + 0.5);
    };

    tci::CytnxTensor<cytnx::cytnx_complex128> tensor;
    CHECK_NOTHROW(tensor = tci::random<tci::CytnxTensor<cytnx::cytnx_complex128>>(ctx, shape, gen));
    CHECK(counter == 4);  // 4 complex elements = 4 gen() calls

    auto result_shape = tci::shape(ctx, tensor);
    CHECK(result_shape == shape);

    auto elem_11 = tci::get_elem(ctx, tensor, {1, 1});
    CHECK(std::abs(tci::real(elem_11) - 3.0) < 1e-10);
    CHECK(std::abs(tci::imag(elem_11) - 3.5) < 1e-10);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Tensor Properties") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 3, 4};
  tci::CytnxTensor<cytnx::cytnx_complex128> tensor;
  tci::zeros(ctx, shape, tensor);

  SUBCASE("Tensor rank") { CHECK(tci::rank(ctx, tensor) == 3); }

  SUBCASE("Tensor shape") {
    auto result_shape = tci::shape(ctx, tensor);
    CHECK(result_shape.size() == 3);
    CHECK(result_shape[0] == 2);
    CHECK(result_shape[1] == 3);
    CHECK(result_shape[2] == 4);
  }

  SUBCASE("Tensor size") { CHECK(tci::size(ctx, tensor) == 24); }

  SUBCASE("Tensor size in bytes") {
    auto bytes = tci::size_bytes(ctx, tensor);
    CHECK(bytes > 0);
    // Expected: 24 elements * 16 bytes per complex128 = 384 bytes
    CHECK(bytes == 384);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Tensor Operations") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  SUBCASE("Copy tensor") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 2};
    tci::CytnxTensor<cytnx::cytnx_complex128> original, copy;

    tci::zeros(ctx, shape, original);
    CHECK_NOTHROW(tci::copy(ctx, original, copy));

    CHECK(tci::rank(ctx, copy) == tci::rank(ctx, original));
    CHECK(tci::shape(ctx, copy) == tci::shape(ctx, original));
  }

  SUBCASE("Element access") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 2};
    tci::CytnxTensor<cytnx::cytnx_complex128> tensor;
    tci::zeros(ctx, shape, tensor);

    // Set element at (0,0) to (1.0, 0.0)
    tci::elem_coors_t<tci::CytnxTensor<cytnx::cytnx_complex128>> coord = {0, 0};
    tci::set_elem(ctx, tensor, coord, cytnx::cytnx_complex128(1.0, 0.0));

    // Get element back - should be exactly what we set
    auto elem = tci::get_elem(ctx, tensor, coord);
    CHECK(std::abs(tci::real(elem) - 1.0) < 1e-10);
    CHECK(std::abs(tci::imag(elem) - 0.0) < 1e-10);
  }

  SUBCASE("Norm calculation") {
    // Test with 3x3 identity matrix
    tci::CytnxTensor<cytnx::cytnx_complex128> identity;
    tci::eye(ctx, 3, identity);

    auto norm_val = tci::norm(ctx, identity);
    // Frobenius norm of 3x3 identity should be sqrt(3)
    CHECK(std::abs(tci::real(norm_val) - std::sqrt(3.0)) < 1e-10);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI assign_from_container") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  SUBCASE("Create tensor from std::vector with row-major indexing") {
    // Create a 2x3 tensor from container
    std::vector<cytnx::cytnx_complex128> container
        = {cytnx::cytnx_complex128(1.0, 0.0), cytnx::cytnx_complex128(2.0, 0.0),
           cytnx::cytnx_complex128(3.0, 0.0), cytnx::cytnx_complex128(4.0, 0.0),
           cytnx::cytnx_complex128(5.0, 0.0), cytnx::cytnx_complex128(6.0, 0.0)};

    std::function<std::ptrdiff_t(
        const tci::elem_coors_t<tci::CytnxTensor<cytnx::cytnx_complex128>>&)>
        coors2idx = [](const tci::elem_coors_t<tci::CytnxTensor<cytnx::cytnx_complex128>>& coors)
        -> std::ptrdiff_t {
      return coors[0] * 3 + coors[1];  // row-major for 2x3 matrix
    };

    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 3};
    tci::CytnxTensor<cytnx::cytnx_complex128> tensor;

    CHECK_NOTHROW(
        tci::assign_from_container(ctx, shape, container.begin(), std::move(coors2idx), tensor));

    // Verify tensor properties
    CHECK(tci::rank(ctx, tensor) == 2);
    auto result_shape = tci::shape(ctx, tensor);
    CHECK(result_shape[0] == 2);
    CHECK(result_shape[1] == 3);

    // Verify element values match container
    auto elem_00 = tci::get_elem(ctx, tensor, {0, 0});
    CHECK(std::abs(tci::real(elem_00) - 1.0) < 1e-10);

    auto elem_01 = tci::get_elem(ctx, tensor, {0, 1});
    CHECK(std::abs(tci::real(elem_01) - 2.0) < 1e-10);

    auto elem_10 = tci::get_elem(ctx, tensor, {1, 0});
    CHECK(std::abs(tci::real(elem_10) - 4.0) < 1e-10);

    auto elem_12 = tci::get_elem(ctx, tensor, {1, 2});
    CHECK(std::abs(tci::real(elem_12) - 6.0) < 1e-10);
  }

  SUBCASE("Create tensor from std::vector with custom indexing") {
    // Create a 2x2 tensor with column-major indexing
    std::vector<cytnx::cytnx_complex128> container = {
        cytnx::cytnx_complex128(1.0, 0.0), cytnx::cytnx_complex128(3.0, 0.0),  // column 0
        cytnx::cytnx_complex128(2.0, 0.0), cytnx::cytnx_complex128(4.0, 0.0)   // column 1
    };

    std::function<std::ptrdiff_t(
        const tci::elem_coors_t<tci::CytnxTensor<cytnx::cytnx_complex128>>&)>
        coors2idx = [](const tci::elem_coors_t<tci::CytnxTensor<cytnx::cytnx_complex128>>& coors)
        -> std::ptrdiff_t {
      return coors[1] * 2 + coors[0];  // column-major for 2x2 matrix
    };

    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 2};
    tci::CytnxTensor<cytnx::cytnx_complex128> tensor;

    CHECK_NOTHROW(
        tci::assign_from_container(ctx, shape, container.begin(), std::move(coors2idx), tensor));

    // Verify element values with column-major layout
    auto elem_00 = tci::get_elem(ctx, tensor, {0, 0});
    CHECK(std::abs(tci::real(elem_00) - 1.0) < 1e-10);

    auto elem_01 = tci::get_elem(ctx, tensor, {0, 1});
    CHECK(std::abs(tci::real(elem_01) - 2.0) < 1e-10);

    auto elem_10 = tci::get_elem(ctx, tensor, {1, 0});
    CHECK(std::abs(tci::real(elem_10) - 3.0) < 1e-10);

    auto elem_11 = tci::get_elem(ctx, tensor, {1, 1});
    CHECK(std::abs(tci::real(elem_11) - 4.0) < 1e-10);
  }

  SUBCASE("Out-of-place version") {
    std::vector<cytnx::cytnx_complex128> container
        = {cytnx::cytnx_complex128(7.0, 0.0), cytnx::cytnx_complex128(8.0, 0.0),
           cytnx::cytnx_complex128(9.0, 0.0)};

    std::function<std::ptrdiff_t(
        const tci::elem_coors_t<tci::CytnxTensor<cytnx::cytnx_complex128>>&)>
        coors2idx = [](const tci::elem_coors_t<tci::CytnxTensor<cytnx::cytnx_complex128>>& coors)
        -> std::ptrdiff_t {
      return coors[0];  // simple linear indexing for 1D tensor
    };

    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {3};

    tci::CytnxTensor<cytnx::cytnx_complex128> tensor;
    CHECK_NOTHROW(tensor = tci::assign_from_container<tci::CytnxTensor<cytnx::cytnx_complex128>>(
                      ctx, shape, container.begin(), std::move(coors2idx)));

    // Verify elements
    auto elem_0 = tci::get_elem(ctx, tensor, {0});
    CHECK(std::abs(tci::real(elem_0) - 7.0) < 1e-10);

    auto elem_2 = tci::get_elem(ctx, tensor, {2});
    CHECK(std::abs(tci::real(elem_2) - 9.0) < 1e-10);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Tensor Manipulation") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  SUBCASE("Reshape operations") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> original_shape = {2, 3, 4};
    tci::CytnxTensor<cytnx::cytnx_complex128> tensor;
    tci::zeros(ctx, original_shape, tensor);
    tci::fill(ctx, original_shape, cytnx::cytnx_complex128(1.0, 0.0), tensor);

    // Test in-place reshape
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> new_shape = {6, 4};
    CHECK_NOTHROW(tci::reshape(ctx, tensor, new_shape));

    auto result_shape = tci::shape(ctx, tensor);
    CHECK(result_shape == new_shape);
    CHECK(tci::size(ctx, tensor) == 24);  // Same total size
  }

  SUBCASE("Transpose operations") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 3, 4};
    tci::CytnxTensor<cytnx::cytnx_complex128> tensor, transposed;
    tci::zeros(ctx, shape, tensor);
    tci::fill(ctx, shape, cytnx::cytnx_complex128(1.0, 0.0), tensor);

    // Test out-of-place transpose
    tci::List<tci::bond_idx_t<tci::CytnxTensor<cytnx::cytnx_complex128>>> new_order
        = {2, 0, 1};  // 4,2,3
    CHECK_NOTHROW(tci::transpose(ctx, tensor, new_order, transposed));

    auto result_shape = tci::shape(ctx, transposed);
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> expected_shape = {4, 2, 3};
    CHECK(result_shape == expected_shape);
  }

  SUBCASE("Complex operations") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 2};
    tci::CytnxTensor<cytnx::cytnx_complex128> tensor;
    tci::fill(ctx, shape, cytnx::cytnx_complex128(2.0, 3.0), tensor);

    // Test complex conjugate
    tci::CytnxTensor<cytnx::cytnx_complex128> conjugated;
    CHECK_NOTHROW(tci::cplx_conj(ctx, tensor, conjugated));

    // Test real/imaginary part extraction
    tci::real_ten_t<tci::CytnxTensor<cytnx::cytnx_complex128>> real_part, imag_part;
    CHECK_NOTHROW(tci::real(ctx, tensor, real_part));
    CHECK_NOTHROW(tci::imag(ctx, tensor, imag_part));

    // Verify real part
    auto real_elem = tci::get_elem(ctx, real_part, {0, 0});
    CHECK(std::abs(tci::real(real_elem) - 2.0) < 1e-10);

    // Verify imaginary part
    auto imag_elem = tci::get_elem(ctx, imag_part, {0, 0});
    CHECK(std::abs(tci::real(imag_elem) - 3.0) < 1e-10);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Advanced Tensor Manipulation") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  SUBCASE("Concatenate 2D tensors") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 3};
    tci::CytnxTensor<cytnx::cytnx_complex128> tensor1, tensor2, result;

    tci::fill(ctx, shape, cytnx::cytnx_complex128(1.0, 0.0), tensor1);
    tci::fill(ctx, shape, cytnx::cytnx_complex128(2.0, 0.0), tensor2);

    // Test vertical concatenation (bond_idx = 0)
    std::vector<tci::CytnxTensor<cytnx::cytnx_complex128>> tensors = {tensor1, tensor2};
    CHECK_NOTHROW(tci::concatenate(ctx, tensors, 0, result));

    auto result_shape = tci::shape(ctx, result);
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> expected_vertical_shape = {4, 3};
    CHECK(result_shape == expected_vertical_shape);  // 2+2 = 4 rows

    // Test horizontal concatenation (bond_idx = 1)
    CHECK_NOTHROW(tci::concatenate(ctx, tensors, 1, result));
    result_shape = tci::shape(ctx, result);
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> expected_horizontal_shape = {2, 6};
    CHECK(result_shape == expected_horizontal_shape);  // 3+3 = 6 columns
  }

  SUBCASE("Concatenate multiple tensors with value verification") {
    // Based on documentation example: {2,3,4} + {2,1,4} + {2,2,4} = {2,6,4}
    tci::CytnxTensor<cytnx::cytnx_complex128> a, b, c, d;

    // Create tensors with specific shapes as in the documentation
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape_a = {2, 3, 4};
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape_b = {2, 1, 4};
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape_c = {2, 2, 4};

    tci::fill(ctx, shape_a, cytnx::cytnx_complex128(1.0, 0.0), a);
    tci::fill(ctx, shape_b, cytnx::cytnx_complex128(2.0, 0.0), b);
    tci::fill(ctx, shape_c, cytnx::cytnx_complex128(3.0, 0.0), c);

    // Concatenate along bond 1 (middle dimension)
    std::vector<tci::CytnxTensor<cytnx::cytnx_complex128>> tensors = {a, b, c};
    CHECK_NOTHROW(tci::concatenate(ctx, tensors, 1, d));

    // Verify final shape
    auto result_shape = tci::shape(ctx, d);
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> expected_shape = {2, 6, 4};  // 3+1+2=6
    CHECK(result_shape == expected_shape);

    // Verify element values at specific positions
    auto el1 = tci::get_elem(ctx, b, {0, 0, 0});  // Should be 2.0
    auto el2 = tci::get_elem(ctx, d, {0, 3, 0});  // b starts at position 3 in concatenated tensor
    CHECK(el1 == el2);

    auto el3 = tci::get_elem(ctx, c, {0, 0, 0});  // Should be 3.0
    auto el4 = tci::get_elem(ctx, d, {0, 4, 0});  // c starts at position 4 in concatenated tensor
    CHECK(el3 == el4);
  }

  SUBCASE("Concatenate error cases") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape_3d = {2, 3, 4};
    tci::CytnxTensor<cytnx::cytnx_complex128> tensor_3d;
    tci::fill(ctx, shape_3d, cytnx::cytnx_complex128(1.0, 0.0), tensor_3d);

    std::vector<tci::CytnxTensor<cytnx::cytnx_complex128>> tensors_3d = {tensor_3d};
    tci::CytnxTensor<cytnx::cytnx_complex128> result;

    // Test out of bounds dimension index
    CHECK_THROWS_AS(tci::concatenate(ctx, tensors_3d, 3, result), std::invalid_argument);
    CHECK_THROWS_AS(tci::concatenate(ctx, tensors_3d, 100, result), std::invalid_argument);

    // Test empty tensor list
    std::vector<tci::CytnxTensor<cytnx::cytnx_complex128>> empty_tensors;
    CHECK_THROWS_AS(tci::concatenate(ctx, empty_tensors, 0, result), std::invalid_argument);

    // Test incompatible shapes
    tci::CytnxTensor<cytnx::cytnx_complex128> tensor_incompatible;
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> incompatible_shape
        = {3, 3, 4};  // Different first dimension
    tci::fill(ctx, incompatible_shape, cytnx::cytnx_complex128(1.0, 0.0), tensor_incompatible);
    std::vector<tci::CytnxTensor<cytnx::cytnx_complex128>> incompatible_tensors
        = {tensor_3d, tensor_incompatible};
    CHECK_THROWS_AS(tci::concatenate(ctx, incompatible_tensors, 1, result), std::invalid_argument);
  }

  SUBCASE("Extract sub-tensor") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {4, 6};
    tci::CytnxTensor<cytnx::cytnx_complex128> tensor, sub_result;
    tci::zeros(ctx, shape, tensor);
    tci::fill(ctx, shape, cytnx::cytnx_complex128(1.0, 0.0), tensor);

    // Set some specific values for verification
    tci::set_elem(ctx, tensor, {1, 2}, cytnx::cytnx_complex128(5.0, 0.0));
    tci::set_elem(ctx, tensor, {2, 3}, cytnx::cytnx_complex128(7.0, 0.0));

    // Extract sub-tensor [1:3, 2:5)
    tci::List<tci::Pair<tci::elem_coor_t<tci::CytnxTensor<cytnx::cytnx_complex128>>,
                        tci::elem_coor_t<tci::CytnxTensor<cytnx::cytnx_complex128>>>>
        coor_pairs = {{1, 3}, {2, 5}};
    CHECK_NOTHROW(tci::extract_sub(ctx, tensor, coor_pairs, sub_result));

    auto sub_shape = tci::shape(ctx, sub_result);
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> expected_sub_shape = {2, 3};
    CHECK(sub_shape == expected_sub_shape);  // (3-1) x (5-2)

    // Verify extracted value
    auto extracted_elem = tci::get_elem(ctx, sub_result, {0, 0});  // Should be (1,2) from original
    CHECK(std::abs(tci::real(extracted_elem) - 5.0) < 1e-10);
  }

  SUBCASE("Replace sub-tensor") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> main_shape = {4, 4};
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> sub_shape = {2, 2};
    tci::CytnxTensor<cytnx::cytnx_complex128> main_tensor, sub_tensor, result;

    tci::zeros(ctx, main_shape, main_tensor);
    tci::fill(ctx, sub_shape, cytnx::cytnx_complex128(3.0, 0.0), sub_tensor);

    // Replace starting at position (1, 1)
    tci::elem_coors_t<tci::CytnxTensor<cytnx::cytnx_complex128>> begin_pt = {1, 1};
    CHECK_NOTHROW(tci::replace_sub(ctx, main_tensor, sub_tensor, begin_pt, result));

    // Verify replacement
    auto replaced_elem = tci::get_elem(ctx, result, {1, 1});
    CHECK(std::abs(tci::real(replaced_elem) - 3.0) < 1e-10);

    auto replaced_elem2 = tci::get_elem(ctx, result, {2, 2});
    CHECK(std::abs(tci::real(replaced_elem2) - 3.0) < 1e-10);

    // Verify non-replaced area
    auto non_replaced = tci::get_elem(ctx, result, {0, 0});
    CHECK(std::abs(tci::real(non_replaced) - 0.0) < 1e-10);
  }

  SUBCASE("Expand tensor dimensions") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> original_shape = {2, 3};
    tci::CytnxTensor<cytnx::cytnx_complex128> tensor, expanded;
    tci::zeros(ctx, original_shape, tensor);
    tci::fill(ctx, original_shape, cytnx::cytnx_complex128(1.0, 0.0), tensor);

    // Expand bond 0 by 1, bond 1 by 2
    tci::Map<tci::bond_idx_t<tci::CytnxTensor<cytnx::cytnx_complex128>>,
             tci::bond_dim_t<tci::CytnxTensor<cytnx::cytnx_complex128>>>
        increment_map = {{0, 1}, {1, 2}};
    CHECK_NOTHROW(tci::expand(ctx, tensor, increment_map, expanded));

    auto expanded_shape = tci::shape(ctx, expanded);
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> expected_expanded_shape = {3, 5};
    CHECK(expanded_shape == expected_expanded_shape);  // 2+1, 3+2

    // Verify original data is preserved (at beginning)
    auto elem = tci::get_elem(ctx, expanded, {0, 0});
    CHECK(std::abs(tci::real(elem) - 1.0) < 1e-10);

    // Verify expanded area is zero
    auto zero_elem = tci::get_elem(ctx, expanded, {2, 0});  // Expanded row
    CHECK(std::abs(tci::real(zero_elem) - 0.0) < 1e-10);
  }

  SUBCASE("For each with coordinates") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 3};
    tci::CytnxTensor<cytnx::cytnx_complex128> tensor;
    tci::zeros(ctx, shape, tensor);

    // Set values based on coordinates using for_each_with_coors
    std::function<void(tci::elem_t<tci::CytnxTensor<cytnx::cytnx_complex128>>&,
                       const tci::elem_coors_t<tci::CytnxTensor<cytnx::cytnx_complex128>>&)>
        set_coords
        = [](tci::elem_t<tci::CytnxTensor<cytnx::cytnx_complex128>>& elem,
             const tci::elem_coors_t<tci::CytnxTensor<cytnx::cytnx_complex128>>& coords) {
            elem = cytnx::cytnx_complex128(coords[0] * 10 + coords[1], 0.0);
          };

    CHECK_NOTHROW(tci::for_each_with_coors(ctx, tensor, std::move(set_coords)));

    // Verify values
    auto elem_01 = tci::get_elem(ctx, tensor, {0, 1});
    CHECK(std::abs(tci::real(elem_01) - 1.0) < 1e-10);  // 0*10 + 1 = 1

    auto elem_12 = tci::get_elem(ctx, tensor, {1, 2});
    CHECK(std::abs(tci::real(elem_12) - 12.0) < 1e-10);  // 1*10 + 2 = 12
  }

  tci::destroy_context(ctx);
}
