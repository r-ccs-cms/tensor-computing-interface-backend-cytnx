#include <doctest/doctest.h>
#include <tci/tci.h>
#include <cmath>
#include <cytnx.hpp>

TEST_CASE("tci::to_cplx API functionality") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("Convert real tensor to complex (out-of-place version)") {
    // Create a real tensor with some values
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    cytnx::Tensor real_tensor;
    tci::zeros(ctx, shape, real_tensor);

    // Set some real values
    tci::elem_coors_t<cytnx::Tensor> coord00 = {0, 0};
    tci::elem_coors_t<cytnx::Tensor> coord01 = {0, 1};
    tci::elem_coors_t<cytnx::Tensor> coord10 = {1, 0};
    tci::elem_coors_t<cytnx::Tensor> coord11 = {1, 1};

    tci::set_elem(ctx, real_tensor, coord00, cytnx::cytnx_complex128(1.5, 0.0));
    tci::set_elem(ctx, real_tensor, coord01, cytnx::cytnx_complex128(2.5, 0.0));
    tci::set_elem(ctx, real_tensor, coord10, cytnx::cytnx_complex128(3.5, 0.0));
    tci::set_elem(ctx, real_tensor, coord11, cytnx::cytnx_complex128(4.5, 0.0));

    // Convert to complex using return version
    auto complex_tensor = tci::to_cplx(ctx, real_tensor);

    // Verify the conversion preserved real parts and has zero imaginary parts
    auto elem00 = tci::get_elem(ctx, complex_tensor, coord00);
    auto elem01 = tci::get_elem(ctx, complex_tensor, coord01);
    auto elem10 = tci::get_elem(ctx, complex_tensor, coord10);
    auto elem11 = tci::get_elem(ctx, complex_tensor, coord11);

    // This will FAIL if to_cplx is not properly implemented
    CHECK(std::abs(tci::real(elem00) - 1.5) < 1e-10);
    CHECK(std::abs(tci::imag(elem00) - 0.0) < 1e-10);
    CHECK(std::abs(tci::real(elem01) - 2.5) < 1e-10);
    CHECK(std::abs(tci::imag(elem01) - 0.0) < 1e-10);
    CHECK(std::abs(tci::real(elem10) - 3.5) < 1e-10);
    CHECK(std::abs(tci::imag(elem10) - 0.0) < 1e-10);
    CHECK(std::abs(tci::real(elem11) - 4.5) < 1e-10);
    CHECK(std::abs(tci::imag(elem11) - 0.0) < 1e-10);
  }

  SUBCASE("Convert real tensor to complex (in-place version)") {
    // Create a real tensor with some values
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    cytnx::Tensor real_tensor;
    tci::zeros(ctx, shape, real_tensor);

    // Set some real values
    tci::elem_coors_t<cytnx::Tensor> coord00 = {0, 0};
    tci::elem_coors_t<cytnx::Tensor> coord11 = {1, 1};

    tci::set_elem(ctx, real_tensor, coord00, cytnx::cytnx_complex128(7.25, 0.0));
    tci::set_elem(ctx, real_tensor, coord11, cytnx::cytnx_complex128(8.75, 0.0));

    // Convert to complex using output parameter version
    tci::cplx_ten_t<cytnx::Tensor> complex_output;
    tci::to_cplx(ctx, real_tensor, complex_output);

    // Verify the conversion preserved values correctly
    auto elem00 = tci::get_elem(ctx, complex_output, coord00);
    auto elem11 = tci::get_elem(ctx, complex_output, coord11);

    // This will FAIL if to_cplx output version is not properly implemented
    CHECK(std::abs(tci::real(elem00) - 7.25) < 1e-10);
    CHECK(std::abs(tci::imag(elem00) - 0.0) < 1e-10);
    CHECK(std::abs(tci::real(elem11) - 8.75) < 1e-10);
    CHECK(std::abs(tci::imag(elem11) - 0.0) < 1e-10);
  }

  SUBCASE("Convert already complex tensor to complex should preserve values") {
    // Create a complex tensor with both real and imaginary parts
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    cytnx::Tensor complex_tensor;
    tci::zeros(ctx, shape, complex_tensor);

    // Set complex values
    tci::elem_coors_t<cytnx::Tensor> coord00 = {0, 0};
    tci::elem_coors_t<cytnx::Tensor> coord11 = {1, 1};

    tci::set_elem(ctx, complex_tensor, coord00, cytnx::cytnx_complex128(3.14, 2.71));
    tci::set_elem(ctx, complex_tensor, coord11, cytnx::cytnx_complex128(-1.41, 1.73));

    // Convert complex to complex
    auto result_tensor = tci::to_cplx(ctx, complex_tensor);

    // Verify values are preserved
    auto elem00 = tci::get_elem(ctx, result_tensor, coord00);
    auto elem11 = tci::get_elem(ctx, result_tensor, coord11);

    // This will FAIL if to_cplx doesn't handle complex input correctly
    CHECK(std::abs(tci::real(elem00) - 3.14) < 1e-10);
    CHECK(std::abs(tci::imag(elem00) - 2.71) < 1e-10);
    CHECK(std::abs(tci::real(elem11) - (-1.41)) < 1e-10);
    CHECK(std::abs(tci::imag(elem11) - 1.73) < 1e-10);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("tci::shrink API functionality") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("Shrink 3x3 tensor to 2x2 (in-place version)") {
    // Create a 3x3 tensor with known values
    tci::shape_t<cytnx::Tensor> shape = {3, 3};
    cytnx::Tensor tensor;
    tci::zeros(ctx, shape, tensor);

    // Fill with distinct values for verification
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        tci::elem_coors_t<cytnx::Tensor> coord = {static_cast<tci::elem_coor_t<cytnx::Tensor>>(i),
                                                   static_cast<tci::elem_coor_t<cytnx::Tensor>>(j)};
        double value = i * 3 + j + 1;  // Values 1-9
        tci::set_elem(ctx, tensor, coord, cytnx::cytnx_complex128(value, 0.0));
      }
    }

    // Shrink to extract 2x2 sub-tensor from [0:2, 0:2]
    tci::bond_idx_elem_coor_pair_map<cytnx::Tensor> shrink_map;
    shrink_map[0] = std::make_pair(0, 1);  // Bond 0: from 0 to 1 (2 elements)
    shrink_map[1] = std::make_pair(0, 1);  // Bond 1: from 0 to 1 (2 elements)

    tci::shrink(ctx, tensor, shrink_map);

    // Verify the result is 2x2 with correct values
    auto result_shape = tci::shape(ctx, tensor);
    CHECK(result_shape.size() == 2);
    CHECK(result_shape[0] == 2);
    CHECK(result_shape[1] == 2);

    // Check values: should be top-left 2x2 of original
    tci::elem_coors_t<cytnx::Tensor> coord00 = {0, 0};
    tci::elem_coors_t<cytnx::Tensor> coord01 = {0, 1};
    tci::elem_coors_t<cytnx::Tensor> coord10 = {1, 0};
    tci::elem_coors_t<cytnx::Tensor> coord11 = {1, 1};

    auto elem00 = tci::get_elem(ctx, tensor, coord00);
    auto elem01 = tci::get_elem(ctx, tensor, coord01);
    auto elem10 = tci::get_elem(ctx, tensor, coord10);
    auto elem11 = tci::get_elem(ctx, tensor, coord11);

    // This will FAIL if shrink is not properly implemented
    CHECK(std::abs(tci::real(elem00) - 1.0) < 1e-10);  // Was (0,0) = 1
    CHECK(std::abs(tci::real(elem01) - 2.0) < 1e-10);  // Was (0,1) = 2
    CHECK(std::abs(tci::real(elem10) - 4.0) < 1e-10);  // Was (1,0) = 4
    CHECK(std::abs(tci::real(elem11) - 5.0) < 1e-10);  // Was (1,1) = 5
  }

  SUBCASE("Shrink 4x4 tensor to 2x2 (out-of-place version)") {
    // Create a 4x4 tensor
    tci::shape_t<cytnx::Tensor> shape = {4, 4};
    cytnx::Tensor input_tensor;
    tci::zeros(ctx, shape, input_tensor);

    // Fill with values in center 2x2 region [1:3, 1:3]
    tci::elem_coors_t<cytnx::Tensor> coord11 = {1, 1};
    tci::elem_coors_t<cytnx::Tensor> coord12 = {1, 2};
    tci::elem_coors_t<cytnx::Tensor> coord21 = {2, 1};
    tci::elem_coors_t<cytnx::Tensor> coord22 = {2, 2};

    tci::set_elem(ctx, input_tensor, coord11, cytnx::cytnx_complex128(11.0, 0.0));
    tci::set_elem(ctx, input_tensor, coord12, cytnx::cytnx_complex128(12.0, 0.0));
    tci::set_elem(ctx, input_tensor, coord21, cytnx::cytnx_complex128(21.0, 0.0));
    tci::set_elem(ctx, input_tensor, coord22, cytnx::cytnx_complex128(22.0, 0.0));

    // Shrink to extract center 2x2 sub-tensor
    tci::bond_idx_elem_coor_pair_map<cytnx::Tensor> shrink_map;
    shrink_map[0] = std::make_pair(1, 2);  // Bond 0: from 1 to 2 (2 elements)
    shrink_map[1] = std::make_pair(1, 2);  // Bond 1: from 1 to 2 (2 elements)

    cytnx::Tensor output_tensor;
    tci::shrink(ctx, input_tensor, shrink_map, output_tensor);

    // Verify the result
    auto result_shape = tci::shape(ctx, output_tensor);
    CHECK(result_shape.size() == 2);
    CHECK(result_shape[0] == 2);
    CHECK(result_shape[1] == 2);

    // Check extracted values
    tci::elem_coors_t<cytnx::Tensor> out_coord00 = {0, 0};
    tci::elem_coors_t<cytnx::Tensor> out_coord01 = {0, 1};
    tci::elem_coors_t<cytnx::Tensor> out_coord10 = {1, 0};
    tci::elem_coors_t<cytnx::Tensor> out_coord11 = {1, 1};

    auto elem00 = tci::get_elem(ctx, output_tensor, out_coord00);
    auto elem01 = tci::get_elem(ctx, output_tensor, out_coord01);
    auto elem10 = tci::get_elem(ctx, output_tensor, out_coord10);
    auto elem11 = tci::get_elem(ctx, output_tensor, out_coord11);

    // This will FAIL if shrink out-of-place version is not properly implemented
    CHECK(std::abs(tci::real(elem00) - 11.0) < 1e-10);  // From (1,1)
    CHECK(std::abs(tci::real(elem01) - 12.0) < 1e-10);  // From (1,2)
    CHECK(std::abs(tci::real(elem10) - 21.0) < 1e-10);  // From (2,1)
    CHECK(std::abs(tci::real(elem11) - 22.0) < 1e-10);  // From (2,2)
  }

  SUBCASE("Shrink should preserve complex values") {
    // Create a 3x3 tensor with complex values
    tci::shape_t<cytnx::Tensor> shape = {3, 3};
    cytnx::Tensor tensor;
    tci::zeros(ctx, shape, tensor);

    // Set complex values in top-left 2x2
    tci::elem_coors_t<cytnx::Tensor> coord00 = {0, 0};
    tci::elem_coors_t<cytnx::Tensor> coord01 = {0, 1};
    tci::elem_coors_t<cytnx::Tensor> coord10 = {1, 0};
    tci::elem_coors_t<cytnx::Tensor> coord11 = {1, 1};

    tci::set_elem(ctx, tensor, coord00, cytnx::cytnx_complex128(1.5, 2.5));
    tci::set_elem(ctx, tensor, coord01, cytnx::cytnx_complex128(3.5, 4.5));
    tci::set_elem(ctx, tensor, coord10, cytnx::cytnx_complex128(5.5, 6.5));
    tci::set_elem(ctx, tensor, coord11, cytnx::cytnx_complex128(7.5, 8.5));

    // Shrink to 2x2
    tci::bond_idx_elem_coor_pair_map<cytnx::Tensor> shrink_map;
    shrink_map[0] = std::make_pair(0, 1);
    shrink_map[1] = std::make_pair(0, 1);

    cytnx::Tensor output;
    tci::shrink(ctx, tensor, shrink_map, output);

    // Verify complex values are preserved
    auto elem00 = tci::get_elem(ctx, output, {0, 0});
    auto elem01 = tci::get_elem(ctx, output, {0, 1});
    auto elem10 = tci::get_elem(ctx, output, {1, 0});
    auto elem11 = tci::get_elem(ctx, output, {1, 1});

    // This will FAIL if shrink doesn't preserve complex values correctly
    CHECK(std::abs(tci::real(elem00) - 1.5) < 1e-10);
    CHECK(std::abs(tci::imag(elem00) - 2.5) < 1e-10);
    CHECK(std::abs(tci::real(elem01) - 3.5) < 1e-10);
    CHECK(std::abs(tci::imag(elem01) - 4.5) < 1e-10);
    CHECK(std::abs(tci::real(elem10) - 5.5) < 1e-10);
    CHECK(std::abs(tci::imag(elem10) - 6.5) < 1e-10);
    CHECK(std::abs(tci::real(elem11) - 7.5) < 1e-10);
    CHECK(std::abs(tci::imag(elem11) - 8.5) < 1e-10);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("tci::real and tci::imag extraction functionality") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("Extract real part from complex tensor (out-of-place version)") {
    // Create a complex tensor
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    cytnx::Tensor complex_tensor;
    tci::zeros(ctx, shape, complex_tensor);

    // Set complex values
    tci::elem_coors_t<cytnx::Tensor> coord00 = {0, 0};
    tci::elem_coors_t<cytnx::Tensor> coord11 = {1, 1};

    tci::set_elem(ctx, complex_tensor, coord00, cytnx::cytnx_complex128(3.14, 2.71));
    tci::set_elem(ctx, complex_tensor, coord11, cytnx::cytnx_complex128(-1.59, 0.58));

    // Extract real part
    auto real_tensor = tci::real(ctx, complex_tensor);

    // Verify only real parts are extracted
    auto elem00 = tci::get_elem(ctx, real_tensor, coord00);
    auto elem11 = tci::get_elem(ctx, real_tensor, coord11);

    // This will FAIL if real extraction is not properly implemented
    CHECK(std::abs(tci::real(elem00) - 3.14) < 1e-10);
    CHECK(std::abs(tci::imag(elem00) - 0.0) < 1e-10);  // Should be zero
    CHECK(std::abs(tci::real(elem11) - (-1.59)) < 1e-10);
    CHECK(std::abs(tci::imag(elem11) - 0.0) < 1e-10);  // Should be zero
  }

  SUBCASE("Extract imaginary part from complex tensor (out-of-place version)") {
    // Create a complex tensor
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    cytnx::Tensor complex_tensor;
    tci::zeros(ctx, shape, complex_tensor);

    // Set complex values
    tci::elem_coors_t<cytnx::Tensor> coord00 = {0, 0};
    tci::elem_coors_t<cytnx::Tensor> coord11 = {1, 1};

    tci::set_elem(ctx, complex_tensor, coord00, cytnx::cytnx_complex128(3.14, 2.71));
    tci::set_elem(ctx, complex_tensor, coord11, cytnx::cytnx_complex128(-1.59, 0.58));

    // Extract imaginary part
    auto imag_tensor = tci::imag(ctx, complex_tensor);

    // Verify only imaginary parts are extracted
    auto elem00 = tci::get_elem(ctx, imag_tensor, coord00);
    auto elem11 = tci::get_elem(ctx, imag_tensor, coord11);

    // This will FAIL if imag extraction is not properly implemented
    CHECK(std::abs(tci::real(elem00) - 2.71) < 1e-10);  // Imag part goes to real part of result
    CHECK(std::abs(tci::imag(elem00) - 0.0) < 1e-10);   // Should be zero
    CHECK(std::abs(tci::real(elem11) - 0.58) < 1e-10);  // Imag part goes to real part of result
    CHECK(std::abs(tci::imag(elem11) - 0.0) < 1e-10);   // Should be zero
  }

  SUBCASE("Extract real and imaginary parts (in-place versions)") {
    // Create a complex tensor
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    cytnx::Tensor complex_tensor;
    tci::zeros(ctx, shape, complex_tensor);

    // Set complex values
    tci::elem_coors_t<cytnx::Tensor> coord00 = {0, 0};
    tci::elem_coors_t<cytnx::Tensor> coord11 = {1, 1};

    tci::set_elem(ctx, complex_tensor, coord00, cytnx::cytnx_complex128(5.25, 7.75));
    tci::set_elem(ctx, complex_tensor, coord11, cytnx::cytnx_complex128(-2.25, -3.75));

    // Extract using in-place versions
    tci::real_ten_t<cytnx::Tensor> real_output, imag_output;
    tci::real(ctx, complex_tensor, real_output);
    tci::imag(ctx, complex_tensor, imag_output);

    // Verify real part extraction
    auto real_elem00 = tci::get_elem(ctx, real_output, coord00);
    auto real_elem11 = tci::get_elem(ctx, real_output, coord11);

    // Verify imaginary part extraction
    auto imag_elem00 = tci::get_elem(ctx, imag_output, coord00);
    auto imag_elem11 = tci::get_elem(ctx, imag_output, coord11);

    // This will FAIL if in-place versions are not properly implemented
    CHECK(std::abs(tci::real(real_elem00) - 5.25) < 1e-10);
    CHECK(std::abs(tci::real(real_elem11) - (-2.25)) < 1e-10);
    CHECK(std::abs(tci::real(imag_elem00) - 7.75) < 1e-10);
    CHECK(std::abs(tci::real(imag_elem11) - (-3.75)) < 1e-10);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("tci::cplx_conj complex conjugate functionality") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("Complex conjugate (in-place version)") {
    // Create a complex tensor
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    cytnx::Tensor tensor;
    tci::zeros(ctx, shape, tensor);

    // Set complex values
    tci::elem_coors_t<cytnx::Tensor> coord00 = {0, 0};
    tci::elem_coors_t<cytnx::Tensor> coord01 = {0, 1};
    tci::elem_coors_t<cytnx::Tensor> coord10 = {1, 0};
    tci::elem_coors_t<cytnx::Tensor> coord11 = {1, 1};

    tci::set_elem(ctx, tensor, coord00, cytnx::cytnx_complex128(1.0, 2.0));
    tci::set_elem(ctx, tensor, coord01, cytnx::cytnx_complex128(-3.0, 4.0));
    tci::set_elem(ctx, tensor, coord10, cytnx::cytnx_complex128(5.0, -6.0));
    tci::set_elem(ctx, tensor, coord11, cytnx::cytnx_complex128(-7.0, -8.0));

    // Apply complex conjugate in-place
    tci::cplx_conj(ctx, tensor);

    // Verify conjugation: real parts unchanged, imaginary parts negated
    auto elem00 = tci::get_elem(ctx, tensor, coord00);
    auto elem01 = tci::get_elem(ctx, tensor, coord01);
    auto elem10 = tci::get_elem(ctx, tensor, coord10);
    auto elem11 = tci::get_elem(ctx, tensor, coord11);

    // This will FAIL if cplx_conj is not properly implemented
    CHECK(std::abs(tci::real(elem00) - 1.0) < 1e-10);
    CHECK(std::abs(tci::imag(elem00) - (-2.0)) < 1e-10);  // 2.0 -> -2.0
    CHECK(std::abs(tci::real(elem01) - (-3.0)) < 1e-10);
    CHECK(std::abs(tci::imag(elem01) - (-4.0)) < 1e-10);  // 4.0 -> -4.0
    CHECK(std::abs(tci::real(elem10) - 5.0) < 1e-10);
    CHECK(std::abs(tci::imag(elem10) - 6.0) < 1e-10);     // -6.0 -> 6.0
    CHECK(std::abs(tci::real(elem11) - (-7.0)) < 1e-10);
    CHECK(std::abs(tci::imag(elem11) - 8.0) < 1e-10);     // -8.0 -> 8.0
  }

  SUBCASE("Complex conjugate (out-of-place version)") {
    // Create a complex tensor
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    cytnx::Tensor input_tensor;
    tci::zeros(ctx, shape, input_tensor);

    // Set complex values
    tci::elem_coors_t<cytnx::Tensor> coord00 = {0, 0};
    tci::elem_coors_t<cytnx::Tensor> coord11 = {1, 1};

    tci::set_elem(ctx, input_tensor, coord00, cytnx::cytnx_complex128(3.14, 2.71));
    tci::set_elem(ctx, input_tensor, coord11, cytnx::cytnx_complex128(-1.41, -1.73));

    // Apply complex conjugate out-of-place
    cytnx::Tensor output_tensor;
    tci::cplx_conj(ctx, input_tensor, output_tensor);

    // Verify input is unchanged
    auto input_elem00 = tci::get_elem(ctx, input_tensor, coord00);
    auto input_elem11 = tci::get_elem(ctx, input_tensor, coord11);
    CHECK(std::abs(tci::real(input_elem00) - 3.14) < 1e-10);
    CHECK(std::abs(tci::imag(input_elem00) - 2.71) < 1e-10);
    CHECK(std::abs(tci::real(input_elem11) - (-1.41)) < 1e-10);
    CHECK(std::abs(tci::imag(input_elem11) - (-1.73)) < 1e-10);

    // Verify output has conjugated values
    auto output_elem00 = tci::get_elem(ctx, output_tensor, coord00);
    auto output_elem11 = tci::get_elem(ctx, output_tensor, coord11);

    // This will FAIL if cplx_conj out-of-place version is not properly implemented
    CHECK(std::abs(tci::real(output_elem00) - 3.14) < 1e-10);    // Real unchanged
    CHECK(std::abs(tci::imag(output_elem00) - (-2.71)) < 1e-10); // Imag negated
    CHECK(std::abs(tci::real(output_elem11) - (-1.41)) < 1e-10); // Real unchanged
    CHECK(std::abs(tci::imag(output_elem11) - 1.73) < 1e-10);    // Imag negated
  }

  tci::destroy_context(ctx);
}