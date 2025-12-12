#include <doctest/doctest.h>
#include <tci/tci.h>

#include <cmath>
#include <cytnx.hpp>

TEST_CASE("TCI Norm Calculation") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  SUBCASE("3x3 identity matrix Frobenius norm should be sqrt(3)") {
    tci::CytnxTensor<cytnx::cytnx_complex128> identity;
    tci::eye(ctx, 3, identity);

    auto norm_val = tci::norm(ctx, identity);
    double expected_norm = std::sqrt(3.0);

    // This will FAIL if norm is a placeholder returning 1.0
    CHECK(std::abs(norm_val - expected_norm) < 1e-10);
  }

  SUBCASE("2x2 identity matrix Frobenius norm should be sqrt(2)") {
    tci::CytnxTensor<cytnx::cytnx_complex128> identity;
    tci::eye(ctx, 2, identity);

    auto norm_val = tci::norm(ctx, identity);
    double expected_norm = std::sqrt(2.0);

    // This will FAIL if norm is a placeholder returning 1.0
    CHECK(std::abs(norm_val - expected_norm) < 1e-10);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Expand Operations") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  SUBCASE("Expand tensor dimensions - in-place version") {
    // Test based on documentation example: {2, 2, 2} -> {3, 4, 2}
    tci::CytnxTensor<cytnx::cytnx_complex128> a;
    tci::zeros(ctx, {2, 2, 2}, a);

    // Apply expand operation: {{1, 2}, {0, 1}} means bond 1 +2, bond 0 +1
    tci::Map<tci::bond_idx_t<tci::CytnxTensor<cytnx::cytnx_complex128>>,
             tci::bond_dim_t<tci::CytnxTensor<cytnx::cytnx_complex128>>>
        bond_map = {{1, 2}, {0, 1}};
    CHECK_NOTHROW(tci::expand(ctx, a, bond_map));

    // Verify new shape is {3, 4, 2}
    auto s = tci::shape(ctx, a);
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> expected_shape = {3, 4, 2};
    CHECK(s == expected_shape);

    // Verify expanded elements are zero (as documented)
    auto el = tci::get_elem(ctx, a, {2, 3, 0});
    CHECK(std::abs(tci::real(el) - 0.0) < 1e-10);
  }

  SUBCASE("Expand tensor dimensions - out-of-place version") {
    tci::CytnxTensor<cytnx::cytnx_complex128> a, expanded;
    tci::zeros(ctx, {2, 2, 2}, a);

    // Set a non-zero element to verify preservation
    tci::set_elem(ctx, a, {1, 1, 1}, cytnx::cytnx_complex128(5.0, 0.0));

    tci::Map<tci::bond_idx_t<tci::CytnxTensor<cytnx::cytnx_complex128>>,
             tci::bond_dim_t<tci::CytnxTensor<cytnx::cytnx_complex128>>>
        bond_map = {{1, 2}, {0, 1}};
    CHECK_NOTHROW(tci::expand(ctx, a, bond_map, expanded));

    // Verify new shape
    auto s = tci::shape(ctx, expanded);
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> expected_shape = {3, 4, 2};
    CHECK(s == expected_shape);

    // Verify original data is preserved
    auto preserved_el = tci::get_elem(ctx, expanded, {1, 1, 1});
    CHECK(std::abs(tci::real(preserved_el) - 5.0) < 1e-10);

    // Verify expanded region is zero
    auto zero_el = tci::get_elem(ctx, expanded, {2, 3, 0});
    CHECK(std::abs(tci::real(zero_el) - 0.0) < 1e-10);
  }

  SUBCASE("Expand single dimension") {
    tci::CytnxTensor<cytnx::cytnx_complex128> a;
    tci::zeros(ctx, {3, 3}, a);

    // Set some test data
    tci::set_elem(ctx, a, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, a, {2, 2}, cytnx::cytnx_complex128(2.0, 0.0));

    // Expand only dimension 1 by 2
    tci::Map<tci::bond_idx_t<tci::CytnxTensor<cytnx::cytnx_complex128>>,
             tci::bond_dim_t<tci::CytnxTensor<cytnx::cytnx_complex128>>>
        bond_map = {{1, 2}};
    tci::expand(ctx, a, bond_map);

    // Verify shape changes from {3, 3} to {3, 5}
    auto s = tci::shape(ctx, a);
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> expected_shape = {3, 5};
    CHECK(s == expected_shape);

    // Verify original data preservation
    auto el1 = tci::get_elem(ctx, a, {0, 0});
    CHECK(std::abs(tci::real(el1) - 1.0) < 1e-10);
    auto el2 = tci::get_elem(ctx, a, {2, 2});
    CHECK(std::abs(tci::real(el2) - 2.0) < 1e-10);

    // Verify expanded area is zero
    auto zero_el = tci::get_elem(ctx, a, {0, 4});
    CHECK(std::abs(tci::real(zero_el) - 0.0) < 1e-10);
  }

  SUBCASE("Expand with invalid bond index should throw") {
    tci::CytnxTensor<cytnx::cytnx_complex128> a;
    tci::zeros(ctx, {2, 2}, a);

    // Try to expand bond index 3 which doesn't exist (tensor is 2D)
    tci::Map<tci::bond_idx_t<tci::CytnxTensor<cytnx::cytnx_complex128>>,
             tci::bond_dim_t<tci::CytnxTensor<cytnx::cytnx_complex128>>>
        invalid_map = {{3, 1}};
    CHECK_THROWS(tci::expand(ctx, a, invalid_map));
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Extract Sub Operations") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  SUBCASE("Extract sub-tensor - out-of-place version (documentation example)") {
    // Test based on documentation example: {3, 4, 2} with {{1, 3}, {0, 2}, {0, 2}}
    tci::CytnxTensor<cytnx::cytnx_complex128> a, sub;
    tci::zeros(ctx, {3, 4, 2}, a);  // create test tensor

    // Set known values for verification
    tci::set_elem(ctx, a, {1, 0, 0}, cytnx::cytnx_complex128(42.0, 0.0));
    tci::set_elem(ctx, a, {2, 1, 1}, cytnx::cytnx_complex128(13.0, 0.0));

    // Extract subtensor as per documentation: {{1, 3}, {0, 2}, {0, 2}}
    tci::List<tci::Pair<tci::elem_coor_t<tci::CytnxTensor<cytnx::cytnx_complex128>>,
                        tci::elem_coor_t<tci::CytnxTensor<cytnx::cytnx_complex128>>>>
        coor_pairs = {{1, 3}, {0, 2}, {0, 2}};
    CHECK_NOTHROW(tci::extract_sub(ctx, a, coor_pairs, sub));

    // Verify shape: {3,4,2} -> {2,2,2} (ranges: [1,3), [0,2), [0,2))
    auto sub_shape = tci::shape(ctx, sub);
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> expected_shape = {2, 2, 2};
    CHECK(sub_shape == expected_shape);

    // Verify element mapping as in documentation: el1 == el2
    auto el1 = tci::get_elem(ctx, a, {1, 0, 0});
    auto el2 = tci::get_elem(ctx, sub, {0, 0, 0});  // (1,0,0) maps to (0,0,0)
    CHECK(std::abs(tci::real(el1) - tci::real(el2)) < 1e-10);
    CHECK(std::abs(tci::imag(el1) - tci::imag(el2)) < 1e-10);

    // Verify another element mapping
    auto el3 = tci::get_elem(ctx, a, {2, 1, 1});
    auto el4 = tci::get_elem(ctx, sub, {1, 1, 1});  // (2,1,1) maps to (1,1,1)
    CHECK(std::abs(tci::real(el3) - tci::real(el4)) < 1e-10);
  }

  SUBCASE("Extract sub-tensor - in-place version") {
    tci::CytnxTensor<cytnx::cytnx_complex128> a;
    tci::zeros(ctx, {4, 3, 2}, a);

    // Set known values
    tci::set_elem(ctx, a, {0, 0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, a, {1, 1, 1}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, a, {2, 2, 0}, cytnx::cytnx_complex128(3.0, 0.0));

    // Extract middle portion: {{1, 3}, {1, 3}, {0, 2}}
    tci::List<tci::Pair<tci::elem_coor_t<tci::CytnxTensor<cytnx::cytnx_complex128>>,
                        tci::elem_coor_t<tci::CytnxTensor<cytnx::cytnx_complex128>>>>
        coor_pairs = {{1, 3}, {1, 3}, {0, 2}};
    CHECK_NOTHROW(tci::extract_sub(ctx, a, coor_pairs));

    // Verify new shape: {4,3,2} -> {2,2,2}
    auto new_shape = tci::shape(ctx, a);
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> expected_shape = {2, 2, 2};
    CHECK(new_shape == expected_shape);

    // Verify element at (1,1,1) in original is now at (0,0,1)
    auto extracted_elem = tci::get_elem(ctx, a, {0, 0, 1});
    CHECK(std::abs(tci::real(extracted_elem) - 2.0) < 1e-10);
  }

  SUBCASE("Extract single element") {
    tci::CytnxTensor<cytnx::cytnx_complex128> a, sub;
    tci::zeros(ctx, {3, 3}, a);
    tci::set_elem(ctx, a, {1, 2}, cytnx::cytnx_complex128(99.0, 0.0));

    // Extract single element at position (1,2)
    tci::List<tci::Pair<tci::elem_coor_t<tci::CytnxTensor<cytnx::cytnx_complex128>>,
                        tci::elem_coor_t<tci::CytnxTensor<cytnx::cytnx_complex128>>>>
        coor_pairs = {{1, 2}, {2, 3}};
    tci::extract_sub(ctx, a, coor_pairs, sub);

    // Should result in 1x1 tensor
    auto sub_shape = tci::shape(ctx, sub);
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> expected_shape = {1, 1};
    CHECK(sub_shape == expected_shape);

    auto extracted_val = tci::get_elem(ctx, sub, {0, 0});
    CHECK(std::abs(tci::real(extracted_val) - 99.0) < 1e-10);
  }

  SUBCASE("Error handling - invalid coordinate pairs") {
    tci::CytnxTensor<cytnx::cytnx_complex128> a;
    tci::zeros(ctx, {3, 3}, a);

    // Wrong number of coordinate pairs (tensor is 2D, but 3 pairs provided)
    tci::List<tci::Pair<tci::elem_coor_t<tci::CytnxTensor<cytnx::cytnx_complex128>>,
                        tci::elem_coor_t<tci::CytnxTensor<cytnx::cytnx_complex128>>>>
        wrong_count = {{0, 2}, {0, 2}, {0, 1}};
    CHECK_THROWS(tci::extract_sub(ctx, a, wrong_count));

    // Invalid range (start >= end)
    tci::List<tci::Pair<tci::elem_coor_t<tci::CytnxTensor<cytnx::cytnx_complex128>>,
                        tci::elem_coor_t<tci::CytnxTensor<cytnx::cytnx_complex128>>>>
        invalid_range = {{2, 1}, {0, 2}};
    CHECK_THROWS(tci::extract_sub(ctx, a, invalid_range));

    // Out of bounds range
    tci::List<tci::Pair<tci::elem_coor_t<tci::CytnxTensor<cytnx::cytnx_complex128>>,
                        tci::elem_coor_t<tci::CytnxTensor<cytnx::cytnx_complex128>>>>
        out_of_bounds = {{0, 2}, {0, 5}};  // 5 > tensor dimension 3
    CHECK_THROWS(tci::extract_sub(ctx, a, out_of_bounds));
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Replace Sub Operations") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  SUBCASE("Replace sub-tensor - in-place version (documentation example)") {
    // Test based on documentation example: {3, 4, 2} with {2, 2, 2} sub at {1, 2, 0}
    tci::CytnxTensor<cytnx::cytnx_complex128> a, sub;
    tci::zeros(ctx, {3, 4, 2}, a);
    tci::zeros(ctx, {2, 2, 2}, sub);

    // Set known values in sub-tensor
    tci::set_elem(ctx, sub, {0, 0, 0}, cytnx::cytnx_complex128(42.0, 0.0));
    tci::set_elem(ctx, sub, {1, 1, 1}, cytnx::cytnx_complex128(13.0, 0.0));

    // Replace subtensor as per documentation: sub at {1, 2, 0}
    tci::elem_coors_t<tci::CytnxTensor<cytnx::cytnx_complex128>> begin_pt = {1, 2, 0};
    CHECK_NOTHROW(tci::replace_sub(ctx, a, sub, begin_pt));

    // Verify element mapping as in documentation: el1 == el2
    auto el1 = tci::get_elem(ctx, a, {1, 2, 0});    // begin_pt position
    auto el2 = tci::get_elem(ctx, sub, {0, 0, 0});  // sub origin
    CHECK(std::abs(tci::real(el1) - tci::real(el2)) < 1e-10);
    CHECK(std::abs(tci::real(el1) - 42.0) < 1e-10);

    // Verify another element mapping
    auto el3 = tci::get_elem(ctx, a, {2, 3, 1});    // begin_pt + {1,1,1}
    auto el4 = tci::get_elem(ctx, sub, {1, 1, 1});  // sub position
    CHECK(std::abs(tci::real(el3) - tci::real(el4)) < 1e-10);
    CHECK(std::abs(tci::real(el3) - 13.0) < 1e-10);

    // Verify area outside replacement is unchanged (should remain 0)
    auto unchanged_elem = tci::get_elem(ctx, a, {0, 0, 0});
    CHECK(std::abs(tci::real(unchanged_elem) - 0.0) < 1e-10);
  }

  SUBCASE("Replace sub-tensor - out-of-place version") {
    tci::CytnxTensor<cytnx::cytnx_complex128> a, sub, result;
    tci::zeros(ctx, {4, 4}, a);
    tci::zeros(ctx, {2, 2}, sub);
    tci::fill(ctx, {2, 2}, cytnx::cytnx_complex128(1.0, 0.0), sub);  // sub-tensor filled with 1.0

    // Set initial value in main tensor
    tci::set_elem(ctx, a, {0, 0}, cytnx::cytnx_complex128(99.0, 0.0));

    // Replace 2x2 sub at position (1,1)
    tci::elem_coors_t<tci::CytnxTensor<cytnx::cytnx_complex128>> begin_pt = {1, 1};
    CHECK_NOTHROW(tci::replace_sub(ctx, a, sub, begin_pt, result));

    // Verify replacement
    auto replaced_elem1 = tci::get_elem(ctx, result, {1, 1});
    CHECK(std::abs(tci::real(replaced_elem1) - 1.0) < 1e-10);
    auto replaced_elem2 = tci::get_elem(ctx, result, {2, 2});
    CHECK(std::abs(tci::real(replaced_elem2) - 1.0) < 1e-10);

    // Verify original area unchanged
    auto unchanged_elem = tci::get_elem(ctx, result, {0, 0});
    CHECK(std::abs(tci::real(unchanged_elem) - 99.0) < 1e-10);

    // Verify original tensor is unmodified
    auto orig_elem = tci::get_elem(ctx, a, {1, 1});
    CHECK(std::abs(tci::real(orig_elem) - 0.0) < 1e-10);
  }

  SUBCASE("Replace single element") {
    tci::CytnxTensor<cytnx::cytnx_complex128> a, sub;
    tci::zeros(ctx, {3, 3}, a);
    tci::zeros(ctx, {1, 1}, sub);
    tci::set_elem(ctx, sub, {0, 0}, cytnx::cytnx_complex128(777.0, 0.0));

    // Replace single element at position (1,2)
    tci::elem_coors_t<tci::CytnxTensor<cytnx::cytnx_complex128>> begin_pt = {1, 2};
    tci::replace_sub(ctx, a, sub, begin_pt);

    // Verify replacement
    auto replaced_val = tci::get_elem(ctx, a, {1, 2});
    CHECK(std::abs(tci::real(replaced_val) - 777.0) < 1e-10);

    // Verify other elements unchanged
    auto other_elem = tci::get_elem(ctx, a, {0, 0});
    CHECK(std::abs(tci::real(other_elem) - 0.0) < 1e-10);
  }

  SUBCASE("Replace with boundary conditions") {
    tci::CytnxTensor<cytnx::cytnx_complex128> a, sub;
    tci::zeros(ctx, {3, 3}, a);
    tci::zeros(ctx, {2, 2}, sub);
    tci::fill(ctx, {2, 2}, cytnx::cytnx_complex128(1.0, 0.0), sub);

    // Replace at boundary (should fit exactly)
    tci::elem_coors_t<tci::CytnxTensor<cytnx::cytnx_complex128>> begin_pt = {1, 1};
    tci::replace_sub(ctx, a, sub, begin_pt);

    // Verify corner replacement
    auto corner_elem = tci::get_elem(ctx, a, {2, 2});
    CHECK(std::abs(tci::real(corner_elem) - 1.0) < 1e-10);
  }

  SUBCASE("Error handling - dimension mismatch") {
    tci::CytnxTensor<cytnx::cytnx_complex128> a, sub;
    tci::zeros(ctx, {3, 3}, a);       // 2D tensor
    tci::zeros(ctx, {2, 2, 2}, sub);  // 3D tensor

    tci::elem_coors_t<tci::CytnxTensor<cytnx::cytnx_complex128>> begin_pt = {0, 0};
    CHECK_THROWS(tci::replace_sub(ctx, a, sub, begin_pt));
  }

  SUBCASE("Error handling - begin_pt dimension mismatch") {
    tci::CytnxTensor<cytnx::cytnx_complex128> a, sub;
    tci::zeros(ctx, {3, 3}, a);
    tci::zeros(ctx, {2, 2}, sub);

    // Wrong number of coordinates in begin_pt
    tci::elem_coors_t<tci::CytnxTensor<cytnx::cytnx_complex128>> wrong_begin_pt
        = {0};  // only 1 coordinate for 2D tensor
    CHECK_THROWS(tci::replace_sub(ctx, a, sub, wrong_begin_pt));
  }

  SUBCASE("Error handling - sub-tensor exceeds bounds") {
    tci::CytnxTensor<cytnx::cytnx_complex128> a, sub;
    tci::zeros(ctx, {3, 3}, a);
    tci::zeros(ctx, {2, 2}, sub);

    // begin_pt + sub_shape would exceed tensor bounds
    tci::elem_coors_t<tci::CytnxTensor<cytnx::cytnx_complex128>> out_of_bounds_pt
        = {2, 2};  // 2+2 > 3
    CHECK_THROWS(tci::replace_sub(ctx, a, sub, out_of_bounds_pt));
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Diagonal Operations") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  SUBCASE("Vector to diagonal matrix conversion") {
    // Create a vector [1, 2, 3]
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {3};
    tci::CytnxTensor<cytnx::cytnx_complex128> vector;
    tci::zeros(ctx, shape, vector);

    tci::set_elem(ctx, vector, {0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, vector, {1}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, vector, {2}, cytnx::cytnx_complex128(3.0, 0.0));

    // Convert to diagonal matrix
    tci::diag(ctx, vector);

    // Should now be 3x3 matrix
    CHECK(tci::order(ctx, vector) == 2);
    CHECK(tci::shape(ctx, vector)[0] == 3);
    CHECK(tci::shape(ctx, vector)[1] == 3);

    // Check diagonal elements
    CHECK(std::abs(tci::real(tci::get_elem(ctx, vector, {0, 0})) - 1.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, vector, {1, 1})) - 2.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, vector, {2, 2})) - 3.0) < 1e-10);

    // Check off-diagonal elements are zero
    CHECK(std::abs(tci::real(tci::get_elem(ctx, vector, {0, 1}))) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, vector, {1, 0}))) < 1e-10);
  }

  SUBCASE("Diagonal matrix to vector conversion") {
    tci::CytnxTensor<cytnx::cytnx_complex128> identity;
    tci::eye(ctx, 3, identity);

    // Convert to diagonal vector
    tci::diag(ctx, identity);

    // Should now be order-1 vector
    CHECK(tci::order(ctx, identity) == 1);
    CHECK(tci::size(ctx, identity) == 3);

    // Check elements are [1, 1, 1]
    CHECK(std::abs(tci::real(tci::get_elem(ctx, identity, {0})) - 1.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, identity, {1})) - 1.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, identity, {2})) - 1.0) < 1e-10);
  }

  tci::destroy_context(ctx);
}

// FIXME: Trace operations cause AddressSanitizer container-overflow in Cytnx internal code
// This appears to be a bug in the Cytnx library itself (UniTensor::UniTensor constructor)
// Skip this entire test case until the Cytnx library issue is resolved
TEST_CASE("TCI Trace Operations" * doctest::skip()) {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  SUBCASE("Matrix trace calculation") {
    // Create a 3x3 matrix with specific diagonal values
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {3, 3};
    tci::CytnxTensor<cytnx::cytnx_complex128> matrix;
    tci::zeros(ctx, shape, matrix);

    // Set diagonal to [2, 3, 4]
    tci::set_elem(ctx, matrix, {0, 0}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 1}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, matrix, {2, 2}, cytnx::cytnx_complex128(4.0, 0.0));

    // Set some off-diagonal elements
    tci::set_elem(ctx, matrix, {0, 1}, cytnx::cytnx_complex128(5.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 2}, cytnx::cytnx_complex128(6.0, 0.0));

    // Calculate trace (sum of diagonal elements)
    tci::bond_idx_pairs_t<tci::CytnxTensor<cytnx::cytnx_complex128>> pairs = {{0, 1}};
    tci::trace(ctx, matrix, pairs);

    // Should now be scalar with value 2+3+4 = 9
    CHECK(tci::order(ctx, matrix) == 0);  // Scalar tensor
    CHECK(std::abs(tci::real(tci::get_elem(ctx, matrix, {})) - 9.0) < 1e-10);
  }

  SUBCASE("3D tensor trace calculation") {
    // Create a 2x3x2 tensor and trace over first and last dimensions
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 3, 2};
    tci::CytnxTensor<cytnx::cytnx_complex128> tensor3d;
    tci::zeros(ctx, shape, tensor3d);

    // Set diagonal elements T[i,j,i] where i=0,1 and j=0,1,2
    tci::set_elem(ctx, tensor3d, {0, 0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, tensor3d, {0, 1, 0}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, tensor3d, {0, 2, 0}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, tensor3d, {1, 0, 1}, cytnx::cytnx_complex128(4.0, 0.0));
    tci::set_elem(ctx, tensor3d, {1, 1, 1}, cytnx::cytnx_complex128(5.0, 0.0));
    tci::set_elem(ctx, tensor3d, {1, 2, 1}, cytnx::cytnx_complex128(6.0, 0.0));

    // Set some non-diagonal elements
    tci::set_elem(ctx, tensor3d, {0, 0, 1}, cytnx::cytnx_complex128(10.0, 0.0));
    tci::set_elem(ctx, tensor3d, {1, 1, 0}, cytnx::cytnx_complex128(20.0, 0.0));

    // Calculate trace over dimensions 0 and 2
    tci::bond_idx_pairs_t<tci::CytnxTensor<cytnx::cytnx_complex128>> pairs = {{0, 2}};
    tci::CytnxTensor<cytnx::cytnx_complex128> result;
    tci::trace(ctx, tensor3d, pairs, result);

    // Result should be 1D tensor with shape [3] containing [1+4, 2+5, 3+6] = [5, 7, 9]
    auto result_shape = tci::shape(ctx, result);
    CHECK(result_shape.size() == 1);
    CHECK(result_shape[0] == 3);

    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {0})) - 5.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {1})) - 7.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {2})) - 9.0) < 1e-10);
  }

  SUBCASE("4D tensor trace calculation") {
    // Create a 2x2x3x2 tensor and trace over dimensions 0 and 3
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 2, 3, 2};
    tci::CytnxTensor<cytnx::cytnx_complex128> tensor4d;
    tci::zeros(ctx, shape, tensor4d);

    // Set diagonal elements T[i,j,k,i] where i=0,1, j=0,1, k=0,1,2
    for (size_t j = 0; j < 2; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        tci::set_elem(ctx, tensor4d, {0, j, k, 0}, cytnx::cytnx_complex128(1.0 + j + k, 0.0));
        tci::set_elem(ctx, tensor4d, {1, j, k, 1}, cytnx::cytnx_complex128(10.0 + j + k, 0.0));
      }
    }

    // Calculate trace over dimensions 0 and 3
    tci::bond_idx_pairs_t<tci::CytnxTensor<cytnx::cytnx_complex128>> pairs = {{0, 3}};
    tci::CytnxTensor<cytnx::cytnx_complex128> result;
    tci::trace(ctx, tensor4d, pairs, result);

    // Result should be 2D tensor with shape [2, 3]
    auto result_shape = tci::shape(ctx, result);
    CHECK(result_shape.size() == 2);
    CHECK(result_shape[0] == 2);
    CHECK(result_shape[1] == 3);

    // Check specific values: result[j,k] = T[0,j,k,0] + T[1,j,k,1]
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {0, 0})) - 11.0)
          < 1e-10);  // (1+0+0) + (10+0+0) = 11
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {0, 1})) - 13.0)
          < 1e-10);  // (1+0+1) + (10+0+1) = 13
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {1, 2})) - 17.0)
          < 1e-10);  // (1+1+2) + (10+1+2) = 17
  }

  SUBCASE("Complex tensor trace calculation") {
    // Test with complex numbers
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {3, 2, 3};
    tci::CytnxTensor<cytnx::cytnx_complex128> tensor_complex;
    tci::zeros(ctx, shape, tensor_complex);

    // Set complex diagonal elements T[i,j,i]
    tci::set_elem(ctx, tensor_complex, {0, 0, 0}, cytnx::cytnx_complex128(1.0, 2.0));
    tci::set_elem(ctx, tensor_complex, {0, 1, 0}, cytnx::cytnx_complex128(2.0, -1.0));
    tci::set_elem(ctx, tensor_complex, {1, 0, 1}, cytnx::cytnx_complex128(-1.0, 3.0));
    tci::set_elem(ctx, tensor_complex, {1, 1, 1}, cytnx::cytnx_complex128(0.5, -2.5));
    tci::set_elem(ctx, tensor_complex, {2, 0, 2}, cytnx::cytnx_complex128(3.0, 1.0));
    tci::set_elem(ctx, tensor_complex, {2, 1, 2}, cytnx::cytnx_complex128(-0.5, 4.0));

    // Calculate trace over dimensions 0 and 2
    tci::bond_idx_pairs_t<tci::CytnxTensor<cytnx::cytnx_complex128>> pairs = {{0, 2}};
    tci::CytnxTensor<cytnx::cytnx_complex128> result;
    tci::trace(ctx, tensor_complex, pairs, result);

    // Result should be 1D tensor with shape [2]
    auto result_shape = tci::shape(ctx, result);
    CHECK(result_shape.size() == 1);
    CHECK(result_shape[0] == 2);

    // Check complex trace values
    auto elem0 = tci::get_elem(ctx, result, {0});  // (1+2i) + (-1+3i) + (3+1i) = 3+6i
    auto elem1 = tci::get_elem(ctx, result, {1});  // (2-1i) + (0.5-2.5i) + (-0.5+4i) = 2+0.5i

    CHECK(std::abs(tci::real(elem0) - 3.0) < 1e-10);
    CHECK(std::abs(tci::imag(elem0) - 6.0) < 1e-10);
    CHECK(std::abs(tci::real(elem1) - 2.0) < 1e-10);
    CHECK(std::abs(tci::imag(elem1) - 0.5) < 1e-10);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Tensor Contraction") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  SUBCASE("Einstein notation contraction") {
    // Create two tensors for contraction
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape_a = {2, 3};
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape_b = {3, 2};

    tci::CytnxTensor<cytnx::cytnx_complex128> a, b, c;
    tci::zeros(ctx, shape_a, a);
    tci::fill(ctx, shape_a, cytnx::cytnx_complex128(1.0, 0.0), a);

    tci::zeros(ctx, shape_b, b);
    tci::fill(ctx, shape_b, cytnx::cytnx_complex128(1.0, 0.0), b);

    // Test string notation contraction
    // TODO: Uncomment when contract is fully implemented
    // CHECK_THROWS_AS(tci::contract(ctx, a, "ij", b, "jk", c, "ik"), std::runtime_error);
    (void)c;  // Suppress unused variable warning
  }

  SUBCASE("Label-based contraction") {
    // Create two tensors for contraction
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape_a = {2, 3};
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape_b = {3, 2};

    tci::CytnxTensor<cytnx::cytnx_complex128> a, b, c;
    tci::zeros(ctx, shape_a, a);
    tci::fill(ctx, shape_a, cytnx::cytnx_complex128(1.0, 0.0), a);

    tci::zeros(ctx, shape_b, b);
    tci::fill(ctx, shape_b, cytnx::cytnx_complex128(1.0, 0.0), b);

    // Test label-based contraction
    tci::List<tci::bond_label_t<tci::CytnxTensor<cytnx::cytnx_complex128>>> labs_a = {1, -1};
    tci::List<tci::bond_label_t<tci::CytnxTensor<cytnx::cytnx_complex128>>> labs_b = {-1, 2};
    tci::List<tci::bond_label_t<tci::CytnxTensor<cytnx::cytnx_complex128>>> labs_c = {1, 2};

    // TODO: Uncomment when contract is fully implemented
    // CHECK_THROWS_AS(tci::contract(ctx, a, labs_a, b, labs_b, c, labs_c), std::runtime_error);
    (void)c;  // Suppress unused variable warning
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Tensor Contraction") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  SUBCASE("Matrix multiplication via contraction: ij,jk->ik") {
    // Create two 2x2 matrices for testing
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 2};
    tci::CytnxTensor<cytnx::cytnx_complex128> a, b, c;
    tci::zeros(ctx, shape, a);
    tci::zeros(ctx, shape, b);

    // Fill matrix A: [[1, 2], [3, 4]]
    tci::set_elem(ctx, a, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, a, {0, 1}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, a, {1, 0}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, a, {1, 1}, cytnx::cytnx_complex128(4.0, 0.0));

    // Fill matrix B: [[5, 6], [7, 8]]
    tci::set_elem(ctx, b, {0, 0}, cytnx::cytnx_complex128(5.0, 0.0));
    tci::set_elem(ctx, b, {0, 1}, cytnx::cytnx_complex128(6.0, 0.0));
    tci::set_elem(ctx, b, {1, 0}, cytnx::cytnx_complex128(7.0, 0.0));
    tci::set_elem(ctx, b, {1, 1}, cytnx::cytnx_complex128(8.0, 0.0));

    // Contract using Einstein notation: A * B
    tci::contract(ctx, a, "ij", b, "jk", c, "ik");

    // Check result: A*B = [[19, 22], [43, 50]]
    auto c00 = tci::get_elem(ctx, c, {0, 0});
    auto c01 = tci::get_elem(ctx, c, {0, 1});
    auto c10 = tci::get_elem(ctx, c, {1, 0});
    auto c11 = tci::get_elem(ctx, c, {1, 1});

    CHECK(std::abs(tci::real(c00) - 19.0) < 1e-10);
    CHECK(std::abs(tci::real(c01) - 22.0) < 1e-10);
    CHECK(std::abs(tci::real(c10) - 43.0) < 1e-10);
    CHECK(std::abs(tci::real(c11) - 50.0) < 1e-10);
  }

  SUBCASE("NCON notation basic test") {
    // Create test tensors for contraction
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape_a = {2, 3};
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape_b = {3, 2};
    tci::CytnxTensor<cytnx::cytnx_complex128> a, b, c;
    tci::zeros(ctx, shape_a, a);
    tci::zeros(ctx, shape_b, b);

    // Fill with simple test values
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        tci::set_elem(ctx, a,
                      {static_cast<tci::elem_coor_t<tci::CytnxTensor<cytnx::cytnx_complex128>>>(i),
                       static_cast<tci::elem_coor_t<tci::CytnxTensor<cytnx::cytnx_complex128>>>(j)},
                      cytnx::cytnx_complex128(i * 3 + j + 1, 0.0));
      }
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 2; ++j) {
        tci::set_elem(ctx, b,
                      {static_cast<tci::elem_coor_t<tci::CytnxTensor<cytnx::cytnx_complex128>>>(i),
                       static_cast<tci::elem_coor_t<tci::CytnxTensor<cytnx::cytnx_complex128>>>(j)},
                      cytnx::cytnx_complex128(i * 2 + j + 1, 0.0));
      }
    }

    // Test NCON with mixed positive/negative output labels
    tci::List<tci::bond_label_t<tci::CytnxTensor<cytnx::cytnx_complex128>>> bd_labs_a
        = {1, 2};  // positive labels for tensor a
    tci::List<tci::bond_label_t<tci::CytnxTensor<cytnx::cytnx_complex128>>> bd_labs_b
        = {2, 3};  // mixed: 2 (contract), 3 (positive output)
    tci::List<tci::bond_label_t<tci::CytnxTensor<cytnx::cytnx_complex128>>> bd_labs_c
        = {1, 3};  // Output labels matching the free axes

    // Perform contraction
    tci::contract(ctx, a, bd_labs_a, b, bd_labs_b, c, bd_labs_c);

    // Verify contraction occurred (shape should be 2x2)
    auto c_shape = tci::shape(ctx, c);
    CHECK(c_shape.size() == 2);
    CHECK(c_shape[0] == 2);
    CHECK(c_shape[1] == 2);
  }

  SUBCASE("Vector dot product via contraction: i,i->") {
    // Create two vectors for dot product
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {3};
    tci::CytnxTensor<cytnx::cytnx_complex128> a, b, c;
    tci::zeros(ctx, shape, a);
    tci::zeros(ctx, shape, b);

    // Fill vectors: a = [1, 2, 3], b = [4, 5, 6]
    tci::set_elem(ctx, a, {0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, a, {1}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, a, {2}, cytnx::cytnx_complex128(3.0, 0.0));

    tci::set_elem(ctx, b, {0}, cytnx::cytnx_complex128(4.0, 0.0));
    tci::set_elem(ctx, b, {1}, cytnx::cytnx_complex128(5.0, 0.0));
    tci::set_elem(ctx, b, {2}, cytnx::cytnx_complex128(6.0, 0.0));

    // Contract to compute dot product: sum_i a[i] * b[i] = 32
    tci::contract(ctx, a, "i", b, "i", c, "");

    // Result should be scalar-like with value 32 (Cytnx returns [1] shape)
    auto c_shape = tci::shape(ctx, c);
    CHECK(c_shape.size() == 1);                    // Cytnx scalar result is [1] shape
    CHECK(c_shape[0] == 1);                        // Single element
    auto dot_result = tci::get_elem(ctx, c, {0});  // Access single element
    CHECK(std::abs(tci::real(dot_result) - 32.0) < 1e-10);
  }

  SUBCASE("Outer product via contraction: i,j->ij") {
    // Create two vectors for outer product
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape_a = {2};
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape_b = {3};
    tci::CytnxTensor<cytnx::cytnx_complex128> a, b, c;
    tci::zeros(ctx, shape_a, a);
    tci::zeros(ctx, shape_b, b);

    // Fill vectors: a = [1, 2], b = [3, 4, 5]
    tci::set_elem(ctx, a, {0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, a, {1}, cytnx::cytnx_complex128(2.0, 0.0));

    tci::set_elem(ctx, b, {0}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, b, {1}, cytnx::cytnx_complex128(4.0, 0.0));
    tci::set_elem(ctx, b, {2}, cytnx::cytnx_complex128(5.0, 0.0));

    // Contract to compute outer product: a[i] * b[j] -> c[i,j]
    tci::contract(ctx, a, "i", b, "j", c, "ij");

    // Result should be 2x3 matrix
    auto c_shape = tci::shape(ctx, c);
    CHECK(c_shape.size() == 2);
    CHECK(c_shape[0] == 2);
    CHECK(c_shape[1] == 3);

    // Check specific values: c[0,0] = 1*3 = 3, c[1,2] = 2*5 = 10
    auto c00 = tci::get_elem(ctx, c, {0, 0});
    auto c12 = tci::get_elem(ctx, c, {1, 2});
    CHECK(std::abs(tci::real(c00) - 3.0) < 1e-10);
    CHECK(std::abs(tci::real(c12) - 10.0) < 1e-10);
  }

  SUBCASE("Duplicate labels within single tensor should throw") {
    // Test that repeated labels within a single tensor are detected and rejected
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape_a = {3, 3, 4};
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape_b = {5};
    tci::CytnxTensor<cytnx::cytnx_complex128> a, b, c;
    tci::zeros(ctx, shape_a, a);
    tci::zeros(ctx, shape_b, b);

    // Test case 1: Duplicate labels in first tensor using string notation
    // "iij" has label 'i' repeated twice within tensor a
    CHECK_THROWS_AS(tci::contract(ctx, a, "iij", b, "k", c, "jk"), std::invalid_argument);

    // Test case 2: Duplicate labels in second tensor using string notation
    // "kk" has label 'k' repeated twice within tensor b - but b is rank-1, so this is invalid
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape_b2 = {5, 5};
    tci::CytnxTensor<cytnx::cytnx_complex128> b2;
    tci::zeros(ctx, shape_b2, b2);
    CHECK_THROWS_AS(tci::contract(ctx, a, "ijk", b2, "kk", c, "ij"), std::invalid_argument);

    // Test case 3: Duplicate labels in first tensor using label list notation
    tci::List<tci::bond_label_t<tci::CytnxTensor<cytnx::cytnx_complex128>>> bd_labs_a_dup
        = {1, 1, 2};  // Label 1 appears twice
    tci::List<tci::bond_label_t<tci::CytnxTensor<cytnx::cytnx_complex128>>> bd_labs_b_single
        = {3};
    tci::List<tci::bond_label_t<tci::CytnxTensor<cytnx::cytnx_complex128>>> bd_labs_c_out
        = {2, 3};
    CHECK_THROWS_AS(
        tci::contract(ctx, a, bd_labs_a_dup, b, bd_labs_b_single, c, bd_labs_c_out),
        std::invalid_argument);

    // Test case 4: No duplicate - should succeed
    tci::List<tci::bond_label_t<tci::CytnxTensor<cytnx::cytnx_complex128>>> bd_labs_a_valid
        = {1, 2, 3};
    tci::List<tci::bond_label_t<tci::CytnxTensor<cytnx::cytnx_complex128>>> bd_labs_b_valid
        = {4};
    tci::List<tci::bond_label_t<tci::CytnxTensor<cytnx::cytnx_complex128>>> bd_labs_c_valid
        = {1, 2, 3, 4};
    CHECK_NOTHROW(tci::contract(ctx, a, bd_labs_a_valid, b, bd_labs_b_valid, c, bd_labs_c_valid));
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Stack Operations") {
  tci::context_handle_t<tci::CytnxTensor<cytnx::cytnx_complex128>> ctx;
  tci::create_context(ctx);

  SUBCASE("Stack three 2x4 tensors along axis 1") {
    // Create three tensors with identical shape {2, 4} - matching documentation example
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 4};
    tci::CytnxTensor<cytnx::cytnx_complex128> a, b, c, d;

    // Create simple tensors with known values for debugging
    tci::zeros(ctx, shape, a);
    tci::zeros(ctx, shape, b);
    tci::zeros(ctx, shape, c);

    // Fill with simple test patterns
    tci::set_elem(ctx, a, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, b, {0, 0}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, c, {0, 0}, cytnx::cytnx_complex128(3.0, 0.0));

    // Stack tensors along axis 1 (matching documentation example)
    tci::List<tci::CytnxTensor<cytnx::cytnx_complex128>> tensors = {a, b, c};
    tci::stack(ctx, tensors, 1, d);

    // Verify output shape is {2, 3, 4} - new dimension of size 3 at position 1
    auto result_shape = tci::shape(ctx, d);
    CHECK(result_shape.size() == 3);
    CHECK(result_shape[0] == 2);  // Original dimension 0
    CHECK(result_shape[1] == 3);  // New stacking dimension (3 tensors)
    CHECK(result_shape[2] == 4);  // Original dimension 1

    // Verify element correspondence: tensor a[0,0] should equal d[0,0,0]
    auto el_a = tci::get_elem(ctx, a, {0, 0});
    auto el_d_a = tci::get_elem(ctx, d, {0, 0, 0});
    CHECK(std::abs(tci::real(el_a) - tci::real(el_d_a)) < 1e-10);

    // Verify tensor b[0,0] should equal d[0,1,0]
    auto el_b = tci::get_elem(ctx, b, {0, 0});
    auto el_d_b = tci::get_elem(ctx, d, {0, 1, 0});
    CHECK(std::abs(tci::real(el_b) - tci::real(el_d_b)) < 1e-10);

    // Verify tensor c[0,0] should equal d[0,2,0]
    auto el_c = tci::get_elem(ctx, c, {0, 0});
    auto el_d_c = tci::get_elem(ctx, d, {0, 2, 0});
    CHECK(std::abs(tci::real(el_c) - tci::real(el_d_c)) < 1e-10);
  }

  SUBCASE("Stack two 3x3 tensors along axis 0") {
    // Create two simple test tensors
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {3, 3};
    tci::CytnxTensor<cytnx::cytnx_complex128> a, b, result;
    tci::zeros(ctx, shape, a);
    tci::zeros(ctx, shape, b);

    // Set specific test values
    tci::set_elem(ctx, a, {1, 1}, cytnx::cytnx_complex128(5.0, 0.0));
    tci::set_elem(ctx, b, {2, 2}, cytnx::cytnx_complex128(6.0, 0.0));

    // Stack along axis 0
    tci::List<tci::CytnxTensor<cytnx::cytnx_complex128>> tensors = {a, b};
    tci::stack(ctx, tensors, 0, result);

    // Verify output shape is {2, 3, 3}
    auto result_shape = tci::shape(ctx, result);
    CHECK(result_shape.size() == 3);
    CHECK(result_shape[0] == 2);  // New stacking dimension
    CHECK(result_shape[1] == 3);  // Original dimension 0
    CHECK(result_shape[2] == 3);  // Original dimension 1

    // Verify element values
    auto el_a11 = tci::get_elem(ctx, result, {0, 1, 1});  // First tensor at (1,1)
    auto el_b22 = tci::get_elem(ctx, result, {1, 2, 2});  // Second tensor at (2,2)
    CHECK(std::abs(tci::real(el_a11) - 5.0) < 1e-10);
    CHECK(std::abs(tci::real(el_b22) - 6.0) < 1e-10);
  }

  SUBCASE("Stack along last dimension (axis 2)") {
    // Create two 2x3 tensors
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape = {2, 3};
    tci::CytnxTensor<cytnx::cytnx_complex128> a, b, result;
    tci::zeros(ctx, shape, a);
    tci::zeros(ctx, shape, b);

    // Set specific test values
    tci::set_elem(ctx, a, {1, 2}, cytnx::cytnx_complex128(7.0, 0.0));
    tci::set_elem(ctx, b, {1, 2}, cytnx::cytnx_complex128(8.0, 0.0));

    // Stack along axis 2 (end)
    tci::List<tci::CytnxTensor<cytnx::cytnx_complex128>> tensors = {a, b};
    tci::stack(ctx, tensors, 2, result);

    // Verify output shape is {2, 3, 2}
    auto result_shape = tci::shape(ctx, result);
    CHECK(result_shape.size() == 3);
    CHECK(result_shape[0] == 2);  // Original dimension 0
    CHECK(result_shape[1] == 3);  // Original dimension 1
    CHECK(result_shape[2] == 2);  // New stacking dimension

    // Verify element values
    auto el_zeros = tci::get_elem(ctx, result, {1, 2, 0});  // From first tensor
    auto el_ones = tci::get_elem(ctx, result, {1, 2, 1});   // From second tensor
    CHECK(std::abs(tci::real(el_zeros) - 7.0) < 1e-10);
    CHECK(std::abs(tci::real(el_ones) - 8.0) < 1e-10);
  }

  SUBCASE("Error handling: empty tensor list") {
    tci::CytnxTensor<cytnx::cytnx_complex128> result;
    tci::List<tci::CytnxTensor<cytnx::cytnx_complex128>> empty_tensors = {};

    // Should throw exception for empty input
    CHECK_THROWS_AS(tci::stack(ctx, empty_tensors, 0, result), std::invalid_argument);
  }

  SUBCASE("Error handling: mismatched tensor shapes") {
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape1 = {2, 3};
    tci::shape_t<tci::CytnxTensor<cytnx::cytnx_complex128>> shape2 = {2, 4};  // Different shape
    tci::CytnxTensor<cytnx::cytnx_complex128> a, b, result;
    tci::zeros(ctx, shape1, a);
    tci::zeros(ctx, shape2, b);

    tci::List<tci::CytnxTensor<cytnx::cytnx_complex128>> tensors = {a, b};

    // Should throw exception for mismatched shapes
    CHECK_THROWS_AS(tci::stack(ctx, tensors, 0, result), std::invalid_argument);
  }

  tci::destroy_context(ctx);
}