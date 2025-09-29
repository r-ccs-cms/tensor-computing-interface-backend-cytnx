#include <doctest/doctest.h>
#include <tci/tci.h>
#include <cmath>
#include <cytnx.hpp>

TEST_CASE("TCI Norm Calculation") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("3x3 identity matrix Frobenius norm should be sqrt(3)") {
    cytnx::Tensor identity;
    tci::eye(ctx, 3, identity);

    auto norm_val = tci::norm(ctx, identity);
    double expected_norm = std::sqrt(3.0);

    // This will FAIL if norm is a placeholder returning 1.0
    CHECK(std::abs(norm_val - expected_norm) < 1e-10);
  }

  SUBCASE("2x2 identity matrix Frobenius norm should be sqrt(2)") {
    cytnx::Tensor identity;
    tci::eye(ctx, 2, identity);

    auto norm_val = tci::norm(ctx, identity);
    double expected_norm = std::sqrt(2.0);

    // This will FAIL if norm is a placeholder returning 1.0
    CHECK(std::abs(norm_val - expected_norm) < 1e-10);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Diagonal Operations") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("Vector to diagonal matrix conversion") {
    // Create a vector [1, 2, 3]
    tci::shape_t<cytnx::Tensor> shape = {3};
    cytnx::Tensor vector;
    tci::zeros(ctx, shape, vector);

    tci::set_elem(ctx, vector, {0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, vector, {1}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, vector, {2}, cytnx::cytnx_complex128(3.0, 0.0));

    // Convert to diagonal matrix
    tci::diag(ctx, vector);

    // Should now be 3x3 matrix
    CHECK(tci::rank(ctx, vector) == 2);
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
    cytnx::Tensor identity;
    tci::eye(ctx, 3, identity);

    // Convert to diagonal vector
    tci::diag(ctx, identity);

    // Should now be rank-1 vector
    CHECK(tci::rank(ctx, identity) == 1);
    CHECK(tci::size(ctx, identity) == 3);

    // Check elements are [1, 1, 1]
    CHECK(std::abs(tci::real(tci::get_elem(ctx, identity, {0})) - 1.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, identity, {1})) - 1.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, identity, {2})) - 1.0) < 1e-10);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Trace Operations") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("Matrix trace calculation") {
    // Create a 3x3 matrix with specific diagonal values
    tci::shape_t<cytnx::Tensor> shape = {3, 3};
    cytnx::Tensor matrix;
    tci::zeros(ctx, shape, matrix);

    // Set diagonal to [2, 3, 4]
    tci::set_elem(ctx, matrix, {0, 0}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 1}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, matrix, {2, 2}, cytnx::cytnx_complex128(4.0, 0.0));

    // Set some off-diagonal elements
    tci::set_elem(ctx, matrix, {0, 1}, cytnx::cytnx_complex128(5.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 2}, cytnx::cytnx_complex128(6.0, 0.0));

    // Calculate trace (sum of diagonal elements)
    tci::bond_idx_pairs_t<cytnx::Tensor> pairs = {{0, 1}};
    tci::trace(ctx, matrix, pairs);

    // Should now be scalar with value 2+3+4 = 9
    CHECK(tci::rank(ctx, matrix) == 0);  // Scalar tensor
    CHECK(std::abs(tci::real(tci::get_elem(ctx, matrix, {})) - 9.0) < 1e-10);
  }

  SUBCASE("3D tensor trace calculation") {
    // Create a 2x3x2 tensor and trace over first and last dimensions
    tci::shape_t<cytnx::Tensor> shape = {2, 3, 2};
    cytnx::Tensor tensor3d;
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
    tci::bond_idx_pairs_t<cytnx::Tensor> pairs = {{0, 2}};
    cytnx::Tensor result;
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
    tci::shape_t<cytnx::Tensor> shape = {2, 2, 3, 2};
    cytnx::Tensor tensor4d;
    tci::zeros(ctx, shape, tensor4d);

    // Set diagonal elements T[i,j,k,i] where i=0,1, j=0,1, k=0,1,2
    for (size_t j = 0; j < 2; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        tci::set_elem(ctx, tensor4d, {0, j, k, 0}, cytnx::cytnx_complex128(1.0 + j + k, 0.0));
        tci::set_elem(ctx, tensor4d, {1, j, k, 1}, cytnx::cytnx_complex128(10.0 + j + k, 0.0));
      }
    }

    // Calculate trace over dimensions 0 and 3
    tci::bond_idx_pairs_t<cytnx::Tensor> pairs = {{0, 3}};
    cytnx::Tensor result;
    tci::trace(ctx, tensor4d, pairs, result);

    // Result should be 2D tensor with shape [2, 3]
    auto result_shape = tci::shape(ctx, result);
    CHECK(result_shape.size() == 2);
    CHECK(result_shape[0] == 2);
    CHECK(result_shape[1] == 3);

    // Check specific values: result[j,k] = T[0,j,k,0] + T[1,j,k,1]
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {0, 0})) - 11.0) < 1e-10);  // (1+0+0) + (10+0+0) = 11
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {0, 1})) - 13.0) < 1e-10);  // (1+0+1) + (10+0+1) = 13
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {1, 2})) - 17.0) < 1e-10);  // (1+1+2) + (10+1+2) = 17
  }

  SUBCASE("Complex tensor trace calculation") {
    // Test with complex numbers
    tci::shape_t<cytnx::Tensor> shape = {3, 2, 3};
    cytnx::Tensor tensor_complex;
    tci::zeros(ctx, shape, tensor_complex);

    // Set complex diagonal elements T[i,j,i]
    tci::set_elem(ctx, tensor_complex, {0, 0, 0}, cytnx::cytnx_complex128(1.0, 2.0));
    tci::set_elem(ctx, tensor_complex, {0, 1, 0}, cytnx::cytnx_complex128(2.0, -1.0));
    tci::set_elem(ctx, tensor_complex, {1, 0, 1}, cytnx::cytnx_complex128(-1.0, 3.0));
    tci::set_elem(ctx, tensor_complex, {1, 1, 1}, cytnx::cytnx_complex128(0.5, -2.5));
    tci::set_elem(ctx, tensor_complex, {2, 0, 2}, cytnx::cytnx_complex128(3.0, 1.0));
    tci::set_elem(ctx, tensor_complex, {2, 1, 2}, cytnx::cytnx_complex128(-0.5, 4.0));

    // Calculate trace over dimensions 0 and 2
    tci::bond_idx_pairs_t<cytnx::Tensor> pairs = {{0, 2}};
    cytnx::Tensor result;
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
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("Einstein notation contraction") {
    // Create two tensors for contraction
    tci::shape_t<cytnx::Tensor> shape_a = {2, 3};
    tci::shape_t<cytnx::Tensor> shape_b = {3, 2};

    cytnx::Tensor a, b, c;
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
    tci::shape_t<cytnx::Tensor> shape_a = {2, 3};
    tci::shape_t<cytnx::Tensor> shape_b = {3, 2};

    cytnx::Tensor a, b, c;
    tci::zeros(ctx, shape_a, a);
    tci::fill(ctx, shape_a, cytnx::cytnx_complex128(1.0, 0.0), a);

    tci::zeros(ctx, shape_b, b);
    tci::fill(ctx, shape_b, cytnx::cytnx_complex128(1.0, 0.0), b);

    // Test label-based contraction
    tci::List<tci::bond_label_t<cytnx::Tensor>> labs_a = {1, -1};
    tci::List<tci::bond_label_t<cytnx::Tensor>> labs_b = {-1, 2};
    tci::List<tci::bond_label_t<cytnx::Tensor>> labs_c = {1, 2};

    // TODO: Uncomment when contract is fully implemented
    // CHECK_THROWS_AS(tci::contract(ctx, a, labs_a, b, labs_b, c, labs_c), std::runtime_error);
    (void)c;  // Suppress unused variable warning
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Tensor Contraction") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("Matrix multiplication via contraction: ij,jk->ik") {
    // Create two 2x2 matrices for testing
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    cytnx::Tensor a, b, c;
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

  SUBCASE("Abnormal NCON: mixing positive and negative output labels") {
    // Create test tensors for abnormal NCON demonstration
    tci::shape_t<cytnx::Tensor> shape_a = {2, 3};
    tci::shape_t<cytnx::Tensor> shape_b = {3, 2};
    cytnx::Tensor a, b, c;
    tci::zeros(ctx, shape_a, a);
    tci::zeros(ctx, shape_b, b);

    // Fill with simple test values
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        tci::set_elem(ctx, a, {static_cast<tci::elem_coor_t<cytnx::Tensor>>(i), static_cast<tci::elem_coor_t<cytnx::Tensor>>(j)}, cytnx::cytnx_complex128(i * 3 + j + 1, 0.0));
      }
    }
    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 2; ++j) {
        tci::set_elem(ctx, b, {static_cast<tci::elem_coor_t<cytnx::Tensor>>(i), static_cast<tci::elem_coor_t<cytnx::Tensor>>(j)}, cytnx::cytnx_complex128(i * 2 + j + 1, 0.0));
      }
    }

    // Test abnormal NCON with mixed positive/negative output labels
    tci::List<tci::bond_label_t<cytnx::Tensor>> bd_labs_a = {1, 2};  // positive labels for tensor a
    tci::List<tci::bond_label_t<cytnx::Tensor>> bd_labs_b
        = {2, 3};  // mixed: 2 (contract), 3 (positive output)
    tci::List<tci::bond_label_t<cytnx::Tensor>> bd_labs_c
        = {-1, 3};  // ABNORMAL: mixing negative (-1) and positive (3)

    // This should handle abnormal NCON gracefully
    tci::contract(ctx, a, bd_labs_a, b, bd_labs_b, c, bd_labs_c);

    // Verify contraction occurred (shape should be 2x2)
    CHECK(c.shape().size() == 2);
    CHECK(c.shape()[0] == 2);
    CHECK(c.shape()[1] == 2);
  }

  SUBCASE("Vector dot product via contraction: i,i->") {
    // Create two vectors for dot product
    tci::shape_t<cytnx::Tensor> shape = {3};
    cytnx::Tensor a, b, c;
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
    CHECK(c.shape().size() == 1);  // Cytnx scalar result is [1] shape
    CHECK(c.shape()[0] == 1);      // Single element
    auto dot_result = tci::get_elem(ctx, c, {0});  // Access single element
    CHECK(std::abs(tci::real(dot_result) - 32.0) < 1e-10);
  }

  SUBCASE("Outer product via contraction: i,j->ij") {
    // Create two vectors for outer product
    tci::shape_t<cytnx::Tensor> shape_a = {2};
    tci::shape_t<cytnx::Tensor> shape_b = {3};
    cytnx::Tensor a, b, c;
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
    CHECK(c.shape().size() == 2);
    CHECK(c.shape()[0] == 2);
    CHECK(c.shape()[1] == 3);

    // Check specific values: c[0,0] = 1*3 = 3, c[1,2] = 2*5 = 10
    auto c00 = tci::get_elem(ctx, c, {0, 0});
    auto c12 = tci::get_elem(ctx, c, {1, 2});
    CHECK(std::abs(tci::real(c00) - 3.0) < 1e-10);
    CHECK(std::abs(tci::real(c12) - 10.0) < 1e-10);
  }

  tci::destroy_context(ctx);
}