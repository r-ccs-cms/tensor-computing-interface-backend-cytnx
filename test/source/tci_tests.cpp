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

TEST_CASE("TCI Size Bytes Calculation") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("Size bytes should be accurate") {
    tci::shape_t<cytnx::Tensor> shape = {2, 3};
    cytnx::Tensor tensor;
    tci::zeros(ctx, shape, tensor);

    auto bytes = tci::size_bytes(ctx, tensor);
    size_t expected_bytes = 6 * sizeof(cytnx::cytnx_complex128);  // 6 * 16 = 96 bytes

    // This will FAIL if size_bytes calculation is wrong
    CHECK(bytes == expected_bytes);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Tensor Equality") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("Different tensors should not be equal") {
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    cytnx::Tensor a, b;

    tci::zeros(ctx, shape, a);
    tci::zeros(ctx, shape, b);

    // Set different values
    tci::elem_coors_t<cytnx::Tensor> coord = {0, 0};
    tci::set_elem(ctx, a, coord, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, b, coord, cytnx::cytnx_complex128(2.0, 0.0));

    bool are_equal = tci::eq(ctx, a, b, cytnx::cytnx_complex128(1e-10, 0));

    // This will FAIL if eq is a placeholder that only checks shapes
    CHECK(are_equal == false);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI SVD Decomposition") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("SVD of 2x2 identity should produce correct singular values") {
    cytnx::Tensor identity;
    tci::eye(ctx, 2, identity);

    cytnx::Tensor u, v_dag;
    cytnx::Tensor s_diag;

    tci::svd(ctx, identity, 1, u, s_diag, v_dag);

    // For 2x2 identity matrix, singular values should be [1.0, 1.0]
    auto s_0 = tci::get_elem(ctx, s_diag, {0});
    auto s_1 = tci::get_elem(ctx, s_diag, {1});

    // Now SVD is properly implemented, not a placeholder
    CHECK(std::abs(tci::real(s_0) - 1.0) < 1e-10);
    CHECK(std::abs(tci::real(s_1) - 1.0) < 1e-10);

    // Verify s_diag is rank-1 (vector of singular values)
    CHECK(tci::rank(ctx, s_diag) == 1);

    // Verify s_diag has 2 elements (for 2x2 matrix)
    CHECK(tci::size(ctx, s_diag) == 2);
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

TEST_CASE("TCI Matrix Decomposition - QR") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("QR decomposition of 3x3 matrix") {
    // Create a 3x3 test matrix
    tci::shape_t<cytnx::Tensor> shape = {3, 3};
    cytnx::Tensor matrix;
    tci::zeros(ctx, shape, matrix);

    // Fill with test values
    tci::set_elem(ctx, matrix, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, matrix, {0, 1}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, matrix, {0, 2}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 0}, cytnx::cytnx_complex128(4.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 1}, cytnx::cytnx_complex128(5.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 2}, cytnx::cytnx_complex128(6.0, 0.0));
    tci::set_elem(ctx, matrix, {2, 0}, cytnx::cytnx_complex128(7.0, 0.0));
    tci::set_elem(ctx, matrix, {2, 1}, cytnx::cytnx_complex128(8.0, 0.0));
    tci::set_elem(ctx, matrix, {2, 2}, cytnx::cytnx_complex128(9.0, 0.0));

    cytnx::Tensor q, r;
    // TODO: Uncomment when QR is implemented
    // CHECK_THROWS_AS(tci::qr(ctx, matrix, 2, q, r), std::runtime_error);
    (void)q;
    (void)r;  // Suppress unused variable warnings
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Matrix Decomposition - LQ") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("LQ decomposition of 3x3 matrix") {
    // Create a 3x3 test matrix
    tci::shape_t<cytnx::Tensor> shape = {3, 3};
    cytnx::Tensor matrix;
    tci::zeros(ctx, shape, matrix);

    // Fill with test values
    tci::set_elem(ctx, matrix, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, matrix, {0, 1}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 0}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 1}, cytnx::cytnx_complex128(4.0, 0.0));
    tci::set_elem(ctx, matrix, {2, 0}, cytnx::cytnx_complex128(5.0, 0.0));
    tci::set_elem(ctx, matrix, {2, 1}, cytnx::cytnx_complex128(6.0, 0.0));

    cytnx::Tensor l, q;
    tci::lq(ctx, matrix, 2, l, q);

    CHECK(tci::rank(ctx, l) == 3);
    CHECK(tci::rank(ctx, q) == 1);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Matrix Decomposition - Truncated SVD") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("Truncated SVD with chi_max constraint") {
    // Create a 4x4 matrix for truncation testing
    tci::shape_t<cytnx::Tensor> shape = {4, 4};
    cytnx::Tensor matrix;
    tci::zeros(ctx, shape, matrix);

    // Create a matrix with known singular values
    for (int i = 0; i < 4; ++i) {
      tci::set_elem(ctx, matrix, {static_cast<tci::elem_coor_t<cytnx::Tensor>>(i), static_cast<tci::elem_coor_t<cytnx::Tensor>>(i)}, cytnx::cytnx_complex128(4.0 - i, 0.0));  // [4,3,2,1]
    }

    cytnx::Tensor u, v_dag;
    cytnx::Tensor s_diag;
    double trunc_err;

    // Test truncated SVD implementation
    CHECK_NOTHROW(tci::trunc_svd(ctx, matrix, 2, u, s_diag, v_dag, trunc_err, 2, 0.1));

    // Verify dimensions (should be truncated to at most chi_max=2)
    auto u_shape = tci::shape(ctx, u);
    auto s_shape = tci::shape(ctx, s_diag);
    auto v_dag_shape = tci::shape(ctx, v_dag);

    CHECK(u_shape.size() == 2);
    CHECK(v_dag_shape.size() == 2);
    CHECK(s_shape.size() == 1);

    // Check that the number of singular values doesn't exceed chi_max
    CHECK(s_shape[0] <= 2);

    // Verify that truncation error is non-negative
    CHECK(trunc_err >= 0.0);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Eigenvalue Problems") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("Eigenvalues of diagonal matrix") {
    // Create a 3x3 diagonal matrix
    cytnx::Tensor diagonal;
    tci::eye(ctx, 3, diagonal);

    // Scale diagonal elements to [1, 2, 3]
    tci::set_elem(ctx, diagonal, {1, 1}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, diagonal, {2, 2}, cytnx::cytnx_complex128(3.0, 0.0));

    cytnx::Tensor eigenvals;
    // This should fail with invalid argument (matrix must be square)
    CHECK_THROWS_AS(tci::eigvals(ctx, diagonal, 2, eigenvals), std::invalid_argument);
  }

  SUBCASE("Eigenvalues of symmetric matrix") {
    // Create a symmetric 2x2 matrix
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    cytnx::Tensor symmetric;
    tci::zeros(ctx, shape, symmetric);

    tci::set_elem(ctx, symmetric, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, symmetric, {0, 1}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, symmetric, {1, 0}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, symmetric, {1, 1}, cytnx::cytnx_complex128(3.0, 0.0));

    cytnx::Tensor eigenvals;
    // This should fail with invalid argument (matrix must be square)
    CHECK_THROWS_AS(tci::eigvalsh(ctx, symmetric, 2, eigenvals), std::invalid_argument);
  }

  SUBCASE("Eigenvalues and eigenvectors") {
    cytnx::Tensor matrix;
    tci::eye(ctx, 2, matrix);

    cytnx::Tensor eigenvals, eigenvecs;
    tci::eig(ctx, matrix, 1, eigenvals, eigenvecs);

    CHECK(tci::rank(ctx, eigenvals) == 1);
    CHECK(tci::size(ctx, eigenvals) == 2);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, eigenvals, {0})) - 1.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, eigenvals, {1})) - 1.0) < 1e-10);

    CHECK(tci::rank(ctx, eigenvecs) == 2);
    CHECK(tci::shape(ctx, eigenvecs)[0] == 2);
    CHECK(tci::shape(ctx, eigenvecs)[1] == 2);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, eigenvecs, {0, 0})) - 1.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, eigenvecs, {1, 1})) - 1.0) < 1e-10);
  }

  SUBCASE("Symmetric eigendecomposition") {
    cytnx::Tensor matrix;
    tci::eye(ctx, 2, matrix);

    cytnx::Tensor eigenvals, eigenvecs;
    tci::eigh(ctx, matrix, 1, eigenvals, eigenvecs);

    CHECK(tci::rank(ctx, eigenvals) == 1);
    CHECK(tci::size(ctx, eigenvals) == 2);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, eigenvals, {0})) - 1.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, eigenvals, {1})) - 1.0) < 1e-10);

    CHECK(tci::rank(ctx, eigenvecs) == 2);
    CHECK(tci::shape(ctx, eigenvecs)[0] == 2);
    CHECK(tci::shape(ctx, eigenvecs)[1] == 2);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, eigenvecs, {0, 0})) - 1.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, eigenvecs, {1, 1})) - 1.0) < 1e-10);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Advanced Linear Algebra") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("Matrix exponential") {
    // Create a small 2x2 matrix
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    cytnx::Tensor matrix;
    tci::zeros(ctx, shape, matrix);

    tci::set_elem(ctx, matrix, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 1}, cytnx::cytnx_complex128(2.0, 0.0));

    // Test matrix exponential (should work now)
    cytnx::Tensor result;
    CHECK_NOTHROW(tci::exp(ctx, matrix, 1, result));

    // Verify dimensions are preserved
    auto result_shape = tci::shape(ctx, result);
    CHECK(result_shape.size() == 2);
    CHECK(result_shape[0] == 2);
    CHECK(result_shape[1] == 2);

    // Test in-place version
    cytnx::Tensor matrix_copy;
    tci::copy(ctx, matrix, matrix_copy);
    CHECK_NOTHROW(tci::exp(ctx, matrix_copy, 1));

    // Results should be the same
    CHECK(tci::eq(ctx, result, matrix_copy, 1e-12));
  }

  SUBCASE("Matrix inverse") {
    // Create an invertible 2x2 matrix
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    cytnx::Tensor matrix;
    tci::zeros(ctx, shape, matrix);

    tci::set_elem(ctx, matrix, {0, 0}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, matrix, {0, 1}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 1}, cytnx::cytnx_complex128(1.0, 0.0));

    // TODO: Uncomment when inverse is implemented
    // CHECK_THROWS_AS(tci::inverse(ctx, matrix, 1), std::runtime_error);
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

TEST_CASE("TCI Truncated SVD") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("Truncated SVD with max bond dimension") {
    // Create a 4x4 matrix with known singular values
    tci::shape_t<cytnx::Tensor> shape = {4, 4};
    cytnx::Tensor matrix;
    tci::zeros(ctx, shape, matrix);

    // Fill diagonal with values [3, 2, 1, 0.1] for known singular values
    tci::set_elem(ctx, matrix, {0, 0}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 1}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, matrix, {2, 2}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, matrix, {3, 3}, cytnx::cytnx_complex128(0.1, 0.0));

    cytnx::Tensor u, v_dag;
    tci::real_ten_t<cytnx::Tensor> s_diag;
    tci::real_t<cytnx::Tensor> trunc_err;

    // Test truncated SVD with chi_max=2 and s_min=0.5
    CHECK_NOTHROW(tci::trunc_svd(ctx, matrix, 2, u, s_diag, v_dag, trunc_err, 2, 0.5));

    // Verify dimensions (should be truncated to at most chi_max=2)
    auto s_shape = tci::shape(ctx, s_diag);
    CHECK(s_shape.size() == 1);
    CHECK(s_shape[0] <= 2);

    // With s_min=0.5, values 0.1 should be truncated, so we expect 2 singular values (3.0, 2.0, 1.0 >= 0.5)
    // But limited by chi_max=2, so we should get 2 values
    CHECK(s_shape[0] <= 2);

    // Verify that truncation error represents information lost
    CHECK(trunc_err >= 0.0);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI QR Decomposition") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("QR decomposition of square matrix") {
    // Create a 3x3 matrix
    tci::shape_t<cytnx::Tensor> shape = {3, 3};
    cytnx::Tensor matrix;
    tci::zeros(ctx, shape, matrix);

    // Fill with test values
    tci::set_elem(ctx, matrix, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, matrix, {0, 1}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, matrix, {0, 2}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 0}, cytnx::cytnx_complex128(4.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 1}, cytnx::cytnx_complex128(5.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 2}, cytnx::cytnx_complex128(6.0, 0.0));
    tci::set_elem(ctx, matrix, {2, 0}, cytnx::cytnx_complex128(7.0, 0.0));
    tci::set_elem(ctx, matrix, {2, 1}, cytnx::cytnx_complex128(8.0, 0.0));
    tci::set_elem(ctx, matrix, {2, 2}, cytnx::cytnx_complex128(9.0, 0.0));

    cytnx::Tensor q, r;

    // Perform QR decomposition
    tci::qr(ctx, matrix, 1, q, r);

    // Q should be orthogonal, R should be upper triangular
    CHECK(q.shape().size() == 2);  // 3x3 matrix -> 2D tensors
    CHECK(r.shape().size() == 2);  // 3x3 matrix -> 2D tensors
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI LQ Decomposition") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("LQ decomposition of square matrix") {
    // Create a 3x3 matrix
    tci::shape_t<cytnx::Tensor> shape = {3, 3};
    cytnx::Tensor matrix;
    tci::zeros(ctx, shape, matrix);

    // Fill with test values
    tci::set_elem(ctx, matrix, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, matrix, {0, 1}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, matrix, {0, 2}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 0}, cytnx::cytnx_complex128(4.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 1}, cytnx::cytnx_complex128(5.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 2}, cytnx::cytnx_complex128(6.0, 0.0));
    tci::set_elem(ctx, matrix, {2, 0}, cytnx::cytnx_complex128(7.0, 0.0));
    tci::set_elem(ctx, matrix, {2, 1}, cytnx::cytnx_complex128(8.0, 0.0));
    tci::set_elem(ctx, matrix, {2, 2}, cytnx::cytnx_complex128(9.0, 0.0));

    cytnx::Tensor l, q;

    // Perform LQ decomposition
    tci::lq(ctx, matrix, 1, l, q);

    // L should be lower triangular, Q should be orthogonal
    CHECK(l.shape().size() == 2);  // 3x3 matrix -> 2D tensors
    CHECK(q.shape().size() == 2);  // 3x3 matrix -> 2D tensors
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI Eigenvalue Functions") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("General matrix eigenvalues") {
    // Create a 2x2 matrix
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    cytnx::Tensor matrix;
    tci::zeros(ctx, shape, matrix);

    // Fill with test values
    tci::set_elem(ctx, matrix, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, matrix, {0, 1}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 0}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 1}, cytnx::cytnx_complex128(4.0, 0.0));

    tci::cplx_ten_t<cytnx::Tensor> eigenvalues;

    // Perform eigenvalue calculation
    tci::eigvals(ctx, matrix, 1, eigenvalues);

    // Should have 2 eigenvalues
    CHECK(eigenvalues.shape()[0] == 2);
  }

  SUBCASE("Symmetric matrix eigenvalues") {
    // Create a symmetric 2x2 matrix
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    cytnx::Tensor matrix;
    tci::zeros(ctx, shape, matrix);

    // Fill with symmetric values
    tci::set_elem(ctx, matrix, {0, 0}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, matrix, {0, 1}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 1}, cytnx::cytnx_complex128(3.0, 0.0));

    tci::real_ten_t<cytnx::Tensor> eigenvalues;

    // Perform symmetric eigenvalue calculation
    tci::eigvalsh(ctx, matrix, 1, eigenvalues);

    // Should have 2 real eigenvalues
    CHECK(eigenvalues.shape()[0] == 2);
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

TEST_CASE("tci::for_each API compliance test") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("for_each modifying version - basic element modification") {
    // Create a test tensor with known values
    tci::shape_t<cytnx::Tensor> shape = {2, 3};
    cytnx::Tensor tensor;
    tci::zeros(ctx, shape, tensor);

    // Set initial values: [1, 2, 3, 4, 5, 6]
    tci::set_elem(ctx, tensor, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, tensor, {0, 1}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, tensor, {0, 2}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, tensor, {1, 0}, cytnx::cytnx_complex128(4.0, 0.0));
    tci::set_elem(ctx, tensor, {1, 1}, cytnx::cytnx_complex128(5.0, 0.0));
    tci::set_elem(ctx, tensor, {1, 2}, cytnx::cytnx_complex128(6.0, 0.0));

    // Define function to double each element
    std::function<void(tci::elem_t<cytnx::Tensor>&)> double_func =
        [](tci::elem_t<cytnx::Tensor>& elem) {
            auto real_part = tci::real(elem) * 2.0;
            auto imag_part = tci::imag(elem) * 2.0;
            elem = cytnx::cytnx_complex128(real_part, imag_part);
        };

    // Apply for_each to modify all elements
    CHECK_NOTHROW(tci::for_each(ctx, tensor, std::move(double_func)));

    // Verify all elements were doubled: [2, 4, 6, 8, 10, 12]
    CHECK(std::abs(tci::real(tci::get_elem(ctx, tensor, {0, 0})) - 2.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, tensor, {0, 1})) - 4.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, tensor, {0, 2})) - 6.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, tensor, {1, 0})) - 8.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, tensor, {1, 1})) - 10.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, tensor, {1, 2})) - 12.0) < 1e-10);
  }

  SUBCASE("for_each basic functionality verification") {
    // Test that for_each properly iterates through all elements in storage order
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    cytnx::Tensor tensor;
    tci::zeros(ctx, shape, tensor);

    // Set unique values to verify order
    tci::set_elem(ctx, tensor, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, tensor, {0, 1}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, tensor, {1, 0}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, tensor, {1, 1}, cytnx::cytnx_complex128(4.0, 0.0));

    // Count elements and verify access
    int count = 0;
    double sum = 0.0;
    std::function<void(tci::elem_t<cytnx::Tensor>&)> count_and_sum =
        [&count, &sum](tci::elem_t<cytnx::Tensor>& elem) {
          count++;
          sum += tci::real(elem);
        };

    CHECK_NOTHROW(tci::for_each(ctx, tensor, std::move(count_and_sum)));

    // Verify all elements were processed
    CHECK(count == 4);
    CHECK(std::abs(sum - 10.0) < 1e-10);  // 1 + 2 + 3 + 4 = 10
  }

  tci::destroy_context(ctx);
}

TEST_CASE("tci::linear_combine API compliance test") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("linear_combine uniform coefficients - basic tensor addition") {
    // Create test tensors with known values
    tci::shape_t<cytnx::Tensor> shape = {2, 2};

    cytnx::Tensor tensor_a, tensor_b, tensor_c, result;
    tci::zeros(ctx, shape, tensor_a);
    tci::zeros(ctx, shape, tensor_b);
    tci::zeros(ctx, shape, tensor_c);

    // Set tensor_a = [[1, 2], [3, 4]]
    tci::set_elem(ctx, tensor_a, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, tensor_a, {0, 1}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, tensor_a, {1, 0}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, tensor_a, {1, 1}, cytnx::cytnx_complex128(4.0, 0.0));

    // Set tensor_b = [[5, 6], [7, 8]]
    tci::set_elem(ctx, tensor_b, {0, 0}, cytnx::cytnx_complex128(5.0, 0.0));
    tci::set_elem(ctx, tensor_b, {0, 1}, cytnx::cytnx_complex128(6.0, 0.0));
    tci::set_elem(ctx, tensor_b, {1, 0}, cytnx::cytnx_complex128(7.0, 0.0));
    tci::set_elem(ctx, tensor_b, {1, 1}, cytnx::cytnx_complex128(8.0, 0.0));

    // Set tensor_c = [[1, 1], [1, 1]]
    tci::set_elem(ctx, tensor_c, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, tensor_c, {0, 1}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, tensor_c, {1, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, tensor_c, {1, 1}, cytnx::cytnx_complex128(1.0, 0.0));

    // Test uniform linear combination (simple addition)
    tci::List<cytnx::Tensor> tensors = {tensor_a, tensor_b, tensor_c};
    CHECK_NOTHROW(tci::linear_combine(ctx, tensors, result));

    // Expected result: [[7, 9], [11, 13]] = [[1+5+1, 2+6+1], [3+7+1, 4+8+1]]
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {0, 0})) - 7.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {0, 1})) - 9.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {1, 0})) - 11.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {1, 1})) - 13.0) < 1e-10);
  }

  SUBCASE("linear_combine with specified coefficients - weighted combination") {
    // Create test tensors
    tci::shape_t<cytnx::Tensor> shape = {2, 2};

    cytnx::Tensor tensor_a, tensor_b, result;
    tci::zeros(ctx, shape, tensor_a);
    tci::zeros(ctx, shape, tensor_b);

    // Set tensor_a = [[2, 4], [6, 8]]
    tci::set_elem(ctx, tensor_a, {0, 0}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, tensor_a, {0, 1}, cytnx::cytnx_complex128(4.0, 0.0));
    tci::set_elem(ctx, tensor_a, {1, 0}, cytnx::cytnx_complex128(6.0, 0.0));
    tci::set_elem(ctx, tensor_a, {1, 1}, cytnx::cytnx_complex128(8.0, 0.0));

    // Set tensor_b = [[1, 3], [5, 7]]
    tci::set_elem(ctx, tensor_b, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, tensor_b, {0, 1}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, tensor_b, {1, 0}, cytnx::cytnx_complex128(5.0, 0.0));
    tci::set_elem(ctx, tensor_b, {1, 1}, cytnx::cytnx_complex128(7.0, 0.0));

    // Test weighted linear combination: 0.5 * tensor_a + 2.0 * tensor_b
    tci::List<cytnx::Tensor> tensors = {tensor_a, tensor_b};
    tci::List<tci::elem_t<cytnx::Tensor>> coefficients = {
        cytnx::cytnx_complex128(0.5, 0.0),
        cytnx::cytnx_complex128(2.0, 0.0)
    };

    CHECK_NOTHROW(tci::linear_combine(ctx, tensors, coefficients, result));

    // Expected result: [[3, 8], [13, 18]] = [[0.5*2+2*1, 0.5*4+2*3], [0.5*6+2*5, 0.5*8+2*7]]
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {0, 0})) - 3.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {0, 1})) - 8.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {1, 0})) - 13.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {1, 1})) - 18.0) < 1e-10);
  }

  SUBCASE("linear_combine edge cases") {
    tci::shape_t<cytnx::Tensor> shape = {1, 1};
    cytnx::Tensor single_tensor, result;
    tci::zeros(ctx, shape, single_tensor);
    tci::set_elem(ctx, single_tensor, {0, 0}, cytnx::cytnx_complex128(5.0, 0.0));

    // Test single tensor uniform combination
    tci::List<cytnx::Tensor> single_list = {single_tensor};
    CHECK_NOTHROW(tci::linear_combine(ctx, single_list, result));
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {0, 0})) - 5.0) < 1e-10);

    // Test single tensor with coefficient
    tci::List<tci::elem_t<cytnx::Tensor>> single_coef = {cytnx::cytnx_complex128(3.0, 0.0)};
    CHECK_NOTHROW(tci::linear_combine(ctx, single_list, single_coef, result));
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {0, 0})) - 15.0) < 1e-10);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("tci::normalize API compliance test") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("normalize in-place version - basic normalization") {
    // Create test tensor with known values
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    cytnx::Tensor tensor;
    tci::zeros(ctx, shape, tensor);

    // Set tensor = [[3, 4], [0, 0]] with norm = 5
    tci::set_elem(ctx, tensor, {0, 0}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, tensor, {0, 1}, cytnx::cytnx_complex128(4.0, 0.0));
    tci::set_elem(ctx, tensor, {1, 0}, cytnx::cytnx_complex128(0.0, 0.0));
    tci::set_elem(ctx, tensor, {1, 1}, cytnx::cytnx_complex128(0.0, 0.0));

    // Normalize and check returned original norm
    auto original_norm = tci::normalize(ctx, tensor);

    // Verify original norm was 5 (3² + 4² = 9 + 16 = 25, √25 = 5)
    CHECK(std::abs(tci::real(original_norm) - 5.0) < 1e-10);
    CHECK(std::abs(tci::imag(original_norm)) < 1e-10);

    // Verify normalized tensor: [[3/5, 4/5], [0, 0]] = [[0.6, 0.8], [0, 0]]
    CHECK(std::abs(tci::real(tci::get_elem(ctx, tensor, {0, 0})) - 0.6) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, tensor, {0, 1})) - 0.8) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, tensor, {1, 0})) - 0.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, tensor, {1, 1})) - 0.0) < 1e-10);

    // Verify new norm is 1
    auto new_norm = tci::norm(ctx, tensor);
    CHECK(std::abs(new_norm - 1.0) < 1e-10);
  }

  SUBCASE("normalize out-of-place version - preserve original") {
    // Create test tensor with known values
    tci::shape_t<cytnx::Tensor> shape = {3, 1};
    cytnx::Tensor original, normalized;
    tci::zeros(ctx, shape, original);

    // Set original = [[2], [2], [1]] with norm = 3 (2² + 2² + 1² = 9, √9 = 3)
    tci::set_elem(ctx, original, {0, 0}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, original, {1, 0}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, original, {2, 0}, cytnx::cytnx_complex128(1.0, 0.0));

    // Normalize out-of-place
    auto original_norm = tci::normalize(ctx, original, normalized);

    // Verify original norm was 3
    CHECK(std::abs(tci::real(original_norm) - 3.0) < 1e-10);

    // Verify original tensor is unchanged
    CHECK(std::abs(tci::real(tci::get_elem(ctx, original, {0, 0})) - 2.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, original, {1, 0})) - 2.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, original, {2, 0})) - 1.0) < 1e-10);

    // Verify normalized tensor: [[2/3], [2/3], [1/3]]
    CHECK(std::abs(tci::real(tci::get_elem(ctx, normalized, {0, 0})) - (2.0/3.0)) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, normalized, {1, 0})) - (2.0/3.0)) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, normalized, {2, 0})) - (1.0/3.0)) < 1e-10);

    // Verify normalized tensor has norm 1
    auto new_norm = tci::norm(ctx, normalized);
    CHECK(std::abs(new_norm - 1.0) < 1e-10);
  }

  SUBCASE("normalize edge cases") {
    tci::shape_t<cytnx::Tensor> shape = {2, 2};

    // Test with single non-zero element
    cytnx::Tensor single_elem;
    tci::zeros(ctx, shape, single_elem);
    tci::set_elem(ctx, single_elem, {1, 1}, cytnx::cytnx_complex128(7.0, 0.0));

    auto norm1 = tci::normalize(ctx, single_elem);
    CHECK(std::abs(tci::real(norm1) - 7.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, single_elem, {1, 1})) - 1.0) < 1e-10);

    // Test with zero tensor (should not crash, original implementation handles this)
    cytnx::Tensor zero_tensor;
    tci::zeros(ctx, shape, zero_tensor);

    auto norm_zero = tci::normalize(ctx, zero_tensor);
    CHECK(std::abs(tci::real(norm_zero) - 0.0) < 1e-10);
    // Zero tensor should remain zero after normalization
    CHECK(std::abs(tci::real(tci::get_elem(ctx, zero_tensor, {0, 0})) - 0.0) < 1e-10);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, zero_tensor, {1, 1})) - 0.0) < 1e-10);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("tci::allocate API compliance test") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("allocate in-place version - basic allocation") {
    // Test basic allocation with different shapes
    tci::shape_t<cytnx::Tensor> shape = {2, 3};
    cytnx::Tensor tensor;

    // Allocate memory for tensor
    CHECK_NOTHROW(tci::allocate(ctx, shape, tensor));

    // Verify shape is correct
    CHECK(tci::rank(ctx, tensor) == 2);
    auto result_shape = tci::shape(ctx, tensor);
    CHECK(result_shape.size() == 2);
    CHECK(result_shape[0] == 2);
    CHECK(result_shape[1] == 3);

    // Verify size (total elements)
    CHECK(tci::size(ctx, tensor) == 6);  // 2 * 3 = 6 elements

    // Note: Memory is allocated but not initialized according to API spec
    // We can verify tensor is properly allocated by checking it's not empty
    CHECK(tensor.storage().size() == 6);
  }

  SUBCASE("allocate out-of-place version - return allocated tensor") {
    // Test out-of-place allocation
    tci::shape_t<cytnx::Tensor> shape = {3, 2, 2};

    // Allocate and return tensor
    cytnx::Tensor result;
    CHECK_NOTHROW(result = tci::allocate<cytnx::Tensor>(ctx, shape));

    // Verify 3D tensor properties
    CHECK(tci::rank(ctx, result) == 3);
    auto result_shape = tci::shape(ctx, result);
    CHECK(result_shape.size() == 3);
    CHECK(result_shape[0] == 3);
    CHECK(result_shape[1] == 2);
    CHECK(result_shape[2] == 2);

    // Verify total size
    CHECK(tci::size(ctx, result) == 12);  // 3 * 2 * 2 = 12 elements
  }

  SUBCASE("allocate different tensor shapes") {
    // Test 1D vector allocation
    tci::shape_t<cytnx::Tensor> shape_1d = {5};
    cytnx::Tensor vector;
    CHECK_NOTHROW(tci::allocate(ctx, shape_1d, vector));

    CHECK(tci::rank(ctx, vector) == 1);
    CHECK(tci::shape(ctx, vector)[0] == 5);
    CHECK(tci::size(ctx, vector) == 5);

    // Test scalar-like allocation
    tci::shape_t<cytnx::Tensor> shape_scalar = {1};
    cytnx::Tensor scalar;
    CHECK_NOTHROW(tci::allocate(ctx, shape_scalar, scalar));

    CHECK(tci::rank(ctx, scalar) == 1);
    CHECK(tci::shape(ctx, scalar)[0] == 1);
    CHECK(tci::size(ctx, scalar) == 1);

    // Test higher-dimensional tensor
    tci::shape_t<cytnx::Tensor> shape_4d = {2, 3, 2, 2};
    auto tensor_4d = tci::allocate<cytnx::Tensor>(ctx, shape_4d);

    CHECK(tci::rank(ctx, tensor_4d) == 4);
    auto shape_4d_result = tci::shape(ctx, tensor_4d);
    CHECK(shape_4d_result[0] == 2);
    CHECK(shape_4d_result[1] == 3);
    CHECK(shape_4d_result[2] == 2);
    CHECK(shape_4d_result[3] == 2);
    CHECK(tci::size(ctx, tensor_4d) == 24);  // 2*3*2*2 = 24
  }

  tci::destroy_context(ctx);
}

TEST_CASE("tci::save API compliance test") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  // Create test tensor with known values
  tci::shape_t<cytnx::Tensor> shape = {2, 3};
  auto test_tensor = tci::zeros<cytnx::Tensor>(ctx, shape);

    // Fill with known values for verification
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      tci::set_elem(ctx, test_tensor, {static_cast<unsigned long long>(i), static_cast<unsigned long long>(j)}, cytnx::cytnx_complex128(i * 3 + j + 1, 0.0));
    }
  }

    // Test 1: Save to file path (std::string)
    {
      std::string filename = "/tmp/claude/tci_save_test_string.cytn";

      // Ensure directory exists
      std::filesystem::create_directories("/tmp/claude");

      // Save tensor
      tci::save(ctx, test_tensor, filename);

      // Verify file exists
      CHECK(std::filesystem::exists(filename));

      // Load back and verify contents
      auto loaded = tci::load<cytnx::Tensor>(ctx, filename);
      CHECK(tci::shape(ctx, loaded) == tci::shape(ctx, test_tensor));
      CHECK(tci::size(ctx, loaded) == tci::size(ctx, test_tensor));

      // Cleanup
      std::filesystem::remove(filename);
    }

    // Test 2: Save to file path (const char*)
    {
      const char* filename = "/tmp/claude/tci_save_test_cstr.cytn";
      std::string filename_str(filename);

      tci::save(ctx, test_tensor, filename_str);

      CHECK(std::filesystem::exists(filename_str));

      auto loaded = tci::load<cytnx::Tensor>(ctx, filename_str);
      CHECK(tci::shape(ctx, loaded) == tci::shape(ctx, test_tensor));

      std::filesystem::remove(filename_str);
    }

    // Test 3: Save to filesystem::path
    {
      std::filesystem::path filepath = "/tmp/claude/tci_save_test_path.cytn";

      tci::save(ctx, test_tensor, filepath);

      CHECK(std::filesystem::exists(filepath));

      auto loaded = tci::load<cytnx::Tensor>(ctx, filepath);
      CHECK(tci::shape(ctx, loaded) == tci::shape(ctx, test_tensor));

      std::filesystem::remove(filepath);
    }

    // Test 4: Save to output stream (stringstream)
    {
      std::ostringstream oss;

      tci::save(ctx, test_tensor, oss);

      // Verify stream has content
      CHECK(!oss.str().empty());

      // Test loading from corresponding input stream
      std::istringstream iss(oss.str());
      auto loaded = tci::load<cytnx::Tensor>(ctx, iss);

      CHECK(tci::shape(ctx, loaded) == tci::shape(ctx, test_tensor));
      CHECK(tci::size(ctx, loaded) == tci::size(ctx, test_tensor));
    }

    // Test 5: Save to file output stream
    {
      std::string filename = "/tmp/claude/tci_save_test_ofstream.cytn";

      {
        std::ofstream ofs(filename, std::ios::binary);
        tci::save(ctx, test_tensor, ofs);
      }  // File automatically closed

      CHECK(std::filesystem::exists(filename));

      auto loaded = tci::load<cytnx::Tensor>(ctx, filename);
      CHECK(tci::shape(ctx, loaded) == tci::shape(ctx, test_tensor));

      std::filesystem::remove(filename);
    }

    // Test 6: Save with automatic directory creation
    {
      std::string nested_path = "/tmp/claude/nested/deep/tci_save_test_auto_dir.cytn";

      tci::save(ctx, test_tensor, nested_path);

      CHECK(std::filesystem::exists(nested_path));

      auto loaded = tci::load<cytnx::Tensor>(ctx, nested_path);
      CHECK(tci::shape(ctx, loaded) == tci::shape(ctx, test_tensor));

      // Cleanup nested directories
      std::filesystem::remove_all("/tmp/claude/nested");
    }

    // Test 7: Save large tensor to verify performance
    {
      tci::shape_t<cytnx::Tensor> large_shape = {10, 10, 10};
      auto large_tensor = tci::zeros<cytnx::Tensor>(ctx, large_shape);
      // Fill with ones
      for (int i = 0; i < 10; ++i) {
        for (int j = 0; j < 10; ++j) {
          for (int k = 0; k < 10; ++k) {
            tci::set_elem(ctx, large_tensor, {static_cast<unsigned long long>(i), static_cast<unsigned long long>(j), static_cast<unsigned long long>(k)}, cytnx::cytnx_complex128(1.0, 0.0));
          }
        }
      }

      std::string filename = "/tmp/claude/tci_save_test_large.cytn";

      tci::save(ctx, large_tensor, filename);

      CHECK(std::filesystem::exists(filename));

      auto loaded = tci::load<cytnx::Tensor>(ctx, filename);
      CHECK(tci::shape(ctx, loaded) == tci::shape(ctx, large_tensor));
      CHECK(tci::size(ctx, loaded) == 1000);  // 10*10*10

      std::filesystem::remove(filename);
    }

    // Test 8: Save different data types (complex tensor)
    {
      auto complex_tensor = tci::to_cplx(ctx, test_tensor);

      std::string filename = "/tmp/claude/tci_save_test_complex.cytn";

      tci::save(ctx, complex_tensor, filename);

      CHECK(std::filesystem::exists(filename));

      auto loaded = tci::load<cytnx::Tensor>(ctx, filename);
      CHECK(tci::shape(ctx, loaded) == tci::shape(ctx, complex_tensor));
      CHECK(loaded.dtype() == complex_tensor.dtype());

      std::filesystem::remove(filename);
    }

  tci::destroy_context(ctx);
}

TEST_CASE("tci::load API compliance test") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  // Create reference tensor for comparison
  tci::shape_t<cytnx::Tensor> shape = {2, 3};
  auto reference_tensor = tci::zeros<cytnx::Tensor>(ctx, shape);

  // Fill with known values
  for (int i = 0; i < 2; ++i) {
    for (int j = 0; j < 3; ++j) {
      tci::set_elem(ctx, reference_tensor, {static_cast<unsigned long long>(i), static_cast<unsigned long long>(j)}, cytnx::cytnx_complex128(i * 3 + j + 1, 0.0));
    }
  }

  // First save the reference tensor to use in load tests
  std::string reference_filename = "/tmp/claude/tci_load_reference.cytn";
  std::filesystem::create_directories("/tmp/claude");
  tci::save(ctx, reference_tensor, reference_filename);

  // Test 1: Load from file path (std::string) - in-place version
  {
    cytnx::Tensor loaded_tensor;

    tci::load(ctx, reference_filename, loaded_tensor);

    CHECK(tci::rank(ctx, loaded_tensor) == tci::rank(ctx, reference_tensor));
    CHECK(tci::size(ctx, loaded_tensor) == tci::size(ctx, reference_tensor));
    CHECK(tci::shape(ctx, loaded_tensor) == tci::shape(ctx, reference_tensor));

    // Verify values match
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        auto expected = tci::get_elem(ctx, reference_tensor, {static_cast<unsigned long long>(i), static_cast<unsigned long long>(j)});
        auto actual = tci::get_elem(ctx, loaded_tensor, {static_cast<unsigned long long>(i), static_cast<unsigned long long>(j)});
        CHECK(std::abs(tci::real(expected) - tci::real(actual)) < 1e-10);
      }
    }
  }

  // Test 2: Load from file path (std::string) - out-of-place version
  {
    auto loaded_tensor = tci::load<cytnx::Tensor>(ctx, reference_filename);

    CHECK(tci::rank(ctx, loaded_tensor) == tci::rank(ctx, reference_tensor));
    CHECK(tci::size(ctx, loaded_tensor) == tci::size(ctx, reference_tensor));
    CHECK(tci::shape(ctx, loaded_tensor) == tci::shape(ctx, reference_tensor));

    // Verify first and last elements
    auto expected_first = tci::get_elem(ctx, reference_tensor, {0ULL, 0ULL});
    auto actual_first = tci::get_elem(ctx, loaded_tensor, {0ULL, 0ULL});
    CHECK(std::abs(tci::real(expected_first) - tci::real(actual_first)) < 1e-10);

    auto expected_last = tci::get_elem(ctx, reference_tensor, {1ULL, 2ULL});
    auto actual_last = tci::get_elem(ctx, loaded_tensor, {1ULL, 2ULL});
    CHECK(std::abs(tci::real(expected_last) - tci::real(actual_last)) < 1e-10);
  }

  // Test 3: Load from filesystem::path - in-place version
  {
    std::filesystem::path filepath = reference_filename;
    cytnx::Tensor loaded_tensor;

    tci::load(ctx, filepath, loaded_tensor);

    CHECK(tci::shape(ctx, loaded_tensor) == tci::shape(ctx, reference_tensor));
    CHECK(tci::size(ctx, loaded_tensor) == tci::size(ctx, reference_tensor));
  }

  // Test 4: Load from filesystem::path - out-of-place version
  {
    std::filesystem::path filepath = reference_filename;
    auto loaded_tensor = tci::load<cytnx::Tensor>(ctx, filepath);

    CHECK(tci::shape(ctx, loaded_tensor) == tci::shape(ctx, reference_tensor));
  }

  // Test 5: Load from input stream (ifstream) - in-place version
  {
    std::ifstream ifs(reference_filename, std::ios::binary);
    CHECK(ifs.is_open());

    cytnx::Tensor loaded_tensor;
    tci::load(ctx, ifs, loaded_tensor);

    CHECK(tci::rank(ctx, loaded_tensor) == tci::rank(ctx, reference_tensor));
    CHECK(tci::shape(ctx, loaded_tensor) == tci::shape(ctx, reference_tensor));
  }

  // Test 6: Load from input stream (ifstream) - out-of-place version
  {
    std::ifstream ifs(reference_filename, std::ios::binary);
    CHECK(ifs.is_open());

    auto loaded_tensor = tci::load<cytnx::Tensor>(ctx, ifs);

    CHECK(tci::shape(ctx, loaded_tensor) == tci::shape(ctx, reference_tensor));
  }

  // Test 7: Load from stringstream (in-place)
  {
    // First save to stringstream
    std::ostringstream oss;
    tci::save(ctx, reference_tensor, oss);

    // Then load from stringstream
    std::istringstream iss(oss.str());
    cytnx::Tensor loaded_tensor;

    tci::load(ctx, iss, loaded_tensor);

    CHECK(tci::rank(ctx, loaded_tensor) == tci::rank(ctx, reference_tensor));
    CHECK(tci::shape(ctx, loaded_tensor) == tci::shape(ctx, reference_tensor));

    // Verify some values
    auto expected = tci::get_elem(ctx, reference_tensor, {0ULL, 1ULL});
    auto actual = tci::get_elem(ctx, loaded_tensor, {0ULL, 1ULL});
    CHECK(std::abs(tci::real(expected) - tci::real(actual)) < 1e-10);
  }

  // Test 8: Load from stringstream (out-of-place)
  {
    std::ostringstream oss;
    tci::save(ctx, reference_tensor, oss);

    std::istringstream iss(oss.str());
    auto loaded_tensor = tci::load<cytnx::Tensor>(ctx, iss);

    CHECK(tci::shape(ctx, loaded_tensor) == tci::shape(ctx, reference_tensor));
  }

  // Test 9: Load complex tensor
  {
    auto complex_tensor = tci::to_cplx(ctx, reference_tensor);
    std::string complex_filename = "/tmp/claude/tci_load_complex.cytn";

    tci::save(ctx, complex_tensor, complex_filename);

    auto loaded_complex = tci::load<cytnx::Tensor>(ctx, complex_filename);

    CHECK(tci::shape(ctx, loaded_complex) == tci::shape(ctx, complex_tensor));
    CHECK(loaded_complex.dtype() == complex_tensor.dtype());

    std::filesystem::remove(complex_filename);
  }

  // Test 10: Load large tensor
  {
    tci::shape_t<cytnx::Tensor> large_shape = {5, 4, 3};
    auto large_tensor = tci::zeros<cytnx::Tensor>(ctx, large_shape);

    // Set some known values
    tci::set_elem(ctx, large_tensor, {0ULL, 0ULL, 0ULL}, cytnx::cytnx_complex128(100.0, 0.0));
    tci::set_elem(ctx, large_tensor, {4ULL, 3ULL, 2ULL}, cytnx::cytnx_complex128(200.0, 0.0));

    std::string large_filename = "/tmp/claude/tci_load_large.cytn";
    tci::save(ctx, large_tensor, large_filename);

    auto loaded_large = tci::load<cytnx::Tensor>(ctx, large_filename);

    CHECK(tci::shape(ctx, loaded_large) == tci::shape(ctx, large_tensor));
    CHECK(tci::size(ctx, loaded_large) == 60);  // 5*4*3

    // Verify specific values
    auto val1 = tci::get_elem(ctx, loaded_large, {0ULL, 0ULL, 0ULL});
    auto val2 = tci::get_elem(ctx, loaded_large, {4ULL, 3ULL, 2ULL});
    CHECK(std::abs(tci::real(val1) - 100.0) < 1e-10);
    CHECK(std::abs(tci::real(val2) - 200.0) < 1e-10);

    std::filesystem::remove(large_filename);
  }

  // Test 11: Error handling - non-existent file
  {
    std::string nonexistent = "/tmp/claude/does_not_exist.cytn";
    cytnx::Tensor error_tensor;

    CHECK_THROWS(tci::load(ctx, nonexistent, error_tensor));
  }

  // Cleanup
  std::filesystem::remove(reference_filename);

  tci::destroy_context(ctx);
}

TEST_CASE("tci::clear API compliance test") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  // Test 1: Clear a tensor with existing data
  {
    tci::shape_t<cytnx::Tensor> shape = {2, 3};
    auto test_tensor = tci::zeros<cytnx::Tensor>(ctx, shape);

    // Fill with known values
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        tci::set_elem(ctx, test_tensor, {static_cast<unsigned long long>(i), static_cast<unsigned long long>(j)}, cytnx::cytnx_complex128(i * 3 + j + 1, 0.0));
      }
    }

    // Verify tensor has data before clearing
    CHECK(tci::size(ctx, test_tensor) == 6);
    CHECK(tci::rank(ctx, test_tensor) == 2);

    // Clear the tensor
    tci::clear(ctx, test_tensor);

    // Verify tensor is cleared (should be in uninitialized state)
    // After clearing, the tensor should be in a default state
    // Check that it's no longer the same as before
    bool is_cleared = true;
    try {
      // Attempting operations on cleared tensor may throw or return default values
      auto size_after = tci::size(ctx, test_tensor);
      auto rank_after = tci::rank(ctx, test_tensor);

      // If we can access these, they should be different from the original
      is_cleared = (size_after == 0 || rank_after == 0);
    } catch (...) {
      // If operations throw, the tensor is properly cleared
      is_cleared = true;
    }
    CHECK(is_cleared);
  }

  // Test 2: Clear an already empty tensor
  {
    cytnx::Tensor empty_tensor;

    // Clear an already empty tensor (should not crash)
    CHECK_NOTHROW(tci::clear(ctx, empty_tensor));
  }

  // Test 3: Clear a large tensor
  {
    tci::shape_t<cytnx::Tensor> large_shape = {10, 10, 5};
    auto large_tensor = tci::zeros<cytnx::Tensor>(ctx, large_shape);

    // Verify it has data
    CHECK(tci::size(ctx, large_tensor) == 500);  // 10*10*5

    // Clear the tensor
    tci::clear(ctx, large_tensor);

    // Verify clearing worked
    // After clearing, the tensor should be in a default uninitialized state
    // We check this by trying to verify if it's different from a valid tensor
    bool large_cleared = false;
    try {
      // A cleared tensor may have undefined behavior for size operations
      // but should not crash. The key test is that it's different from before
      auto size_after = tci::size(ctx, large_tensor);
      auto rank_after = tci::rank(ctx, large_tensor);

      // An empty/cleared tensor typically has size 0 or rank 0
      large_cleared = (size_after == 0 || rank_after == 0 || size_after != 500);
    } catch (...) {
      // If operations throw, the tensor is properly cleared
      large_cleared = true;
    }
    CHECK(large_cleared);
  }

  // Test 4: Clear a complex tensor
  {
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    auto complex_tensor = tci::zeros<cytnx::Tensor>(ctx, shape);

    // Convert to complex (though zeros already creates complex type)
    auto cplx_tensor = tci::to_cplx(ctx, complex_tensor);

    // Set some complex values
    tci::set_elem(ctx, cplx_tensor, {0ULL, 0ULL}, cytnx::cytnx_complex128(1.0, 2.0));
    tci::set_elem(ctx, cplx_tensor, {1ULL, 1ULL}, cytnx::cytnx_complex128(3.0, 4.0));

    // Verify it has the expected data type and size
    CHECK(tci::size(ctx, cplx_tensor) == 4);

    // Clear the tensor
    tci::clear(ctx, cplx_tensor);

    // Verify clearing worked
    bool cplx_cleared = false;
    try {
      auto size_after = tci::size(ctx, cplx_tensor);
      auto rank_after = tci::rank(ctx, cplx_tensor);

      // Check if tensor state changed after clearing
      cplx_cleared = (size_after == 0 || rank_after == 0 || size_after != 4);
    } catch (...) {
      cplx_cleared = true;
    }
    CHECK(cplx_cleared);
  }

  // Test 5: Clear tensor created by different construction methods
  {
    // Test clearing a tensor created with allocate
    tci::shape_t<cytnx::Tensor> shape = {3, 3};
    auto allocated_tensor = tci::allocate<cytnx::Tensor>(ctx, shape);

    CHECK_NOTHROW(tci::clear(ctx, allocated_tensor));

    // Test clearing a tensor created with eye
    auto eye_tensor = tci::eye<cytnx::Tensor>(ctx, 3);

    CHECK_NOTHROW(tci::clear(ctx, eye_tensor));

    // Test clearing a copied tensor
    auto original = tci::zeros<cytnx::Tensor>(ctx, shape);
    auto copied = tci::copy<cytnx::Tensor>(ctx, original);

    CHECK_NOTHROW(tci::clear(ctx, copied));
  }

  // Test 6: Multiple clears on same tensor
  {
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    auto test_tensor = tci::zeros<cytnx::Tensor>(ctx, shape);

    // Clear multiple times (should not crash)
    CHECK_NOTHROW(tci::clear(ctx, test_tensor));
    CHECK_NOTHROW(tci::clear(ctx, test_tensor));
    CHECK_NOTHROW(tci::clear(ctx, test_tensor));
  }

  // Test 7: Clear then reallocate
  {
    tci::shape_t<cytnx::Tensor> shape = {2, 3};
    auto test_tensor = tci::zeros<cytnx::Tensor>(ctx, shape);

    // Set some values
    tci::set_elem(ctx, test_tensor, {0ULL, 0ULL}, cytnx::cytnx_complex128(42.0, 0.0));

    // Clear the tensor
    tci::clear(ctx, test_tensor);

    // Reallocate with different shape
    tci::shape_t<cytnx::Tensor> new_shape = {3, 2};
    tci::zeros(ctx, new_shape, test_tensor);

    // Verify new tensor works properly
    CHECK(tci::size(ctx, test_tensor) == 6);
    CHECK(tci::rank(ctx, test_tensor) == 2);

    auto shape_result = tci::shape(ctx, test_tensor);
    CHECK(shape_result[0] == 3);
    CHECK(shape_result[1] == 2);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("tci::move API compliance test") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  // Test 1: In-place move with data preservation
  {
    // Create source tensor with known data
    tci::shape_t<cytnx::Tensor> shape = {2, 3};
    auto source_tensor = tci::zeros<cytnx::Tensor>(ctx, shape);

    // Fill with known values
    for (int i = 0; i < 2; ++i) {
      for (int j = 0; j < 3; ++j) {
        tci::set_elem(ctx, source_tensor, {static_cast<unsigned long long>(i), static_cast<unsigned long long>(j)}, cytnx::cytnx_complex128(i * 3 + j + 1, 0.0));
      }
    }

    // Store original values for verification
    auto orig_val_00 = tci::get_elem(ctx, source_tensor, {0ULL, 0ULL});
    auto orig_val_12 = tci::get_elem(ctx, source_tensor, {1ULL, 2ULL});

    // Create destination tensor
    cytnx::Tensor dest_tensor;

    // Perform in-place move
    tci::move(ctx, source_tensor, dest_tensor);

    // Verify destination has the moved data
    CHECK(tci::size(ctx, dest_tensor) == 6);
    CHECK(tci::rank(ctx, dest_tensor) == 2);
    CHECK(tci::shape(ctx, dest_tensor) == shape);

    // Verify data integrity
    auto moved_val_00 = tci::get_elem(ctx, dest_tensor, {0ULL, 0ULL});
    auto moved_val_12 = tci::get_elem(ctx, dest_tensor, {1ULL, 2ULL});
    CHECK(std::abs(tci::real(moved_val_00) - tci::real(orig_val_00)) < 1e-10);
    CHECK(std::abs(tci::real(moved_val_12) - tci::real(orig_val_12)) < 1e-10);

    // Verify source is cleared (moved from)
    bool source_cleared = false;
    try {
      auto size_after = tci::size(ctx, source_tensor);
      auto rank_after = tci::rank(ctx, source_tensor);
      source_cleared = (size_after == 0 || rank_after == 0 || size_after != 6);
    } catch (...) {
      source_cleared = true;
    }
    CHECK(source_cleared);
  }

  // Test 2: Out-of-place move with data preservation
  {
    // Create source tensor
    tci::shape_t<cytnx::Tensor> shape = {3, 2};
    auto source_tensor = tci::zeros<cytnx::Tensor>(ctx, shape);

    // Set specific values
    tci::set_elem(ctx, source_tensor, {0ULL, 0ULL}, cytnx::cytnx_complex128(10.0, 5.0));
    tci::set_elem(ctx, source_tensor, {2ULL, 1ULL}, cytnx::cytnx_complex128(20.0, -3.0));

    // Store original values
    auto orig_val_00 = tci::get_elem(ctx, source_tensor, {0ULL, 0ULL});
    auto orig_val_21 = tci::get_elem(ctx, source_tensor, {2ULL, 1ULL});

    // Perform out-of-place move
    auto moved_tensor = tci::move<cytnx::Tensor>(ctx, source_tensor);

    // Verify moved tensor has correct properties
    CHECK(tci::size(ctx, moved_tensor) == 6);
    CHECK(tci::rank(ctx, moved_tensor) == 2);
    CHECK(tci::shape(ctx, moved_tensor) == shape);

    // Verify data integrity
    auto moved_val_00 = tci::get_elem(ctx, moved_tensor, {0ULL, 0ULL});
    auto moved_val_21 = tci::get_elem(ctx, moved_tensor, {2ULL, 1ULL});
    CHECK(std::abs(tci::real(moved_val_00) - tci::real(orig_val_00)) < 1e-10);
    CHECK(std::abs(tci::imag(moved_val_00) - tci::imag(orig_val_00)) < 1e-10);
    CHECK(std::abs(tci::real(moved_val_21) - tci::real(orig_val_21)) < 1e-10);
    CHECK(std::abs(tci::imag(moved_val_21) - tci::imag(orig_val_21)) < 1e-10);

    // Verify source is cleared
    bool source_cleared = false;
    try {
      auto size_after = tci::size(ctx, source_tensor);
      auto rank_after = tci::rank(ctx, source_tensor);
      source_cleared = (size_after == 0 || rank_after == 0 || size_after != 6);
    } catch (...) {
      source_cleared = true;
    }
    CHECK(source_cleared);
  }

  // Test 3: Move large tensor (performance test)
  {
    tci::shape_t<cytnx::Tensor> large_shape = {10, 10, 5};
    auto large_tensor = tci::zeros<cytnx::Tensor>(ctx, large_shape);

    // Set some values
    tci::set_elem(ctx, large_tensor, {0ULL, 0ULL, 0ULL}, cytnx::cytnx_complex128(100.0, 0.0));
    tci::set_elem(ctx, large_tensor, {9ULL, 9ULL, 4ULL}, cytnx::cytnx_complex128(999.0, 0.0));

    cytnx::Tensor dest_large;

    // Perform move
    CHECK_NOTHROW(tci::move(ctx, large_tensor, dest_large));

    // Verify moved tensor
    CHECK(tci::size(ctx, dest_large) == 500);  // 10*10*5

    // Verify specific values
    auto val1 = tci::get_elem(ctx, dest_large, {0ULL, 0ULL, 0ULL});
    auto val2 = tci::get_elem(ctx, dest_large, {9ULL, 9ULL, 4ULL});
    CHECK(std::abs(tci::real(val1) - 100.0) < 1e-10);
    CHECK(std::abs(tci::real(val2) - 999.0) < 1e-10);
  }

  // Test 4: Move complex tensor preserving type
  {
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    auto complex_tensor = tci::zeros<cytnx::Tensor>(ctx, shape);

    // Set complex values
    tci::set_elem(ctx, complex_tensor, {0ULL, 0ULL}, cytnx::cytnx_complex128(1.0, 2.0));
    tci::set_elem(ctx, complex_tensor, {1ULL, 1ULL}, cytnx::cytnx_complex128(-3.0, 4.0));

    auto moved = tci::move<cytnx::Tensor>(ctx, complex_tensor);

    // Verify type preservation
    CHECK(moved.dtype() == cytnx::Type.ComplexDouble);

    // Verify complex values
    auto val1 = tci::get_elem(ctx, moved, {0ULL, 0ULL});
    auto val2 = tci::get_elem(ctx, moved, {1ULL, 1ULL});
    CHECK(std::abs(tci::real(val1) - 1.0) < 1e-10);
    CHECK(std::abs(tci::imag(val1) - 2.0) < 1e-10);
    CHECK(std::abs(tci::real(val2) + 3.0) < 1e-10);  // -3.0
    CHECK(std::abs(tci::imag(val2) - 4.0) < 1e-10);
  }

  // Test 5: Move tensor created by different construction methods
  {
    // Move an eye tensor
    auto eye_tensor = tci::eye<cytnx::Tensor>(ctx, 3);
    auto moved_eye = tci::move<cytnx::Tensor>(ctx, eye_tensor);

    CHECK(tci::size(ctx, moved_eye) == 9);
    CHECK(tci::rank(ctx, moved_eye) == 2);

    // Verify diagonal elements are 1
    auto diag00 = tci::get_elem(ctx, moved_eye, {0ULL, 0ULL});
    auto diag11 = tci::get_elem(ctx, moved_eye, {1ULL, 1ULL});
    auto diag22 = tci::get_elem(ctx, moved_eye, {2ULL, 2ULL});
    CHECK(std::abs(tci::real(diag00) - 1.0) < 1e-10);
    CHECK(std::abs(tci::real(diag11) - 1.0) < 1e-10);
    CHECK(std::abs(tci::real(diag22) - 1.0) < 1e-10);

    // Verify off-diagonal elements are 0
    auto off_diag01 = tci::get_elem(ctx, moved_eye, {0ULL, 1ULL});
    CHECK(std::abs(tci::real(off_diag01)) < 1e-10);
  }

  // Test 6: Move tensor to already occupied destination
  {
    // Create two tensors with different data
    tci::shape_t<cytnx::Tensor> shape1 = {2, 2};
    tci::shape_t<cytnx::Tensor> shape2 = {3, 3};

    auto tensor1 = tci::zeros<cytnx::Tensor>(ctx, shape1);
    auto tensor2 = tci::zeros<cytnx::Tensor>(ctx, shape2);

    tci::set_elem(ctx, tensor1, {0ULL, 0ULL}, cytnx::cytnx_complex128(42.0, 0.0));
    tci::set_elem(ctx, tensor2, {0ULL, 0ULL}, cytnx::cytnx_complex128(99.0, 0.0));

    // Move tensor1 into tensor2 (should replace tensor2)
    tci::move(ctx, tensor1, tensor2);

    // Verify tensor2 now has tensor1's properties and data
    CHECK(tci::size(ctx, tensor2) == 4);  // 2*2, not 3*3
    CHECK(tci::rank(ctx, tensor2) == 2);

    auto val = tci::get_elem(ctx, tensor2, {0ULL, 0ULL});
    CHECK(std::abs(tci::real(val) - 42.0) < 1e-10);

    // tensor1 should be cleared
    bool tensor1_cleared = false;
    try {
      auto size_after = tci::size(ctx, tensor1);
      tensor1_cleared = (size_after == 0 || size_after != 4);
    } catch (...) {
      tensor1_cleared = true;
    }
    CHECK(tensor1_cleared);
  }

  // Test 7: Move empty/uninitialized tensor
  {
    cytnx::Tensor empty_tensor;
    cytnx::Tensor dest_tensor;

    // Move empty tensor should not crash
    CHECK_NOTHROW(tci::move(ctx, empty_tensor, dest_tensor));
  }

  // Test 8: Self-assignment protection (though not recommended)
  {
    auto test_tensor = tci::zeros<cytnx::Tensor>(ctx, {2, 2});
    tci::set_elem(ctx, test_tensor, {0ULL, 0ULL}, cytnx::cytnx_complex128(123.0, 0.0));

    // Self-move should not crash (though behavior is undefined)
    CHECK_NOTHROW(tci::move(ctx, test_tensor, test_tensor));
  }

  tci::destroy_context(ctx);
}

TEST_CASE("tci::scale API functionality") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  // Test 1: In-place scaling with positive factor
  {
    cytnx::Tensor tensor;
    tci::fill(ctx, {2, 3}, cytnx::cytnx_complex128(4.0, 2.0), tensor);
    tci::elem_t<cytnx::Tensor> scale_factor = cytnx::cytnx_complex128(2.5, 0.0);

    CHECK_NOTHROW(tci::scale(ctx, tensor, scale_factor));

    auto result = tci::get_elem(ctx, tensor, {0ULL, 0ULL});
    auto expected = cytnx::cytnx_complex128(10.0, 5.0);  // (4.0 + 2.0i) * 2.5
    CHECK(std::abs(tci::real(result) - tci::real(expected)) < 1e-10);
    CHECK(std::abs(tci::imag(result) - tci::imag(expected)) < 1e-10);
  }

  // Test 2: In-place scaling with negative factor
  {
    cytnx::Tensor tensor;
    tci::fill(ctx, {2, 2}, cytnx::cytnx_complex128(3.0, -1.0), tensor);
    tci::elem_t<cytnx::Tensor> scale_factor = cytnx::cytnx_complex128(-1.5, 0.0);

    CHECK_NOTHROW(tci::scale(ctx, tensor, scale_factor));

    auto result = tci::get_elem(ctx, tensor, {1ULL, 1ULL});
    auto expected = cytnx::cytnx_complex128(-4.5, 1.5);  // (3.0 - 1.0i) * -1.5
    CHECK(std::abs(tci::real(result) - tci::real(expected)) < 1e-10);
    CHECK(std::abs(tci::imag(result) - tci::imag(expected)) < 1e-10);
  }

  // Test 3: In-place scaling with complex factor
  {
    cytnx::Tensor tensor;
    tci::fill(ctx, {1, 2}, cytnx::cytnx_complex128(2.0, 3.0), tensor);
    tci::elem_t<cytnx::Tensor> scale_factor = cytnx::cytnx_complex128(1.0, 2.0);

    CHECK_NOTHROW(tci::scale(ctx, tensor, scale_factor));

    auto result = tci::get_elem(ctx, tensor, {0ULL, 0ULL});
    // (2.0 + 3.0i) * (1.0 + 2.0i) = 2.0 + 4.0i + 3.0i + 6.0i² = 2.0 + 7.0i - 6.0 = -4.0 + 7.0i
    auto expected = cytnx::cytnx_complex128(-4.0, 7.0);
    CHECK(std::abs(tci::real(result) - tci::real(expected)) < 1e-10);
    CHECK(std::abs(tci::imag(result) - tci::imag(expected)) < 1e-10);
  }

  // Test 4: In-place scaling with zero
  {
    cytnx::Tensor tensor;
    tci::fill(ctx, {2, 2}, cytnx::cytnx_complex128(5.0, -2.0), tensor);
    tci::elem_t<cytnx::Tensor> scale_factor = cytnx::cytnx_complex128(0.0, 0.0);

    CHECK_NOTHROW(tci::scale(ctx, tensor, scale_factor));

    auto result = tci::get_elem(ctx, tensor, {0ULL, 1ULL});
    CHECK(std::abs(tci::real(result)) < 1e-15);
    CHECK(std::abs(tci::imag(result)) < 1e-15);
  }

  // Test 5: In-place scaling with identity (1.0)
  {
    cytnx::Tensor original;
    tci::fill(ctx, {3, 1}, cytnx::cytnx_complex128(7.0, -3.0), original);
    auto tensor = tci::copy(ctx, original);
    tci::elem_t<cytnx::Tensor> scale_factor = cytnx::cytnx_complex128(1.0, 0.0);

    CHECK_NOTHROW(tci::scale(ctx, tensor, scale_factor));

    // Should remain unchanged
    auto original_elem = tci::get_elem(ctx, original, {2ULL, 0ULL});
    auto result_elem = tci::get_elem(ctx, tensor, {2ULL, 0ULL});
    CHECK(std::abs(tci::real(original_elem) - tci::real(result_elem)) < 1e-15);
    CHECK(std::abs(tci::imag(original_elem) - tci::imag(result_elem)) < 1e-15);
  }

  // Test 6: Out-of-place scaling with different tensors
  {
    cytnx::Tensor input, output;
    tci::fill(ctx, {2, 3}, cytnx::cytnx_complex128(8.0, 1.0), input);
    tci::allocate(ctx, {2, 3}, output);
    tci::elem_t<cytnx::Tensor> scale_factor = cytnx::cytnx_complex128(0.5, -0.25);

    CHECK_NOTHROW(tci::scale(ctx, input, scale_factor, output));

    // Input should remain unchanged
    auto input_elem = tci::get_elem(ctx, input, {0ULL, 0ULL});
    auto input_expected = cytnx::cytnx_complex128(8.0, 1.0);
    CHECK(std::abs(tci::real(input_elem) - tci::real(input_expected)) < 1e-15);
    CHECK(std::abs(tci::imag(input_elem) - tci::imag(input_expected)) < 1e-15);

    // Output should contain scaled values
    auto output_elem = tci::get_elem(ctx, output, {0ULL, 0ULL});
    // (8.0 + 1.0i) * (0.5 - 0.25i) = 4.0 - 2.0i + 0.5i - 0.25i² = 4.0 - 1.5i + 0.25 = 4.25 - 1.5i
    auto output_expected = cytnx::cytnx_complex128(4.25, -1.5);
    CHECK(std::abs(tci::real(output_elem) - tci::real(output_expected)) < 1e-10);
    CHECK(std::abs(tci::imag(output_elem) - tci::imag(output_expected)) < 1e-10);
  }

  // Test 7: Out-of-place scaling with larger tensor shapes
  {
    cytnx::Tensor input;
    tci::zeros(ctx, {4, 5}, input);
    tci::set_elem(ctx, input, {0ULL, 0ULL}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, input, {3ULL, 4ULL}, cytnx::cytnx_complex128(0.0, 1.0));

    cytnx::Tensor output;
    tci::allocate(ctx, {4, 5}, output);
    tci::elem_t<cytnx::Tensor> scale_factor = cytnx::cytnx_complex128(3.0, 4.0);

    CHECK_NOTHROW(tci::scale(ctx, input, scale_factor, output));

    // Check specific elements
    auto result1 = tci::get_elem(ctx, output, {0ULL, 0ULL});
    auto expected1 = cytnx::cytnx_complex128(3.0, 4.0);  // (1.0 + 0.0i) * (3.0 + 4.0i)
    CHECK(std::abs(tci::real(result1) - tci::real(expected1)) < 1e-10);
    CHECK(std::abs(tci::imag(result1) - tci::imag(expected1)) < 1e-10);

    auto result2 = tci::get_elem(ctx, output, {3ULL, 4ULL});
    auto expected2 = cytnx::cytnx_complex128(-4.0, 3.0);  // (0.0 + 1.0i) * (3.0 + 4.0i) = 3.0i + 4.0i² = -4.0 + 3.0i
    CHECK(std::abs(tci::real(result2) - tci::real(expected2)) < 1e-10);
    CHECK(std::abs(tci::imag(result2) - tci::imag(expected2)) < 1e-10);

    // Check zero elements remain zero
    auto result_zero = tci::get_elem(ctx, output, {1ULL, 1ULL});
    CHECK(std::abs(tci::real(result_zero)) < 1e-15);
    CHECK(std::abs(tci::imag(result_zero)) < 1e-15);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("tci::shrink API functionality") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  // Test 1: In-place shrinking of 2D tensor - extract central 2x2 from 4x4
  {
    cytnx::Tensor tensor;
    tci::zeros(ctx, {4, 4}, tensor);

    // Fill with identifiable pattern: [row][col] = row*10 + col
    for (cytnx::cytnx_uint64 i = 0; i < 4; ++i) {
      for (cytnx::cytnx_uint64 j = 0; j < 4; ++j) {
        tci::set_elem(ctx, tensor, {i, j}, cytnx::cytnx_complex128(i*10.0 + j, 0.0));
      }
    }

    // Shrink to extract central 2x2 (indices 1:3, 1:3)
    tci::bond_idx_elem_coor_pair_map<cytnx::Tensor> shrink_map;
    shrink_map[0] = {1ULL, 3ULL};  // rows 1-2 (range [1, 3))
    shrink_map[1] = {1ULL, 3ULL};  // cols 1-2 (range [1, 3))

    CHECK_NOTHROW(tci::shrink(ctx, tensor, shrink_map));

    // Check resulting shape
    auto result_shape = tci::shape(ctx, tensor);
    CHECK(result_shape[0] == 2ULL);
    CHECK(result_shape[1] == 2ULL);

    // Check values: should be [11, 12; 21, 22]
    auto elem_00 = tci::get_elem(ctx, tensor, {0ULL, 0ULL});
    auto elem_01 = tci::get_elem(ctx, tensor, {0ULL, 1ULL});
    auto elem_10 = tci::get_elem(ctx, tensor, {1ULL, 0ULL});
    auto elem_11 = tci::get_elem(ctx, tensor, {1ULL, 1ULL});

    CHECK(std::abs(tci::real(elem_00) - 11.0) < 1e-15);
    CHECK(std::abs(tci::real(elem_01) - 12.0) < 1e-15);
    CHECK(std::abs(tci::real(elem_10) - 21.0) < 1e-15);
    CHECK(std::abs(tci::real(elem_11) - 22.0) < 1e-15);
  }

  // Test 2: In-place shrinking with partial dimension specification
  {
    cytnx::Tensor tensor;
    tci::zeros(ctx, {3, 5, 2}, tensor);

    // Fill with identifiable pattern
    for (cytnx::cytnx_uint64 i = 0; i < 3; ++i) {
      for (cytnx::cytnx_uint64 j = 0; j < 5; ++j) {
        for (cytnx::cytnx_uint64 k = 0; k < 2; ++k) {
          tci::set_elem(ctx, tensor, {i, j, k},
                       cytnx::cytnx_complex128(i*100.0 + j*10.0 + k, 0.0));
        }
      }
    }

    // Shrink only middle dimension (keep first slice of first dim, middle 3 of second dim, all of third dim)
    tci::bond_idx_elem_coor_pair_map<cytnx::Tensor> shrink_map;
    shrink_map[0] = {0ULL, 1ULL};  // first slice only
    shrink_map[1] = {1ULL, 4ULL};  // middle 3 slices

    CHECK_NOTHROW(tci::shrink(ctx, tensor, shrink_map));

    // Check resulting shape
    auto result_shape = tci::shape(ctx, tensor);
    CHECK(result_shape[0] == 1ULL);
    CHECK(result_shape[1] == 3ULL);
    CHECK(result_shape[2] == 2ULL);

    // Check specific values
    auto elem = tci::get_elem(ctx, tensor, {0ULL, 0ULL, 1ULL});
    CHECK(std::abs(tci::real(elem) - 11.0) < 1e-15);  // was [0,1,1] = 0*100 + 1*10 + 1 = 11

    elem = tci::get_elem(ctx, tensor, {0ULL, 2ULL, 0ULL});
    CHECK(std::abs(tci::real(elem) - 30.0) < 1e-15);  // was [0,3,0] = 0*100 + 3*10 + 0 = 30
  }

  // Test 3: Out-of-place shrinking preserves original
  {
    cytnx::Tensor input, output;
    tci::fill(ctx, {3, 3}, cytnx::cytnx_complex128(42.0, 7.0), input);

    // Set corner elements to different values
    tci::set_elem(ctx, input, {0ULL, 0ULL}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, input, {2ULL, 2ULL}, cytnx::cytnx_complex128(9.0, 0.0));

    // Shrink to extract 2x2 from top-left
    tci::bond_idx_elem_coor_pair_map<cytnx::Tensor> shrink_map;
    shrink_map[0] = {0ULL, 2ULL};  // first 2 rows
    shrink_map[1] = {0ULL, 2ULL};  // first 2 cols

    CHECK_NOTHROW(tci::shrink(ctx, input, shrink_map, output));

    // Input should be unchanged
    auto input_shape = tci::shape(ctx, input);
    CHECK(input_shape[0] == 3ULL);
    CHECK(input_shape[1] == 3ULL);

    auto input_corner = tci::get_elem(ctx, input, {2ULL, 2ULL});
    CHECK(std::abs(tci::real(input_corner) - 9.0) < 1e-15);

    // Output should have shrunk dimensions
    auto output_shape = tci::shape(ctx, output);
    CHECK(output_shape[0] == 2ULL);
    CHECK(output_shape[1] == 2ULL);

    // Output should contain the extracted values
    auto output_elem = tci::get_elem(ctx, output, {0ULL, 0ULL});
    CHECK(std::abs(tci::real(output_elem) - 1.0) < 1e-15);

    auto output_fill = tci::get_elem(ctx, output, {1ULL, 1ULL});
    CHECK(std::abs(tci::real(output_fill) - 42.0) < 1e-15);
    CHECK(std::abs(tci::imag(output_fill) - 7.0) < 1e-15);
  }

  // Test 4: Single element extraction
  {
    cytnx::Tensor tensor;
    tci::zeros(ctx, {4, 3}, tensor);
    tci::set_elem(ctx, tensor, {2ULL, 1ULL}, cytnx::cytnx_complex128(123.456, -78.9));

    // Extract single element
    tci::bond_idx_elem_coor_pair_map<cytnx::Tensor> shrink_map;
    shrink_map[0] = {2ULL, 3ULL};  // single row
    shrink_map[1] = {1ULL, 2ULL};  // single col

    CHECK_NOTHROW(tci::shrink(ctx, tensor, shrink_map));

    // Should be 1x1 tensor
    auto shape = tci::shape(ctx, tensor);
    CHECK(shape[0] == 1ULL);
    CHECK(shape[1] == 1ULL);

    auto elem = tci::get_elem(ctx, tensor, {0ULL, 0ULL});
    CHECK(std::abs(tci::real(elem) - 123.456) < 1e-10);
    CHECK(std::abs(tci::imag(elem) + 78.9) < 1e-10);
  }

  // Test 5: Full slice extraction (should be no-op)
  {
    cytnx::Tensor original, shrunk;
    tci::fill(ctx, {2, 3, 4}, cytnx::cytnx_complex128(1.5, -2.5), original);

    // Extract full ranges
    tci::bond_idx_elem_coor_pair_map<cytnx::Tensor> shrink_map;
    shrink_map[0] = {0ULL, 2ULL};  // full first dimension
    shrink_map[1] = {0ULL, 3ULL};  // full second dimension
    shrink_map[2] = {0ULL, 4ULL};  // full third dimension

    CHECK_NOTHROW(tci::shrink(ctx, original, shrink_map, shrunk));

    // Should have same shape
    auto orig_shape = tci::shape(ctx, original);
    auto shrunk_shape = tci::shape(ctx, shrunk);
    CHECK(orig_shape[0] == shrunk_shape[0]);
    CHECK(orig_shape[1] == shrunk_shape[1]);
    CHECK(orig_shape[2] == shrunk_shape[2]);

    // Values should match
    auto orig_elem = tci::get_elem(ctx, original, {1ULL, 2ULL, 3ULL});
    auto shrunk_elem = tci::get_elem(ctx, shrunk, {1ULL, 2ULL, 3ULL});
    CHECK(std::abs(tci::real(orig_elem) - tci::real(shrunk_elem)) < 1e-15);
    CHECK(std::abs(tci::imag(orig_elem) - tci::imag(shrunk_elem)) < 1e-15);
  }

  // Test 6: Edge case - extract first row only
  {
    cytnx::Tensor tensor;
    tci::zeros(ctx, {5, 3}, tensor);

    // Set first row to different values
    tci::set_elem(ctx, tensor, {0ULL, 0ULL}, cytnx::cytnx_complex128(10.0, 0.0));
    tci::set_elem(ctx, tensor, {0ULL, 1ULL}, cytnx::cytnx_complex128(20.0, 0.0));
    tci::set_elem(ctx, tensor, {0ULL, 2ULL}, cytnx::cytnx_complex128(30.0, 0.0));

    tci::bond_idx_elem_coor_pair_map<cytnx::Tensor> shrink_map;
    shrink_map[0] = {0ULL, 1ULL};  // first row only

    CHECK_NOTHROW(tci::shrink(ctx, tensor, shrink_map));

    auto shape = tci::shape(ctx, tensor);
    CHECK(shape[0] == 1ULL);
    CHECK(shape[1] == 3ULL);

    CHECK(std::abs(tci::real(tci::get_elem(ctx, tensor, {0ULL, 0ULL})) - 10.0) < 1e-15);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, tensor, {0ULL, 1ULL})) - 20.0) < 1e-15);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, tensor, {0ULL, 2ULL})) - 30.0) < 1e-15);
  }

  // Test 7: Edge case - extract last column only
  {
    cytnx::Tensor tensor, result;
    tci::zeros(ctx, {3, 4}, tensor);

    // Set last column to different values
    tci::set_elem(ctx, tensor, {0ULL, 3ULL}, cytnx::cytnx_complex128(100.0, 0.0));
    tci::set_elem(ctx, tensor, {1ULL, 3ULL}, cytnx::cytnx_complex128(200.0, 0.0));
    tci::set_elem(ctx, tensor, {2ULL, 3ULL}, cytnx::cytnx_complex128(300.0, 0.0));

    tci::bond_idx_elem_coor_pair_map<cytnx::Tensor> shrink_map;
    shrink_map[1] = {3ULL, 4ULL};  // last column only

    CHECK_NOTHROW(tci::shrink(ctx, tensor, shrink_map, result));

    auto shape = tci::shape(ctx, result);
    CHECK(shape[0] == 3ULL);
    CHECK(shape[1] == 1ULL);

    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {0ULL, 0ULL})) - 100.0) < 1e-15);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {1ULL, 0ULL})) - 200.0) < 1e-15);
    CHECK(std::abs(tci::real(tci::get_elem(ctx, result, {2ULL, 0ULL})) - 300.0) < 1e-15);
  }

  tci::destroy_context(ctx);
}

TEST_CASE("tci::to_cplx API functionality") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  // Test 1: Convert complex tensor to complex (should be a copy, in-place)
  {
    cytnx::Tensor complex_input, complex_output;
    tci::zeros(ctx, {2, 3}, complex_input);

    // Set complex tensor values
    tci::set_elem(ctx, complex_input, {0ULL, 0ULL}, cytnx::cytnx_complex128(3.14, 0.0));
    tci::set_elem(ctx, complex_input, {1ULL, 2ULL}, cytnx::cytnx_complex128(-2.71, 0.0));

    CHECK_NOTHROW(tci::to_cplx(ctx, complex_input, complex_output));

    // Check shape is preserved
    auto input_shape = tci::shape(ctx, complex_input);
    auto output_shape = tci::shape(ctx, complex_output);
    CHECK(input_shape[0] == output_shape[0]);
    CHECK(input_shape[1] == output_shape[1]);

    // Check values are preserved exactly (complex -> complex copy)
    auto elem1 = tci::get_elem(ctx, complex_output, {0ULL, 0ULL});
    auto elem2 = tci::get_elem(ctx, complex_output, {1ULL, 2ULL});
    auto elem_zero = tci::get_elem(ctx, complex_output, {0ULL, 1ULL});

    CHECK(std::abs(tci::real(elem1) - 3.14) < 1e-15);
    CHECK(std::abs(tci::imag(elem1)) < 1e-15);  // Imaginary part should be zero
    CHECK(std::abs(tci::real(elem2) + 2.71) < 1e-15);
    CHECK(std::abs(tci::imag(elem2)) < 1e-15);
    CHECK(std::abs(tci::real(elem_zero)) < 1e-15);
    CHECK(std::abs(tci::imag(elem_zero)) < 1e-15);
  }

  // Test 2: Out-of-place conversion from complex to complex (copy)
  {
    cytnx::Tensor complex_input;
    tci::zeros(ctx, {2, 2}, complex_input);

    tci::set_elem(ctx, complex_input, {0ULL, 0ULL}, cytnx::cytnx_complex128(7.0, -3.0));
    tci::set_elem(ctx, complex_input, {1ULL, 0ULL}, cytnx::cytnx_complex128(0.0, 5.5));

    auto complex_result = tci::to_cplx(ctx, complex_input);

    // Values should be identical
    auto input_elem1 = tci::get_elem(ctx, complex_input, {0ULL, 0ULL});
    auto result_elem1 = tci::get_elem(ctx, complex_result, {0ULL, 0ULL});
    auto input_elem2 = tci::get_elem(ctx, complex_input, {1ULL, 0ULL});
    auto result_elem2 = tci::get_elem(ctx, complex_result, {1ULL, 0ULL});

    CHECK(std::abs(tci::real(input_elem1) - tci::real(result_elem1)) < 1e-15);
    CHECK(std::abs(tci::imag(input_elem1) - tci::imag(result_elem1)) < 1e-15);
    CHECK(std::abs(tci::real(input_elem2) - tci::real(result_elem2)) < 1e-15);
    CHECK(std::abs(tci::imag(input_elem2) - tci::imag(result_elem2)) < 1e-15);

    CHECK(std::abs(tci::real(result_elem1) - 7.0) < 1e-15);
    CHECK(std::abs(tci::imag(result_elem1) + 3.0) < 1e-15);
    CHECK(std::abs(tci::real(result_elem2)) < 1e-15);
    CHECK(std::abs(tci::imag(result_elem2) - 5.5) < 1e-15);
  }

  // Test 3: Complex tensor with both real and imaginary parts
  {
    cytnx::Tensor input, output;
    tci::zeros(ctx, {2, 3}, input);

    // Set various complex values
    tci::set_elem(ctx, input, {0ULL, 0ULL}, cytnx::cytnx_complex128(1.5, 2.5));
    tci::set_elem(ctx, input, {1ULL, 1ULL}, cytnx::cytnx_complex128(-3.0, 4.0));
    tci::set_elem(ctx, input, {0ULL, 2ULL}, cytnx::cytnx_complex128(0.0, -7.5));

    CHECK_NOTHROW(tci::to_cplx(ctx, input, output));

    // Check that complex values are preserved exactly
    auto elem1 = tci::get_elem(ctx, output, {0ULL, 0ULL});
    auto elem2 = tci::get_elem(ctx, output, {1ULL, 1ULL});
    auto elem3 = tci::get_elem(ctx, output, {0ULL, 2ULL});

    CHECK(std::abs(tci::real(elem1) - 1.5) < 1e-15);
    CHECK(std::abs(tci::imag(elem1) - 2.5) < 1e-15);
    CHECK(std::abs(tci::real(elem2) + 3.0) < 1e-15);
    CHECK(std::abs(tci::imag(elem2) - 4.0) < 1e-15);
    CHECK(std::abs(tci::real(elem3)) < 1e-15);
    CHECK(std::abs(tci::imag(elem3) + 7.5) < 1e-15);

    // Verify input is unchanged
    auto input_elem = tci::get_elem(ctx, input, {0ULL, 0ULL});
    CHECK(std::abs(tci::real(input_elem) - 1.5) < 1e-15);
    CHECK(std::abs(tci::imag(input_elem) - 2.5) < 1e-15);
  }

  // Test 4: Larger tensor test
  {
    cytnx::Tensor large_input, large_output;
    tci::zeros(ctx, {3, 4, 2}, large_input);

    // Fill with pattern: real = i*100 + j*10 + k, imag = i + j + k
    for (cytnx::cytnx_uint64 i = 0; i < 3; ++i) {
      for (cytnx::cytnx_uint64 j = 0; j < 4; ++j) {
        for (cytnx::cytnx_uint64 k = 0; k < 2; ++k) {
          double real_val = i*100.0 + j*10.0 + k;
          double imag_val = i + j + k;
          tci::set_elem(ctx, large_input, {i, j, k}, cytnx::cytnx_complex128(real_val, imag_val));
        }
      }
    }

    CHECK_NOTHROW(tci::to_cplx(ctx, large_input, large_output));

    // Check a few sample values
    auto elem_start = tci::get_elem(ctx, large_output, {0ULL, 0ULL, 0ULL});
    auto elem_mid = tci::get_elem(ctx, large_output, {1ULL, 2ULL, 1ULL});
    auto elem_end = tci::get_elem(ctx, large_output, {2ULL, 3ULL, 1ULL});

    CHECK(std::abs(tci::real(elem_start) - 0.0) < 1e-15);
    CHECK(std::abs(tci::imag(elem_start) - 0.0) < 1e-15);

    CHECK(std::abs(tci::real(elem_mid) - 121.0) < 1e-15);  // 1*100 + 2*10 + 1
    CHECK(std::abs(tci::imag(elem_mid) - 4.0) < 1e-15);    // 1 + 2 + 1

    CHECK(std::abs(tci::real(elem_end) - 231.0) < 1e-15);  // 2*100 + 3*10 + 1
    CHECK(std::abs(tci::imag(elem_end) - 6.0) < 1e-15);    // 2 + 3 + 1

    // Check shape preservation
    auto input_shape = tci::shape(ctx, large_input);
    auto output_shape = tci::shape(ctx, large_output);
    CHECK(input_shape[0] == output_shape[0]);
    CHECK(input_shape[1] == output_shape[1]);
    CHECK(input_shape[2] == output_shape[2]);
  }

  // Test 5: Edge case - single element tensor
  {
    cytnx::Tensor scalar_input, scalar_output;
    tci::zeros(ctx, {1}, scalar_input);
    tci::set_elem(ctx, scalar_input, {0ULL}, cytnx::cytnx_complex128(9.876, -1.234));

    CHECK_NOTHROW(tci::to_cplx(ctx, scalar_input, scalar_output));

    auto result = tci::get_elem(ctx, scalar_output, {0ULL});
    CHECK(std::abs(tci::real(result) - 9.876) < 1e-15);
    CHECK(std::abs(tci::imag(result) + 1.234) < 1e-15);
  }

  tci::destroy_context(ctx);
}

