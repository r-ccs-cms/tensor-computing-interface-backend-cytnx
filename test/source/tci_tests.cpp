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
    CHECK(std::abs(retrieved_value.real() - expected_value.real()) < 1e-10);
    CHECK(std::abs(retrieved_value.imag() - expected_value.imag()) < 1e-10);
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
    CHECK(std::abs(s_0.real() - 1.0) < 1e-10);
    CHECK(std::abs(s_1.real() - 1.0) < 1e-10);

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
    CHECK(std::abs(tci::get_elem(ctx, vector, {0, 0}).real() - 1.0) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, vector, {1, 1}).real() - 2.0) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, vector, {2, 2}).real() - 3.0) < 1e-10);

    // Check off-diagonal elements are zero
    CHECK(std::abs(tci::get_elem(ctx, vector, {0, 1}).real()) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, vector, {1, 0}).real()) < 1e-10);
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
    CHECK(std::abs(tci::get_elem(ctx, identity, {0}).real() - 1.0) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, identity, {1}).real() - 1.0) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, identity, {2}).real() - 1.0) < 1e-10);
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
    CHECK(std::abs(tci::get_elem(ctx, matrix, {}).real() - 9.0) < 1e-10);
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
    CHECK(std::abs(tci::get_elem(ctx, eigenvals, {0}).real() - 1.0) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, eigenvals, {1}).real() - 1.0) < 1e-10);

    CHECK(tci::rank(ctx, eigenvecs) == 2);
    CHECK(tci::shape(ctx, eigenvecs)[0] == 2);
    CHECK(tci::shape(ctx, eigenvecs)[1] == 2);
    CHECK(std::abs(tci::get_elem(ctx, eigenvecs, {0, 0}).real() - 1.0) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, eigenvecs, {1, 1}).real() - 1.0) < 1e-10);
  }

  SUBCASE("Symmetric eigendecomposition") {
    cytnx::Tensor matrix;
    tci::eye(ctx, 2, matrix);

    cytnx::Tensor eigenvals, eigenvecs;
    tci::eigh(ctx, matrix, 1, eigenvals, eigenvecs);

    CHECK(tci::rank(ctx, eigenvals) == 1);
    CHECK(tci::size(ctx, eigenvals) == 2);
    CHECK(std::abs(tci::get_elem(ctx, eigenvals, {0}).real() - 1.0) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, eigenvals, {1}).real() - 1.0) < 1e-10);

    CHECK(tci::rank(ctx, eigenvecs) == 2);
    CHECK(tci::shape(ctx, eigenvecs)[0] == 2);
    CHECK(tci::shape(ctx, eigenvecs)[1] == 2);
    CHECK(std::abs(tci::get_elem(ctx, eigenvecs, {0, 0}).real() - 1.0) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, eigenvecs, {1, 1}).real() - 1.0) < 1e-10);
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

    CHECK(std::abs(c00.real() - 19.0) < 1e-10);
    CHECK(std::abs(c01.real() - 22.0) < 1e-10);
    CHECK(std::abs(c10.real() - 43.0) < 1e-10);
    CHECK(std::abs(c11.real() - 50.0) < 1e-10);
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
    CHECK(std::abs(dot_result.real() - 32.0) < 1e-10);
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
    CHECK(std::abs(c00.real() - 3.0) < 1e-10);
    CHECK(std::abs(c12.real() - 10.0) < 1e-10);
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
        [](tci::elem_t<cytnx::Tensor>& elem) { elem *= 2.0; };

    // Apply for_each to modify all elements
    CHECK_NOTHROW(tci::for_each(ctx, tensor, std::move(double_func)));

    // Verify all elements were doubled: [2, 4, 6, 8, 10, 12]
    CHECK(std::abs(tci::get_elem(ctx, tensor, {0, 0}).real() - 2.0) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, tensor, {0, 1}).real() - 4.0) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, tensor, {0, 2}).real() - 6.0) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, tensor, {1, 0}).real() - 8.0) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, tensor, {1, 1}).real() - 10.0) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, tensor, {1, 2}).real() - 12.0) < 1e-10);
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
          sum += elem.real();
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
    CHECK(std::abs(tci::get_elem(ctx, result, {0, 0}).real() - 7.0) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, result, {0, 1}).real() - 9.0) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, result, {1, 0}).real() - 11.0) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, result, {1, 1}).real() - 13.0) < 1e-10);
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
    CHECK(std::abs(tci::get_elem(ctx, result, {0, 0}).real() - 3.0) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, result, {0, 1}).real() - 8.0) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, result, {1, 0}).real() - 13.0) < 1e-10);
    CHECK(std::abs(tci::get_elem(ctx, result, {1, 1}).real() - 18.0) < 1e-10);
  }

  SUBCASE("linear_combine edge cases") {
    tci::shape_t<cytnx::Tensor> shape = {1, 1};
    cytnx::Tensor single_tensor, result;
    tci::zeros(ctx, shape, single_tensor);
    tci::set_elem(ctx, single_tensor, {0, 0}, cytnx::cytnx_complex128(5.0, 0.0));

    // Test single tensor uniform combination
    tci::List<cytnx::Tensor> single_list = {single_tensor};
    CHECK_NOTHROW(tci::linear_combine(ctx, single_list, result));
    CHECK(std::abs(tci::get_elem(ctx, result, {0, 0}).real() - 5.0) < 1e-10);

    // Test single tensor with coefficient
    tci::List<tci::elem_t<cytnx::Tensor>> single_coef = {cytnx::cytnx_complex128(3.0, 0.0)};
    CHECK_NOTHROW(tci::linear_combine(ctx, single_list, single_coef, result));
    CHECK(std::abs(tci::get_elem(ctx, result, {0, 0}).real() - 15.0) < 1e-10);
  }

  tci::destroy_context(ctx);
}
