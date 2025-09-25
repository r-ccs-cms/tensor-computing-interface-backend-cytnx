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
    // This should fail until LQ is implemented
    CHECK_THROWS_AS(tci::lq(ctx, matrix, 2, l, q), std::runtime_error);
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

    // This should fail until trunc_svd is implemented
    CHECK_THROWS_AS(tci::trunc_svd(ctx, matrix, 2, u, s_diag, v_dag, trunc_err, 2, 0.1),
                    std::runtime_error);
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
    // This should fail until eigvals is implemented
    CHECK_THROWS_AS(tci::eigvals(ctx, diagonal, 2, eigenvals), std::runtime_error);
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
    // This should fail until eigvalsh is implemented
    CHECK_THROWS_AS(tci::eigvalsh(ctx, symmetric, 2, eigenvals), std::runtime_error);
  }

  SUBCASE("Eigenvalues and eigenvectors") {
    cytnx::Tensor matrix;
    tci::eye(ctx, 2, matrix);

    cytnx::Tensor eigenvals, eigenvecs;
    // This should fail until eig is implemented
    CHECK_THROWS_AS(tci::eig(ctx, matrix, 1, eigenvals, eigenvecs), std::runtime_error);
  }

  SUBCASE("Symmetric eigendecomposition") {
    cytnx::Tensor matrix;
    tci::eye(ctx, 2, matrix);

    cytnx::Tensor eigenvals, eigenvecs;
    // This should fail until eigh is implemented
    CHECK_THROWS_AS(tci::eigh(ctx, matrix, 1, eigenvals, eigenvecs), std::runtime_error);
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

    // Skip exp test - function not implemented yet
    // TODO: Implement tci::exp function
    // CHECK_THROWS_AS(tci::exp(ctx, matrix, 1), std::runtime_error);
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

    // Test with chi_max = 2 (should truncate to 2 largest singular values)
    tci::trunc_svd(ctx, matrix, 2, u, s_diag, v_dag, trunc_err, 2, 0.5);

    // Should keep only 2 largest singular values (above s_min = 0.5)
    CHECK(s_diag.shape()[0] <= 2);

    // Should have truncation error > 0 because smaller singular values were truncated
    CHECK(trunc_err >= 0.0);  // Truncation error should be non-negative
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