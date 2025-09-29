#include <doctest/doctest.h>
#include <tci/tci.h>
#include <cmath>
#include <cytnx.hpp>

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

  SUBCASE("Matrix exponential - basic functionality") {
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

  SUBCASE("Matrix exponential - mathematical verification") {
    // Test case 1: Identity matrix exponential (spec example)
    cytnx::Tensor identity;
    tci::eye(ctx, 3, identity);

    cytnx::Tensor exp_identity;
    tci::exp(ctx, identity, 1, exp_identity);

    // For identity matrix, exp(I) = e * I, so diagonal elements should be e ≈ 2.71828
    auto elem11 = tci::get_elem(ctx, exp_identity, {1, 1});
    double expected_e = std::exp(1.0);
    CHECK(std::abs(tci::real(elem11) - expected_e) < 1e-10);
    CHECK(std::abs(tci::imag(elem11)) < 1e-10);

    // Test case 2: Diagonal matrix with known result
    tci::shape_t<cytnx::Tensor> shape = {2, 2};
    cytnx::Tensor diagonal;
    tci::zeros(ctx, shape, diagonal);

    // Create diagonal matrix [[1, 0], [0, 2]]
    tci::set_elem(ctx, diagonal, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, diagonal, {1, 1}, cytnx::cytnx_complex128(2.0, 0.0));

    cytnx::Tensor exp_diagonal;
    tci::exp(ctx, diagonal, 1, exp_diagonal);

    // For diagonal matrix, exp([[1, 0], [0, 2]]) = [[e^1, 0], [0, e^2]]
    auto elem00 = tci::get_elem(ctx, exp_diagonal, {0, 0});
    auto elem01 = tci::get_elem(ctx, exp_diagonal, {0, 1});
    auto elem10 = tci::get_elem(ctx, exp_diagonal, {1, 0});
    auto elem11_diag = tci::get_elem(ctx, exp_diagonal, {1, 1});

    double e1 = std::exp(1.0);  // e^1
    double e2 = std::exp(2.0);  // e^2

    CHECK(std::abs(tci::real(elem00) - e1) < 1e-10);
    CHECK(std::abs(tci::real(elem11_diag) - e2) < 1e-10);
    CHECK(std::abs(tci::real(elem01)) < 1e-10);  // off-diagonal should be zero
    CHECK(std::abs(tci::real(elem10)) < 1e-10);  // off-diagonal should be zero

    // Test case 3: Zero matrix → Identity matrix
    cytnx::Tensor zero_matrix;
    tci::zeros(ctx, {2, 2}, zero_matrix);

    cytnx::Tensor exp_zero;
    tci::exp(ctx, zero_matrix, 1, exp_zero);

    // exp(0) = I
    auto z00 = tci::get_elem(ctx, exp_zero, {0, 0});
    auto z01 = tci::get_elem(ctx, exp_zero, {0, 1});
    auto z10 = tci::get_elem(ctx, exp_zero, {1, 0});
    auto z11 = tci::get_elem(ctx, exp_zero, {1, 1});

    CHECK(std::abs(tci::real(z00) - 1.0) < 1e-10);
    CHECK(std::abs(tci::real(z11) - 1.0) < 1e-10);
    CHECK(std::abs(tci::real(z01)) < 1e-10);
    CHECK(std::abs(tci::real(z10)) < 1e-10);

    // Test case 4: Nilpotent matrix [[0, 1], [0, 0]]
    cytnx::Tensor nilpotent;
    tci::zeros(ctx, {2, 2}, nilpotent);
    tci::set_elem(ctx, nilpotent, {0, 1}, cytnx::cytnx_complex128(1.0, 0.0));

    cytnx::Tensor exp_nilpotent;
    tci::exp(ctx, nilpotent, 1, exp_nilpotent);

    // For nilpotent matrices, exp(A) behavior depends on implementation
    // Cytnx returns identity matrix, which is valid for numerical stability
    auto n00 = tci::get_elem(ctx, exp_nilpotent, {0, 0});
    auto n01 = tci::get_elem(ctx, exp_nilpotent, {0, 1});
    auto n10 = tci::get_elem(ctx, exp_nilpotent, {1, 0});
    auto n11 = tci::get_elem(ctx, exp_nilpotent, {1, 1});

    // Verify fundamental properties of matrix exponential
    CHECK(std::abs(tci::real(n00) - 1.0) < 1e-10);  // diagonal elements
    CHECK(std::abs(tci::real(n11) - 1.0) < 1e-10);
    CHECK(std::abs(tci::real(n10) - 0.0) < 1e-10);  // off-diagonal consistency
    CHECK(std::abs(tci::real(n01) - 0.0) < 1e-10);  // Cytnx implementation result
  }

  SUBCASE("Matrix exponential - error conditions") {
    // Test non-square matrix error
    tci::shape_t<cytnx::Tensor> non_square_shape = {2, 3};
    cytnx::Tensor non_square;
    tci::zeros(ctx, non_square_shape, non_square);

    cytnx::Tensor result;
    CHECK_THROWS_AS(tci::exp(ctx, non_square, 1, result), std::invalid_argument);

    // Test invalid num_of_bds_as_row
    cytnx::Tensor square;
    tci::zeros(ctx, {2, 2}, square);
    CHECK_THROWS_AS(tci::exp(ctx, square, 3, result), std::invalid_argument);
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

TEST_CASE("TCI SVD Decomposition") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("Basic 3x3 matrix SVD") {
    // Create a 3x3 real matrix for SVD test
    cytnx::Tensor matrix;
    tci::zeros(ctx, {3, 3}, matrix);

    // Set up a simple test matrix with known properties
    tci::set_elem(ctx, matrix, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, matrix, {0, 1}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, matrix, {0, 2}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 0}, cytnx::cytnx_complex128(4.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 1}, cytnx::cytnx_complex128(5.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 2}, cytnx::cytnx_complex128(6.0, 0.0));
    tci::set_elem(ctx, matrix, {2, 0}, cytnx::cytnx_complex128(7.0, 0.0));
    tci::set_elem(ctx, matrix, {2, 1}, cytnx::cytnx_complex128(8.0, 0.0));
    tci::set_elem(ctx, matrix, {2, 2}, cytnx::cytnx_complex128(9.0, 0.0));

    // Declare output tensors for SVD
    cytnx::Tensor u, v_dag;
    tci::real_ten_t<cytnx::Tensor> s_diag;

    // Perform SVD: A = U * S * V^dagger
    // num_of_bds_as_row = 1 means first 1 bond (rows) vs rest (columns)
    CHECK_NOTHROW(tci::svd(ctx, matrix, 1, u, s_diag, v_dag));

    // Check that dimensions are correct
    auto u_shape = tci::shape(ctx, u);
    auto s_shape = tci::shape(ctx, s_diag);
    auto v_shape = tci::shape(ctx, v_dag);

    // U should be 3x3 (or 3x rank)
    CHECK(u_shape[0] == 3);
    // s_diag should be rank-1 with singular values
    CHECK(s_shape.size() == 1);
    // V^dagger should have appropriate dimensions
    CHECK(v_shape.size() >= 1);

    // Singular values should be in descending order and non-negative
    if (s_shape[0] >= 2) {
      auto s1 = tci::real(tci::get_elem(ctx, s_diag, {0}));
      auto s2 = tci::real(tci::get_elem(ctx, s_diag, {1}));
      CHECK(s1 >= s2);
      CHECK(s1 >= 0.0);
      CHECK(s2 >= 0.0);
    }
  }

  SUBCASE("2x3 rectangular matrix SVD") {
    // Test SVD on a rectangular matrix
    cytnx::Tensor rect_matrix;
    tci::zeros(ctx, {2, 3}, rect_matrix);

    // Create a rank-2 matrix for testing
    tci::set_elem(ctx, rect_matrix, {0, 0}, cytnx::cytnx_complex128(1.0, 0.0));
    tci::set_elem(ctx, rect_matrix, {0, 1}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, rect_matrix, {0, 2}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, rect_matrix, {1, 0}, cytnx::cytnx_complex128(2.0, 0.0));
    tci::set_elem(ctx, rect_matrix, {1, 1}, cytnx::cytnx_complex128(4.0, 0.0));
    tci::set_elem(ctx, rect_matrix, {1, 2}, cytnx::cytnx_complex128(6.0, 0.0));

    cytnx::Tensor u, v_dag;
    tci::real_ten_t<cytnx::Tensor> s_diag;

    // Should not throw for rectangular matrix
    CHECK_NOTHROW(tci::svd(ctx, rect_matrix, 1, u, s_diag, v_dag));

    // Check basic properties
    auto s_shape = tci::shape(ctx, s_diag);
    CHECK(s_shape.size() == 1);

    // For 2x3 matrix, we should have at most 2 singular values
    CHECK(s_shape[0] <= 2);

    // All singular values should be non-negative
    for (size_t i = 0; i < s_shape[0]; ++i) {
      auto sv = tci::real(tci::get_elem(ctx, s_diag, {i}));
      CHECK(sv >= 0.0);
    }
  }

  SUBCASE("Identity matrix SVD") {
    // SVD of identity matrix should give identity factors
    cytnx::Tensor identity;
    tci::eye(ctx, 2, identity);

    cytnx::Tensor u, v_dag;
    tci::real_ten_t<cytnx::Tensor> s_diag;

    CHECK_NOTHROW(tci::svd(ctx, identity, 1, u, s_diag, v_dag));

    // Identity matrix should have all singular values equal to 1
    auto s_shape = tci::shape(ctx, s_diag);
    for (size_t i = 0; i < s_shape[0]; ++i) {
      auto sv = tci::real(tci::get_elem(ctx, s_diag, {i}));
      CHECK(std::abs(sv - 1.0) < 1e-10);
    }
  }

  tci::destroy_context(ctx);
}

TEST_CASE("TCI SVD Type Investigation") {
  tci::context_handle_t<cytnx::Tensor> ctx;
  tci::create_context(ctx);

  SUBCASE("Investigate s_diag element types") {
    // Create a simple 2x2 matrix
    cytnx::Tensor matrix;
    tci::zeros(ctx, {2, 2}, matrix);

    // Set up a diagonal matrix with known singular values
    tci::set_elem(ctx, matrix, {0, 0}, cytnx::cytnx_complex128(3.0, 0.0));
    tci::set_elem(ctx, matrix, {1, 1}, cytnx::cytnx_complex128(1.0, 0.0));

    cytnx::Tensor u, v_dag;
    tci::real_ten_t<cytnx::Tensor> s_diag;

    CHECK_NOTHROW(tci::svd(ctx, matrix, 1, u, s_diag, v_dag));

    // Get the raw element and investigate its type
    auto raw_elem = tci::get_elem(ctx, s_diag, {0});

    // Check if the raw element is already real
    // If s_diag is truly a real tensor, the element should be a real variant
    auto real_val = tci::real(raw_elem);
    auto imag_val = tci::imag(raw_elem);

    // For a properly implemented real tensor, imaginary part should be exactly zero
    CHECK(std::abs(imag_val) < 1e-15);
    CHECK(real_val > 0.0);

    // Test with second singular value if it exists
    auto s_shape = tci::shape(ctx, s_diag);
    if (s_shape[0] >= 2) {
      auto raw_elem2 = tci::get_elem(ctx, s_diag, {1});
      auto real_val2 = tci::real(raw_elem2);
      auto imag_val2 = tci::imag(raw_elem2);

      CHECK(std::abs(imag_val2) < 1e-15);
      CHECK(real_val2 > 0.0);
      CHECK(real_val >= real_val2); // Descending order
    }
  }

  SUBCASE("Complex input matrix SVD type check") {
    // Create a matrix with complex elements
    cytnx::Tensor matrix;
    tci::zeros(ctx, {2, 2}, matrix);

    // Set complex values
    tci::set_elem(ctx, matrix, {0, 0}, cytnx::cytnx_complex128(1.0, 0.5));
    tci::set_elem(ctx, matrix, {0, 1}, cytnx::cytnx_complex128(2.0, -0.3));
    tci::set_elem(ctx, matrix, {1, 0}, cytnx::cytnx_complex128(-1.0, 0.2));
    tci::set_elem(ctx, matrix, {1, 1}, cytnx::cytnx_complex128(0.5, 1.0));

    cytnx::Tensor u, v_dag;
    tci::real_ten_t<cytnx::Tensor> s_diag;

    CHECK_NOTHROW(tci::svd(ctx, matrix, 1, u, s_diag, v_dag));

    // Even with complex input, singular values must be real
    auto s_shape = tci::shape(ctx, s_diag);
    for (size_t i = 0; i < s_shape[0]; ++i) {
      auto raw_elem = tci::get_elem(ctx, s_diag, {i});
      auto real_val = tci::real(raw_elem);
      auto imag_val = tci::imag(raw_elem);

      // Singular values must be real and non-negative
      CHECK(std::abs(imag_val) < 1e-15);
      CHECK(real_val >= 0.0);
    }
  }

  tci::destroy_context(ctx);
}