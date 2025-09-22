#include <doctest/doctest.h>
#include <tci/tci.h>
#include <cytnx.hpp>
#include <cmath>

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

/*
TEST_CASE("TCI SVD Decomposition") {  // Disabled until basic functions are implemented
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

        // This will FAIL with current placeholder implementation
        // because s_diag is just a clone of identity (2x2 matrix), not singular values
        CHECK(std::abs(s_0.real() - 1.0) < 1e-10);
        CHECK(std::abs(s_1.real() - 1.0) < 1e-10);

        // Verify s_diag is rank-1 (vector of singular values)
        CHECK(tci::rank(ctx, s_diag) == 1);

        // Verify s_diag has 2 elements (for 2x2 matrix)
        CHECK(tci::size(ctx, s_diag) == 2);
    }

    tci::destroy_context(ctx);
}
*/