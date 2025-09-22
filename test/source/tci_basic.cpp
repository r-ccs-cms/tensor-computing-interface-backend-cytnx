#include <doctest/doctest.h>
#include <tci/tci.h>
#include <cytnx.hpp>

TEST_CASE("TCI Context Management") {
    tci::context_handle_t<cytnx::Tensor> ctx;

    SUBCASE("Create and destroy context") {
        CHECK_NOTHROW(tci::create_context(ctx));
        CHECK_NOTHROW(tci::destroy_context(ctx));
    }
}

TEST_CASE("TCI Tensor Creation") {
    tci::context_handle_t<cytnx::Tensor> ctx;
    tci::create_context(ctx);

    SUBCASE("Create zero tensor") {
        tci::shape_t<cytnx::Tensor> shape = {2, 3};
        cytnx::Tensor tensor;

        CHECK_NOTHROW(tci::zeros(ctx, shape, tensor));
        CHECK(tci::rank(ctx, tensor) == 2);

        auto result_shape = tci::shape(ctx, tensor);
        CHECK(result_shape.size() == 2);
        CHECK(result_shape[0] == 2);
        CHECK(result_shape[1] == 3);
    }

    SUBCASE("Create identity matrix") {
        cytnx::Tensor identity;

        CHECK_NOTHROW(tci::eye(ctx, 3, identity));
        CHECK(tci::rank(ctx, identity) == 2);

        auto result_shape = tci::shape(ctx, identity);
        CHECK(result_shape.size() == 2);
        CHECK(result_shape[0] == 3);
        CHECK(result_shape[1] == 3);
    }

    tci::destroy_context(ctx);
}

TEST_CASE("TCI Tensor Properties") {
    tci::context_handle_t<cytnx::Tensor> ctx;
    tci::create_context(ctx);

    tci::shape_t<cytnx::Tensor> shape = {2, 3, 4};
    cytnx::Tensor tensor;
    tci::zeros(ctx, shape, tensor);

    SUBCASE("Tensor rank") {
        CHECK(tci::rank(ctx, tensor) == 3);
    }

    SUBCASE("Tensor shape") {
        auto result_shape = tci::shape(ctx, tensor);
        CHECK(result_shape.size() == 3);
        CHECK(result_shape[0] == 2);
        CHECK(result_shape[1] == 3);
        CHECK(result_shape[2] == 4);
    }

    SUBCASE("Tensor size") {
        CHECK(tci::size(ctx, tensor) == 24);
    }

    SUBCASE("Tensor size in bytes") {
        auto bytes = tci::size_bytes(ctx, tensor);
        CHECK(bytes > 0);
        // Expected: 24 elements * 16 bytes per complex128 = 384 bytes
        CHECK(bytes == 384);
    }

    tci::destroy_context(ctx);
}

TEST_CASE("TCI Tensor Operations") {
    tci::context_handle_t<cytnx::Tensor> ctx;
    tci::create_context(ctx);

    SUBCASE("Copy tensor") {
        tci::shape_t<cytnx::Tensor> shape = {2, 2};
        cytnx::Tensor original, copy;

        tci::zeros(ctx, shape, original);
        CHECK_NOTHROW(tci::copy(ctx, original, copy));

        CHECK(tci::rank(ctx, copy) == tci::rank(ctx, original));
        CHECK(tci::shape(ctx, copy) == tci::shape(ctx, original));
    }

    SUBCASE("Element access") {
        tci::shape_t<cytnx::Tensor> shape = {2, 2};
        cytnx::Tensor tensor;
        tci::zeros(ctx, shape, tensor);

        // Set element at (0,0) to (1.0, 0.0)
        tci::elem_coors_t<cytnx::Tensor> coord = {0, 0};
        tci::set_elem(ctx, tensor, coord, cytnx::cytnx_complex128(1.0, 0.0));

        // Get element back - should be exactly what we set
        auto elem = tci::get_elem(ctx, tensor, coord);
        CHECK(std::abs(elem.real() - 1.0) < 1e-10);
        CHECK(std::abs(elem.imag() - 0.0) < 1e-10);
    }

    SUBCASE("Norm calculation") {
        // Test with 3x3 identity matrix
        cytnx::Tensor identity;
        tci::eye(ctx, 3, identity);

        auto norm_val = tci::norm(ctx, identity);
        // Frobenius norm of 3x3 identity should be sqrt(3)
        CHECK(std::abs(norm_val - std::sqrt(3.0)) < 1e-10);
    }

    tci::destroy_context(ctx);
}