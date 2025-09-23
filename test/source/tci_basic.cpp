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

    SUBCASE("Create random tensor (in-place)") {
        tci::shape_t<cytnx::Tensor> shape = {2, 3};
        cytnx::Tensor tensor;
        std::size_t counter = 0;

        auto gen = [&]() -> cytnx::cytnx_complex128 {
            return cytnx::cytnx_complex128(static_cast<double>(counter++), 0.0);
        };

        CHECK_NOTHROW(tci::random(ctx, shape, gen, tensor));
        CHECK(counter == 6);

        auto result_shape = tci::shape(ctx, tensor);
        CHECK(result_shape == shape);

        auto elem_00 = tci::get_elem(ctx, tensor, {0, 0});
        CHECK(std::abs(elem_00.real() - 0.0) < 1e-10);
        CHECK(std::abs(elem_00.imag()) < 1e-10);

        auto elem_01 = tci::get_elem(ctx, tensor, {0, 1});
        CHECK(std::abs(elem_01.real() - 1.0) < 1e-10);

        auto elem_12 = tci::get_elem(ctx, tensor, {1, 2});
        CHECK(std::abs(elem_12.real() - 5.0) < 1e-10);
    }

    SUBCASE("Create random tensor (out-of-place)") {
        tci::shape_t<cytnx::Tensor> shape = {2, 2};
        std::size_t counter = 0;

        auto gen = [&]() -> cytnx::cytnx_complex128 {
            return cytnx::cytnx_complex128(static_cast<double>(counter++), 0.0);
        };

        cytnx::Tensor tensor;
        CHECK_NOTHROW(tensor = tci::random(ctx, shape, gen));
        CHECK(counter == 4);

        auto result_shape = tci::shape(ctx, tensor);
        CHECK(result_shape == shape);

        auto elem_11 = tci::get_elem(ctx, tensor, {1, 1});
        CHECK(std::abs(elem_11.real() - 3.0) < 1e-10);
        CHECK(std::abs(elem_11.imag()) < 1e-10);
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

TEST_CASE("TCI assign_from_container") {
    tci::context_handle_t<cytnx::Tensor> ctx;
    tci::create_context(ctx);

    SUBCASE("Create tensor from std::vector with row-major indexing") {
        // Create a 2x3 tensor from container
        std::vector<cytnx::cytnx_complex128> container = {
            cytnx::cytnx_complex128(1.0, 0.0), cytnx::cytnx_complex128(2.0, 0.0), cytnx::cytnx_complex128(3.0, 0.0),
            cytnx::cytnx_complex128(4.0, 0.0), cytnx::cytnx_complex128(5.0, 0.0), cytnx::cytnx_complex128(6.0, 0.0)
        };

        auto coors2idx = [](const tci::elem_coors_t<cytnx::Tensor> &coors) -> std::ptrdiff_t {
            return coors[0] * 3 + coors[1]; // row-major for 2x3 matrix
        };

        tci::shape_t<cytnx::Tensor> shape = {2, 3};
        cytnx::Tensor tensor;

        CHECK_NOTHROW(tci::assign_from_container(ctx, shape, container.begin(), coors2idx, tensor));

        // Verify tensor properties
        CHECK(tci::rank(ctx, tensor) == 2);
        auto result_shape = tci::shape(ctx, tensor);
        CHECK(result_shape[0] == 2);
        CHECK(result_shape[1] == 3);

        // Verify element values match container
        auto elem_00 = tci::get_elem(ctx, tensor, {0, 0});
        CHECK(std::abs(elem_00.real() - 1.0) < 1e-10);

        auto elem_01 = tci::get_elem(ctx, tensor, {0, 1});
        CHECK(std::abs(elem_01.real() - 2.0) < 1e-10);

        auto elem_10 = tci::get_elem(ctx, tensor, {1, 0});
        CHECK(std::abs(elem_10.real() - 4.0) < 1e-10);

        auto elem_12 = tci::get_elem(ctx, tensor, {1, 2});
        CHECK(std::abs(elem_12.real() - 6.0) < 1e-10);
    }

    SUBCASE("Create tensor from std::vector with custom indexing") {
        // Create a 2x2 tensor with column-major indexing
        std::vector<cytnx::cytnx_complex128> container = {
            cytnx::cytnx_complex128(1.0, 0.0), cytnx::cytnx_complex128(3.0, 0.0), // column 0
            cytnx::cytnx_complex128(2.0, 0.0), cytnx::cytnx_complex128(4.0, 0.0)  // column 1
        };

        auto coors2idx = [](const tci::elem_coors_t<cytnx::Tensor> &coors) -> std::ptrdiff_t {
            return coors[1] * 2 + coors[0]; // column-major for 2x2 matrix
        };

        tci::shape_t<cytnx::Tensor> shape = {2, 2};
        cytnx::Tensor tensor;

        CHECK_NOTHROW(tci::assign_from_container(ctx, shape, container.begin(), coors2idx, tensor));

        // Verify element values with column-major layout
        auto elem_00 = tci::get_elem(ctx, tensor, {0, 0});
        CHECK(std::abs(elem_00.real() - 1.0) < 1e-10);

        auto elem_01 = tci::get_elem(ctx, tensor, {0, 1});
        CHECK(std::abs(elem_01.real() - 2.0) < 1e-10);

        auto elem_10 = tci::get_elem(ctx, tensor, {1, 0});
        CHECK(std::abs(elem_10.real() - 3.0) < 1e-10);

        auto elem_11 = tci::get_elem(ctx, tensor, {1, 1});
        CHECK(std::abs(elem_11.real() - 4.0) < 1e-10);
    }

    SUBCASE("Out-of-place version") {
        std::vector<cytnx::cytnx_complex128> container = {
            cytnx::cytnx_complex128(7.0, 0.0), cytnx::cytnx_complex128(8.0, 0.0), cytnx::cytnx_complex128(9.0, 0.0)
        };

        auto coors2idx = [](const tci::elem_coors_t<cytnx::Tensor> &coors) -> std::ptrdiff_t {
            return coors[0]; // simple linear indexing for 1D tensor
        };

        tci::shape_t<cytnx::Tensor> shape = {3};

        cytnx::Tensor tensor;
        CHECK_NOTHROW(tensor = tci::assign_from_container<cytnx::Tensor>(ctx, shape, container.begin(), coors2idx));

        // Verify elements
        auto elem_0 = tci::get_elem(ctx, tensor, {0});
        CHECK(std::abs(elem_0.real() - 7.0) < 1e-10);

        auto elem_2 = tci::get_elem(ctx, tensor, {2});
        CHECK(std::abs(elem_2.real() - 9.0) < 1e-10);
    }

    tci::destroy_context(ctx);
}

TEST_CASE("TCI Tensor Manipulation") {
    tci::context_handle_t<cytnx::Tensor> ctx;
    tci::create_context(ctx);

    SUBCASE("Reshape operations") {
        tci::shape_t<cytnx::Tensor> original_shape = {2, 3, 4};
        cytnx::Tensor tensor;
        tci::ones(ctx, original_shape, tensor);

        // Test in-place reshape
        tci::shape_t<cytnx::Tensor> new_shape = {6, 4};
        CHECK_NOTHROW(tci::reshape(ctx, tensor, new_shape));

        auto result_shape = tci::shape(ctx, tensor);
        CHECK(result_shape == new_shape);
        CHECK(tci::size(ctx, tensor) == 24); // Same total size
    }

    SUBCASE("Transpose operations") {
        tci::shape_t<cytnx::Tensor> shape = {2, 3, 4};
        cytnx::Tensor tensor, transposed;
        tci::ones(ctx, shape, tensor);

        // Test out-of-place transpose
        std::vector<std::size_t> new_order = {2, 0, 1}; // 4,2,3
        CHECK_NOTHROW(tci::transpose(ctx, tensor, new_order, transposed));

        auto result_shape = tci::shape(ctx, transposed);
        CHECK(result_shape == std::vector<std::size_t>{4, 2, 3});
    }

    SUBCASE("Complex operations") {
        tci::shape_t<cytnx::Tensor> shape = {2, 2};
        cytnx::Tensor tensor;
        tci::fill(ctx, shape, cytnx::cytnx_complex128(2.0, 3.0), tensor);

        // Test complex conjugate
        cytnx::Tensor conjugated;
        CHECK_NOTHROW(tci::cplx_conj(ctx, tensor, conjugated));

        // Test real/imaginary part extraction
        tci::real_ten_t<cytnx::Tensor> real_part, imag_part;
        CHECK_NOTHROW(tci::real(ctx, tensor, real_part));
        CHECK_NOTHROW(tci::imag(ctx, tensor, imag_part));

        // Verify real part
        auto real_elem = tci::get_elem(ctx, real_part, {0, 0});
        CHECK(std::abs(real_elem.real() - 2.0) < 1e-10);

        // Verify imaginary part
        auto imag_elem = tci::get_elem(ctx, imag_part, {0, 0});
        CHECK(std::abs(imag_elem.real() - 3.0) < 1e-10);
    }

    tci::destroy_context(ctx);
}
