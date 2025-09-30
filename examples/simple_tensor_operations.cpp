/**
 * @file simple_tensor_operations.cpp
 * @brief Simple examples demonstrating TCI usage for basic tensor operations
 *
 * This file contains practical examples showing how to use the TCI
 * (Tensor Computing Interface) for common tensor computations.
 */

#include <tci/tci.h>
#include <iostream>
#include <vector>
#include <complex>
#include <iomanip>
#include <random>
#include <functional>

using Ten = cytnx::Tensor;
using namespace tci;

/**
 * @brief Demonstrate basic tensor creation and manipulation
 */
void basic_tensor_operations() {
    std::cout << "\n=== Basic Tensor Operations ===" << std::endl;

    // Create context
    auto ctx = create_context<context_handle_t<Ten>>();

    // Create tensors of different types
    Ten zeros_tensor = zeros<Ten>(ctx, {3, 4});
    Ten ones_tensor = fill<Ten>(ctx, {2, 3}, std::complex<double>(1.0, 0.0));
    Ten eye_tensor = eye<Ten>(ctx, 4);

    // High-quality random tensor generation using MT19937 and uniform distribution
    // Header-only template implementation allows direct lambda usage without linkage issues
    std::mt19937 rng(std::random_device{}());
    std::uniform_real_distribution<double> dist(-1.0, 1.0);
    auto random_gen1 = [&rng, &dist]() { return std::complex<double>(dist(rng), dist(rng)); };
    Ten random_tensor = random<Ten>(ctx, {2, 2, 2}, random_gen1);

    std::cout << "Created tensors with shapes:" << std::endl;
    std::cout << "  zeros: ";
    auto zero_shape = shape(ctx, zeros_tensor);
    for (auto dim : zero_shape) std::cout << dim << " ";
    std::cout << std::endl;

    std::cout << "  identity: " << shape(ctx, eye_tensor)[0] << "×" << shape(ctx, eye_tensor)[1] << std::endl;
    std::cout << "  random: rank-" << rank(ctx, random_tensor) << " tensor" << std::endl;

    // Demonstrate element access
    std::cout << "\nElement access:" << std::endl;
    auto eye_elem = get_elem(ctx, eye_tensor, {1, 1});
    // Note: elem_t for cytnx::Tensor is std::variant
    std::visit([](auto&& val) {
      std::cout << "  eye[1,1] = " << std::real(val) << " + " << std::imag(val) << "i" << std::endl;
    }, eye_elem);

    // Set an element
    set_elem(ctx, zeros_tensor, {1, 2}, std::complex<double>(3.14, 0));
    auto set_elem_val = get_elem(ctx, zeros_tensor, {1, 2});
    std::visit([](auto&& val) {
      std::cout << "  After setting zeros[1,2] = π: " << std::real(val) << std::endl;
    }, set_elem_val);

    // Cleanup
    destroy_context(ctx);
}

/**
 * @brief Demonstrate linear algebra operations
 */
void linear_algebra_operations() {
    std::cout << "\n=== Linear Algebra Operations ===" << std::endl;

    auto ctx = create_context<context_handle_t<Ten>>();

    // Create a test matrix
    Ten A = zeros<Ten>(ctx, {3, 3});

    // Fill with some interesting values: A = [[2, 1, 0], [1, 2, 1], [0, 1, 2]]
    set_elem(ctx, A, {0, 0}, std::complex<double>(2, 0));
    set_elem(ctx, A, {0, 1}, std::complex<double>(1, 0));
    set_elem(ctx, A, {1, 0}, std::complex<double>(1, 0));
    set_elem(ctx, A, {1, 1}, std::complex<double>(2, 0));
    set_elem(ctx, A, {1, 2}, std::complex<double>(1, 0));
    set_elem(ctx, A, {2, 1}, std::complex<double>(1, 0));
    set_elem(ctx, A, {2, 2}, std::complex<double>(2, 0));

    std::cout << "Matrix A created (3×3 tridiagonal)" << std::endl;

    // Calculate norm
    auto matrix_norm = norm(ctx, A);
    std::cout << "  Frobenius norm: " << matrix_norm << std::endl;

    // SVD decomposition
    Ten U, S, V_dag;
    svd(ctx, A, 1, U, S, V_dag);  // Treat first index as row

    std::cout << "  SVD completed" << std::endl;
    std::cout << "    U shape: " << shape(ctx, U)[0] << "×" << shape(ctx, U)[1] << std::endl;
    std::cout << "    S shape: " << shape(ctx, S)[0] << " (singular values)" << std::endl;
    std::cout << "    V† shape: " << shape(ctx, V_dag)[0] << "×" << shape(ctx, V_dag)[1] << std::endl;

    // Print singular values
    std::cout << "  Singular values: ";
    for (size_t i = 0; i < shape(ctx, S)[0]; ++i) {
        auto sv = get_elem(ctx, S, {static_cast<elem_coor_t<Ten>>(i)});
        std::visit([](auto&& val) {
            std::cout << std::setprecision(3) << std::real(val) << " ";
        }, sv);
    }
    std::cout << std::endl;

    // Eigenvalue decomposition (for symmetric matrix)
    Ten eigenvals, eigenvecs;
    eigh(ctx, A, 1, eigenvals, eigenvecs);

    std::cout << "  Eigenvalues: ";
    for (size_t i = 0; i < shape(ctx, eigenvals)[0]; ++i) {
        auto ev = get_elem(ctx, eigenvals, {static_cast<elem_coor_t<Ten>>(i)});
        std::visit([](auto&& val) {
            std::cout << std::setprecision(3) << std::real(val) << " ";
        }, ev);
    }
    std::cout << std::endl;

    destroy_context(ctx);
}

/**
 * @brief Demonstrate tensor contractions
 */
void tensor_contraction_example() {
    std::cout << "\n=== Tensor Contractions ===" << std::endl;

    auto ctx = create_context<context_handle_t<Ten>>();

    // Create tensors for contraction
    std::mt19937 rng2(std::random_device{}());
    std::uniform_real_distribution<double> dist2(-1.0, 1.0);
    auto random_gen2 = [&rng2, &dist2]() { return std::complex<double>(dist2(rng2), dist2(rng2)); };
    Ten A = random<Ten>(ctx, {3, 4, 5}, random_gen2);
    Ten B = random<Ten>(ctx, {5, 6, 7}, random_gen2);

    std::cout << "Created tensors:" << std::endl;
    std::cout << "  A: 3×4×5" << std::endl;
    std::cout << "  B: 5×6×7" << std::endl;

    // Matrix multiplication: contract over shared dimension
    Ten C;
    contract(ctx, A, {2}, B, {0}, C, {0, 1, 2, 3}); // Contract A's last index with B's first index

    auto c_shape = shape(ctx, C);
    std::cout << "  C = A×B: ";
    for (auto dim : c_shape) std::cout << dim << "×";
    std::cout << "\b " << std::endl; // Remove last ×

    // More complex contraction using Einstein notation
    std::mt19937 rng3(std::random_device{}());
    std::uniform_real_distribution<double> dist3(-1.0, 1.0);
    auto random_gen3 = [&rng3, &dist3]() { return std::complex<double>(dist3(rng3), dist3(rng3)); };
    Ten D = random<Ten>(ctx, {3, 4}, random_gen3);
    Ten E = random<Ten>(ctx, {4, 5}, random_gen3);

    Ten F;
    // Einstein notation: "ij,jk->ik" (matrix multiplication)
    contract(ctx, D, "ij", E, "jk", F, "ik");

    std::cout << "  Matrix multiplication via Einstein notation:" << std::endl;
    std::cout << "    D(3×4) × E(4×5) = F(" << shape(ctx, F)[0] << "×" << shape(ctx, F)[1] << ")" << std::endl;

    destroy_context(ctx);
}

/**
 * @brief Demonstrate I/O and utility functions
 */
void io_and_utilities() {
    std::cout << "\n=== I/O and Utility Functions ===" << std::endl;

    auto ctx = create_context<context_handle_t<Ten>>();

    // Create a test tensor
    Ten test_tensor = eye<Ten>(ctx, 3);

    std::cout << "Test tensor (3×3 identity):" << std::endl;
    show(ctx, test_tensor); // Print tensor

    // Test equality comparison
    Ten test_tensor2 = eye<Ten>(ctx, 3);
    bool are_equal = eq(ctx, test_tensor, test_tensor2, std::complex<double>(1e-10, 0));
    std::cout << "  Tensors are equal: " << (are_equal ? "true" : "false") << std::endl;

    // Convert to STL container functionality disabled due to template linkage issues
    // to_container function suffers from same lambda linkage problem as random() previously did
    // Requires similar header-only implementation to resolve template instantiation with lambdas
    /*
    std::vector<std::complex<double>> container(9);
    auto row_major_map = [](const elem_coors_t<Ten>& coors) -> std::ptrdiff_t {
        return coors[0] * 3 + coors[1]; // 3×3 tensor, row-major mapping function
    };

    to_container(ctx, test_tensor, container.begin(), row_major_map);
    */

    // std::cout << "  Converted to std::vector: ";
    // for (size_t i = 0; i < 9; ++i) {
    //     std::cout << std::setprecision(1) << container[i].real() << " ";
    //     if ((i + 1) % 3 == 0) std::cout << "| ";
    // }
    // std::cout << std::endl;

    // Memory usage information
    auto memory_usage = size_bytes(ctx, test_tensor);
    std::cout << "  Memory usage: " << memory_usage << " bytes" << std::endl;

    destroy_context(ctx);
}

/**
 * @brief Demonstrate TCI_VERBOSE environment variable
 */
void demonstrate_verbose_output() {
    std::cout << "\n=== TCI_VERBOSE Demonstration ===" << std::endl;
    std::cout << "Set TCI_VERBOSE environment variable to see debug output:" << std::endl;
    std::cout << "  TCI_VERBOSE=1  - Function calls" << std::endl;
    std::cout << "  TCI_VERBOSE=2  - Function calls + timing" << std::endl;

    auto ctx = create_context<context_handle_t<Ten>>();

    // These operations will show debug output if TCI_VERBOSE is set
    Ten A = eye<Ten>(ctx, 2);
    Ten B = zeros<Ten>(ctx, {2, 2});

    // This will trigger TCI_VERBOSE output for show, eq, and convert functions
    show(ctx, A);
    bool equal = eq(ctx, A, B, std::complex<double>(1e-6, 0));

    Ten C;
    auto ctx2 = create_context<context_handle_t<Ten>>();
    convert(ctx, A, ctx2, C);

    destroy_context(ctx2);
    destroy_context(ctx);
}

int main() {
    std::cout << "TCI (Tensor Computing Interface) - Usage Examples" << std::endl;
    std::cout << "=================================================" << std::endl;

    basic_tensor_operations();
    linear_algebra_operations();
    tensor_contraction_example();
    io_and_utilities();
    demonstrate_verbose_output();

    std::cout << "\n=== All Examples Completed Successfully! ===" << std::endl;
    return 0;
}