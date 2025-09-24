# TCI User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [API Reference](#api-reference)
5. [Advanced Usage](#advanced-usage)
6. [Performance Optimization](#performance-optimization)
7. [Debugging and Profiling](#debugging-and-profiling)
8. [Examples](#examples)

## Introduction

The Tensor Computing Interface (TCI) provides a unified, high-level API for tensor computations that can work with different backend tensor libraries. This implementation uses Cytnx as the backend, providing GPU acceleration and optimized numerical algorithms.

### Key Features

- **Universal Interface**: Same API works across different tensor backends
- **GPU Acceleration**: Automatic GPU support via Cytnx backend
- **Type Safety**: Template-based design with compile-time type checking
- **Performance**: Direct calls to optimized backend implementations
- **Debugging Support**: Built-in verbose mode and profiling tools

## Quick Start

### Basic Setup

```cpp
#include <tci/tci.h>

using Ten = cytnx::Tensor;
using namespace tci;

int main() {
    // Create context
    auto ctx = create_context<context_handle_t<Ten>>();

    // Your tensor operations here

    // Cleanup
    destroy_context(ctx);
    return 0;
}
```

### Environment Variables

Set these before running your program:

```bash
# Enable debug output
export TCI_VERBOSE=1  # Function calls only
export TCI_VERBOSE=2  # Function calls + timing

# macOS: Required for proper library linking
source setup_env.sh   # Sets up OpenBLAS, LLVM, libomp paths
```

## Core Concepts

### Contexts

All TCI operations require a context handle that manages the underlying tensor library state:

```cpp
auto ctx = create_context<context_handle_t<Ten>>();
// Use ctx for all operations
destroy_context(ctx);
```

### Type System

TCI uses a trait-based type system that automatically extracts type information:

```cpp
// These types are automatically deduced from your tensor type
using elem_t = tci::elem_t<Ten>;           // Element type (complex<double>)
using shape_t = tci::shape_t<Ten>;         // Shape type (vector<size_t>)
using context_handle_t = tci::context_handle_t<Ten>; // Context type
```

### Memory Management

TCI provides both in-place and out-of-place versions of most operations:

```cpp
// Out-of-place (creates new tensor)
Ten result = zeros<Ten>(ctx, {3, 4});

// In-place (modifies existing tensor)
Ten tensor = eye<Ten>(ctx, 3);
normalize(ctx, tensor); // Modifies tensor in-place
```

## API Reference

### Tensor Creation

```cpp
// Basic constructors
Ten zeros_tensor = zeros<Ten>(ctx, {3, 4, 5});
Ten ones_tensor = ones<Ten>(ctx, {2, 3});
Ten eye_tensor = eye<Ten>(ctx, 4);  // 4x4 identity matrix
Ten random_tensor = random<Ten>(ctx, {2, 2}, random_generator);

// From containers
std::vector<double> data = {1, 2, 3, 4, 5, 6};
auto coors2idx = [](const auto& coors) { return coors[0] * 3 + coors[1]; };
Ten from_container = assign_from_container<Ten>(ctx, {2, 3}, data.begin(), coors2idx);
```

### Information and Access

```cpp
// Tensor properties
auto r = rank(ctx, tensor);           // Number of dimensions
auto s = shape(ctx, tensor);          // Shape vector
auto total_size = size(ctx, tensor);  // Total number of elements
auto memory = size_bytes(ctx, tensor); // Memory usage in bytes

// Element access
auto element = get_elem(ctx, tensor, {1, 2, 0});  // Read element
set_elem(ctx, tensor, {1, 2, 0}, std::complex<double>(3.14, 0));  // Write element
```

### Linear Algebra

```cpp
// Basic operations
auto tensor_norm = norm(ctx, tensor);
auto normalized = normalize(ctx, tensor);
auto scaled = scale(ctx, tensor, 2.0);

// Matrix decompositions
Ten U, S, V_dag;
svd(ctx, matrix, 1, U, S, V_dag);  // Singular value decomposition

Ten eigenvals, eigenvecs;
eigh(ctx, symmetric_matrix, 1, eigenvals, eigenvecs);  // Eigendecomposition

// Tensor contractions
Ten result;
contract(ctx, A, {2}, B, {0}, result, {0, 1, 2, 3});  // Contract specified indices
contract(ctx, A, "ij", B, "jk", result, "ik");        // Einstein notation
```

### Tensor Manipulation

```cpp
// Shape operations
Ten reshaped = reshape(ctx, tensor, {6, 4});
Ten transposed = transpose(ctx, tensor, {1, 0, 2});

// Advanced operations
Ten expanded = expand(ctx, tensor, {{1, 2}});  // Expand dimension 1 by 2
Ten sub_tensor = extract_sub(ctx, tensor, {{0, 2}, {1, 3}});  // Extract subtensor
```

### I/O Operations

```cpp
// Save and load tensors
save(ctx, tensor, "data/tensor.dat");
Ten loaded = load<Ten>(ctx, "data/tensor.dat");
```

### Utility Functions

```cpp
// Display tensor
show(ctx, tensor);  // Print to stdout

// Compare tensors
bool equal = eq(ctx, tensor1, tensor2, 1e-10);  // With tolerance

// Convert to STL containers
std::vector<std::complex<double>> container(total_elements);
to_container(ctx, tensor, container.begin(), coordinate_mapping_function);

// Convert between contexts (e.g., CPU ↔ GPU)
Ten gpu_tensor;
convert(ctx_cpu, cpu_tensor, ctx_gpu, gpu_tensor);
```

## Advanced Usage

### Custom Coordinate Mappings

When using `to_container` or `assign_from_container`, you need to provide a coordinate mapping function:

```cpp
// Row-major mapping for 3D tensor with shape [A, B, C]
auto row_major = [B, C](const elem_coors_t<Ten>& coors) -> std::ptrdiff_t {
    return coors[0] * B * C + coors[1] * C + coors[2];
};

// Column-major mapping
auto col_major = [A, B](const elem_coors_t<Ten>& coors) -> std::ptrdiff_t {
    return coors[2] * A * B + coors[1] * A + coors[0];
};
```

### Working with Complex Numbers

```cpp
// TCI uses std::complex<double> for all elements
Ten complex_tensor = zeros<Ten>(ctx, {2, 2});
set_elem(ctx, complex_tensor, {0, 0}, std::complex<double>(1.0, 2.0));

// Extract real/imaginary parts
Ten real_part = real(ctx, complex_tensor);
Ten imag_part = imag(ctx, complex_tensor);

// Convert real to complex
Ten real_tensor = ones<Ten>(ctx, {2, 2});
Ten as_complex = to_cplx(ctx, real_tensor);
```

### Error Handling

TCI functions may throw exceptions for invalid operations:

```cpp
try {
    // This will throw if shapes don't match
    auto result = contract(ctx, A, {0}, B, {1}, C, {0, 1});
} catch (const std::exception& e) {
    std::cerr << "TCI Error: " << e.what() << std::endl;
}
```

## Performance Optimization

### Bond Dimension Management

For tensor network algorithms, managing bond dimensions is crucial:

```cpp
// Use truncated SVD to control bond dimensions
Ten U, S, V_dag;
real_t<Ten> trunc_error = 0.0;
size_t chi_min = 1, chi_max = 64;
double error_threshold = 1e-12;

trunc_svd(ctx, large_tensor, 1, U, S, V_dag, trunc_error,
          chi_min, chi_max, error_threshold, error_threshold);

std::cout << "Truncation error: " << trunc_error << std::endl;
```

### Memory Management

```cpp
// Prefer in-place operations when possible
normalize(ctx, tensor);  // In-place normalization

// Clear unused tensors
clear(ctx, unused_tensor);

// Check memory usage
auto memory_usage = size_bytes(ctx, tensor);
std::cout << "Memory usage: " << memory_usage / 1024 / 1024 << " MB" << std::endl;
```

### GPU Acceleration

```cpp
// Create GPU context (device ID >= 0)
auto gpu_ctx = create_context<context_handle_t<Ten>>();
// Note: GPU context creation depends on Cytnx configuration

// Transfer tensors between CPU and GPU
Ten gpu_tensor;
convert(cpu_ctx, cpu_tensor, gpu_ctx, gpu_tensor);

// Operations on GPU tensors automatically use GPU acceleration
auto result = contract(gpu_ctx, gpu_A, {1}, gpu_B, {0}, gpu_result, {0, 1});
```

## Debugging and Profiling

### Verbose Output

Set `TCI_VERBOSE` environment variable:

```bash
# Show function calls only
TCI_VERBOSE=1 ./my_program

# Show function calls and timing information
TCI_VERBOSE=2 ./my_program
```

Example output:
```
[TCI] tci::svd - shape=[100,50], dtype=10
[TCI] tci::svd took 15.234 ms
```

### Manual Timing

```cpp
#include <chrono>

auto start = std::chrono::high_resolution_clock::now();

// Your TCI operations here
auto result = contract(ctx, A, {1}, B, {0}, C, {0, 1});

auto end = std::chrono::high_resolution_clock::now();
auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
std::cout << "Operation took: " << duration.count() << " ms" << std::endl;
```

### Memory Profiling

```cpp
// Track memory usage over time
std::vector<size_t> memory_history;

for (int i = 0; i < num_iterations; ++i) {
    // Your operations

    // Record memory usage
    size_t total_memory = 0;
    for (const auto& tensor : all_tensors) {
        total_memory += size_bytes(ctx, tensor);
    }
    memory_history.push_back(total_memory);
}

// Analyze memory growth
auto max_memory = *std::max_element(memory_history.begin(), memory_history.end());
std::cout << "Peak memory usage: " << max_memory / 1024 / 1024 << " MB" << std::endl;
```

## Examples

### Complete Example: Matrix Multiplication Chain

```cpp
#include <tci/tci.h>
#include <iostream>
#include <vector>

using Ten = cytnx::Tensor;
using namespace tci;

int main() {
    // Setup
    auto ctx = create_context<context_handle_t<Ten>>();

    // Create random matrices
    auto rand_gen = []() { return std::complex<double>(std::rand() / double(RAND_MAX), 0); };

    Ten A = random<Ten>(ctx, {100, 50}, rand_gen);
    Ten B = random<Ten>(ctx, {50, 30}, rand_gen);
    Ten C = random<Ten>(ctx, {30, 20}, rand_gen);

    std::cout << "Computing A(100×50) × B(50×30) × C(30×20)" << std::endl;

    // Method 1: Step by step
    Ten AB, ABC;
    contract(ctx, A, {1}, B, {0}, AB, {0, 1});    // A × B
    contract(ctx, AB, {1}, C, {0}, ABC, {0, 1});  // (A × B) × C

    std::cout << "Result shape: " << shape(ctx, ABC)[0] << "×" << shape(ctx, ABC)[1] << std::endl;
    std::cout << "Result norm: " << norm(ctx, ABC) << std::endl;

    // Method 2: Einstein notation
    Ten ABC_einstein;
    // This would require a 3-tensor contraction - use step-by-step for now

    // Cleanup
    destroy_context(ctx);

    return 0;
}
```

### Complete Example: SVD Analysis

```cpp
#include <tci/tci.h>
#include <iostream>
#include <vector>
#include <iomanip>

using Ten = cytnx::Tensor;
using namespace tci;

int main() {
    auto ctx = create_context<context_handle_t<Ten>>();

    // Create a low-rank matrix (rank 3 in 10×8 matrix)
    Ten A = zeros<Ten>(ctx, {10, 8});

    // Fill with rank-3 structure: A = U * S * V^T
    // where U is 10×3, S is 3×3 diagonal, V^T is 3×8

    auto rand_gen = []() { return std::complex<double>(std::rand() / double(RAND_MAX) - 0.5, 0); };
    Ten U_small = random<Ten>(ctx, {10, 3}, rand_gen);
    Ten V_small = random<Ten>(ctx, {3, 8}, rand_gen);

    // Create diagonal matrix with specific singular values
    Ten S_small = zeros<Ten>(ctx, {3, 3});
    set_elem(ctx, S_small, {0, 0}, std::complex<double>(10.0, 0));
    set_elem(ctx, S_small, {1, 1}, std::complex<double>(5.0, 0));
    set_elem(ctx, S_small, {2, 2}, std::complex<double>(1.0, 0));

    // Construct A = U * S * V
    Ten US, A_constructed;
    contract(ctx, U_small, {1}, S_small, {0}, US, {0, 1});
    contract(ctx, US, {1}, V_small, {0}, A_constructed, {0, 1});

    std::cout << "Constructed 10×8 matrix with known rank-3 structure" << std::endl;
    std::cout << "Expected singular values: 10.0, 5.0, 1.0, 0.0, ..." << std::endl;

    // Perform SVD
    Ten U, S, V_dag;
    svd(ctx, A_constructed, 1, U, S, V_dag);

    std::cout << "SVD Results:" << std::endl;
    std::cout << "Singular values: ";

    auto s_shape = shape(ctx, S);
    for (size_t i = 0; i < s_shape[0]; ++i) {
        auto sv = get_elem(ctx, S, {static_cast<elem_coor_t<Ten>>(i)});
        std::cout << std::setprecision(3) << sv.real() << " ";
    }
    std::cout << std::endl;

    // Test truncated SVD
    Ten U_trunc, S_trunc, V_dag_trunc;
    real_t<Ten> trunc_error = 0.0;

    trunc_svd(ctx, A_constructed, 1, U_trunc, S_trunc, V_dag_trunc, trunc_error,
              1, 3, 1e-10, 1e-14); // Keep only top 3 singular values

    std::cout << "Truncated SVD (keeping 3 singular values):" << std::endl;
    std::cout << "Truncation error: " << trunc_error << std::endl;

    // Verify reconstruction
    Ten USV_trunc;
    Ten US_trunc;
    Ten S_trunc_diag = diag(ctx, S_trunc);

    contract(ctx, U_trunc, {1}, S_trunc_diag, {0}, US_trunc, {0, 1});
    contract(ctx, US_trunc, {1}, V_dag_trunc, {0}, USV_trunc, {0, 1});

    // Check reconstruction error
    Ten diff = USV_trunc - A_constructed;  // Should be very small
    auto reconstruction_error = norm(ctx, diff);

    std::cout << "Reconstruction error: " << reconstruction_error << std::endl;
    std::cout << "(Should be approximately equal to truncation error)" << std::endl;

    destroy_context(ctx);
    return 0;
}
```

### Running Examples

To run the included examples:

```bash
# Build with examples enabled
cmake -S . -B build -DBUILD_EXAMPLES=ON
cmake --build build --parallel 4

# Run simple operations example
./build/examples/simple_tensor_operations

# Run with verbose output
TCI_VERBOSE=2 ./build/examples/simple_tensor_operations
```

## Troubleshooting

### Common Issues

1. **Missing BLAS libraries**: Make sure OpenBLAS is installed and environment variables are set
2. **Linking errors**: Use the provided `setup_env.sh` script on macOS
3. **Runtime crashes**: Check tensor shapes are compatible for operations
4. **Poor performance**: Enable GPU support and use appropriate bond dimensions

### Getting Help

1. Check the error message for specific function and parameter information
2. Enable `TCI_VERBOSE=2` to see detailed function call information
3. Verify tensor shapes using `tci::shape()` before operations
4. Use `tci::show()` to inspect tensor contents

For more examples and advanced usage patterns, see the `test/` directory in the source code, particularly the iTEBD integration test which demonstrates a complete real-world application.