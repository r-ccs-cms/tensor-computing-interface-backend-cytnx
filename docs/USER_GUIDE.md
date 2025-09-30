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
#include <tci/cytnx_typed_tensor.h>
#include <tci/cytnx_typed_tensor_impl.h>

// Use CytnxTensor for type-safe operations
using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
using Elem = tci::elem_t<Tensor>;  // = std::complex<double>

int main() {
    // Create context (CPU)
    int ctx = -1;

    // Your tensor operations here

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

### CytnxTensor: Type-Safe Tensor Wrapper

TCI provides `CytnxTensor<ElemT>`, a type-safe wrapper around Cytnx tensors that enables direct arithmetic operations on elements.

#### Key Features

- **Direct Arithmetic**: Use `+`, `-`, `*`, `/`, `std::sqrt`, `std::exp`, etc. on elements
- **Compile-Time Type Safety**: Element type fixed at compile time
- **Clean Code**: No need for `std::visit` or variant handling

#### Basic Usage

```cpp
using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
using Elem = tci::elem_t<Tensor>;  // = std::complex<double>

int ctx = -1;  // CPU context

// Create and allocate tensor
Tensor tensor;
tci::allocate(ctx, {100, 100}, tensor);

// Direct element access
auto elem = tci::get_elem(ctx, tensor, {0, 0});
// elem is std::complex<double>, can use directly!
elem = elem * 2.0 + std::complex<double>(1.0, 0.0);
```

#### Element-wise Operations with for_each

```cpp
// Apply mathematical functions to all elements
tci::for_each(ctx, tensor, [](Elem& elem) {
    elem = std::sqrt(elem) * 2.0;
    elem = elem + Elem{1.0, 0.5};
    elem = elem / std::abs(elem);  // Normalize
});

// Conditional operations
tci::for_each(ctx, tensor, [](Elem& elem) {
    double magnitude = std::abs(elem);
    if (magnitude > 1e-10) {
        elem = elem / magnitude;  // Normalize non-zero elements
    } else {
        elem = Elem{0.0, 0.0};
    }
});
```

#### Available Element Types

```cpp
// Double precision complex (recommended)
using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;  // std::complex<double>

// Single precision complex
using Tensor = tci::CytnxTensor<cytnx::cytnx_complex64>;   // std::complex<float>

// Double precision real
using Tensor = tci::CytnxTensor<cytnx::cytnx_double>;      // double

// Single precision real
using Tensor = tci::CytnxTensor<cytnx::cytnx_float>;       // float
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
using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
int ctx = -1;  // CPU

// Allocate tensor with shape
Tensor tensor;
tci::allocate(ctx, {3, 4, 5}, tensor);

// Or use the return-value version
auto tensor2 = tci::allocate<Tensor>(ctx, {2, 3});
```

**Note**: For `cytnx::Tensor` (advanced users), use the template-based creation:
```cpp
using Ten = cytnx::Tensor;
auto ctx = tci::create_context<tci::context_handle_t<Ten>>();
Ten zeros_tensor = tci::zeros<Ten>(ctx, {3, 4, 5});
```

### Information and Access

```cpp
using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
using Elem = tci::elem_t<Tensor>;

// Tensor properties (via backend)
auto shape = tensor.backend.shape();  // Get shape
auto rank = shape.size();             // Number of dimensions

// Element access through TCI
auto element = tci::get_elem(ctx, tensor, {1, 2, 0});  // Read element
// element is Elem (std::complex<double>), can use directly!

// Direct backend access for setting
tensor.backend.at<Elem>({1, 2, 0}) = Elem{3.14, 0.0};
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

### Using cytnx::Tensor (Dynamic Typing)

For advanced users who need runtime type flexibility or full Cytnx API compatibility, `cytnx::Tensor` is available:

```cpp
#include <tci/tci.h>

using Ten = cytnx::Tensor;
auto ctx = tci::create_context<tci::context_handle_t<Ten>>();

// Create tensors with template-based API
Ten zeros_tensor = tci::zeros<Ten>(ctx, {3, 4});
Ten eye_tensor = tci::eye<Ten>(ctx, 3);

// Element access returns std::variant
auto elem = tci::get_elem(ctx, zeros_tensor, {0, 0});
// elem is std::variant<double, float, complex<double>, complex<float>>

// Must use std::visit for access
std::visit([](auto&& val) {
    std::cout << "Element: " << val << std::endl;
}, elem);

tci::destroy_context(ctx);
```

**When to use cytnx::Tensor:**
- You need runtime type flexibility
- You're integrating with existing Cytnx code
- You need advanced Cytnx features not yet supported by CytnxTensor

**Limitation:** Cannot use arithmetic operators directly on `elem_t` (it's a variant).

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

### Complete Example: Element-wise Operations

```cpp
#include <tci/tci.h>
#include <tci/cytnx_typed_tensor.h>
#include <tci/cytnx_typed_tensor_impl.h>
#include <iostream>
#include <cmath>

using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
using Elem = tci::elem_t<Tensor>;

int main() {
    int ctx = -1;  // CPU

    // Create tensor
    Tensor tensor;
    tci::allocate(ctx, {100, 100}, tensor);

    // Initialize with random-like values
    for (size_t i = 0; i < 100; ++i) {
        for (size_t j = 0; j < 100; ++j) {
            double val = (i * 100 + j) * 0.01;
            tensor.backend.at<Elem>({i, j}) = Elem{val, val * 0.5};
        }
    }

    std::cout << "Applying element-wise operations..." << std::endl;

    // Apply mathematical transformations
    tci::for_each(ctx, tensor, [](Elem& elem) {
        // Square root
        elem = std::sqrt(elem);
        // Scale
        elem = elem * 2.0;
        // Add constant
        elem = elem + Elem{1.0, 0.5};
    });

    // Normalize all elements
    tci::for_each(ctx, tensor, [](Elem& elem) {
        double mag = std::abs(elem);
        if (mag > 1e-10) {
            elem = elem / mag;
        }
    });

    std::cout << "Operations completed!" << std::endl;

    return 0;
}
```

### Complete Example: Conditional Processing

```cpp
#include <tci/tci.h>
#include <tci/cytnx_typed_tensor.h>
#include <tci/cytnx_typed_tensor_impl.h>
#include <iostream>

using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
using Elem = tci::elem_t<Tensor>;

int main() {
    int ctx = -1;

    // Create and initialize tensor
    Tensor data;
    tci::allocate(ctx, {1000}, data);

    // Fill with varying values
    for (size_t i = 0; i < 1000; ++i) {
        double x = (i - 500) * 0.01;
        data.backend.at<Elem>({i}) = Elem{x, x * x};
    }

    std::cout << "Processing data with conditional logic..." << std::endl;

    // Count elements by category
    size_t small_count = 0, medium_count = 0, large_count = 0;

    tci::for_each(ctx, static_cast<const Tensor&>(data),
                  [&](const Elem& elem) {
        double magnitude = std::abs(elem);
        if (magnitude < 1.0) {
            small_count++;
        } else if (magnitude < 5.0) {
            medium_count++;
        } else {
            large_count++;
        }
    });

    std::cout << "Small magnitude: " << small_count << std::endl;
    std::cout << "Medium magnitude: " << medium_count << std::endl;
    std::cout << "Large magnitude: " << large_count << std::endl;

    // Clip extreme values
    tci::for_each(ctx, data, [](Elem& elem) {
        double magnitude = std::abs(elem);
        if (magnitude > 10.0) {
            elem = elem / magnitude * 10.0;  // Clip to magnitude 10
        }
    });

    std::cout << "Clipped extreme values" << std::endl;

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