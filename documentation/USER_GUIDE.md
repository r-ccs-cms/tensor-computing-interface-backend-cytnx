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

// Use CytnxTensor for type-safe operations
using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
using Elem = tci::elem_t<Tensor>;  // = std::complex<double>

int main() {
    // Create context (CPU only in current implementation)
    tci::context_handle_t<Tensor> ctx;
    tci::create_context(ctx);

    // Your tensor operations here

    tci::destroy_context(ctx);
    return 0;
}
```

### Environment Variables

Set these before running your program:

```bash
# Enable debug output
export TCI_VERBOSE=1  # Function calls only
export TCI_VERBOSE=2  # Function calls + timing
```

## Core Concepts

### Contexts

TCI operations require a context handle. In the Cytnx backend implementation, the context is simply an integer representing the device ID (CPU=-1, GPU=0,1,2...).

**Note:** The current implementation only supports CPU context. GPU context creation is not yet implemented.

```cpp
// Create CPU context (specify backend tensor type)
using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
tci::context_handle_t<Tensor> ctx;
tci::create_context(ctx);

// Use ctx for all operations
// (In current implementation, ctx is rarely used by functions)

tci::destroy_context(ctx);
```

### Type System

TCI uses a trait-based type system that automatically extracts type information:

```cpp
// These types are automatically deduced from your tensor type
using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
using elem_t = tci::elem_t<Tensor>;                  // Element type (complex<double>)
using shape_t = tci::shape_t<Tensor>;                // Shape type (vector<size_t>)
using context_handle_t = tci::context_handle_t<Tensor>;  // Context type
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

tci::context_handle_t<Tensor> ctx;
tci::create_context(ctx);

// Create and allocate tensor
Tensor tensor;
tci::allocate(ctx, {100, 100}, tensor);

// Direct element access
auto elem = tci::get_elem(ctx, tensor, {0, 0});
// elem is std::complex<double>, can use directly!
elem = elem * 2.0 + Elem{1.0, 0.0};
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
using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;

// Out-of-place (creates new tensor)
Tensor result = zeros<Tensor>(ctx, {3, 4});

// In-place (modifies existing tensor)
Tensor tensor = eye<Tensor>(ctx, 3);
normalize(ctx, tensor); // Modifies tensor in-place
```

## API Reference

### Tensor Creation

```cpp
using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
tci::context_handle_t<Tensor> ctx;
tci::create_context(ctx);

// Allocate tensor with shape
Tensor tensor;
tci::allocate(ctx, {3, 4, 5}, tensor);

// Or use the return-value version
auto tensor2 = tci::allocate<Tensor>(ctx, {2, 3});
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
using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;

// Basic operations
auto tensor_norm = norm(ctx, tensor);
auto normalized = normalize(ctx, tensor);
auto scaled = scale(ctx, tensor, 2.0);

// Matrix decompositions
Tensor U, V_dag;
real_ten_t<Tensor> S;
svd(ctx, matrix, 1, U, S, V_dag);  // Singular value decomposition

Tensor eigenvecs;
real_ten_t<Tensor> eigenvals;
eigh(ctx, symmetric_matrix, 1, eigenvals, eigenvecs);  // Eigendecomposition

// Tensor contractions
Tensor result;
contract(ctx, A, {2}, B, {0}, result, {0, 1, 2, 3});  // Contract specified indices
contract(ctx, A, "ij", B, "jk", result, "ik");        // Einstein notation
```

### Tensor Manipulation

```cpp
using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;

// Shape operations
Tensor reshaped = reshape(ctx, tensor, {6, 4});
Tensor transposed = transpose(ctx, tensor, {1, 0, 2});

// Advanced operations
Tensor expanded = expand(ctx, tensor, {{1, 2}});  // Expand dimension 1 by 2
Tensor sub_tensor = extract_sub(ctx, tensor, {{0, 2}, {1, 3}});  // Extract subtensor
```

### I/O Operations

```cpp
using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;

// Save and load tensors
save(ctx, tensor, "data/tensor.dat");
Tensor loaded = load<Tensor>(ctx, "data/tensor.dat");
```

### Utility Functions

```cpp
using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;

// Display tensor
show(ctx, tensor);  // Print to stdout

// Compare tensors
bool equal = eq(ctx, tensor1, tensor2, 1e-10);  // With tolerance

// Convert to STL containers
std::vector<std::complex<double>> container(total_elements);
to_container(ctx, tensor, container.begin(), coordinate_mapping_function);

// Convert between contexts (e.g., CPU ↔ GPU)
Tensor gpu_tensor;
convert(ctx_cpu, cpu_tensor, ctx_gpu, gpu_tensor);
```

## Advanced Usage

### Custom Coordinate Mappings

When using `to_container` or `assign_from_container`, you need to provide a coordinate mapping function:

```cpp
using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;

// Row-major mapping for 3D tensor with shape [A, B, C]
auto row_major = [B, C](const elem_coors_t<Tensor>& coors) -> std::ptrdiff_t {
    return coors[0] * B * C + coors[1] * C + coors[2];
};

// Column-major mapping
auto col_major = [A, B](const elem_coors_t<Tensor>& coors) -> std::ptrdiff_t {
    return coors[2] * A * B + coors[1] * A + coors[0];
};
```

### Working with Complex Numbers

```cpp
using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
using Elem = tci::elem_t<Tensor>;
tci::context_handle_t<Tensor> ctx;
tci::create_context(ctx);

// Create complex tensor
Tensor complex_tensor;
tci::allocate(ctx, {2, 2}, complex_tensor);

// Set complex values using Elem type alias (recommended)
complex_tensor.backend.at<Elem>({0, 0}) = Elem{1.0, 2.0};

// Extract real/imaginary parts
auto real_part = tci::real(ctx, complex_tensor);
auto imag_part = tci::imag(ctx, complex_tensor);

// Convert real to complex
using RealTensor = tci::real_ten_t<Tensor>;
RealTensor real_tensor = tci::ones<RealTensor>(ctx, {2, 2});
Tensor as_complex = tci::to_cplx(ctx, real_tensor);
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
using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;

// Use truncated SVD to control bond dimensions
Tensor U, V_dag;
real_ten_t<Tensor> S;
real_t<Tensor> trunc_error = 0.0;
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

**Note:** GPU support is not yet implemented in the current version of TCI with Cytnx backend.

- `create_context()` always returns CPU context (-1)
- GPU context creation is planned for future releases
- The `convert()` function has device transfer capability, but GPU contexts are not available yet

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
#include <iostream>
#include <cmath>

using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
using Elem = tci::elem_t<Tensor>;

int main() {
    tci::context_handle_t<Tensor> ctx;
    tci::create_context(ctx);

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

    tci::destroy_context(ctx);
    return 0;
}
```

### Complete Example: Conditional Processing

```cpp
#include <tci/tci.h>
#include <iostream>

using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
using Elem = tci::elem_t<Tensor>;

int main() {
    tci::context_handle_t<Tensor> ctx;
    tci::create_context(ctx);

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

    tci::destroy_context(ctx);
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

