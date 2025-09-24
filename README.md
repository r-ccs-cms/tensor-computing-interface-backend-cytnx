# TCI (Tensor Computing Interface) - Cytnx Backend

TCI (Tensor Computing Interface) is a universal C++ interface for tensor computing libraries. This repository provides the Cytnx backend implementation, allowing TCI APIs to work with the [Cytnx](https://github.com/Cytnx-dev/Cytnx) tensor library.

## Features

- **Universal Tensor Interface**: Consistent API across different tensor libraries
- **Cytnx Integration**: Full integration with Cytnx tensor operations
- **Template-Based Design**: Type-safe generic programming interface
- **High Performance**: Direct calls to optimized Cytnx backend
- **Modern CMake**: Clean build system with dependency management

## C++ Standard Requirements and Important Caveats

### User Code Requirements

- **User code**: Must be compiled with **C++17** standard
- **TCI Public API**: Designed to be fully C++17 compatible

### Implementation Requirements

- **TCI Implementation**: Uses C++17 features only, but requires C++20 compilation
- **Cytnx Backend**: Requires C++20 standard for compilation
- **Build System**: CMake with C++20 due to Cytnx dependency

### Why This Mixed Approach?

The TCI implementation acts as a **translation layer** between the C++17 user interface and the C++20 Cytnx library:

```
[User Code (C++17)] → [TCI API (C++17)] → [TCI Implementation (C++17 features, C++20 compile)] → [Cytnx (C++20)]
                                                     ↑
                                              Abstraction boundary
```

**Key Points:**

1. **TCI implementation code** uses only C++17 language features
2. **Compilation requires C++20** due to Cytnx header dependencies
3. **User projects** remain fully C++17 compatible
4. **No C++20 features leak** into the public TCI API

This design ensures maximum compatibility while leveraging Cytnx's optimized implementations.

## Dependencies

### Required

- **CMake 3.24+**: Required for Cytnx
- **C++20 capable compiler**: GCC 10+, Clang 12+, or MSVC 2019+
- **BLAS/LAPACK**: OpenBLAS recommended
- **Cytnx**: Included as git submodule

### Optional

- **OpenMP**: For parallel computations
- **CUDA**: For GPU acceleration (if supported by Cytnx)

## Quick Start

### 1. Clone with Submodules

```bash
git clone --recursive https://github.com/your-org/tensor-computing-interface-backend-cytnx.git
cd tensor-computing-interface-backend-cytnx
```

### 2. Install Dependencies (macOS with Homebrew)

```bash
brew install openblas llvm libomp cmake
```

### 3. Configure and Build

```bash
# Set environment variables for proper library linking
export CPPFLAGS="-I/opt/homebrew/opt/openblas/include -I/opt/homebrew/opt/llvm/include -I/opt/homebrew/opt/libomp/include"
export LDFLAGS="-L/opt/homebrew/opt/openblas/lib -L/opt/homebrew/opt/llvm/lib -L/opt/homebrew/opt/libomp/lib"

# Configure with CMake
cmake -S . -B build \
  -DBLAS_LIBRARIES="/opt/homebrew/opt/openblas/lib/libopenblas.dylib" \
  -DLAPACK_LIBRARIES="/opt/homebrew/opt/openblas/lib/libopenblas.dylib" \
  -DBUILD_PYTHON=OFF

# Build
cmake --build build --parallel 4
```

### 4. Run Tests

```bash
# Build and run tests
cmake -S test -B build/test
cmake --build build/test
cd build/test && ./TCITests
```

## Usage Example

```cpp
#include "tci/tci.h"
#include <cytnx.hpp>

int main() {
    // Create TCI context
    tci::context_handle_t<cytnx::Tensor> ctx;
    tci::create_context(ctx);

    // Create tensors using TCI API
    tci::shape_t<cytnx::Tensor> shape = {3, 4};
    cytnx::Tensor a, b, c;

    // Initialize tensors
    tci::zeros(ctx, shape, a);
    tci::eye(ctx, 3, b);

    // Perform operations
    tci::copy(ctx, a, c);
    auto norm_val = tci::norm(ctx, b);

    // Display results
    tci::show(ctx, b);

    // Cleanup
    tci::destroy_context(ctx);
    return 0;
}
```

### Compiling User Code (C++17)

```bash
# Your project can use C++17
g++ -std=c++17 -I<tci-install>/include your_code.cpp -lTCI -lcytnx
```

## Project Structure

```
├── include/tci/           # Public TCI API headers (C++17 compatible)
├── source/               # TCI implementation (C++17 features, C++20 compile)
├── test/                 # Test suite
├── external/Cytnx/       # Cytnx library submodule
└── CMakeLists.txt        # Build configuration
```

## Contributing

1. Ensure your contributions maintain C++17 feature compatibility
2. Only use C++20 features if absolutely necessary for Cytnx integration
3. Add tests for new functionality
4. Follow the existing code style

## Development Notes

### Build System

- TCI library compiles with C++20 (required by Cytnx)
- Public headers remain C++17 compatible
- Use explicit template specialization to hide implementation details

### Testing

```bash
# Build and run test suite
cmake -S test -B build/test
cmake --build build/test
cd build/test && ./TCITests

# With doctest, you can also run specific test cases
./build/test/TCITests --test-case="*Context*"
```

## License

[Apache-2.0 License](LICENSE)

## Related Projects

- [Cytnx](https://github.com/Cytnx-dev/Cytnx): High-performance tensor network library
- [TCI Specification](link-to-spec): Universal tensor computing interface specification

