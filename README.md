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
- **C++20 capable compiler**: GCC 10+, Clang 12+
- **BLAS/LAPACK**: OpenBLAS recommended
- **Cytnx**: Included as git submodule
- **OpenMP**: For parallel computations

### Optional

- **CUDA**: For GPU acceleration (via Cytnx)

## Quick Start

### 1. Clone with Submodules

```bash
git clone --recursive https://github.com/r-ccs-cms/tensor-computing-interface-backend-cytnx.git
cd tensor-computing-interface-backend-cytnx
```

### 2. Install Dependencies (macOS with Homebrew)

```bash
brew install openblas llvm libomp cmake boost arpack
```

llvm (LLVM Clang) is optional; Apple Clang may work.

### 3. Configure and Build

#### Option A: Using CMake Presets (Recommended)

```bash
# Configure and build with Homebrew preset (automatically handles dependencies)
cmake --preset brew-release
cmake --build --preset brew-release
ctest --preset brew-release
```

#### Option B: Manual Configuration

**For development (with debugging features):**

```bash
cmake --preset brew-debug
cmake --build --preset brew-debug
ctest --preset brew-debug
```

#### Manual Configuration (Alternative)

```bash
# Configure with toolchain file
cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/macos-homebrew.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 \
  -DBUILD_PYTHON=OFF

# Build
cmake --build build --parallel 8
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
├── include/tci/                    # Public TCI API headers (C++17 compatible)
├── source/                        # TCI implementation (C++17 features, C++20 compile)
├── test/                          # Test suite
├── external/Cytnx/                # Cytnx library submodule
├── cmake/toolchains/              # CMake toolchain files
│   └── macos-homebrew.cmake       # Homebrew dependency configuration
├── CMakePresets.json              # CMake preset configurations
└── CMakeLists.txt                 # Build configuration
```

## Documentation

Currently, document generation depends on Doxygen and uv.

```bash
# generate docs
cmake -S documentation -B build/doc
cmake --build build/doc --target GenerateDocs
# view the docs
open build/doc/doxygen/html/index.html
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
- **CMake presets** provide simplified build configuration
- **Homebrew toolchain** automatically handles keg-only dependency paths

### Testing

#### Using CMake Presets (Recommended)

**Production Testing**

```bash
# Configure, build and run release tests
cmake --preset brew-release
cmake --build --preset brew-release
ctest --preset brew-release
```

**Development Testing**

```bash
# Configure, build and run debug tests
cmake --preset brew-debug
cmake --build --preset brew-debug
ctest --preset brew-debug

# Run tests directly for more detailed output
./build-debug/test/TCITests

# Run specific test cases
./build-debug/test/TCITests --test-case="*template*"
```

#### Manual Test Configuration

```bash
# Configure test build manually
cmake -S . -B build-test \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/macos-homebrew.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 \
  -DTCI_BUILD_TESTS=ON \
  -DBUILD_PYTHON=OFF

# Build and run tests
cmake --build build-test --parallel 8
./build-test/test/TCITests
```

#### Test Development Workflow

**Quick development cycle:**

```bash
# Debug build
cmake --preset brew-debug && cmake --build --preset brew-debug && ctest --preset brew-debug
```

**Release validation:**

```bash
# Release build testing
cmake --preset brew-release && cmake --build --preset brew-release && ctest --preset brew-release
```

## License

[Apache-2.0 License](LICENSE)

## Related Projects

- [Cytnx](https://github.com/Cytnx-dev/Cytnx): High-performance tensor network library
- [TCI Specification](link-to-spec): Universal tensor computing interface specification
