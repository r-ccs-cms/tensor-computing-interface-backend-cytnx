# TCI (Tensor Computing Interface) - Cytnx Backend

[TCI (Tensor Computing Interface)](https://arxiv.org/abs/2512.23917) is a universal C++ interface for tensor computing libraries. This repository provides the Cytnx backend implementation, allowing TCI APIs to work with the [Cytnx](https://github.com/Cytnx-dev/Cytnx) tensor library.

## C++ Standard Requirements

- **Public API**: C++17 compatible — no C++20 features leak into user-facing headers
- **TCI build**: C++20 (required by Cytnx headers)

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

### 2. Install Dependencies

#### macOS with Homebrew

```bash
brew install openblas llvm libomp cmake boost arpack ninja
```

llvm (LLVM Clang) is optional; Apple Clang may work.

#### Linux with Intel oneAPI (HPC environments)

```bash
module load intel/oneapi  # Provides icx, icpx, ifort, and MKL
module load boost         # Required by Cytnx
```

ARPACK-NG will be automatically built from source if not found. To use a pre-built ARPACK, set `ARPACK_ROOT` or pass `-DARPACK_ROOT=/path/to/arpack`.

### 3. Configure, Build, and Test

#### Using CMake Presets (Recommended)

**macOS with Homebrew:**

```bash
cmake --preset brew-debug
cmake --build --preset brew-debug
ctest --preset brew-debug
```

**Linux with Intel oneAPI:**

```bash
module load intel/oneapi boost

cmake --preset intel-debug
cmake --build --preset intel-debug
ctest --preset intel-debug
```

Replace `debug` with `release` for optimized builds.

#### Manual Configuration

```bash
cmake -S . -B build \
  -DCMAKE_TOOLCHAIN_FILE=cmake/toolchains/macos-homebrew.cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CXX_STANDARD=20 \
  -DBUILD_PYTHON=OFF \
  -DTCI_BUILD_TESTS=ON

cmake --build build --parallel 8
```

### Running Tests Directly

```bash
# All tests
./build-debug/test/TCITests

# Specific test cases
./build-debug/test/TCITests --test-case="*template*"
```

## Disabling the Deprecated API

Define `TCI_NO_DEPRECATED_API` to suppress deprecated `template <typename ElemT>` overloads:

```cpp
#define TCI_NO_DEPRECATED_API
#include "tci/tci.h"
```

Or via CMake:

```bash
target_compile_definitions(your_target PRIVATE TCI_NO_DEPRECATED_API)
```

## Usage Example

### Basic Usage with CytnxTensor

```cpp
#include "tci/tci.h"

int main() {
    using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;

    tci::context_handle_t<Tensor> ctx;
    tci::create_context(ctx);

    Tensor a = tci::zeros<Tensor>(ctx, {3, 4});
    Tensor b = tci::eye<Tensor>(ctx, 3);

    Tensor c;
    tci::copy(ctx, a, c);
    auto norm_val = tci::norm(ctx, b);

    tci::show(ctx, b);

    tci::destroy_context(ctx);
    return 0;
}
```

### Element-wise Arithmetic

```cpp
#include "tci/tci.h"

int main() {
    using Tensor = tci::CytnxTensor<cytnx::cytnx_complex128>;
    using Elem = tci::elem_t<Tensor>;  // std::complex<double>

    tci::context_handle_t<Tensor> ctx;
    tci::create_context(ctx);

    Tensor tensor;
    tci::allocate(ctx, {100, 100}, tensor);

    tci::for_each(ctx, tensor, [](Elem& elem) {
        elem = std::sqrt(elem) * 2.0;
        elem = elem / std::abs(elem);
    });

    tci::destroy_context(ctx);
    return 0;
}
```

- `CytnxTensor<ElemT>` fixes the element type at compile time, so `tci::elem_t<Tensor>` resolves to `ElemT` and supports direct arithmetic.
- The wrapper exposes the underlying Cytnx tensor via `Tensor::backend` when lower-level Cytnx APIs are required.

### Compiling User Code (C++17)

```bash
g++ -std=c++17 -I<tci-install>/include your_code.cpp -lTCI -lcytnx
```

## Documentation

```bash
# Requires Doxygen and uv
cmake -S documentation -B build/doc
cmake --build build/doc --target GenerateDocs
open build/doc/doxygen/html/index.html
```

## License

[Apache-2.0 License](LICENSE)

