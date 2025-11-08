# Linux + Intel oneAPI toolchain for TCI project
# Uses environment variables set by 'module load intel/oneapi' or similar
# Requires: CC=icx, CXX=icpx, MKLROOT to be set in environment

# Use compilers from environment if available, otherwise search for them
if(DEFINED ENV{CC})
  set(CMAKE_C_COMPILER $ENV{CC} CACHE FILEPATH "C compiler")
else()
  find_program(CMAKE_C_COMPILER NAMES icx icc REQUIRED)
  # Set environment variable so subprojects (Cytnx, ARPACK) use the same compiler
  set(ENV{CC} ${CMAKE_C_COMPILER})
endif()

if(DEFINED ENV{CXX})
  set(CMAKE_CXX_COMPILER $ENV{CXX} CACHE FILEPATH "C++ compiler")
else()
  find_program(CMAKE_CXX_COMPILER NAMES icpx icpc REQUIRED)
  # Set environment variable so subprojects (Cytnx, ARPACK) use the same compiler
  set(ENV{CXX} ${CMAKE_CXX_COMPILER})
endif()

# Also find and set Fortran compiler for consistency
if(NOT DEFINED ENV{FC})
  find_program(INTEL_FORTRAN_COMPILER NAMES ifx ifort)
  if(INTEL_FORTRAN_COMPILER)
    set(ENV{FC} ${INTEL_FORTRAN_COMPILER})
  endif()
endif()

# Set Intel linker to handle LLVM bitcode from Intel compilers
# Intel compilers may produce LLVM bitcode that GNU ld cannot handle
# Use Intel's linker (xild) or LLVM's lld
set(CMAKE_LINKER "ld.lld" CACHE FILEPATH "Linker")
set(CMAKE_AR "llvm-ar" CACHE FILEPATH "Archiver")
set(CMAKE_RANLIB "llvm-ranlib" CACHE FILEPATH "Ranlib")

# Boost configuration (from environment modules)
# Check for BOOST_ROOT or Boost_ROOT from environment modules
if(DEFINED ENV{BOOST_ROOT})
  set(BOOST_ROOT $ENV{BOOST_ROOT})
  list(PREPEND CMAKE_PREFIX_PATH "${BOOST_ROOT}")
elseif(DEFINED ENV{Boost_ROOT})
  set(BOOST_ROOT $ENV{Boost_ROOT})
  list(PREPEND CMAKE_PREFIX_PATH "${BOOST_ROOT}")
endif()

# Intel MKL configuration
# MKLROOT should be set by Intel oneAPI environment (e.g., /opt/intel/oneapi/mkl/latest)
if(DEFINED ENV{MKLROOT})
  set(MKLROOT $ENV{MKLROOT})

  # Add MKL to CMake search paths
  list(PREPEND CMAKE_PREFIX_PATH "${MKLROOT}")

  # Set up MKL library paths
  if(EXISTS "${MKLROOT}/lib/intel64")
    list(PREPEND CMAKE_LIBRARY_PATH "${MKLROOT}/lib/intel64")
  endif()

  if(EXISTS "${MKLROOT}/include")
    list(PREPEND CMAKE_INCLUDE_PATH "${MKLROOT}/include")
  endif()

  # Prefer Intel MKL for BLAS/LAPACK
  set(BLA_VENDOR Intel10_64lp CACHE STRING "Use Intel MKL")

  # Set up pkg-config for MKL if available
  if(EXISTS "${MKLROOT}/lib/pkgconfig")
    if(DEFINED ENV{PKG_CONFIG_PATH} AND NOT "$ENV{PKG_CONFIG_PATH}" STREQUAL "")
      set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${MKLROOT}/lib/pkgconfig")
    else()
      set(ENV{PKG_CONFIG_PATH} "${MKLROOT}/lib/pkgconfig")
    endif()
  endif()
else()
  message(WARNING "MKLROOT environment variable not set. Intel MKL may not be found.\n"
                  "Please ensure Intel oneAPI module is loaded:\n"
                  "  module load intel/oneapi")
endif()

# Intel-specific compiler flags for portable optimization
# These flags work across different Intel and compatible CPUs
# Disable IPO/LTO to avoid LLVM bitcode in static libraries
# -fno-lto: disable Link Time Optimization
# -no-ipo: disable Interprocedural Optimization (Intel-specific)
set(INTEL_CXX_FLAGS_INIT "-fp-model=precise -fno-lto -no-ipo")

# Add Intel flags to CMake defaults
set(CMAKE_CXX_FLAGS_INIT "${CMAKE_CXX_FLAGS_INIT} ${INTEL_CXX_FLAGS_INIT}")
set(CMAKE_C_FLAGS_INIT "${CMAKE_C_FLAGS_INIT} ${INTEL_CXX_FLAGS_INIT}")

# Intel-specific optimization flags by build type
set(CMAKE_CXX_FLAGS_RELEASE_INIT "-O3 -DNDEBUG")
set(CMAKE_CXX_FLAGS_DEBUG_INIT "-O0 -g")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO_INIT "-O2 -g -DNDEBUG")

# Disable IPO/LTO globally to prevent LLVM bitcode in archives
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF CACHE BOOL "Disable IPO/LTO" FORCE)
