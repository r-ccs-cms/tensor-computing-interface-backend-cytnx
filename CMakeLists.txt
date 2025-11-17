cmake_minimum_required(VERSION 3.24) # Required for Cytnx

# Fix toolchain file path for subdirectories (like Cytnx)
# Convert relative paths to absolute paths BEFORE project() to ensure subprojects can find the toolchain
if(CMAKE_TOOLCHAIN_FILE AND NOT IS_ABSOLUTE "${CMAKE_TOOLCHAIN_FILE}")
  get_filename_component(CMAKE_TOOLCHAIN_FILE "${CMAKE_TOOLCHAIN_FILE}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  set(CMAKE_TOOLCHAIN_FILE "${CMAKE_TOOLCHAIN_FILE}" CACHE FILEPATH "Toolchain file" FORCE)
endif()

# ---- Project ----

if(USE_CUDA)
  set(TCI_LANGUAGES CXX CUDA)
else()
  set(TCI_LANGUAGES CXX)
endif()

project(
  TCI
  VERSION 0.2.0
  LANGUAGES ${TCI_LANGUAGES}
)

# ---- Version Information ----

set(TCI_SPEC_VERSION "1.0-rc.1")

# ---- Build Options ----

option(ENABLE_ASAN_ODR "Enable AddressSanitizer ODR violation detection" OFF)
option(ENABLE_ASAN_ODR_STRICT "Enable strict ASAN ODR detection (level 2)" OFF)
option(ENABLE_ODR_INDICATOR "Enable ODR indicator symbols (default ON for non-Windows)" ON)

# ---- Include guards ----

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

# ---- Add dependencies via CPM ----
# see https://github.com/TheLartians/CPM.cmake for more info

include(cmake/CPM.cmake)

# ---- Find or build ARPACK dependency ----
# ARPACK is required by Cytnx for eigenvalue computations
# Users can specify existing ARPACK installation via:
#   - ARPACK_ROOT environment variable
#   - -DARPACK_ROOT=/path/to/arpack on cmake command line
# If not found, ARPACK-NG will be automatically built from source

# Ensure consistent BLAS/LAPACK usage
# The toolchain file may have already set BLA_VENDOR, preserve it
if(NOT DEFINED BLA_VENDOR)
  # If not set by toolchain, use default (OpenBLAS for macOS, Intel for Intel compiler)
  if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    set(BLA_VENDOR Intel10_64lp CACHE STRING "Use Intel MKL")
  endif()
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# First, try to find existing ARPACK installation
find_package(ARPACK QUIET)

if(NOT ARPACK_FOUND)
  # If not found, build ARPACK-NG from source
  include(ExternalARPACK)
else()
  message(STATUS "Using existing ARPACK: ${ARPACK_LIBRARIES}")
endif()

# ---- Add Cytnx as a subdirectory ----
# Provide ARPACK to Cytnx
# Cytnx expects ARPACK_LIB variable to be set
# Note: ARPACK requires Fortran runtime (ifcore) when built with Intel Fortran

# TODO: Review if this manual INTERFACE_LINK_LIBRARIES extraction is necessary.
# This was added to work around Cytnx's find_library(ARPACK_LIB) overwriting our settings,
# but may be overly complex. Consider if Cytnx's CMakeLists should be patched instead.
if(TARGET ARPACK::ARPACK)
  # Get all libraries needed for ARPACK (including Fortran runtime)
  get_target_property(ARPACK_INTERFACE_LIBS ARPACK::ARPACK INTERFACE_LINK_LIBRARIES)
  if(ARPACK_INTERFACE_LIBS)
    # Combine ARPACK library and its dependencies (e.g., Fortran runtime)
    set(ARPACK_LIB "${ARPACK_LIBRARIES};${ARPACK_INTERFACE_LIBS}")
  else()
    set(ARPACK_LIB ${ARPACK_LIBRARIES})
  endif()
else()
  set(ARPACK_LIB ${ARPACK_LIBRARIES})
endif()

# Mark ARPACK_LIB as found to prevent Cytnx from re-searching
# Cytnx's find_library(ARPACK_LIB ...) will be skipped if this variable is already set
set(ARPACK_LIB "${ARPACK_LIB}" CACHE STRING "ARPACK library with dependencies for Cytnx" FORCE)

# Ensure Cytnx uses the same BLAS/LAPACK as the rest of the project
# This prevents linking both OpenBLAS and MKL
# Cytnx will inherit BLA_VENDOR cache variable if set

# Provide the already-found BLAS/LAPACK libraries to Cytnx
# This prevents Cytnx from searching again and finding OpenBLAS
if(BLAS_FOUND AND LAPACK_FOUND)
  set(BLAS_LIBRARIES ${BLAS_LIBRARIES} CACHE STRING "BLAS libraries" FORCE)
  set(LAPACK_LIBRARIES ${LAPACK_LIBRARIES} CACHE STRING "LAPACK libraries" FORCE)
  set(BLAS_FOUND TRUE CACHE BOOL "BLAS found" FORCE)
  set(LAPACK_FOUND TRUE CACHE BOOL "LAPACK found" FORCE)

  # For Intel MKL, enable USE_MKL in Cytnx
  # This will make Cytnx use MKL and properly set up include directories
  if(DEFINED ENV{MKLROOT})
    set(USE_MKL ON CACHE BOOL "Enable MKL in Cytnx" FORCE)
  endif()
endif()

# Fix: Set Kron.cpp to compile with nvcc when USE_CUDA is enabled
if(USE_CUDA)
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/external/Cytnx/src/linalg/Kron.cpp
    PROPERTIES LANGUAGE CUDA
  )
endif()

# Disable LTO/IPO for Cytnx to prevent LLVM bitcode in static libraries
# Intel compilers produce LLVM IR bitcode with LTO, which causes linker errors
set(CMAKE_INTERPROCEDURAL_OPTIMIZATION OFF)

add_subdirectory(external/Cytnx)

# Force disable IPO/LTO on Cytnx target after it's created
# Cytnx enables CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE by default
# This causes LLVM bitcode in static libraries with Intel compilers
if(TARGET cytnx)
  set_target_properties(cytnx PROPERTIES INTERPROCEDURAL_OPTIMIZATION FALSE)

  # Apply compiler-specific flags for disabling LTO/IPO
  if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    # Intel compiler specific flags
    target_compile_options(cytnx PRIVATE -fno-lto -no-ipo)
    target_link_options(cytnx PRIVATE -fno-lto -no-ipo)
  else()
    # For Clang/GCC, only use -fno-lto (they don't have -no-ipo)
    target_compile_options(cytnx PRIVATE -fno-lto)
    target_link_options(cytnx PRIVATE -fno-lto)
  endif()
endif()

# ---- Generate version header ----

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/tci/version.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/tci/version.h"
  @ONLY
)

configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/include/tci/detail/version_impl.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/include/tci/detail/version_impl.h"
  @ONLY
)

# ---- Add source files ----

# Note: globbing sources is considered bad practice as CMake's generators may not detect new files
# automatically. Keep that in mind when changing files, or explicitly mention them here.
file(GLOB_RECURSE headers CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
file(GLOB_RECURSE sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp")

# ---- Create library ----

# Note: for header-only libraries change all PUBLIC flags to INTERFACE and create an interface
# target: add_library(${PROJECT_NAME} INTERFACE)
add_library(${PROJECT_NAME} ${headers} ${sources})
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

# being a cross-platform target, we enforce standards conformance on MSVC
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")

# ---- AddressSanitizer ODR Detection Configuration ----

# Function to configure ASAN ODR detection
function(configure_asan_odr_detection target)
    if(ENABLE_ASAN_ODR OR ENABLE_ASAN_ODR_STRICT)
        # Check if compiler supports AddressSanitizer
        if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
            message(STATUS "Enabling AddressSanitizer ODR detection for ${target}")

            # Basic AddressSanitizer flags
            target_compile_options(${target} PUBLIC
                -fsanitize=address
                -g
                -O1
                -fno-omit-frame-pointer
                -fno-optimize-sibling-calls
            )

            target_link_options(${target} PUBLIC
                -fsanitize=address
            )

            # ODR indicator symbols (enabled by default on non-Windows)
            if(ENABLE_ODR_INDICATOR AND NOT WIN32)
                target_compile_options(${target} PUBLIC
                    -fsanitize-address-use-odr-indicator
                )
            elseif(NOT ENABLE_ODR_INDICATOR)
                target_compile_options(${target} PUBLIC
                    -fno-sanitize-address-use-odr-indicator
                )
            endif()

            # Additional flags for better ODR detection
            target_compile_options(${target} PUBLIC
                -fno-sanitize-recover=address  # Stop on first violation
                -fno-common                    # Prevent common symbol merging
            )

            # Set runtime environment variables via CMake
            set_target_properties(${target} PROPERTIES
                ENVIRONMENT "ASAN_OPTIONS=detect_odr_violation=$<IF:$<BOOL:${ENABLE_ASAN_ODR_STRICT}>,2,1>:abort_on_error=1:halt_on_error=1"
            )

            message(STATUS "ASAN ODR detection level: $<IF:$<BOOL:${ENABLE_ASAN_ODR_STRICT}>,2 (strict),1 (normal)>")

        else()
            message(WARNING "AddressSanitizer ODR detection requested but compiler ${CMAKE_CXX_COMPILER_ID} may not support it")
        endif()
    endif()
endfunction()

# Apply ASAN ODR configuration to main target
configure_asan_odr_detection(${PROJECT_NAME})

# Link dependencies
target_link_libraries(${PROJECT_NAME} PUBLIC cytnx)

# If ARPACK was built as an imported target, link it directly to TCI
# This ensures that ARPACK's INTERFACE_LINK_LIBRARIES (including ifcore) are properly propagated
if(TARGET ARPACK::ARPACK)
  target_link_libraries(${PROJECT_NAME} PUBLIC ARPACK::ARPACK)
endif()

# TODO: Simplify this manual library propagation logic.
# This complex loop was added to work around static library linking issues with Intel compilers
# where INTERFACE_LINK_LIBRARIES weren't being properly propagated.
# Consider:
# 1. Whether this is still needed after IPO/LTO fixes
# 2. If CMake's native transitive dependency handling is sufficient
# Since TCI is a static library, we need to explicitly propagate all of Cytnx's
# INTERFACE_LINK_LIBRARIES to TCI's own INTERFACE_LINK_LIBRARIES.
# This ensures that downstream targets (examples, tests) get all necessary libraries.
get_target_property(CYTNX_INTERFACE_LIBS cytnx INTERFACE_LINK_LIBRARIES)
if(CYTNX_INTERFACE_LIBS)
  # Remove any incomplete "-lifcore" entries, but keep full paths
  list(FILTER CYTNX_INTERFACE_LIBS EXCLUDE REGEX "^-lifcore$")

  # Add libraries to TCI's interface, but skip CMake target names (like Boost::boost)
  # Only add actual library files or linker flags
  foreach(lib ${CYTNX_INTERFACE_LIBS})
    # Check if lib is an actual file path or linker flag, not a target name
    if(lib MATCHES "^/" OR lib MATCHES "^-l" OR lib MATCHES "^\\.\\.")
      target_link_libraries(${PROJECT_NAME} PUBLIC ${lib})
    endif()
  endforeach()
endif()

# Create a flag boundary: Remove Cytnx's -w flag from INTERFACE propagation
# This allows TCI's tests and examples to show deprecation warnings
# Note: Cytnx itself will still compile with -w, but it won't propagate to downstream targets
get_target_property(CYTNX_INTERFACE_COMPILE_OPTIONS cytnx INTERFACE_COMPILE_OPTIONS)
if(CYTNX_INTERFACE_COMPILE_OPTIONS)
  list(REMOVE_ITEM CYTNX_INTERFACE_COMPILE_OPTIONS "-w")
  set_target_properties(cytnx PROPERTIES INTERFACE_COMPILE_OPTIONS "${CYTNX_INTERFACE_COMPILE_OPTIONS}")
endif()

target_include_directories(
  ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                         $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/include>
                         $<INSTALL_INTERFACE:include/${PROJECT_NAME}-${PROJECT_VERSION}>
                         $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/external/Cytnx/include>
)

# ---- Add alias target for tests ----
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

# ---- Add examples ----
option(BUILD_EXAMPLES "Build example programs" ON)
if(BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# ---- Optional test suite ----
option(TCI_BUILD_TESTS "Build the TCI test suite" OFF)
if(TCI_BUILD_TESTS)
  add_subdirectory(test)
endif()
