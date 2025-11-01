cmake_minimum_required(VERSION 3.24) # Required for Cytnx

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

# ---- Add Cytnx as a subdirectory ----
# Fix: Set Kron.cpp to compile with nvcc when USE_CUDA is enabled
if(USE_CUDA)
  set_source_files_properties(
    ${CMAKE_CURRENT_SOURCE_DIR}/external/Cytnx/src/linalg/Kron.cpp
    PROPERTIES LANGUAGE CUDA
  )
endif()
add_subdirectory(external/Cytnx)

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
