# macOS + Homebrew toolchain for TCI project
# Automatically configures keg-only dependencies in CMAKE_PREFIX_PATH and PKG_CONFIG_PATH
# Eliminates the need for manual environment variable exports

if(NOT DEFINED HOMEBREW_PREFIX)
  execute_process(
    COMMAND brew --prefix
    OUTPUT_VARIABLE HOMEBREW_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
endif()

if(NOT HOMEBREW_PREFIX)
  message(FATAL_ERROR "Homebrew not found. Please install Homebrew first.")
endif()

# Define required and optional keg-only libraries
# Required: OpenBLAS for linear algebra, Boost for C++ utilities
# Optional: LLVM and OpenMP for enhanced compilation and parallelization
set(_required_keg_roots
  "${HOMEBREW_PREFIX}/opt/openblas"
  "${HOMEBREW_PREFIX}/opt/boost"
)

set(_optional_keg_roots
  "${HOMEBREW_PREFIX}/opt/llvm"
  "${HOMEBREW_PREFIX}/opt/libomp"
)

# Validate required libraries are installed
foreach(_root IN LISTS _required_keg_roots)
  if(NOT EXISTS "${_root}")
    get_filename_component(_lib_name "${_root}" NAME)
    message(FATAL_ERROR "Required library not found: ${_lib_name}. Please install with: brew install ${_lib_name}")
  endif()
endforeach()

# Collect paths for all available libraries (required + optional)
set(_existing_keg_roots)
foreach(_root IN LISTS _required_keg_roots _optional_keg_roots)
  if(EXISTS "${_root}")
    list(APPEND _existing_keg_roots "${_root}")
    get_filename_component(_lib_name "${_root}" NAME)
    message(STATUS "Found library: ${_lib_name}")
  endif()
endforeach()

# Configure CMake to find libraries in keg-only locations
list(PREPEND CMAKE_PREFIX_PATH ${_existing_keg_roots} "${HOMEBREW_PREFIX}")

# Set up pkg-config search paths for libraries with .pc files
foreach(_root IN LISTS _existing_keg_roots)
  if(EXISTS "${_root}/lib/pkgconfig")
    if(DEFINED ENV{PKG_CONFIG_PATH} AND NOT "$ENV{PKG_CONFIG_PATH}" STREQUAL "")
      set(ENV{PKG_CONFIG_PATH} "$ENV{PKG_CONFIG_PATH}:${_root}/lib/pkgconfig")
    else()
      set(ENV{PKG_CONFIG_PATH} "${_root}/lib/pkgconfig")
    endif()
  endif()
endforeach()

# Set legacy environment variables for backward compatibility with existing build scripts
set(_cppflags_parts)
set(_ldflags_parts)
foreach(_root IN LISTS _existing_keg_roots)
  if(EXISTS "${_root}/include")
    list(APPEND _cppflags_parts "-I${_root}/include")
  endif()
  if(EXISTS "${_root}/lib")
    list(APPEND _ldflags_parts "-L${_root}/lib")
  endif()
endforeach()

string(REPLACE ";" " " _cppflags_str "${_cppflags_parts}")
string(REPLACE ";" " " _ldflags_str "${_ldflags_parts}")
set(ENV{CPPFLAGS} "${_cppflags_str}")
set(ENV{LDFLAGS} "${_ldflags_str}")

# Prefer OpenBLAS for BLAS/LAPACK operations to ensure consistent linear algebra performance
set(BLA_VENDOR OpenBLAS CACHE STRING "Prefer OpenBLAS")

# Report configuration status for debugging
message(STATUS "Homebrew prefix: ${HOMEBREW_PREFIX}")
message(STATUS "CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}")
message(STATUS "PKG_CONFIG_PATH: $ENV{PKG_CONFIG_PATH}")
message(STATUS "CPPFLAGS: $ENV{CPPFLAGS}")
message(STATUS "LDFLAGS: $ENV{LDFLAGS}")