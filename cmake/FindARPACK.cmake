# FindARPACK.cmake
# Finds ARPACK library (ARPACK-NG or compatible)
#
# Usage:
#   find_package(ARPACK)
#
# User can specify ARPACK location via:
#   - ARPACK_ROOT environment variable
#   - -DARPACK_ROOT=/path/to/arpack on cmake command line
#
# This module defines:
#   ARPACK_FOUND - system has ARPACK
#   ARPACK_LIBRARIES - ARPACK library
#   ARPACK_INCLUDE_DIRS - ARPACK include directories (if any)

# Search for ARPACK library in standard locations and user-specified paths
find_library(ARPACK_LIBRARY
  NAMES arpack libarpack
  HINTS
    ${ARPACK_ROOT}
    $ENV{ARPACK_ROOT}
    ${CMAKE_PREFIX_PATH}
  PATH_SUFFIXES
    lib
    lib64
    lib/x86_64-linux-gnu
  DOC "Path to ARPACK library"
)

# ARPACK typically doesn't have headers (Fortran library with C bindings)
# But some installations might have them
find_path(ARPACK_INCLUDE_DIR
  NAMES arpack.h arpackdef.h
  HINTS
    ${ARPACK_ROOT}
    $ENV{ARPACK_ROOT}
    ${CMAKE_PREFIX_PATH}
  PATH_SUFFIXES
    include
    include/arpack
  DOC "Path to ARPACK include directory"
)

# Handle standard find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ARPACK
  REQUIRED_VARS ARPACK_LIBRARY
  FOUND_VAR ARPACK_FOUND
)

if(ARPACK_FOUND)
  set(ARPACK_LIBRARIES ${ARPACK_LIBRARY})

  if(ARPACK_INCLUDE_DIR)
    set(ARPACK_INCLUDE_DIRS ${ARPACK_INCLUDE_DIR})
  else()
    set(ARPACK_INCLUDE_DIRS "")
  endif()

  # Create imported target for modern CMake usage
  if(NOT TARGET ARPACK::ARPACK)
    add_library(ARPACK::ARPACK UNKNOWN IMPORTED)
    set_target_properties(ARPACK::ARPACK PROPERTIES
      IMPORTED_LOCATION "${ARPACK_LIBRARY}"
    )
    if(ARPACK_INCLUDE_DIR)
      set_target_properties(ARPACK::ARPACK PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ARPACK_INCLUDE_DIR}"
      )
    endif()
  endif()

  mark_as_advanced(ARPACK_LIBRARY ARPACK_INCLUDE_DIR)
endif()
