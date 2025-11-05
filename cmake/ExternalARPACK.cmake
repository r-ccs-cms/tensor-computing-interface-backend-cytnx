# ExternalARPACK.cmake
# Automatically downloads and builds ARPACK-NG when not found on the system
#
# This module is used when find_package(ARPACK) fails
# It builds ARPACK-NG from source using the detected BLAS/LAPACK
#
# NOTE: Downloads source during configure phase using FetchContent_Populate,
# but builds during build phase using ExternalProject_Add with SOURCE_DIR

include(FetchContent)
include(ExternalProject)

message(STATUS "ARPACK not found. Will build ARPACK-NG from source.")

# Detect BLAS/LAPACK for ARPACK backend
# This should be MKL when using Intel toolchain
find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)

# Check if Fortran compiler is available (required for ARPACK-NG)
# For Intel compilers, prefer Intel Fortran (ifort/ifx) over GNU Fortran
include(CheckLanguage)

# If using Intel C/C++ compilers, try to find Intel Fortran
if(CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
  # Try Intel Fortran compilers in order of preference
  find_program(INTEL_FORTRAN_COMPILER
    NAMES ifx ifort
    HINTS
      ENV{ONEAPI_ROOT}/compiler/latest/linux/bin
      /opt/intel/oneapi/compiler/latest/linux/bin
      ${CMAKE_CXX_COMPILER_DIR}
  )

  if(INTEL_FORTRAN_COMPILER)
    set(CMAKE_Fortran_COMPILER ${INTEL_FORTRAN_COMPILER} CACHE FILEPATH "Fortran compiler" FORCE)
  else()
    message(WARNING "Intel C++ compiler is used, but Intel Fortran compiler not found.\n"
                    "Using system Fortran compiler may cause linking issues.")
  endif()
endif()

check_language(Fortran)

if(CMAKE_Fortran_COMPILER)
  enable_language(Fortran)
else()
  message(FATAL_ERROR "Fortran compiler not found. ARPACK-NG requires Fortran.\n"
                      "Please install a Fortran compiler (e.g., gfortran or ifort).\n"
                      "For Intel oneAPI: module load intel/oneapi (includes ifort/ifx)")
endif()

# Download ARPACK-NG source during configure phase (network required)
# This uses FetchContent_Populate which only downloads, doesn't configure
FetchContent_Declare(
  arpack-ng
  GIT_REPOSITORY https://github.com/opencollab/arpack-ng.git
  GIT_TAG 3.9.1
  GIT_SHALLOW TRUE
)

FetchContent_Populate(arpack-ng)

# Set up directories for ExternalProject
set(ARPACK_SOURCE_DIR "${arpack-ng_SOURCE_DIR}")
set(ARPACK_BINARY_DIR "${CMAKE_BINARY_DIR}/external/arpack-ng/build")
set(ARPACK_INSTALL_DIR "${CMAKE_BINARY_DIR}/external/arpack-ng/install")

# Convert CMake lists to space-separated strings for ExternalProject
string(REPLACE ";" " " BLAS_LIBRARIES_STR "${BLAS_LIBRARIES}")
string(REPLACE ";" " " LAPACK_LIBRARIES_STR "${LAPACK_LIBRARIES}")

# Build ARPACK-NG using ExternalProject during build phase (network NOT required)
ExternalProject_Add(arpack-ng-build
  SOURCE_DIR ${ARPACK_SOURCE_DIR}
  BINARY_DIR ${ARPACK_BINARY_DIR}
  INSTALL_DIR ${ARPACK_INSTALL_DIR}
  CMAKE_ARGS
    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
    -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
    -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
    -DCMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER}
    -DCMAKE_INSTALL_PREFIX=${ARPACK_INSTALL_DIR}
    -DBUILD_SHARED_LIBS=OFF
    -DICB=ON
    -DEXAMPLES=OFF
    -DTESTS=OFF
    -DBLAS_LIBRARIES=${BLAS_LIBRARIES_STR}
    -DLAPACK_LIBRARIES=${LAPACK_LIBRARIES_STR}
  BUILD_BYPRODUCTS
    ${ARPACK_INSTALL_DIR}/lib64/libarpack.a
  DOWNLOAD_COMMAND ""
  UPDATE_COMMAND ""
  LOG_CONFIGURE ON
  LOG_BUILD ON
)

# Set variables for use by parent project
set(ARPACK_LIBRARY "${ARPACK_INSTALL_DIR}/lib64/libarpack.a" CACHE FILEPATH "ARPACK library" FORCE)
set(ARPACK_LIBRARIES ${ARPACK_LIBRARY} CACHE STRING "ARPACK libraries" FORCE)
set(ARPACK_FOUND TRUE CACHE BOOL "ARPACK found" FORCE)

# Create imported target for ARPACK
add_library(ARPACK::ARPACK STATIC IMPORTED GLOBAL)
set_target_properties(ARPACK::ARPACK PROPERTIES
  IMPORTED_LOCATION "${ARPACK_LIBRARY}"
)

# ARPACK is built with Fortran, so we need to link Fortran runtime libraries
# Find Fortran implicit link libraries
if(CMAKE_Fortran_COMPILER)
  # Get Fortran compiler's implicit link libraries
  execute_process(
    COMMAND ${CMAKE_Fortran_COMPILER} -v
    OUTPUT_VARIABLE FORTRAN_VERSION_OUTPUT
    ERROR_VARIABLE FORTRAN_VERSION_OUTPUT
  )

  # For Intel Fortran, we need ifcore (Fortran runtime)
  if(CMAKE_Fortran_COMPILER_ID MATCHES "IntelLLVM")
    find_library(FORTRAN_IFCORE_LIB NAMES ifcore
      HINTS
        ${CMAKE_Fortran_IMPLICIT_LINK_DIRECTORIES}
        ENV{LIBRARY_PATH}
      PATH_SUFFIXES lib
    )

    if(FORTRAN_IFCORE_LIB)
      set_property(TARGET ARPACK::ARPACK APPEND PROPERTY
        INTERFACE_LINK_LIBRARIES ${FORTRAN_IFCORE_LIB}
      )
    else()
      message(WARNING "Could not find Intel Fortran runtime library (ifcore). Linking may fail.")
    endif()
  endif()
endif()

# Ensure ARPACK is built before targets that depend on it
add_dependencies(ARPACK::ARPACK arpack-ng-build)
