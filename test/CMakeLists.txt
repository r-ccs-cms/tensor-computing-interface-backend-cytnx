cmake_minimum_required(VERSION 3.24)

project(TCITests LANGUAGES CXX)

# ---- Options ----

option(ENABLE_TEST_COVERAGE "Enable test coverage" OFF)
option(TEST_INSTALLED_VERSION "Test the version found by find_package" OFF)

# --- Import tools ----

include(../cmake/tools.cmake)

# ---- Dependencies ----

# Add doctest via CPM.cmake (header-only mode)
include(../cmake/CPM.cmake)
CPMAddPackage(
  NAME doctest
  GITHUB_REPOSITORY doctest/doctest
  VERSION 2.4.12
  DOWNLOAD_ONLY YES
)

if (doctest_ADDED)
  add_library(doctest INTERFACE)
  target_include_directories(doctest INTERFACE ${doctest_SOURCE_DIR})
  add_library(doctest::doctest ALIAS doctest)
endif()

if(NOT TARGET TCI::TCI)
  if(TEST_INSTALLED_VERSION)
    find_package(TCI REQUIRED)
  else()
    CPMAddPackage(NAME TCI SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)
  endif()
endif()

# ---- Create binary ----

file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/source/*.cpp)
message(STATUS "Test sources found: ${sources}")
add_executable(${PROJECT_NAME} ${sources})
target_link_libraries(${PROJECT_NAME} doctest::doctest TCI::TCI)
set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)
message(STATUS "Created test executable: ${PROJECT_NAME}")

add_custom_target(
  check-format
  COMMAND ${CMAKE_COMMAND} -E echo "Format check skipped: offline configuration"
)

# enable compiler warnings (temporarily disable -Werror for deprecated warnings)
if(NOT TEST_INSTALLED_VERSION)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(TCI PUBLIC -Wall -Wpedantic -Wextra -Wno-error=deprecated-declarations)
  elseif(MSVC)
    target_compile_options(TCI PUBLIC /W4 /WX)
    target_compile_definitions(${PROJECT_NAME} PUBLIC DOCTEST_CONFIG_USE_STD_HEADERS)
  endif()
endif()

# ---- Add TCITests ----

enable_testing()
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME})

# ---- code coverage ----

if(ENABLE_TEST_COVERAGE)
  target_compile_options(TCI PUBLIC -O0 -g -fprofile-arcs -ftest-coverage)
  target_link_options(TCI PUBLIC -fprofile-arcs -ftest-coverage)
endif()
