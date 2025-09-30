cmake_minimum_required(VERSION 3.24)

# ---- Examples for TCI usage ----

# Note: This CMakeLists.txt is included from the parent project
# TCI target and dependencies are already available

# Simple tensor operations example
add_executable(simple_tensor_operations simple_tensor_operations.cpp)
target_link_libraries(simple_tensor_operations PRIVATE TCI::TCI)
set_target_properties(simple_tensor_operations PROPERTIES CXX_STANDARD 17)

# Add debug symbols for better debugging experience
target_compile_options(simple_tensor_operations PRIVATE -g)

# Basic tensor example
add_executable(basic_tensor_example basic_tensor_example.cpp)
target_link_libraries(basic_tensor_example PRIVATE TCI::TCI)
set_target_properties(basic_tensor_example PROPERTIES CXX_STANDARD 17)
target_compile_options(basic_tensor_example PRIVATE -g)

# SVD example
add_executable(svd_example svd_example.cpp)
target_link_libraries(svd_example PRIVATE TCI::TCI)
set_target_properties(svd_example PROPERTIES CXX_STANDARD 17)
target_compile_options(svd_example PRIVATE -g)

# Tensor contraction example
add_executable(tensor_contraction_example tensor_contraction_example.cpp)
target_link_libraries(tensor_contraction_example PRIVATE TCI::TCI)
set_target_properties(tensor_contraction_example PROPERTIES CXX_STANDARD 17)
target_compile_options(tensor_contraction_example PRIVATE -g)

# Install examples (optional)
install(TARGETS simple_tensor_operations basic_tensor_example svd_example tensor_contraction_example
    RUNTIME DESTINATION bin/examples
    COMPONENT examples
)