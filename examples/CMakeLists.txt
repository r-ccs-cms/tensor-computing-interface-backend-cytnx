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

# Install examples (optional)
install(TARGETS simple_tensor_operations
    RUNTIME DESTINATION bin/examples
    COMPONENT examples
)