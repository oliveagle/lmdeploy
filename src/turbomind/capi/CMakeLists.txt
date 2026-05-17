# Copyright (c) OpenMMLab. All rights reserved.
# C API library build for direct FFI integration

add_library(turbomind_c SHARED turbomind_c.cc)
target_include_directories(turbomind_c PUBLIC ${CMAKE_SOURCE_DIR})
target_link_libraries(turbomind_c PUBLIC turbomind)

# Ensure standard lib prefix for dynamic linking
set_target_properties(turbomind_c PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)
