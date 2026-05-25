# Copyright (c) OpenMMLab. All rights reserved.
# C API library build for direct FFI integration

add_library(turbomind_c SHARED turbomind_c.cc)
target_include_directories(turbomind_c PUBLIC ${CMAKE_SOURCE_DIR})
target_include_directories(turbomind_c PRIVATE ${xgrammar_SOURCE_DIR}/include)
target_include_directories(turbomind_c PRIVATE ${xgrammar_SOURCE_DIR}/cpp/include)

# Force-link all objects from static libraries that contain self-registration
# globals (module type registrars).  Without --whole-archive, the linker
# strips registrar objects because nobody references them directly.
target_link_libraries(turbomind_c PUBLIC models turbomind weight_serializer)
set_property(TARGET turbomind_c APPEND PROPERTY LINK_OPTIONS
    "-Wl,--whole-archive" "$<TARGET_FILE:models>" "$<TARGET_FILE:turbomind>" "-Wl,--no-whole-archive")

# Ensure standard lib prefix for dynamic linking
set_target_properties(turbomind_c PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Step-by-step InitFromPath test binary for debugging hangs
if(BUILD_C_FFI)
    add_executable(test_initfrompath_stepby test_initfrompath_stepby.cc)
    target_link_libraries(test_initfrompath_stepby PRIVATE turbomind_c)
    set_target_properties(test_initfrompath_stepby PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
endif()
