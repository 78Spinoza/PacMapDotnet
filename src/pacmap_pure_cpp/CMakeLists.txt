cmake_minimum_required(VERSION 3.15)
project(pacmap CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define PACMAP_EXPORTS for Windows DLL export
add_definitions(-DPACMAP_EXPORTS)

# Main PACMAP library
add_library(pacmap SHARED
    pacmap_simple_wrapper.cpp
    pacmap_simple_minimal.cpp
)

# Windows specific settings
if(WIN32)
    target_compile_definitions(pacmap PRIVATE
        _CRT_SECURE_NO_WARNINGS
        PACMAP_EXPORTS
    )
    set_target_properties(pacmap PROPERTIES
        OUTPUT_NAME "pacmap"
        PREFIX ""  # Remove lib prefix on Windows
        SUFFIX ".dll"
    )
else()
    set_target_properties(pacmap PROPERTIES
        OUTPUT_NAME "pacmap"
        PREFIX "lib"
        SUFFIX ".so"
    )
endif()

# Include directories
target_include_directories(pacmap PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})

# Set output directory
set_target_properties(pacmap PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
)

# Copy the DLL to the root directory for easy access
if(WIN32)
    add_custom_command(TARGET pacmap POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        $<TARGET_FILE:pacmap>
        ${CMAKE_BINARY_DIR}/pacmap.dll
    )
endif()

# =================== TEST EXECUTABLES ===================

# Test executables
add_executable(run_all_tests
    run_all_tests.cpp
)

add_executable(test_minimal_standalone
    test_minimal_standalone.cpp
)


add_executable(test_simple_minimal
    test_simple_minimal.cpp
)

add_executable(test_dll_export
    test_dll_export.cpp
)

add_executable(test_basic_integration
    test_basic_integration.cpp
)

# Link test executables with pacmap library
target_link_libraries(run_all_tests pacmap)
target_link_libraries(test_minimal_standalone pacmap)
target_link_libraries(test_simple_minimal pacmap)
target_link_libraries(test_dll_export pacmap)
target_link_libraries(test_basic_integration pacmap)

# Windows specific settings for tests
if(WIN32)
    target_compile_definitions(run_all_tests PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_definitions(test_minimal_standalone PRIVATE _CRT_SECURE_NO_WARNINGS)
      target_compile_definitions(test_simple_minimal PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_definitions(test_dll_export PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_definitions(test_basic_integration PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Set output directory for tests
set_target_properties(run_all_tests PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(test_minimal_standalone PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)


set_target_properties(test_simple_minimal PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(test_dll_export PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

set_target_properties(test_basic_integration PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)