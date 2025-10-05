cmake_minimum_required(VERSION 3.15)
project(pacmap CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define PACMAP_EXPORTS for Windows DLL export
add_definitions(-DPACMAP_EXPORTS)

# Main PACMAP library with HNSW-enhanced embedding storage
add_library(pacmap SHARED
    pacmap_embedding_storage.h
    pacmap_embedding_storage.cpp
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

# Simple DLL integration test
add_executable(test_simple
    test_simple.cpp
)

# Windows specific settings for tests
if(WIN32)
    target_compile_definitions(test_simple PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Set output directory for tests
set_target_properties(test_simple PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)