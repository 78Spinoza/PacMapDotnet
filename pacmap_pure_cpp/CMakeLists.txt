cmake_minimum_required(VERSION 3.15)
project(pacmap CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define PACMAP_EXPORTS for Windows DLL export
add_definitions(-DPACMAP_EXPORTS -DLZ4_DLL_EXPORT)

# Main PACMAP library with HNSW-enhanced embedding storage and LZ4 compression
add_library(pacmap SHARED
    pacmap_embedding_storage.h
    pacmap_embedding_storage.cpp
)

# Add LZ4 C source and set proper language
target_sources(pacmap PRIVATE lz4.c)
set_source_files_properties(lz4.c PROPERTIES LANGUAGE C)
target_compile_definitions(pacmap PRIVATE LZ4_DLL_EXPORT=1)

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

# Comprehensive test suite
add_executable(test_comprehensive
    test_comprehensive.cpp
)

# Link with pacmap library
target_link_libraries(test_simple pacmap)
target_link_libraries(test_comprehensive pacmap)

# Windows specific settings for tests
if(WIN32)
    target_compile_definitions(test_simple PRIVATE _CRT_SECURE_NO_WARNINGS)
    target_compile_definitions(test_comprehensive PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()

# Set output directory for tests
set_target_properties(test_simple PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)
set_target_properties(test_comprehensive PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)