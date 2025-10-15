cmake_minimum_required(VERSION 3.15)
project(pacmap VERSION 2.8.23 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define PACMAP_EXPORTS for Windows DLL export
add_definitions(-DPACMAP_EXPORTS)

# Define LZ4_FREESTANDING to avoid external dependencies
add_definitions(-DLZ4_FREESTANDING=0)

# Export LZ4 functions from the DLL - let lz4.h define its own API
# add_definitions(-DLZ4LIB_API=)

# Main PACMAP library - CORE IMPLEMENTATION FILES
add_library(pacmap SHARED
    # Core algorithm files
    pacmap_fit.cpp
    pacmap_transform.cpp
    pacmap_triplet_sampling.cpp
    pacmap_optimization.cpp
    pacmap_gradient.cpp
    pacmap_model.cpp
    pacmap_distance.cpp
    pacmap_utils.cpp
  pacmap_persistence.cpp
        pacmap_progress_utils.cpp
    pacmap_quantization.cpp
    pacmap_crc32.cpp
    pacmap_hnsw_utils.cpp

    # Headers (included for compilation)
    pacmap_fit.h
    pacmap_transform.h
    pacmap_triplet_sampling.h
    pacmap_optimization.h
    pacmap_gradient.h
    pacmap_model.h
    pacmap_distance.h
    pacmap_utils.h
    pacmap_persistence.h
    pacmap_progress_utils.h
    pacmap_quantization.h
    pacmap_crc32.h
    pacmap_hnsw_utils.h
    pacmap_simd_utils.h
    pacmap_system_info.h

    # External dependencies
    hnswalg.h
    hnswlib.h
    lz4.h
    space_l2.h
    space_ip.h
    stop_condition.h
    bruteforce.h
    visited_list_pool.h

    # C++ API wrapper
    pacmap_simple_wrapper.cpp
    pacmap_simple_wrapper.h

    # Note: debug_to_file.cpp was moved to backup as it was not part of core implementation
)

# Add LZ4 as C source
target_sources(pacmap PRIVATE lz4.c)
set_source_files_properties(lz4.c PROPERTIES LANGUAGE C)

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

# OpenMP Configuration - CRITICAL for parallel processing
if(MSVC)
    # Enable OpenMP for parallel processing on Windows
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found for MSVC: ${OpenMP_CXX_VERSION}")
        target_compile_options(pacmap PRIVATE /openmp)
        target_compile_definitions(pacmap PRIVATE _OPENMP=201511)  # Define OpenMP macro manually
        target_link_libraries(pacmap OpenMP::OpenMP_CXX)
    else()
        message(FATAL_ERROR "OpenMP not found - required for performance")
    endif()
else()
    # Enable OpenMP for parallel processing on GCC/Clang
    find_package(OpenMP REQUIRED)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found for GCC/Clang: ${OpenMP_CXX_VERSION}")
        target_compile_options(pacmap PRIVATE -fopenmp)
        target_link_libraries(pacmap OpenMP::OpenMP_CXX)
    else()
        message(FATAL_ERROR "OpenMP not found - required for performance")
    endif()
endif()

# Performance optimizations for Release builds
if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    if(MSVC)
        # MSVC optimizations
        target_compile_options(pacmap PRIVATE /O2 /Ob2 /DNDEBUG)
        # Temporarily disable AVX2 to debug C++ test segfault
        # target_compile_options(pacmap PRIVATE /arch:AVX2)
    else()
        # GCC/Clang optimizations
        target_compile_options(pacmap PRIVATE -O3 -DNDEBUG)
        # Enable native optimizations for better performance
        target_compile_options(pacmap PRIVATE -march=native)
    endif()
endif()

# Include directories
target_include_directories(pacmap PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/eigen  # Eigen library for SIMD vectorization
)

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
# Test executables commented out for production builds
# Uncomment these lines if you need to run tests during development

# CRITICAL FIX: Removed test_adam_gaussian - file was deleted during repository cleanup
# This test is no longer needed as the Adam optimizer has been verified and fixed

# C++ Integration Tests - Updated for v2.8.6 double precision
add_executable(test_basic_integration
    test_basic_integration.cpp
)
target_link_libraries(test_basic_integration pacmap)
if(WIN32)
    target_compile_definitions(test_basic_integration PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
set_target_properties(test_basic_integration PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Minimal test to isolate segfault issue
add_executable(test_minimal
    test_minimal.cpp
)
target_link_libraries(test_minimal pacmap)
if(WIN32)
    target_compile_definitions(test_minimal PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
set_target_properties(test_minimal PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# OpenMP isolation test to debug segfault
add_executable(test_openmp_isolation
    test_openmp_isolation.cpp
)
if(WIN32)
    target_compile_options(test_openmp_isolation PRIVATE /openmp)
    target_compile_definitions(test_openmp_isolation PRIVATE _OPENMP=201511)
else()
    target_compile_options(test_openmp_isolation PRIVATE -fopenmp)
endif()
set_target_properties(test_openmp_isolation PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Fit-only test to isolate segfault to fit function
add_executable(test_fit_only
    test_fit_only.cpp
)
target_link_libraries(test_fit_only pacmap)
if(WIN32)
    target_compile_definitions(test_fit_only PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
set_target_properties(test_fit_only PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# Gradient analysis test for cross pattern investigation
add_executable(test_gradient_analysis
    test_gradient_analysis.cpp
)
target_link_libraries(test_gradient_analysis pacmap)
if(WIN32)
    target_compile_definitions(test_gradient_analysis PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
set_target_properties(test_gradient_analysis PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
)

# add_executable(test_mathematical_correctness
#     test_mathematical_correctness.cpp
# )
# target_link_libraries(test_mathematical_correctness pacmap)
# if(WIN32)
#     target_compile_definitions(test_mathematical_correctness PRIVATE _CRT_SECURE_NO_WARNINGS)
# endif()
# set_target_properties(test_mathematical_correctness PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
# )