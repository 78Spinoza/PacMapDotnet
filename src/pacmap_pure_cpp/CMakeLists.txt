cmake_minimum_required(VERSION 3.12)
project(PACMAPWrapper VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Build options
option(BUILD_SHARED_LIBS "Build shared libraries" ON)
option(BUILD_TESTS "Build tests" ON)

# Find dependencies
find_package(OpenMP)

# Include directories (current directory since all files are in root)
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

# Source files - PACMAP implementation
set(PACMAP_WRAPPER_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/pacmap_simple_wrapper.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pacmap_model.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pacmap_utils.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pacmap_triplet_sampling.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pacmap_gradient.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pacmap_optimization.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pacmap_transform.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pacmap_persistence.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pacmap_distance.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/pacmap_crc32.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/lz4.c
)

# PACMAP core headers (review-optimized)
set(PACMAP_CORE_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/pacmap_model.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pacmap_utils.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pacmap_triplet_sampling.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pacmap_gradient.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pacmap_optimization.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pacmap_transform.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pacmap_persistence.h
    ${CMAKE_CURRENT_SOURCE_DIR}/pacmap_simple_wrapper.h
)

# Create the library
add_library(pacmap_wrapper ${PACMAP_WRAPPER_SOURCES})

# Link OpenMP AFTER target is created
if(OpenMP_CXX_FOUND)
    target_link_libraries(pacmap_wrapper PUBLIC OpenMP::OpenMP_CXX)
endif()

# Set library properties
set_target_properties(pacmap_wrapper PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    PUBLIC_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/pacmap_simple_wrapper.h"
)

# Compile definitions
target_compile_definitions(pacmap_wrapper PRIVATE PACMAP_EXPORTS)

# Suppress warnings from third-party headers
if(MSVC)
    target_compile_options(pacmap_wrapper PRIVATE
        /wd4244  # conversion from 'double' to 'float', possible loss of data
        /wd4267  # conversion from 'size_t' to 'int', possible loss of data
        /wd4566  # character represented by universal-character-name cannot be represented
    )
endif()

# Link libraries
target_link_libraries(pacmap_wrapper PRIVATE ${CMAKE_THREAD_LIBS_INIT})

# Platform-specific settings
if(WIN32)
    set_target_properties(pacmap_wrapper PROPERTIES
        OUTPUT_NAME "pacmap"
        SUFFIX ".dll"
    )
    target_compile_definitions(pacmap_wrapper PRIVATE WIN32_WINNT=0x0601)
else()
    set_target_properties(pacmap_wrapper PROPERTIES
        OUTPUT_NAME "pacmap"
        PREFIX "lib"
        SUFFIX ".so"
    )
endif()

# Install targets and create install tree export
install(TARGETS pacmap_wrapper
    EXPORT PACMAPWrapperTargets
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    PUBLIC_HEADER DESTINATION include
)

# Install PACMAP headers
install(FILES ${PACMAP_CORE_HEADERS}
    DESTINATION include/pacmap
)

# Tests - Essential validation tests
if(BUILD_TESTS)
    message(STATUS "Tests enabled - building essential validation tests")

    # Basic PACMAP integration test
    add_executable(test_pacmap_basic test_pacmap_basic.cpp)
    target_link_libraries(test_pacmap_basic pacmap_wrapper)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_pacmap_basic OpenMP::OpenMP_CXX)
    endif()

    # Simple error fixes test
    add_executable(test_error_fixes_simple test_error_fixes_simple.cpp)
    target_link_libraries(test_error_fixes_simple pacmap_wrapper)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_error_fixes_simple OpenMP::OpenMP_CXX)
    endif()

    # Comprehensive pipeline test
    add_executable(test_comprehensive_pipeline test_comprehensive_pipeline.cpp)
    target_link_libraries(test_comprehensive_pipeline pacmap_wrapper)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_comprehensive_pipeline OpenMP::OpenMP_CXX)
    endif()

    # STANDARD comprehensive test with strict pass/fail thresholds
    add_executable(test_standard_comprehensive test_standard_comprehensive.cpp)
    target_link_libraries(test_standard_comprehensive pacmap_wrapper)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_standard_comprehensive OpenMP::OpenMP_CXX)
    endif()

    # Legacy enhanced test (may need updates for current architecture)
    add_executable(test_enhanced_wrapper enhanced_test.cpp)
    target_link_libraries(test_enhanced_wrapper pacmap_wrapper)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_enhanced_wrapper OpenMP::OpenMP_CXX)
    endif()

    # Comprehensive quantization pipeline test with >1% diff stats
    add_executable(test_quantization_comprehensive test_quantization_comprehensive.cpp)
    target_link_libraries(test_quantization_comprehensive pacmap_wrapper)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_quantization_comprehensive OpenMP::OpenMP_CXX)
    endif()

    # Debug quantization save/load test
    add_executable(debug_quantization_simple debug_quantization_simple.cpp)
    target_link_libraries(debug_quantization_simple pacmap_wrapper)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(debug_quantization_simple OpenMP::OpenMP_CXX)
    endif()

    # Debug save/load test
    add_executable(debug_save_load_test debug_save_load_test.cpp)
    target_link_libraries(debug_save_load_test pacmap_wrapper)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(debug_save_load_test OpenMP::OpenMP_CXX)
    endif()

    # Zero-norm handling test
    add_executable(test_zero_norm_handling test_zero_norm_handling.cpp)
    target_link_libraries(test_zero_norm_handling pacmap_wrapper)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_zero_norm_handling OpenMP::OpenMP_CXX)
    endif()

    # ef_search scaling test
    add_executable(test_ef_search_scaling test_ef_search_scaling.cpp)
    target_link_libraries(test_ef_search_scaling pacmap_wrapper)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_ef_search_scaling OpenMP::OpenMP_CXX)
    endif()

    # Simple ef_search scaling test
    add_executable(test_ef_scaling_simple test_ef_scaling_simple.cpp)
    target_link_libraries(test_ef_scaling_simple pacmap_wrapper)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_ef_scaling_simple OpenMP::OpenMP_CXX)
    endif()

    # Distance conversion refactoring demo - DISABLED (missing distance_metrics namespace)
    # add_executable(refactoring_demo refactoring_demo.cpp)
    # target_link_libraries(refactoring_demo uwot_wrapper)
    # if(OpenMP_CXX_FOUND)
    #     target_link_libraries(refactoring_demo OpenMP::OpenMP_CXX)
    # endif()

    # Refactoring validation test - DISABLED (missing distance_metrics namespace)
    # add_executable(test_refactoring_validation test_refactoring_validation.cpp)
    # target_link_libraries(test_refactoring_validation uwot_wrapper)
    # if(OpenMP_CXX_FOUND)
    #     target_link_libraries(test_refactoring_validation OpenMP::OpenMP_CXX)
    # endif()

    # HNSW embedding extraction optimization test
    add_executable(test_embedding_extraction_optimization test_embedding_extraction_optimization.cpp)
    target_link_libraries(test_embedding_extraction_optimization pacmap_wrapper)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_embedding_extraction_optimization OpenMP::OpenMP_CXX)
    endif()

    # Distance helper functions test - DISABLED (missing distance_metrics namespace)
    # add_executable(test_distance_helpers test_distance_helpers.cpp)
    # target_link_libraries(test_distance_helpers pacmap_wrapper)
    # if(OpenMP_CXX_FOUND)
    #     target_link_libraries(test_distance_helpers OpenMP::OpenMP_CXX)
    # endif()

    # Distance simple test - DISABLED (missing distance_metrics namespace)
    # add_executable(test_distance_simple test_distance_simple.cpp)
    # target_link_libraries(test_distance_simple pacmap_wrapper)
    # if(OpenMP_CXX_FOUND)
    #     target_link_libraries(test_distance_simple OpenMP::OpenMP_CXX)
    # endif()

    # Debug transform accuracy test
    add_executable(debug_transform_accuracy debug_transform_accuracy.cpp)
    target_link_libraries(debug_transform_accuracy pacmap_wrapper)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(debug_transform_accuracy OpenMP::OpenMP_CXX)
    endif()

    # Embedding data preservation test
    add_executable(test_embedding_preservation test_embedding_preservation.cpp)
    target_link_libraries(test_embedding_preservation pacmap_wrapper)
    if(OpenMP_CXX_FOUND)
        target_link_libraries(test_embedding_preservation OpenMP::OpenMP_CXX)
    endif()
endif()

# Package configuration
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/PACMAPWrapperConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Export targets for build tree
export(TARGETS pacmap_wrapper
    FILE "${CMAKE_CURRENT_BINARY_DIR}/PACMAPWrapperTargets.cmake"
)

# Install the export set
install(EXPORT PACMAPWrapperTargets
    FILE PACMAPWrapperTargets.cmake
    DESTINATION lib/cmake/PACMAPWrapper
)