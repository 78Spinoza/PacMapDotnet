#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <iomanip>
#include <cassert>
#include "pacmap_simple_wrapper.h"

// Simple test data generator
void generate_test_data(std::vector<float>& data, int n_samples, int n_features, int seed = 42) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    data.resize(n_samples * n_features);
    for (int i = 0; i < n_samples * n_features; ++i) {
        data[i] = dist(rng);
    }
}

// Progress callback for testing
void test_progress_callback_v2(const char* phase, int current, int total, float percent, const char* message) {
    std::cout << "[" << std::setw(12) << phase << "] "
              << std::setw(6) << current << "/" << total
              << " (" << std::fixed << std::setprecision(1) << percent << "%)";
    if (message) {
        std::cout << " - " << message;
    }
    std::cout << std::endl;
}

// Test basic PACMAP functionality
int test_basic_pacmap() {
    std::cout << "=== Testing Basic PACMAP Functionality ===" << std::endl;

    // Test parameters
    const int n_samples = 1000;
    const int n_features = 50;
    const int n_components = 2;
    const int n_neighbors = 10;
    const float MN_ratio = 0.5f;
    const float FP_ratio = 2.0f;
    const float learning_rate = 1.0f;
    const int n_iters = 450;
    const int phase1_iters = 100;
    const int phase2_iters = 100;
    const int phase3_iters = 250;

    // Generate test data
    std::vector<float> data;
    generate_test_data(data, n_samples, n_features);

    // Create PACMAP model
    PacMapModel* model = pacmap_create();
    if (!model) {
        std::cerr << "ERROR: Failed to create PACMAP model" << std::endl;
        return -1;
    }

    // Prepare output embedding
    std::vector<float> embedding(n_samples * n_components);

    std::cout << "Running PACMAP with " << n_samples << " samples, " << n_features
              << " features..." << std::endl;

    // Time the fitting process
    auto start_time = std::chrono::high_resolution_clock::now();

    // Fit PACMAP model
    int result = pacmap_fit_with_progress_v2(
        model,
        data.data(),
        n_samples,
        n_features,
        n_components,
        n_neighbors,
        MN_ratio,
        FP_ratio,
        learning_rate,
        n_iters,
        phase1_iters,
        phase2_iters,
        phase3_iters,
        PACMAP_METRIC_EUCLIDEAN,
        embedding.data(),
        test_progress_callback_v2,
        0,  // force_exact_knn
        -1, // M (auto)
        -1, // ef_construction (auto)
        -1, // ef_search (auto)
        0,  // use_quantization
        42, // random_seed
        1   // autoHNSWParam
    );

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    if (result != PACMAP_SUCCESS) {
        std::cerr << "ERROR: PACMAP fitting failed with error code: " << result << std::endl;
        const char* error_msg = pacmap_get_error_message(result);
        if (error_msg) {
            std::cerr << "Error message: " << error_msg << std::endl;
        }
        pacmap_destroy(model);
        return -1;
    }

    std::cout << "PACMAP fitting completed successfully in " << duration.count() << " ms" << std::endl;

    // Validate embedding
    bool embedding_valid = true;
    for (int i = 0; i < embedding.size(); ++i) {
        if (std::isnan(embedding[i]) || std::isinf(embedding[i])) {
            std::cerr << "ERROR: Invalid embedding value at index " << i << ": " << embedding[i] << std::endl;
            embedding_valid = false;
            break;
        }
    }

    if (!embedding_valid) {
        pacmap_destroy(model);
        return -1;
    }

    // Test model information
    int info_n_samples, info_n_features, info_n_components, info_n_neighbors;
    float info_MN_ratio, info_FP_ratio, info_learning_rate;
    int info_n_iters, info_phase1_iters, info_phase2_iters, info_phase3_iters;
    PacMapMetric info_metric;
    int info_hnsw_M, info_hnsw_ef_construction, info_hnsw_ef_search;

    result = pacmap_get_model_info(
        model,
        &info_n_samples,
        &info_n_features,
        &info_n_components,
        &info_n_neighbors,
        &info_MN_ratio,
        &info_FP_ratio,
        &info_learning_rate,
        &info_n_iters,
        &info_phase1_iters,
        &info_phase2_iters,
        &info_phase3_iters,
        &info_metric,
        &info_hnsw_M,
        &info_hnsw_ef_construction,
        &info_hnsw_ef_search
    );

    if (result == PACMAP_SUCCESS) {
        std::cout << "Model Information:" << std::endl;
        std::cout << "  Samples: " << info_n_samples << std::endl;
        std::cout << "  Features: " << info_n_features << std::endl;
        std::cout << "  Components: " << info_n_components << std::endl;
        std::cout << "  Neighbors: " << info_n_neighbors << std::endl;
        std::cout << "  MN Ratio: " << info_MN_ratio << std::endl;
        std::cout << "  FP Ratio: " << info_FP_ratio << std::endl;
        std::cout << "  Learning Rate: " << info_learning_rate << std::endl;
        std::cout << "  Total Iterations: " << info_n_iters << std::endl;
        std::cout << "  Phase Iterations: " << info_phase1_iters << "/"
                  << info_phase2_iters << "/" << info_phase3_iters << std::endl;
        std::cout << "  Metric: " << pacmap_get_metric_name(info_metric) << std::endl;
        std::cout << "  HNSW M: " << info_hnsw_M << std::endl;
        std::cout << "  HNSW ef_construction: " << info_hnsw_ef_construction << std::endl;
        std::cout << "  HNSW ef_search: " << info_hnsw_ef_search << std::endl;
    }

    // Clean up
    pacmap_destroy(model);

    std::cout << "Basic PACMAP test completed successfully!" << std::endl;
    return 0;
}

// Test model persistence
int test_model_persistence() {
    std::cout << "\n=== Testing Model Persistence ===" << std::endl;

    // Test parameters
    const int n_samples = 500;
    const int n_features = 20;
    const int n_components = 2;

    // Generate test data
    std::vector<float> data;
    generate_test_data(data, n_samples, n_features, 123);

    // Create and fit model
    PacMapModel* model = pacmap_create();
    std::vector<float> embedding1(n_samples * n_components);

    int result = pacmap_fit_with_progress_v2(
        model,
        data.data(),
        n_samples,
        n_features,
        n_components,
        10,  // n_neighbors
        0.5f, // MN_ratio
        2.0f, // FP_ratio
        1.0f, // learning_rate
        450,  // n_iters
        100,  // phase1_iters
        100,  // phase2_iters
        250,  // phase3_iters
        PACMAP_METRIC_EUCLIDEAN,
        embedding1.data(),
        nullptr, // callback
        0, -1, -1, -1, 0, 123, 1
    );

    if (result != PACMAP_SUCCESS) {
        std::cerr << "ERROR: Failed to fit model for persistence test" << std::endl;
        pacmap_destroy(model);
        return -1;
    }

    // Save model
    const char* test_filename = "test_pacmap_model.bin";
    result = pacmap_save_model(model, test_filename);
    if (result != PACMAP_SUCCESS) {
        std::cerr << "ERROR: Failed to save model, error code: " << result << std::endl;
        pacmap_destroy(model);
        return -1;
    }

    std::cout << "Model saved successfully to " << test_filename << std::endl;

    // Destroy original model
    pacmap_destroy(model);

    // Load model
    PacMapModel* loaded_model = pacmap_load_model(test_filename);
    if (!loaded_model) {
        std::cerr << "ERROR: Failed to load model" << std::endl;
        return -1;
    }

    std::cout << "Model loaded successfully" << std::endl;

    // Transform with loaded model
    std::vector<float> embedding2(n_samples * n_components);
    result = pacmap_transform(loaded_model, data.data(), n_samples, n_features, embedding2.data());

    if (result != PACMAP_SUCCESS) {
        std::cerr << "ERROR: Failed to transform with loaded model, error code: " << result << std::endl;
        pacmap_destroy(loaded_model);
        return -1;
    }

    // Compare embeddings (should be identical or very close)
    double max_diff = 0.0;
    for (int i = 0; i < embedding1.size(); ++i) {
        double diff = std::abs(embedding1[i] - embedding2[i]);
        max_diff = std::max(max_diff, diff);
    }

    std::cout << "Maximum embedding difference after save/load: " << max_diff << std::endl;

    // Clean up
    pacmap_destroy(loaded_model);

    // Remove test file (optional)
    // std::remove(test_filename);

    std::cout << "Model persistence test completed successfully!" << std::endl;
    return 0;
}

int main() {
    std::cout << "PACMAP C++ Integration Tests" << std::endl;
    std::cout << "Version: " << pacmap_get_version() << std::endl;
    std::cout << "==========================================\n" << std::endl;

    int result = 0;

    // Run basic functionality test
    if (test_basic_pacmap() != 0) {
        std::cerr << "Basic PACMAP test failed!" << std::endl;
        result = -1;
    }

    // Run persistence test
    if (test_model_persistence() != 0) {
        std::cerr << "Model persistence test failed!" << std::endl;
        result = -1;
    }

    if (result == 0) {
        std::cout << "\n=== ALL TESTS PASSED ===" << std::endl;
    } else {
        std::cout << "\n=== SOME TESTS FAILED ===" << std::endl;
    }

    return result;
}