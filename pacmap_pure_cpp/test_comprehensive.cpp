#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <fstream>
#include <chrono>
#include "pacmap_embedding_storage.h"

// Simple test to verify HNSW vs Raw Data save/load functionality
bool test_hnsw_vs_raw_data() {
    std::cout << "=== Test 1: HNSW vs Raw Data Save/Load ===" << std::endl;

    const int n_samples = 100;
    const int n_features = 10;
    const int embedding_dim = 2;
    const int random_seed = 12345;

    // Generate test data
    std::vector<float> data(n_samples * n_features);
    std::vector<float> embedding_hnsw(n_samples * embedding_dim);
    std::vector<float> embedding_raw(n_samples * embedding_dim);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n_samples * n_features; ++i) {
        data[i] = dist(rng);
    }

    std::cout << "Generated test data: " << n_samples << " samples x " << n_features << " features" << std::endl;
    std::cout << "Data size: " << (n_samples * n_features * sizeof(float) / 1024.0) << " KB" << std::endl;

    // === Test HNSW Mode ===
    std::cout << "\n--- Testing HNSW Mode (force_exact_knn = 0) ---" << std::endl;

    PacMapModel* model_hnsw = pacmap_create();
    if (!model_hnsw) {
        std::cout << "FAILED: Could not create HNSW model" << std::endl;
        return false;
    }

    int result = pacmap_fit_with_progress_v2(
        model_hnsw,
        data.data(),
        n_samples,
        n_features,
        embedding_dim,
        10,     // n_neighbors
        2.0f,   // MN_ratio
        1.0f,   // FP_ratio
        1.0f,   // learning_rate
        100,    // n_iters
        30,     // phase1_iters
        30,     // phase2_iters
        40,     // phase3_iters
        PACMAP_METRIC_EUCLIDEAN,
        embedding_hnsw.data(),
        nullptr, // progress_callback
        0,      // force_exact_knn = 0 (HNSW MODE)
        -1,     // M
        -1,     // ef_construction
        -1,     // ef_search
        0,      // use_quantization
        random_seed,
        1       // autoHNSWParam
    );

    if (result != PACMAP_SUCCESS) {
        std::cout << "FAILED: HNSW model fitting returned error " << result << std::endl;
        pacmap_destroy(model_hnsw);
        return false;
    }

    std::cout << "✓ HNSW model fitted successfully" << std::endl;
    std::cout << "  - Uses HNSW: " << (model_hnsw->uses_hnsw ? "YES" : "NO") << std::endl;
    std::cout << "  - Training data stored: " << (model_hnsw->training_data.empty() ? "NO (efficient!)" : "YES") << std::endl;
    std::cout << "  - Training embedding size: " << model_hnsw->training_embedding.size() << std::endl;

    // Save HNSW model
    result = pacmap_save_model(model_hnsw, "test_hnsw_model.pacmap");
    if (result != PACMAP_SUCCESS) {
        std::cout << "FAILED: HNSW model save returned error " << result << std::endl;
        pacmap_destroy(model_hnsw);
        return false;
    }

    // Check HNSW model file size
    std::ifstream hnsw_file("test_hnsw_model.pacmap", std::ios::binary | std::ios::ate);
    size_t hnsw_file_size = hnsw_file.tellg();
    hnsw_file.close();

    std::cout << "✓ HNSW model saved successfully" << std::endl;
    std::cout << "  - File size: " << (hnsw_file_size / 1024.0) << " KB" << std::endl;

    // Load HNSW model
    PacMapModel* loaded_hnsw = pacmap_load_model("test_hnsw_model.pacmap");
    if (!loaded_hnsw) {
        std::cout << "FAILED: Could not load HNSW model" << std::endl;
        pacmap_destroy(model_hnsw);
        return false;
    }

    std::cout << "✓ HNSW model loaded successfully" << std::endl;
    std::cout << "  - Uses HNSW: " << (loaded_hnsw->uses_hnsw ? "YES" : "NO") << std::endl;
    std::cout << "  - Training data loaded: " << (loaded_hnsw->training_data.empty() ? "NO" : "YES") << std::endl;
    std::cout << "  - Samples: " << pacmap_get_n_samples(loaded_hnsw) << std::endl;

    // === Test Raw Data Mode ===
    std::cout << "\n--- Testing Raw Data Mode (force_exact_knn = 1) ---" << std::endl;

    PacMapModel* model_raw = pacmap_create();
    if (!model_raw) {
        std::cout << "FAILED: Could not create raw data model" << std::endl;
        pacmap_destroy(model_hnsw);
        pacmap_destroy(loaded_hnsw);
        return false;
    }

    result = pacmap_fit_with_progress_v2(
        model_raw,
        data.data(),
        n_samples,
        n_features,
        embedding_dim,
        10,     // n_neighbors
        2.0f,   // MN_ratio
        1.0f,   // FP_ratio
        1.0f,   // learning_rate
        100,    // n_iters
        30,     // phase1_iters
        30,     // phase2_iters
        40,     // phase3_iters
        PACMAP_METRIC_EUCLIDEAN,
        embedding_raw.data(),
        nullptr, // progress_callback
        1,      // force_exact_knn = 1 (RAW DATA MODE)
        -1,     // M
        -1,     // ef_construction
        -1,     // ef_search
        0,      // use_quantization
        random_seed,
        1       // autoHNSWParam
    );

    if (result != PACMAP_SUCCESS) {
        std::cout << "FAILED: Raw data model fitting returned error " << result << std::endl;
        pacmap_destroy(model_hnsw);
        pacmap_destroy(loaded_hnsw);
        pacmap_destroy(model_raw);
        return false;
    }

    std::cout << "✓ Raw data model fitted successfully" << std::endl;
    std::cout << "  - Uses HNSW: " << (model_raw->uses_hnsw ? "YES" : "NO") << std::endl;
    std::cout << "  - Training data stored: " << (model_raw->training_data.empty() ? "NO" : "YES") << std::endl;
    std::cout << "  - Training data size: " << (model_raw->training_data.size() * sizeof(float) / 1024.0) << " KB" << std::endl;

    // Save raw data model
    result = pacmap_save_model(model_raw, "test_raw_model.pacmap");
    if (result != PACMAP_SUCCESS) {
        std::cout << "FAILED: Raw data model save returned error " << result << std::endl;
        pacmap_destroy(model_hnsw);
        pacmap_destroy(loaded_hnsw);
        pacmap_destroy(model_raw);
        return false;
    }

    // Check raw data model file size
    std::ifstream raw_file("test_raw_model.pacmap", std::ios::binary | std::ios::ate);
    size_t raw_file_size = raw_file.tellg();
    raw_file.close();

    std::cout << "✓ Raw data model saved successfully" << std::endl;
    std::cout << "  - File size: " << (raw_file_size / 1024.0) << " KB" << std::endl;

    // Load raw data model
    PacMapModel* loaded_raw = pacmap_load_model("test_raw_model.pacmap");
    if (!loaded_raw) {
        std::cout << "FAILED: Could not load raw data model" << std::endl;
        pacmap_destroy(model_hnsw);
        pacmap_destroy(loaded_hnsw);
        pacmap_destroy(model_raw);
        return false;
    }

    std::cout << "✓ Raw data model loaded successfully" << std::endl;
    std::cout << "  - Uses HNSW: " << (loaded_raw->uses_hnsw ? "YES" : "NO") << std::endl;
    std::cout << "  - Training data loaded: " << (loaded_raw->training_data.empty() ? "NO" : "YES") << std::endl;
    std::cout << "  - Training data size: " << (loaded_raw->training_data.size() * sizeof(float) / 1024.0) << " KB" << std::endl;

    // === Memory Comparison ===
    std::cout << "\n=== Memory Comparison ===" << std::endl;

    double memory_difference = (double)raw_file_size - hnsw_file_size;
    double percent_savings = (memory_difference / raw_file_size) * 100.0;

    std::cout << "Original training data size: " << (n_samples * n_features * sizeof(float) / 1024.0) << " KB" << std::endl;
    std::cout << "HNSW model file: " << (hnsw_file_size / 1024.0) << " KB" << std::endl;
    std::cout << "Raw data model file: " << (raw_file_size / 1024.0) << " KB" << std::endl;

    if (memory_difference > 0) {
        std::cout << "Memory savings with HNSW: " << (memory_difference / 1024.0) << " KB (" << percent_savings << "%)" << std::endl;
        std::cout << "✅ HNSW mode is MORE memory efficient!" << std::endl;
    } else {
        std::cout << "Raw data mode is smaller: " << (-memory_difference / 1024.0) << " KB" << std::endl;
        std::cout << "⚠️  Overhead dominates for small dataset" << std::endl;
    }

    // Cleanup
    pacmap_destroy(model_hnsw);
    pacmap_destroy(loaded_hnsw);
    pacmap_destroy(model_raw);
    pacmap_destroy(loaded_raw);

    // Clean up test files
    std::remove("test_hnsw_model.pacmap");
    std::remove("test_raw_model.pacmap");

    return true;
}

// Test transform functionality and consistency
bool test_transform_consistency() {
    std::cout << "\n=== Test 2: Transform Consistency ===" << std::endl;

    const int n_train_samples = 50;
    const int n_test_samples = 10;
    const int n_features = 5;
    const int embedding_dim = 2;
    const int random_seed = 42;

    // Generate training data
    std::vector<float> train_data(n_train_samples * n_features);
    std::vector<float> train_embedding(n_train_samples * embedding_dim);
    std::vector<float> test_data(n_test_samples * n_features);
    std::vector<float> test_embedding1(n_test_samples * embedding_dim);
    std::vector<float> test_embedding2(n_test_samples * embedding_dim);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (int i = 0; i < n_train_samples * n_features; ++i) {
        train_data[i] = dist(rng);
    }
    for (int i = 0; i < n_test_samples * n_features; ++i) {
        test_data[i] = dist(rng);
    }

    std::cout << "Generated training data: " << n_train_samples << " x " << n_features << std::endl;
    std::cout << "Generated test data: " << n_test_samples << " x " << n_features << std::endl;

    // Fit model
    PacMapModel* model = pacmap_create();
    if (!model) {
        std::cout << "FAILED: Could not create model" << std::endl;
        return false;
    }

    int result = pacmap_fit_with_progress_v2(
        model,
        train_data.data(),
        n_train_samples,
        n_features,
        embedding_dim,
        10, 2.0f, 1.0f, 1.0f, 100, 30, 30, 40,
        PACMAP_METRIC_EUCLIDEAN,
        train_embedding.data(),
        nullptr, 0, -1, -1, -1, 0, random_seed, 1
    );

    if (result != PACMAP_SUCCESS) {
        std::cout << "FAILED: Model fitting returned error " << result << std::endl;
        pacmap_destroy(model);
        return false;
    }

    std::cout << "✓ Model fitted successfully" << std::endl;
    std::cout << "  - Uses HNSW: " << (model->uses_hnsw ? "YES" : "NO") << std::endl;

    // First transform
    result = pacmap_transform(model, test_data.data(), n_test_samples, n_features, test_embedding1.data());
    if (result != PACMAP_SUCCESS) {
        std::cout << "FAILED: First transform returned error " << result << std::endl;
        pacmap_destroy(model);
        return false;
    }

    std::cout << "✓ First transform successful" << std::endl;

    // Second transform (should be identical if using same seed)
    result = pacmap_transform(model, test_data.data(), n_test_samples, n_features, test_embedding2.data());
    if (result != PACMAP_SUCCESS) {
        std::cout << "FAILED: Second transform returned error " << result << std::endl;
        pacmap_destroy(model);
        return false;
    }

    std::cout << "✓ Second transform successful" << std::endl;

    // Check consistency
    bool consistent = true;
    float max_diff = 0.0f;
    for (int i = 0; i < n_test_samples * embedding_dim; ++i) {
        float diff = std::abs(test_embedding1[i] - test_embedding2[i]);
        max_diff = std::max(max_diff, diff);
        if (diff > 1e-6f) {
            consistent = false;
        }
    }

    if (consistent) {
        std::cout << "✓ Transform results are consistent (max diff: " << max_diff << ")" << std::endl;
    } else {
        std::cout << "FAILED: Transform results inconsistent (max diff: " << max_diff << ")" << std::endl;
        pacmap_destroy(model);
        return false;
    }

    // Test transform with different data
    std::vector<float> different_test_data(n_test_samples * n_features);
    for (int i = 0; i < n_test_samples * n_features; ++i) {
        different_test_data[i] = dist(rng) * 2.0f; // Different range
    }

    std::vector<float> different_embedding(n_test_samples * embedding_dim);
    result = pacmap_transform(model, different_test_data.data(), n_test_samples, n_features, different_embedding.data());
    if (result != PACMAP_SUCCESS) {
        std::cout << "FAILED: Transform with different data returned error " << result << std::endl;
        pacmap_destroy(model);
        return false;
    }

    std::cout << "✓ Transform with different data successful" << std::endl;
    std::cout << "  First test embedding[0]: (" << test_embedding1[0] << ", " << test_embedding1[1] << ")" << std::endl;
    std::cout << "  Different data embedding[0]: (" << different_embedding[0] << ", " << different_embedding[1] << ")" << std::endl;

    pacmap_destroy(model);
    return true;
}

// Test save/load consistency
bool test_save_load_consistency() {
    std::cout << "\n=== Test 3: Save/Load Consistency ===" << std::endl;

    const int n_samples = 30;
    const int n_features = 3;
    const int embedding_dim = 2;
    const int random_seed = 123;

    std::vector<float> data(n_samples * n_features);
    std::vector<float> embedding1(n_samples * embedding_dim);
    std::vector<float> embedding2(n_samples * embedding_dim);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n_samples * n_features; ++i) {
        data[i] = dist(rng);
    }

    // Create and fit model
    PacMapModel* model1 = pacmap_create();
    if (!model1) {
        std::cout << "FAILED: Could not create model" << std::endl;
        return false;
    }

    int result = pacmap_fit_with_progress_v2(
        model1, data.data(), n_samples, n_features, embedding_dim,
        5, 2.0f, 1.0f, 1.0f, 50, 15, 15, 20,
        PACMAP_METRIC_EUCLIDEAN, embedding1.data(),
        nullptr, 0, -1, -1, -1, 0, random_seed, 1
    );

    if (result != PACMAP_SUCCESS) {
        std::cout << "FAILED: Model fitting returned error " << result << std::endl;
        pacmap_destroy(model1);
        return false;
    }

    std::cout << "✓ Original model fitted" << std::endl;

    // Save model
    result = pacmap_save_model(model1, "consistency_test_model.pacmap");
    if (result != PACMAP_SUCCESS) {
        std::cout << "FAILED: Model save returned error " << result << std::endl;
        pacmap_destroy(model1);
        return false;
    }

    std::cout << "✓ Model saved" << std::endl;

    // Load model
    PacMapModel* model2 = pacmap_load_model("consistency_test_model.pacmap");
    if (!model2) {
        std::cout << "FAILED: Could not load model" << std::endl;
        pacmap_destroy(model1);
        return false;
    }

    std::cout << "✓ Model loaded" << std::endl;

    // Compare model parameters
    if (pacmap_get_n_samples(model1) != pacmap_get_n_samples(model2) ||
        pacmap_get_n_features(model1) != pacmap_get_n_features(model2) ||
        pacmap_get_n_components(model1) != pacmap_get_n_components(model2)) {
        std::cout << "FAILED: Model parameters don't match" << std::endl;
        pacmap_destroy(model1);
        pacmap_destroy(model2);
        return false;
    }

    std::cout << "✓ Model parameters match" << std::endl;

    // Transform with both models
    std::vector<float> test_data(5 * n_features);
    for (int i = 0; i < 5 * n_features; ++i) {
        test_data[i] = dist(rng);
    }

    std::vector<float> transform1(5 * embedding_dim);
    std::vector<float> transform2(5 * embedding_dim);

    result = pacmap_transform(model1, test_data.data(), 5, n_features, transform1.data());
    if (result != PACMAP_SUCCESS) {
        std::cout << "FAILED: Transform with original model failed" << std::endl;
        pacmap_destroy(model1);
        pacmap_destroy(model2);
        return false;
    }

    result = pacmap_transform(model2, test_data.data(), 5, n_features, transform2.data());
    if (result != PACMAP_SUCCESS) {
        std::cout << "FAILED: Transform with loaded model failed" << std::endl;
        pacmap_destroy(model1);
        pacmap_destroy(model2);
        return false;
    }

    std::cout << "✓ Both models transform successfully" << std::endl;

    // Compare transform results
    bool transform_consistent = true;
    float max_transform_diff = 0.0f;
    for (int i = 0; i < 5 * embedding_dim; ++i) {
        float diff = std::abs(transform1[i] - transform2[i]);
        max_transform_diff = std::max(max_transform_diff, diff);
        if (diff > 1e-6f) {
            transform_consistent = false;
        }
    }

    if (transform_consistent) {
        std::cout << "✓ Transform results consistent (max diff: " << max_transform_diff << ")" << std::endl;
    } else {
        std::cout << "FAILED: Transform results inconsistent (max diff: " << max_transform_diff << ")" << std::endl;
        pacmap_destroy(model1);
        pacmap_destroy(model2);
        return false;
    }

    // Cleanup
    pacmap_destroy(model1);
    pacmap_destroy(model2);
    std::remove("consistency_test_model.pacmap");

    return true;
}

// Test error handling
bool test_error_handling() {
    std::cout << "\n=== Test 4: Error Handling ===" << std::endl;

    // Test invalid parameters
    PacMapModel* model = pacmap_create();
    if (!model) {
        std::cout << "FAILED: Could not create model" << std::endl;
        return false;
    }

    std::cout << "✓ Model created" << std::endl;

    // Test transform with unfitted model
    std::vector<float> test_data(10);
    std::vector<float> embedding(5);
    int result = pacmap_transform(model, test_data.data(), 5, 2, embedding.data());
    if (result != PACMAP_ERROR_MODEL_NOT_FITTED) {
        std::cout << "FAILED: Expected model not fitted error, got " << result << std::endl;
        pacmap_destroy(model);
        return false;
    }

    std::cout << "✓ Correctly detected unfitted model" << std::endl;

    // Test invalid parameters
    result = pacmap_fit_with_progress_v2(
        nullptr, nullptr, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        PACMAP_METRIC_EUCLIDEAN, nullptr, nullptr, 0, -1, -1, -1, 0, -1, 1
    );
    if (result != PACMAP_ERROR_INVALID_PARAMS) {
        std::cout << "FAILED: Expected invalid params error, got " << result << std::endl;
        pacmap_destroy(model);
        return false;
    }

    std::cout << "✓ Correctly detected invalid parameters" << std::endl;

    // Test load non-existent file
    PacMapModel* loaded = pacmap_load_model("non_existent_file.pacmap");
    if (loaded != nullptr) {
        std::cout << "FAILED: Expected null for non-existent file" << std::endl;
        pacmap_destroy(model);
        pacmap_destroy(loaded);
        return false;
    }

    std::cout << "✓ Correctly handled non-existent file" << std::endl;

    pacmap_destroy(model);
    return true;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "PACMAP Comprehensive Test Suite" << std::endl;
    std::cout << "HNSW-Optimized Implementation" << std::endl;
    std::cout << "========================================" << std::endl;

    bool all_passed = true;

    // Run all tests
    all_passed &= test_hnsw_vs_raw_data();
    all_passed &= test_transform_consistency();
    all_passed &= test_save_load_consistency();
    all_passed &= test_error_handling();

    // Final results
    std::cout << "\n========================================" << std::endl;
    std::cout << "FINAL TEST RESULTS" << std::endl;
    std::cout << "========================================" << std::endl;

    if (all_passed) {
        std::cout << "🎉 ALL TESTS PASSED!" << std::endl;
        std::cout << "✅ HNSW vs Raw Data save/load working correctly" << std::endl;
        std::cout << "✅ Transform consistency maintained" << std::endl;
        std::cout << "✅ Save/load functionality working" << std::endl;
        std::cout << "✅ Error handling working correctly" << std::endl;
        std::cout << "\n🚀 Implementation is production ready!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ SOME TESTS FAILED!" << std::endl;
        std::cout << "Please review the test output above for details." << std::endl;
        return 1;
    }
}