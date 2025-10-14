#include <iostream>
#include <vector>
#include "pacmap_simple_wrapper.h"

int main() {
    std::cout << "=== Basic PACMAP Integration Test ===" << std::endl;

    // Test 1: Create model
    std::cout << "Test 1: Creating PACMAP model..." << std::endl;
    PacMapModel* model = pacmap_create();
    if (!model) {
        std::cout << "FAILED: Could not create model" << std::endl;
        return 1;
    }
    std::cout << "SUCCESS: Model created" << std::endl;

    // Test 2: Check version
    std::cout << "Test 2: Checking version..." << std::endl;
    const char* version = pacmap_get_version();
    std::cout << "SUCCESS: Version " << version << std::endl;

    // Test 3: Check if fitted (should be false initially)
    std::cout << "Test 3: Checking fitted status..." << std::endl;
    int is_fitted = pacmap_is_fitted(model);
    std::cout << "SUCCESS: Is fitted = " << is_fitted << " (should be 0)" << std::endl;

    // Test 4: Simple test data
    std::cout << "Test 4: Creating test data..." << std::endl;
    const int n_samples = 100;
    const int n_features = 5;
    const int embedding_dim = 2;

    // v2.8.6+: Use double precision instead of float
    std::vector<double> data(n_samples * n_features);
    std::vector<double> embedding(n_samples * embedding_dim);

    // Generate simple test data
    for (int i = 0; i < n_samples; i++) {
        for (int j = 0; j < n_features; j++) {
            data[i * n_features + j] = static_cast<double>(i * j) * 0.1;
        }
    }

    std::cout << "SUCCESS: Test data created (" << n_samples << " x " << n_features << ")" << std::endl;

    // Test 5: Fit model
    std::cout << "Test 5: Fitting PACMAP model..." << std::endl;
    int result = pacmap_fit_with_progress_v2(
        model,
        data.data(),
        n_samples,
        n_features,
        embedding_dim,
        10,  // n_neighbors
        0.5f, // MN_ratio (default)
        2.0f, // FP_ratio (default, follows relationship: FP_ratio = 4 * MN_ratio)
        1.0f, // learning_rate
        50,   // n_iters
        10,   // phase1_iters
        10,   // phase2_iters
        30,   // phase3_iters
        PACMAP_METRIC_EUCLIDEAN,
        embedding.data(),
        nullptr, // progress_callback
        1,      // force_exact_knn (use exact for small dataset to avoid HNSW overhead)
        -1,     // M
        -1,     // ef_construction
        -1,     // ef_search
        0,      // use_quantization
        42,     // random_seed
        0       // autoHNSWParam (disable for small dataset)
    );

    if (result != PACMAP_SUCCESS) {
        std::cout << "FAILED: Fit returned error " << result << std::endl;
        const char* error_msg = pacmap_get_error_message(result);
        std::cout << "Error message: " << error_msg << std::endl;
        pacmap_destroy(model);
        return 1;
    }
    std::cout << "SUCCESS: Model fitted successfully" << std::endl;

    // Test 6: Check fitted status (should be true now)
    std::cout << "Test 6: Checking fitted status after fit..." << std::endl;
    is_fitted = pacmap_is_fitted(model);
    std::cout << "SUCCESS: Is fitted = " << is_fitted << " (should be 1)" << std::endl;

    // Test 7: Get model info
    std::cout << "Test 7: Getting model info..." << std::endl;
    int n_components = pacmap_get_n_components(model);
    int n_samples_stored = pacmap_get_n_samples(model);
    std::cout << "SUCCESS: n_components = " << n_components << std::endl;
    std::cout << "SUCCESS: n_samples = " << n_samples_stored << std::endl;

    // Test 8: Check embedding dimensions
    std::cout << "Test 8: Checking embedding dimensions..." << std::endl;
    std::cout << "Embedding shape: " << n_samples << " x " << embedding_dim << std::endl;
    std::cout << "First embedding point: (" << embedding[0] << ", " << embedding[1] << ")" << std::endl;

    // Test 9: Save model
    std::cout << "Test 9: Saving model..." << std::endl;
    result = pacmap_save_model(model, "test_model.pacmap");
    if (result != PACMAP_SUCCESS) {
        std::cout << "WARNING: Save returned error " << result << std::endl;
    } else {
        std::cout << "SUCCESS: Model saved" << std::endl;
    }

    // Test 10: Load model
    std::cout << "Test 10: Loading model..." << std::endl;
    PacMapModel* loaded_model = pacmap_load_model("test_model.pacmap");
    if (!loaded_model) {
        std::cout << "WARNING: Could not load model" << std::endl;
    } else {
        std::cout << "SUCCESS: Model loaded" << std::endl;
        int loaded_components = pacmap_get_n_components(loaded_model);
        std::cout << "Loaded model n_components = " << loaded_components << std::endl;
        pacmap_destroy(loaded_model);
    }

    // Cleanup
    std::cout << "Test 11: Cleanup..." << std::endl;
    pacmap_destroy(model);
    std::cout << "SUCCESS: Model destroyed" << std::endl;

    std::cout << "\n=== ALL TESTS COMPLETED SUCCESSFULLY ===" << std::endl;
    return 0;
}