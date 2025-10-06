#include "pacmap_simple_wrapper.h"
#include "pacmap_utils.h"
#include <iostream>
#include <vector>
#include <random>
#include <iomanip>

// Simple test progress callback
void test_progress_callback(const char* phase, int current, int total, float percent, const char* message) {
    std::cout << "Progress: " << phase << " - " << current << "/" << total
              << " (" << std::fixed << std::setprecision(1) << percent << "%)";
    if (message) {
        std::cout << " - " << message;
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "=== PACMAP Simple Minimal Test ===" << std::endl;

    // Test 1: Model creation and destruction
    std::cout << "\n1. Testing model creation..." << std::endl;
    PacMapModel* model = pacmap_create();
    if (!model) {
        std::cerr << "ERROR: Failed to create model" << std::endl;
        return 1;
    }
    std::cout << "SUCCESS: Model created successfully" << std::endl;

    // Test 2: Create simple test data
    std::cout << "\n2. Creating test data..." << std::endl;
    const int n_samples = 100;
    const int n_features = 10;
    const int embedding_dim = 2;

    std::vector<float> data(n_samples * n_features);
    std::vector<float> embedding(n_samples * embedding_dim);

    // Generate random test data
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (int i = 0; i < n_samples * n_features; ++i) {
        data[i] = dist(rng);
    }

    std::cout << "Created " << n_samples << " samples with " << n_features << " features each" << std::endl;

    // Test 3: Run PACMAP fitting
    std::cout << "\n3. Running PACMAP fitting..." << std::endl;
    int result = pacmap_fit_with_progress_v2(
        model,
        data.data(),
        n_samples,
        n_features,
        embedding_dim,
        10,              // n_neighbors
        0.5f,            // MN_ratio
        2.0f,            // FP_ratio
        1.0f,            // learning_rate
        450,             // n_iters
        100,             // phase1_iters
        100,             // phase2_iters
        250,             // phase3_iters
        PACMAP_METRIC_EUCLIDEAN,
        embedding.data(),
        test_progress_callback,
        0,   // force_exact_knn
        -1,  // M
        -1,  // ef_construction
        -1,  // ef_search
        0,   // use_quantization
        42,  // random_seed
        1    // autoHNSWParam
    );

    if (result != PACMAP_SUCCESS) {
        std::cerr << "ERROR: PACMAP fitting failed with code: " << result << std::endl;
        pacmap_destroy(model);
        return 1;
    }
    std::cout << "SUCCESS: PACMAP fitting completed" << std::endl;

    // Test 4: Verify embedding results
    std::cout << "\n4. Verifying embedding results..." << std::endl;
    bool embedding_valid = true;
    for (int i = 0; i < n_samples * embedding_dim; ++i) {
        if (std::isnan(embedding[i]) || std::isinf(embedding[i])) {
            embedding_valid = false;
            break;
        }
    }

    if (!embedding_valid) {
        std::cerr << "ERROR: Embedding contains NaN or Inf values" << std::endl;
        pacmap_destroy(model);
        return 1;
    }

    // Show first few embedding points
    std::cout << "First 5 embedding points:" << std::endl;
    for (int i = 0; i < std::min(5, n_samples); ++i) {
        std::cout << "  Point " << i << ": ("
                  << std::fixed << std::setprecision(3) << embedding[i * embedding_dim]
                  << ", " << embedding[i * embedding_dim + 1] << ")" << std::endl;
    }
    std::cout << "SUCCESS: Embedding validation passed" << std::endl;

    // Test 5: Test save/load functionality
    std::cout << "\n5. Testing save/load functionality..." << std::endl;
    const char* test_filename = "test_model.pacmap";

    // Test save
    if (pacmap_save_model(model, test_filename) != PACMAP_SUCCESS) {
        std::cerr << "ERROR: Failed to save model" << std::endl;
        pacmap_destroy(model);
        return 1;
    }
    std::cout << "SUCCESS: Model saved to " << test_filename << std::endl;

    // Test load
    PacMapModel* loaded_model = pacmap_create();
    if (!loaded_model) {
        std::cerr << "ERROR: Failed to create model for loading" << std::endl;
        pacmap_destroy(model);
        return 1;
    }

    loaded_model = pacmap_load_model(test_filename);
    if (loaded_model == nullptr) {
        std::cerr << "ERROR: Failed to load model" << std::endl;
        pacmap_destroy(model);
        pacmap_destroy(loaded_model);
        return 1;
    }
    std::cout << "SUCCESS: Model loaded successfully" << std::endl;

    // Test 6: Test transform functionality
    std::cout << "\n6. Testing transform functionality..." << std::endl;
    const int n_new_samples = 10;
    std::vector<float> new_data(n_new_samples * n_features);
    std::vector<float> new_embedding(n_new_samples * embedding_dim);

    // Generate new test data
    for (int i = 0; i < n_new_samples * n_features; ++i) {
        new_data[i] = dist(rng);
    }

    if (pacmap_transform(loaded_model, new_data.data(), n_new_samples, n_features, new_embedding.data()) != PACMAP_SUCCESS) {
        std::cerr << "ERROR: Transform failed" << std::endl;
        pacmap_destroy(model);
        pacmap_destroy(loaded_model);
        return 1;
    }

    std::cout << "Transformed " << n_new_samples << " new samples successfully" << std::endl;
    std::cout << "First transformed point: ("
              << std::fixed << std::setprecision(3) << new_embedding[0]
              << ", " << new_embedding[1] << ")" << std::endl;
    std::cout << "SUCCESS: Transform functionality working" << std::endl;

    // Cleanup
    pacmap_destroy(model);
    pacmap_destroy(loaded_model);

    // Remove test file
    std::remove(test_filename);

    std::cout << "\n=== ALL TESTS PASSED ===" << std::endl;
    std::cout << "PACMAP minimal implementation is working correctly!" << std::endl;

    return 0;
}