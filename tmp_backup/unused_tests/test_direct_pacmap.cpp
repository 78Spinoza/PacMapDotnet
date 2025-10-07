#include "pacmap_simple_wrapper.h"
#include <iostream>
#include <vector>
#include <random>
#include <fstream>

int main() {
    std::cout << "=== DIRECT PACMAP FUNCTION TEST ===" << std::endl;

    // Create simple test data
    const int n_samples = 50;
    const int n_features = 3;
    const int n_components = 2;

    std::vector<float> data(n_samples * n_features);
    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Create 2 simple clusters
    for (int i = 0; i < n_samples/2; ++i) {
        for (int j = 0; j < n_features; ++j) {
            data[i * n_features + j] = dist(rng) + 5.0f;  // Cluster 1
        }
    }
    for (int i = n_samples/2; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            data[i * n_features + j] = dist(rng) - 5.0f;  // Cluster 2
        }
    }

    std::cout << "Created " << n_samples << " samples with " << n_features << " features" << std::endl;

    // Create model
    PacMapModel* model = pacmap_create();
    if (!model) {
        std::cout << "FAILED: Could not create model" << std::endl;
        return -1;
    }
    std::cout << "✅ Model created" << std::endl;

    // Prepare output embedding
    std::vector<float> embedding(n_samples * n_components);

    std::cout << "=== CALLING PACMAP FIT DIRECTLY ===" << std::endl;

    // Call PACMAP fit with minimal parameters to trigger debug output
    int result = pacmap_fit_with_progress_v2(
        model,
        data.data(), n_samples, n_features, n_components,
        10, 0.5f, 2.0f,  // n_neighbors, mn_ratio, fp_ratio
        1.0f, 50, 20, 15, 15,  // learning_rate, n_iters, phase1, phase2, phase3
        PACMAP_METRIC_EUCLIDEAN,
        embedding.data(),
        nullptr,  // No callback
        0, 16, 200, 200,  // HNSW params
        0,  // No quantization
        42,  // Random seed
        1   // autoHNSWParam
    );

    std::cout << "PACMAP result: " << result << std::endl;

    if (result == PACMAP_SUCCESS) {
        std::cout << "✅ PACMAP completed successfully" << std::endl;

        // Show some embedding points
        std::cout << "First 5 embedding points:" << std::endl;
        for (int i = 0; i < 5; ++i) {
            std::cout << "  Point " << i << ": ("
                      << embedding[i*2] << ", " << embedding[i*2+1] << ")" << std::endl;
        }

        // Calculate spread
        float min_val = embedding[0], max_val = embedding[0];
        for (float val : embedding) {
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
        std::cout << "Embedding range: [" << min_val << ", " << max_val << "]" << std::endl;

    } else {
        std::cout << "❌ PACMAP failed with error: " << result << std::endl;
        std::cout << "Error message: " << pacmap_get_error_message(result) << std::endl;
    }

    pacmap_destroy(model);
    std::cout << "=== TEST COMPLETE ===" << std::endl;
    return 0;
}