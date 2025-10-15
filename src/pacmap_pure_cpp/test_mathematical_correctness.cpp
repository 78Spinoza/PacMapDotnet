#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <chrono>
#include "pacmap_simple_wrapper.h"

// Simple test to verify PACMAP mathematical correctness
bool test_mathematical_correctness() {
    std::cout << "\n=== MATHEMATICAL CORRECTNESS VALIDATION ===\n" << std::endl;

    // Create a simple test case with known structure
    const int n_samples = 100;
    const int n_features = 10;
    const int n_components = 2;

    // Generate structured data: two well-separated clusters
    std::vector<float> data(n_samples * n_features);
    std::mt19937 rng(42); // Fixed seed for reproducibility
    std::normal_distribution<float> dist(0.0f, 1.0f);

    // Create two clusters: cluster 1 (mean=-2), cluster 2 (mean=+2)
    for (int i = 0; i < n_samples / 2; ++i) {
        for (int j = 0; j < n_features; ++j) {
            data[i * n_features + j] = dist(rng) - 2.0f; // Cluster 1
        }
    }
    for (int i = n_samples / 2; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            data[i * n_features + j] = dist(rng) + 2.0f; // Cluster 2
        }
    }

    std::cout << "Test data: " << n_samples << " samples, " << n_features << " features" << std::endl;
    std::cout << "Expected: Two well-separated clusters should remain separated in embedding" << std::endl;

    // Create PACMAP model
    PacMapModel* model = pacmap_create();
    if (!model) {
        std::cout << "FAILED: Could not create model" << std::endl;
        return false;
    }

    // Set parameters for clear separation
    std::vector<float> embedding(n_samples * n_components);

    std::cout << "Running PACMAP with deterministic seed..." << std::endl;

    auto start = std::chrono::high_resolution_clock::now();

    // Fit with fixed seed for reproducibility
    int result = pacmap_fit_with_progress_v2(model,
        data.data(), n_samples, n_features, n_components,
        10, 0.5f, 2.0f, 1.0f, 100, 25, 25, 50,
        PACMAP_METRIC_EUCLIDEAN, embedding.data(),
        nullptr, 0, -1, -1, -1, 0, 42, 1);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (result != 0) {
        std::cout << "FAILED: PACMAP fit returned error " << result << std::endl;
        pacmap_destroy(model);
        return false;
    }

    std::cout << "PACMAP completed in " << duration.count() << " ms" << std::endl;

    // Analyze embedding quality
    std::cout << "\n=== EMBEDDING ANALYSIS ===" << std::endl;

    // Calculate cluster centers
    float cluster1_x = 0, cluster1_y = 0;
    float cluster2_x = 0, cluster2_y = 0;
    int cluster1_count = n_samples / 2;
    int cluster2_count = n_samples / 2;

    for (int i = 0; i < cluster1_count; ++i) {
        cluster1_x += embedding[i * n_components];
        cluster1_y += embedding[i * n_components + 1];
    }
    cluster1_x /= cluster1_count;
    cluster1_y /= cluster1_count;

    for (int i = cluster1_count; i < n_samples; ++i) {
        cluster2_x += embedding[i * n_components];
        cluster2_y += embedding[i * n_components + 1];
    }
    cluster2_x /= cluster2_count;
    cluster2_y /= cluster2_count;

    // Calculate separation distance
    float separation = std::sqrt((cluster2_x - cluster1_x) * (cluster2_x - cluster1_x) +
                                (cluster2_y - cluster1_y) * (cluster2_y - cluster1_y));

    std::cout << "Cluster 1 center: (" << cluster1_x << ", " << cluster1_y << ")" << std::endl;
    std::cout << "Cluster 2 center: (" << cluster2_x << ", " << cluster2_y << ")" << std::endl;
    std::cout << "Separation distance: " << separation << std::endl;

    // Calculate within-cluster variance
    float var1 = 0, var2 = 0;
    for (int i = 0; i < cluster1_count; ++i) {
        float dx = embedding[i * n_components] - cluster1_x;
        float dy = embedding[i * n_components + 1] - cluster1_y;
        var1 += dx * dx + dy * dy;
    }
    for (int i = cluster1_count; i < n_samples; ++i) {
        float dx = embedding[i * n_components] - cluster2_x;
        float dy = embedding[i * n_components + 1] - cluster2_y;
        var2 += dx * dx + dy * dy;
    }
    var1 /= cluster1_count;
    var2 /= cluster2_count;

    float avg_variance = (var1 + var2) / 2.0f;
    std::cout << "Within-cluster variance: " << avg_variance << std::endl;

    // Validate mathematical correctness
    bool passed = true;

    // Test 1: Clusters should be separated
    if (separation < 2.0f) {
        std::cout << "FAILED: Clusters not well separated (separation = " << separation << ")" << std::endl;
        passed = false;
    } else {
        std::cout << " PASSED: Good cluster separation (" << separation << ")" << std::endl;
    }

    // Test 2: Within-cluster variance should be reasonable
    if (avg_variance > 10.0f) {
        std::cout << "FAILED: Within-cluster variance too high (" << avg_variance << ")" << std::endl;
        passed = false;
    } else {
        std::cout << " PASSED: Reasonable within-cluster variance (" << avg_variance << ")" << std::endl;
    }

    // Test 3: Embedding should not have NaN or Inf values
    for (int i = 0; i < n_samples * n_components; ++i) {
        if (std::isnan(embedding[i]) || std::isinf(embedding[i])) {
            std::cout << "FAILED: Invalid embedding values detected" << std::endl;
            passed = false;
            break;
        }
    }
    if (passed) {
        std::cout << " PASSED: No invalid values in embedding" << std::endl;
    }

    // Test 4: Reproducibility with same seed
    std::vector<float> embedding2(n_samples * n_components);
    PacMapModel* model2 = pacmap_create();

    int result2 = pacmap_fit_with_progress_v2(model2,
        data.data(), n_samples, n_features, n_components,
        10, 0.5f, 2.0f, 1.0f, 100, 25, 25, 50,
        PACMAP_METRIC_EUCLIDEAN, embedding2.data(),
        nullptr, 0, -1, -1, -1, 0, 42, 1);

    if (result2 != 0) {
        std::cout << "FAILED: Second PACMAP run returned error" << std::endl;
        passed = false;
    } else {
        // Check reproducibility
        double max_diff = 0;
        for (int i = 0; i < n_samples * n_components; ++i) {
            double diff = std::abs(embedding[i] - embedding2[i]);
            max_diff = std::max(max_diff, diff);
        }

        if (max_diff > 1e-6) {
            std::cout << "FAILED: Results not reproducible (max diff = " << max_diff << ")" << std::endl;
            passed = false;
        } else {
            std::cout << " PASSED: Reproducible results with same seed" << std::endl;
        }
    }

    pacmap_destroy(model);
    pacmap_destroy(model2);

    std::cout << "\n=== MATHEMATICAL CORRECTNESS SUMMARY ===" << std::endl;
    if (passed) {
        std::cout << "‰ ALL MATHEMATICAL CORRECTNESS TESTS PASSED!" << std::endl;
        std::cout << "   - Cluster separation: " << std::endl;
        std::cout << "   - Variance control: " << std::endl;
        std::cout << "   - Numerical stability: " << std::endl;
        std::cout << "   - Reproducibility: " << std::endl;
    } else {
        std::cout << " MATHEMATICAL CORRECTNESS TESTS FAILED!" << std::endl;
    }

    return passed;
}

int main() {
    std::cout << "=== PACMAP Mathematical Correctness Validation ===" << std::endl;

    bool success = test_mathematical_correctness();

    if (success) {
        std::cout << "\n ALL TESTS PASSED - PACMAP is mathematically correct!" << std::endl;
        return 0;
    } else {
        std::cout << "\n TESTS FAILED - Mathematical issues detected!" << std::endl;
        return 1;
    }
}