#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include "pacmap_simple_wrapper.h"

// Simple test data generator
void create_simple_test_data(float* data, int n_samples, int n_dim) {
    std::mt19937 rng(42);
    std::normal_distribution<float> normal(0.0f, 0.5f);

    // Create 3 clear clusters
    for (int i = 0; i < n_samples; i++) {
        int cluster = i / (n_samples / 3);

        if (cluster == 0) {
            // Cluster 1: centered at (0, 0, 0)
            for (int j = 0; j < n_dim; j++) {
                data[i * n_dim + j] = normal(rng);
            }
        } else if (cluster == 1) {
            // Cluster 2: centered at (3, 3, 3)
            for (int j = 0; j < n_dim; j++) {
                data[i * n_dim + j] = 3.0f + normal(rng);
            }
        } else {
            // Cluster 3: centered at (-3, -3, -3)
            for (int j = 0; j < n_dim; j++) {
                data[i * n_dim + j] = -3.0f + normal(rng);
            }
        }
    }
}

// Progress callback for debugging
void debug_callback(const char* phase, int current, int total, float percent, const char* message) {
    std::cout << "CALLBACK: " << phase << " (" << current << "/" << total << ") "
              << percent << "% - " << (message ? message : "no message") << std::endl;
}

int main() {
    std::cout << "=== STANDALONE PACMAP DEBUG TEST ===" << std::endl;

    // Test parameters
    const int n_samples = 300;
    const int n_dim = 3;
    const int embedding_dim = 2;
    const int n_neighbors = 5;

    std::cout << "Creating test data: " << n_samples << " samples, " << n_dim << " dimensions" << std::endl;

    // Create test data
    std::vector<float> data(n_samples * n_dim);
    create_simple_test_data(data.data(), n_samples, n_dim);

    std::cout << "Sample data points:" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "  Point " << i << ": ";
        for (int j = 0; j < n_dim; j++) {
            std::cout << data[i * n_dim + j] << " ";
        }
        std::cout << std::endl;
    }

    // Create PACMAP model
    std::cout << "\nCreating PACMAP model..." << std::endl;
    PacMapModel* model = pacmap_create();
    if (!model) {
        std::cout << "ERROR: Failed to create model!" << std::endl;
        return 1;
    }
    std::cout << "Model created successfully" << std::endl;

    // Create output buffer
    std::vector<float> embedding(n_samples * embedding_dim);

    // PACMAP parameters
    float mn_ratio = 0.5f;
    float fp_ratio = 2.0f;
    float learning_rate = 1.0f;
    int n_iters = 100;
    int phase1_iters = 100;
    int phase2_iters = 100;
    int phase3_iters = 100;
    PacMapMetric metric = PACMAP_METRIC_EUCLIDEAN;
    int random_seed = 42;

    std::cout << "\nStarting PACMAP fit with debug output..." << std::endl;
    std::cout << "Parameters:" << std::endl;
    std::cout << "  n_neighbors=" << n_neighbors << ", mn_ratio=" << mn_ratio << ", fp_ratio=" << fp_ratio << std::endl;
    std::cout << "  learning_rate=" << learning_rate << ", n_iters=" << n_iters << std::endl;
    std::cout << "  phase1=" << phase1_iters << ", phase2=" << phase2_iters << ", phase3=" << phase3_iters << std::endl;

    auto start_time = std::chrono::high_resolution_clock::now();

    // Run PACMAP with debug output
    int result = pacmap_fit_with_progress_v2(
        model,
        data.data(),
        n_samples,
        n_dim,
        embedding_dim,
        n_neighbors,
        mn_ratio,
        fp_ratio,
        learning_rate,
        n_iters,
        phase1_iters,
        phase2_iters,
        phase3_iters,
        metric,
        embedding.data(),
        debug_callback,
        0,  // force_exact_knn
        16, // M
        200, // ef_construction
        200, // ef_search
        0,  // use_quantization
        random_seed,
        1   // autoHNSWParam
    );

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);

    std::cout << "\nPACMAP completed with result: " << result << std::endl;
    std::cout << "Total time: " << duration.count() << " ms" << std::endl;

    if (result == PACMAP_SUCCESS) {
        std::cout << "\nPACMAP SUCCESS - Embedding results:" << std::endl;
        for (int i = 0; i < std::min(10, n_samples); i++) {
            std::cout << "  Point " << i << ": ";
            for (int j = 0; j < embedding_dim; j++) {
                std::cout << embedding[i * embedding_dim + j] << " ";
            }
            std::cout << std::endl;
        }

        // Calculate embedding variance to check if it's meaningful
        float sum_x = 0.0f, sum_y = 0.0f;
        float sum_sq_x = 0.0f, sum_sq_y = 0.0f;

        for (int i = 0; i < n_samples; i++) {
            float x = embedding[i * embedding_dim + 0];
            float y = embedding[i * embedding_dim + 1];
            sum_x += x;
            sum_y += y;
            sum_sq_x += x * x;
            sum_sq_y += y * y;
        }

        float mean_x = sum_x / n_samples;
        float mean_y = sum_y / n_samples;
        float var_x = (sum_sq_x / n_samples) - (mean_x * mean_x);
        float var_y = (sum_sq_y / n_samples) - (mean_y * mean_y);

        std::cout << "\nEmbedding statistics:" << std::endl;
        std::cout << "  Mean X: " << mean_x << ", Variance X: " << var_x << std::endl;
        std::cout << "  Mean Y: " << mean_y << ", Variance Y: " << var_y << std::endl;

        if (var_x < 1e-6f && var_y < 1e-6f) {
            std::cout << "  WARNING: Embedding has near-zero variance - algorithm may not be working!" << std::endl;
        } else {
            std::cout << "  OK: Embedding shows meaningful variance" << std::endl;
        }
    } else {
        std::cout << "PACMAP FAILED with error code: " << result << std::endl;
        const char* error_msg = pacmap_get_error_message(result);
        std::cout << "Error message: " << (error_msg ? error_msg : "Unknown error") << std::endl;
    }

    // Cleanup
    pacmap_destroy(model);

    std::cout << "\n=== TEST COMPLETE ===" << std::endl;
    return 0;
}