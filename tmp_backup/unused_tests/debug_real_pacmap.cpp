#include "pacmap_simple_wrapper.h"
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <fstream>
#include <iomanip>

// Create clear cluster data that PACMAP should easily separate
void create_cluster_data(std::vector<float>& data, std::vector<int>& labels, int n_samples_per_cluster = 50) {
    const int n_clusters = 3;
    const int n_features = 5;
    const int total_samples = n_clusters * n_samples_per_cluster;

    data.resize(total_samples * n_features);
    labels.resize(total_samples);

    std::mt19937 rng(42);  // Fixed seed for reproducibility
    std::normal_distribution<float> noise(0.0f, 0.1f);  // Small noise

    // Create 3 well-separated clusters in 5D space
    float cluster_centers[3][5] = {
        {5.0f, 5.0f, 5.0f, 5.0f, 5.0f},      // Cluster 0
        {-5.0f, -5.0f, -5.0f, -5.0f, -5.0f}, // Cluster 1
        {5.0f, -5.0f, 5.0f, -5.0f, 5.0f}      // Cluster 2
    };

    for (int cluster = 0; cluster < n_clusters; ++cluster) {
        for (int sample = 0; sample < n_samples_per_cluster; ++sample) {
            int idx = (cluster * n_samples_per_cluster + sample) * n_features;
            labels[cluster * n_samples_per_cluster + sample] = cluster;

            for (int dim = 0; dim < n_features; ++dim) {
                data[idx + dim] = cluster_centers[cluster][dim] + noise(rng);
            }
        }
    }
}

// Calculate simple cluster quality metric
float calculate_cluster_quality(const float* embedding, const int* labels, int n_samples) {
    float total_intra_cluster_dist = 0.0f;
    float total_inter_cluster_dist = 0.0f;
    int intra_pairs = 0, inter_pairs = 0;

    for (int i = 0; i < n_samples; ++i) {
        for (int j = i + 1; j < n_samples; ++j) {
            float dx = embedding[i*2] - embedding[j*2];
            float dy = embedding[i*2+1] - embedding[j*2+1];
            float dist = std::sqrt(dx*dx + dy*dy);

            if (labels[i] == labels[j]) {
                total_intra_cluster_dist += dist;
                intra_pairs++;
            } else {
                total_inter_cluster_dist += dist;
                inter_pairs++;
            }
        }
    }

    if (intra_pairs == 0 || inter_pairs == 0) return 0.0f;

    float avg_intra = total_intra_cluster_dist / intra_pairs;
    float avg_inter = total_inter_cluster_dist / inter_pairs;

    // Higher quality when inter-cluster distances are much larger than intra-cluster
    return avg_inter / (avg_intra + 1e-8f);
}

// Save embedding to CSV for visualization
void save_embedding_csv(const float* embedding, const int* labels, int n_samples, const char* filename) {
    std::ofstream file(filename);
    file << "x,y,label\n";
    for (int i = 0; i < n_samples; ++i) {
        file << embedding[i*2] << "," << embedding[i*2+1] << "," << labels[i] << "\n";
    }
    file.close();
}

int main() {
    std::cout << "=== REAL PACMAP ALGORITHM TEST ===" << std::endl;
    std::cout << "Testing PACMAP with well-separated cluster data" << std::endl;

    // Create test data with 3 clear clusters
    const int n_samples_per_cluster = 50;
    std::vector<float> data;
    std::vector<int> labels;
    create_cluster_data(data, labels, n_samples_per_cluster);

    const int n_samples = static_cast<int>(labels.size());
    const int n_features = 5;
    const int n_components = 2;

    std::cout << "Created " << n_samples << " samples with " << n_features << " features" << std::endl;
    std::cout << "3 clusters, " << n_samples_per_cluster << " samples each" << std::endl;

    // Check original data distances (should show clear separation)
    float orig_inter_dist = 0.0f, orig_intra_dist = 0.0f;
    int orig_inter_pairs = 0, orig_intra_pairs = 0;

    for (int i = 0; i < n_samples; ++i) {
        for (int j = i + 1; j < n_samples; ++j) {
            float dist = 0.0f;
            for (int d = 0; d < n_features; ++d) {
                float diff = data[i*n_features + d] - data[j*n_features + d];
                dist += diff * diff;
            }
            dist = std::sqrt(dist);

            if (labels[i] == labels[j]) {
                orig_intra_dist += dist;
                orig_intra_pairs++;
            } else {
                orig_inter_dist += dist;
                orig_inter_pairs++;
            }
        }
    }

    std::cout << "Original data:" << std::endl;
    std::cout << "  Avg intra-cluster distance: " << (orig_intra_dist/orig_intra_pairs) << std::endl;
    std::cout << "  Avg inter-cluster distance: " << (orig_inter_dist/orig_inter_pairs) << std::endl;
    std::cout << "  Original separation ratio: " << (orig_inter_dist/orig_intra_pairs) / (orig_intra_dist/orig_intra_pairs + 1e-8f) << std::endl;

    // Test PACMAP
    std::cout << "\n=== TESTING PACMAP ===" << std::endl;

    // Create model
    PacMapModel* model = pacmap_create();
    if (!model) {
        std::cout << "FAILED: Could not create PACMAP model" << std::endl;
        return -1;
    }
    std::cout << "✅ Model created" << std::endl;

    // Set good PACMAP parameters for clustering
    int n_neighbors = 10;
    float mn_ratio = 0.5f;
    float fp_ratio = 2.0f;
    int n_iters = 450;  // Standard PACMAP iterations

    // Run PACMAP with detailed parameters
    std::vector<float> embedding(n_samples * n_components);

    std::cout << "Running PACMAP with parameters:" << std::endl;
    std::cout << "  n_neighbors=" << n_neighbors << ", mn_ratio=" << mn_ratio << ", fp_ratio=" << fp_ratio << std::endl;
    std::cout << "  n_iters=" << n_iters << ", learning_rate=1.0 (AdaGrad)" << std::endl;
    std::cout << "  metric=EUCLIDEAN (forced)" << std::endl;

    int result = pacmap_fit_with_progress_v2(
        model,
        data.data(), n_samples, n_features, n_components,
        n_neighbors, mn_ratio, fp_ratio,
        1.0f, n_iters, 100, 150, 200,  // Standard PACMAP phase iterations
        PACMAP_METRIC_EUCLIDEAN,
        embedding.data(),
        nullptr,  // No callback for simplicity
        0, 16, 200, 200,  // HNSW parameters
        0,  // No quantization
        42,  // Fixed seed
        1   // autoHNSWParam
    );

    if (result != PACMAP_SUCCESS) {
        std::cout << "FAILED: PACMAP fitting failed with error " << result << std::endl;
        pacmap_destroy(model);
        return -1;
    }

    std::cout << "✅ PACMAP fitting completed" << std::endl;

    // Analyze embedding quality
    std::cout << "\n=== ANALYZING EMBEDDING QUALITY ===" << std::endl;

    // Calculate embedding statistics
    float min_x = embedding[0], max_x = embedding[0];
    float min_y = embedding[1], max_y = embedding[1];
    float sum_x = 0.0f, sum_y = 0.0f;

    for (int i = 0; i < n_samples; ++i) {
        float x = embedding[i*2];
        float y = embedding[i*2+1];

        min_x = std::min(min_x, x);
        max_x = std::max(max_x, x);
        min_y = std::min(min_y, y);
        max_y = std::max(max_y, y);
        sum_x += x;
        sum_y += y;
    }

    std::cout << "Embedding statistics:" << std::endl;
    std::cout << "  X range: [" << min_x << ", " << max_x << "]" << std::endl;
    std::cout << "  Y range: [" << min_y << ", " << max_y << "]" << std::endl;
    std::cout << "  Mean: (" << (sum_x/n_samples) << ", " << (sum_y/n_samples) << ")" << std::endl;

    // Check if embedding is just noise around origin
    float x_span = max_x - min_x;
    float y_span = max_y - min_y;
    float embedding_span = std::sqrt(x_span*x_span + y_span*y_span);

    if (embedding_span < 1.0f) {
        std::cout << "❌ EMBEDDING IS TOO COMPACT - likely random noise!" << std::endl;
    } else {
        std::cout << "✅ EMBEDDING has reasonable spread" << std::endl;
    }

    // Calculate cluster quality
    float quality = calculate_cluster_quality(embedding.data(), labels.data(), n_samples);
    std::cout << "Cluster separation quality: " << quality << std::endl;

    if (quality < 2.0f) {
        std::cout << "❌ POOR CLUSTERING - PACMAP is not working properly!" << std::endl;
    } else if (quality < 5.0f) {
        std::cout << "⚠️  MODERATE CLUSTERING - PACMAP may have issues" << std::endl;
    } else {
        std::cout << "✅ GOOD CLUSTERING - PACMAP is working!" << std::endl;
    }

    // Show sample points
    std::cout << "\nSample embedding points:" << std::endl;
    for (int i = 0; i < std::min(9, n_samples); ++i) {
        std::cout << "  Point " << i << " (label " << labels[i] << "): ("
                  << std::fixed << std::setprecision(3) << embedding[i*2] << ", "
                  << embedding[i*2+1] << ")" << std::endl;
    }

    // Save for visualization
    save_embedding_csv(embedding.data(), labels.data(), n_samples, "pacmap_cluster_test.csv");
    std::cout << "✅ Embedding saved to pacmap_cluster_test.csv" << std::endl;

    pacmap_destroy(model);

    std::cout << "\n=== CONCLUSION ===" << std::endl;
    if (quality > 5.0f && embedding_span > 1.0f) {
        std::cout << "✅ PACMAP algorithm is working correctly!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ PACMAP algorithm is still producing noise!" << std::endl;
        std::cout << "Quality: " << quality << " (need >5.0), Span: " << embedding_span << " (need >1.0)" << std::endl;
        return -1;
    }
}