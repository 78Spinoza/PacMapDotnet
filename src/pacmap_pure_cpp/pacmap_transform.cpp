#include "pacmap_transform.h"
#include "pacmap_simple_wrapper.h"
#include "pacmap_distance.h"
#include "pacmap_triplet_sampling.h"
#include "pacmap_utils.h"
#include <iostream>
#include <algorithm>
#include <numeric>
#include <cmath>

namespace transform_utils {

    // PACMAP transform: Weighted interpolation using nearest neighbors (internal implementation)
    int internal_pacmap_transform_detailed(
        PacMapModel* model,
        double* new_data,
        int n_new_obs,
        int n_dim,
        double* embedding,
        int* nn_indices,
        double* nn_distances,
        double* confidence_score,
        int* outlier_level,
        double* percentile_rank,
        double* z_score
    ) {
        if (!model || !model->is_fitted || !new_data || !embedding ||
            n_new_obs <= 0 || n_dim != model->n_features) {
            return PACMAP_ERROR_INVALID_PARAMS;
        }

        try {
            std::vector<double> new_embedding(static_cast<size_t>(n_new_obs) * static_cast<size_t>(model->n_components));

            for (int i = 0; i < n_new_obs; i++) {
                // Extract point from new data
                std::vector<double> point(n_dim);
                for (int j = 0; j < n_dim; j++) {
                    size_t idx = static_cast<size_t>(i) * static_cast<size_t>(n_dim) + static_cast<size_t>(j);
                    point[j] = new_data[idx];
                }

                // Normalize point using same preprocessing as training
                std::vector<double> normalized_point = point;
                if (!model->feature_means.empty() && !model->feature_stds.empty()) {
                    for (int j = 0; j < n_dim; j++) {
                        normalized_point[j] = (point[j] - model->feature_means[j]) / (model->feature_stds[j] + 1e-8f);
                    }
                }

                // Find k-nearest neighbors in original space (HNSW or KNN direct mode)
                std::vector<std::pair<float, size_t>> neighbors;

                if (model->force_exact_knn || !model->original_space_index) {
                    // KNN direct mode: brute-force search through training data (double precision)
                    if (!model->training_data.empty()) {
                        std::vector<std::pair<float, size_t>> all_distances;
                        all_distances.reserve(model->n_samples);

                        for (int j = 0; j < model->n_samples; j++) {
                            const double* training_point = &model->training_data[static_cast<size_t>(j) * static_cast<size_t>(n_dim)];

                            // Calculate distance based on metric (double precision internally, return as float)
                            double distance_double = 0.0;
                            switch (model->metric) {
                                case PACMAP_METRIC_EUCLIDEAN:
                                    for (int d = 0; d < n_dim; d++) {
                                        double diff = static_cast<double>(normalized_point[d]) - training_point[d];
                                        distance_double += diff * diff;
                                    }
                                    distance_double = std::sqrt(distance_double);
                                    break;
                                case PACMAP_METRIC_COSINE:
                                    {
                                        double dot_product = 0.0, norm_a = 0.0, norm_b = 0.0;
                                        for (int d = 0; d < n_dim; d++) {
                                            double a = static_cast<double>(normalized_point[d]);
                                            double b = training_point[d];
                                            dot_product += a * b;
                                            norm_a += a * a;
                                            norm_b += b * b;
                                        }
                                        norm_a = std::sqrt(norm_a);
                                        norm_b = std::sqrt(norm_b);
                                        distance_double = (norm_a > 1e-8 && norm_b > 1e-8) ?
                                                  (1.0 - dot_product / (norm_a * norm_b)) : 1.0;
                                    }
                                    break;
                                default:
                                    // ERROR13 FIX: Use true Euclidean distance for default case
                                    for (int d = 0; d < n_dim; d++) {
                                        double diff = static_cast<double>(normalized_point[d]) - training_point[d];
                                        distance_double += diff * diff;
                                    }
                                    distance_double = std::sqrt(std::max(0.0, distance_double));
                                    break;
                            }
                            float distance = static_cast<float>(distance_double);

                            all_distances.emplace_back(distance, j);
                        }

                        // Sort and take k-nearest neighbors
                        std::partial_sort(all_distances.begin(),
                                        all_distances.begin() + std::min(model->n_neighbors, model->n_samples),
                                        all_distances.end());

                        int k_actual = std::min(model->n_neighbors, model->n_samples);
                        for (int k = 0; k < k_actual; k++) {
                            neighbors.push_back(all_distances[k]);
                        }
                    }
                } else {
                    // HNSW mode: use HNSW index for fast search
                    // Convert double to float for HNSW interface
                    std::vector<float> normalized_point_float(n_dim);
                    for (int d = 0; d < n_dim; d++) {
                        normalized_point_float[d] = static_cast<float>(normalized_point[d]);
                    }
                    auto knn_results = model->original_space_index->searchKnn(normalized_point_float.data(), model->n_neighbors);

                    // Convert priority_queue to vector
                    while (!knn_results.empty()) {
                        neighbors.push_back(knn_results.top());
                        knn_results.pop();
                    }
                }

                // Calculate inverse distance weights
                std::vector<float> weights;
                float total_weight = 0.0f;
                for (const auto& neighbor : neighbors) {
                    float distance = neighbor.first;
                    // Convert HNSW distance based on metric
                    switch (model->metric) {
                        case PACMAP_METRIC_EUCLIDEAN:
                            distance = std::sqrt(std::max(0.0f, distance));
                            break;
                        case PACMAP_METRIC_COSINE:
                            distance = std::max(0.0f, std::min(2.0f, 1.0f + distance));
                            break;
                        default:
                            distance = std::max(0.0f, distance);
                            break;
                    }

                    float weight = 1.0f / (distance + 1e-8f);
                    weights.push_back(weight);
                    total_weight += weight;
                }

                // Normalize weights
                for (float& w : weights) {
                    w /= total_weight;
                }

                // Calculate embedding as weighted average of neighbor embeddings
                for (int d = 0; d < model->n_components; d++) {
                    float coord = 0.0f;
                    for (size_t k = 0; k < neighbors.size(); k++) {
                        int neighbor_idx = static_cast<int>(neighbors[k].second);
                        size_t embed_idx = static_cast<size_t>(neighbor_idx) * static_cast<size_t>(model->n_components) + static_cast<size_t>(d);
                        coord += model->embedding[embed_idx] * weights[k];
                    }
                    size_t out_idx = static_cast<size_t>(i) * static_cast<size_t>(model->n_components) + static_cast<size_t>(d);
                    new_embedding[out_idx] = coord;
                }

                // STEP 2: Search in EMBEDDING space for AI inference and safety analysis
                // This follows UMAP's two-step transform approach
                std::vector<int> embedding_neighbors;
                std::vector<float> embedding_distances;

                // Only perform embedding space search if index exists AND detailed info requested
                if (model->embedding_space_index && (nn_indices || confidence_score || outlier_level || percentile_rank || z_score)) {
                    // Search for neighbors in EMBEDDING space using the newly computed embedding coordinates
                    // Convert double to float for HNSW interface
                    std::vector<float> new_embedding_point_float(model->n_components);
                    for (int d = 0; d < model->n_components; d++) {
                        size_t idx = static_cast<size_t>(i) * static_cast<size_t>(model->n_components) + static_cast<size_t>(d);
                        new_embedding_point_float[d] = static_cast<float>(new_embedding[idx]);
                    }
                    const float* new_embedding_point = new_embedding_point_float.data();

                    // Temporarily boost ef for better search quality (store and restore)
                    // Note: HNSW doesn't expose current ef, so we'll use the model parameter
                    size_t boosted_ef = static_cast<size_t>(model->n_neighbors * 32);
                    boosted_ef = std::min(boosted_ef, static_cast<size_t>(400));
                    model->embedding_space_index->setEf(boosted_ef);

                    auto embedding_search_result = model->embedding_space_index->searchKnn(new_embedding_point, model->n_neighbors);
                    model->embedding_space_index->setEf(model->hnsw_ef_search);  // Restore original ef

                    // Extract embedding space neighbors and distances for AI inference
                    while (!embedding_search_result.empty()) {
                        auto pair = embedding_search_result.top();
                        embedding_search_result.pop();

                        int neighbor_idx = static_cast<int>(pair.second);
                        float distance = std::sqrt(std::max(0.0f, pair.first)); // L2Space returns squared distance

                        embedding_neighbors.push_back(neighbor_idx);
                        embedding_distances.push_back(distance);
                    }
                }

                // Store EMBEDDING SPACE neighbor information (this is what AI needs)
                if (nn_indices && nn_distances) {
                    for (size_t k = 0; k < embedding_neighbors.size() && k < static_cast<size_t>(model->n_neighbors); k++) {
                        size_t out_idx = static_cast<size_t>(i) * static_cast<size_t>(model->n_neighbors) + k;
                        nn_indices[out_idx] = embedding_neighbors[k];
                        nn_distances[out_idx] = embedding_distances[k];
                    }
                }

                // Calculate AI inference safety metrics using EMBEDDING space distances
                if (!embedding_distances.empty() && (confidence_score || outlier_level || percentile_rank || z_score)) {
                    float min_dist = *std::min_element(embedding_distances.begin(), embedding_distances.end());
                    float mean_dist = std::accumulate(embedding_distances.begin(), embedding_distances.end(), 0.0f) / embedding_distances.size();

                    // Confidence score (higher = more similar to training data)
                    if (confidence_score) {
                        confidence_score[i] = std::exp(-min_dist / (model->p95_embedding_distance + 1e-8f));
                    }

                    // Outlier level based on embedding space distance percentiles
                    if (outlier_level) {
                        if (min_dist <= model->p95_embedding_distance) {
                            outlier_level[i] = PACMAP_OUTLIER_NORMAL;
                        } else if (min_dist <= model->p99_embedding_distance) {
                            outlier_level[i] = PACMAP_OUTLIER_UNUSUAL;
                        } else if (min_dist <= model->mild_embedding_outlier_threshold) {
                            outlier_level[i] = PACMAP_OUTLIER_MILD;
                        } else if (min_dist <= model->extreme_embedding_outlier_threshold) {
                            outlier_level[i] = PACMAP_OUTLIER_EXTREME;
                        } else {
                            outlier_level[i] = PACMAP_OUTLIER_NOMANSLAND;
                        }
                    }

                    // Percentile rank based on embedding space distances
                    if (percentile_rank) {
                        float rank = (min_dist / model->p95_embedding_distance) * 95.0f;
                        percentile_rank[i] = std::min(99.0f, rank);
                    }

                    // Z-score based on embedding space statistics
                    if (z_score && model->std_embedding_distance > 1e-8f) {
                        z_score[i] = (min_dist - model->mean_embedding_distance) / model->std_embedding_distance;
                    }
                }
            }

            // Copy results to output
            size_t expected_size = static_cast<size_t>(n_new_obs) * static_cast<size_t>(model->n_components);
            for (size_t i = 0; i < expected_size; i++) {
                embedding[i] = new_embedding[i];
            }

            return PACMAP_SUCCESS;
        }
        catch (const std::exception& e) {
            send_error_to_callback(("Transform error: " + std::string(e.what())).c_str());
            return PACMAP_ERROR_MEMORY;
        }
    }

    // Basic PACMAP transform without detailed analysis (internal implementation)
    int internal_pacmap_transform(
        PacMapModel* model,
        double* new_data,
        int n_new_obs,
        int n_dim,
        double* embedding
    ) {
        return transform_utils::internal_pacmap_transform_detailed(model, new_data, n_new_obs, n_dim, embedding,
            nullptr, nullptr, nullptr, nullptr, nullptr, nullptr);
    }
}