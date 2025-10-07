#include "pacmap_persistence.h"
#include "pacmap_quantization.h"
#include "pacmap_utils.h"
#include "pacmap_crc32.h"
#include "space_l2.h"
#include "lz4.h"
#include <algorithm>
#include <vector>
#include <random>
#include <chrono>

namespace persistence_utils {

    // Endian-safe serialization utilities
    namespace endian_utils {

        // Check if system is little-endian
        bool is_little_endian() {
            uint16_t test = 0x1234;
            return *reinterpret_cast<uint8_t*>(&test) == 0x34;
        }

        // Convert to little-endian (portable format)
        template<typename T>
        void to_little_endian(T& value) {
            if (!is_little_endian()) {
                // Swap bytes for big-endian systems
                uint8_t* bytes = reinterpret_cast<uint8_t*>(&value);
                for (size_t i = 0; i < sizeof(T) / 2; ++i) {
                    std::swap(bytes[i], bytes[sizeof(T) - 1 - i]);
                }
            }
        }

        // Convert from little-endian to native
        template<typename T>
        void from_little_endian(T& value) {
            to_little_endian(value); // Same operation - byte swap if needed
        }

        // Safe write with endian conversion
        template<typename T>
        void write_value(std::ostream& stream, const T& value) {
            T temp = value;
            to_little_endian(temp);
            stream.write(reinterpret_cast<const char*>(&temp), sizeof(T));
        }

        // Safe read with endian conversion
        template<typename T>
        bool read_value(std::istream& stream, T& value) {
            if (!stream.read(reinterpret_cast<char*>(&value), sizeof(T))) {
                return false;
            }
            from_little_endian(value);
            return true;
        }
    }


    void save_hnsw_to_stream_compressed(std::ostream& output, hnswlib::HierarchicalNSW<float>* hnsw_index) {
                std::string temp_filename = hnsw_utils::hnsw_stream_utils::generate_unique_temp_filename("hnsw_compressed");

        try {
                        // Save HNSW index to temporary file
            hnsw_index->saveIndex(temp_filename);

            // Read the temporary file
            std::ifstream temp_file(temp_filename, std::ios::binary | std::ios::ate);
            if (!temp_file.is_open()) {
                throw std::runtime_error("Failed to open temporary HNSW file for compression");
            }

            std::streamsize file_size = temp_file.tellg();
                        temp_file.seekg(0, std::ios::beg);

            std::vector<char> uncompressed_data(file_size);
            if (!temp_file.read(uncompressed_data.data(), file_size)) {
                throw std::runtime_error("Failed to read HNSW temporary file");
            }
            temp_file.close();

            // Compress with LZ4
            int max_compressed_size = LZ4_compressBound(static_cast<int>(file_size));
            std::vector<char> compressed_data(max_compressed_size);

            int compressed_size = LZ4_compress_default(
                uncompressed_data.data(), compressed_data.data(),
                static_cast<int>(file_size), max_compressed_size);

            if (compressed_size <= 0) {
                throw std::runtime_error("LZ4 compression failed for HNSW data");
            }

            // Write sizes and compressed data
            uint32_t original_size = static_cast<uint32_t>(file_size);
            uint32_t comp_size = static_cast<uint32_t>(compressed_size);

                        endian_utils::write_value(output, original_size);
            endian_utils::write_value(output, comp_size);
            output.write(compressed_data.data(), compressed_size);

            // Clean up
            temp_utils::safe_remove_file(temp_filename);
        }
        catch (...) {
            temp_utils::safe_remove_file(temp_filename);
            throw;
        }
    }

    void load_hnsw_from_stream_compressed(std::istream& input, hnswlib::HierarchicalNSW<float>* hnsw_index,
        hnswlib::SpaceInterface<float>* space) {
                std::string temp_filename;

        try {
            // Read LZ4 compression headers with validation (endian-safe)
            uint32_t original_size, compressed_size;
            if (!endian_utils::read_value(input, original_size) ||
                !endian_utils::read_value(input, compressed_size)) {
                throw std::runtime_error("Failed to read LZ4 compression headers");
            }


            // Enhanced security validation for LZ4 decompression
            const uint32_t MAX_DECOMPRESSED_SIZE = 100 * 1024 * 1024; // 100MB limit
            const uint32_t MAX_COMPRESSED_SIZE = 80 * 1024 * 1024;    // 80MB limit

            if (original_size > MAX_DECOMPRESSED_SIZE) {
                throw std::runtime_error("LZ4 decompression: HNSW original size too large (potential attack)");
            }
            if (compressed_size > MAX_COMPRESSED_SIZE) {
                throw std::runtime_error("LZ4 decompression: HNSW compressed size too large (potential attack)");
            }
            if (original_size == 0 || compressed_size == 0) {
                throw std::runtime_error("LZ4 decompression: Invalid zero size for HNSW data");
            }

            // Read compressed data with validation
            std::vector<char> compressed_data(compressed_size);
            input.read(compressed_data.data(), compressed_size);

            if (!input.good() || input.gcount() != static_cast<std::streamsize>(compressed_size)) {
                throw std::runtime_error("LZ4 decompression: Failed to read HNSW compressed data");
            }

            // Decompress with LZ4 (bounds-checked)
            std::vector<char> decompressed_data(original_size);
            int decompressed_size = LZ4_decompress_safe(
                compressed_data.data(), decompressed_data.data(),
                static_cast<int>(compressed_size), static_cast<int>(original_size));

            if (decompressed_size <= 0) {
                throw std::runtime_error("LZ4 decompression failed: Malformed HNSW compressed data");
            }
            if (decompressed_size != static_cast<int>(original_size)) {
                throw std::runtime_error("LZ4 decompression failed: HNSW size mismatch");
            }

            // Write to temporary file and load
            temp_filename = hnsw_utils::hnsw_stream_utils::generate_unique_temp_filename("hnsw_decomp");
            std::ofstream temp_file(temp_filename, std::ios::binary);
            if (!temp_file.is_open()) {
                throw std::runtime_error("Failed to create temporary file for HNSW decompression");
            }

            temp_file.write(decompressed_data.data(), original_size);
            temp_file.close();

            // Load from temporary file with correct parameters
            hnsw_index->loadIndex(temp_filename, space, hnsw_index->getCurrentElementCount());

            // Clean up
            temp_utils::safe_remove_file(temp_filename);
        }
        catch (...) {
            if (!temp_filename.empty()) {
                temp_utils::safe_remove_file(temp_filename);
            }
            throw;
        }
    }

    int save_model(PacMapModel* model, const char* filename) {
        if (!model || !model->is_fitted || !filename) {
            return PACMAP_ERROR_INVALID_PARAMS;
        }


        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return PACMAP_ERROR_FILE_IO;
        }

        try {
            // Magic number for file format validation (4 bytes)
            const uint32_t magic = 0x50414D50; // "PAMP" in ASCII
            endian_utils::write_value(file, magic);

            // File format version (4 bytes)
            const uint32_t format_version = 1;
            endian_utils::write_value(file, format_version);

            // Library version (16 bytes) - keep as string for compatibility
            file.write(PACMAP_WRAPPER_VERSION_STRING, 16);

            // Basic model parameters (endian-safe)
            endian_utils::write_value(file, model->n_samples);
            endian_utils::write_value(file, model->n_features);
            endian_utils::write_value(file, model->n_components);
            endian_utils::write_value(file, model->n_neighbors);
            endian_utils::write_value(file, model->mn_ratio);
            endian_utils::write_value(file, model->fp_ratio);
            endian_utils::write_value(file, static_cast<int>(model->metric));
            endian_utils::write_value(file, model->learning_rate);

            // KNN mode flag (critical for persistence)
            int force_exact_knn_int = model->force_exact_knn ? 1 : 0;
            endian_utils::write_value(file, force_exact_knn_int);

            // Random seed (for reproducibility)
            endian_utils::write_value(file, model->random_seed);

            // HNSW parameters (endian-safe)
            endian_utils::write_value(file, model->hnsw_m);
            endian_utils::write_value(file, model->hnsw_ef_construction);
            endian_utils::write_value(file, model->hnsw_ef_search);

            // Neighbor statistics (endian-safe)
            endian_utils::write_value(file, model->mean_original_distance);
            endian_utils::write_value(file, model->std_original_distance);
            endian_utils::write_value(file, model->min_original_distance);
            endian_utils::write_value(file, model->p95_original_distance);
            endian_utils::write_value(file, model->p99_original_distance);
            endian_utils::write_value(file, model->mild_original_outlier_threshold);
            endian_utils::write_value(file, model->extreme_original_outlier_threshold);
            endian_utils::write_value(file, model->median_original_distance);
            endian_utils::write_value(file, model->exact_match_threshold);
            endian_utils::write_value(file, model->hnsw_recall_percentage);

            // Normalization parameters (endian-safe)
            bool has_normalization = !model->feature_means.empty() && !model->feature_stds.empty();
            endian_utils::write_value(file, has_normalization);
            if (has_normalization) {
                // Write feature means (endian-safe)
                for (int i = 0; i < model->n_features; i++) {
                    endian_utils::write_value(file, model->feature_means[i]);
                }
                // Write feature standard deviations (endian-safe)
                for (int i = 0; i < model->n_features; i++) {
                    endian_utils::write_value(file, model->feature_stds[i]);
                }
            }

            // Phase iterators (PACMAP-specific)
            endian_utils::write_value(file, model->phase1_iters);
            endian_utils::write_value(file, model->phase2_iters);
            endian_utils::write_value(file, model->phase3_iters);

            // Embedding space statistics (critical for transform outlier detection and confidence scoring)
            endian_utils::write_value(file, model->min_embedding_distance);
            endian_utils::write_value(file, model->p95_embedding_distance);
            endian_utils::write_value(file, model->p99_embedding_distance);
            endian_utils::write_value(file, model->mild_embedding_outlier_threshold);
            endian_utils::write_value(file, model->extreme_embedding_outlier_threshold);
            endian_utils::write_value(file, model->mean_embedding_distance);
            endian_utils::write_value(file, model->std_embedding_distance);

            // Embedding storage
            size_t embedding_size = model->embedding.size();
            endian_utils::write_value(file, embedding_size);
            bool save_embedding = true;
            endian_utils::write_value(file, save_embedding);

            if (save_embedding && embedding_size > 0) {
                // Compress embedding data with LZ4
                size_t uncompressed_bytes = embedding_size * sizeof(float);
                int max_compressed_size = LZ4_compressBound(static_cast<int>(uncompressed_bytes));
                std::vector<char> compressed_data(max_compressed_size);

                int compressed_bytes = LZ4_compress_default(
                    reinterpret_cast<const char*>(model->embedding.data()),
                    compressed_data.data(),
                    static_cast<int>(uncompressed_bytes),
                    max_compressed_size);

                if (compressed_bytes > 0) {
                    // Write: uncompressed_bytes, compressed_bytes, compressed_data (endian-safe)
                    uint32_t uncompressed_size = static_cast<uint32_t>(uncompressed_bytes);
                    uint32_t comp_size = static_cast<uint32_t>(compressed_bytes);
                    endian_utils::write_value(file, uncompressed_size);
                    endian_utils::write_value(file, comp_size);
                    file.write(compressed_data.data(), compressed_bytes);
                } else {
                    // Compression failed - fall back to uncompressed (endian-safe)
                    uint32_t uncompressed_size = static_cast<uint32_t>(uncompressed_bytes);
                    uint32_t comp_size = 0; // 0 = uncompressed
                    endian_utils::write_value(file, uncompressed_size);
                    endian_utils::write_value(file, comp_size);
                    // Write raw float data (endian conversion needed for floats)
                    for (size_t i = 0; i < embedding_size; i++) {
                        endian_utils::write_value(file, model->embedding[i]);
                    }
                }
            }

            // k-NN data - Only save when NOT using quantization (quantization uses PQ codes instead)
            bool needs_knn = !model->use_quantization;
            endian_utils::write_value(file, needs_knn);

            if (needs_knn) {
                size_t indices_size = model->nn_indices.size();
                size_t distances_size = model->nn_distances.size();
                size_t weights_size = model->nn_weights.size();

                endian_utils::write_value(file, indices_size);
                endian_utils::write_value(file, distances_size);
                endian_utils::write_value(file, weights_size);

                // Write k-NN indices (endian-safe)
                for (size_t i = 0; i < indices_size; i++) {
                    endian_utils::write_value(file, model->nn_indices[i]);
                }
                // Write k-NN distances (endian-safe)
                for (size_t i = 0; i < distances_size; i++) {
                    endian_utils::write_value(file, model->nn_distances[i]);
                }
                // Write k-NN weights (endian-safe)
                for (size_t i = 0; i < weights_size; i++) {
                    endian_utils::write_value(file, model->nn_weights[i]);
                }
            }

            // Quantization data (PQ - Product Quantization) (endian-safe)
            endian_utils::write_value(file, model->use_quantization);
            endian_utils::write_value(file, model->pq_m);

            if (model->use_quantization) {
                // Save PQ codes
                size_t pq_codes_size = model->pq_codes.size();
                endian_utils::write_value(file, pq_codes_size);
                // PQ codes are uint8_t - no endian conversion needed
                if (pq_codes_size > 0) {
                    file.write(reinterpret_cast<const char*>(model->pq_codes.data()), pq_codes_size * sizeof(uint8_t));
                }

                // Save PQ centroids (endian-safe)
                size_t pq_centroids_size = model->pq_centroids.size();
                endian_utils::write_value(file, pq_centroids_size);
                for (size_t i = 0; i < pq_centroids_size; i++) {
                    endian_utils::write_value(file, model->pq_centroids[i]);
                }
            }

            // Save DUAL HNSW indices directly to stream with CRC32 validation
            // Conditional logic based on force_exact_knn flag:
            // - HNSW mode (force_exact_knn=false): Save HNSW original index, don't save training data
            // - KNN direct mode (force_exact_knn=true): Don't save HNSW index, save training data instead
            bool save_original_index = !model->force_exact_knn && model->original_space_index != nullptr && !model->use_quantization;
            bool save_embedding_index = model->embedding_space_index != nullptr && !model->always_save_embedding_data;
            bool save_training_data = model->force_exact_knn && !model->training_data.empty();

            endian_utils::write_value(file, save_original_index);
            endian_utils::write_value(file, save_embedding_index);
            endian_utils::write_value(file, save_training_data);

            // Save original space HNSW index
            if (save_original_index) {
                try {
                    // Save original space HNSW index data with simple temp file approach
                    hnsw_utils::save_hnsw_to_stream_compressed(file, model->original_space_index.get());
                }
                catch (const std::exception&) {
                    // Original HNSW save failed
                    size_t zero_size = 0;
                    endian_utils::write_value(file, zero_size);
                    send_warning_to_callback("Original space HNSW index save failed - transforms may be slower");
                }
            }
            else {
                // No original space HNSW index - write zero size
                size_t zero_size = 0;
                endian_utils::write_value(file, zero_size);
            }

            // Save embedding space HNSW index (always saved, never quantized)
            if (save_embedding_index) {
                try {
                    // Save embedding space HNSW index data with simple temp file approach
                    hnsw_utils::save_hnsw_to_stream_compressed(file, model->embedding_space_index.get());
                }
                catch (const std::exception& e) {
                    // Embedding HNSW save failed - this is critical for AI inference
                    size_t zero_size = 0;
                    endian_utils::write_value(file, zero_size);
                    send_warning_to_callback("Embedding space HNSW index save failed - AI inference will not work");
                }
            }
            else {
                // No embedding space HNSW index - write zero size
                size_t zero_size = 0;
                endian_utils::write_value(file, zero_size);
            }

            // Save training data for KNN direct mode (force_exact_knn=true)
            if (save_training_data) {
                try {
                    size_t training_data_size = model->training_data.size();
                    endian_utils::write_value(file, training_data_size);

                    // Compress training data with LZ4
                    size_t uncompressed_bytes = training_data_size * sizeof(float);
                    int max_compressed_size = LZ4_compressBound(static_cast<int>(uncompressed_bytes));
                    std::vector<char> compressed_data(max_compressed_size);

                    int compressed_bytes = LZ4_compress_default(
                        reinterpret_cast<const char*>(model->training_data.data()),
                        compressed_data.data(),
                        static_cast<int>(uncompressed_bytes),
                        max_compressed_size);

                    if (compressed_bytes > 0) {
                        // Write compressed data
                        uint32_t uncompressed_size = static_cast<uint32_t>(uncompressed_bytes);
                        uint32_t comp_size = static_cast<uint32_t>(compressed_bytes);
                        endian_utils::write_value(file, uncompressed_size);
                        endian_utils::write_value(file, comp_size);
                        file.write(compressed_data.data(), compressed_bytes);
                    } else {
                        // Compression failed - write uncompressed
                        uint32_t uncompressed_size = static_cast<uint32_t>(uncompressed_bytes);
                        uint32_t comp_size = 0; // 0 = uncompressed
                        endian_utils::write_value(file, uncompressed_size);
                        endian_utils::write_value(file, comp_size);
                        // Write raw float data with endian conversion
                        for (size_t i = 0; i < training_data_size; i++) {
                            endian_utils::write_value(file, model->training_data[i]);
                        }
                    }
                }
                catch (const std::exception& e) {
                    // Training data save failed
                    size_t zero_size = 0;
                    endian_utils::write_value(file, zero_size);
                    send_warning_to_callback("Training data save failed - KNN direct mode transform may not work");
                }
            }
            else {
                // No training data - write zero size
                size_t zero_size = 0;
                endian_utils::write_value(file, zero_size);
            }

            // Calculate and save model CRC32 values for integrity validation
            // CRC32 for original space (training data or HNSW index)
            if (model->force_exact_knn && !model->training_data.empty()) {
                model->original_space_crc = crc_utils::compute_vector_crc32(model->training_data);
            } else if (model->original_space_index) {
                // For HNSW index, we'll compute a simple CRC based on key parameters
                // Note: Full HNSW index CRC would require serialization, which is complex
                uint32_t index_params_crc = crc_utils::compute_crc32(&model->n_samples, sizeof(int));
                index_params_crc = crc_utils::combine_crc32(index_params_crc, crc_utils::compute_crc32(&model->n_features, sizeof(int)), sizeof(int));
                index_params_crc = crc_utils::combine_crc32(index_params_crc, crc_utils::compute_crc32(&model->hnsw_m, sizeof(int)), sizeof(int));
                model->original_space_crc = index_params_crc;
            } else {
                model->original_space_crc = 0;
            }

            // CRC32 for embedding space (embedding coordinates and HNSW index)
            if (!model->embedding.empty()) {
                model->embedding_space_crc = crc_utils::compute_vector_crc32(model->embedding);
            } else {
                model->embedding_space_crc = 0;
            }

            // CRC32 for model version and parameters
            uint32_t params_crc = crc_utils::compute_crc32(&model->n_samples, sizeof(int));
            params_crc = crc_utils::combine_crc32(params_crc, crc_utils::compute_crc32(&model->n_features, sizeof(int)), sizeof(int));
            params_crc = crc_utils::combine_crc32(params_crc, crc_utils::compute_crc32(&model->n_components, sizeof(int)), sizeof(int));
            params_crc = crc_utils::combine_crc32(params_crc, crc_utils::compute_crc32(&model->n_neighbors, sizeof(int)), sizeof(int));
            params_crc = crc_utils::combine_crc32(params_crc, crc_utils::compute_crc32(&model->mn_ratio, sizeof(float)), sizeof(float));
            params_crc = crc_utils::combine_crc32(params_crc, crc_utils::compute_crc32(&model->fp_ratio, sizeof(float)), sizeof(float));
            model->model_version_crc = params_crc;

            endian_utils::write_value(file, model->original_space_crc);
            endian_utils::write_value(file, model->embedding_space_crc);
            endian_utils::write_value(file, model->model_version_crc);

            file.close();
            return PACMAP_SUCCESS;
        }
        catch (const std::exception& e) {
            send_error_to_callback(e.what());
            return PACMAP_ERROR_FILE_IO;
        }
    }

    PacMapModel* load_model(const char* filename) {
        if (!filename) {
            return nullptr;
        }

        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            return nullptr;
        }

        PacMapModel* model = nullptr;

        try {
            model = new PacMapModel();
            if (!model) {
                return nullptr;
            }

            // Read and verify magic number and version (endian-safe)
            uint32_t magic;
            if (!endian_utils::read_value(file, magic)) {
                throw std::runtime_error("Failed to read magic number");
            }
            if (magic != 0x50414D50) { // "PAMP" in ASCII
                throw std::runtime_error("Invalid file format: magic number mismatch");
            }

            uint32_t format_version;
            if (!endian_utils::read_value(file, format_version)) {
                throw std::runtime_error("Failed to read format version");
            }
            if (format_version != 1) {
                send_warning_to_callback("Unsupported format version - attempting to load anyway");
            }

            // Read library version (16 bytes) - keep as string for compatibility
            char version[17] = { 0 };
            file.read(version, 16);
            if (strcmp(version, PACMAP_WRAPPER_VERSION_STRING) != 0) {
                send_warning_to_callback("Model version mismatch - attempting to load anyway");
            }

            // Read basic model parameters (endian-safe)
            if (!endian_utils::read_value(file, model->n_samples) ||
                !endian_utils::read_value(file, model->n_features) ||
                !endian_utils::read_value(file, model->n_components) ||
                !endian_utils::read_value(file, model->n_neighbors) ||
                !endian_utils::read_value(file, model->mn_ratio) ||
                !endian_utils::read_value(file, model->fp_ratio)) {
                throw std::runtime_error("Failed to read basic model parameters");
            }

            int metric_value;
            if (!endian_utils::read_value(file, metric_value)) {
                throw std::runtime_error("Failed to read metric parameter");
            }
            model->metric = static_cast<PacMapMetric>(metric_value);

            if (!endian_utils::read_value(file, model->learning_rate)) {
                throw std::runtime_error("Failed to read learning rate");
            }

            // Read force_exact_knn flag (endian-safe)
            int force_exact_knn_int;
            if (!endian_utils::read_value(file, force_exact_knn_int)) {
                throw std::runtime_error("Failed to read force_exact_knn parameter");
            }
            model->force_exact_knn = (force_exact_knn_int != 0);

            // Read random seed (for reproducibility)
            if (!endian_utils::read_value(file, model->random_seed)) {
                throw std::runtime_error("Failed to read random seed");
            }

            // Read HNSW and training data save flags (endian-safe)
            uint32_t save_hnsw_original, save_hnsw_embedding, save_training_data, has_training_data;
            if (!endian_utils::read_value(file, save_hnsw_original) ||
                !endian_utils::read_value(file, save_hnsw_embedding) ||
                !endian_utils::read_value(file, save_training_data) ||
                !endian_utils::read_value(file, has_training_data)) {
                throw std::runtime_error("Failed to read save flags");
            }

            // Read HNSW parameters (endian-safe)
            if (!endian_utils::read_value(file, model->hnsw_m) ||
                !endian_utils::read_value(file, model->hnsw_ef_construction) ||
                !endian_utils::read_value(file, model->hnsw_ef_search)) {
                throw std::runtime_error("Failed to read HNSW parameters");
            }

            // Read neighbor statistics (endian-safe)
            if (!endian_utils::read_value(file, model->mean_original_distance) ||
                !endian_utils::read_value(file, model->std_original_distance) ||
                !endian_utils::read_value(file, model->min_original_distance) ||
                !endian_utils::read_value(file, model->p95_original_distance) ||
                !endian_utils::read_value(file, model->p99_original_distance) ||
                !endian_utils::read_value(file, model->mild_original_outlier_threshold) ||
                !endian_utils::read_value(file, model->extreme_original_outlier_threshold) ||
                !endian_utils::read_value(file, model->median_original_distance) ||
                !endian_utils::read_value(file, model->exact_match_threshold) ||
                !endian_utils::read_value(file, model->hnsw_recall_percentage)) {
                throw std::runtime_error("Failed to read neighbor statistics");
            }

            // Read normalization parameters (endian-safe)
            bool has_normalization;
            if (!endian_utils::read_value(file, has_normalization)) {
                throw std::runtime_error("Failed to read normalization flag");
            }
            if (has_normalization) {
                model->feature_means.resize(model->n_features);
                model->feature_stds.resize(model->n_features);

                // Read feature means (endian-safe)
                for (int i = 0; i < model->n_features; i++) {
                    if (!endian_utils::read_value(file, model->feature_means[i])) {
                        throw std::runtime_error("Failed to read feature means");
                    }
                }

                // Read feature standard deviations (endian-safe)
                for (int i = 0; i < model->n_features; i++) {
                    if (!endian_utils::read_value(file, model->feature_stds[i])) {
                        throw std::runtime_error("Failed to read feature standard deviations");
                    }
                }
                model->use_normalization = true;
            }

            // Read phase iterators (PACMAP-specific)
            if (!endian_utils::read_value(file, model->phase1_iters) ||
                !endian_utils::read_value(file, model->phase2_iters) ||
                !endian_utils::read_value(file, model->phase3_iters)) {
                throw std::runtime_error("Failed to read phase iterators");
            }

            // Read embedding space statistics (critical for transform outlier detection and confidence scoring)
            if (!endian_utils::read_value(file, model->min_embedding_distance) ||
                !endian_utils::read_value(file, model->p95_embedding_distance) ||
                !endian_utils::read_value(file, model->p99_embedding_distance) ||
                !endian_utils::read_value(file, model->mild_embedding_outlier_threshold) ||
                !endian_utils::read_value(file, model->extreme_embedding_outlier_threshold) ||
                !endian_utils::read_value(file, model->mean_embedding_distance) ||
                !endian_utils::read_value(file, model->std_embedding_distance)) {
                throw std::runtime_error("Failed to read embedding space statistics");
            }

            // Read embedding data - handle optimized format for quantized models (endian-safe)
            size_t embedding_size;
            if (!endian_utils::read_value(file, embedding_size)) {
                throw std::runtime_error("Failed to read embedding size");
            }

            // Read optimization flag for quantized models
            bool save_embedding;
            if (!endian_utils::read_value(file, save_embedding)) {
                throw std::runtime_error("Failed to read embedding save flag");
            }

            model->embedding.resize(embedding_size);

            if (save_embedding && embedding_size > 0) {
                // Full embedding data saved (non-quantized or legacy models)
                uint32_t uncompressed_size, compressed_size;
                if (!endian_utils::read_value(file, uncompressed_size) ||
                    !endian_utils::read_value(file, compressed_size)) {
                    throw std::runtime_error("Failed to read embedding compression headers");
                }

                if (compressed_size > 0) {
                    // Enhanced security validation for embedding LZ4 decompression
                    const uint32_t MAX_EMBEDDING_DECOMPRESSED = 200 * 1024 * 1024; // 200MB limit for embeddings
                    const uint32_t MAX_EMBEDDING_COMPRESSED = 150 * 1024 * 1024;   // 150MB limit for compressed

                    if (uncompressed_size > MAX_EMBEDDING_DECOMPRESSED) {
                        throw std::runtime_error("LZ4 decompression: Embedding uncompressed size too large (potential attack)");
                    }
                    if (compressed_size > MAX_EMBEDDING_COMPRESSED) {
                        throw std::runtime_error("LZ4 decompression: Embedding compressed size too large (potential attack)");
                    }
                    if (uncompressed_size == 0) {
                        throw std::runtime_error("LZ4 decompression: Invalid zero uncompressed size for embedding");
                    }

                    // Read compressed embedding data with validation
                    std::vector<char> compressed_data(compressed_size);
                    file.read(compressed_data.data(), compressed_size);

                    if (!file.good() || file.gcount() != static_cast<std::streamsize>(compressed_size)) {
                        throw std::runtime_error("LZ4 decompression: Failed to read embedding compressed data");
                    }

                    // Decompress with LZ4 (bounds-checked)
                    int decompressed_size = LZ4_decompress_safe(
                        compressed_data.data(),
                        reinterpret_cast<char*>(model->embedding.data()),
                        static_cast<int>(compressed_size),
                        static_cast<int>(uncompressed_size));

                    if (decompressed_size <= 0) {
                        throw std::runtime_error("LZ4 decompression failed: Malformed embedding compressed data");
                    }
                    if (decompressed_size != static_cast<int>(uncompressed_size)) {
                        throw std::runtime_error("LZ4 decompression failed: Embedding size mismatch");
                    }
                } else {
                    // Uncompressed embedding data (fallback or old format) - read endian-safe floats
                    for (size_t i = 0; i < embedding_size; i++) {
                        if (!endian_utils::read_value(file, model->embedding[i])) {
                            throw std::runtime_error("Failed to read embedding data");
                        }
                    }
                }
            }

            // Read k-NN data (endian-safe)
            bool needs_knn;
            if (!endian_utils::read_value(file, needs_knn)) {
                throw std::runtime_error("Failed to read k-NN flag");
            }

            if (needs_knn) {
                // k-NN data available for exact reproducibility
                size_t indices_size, distances_size, weights_size;
                if (!endian_utils::read_value(file, indices_size) ||
                    !endian_utils::read_value(file, distances_size) ||
                    !endian_utils::read_value(file, weights_size)) {
                    throw std::runtime_error("Failed to read k-NN sizes");
                }

                // Read k-NN indices (endian-safe)
                if (indices_size > 0) {
                    model->nn_indices.resize(indices_size);
                    for (size_t i = 0; i < indices_size; i++) {
                        if (!endian_utils::read_value(file, model->nn_indices[i])) {
                            throw std::runtime_error("Failed to read k-NN indices");
                        }
                    }
                }
                // Read k-NN distances (endian-safe)
                if (distances_size > 0) {
                    model->nn_distances.resize(distances_size);
                    for (size_t i = 0; i < distances_size; i++) {
                        if (!endian_utils::read_value(file, model->nn_distances[i])) {
                            throw std::runtime_error("Failed to read k-NN distances");
                        }
                    }
                }
                // Read k-NN weights (endian-safe)
                if (weights_size > 0) {
                    model->nn_weights.resize(weights_size);
                    for (size_t i = 0; i < weights_size; i++) {
                        if (!endian_utils::read_value(file, model->nn_weights[i])) {
                            throw std::runtime_error("Failed to read k-NN weights");
                        }
                    }
                }
            }

            // Read quantization data (PQ - Product Quantization) (endian-safe)
            if (!endian_utils::read_value(file, model->use_quantization) ||
                !endian_utils::read_value(file, model->pq_m)) {
                throw std::runtime_error("Failed to read quantization parameters");
            }

            if (model->use_quantization) {
                // Read PQ codes
                size_t pq_codes_size;
                if (!endian_utils::read_value(file, pq_codes_size)) {
                    throw std::runtime_error("Failed to read PQ codes size");
                }
                if (pq_codes_size > 0) {
                    model->pq_codes.resize(pq_codes_size);
                    // PQ codes are uint8_t - no endian conversion needed
                    file.read(reinterpret_cast<char*>(model->pq_codes.data()), pq_codes_size * sizeof(uint8_t));
                }

                // Read PQ centroids (endian-safe)
                size_t pq_centroids_size;
                if (!endian_utils::read_value(file, pq_centroids_size)) {
                    throw std::runtime_error("Failed to read PQ centroids size");
                }
                if (pq_centroids_size > 0) {
                    model->pq_centroids.resize(pq_centroids_size);
                    for (size_t i = 0; i < pq_centroids_size; i++) {
                        if (!endian_utils::read_value(file, model->pq_centroids[i])) {
                            throw std::runtime_error("Failed to read PQ centroids");
                        }
                    }
                }
            }

            // Read training data for KNN direct mode (if available)
            if (has_training_data && save_training_data) {
                size_t training_data_size;
                if (!endian_utils::read_value(file, training_data_size)) {
                    throw std::runtime_error("Failed to read training data size");
                }

                if (training_data_size > 0) {
                    model->training_data.resize(training_data_size);
                    for (size_t i = 0; i < training_data_size; i++) {
                        if (!endian_utils::read_value(file, model->training_data[i])) {
                            throw std::runtime_error("Failed to read training data");
                        }
                    }
                }
            }

            // Read DUAL HNSW indices based on save flags (endian-safe)
            bool has_original_index = (save_hnsw_original != 0);
            bool has_embedding_index = (save_hnsw_embedding != 0);

            // Load original space HNSW index
            if (has_original_index) {
                try {
                    // Create original space HNSW index with saved parameters
                    // Note: Simplified approach - we'll reconstruct the space if needed
                    model->original_space_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                        nullptr, // Space will be set during loading
                        model->n_samples,
                        model->hnsw_m,
                        model->hnsw_ef_construction
                    );

                    model->original_space_index->setEf(model->hnsw_ef_search);

                    // Load original space HNSW data - PROPER LOADING FROM SAVED INDEX
                    try {
   send_warning_to_callback("Loading original space HNSW index from saved data");

                        // Load HNSW index directly from stream (massive performance improvement)
                        // Note: We need to create the appropriate space for loading
                        // For now, we'll use a simple L2 space as placeholder
                        static hnswlib::L2Space l2_space(model->n_features);
                        hnsw_utils::load_hnsw_from_stream_compressed(file, model->original_space_index.get(),
                                                                   &l2_space);

   send_warning_to_callback("Original space HNSW index loaded successfully - no reconstruction needed");
                    }
                    catch (const std::exception&) {
   send_warning_to_callback("Failed to load saved HNSW index, falling back to rebuild");

                        // Fallback: rebuild the index from quantized data if loading fails
                        if (model->use_quantization && !model->pq_codes.empty() && !model->pq_centroids.empty()) {
                            std::vector<float> reconstructed_data(model->n_samples * model->n_features);
                            int subspace_dim = model->n_features / model->pq_m;

                            for (int i = 0; i < model->n_samples; i++) {
                                std::vector<float> reconstructed_point;
                                pq_utils::reconstruct_vector(model->pq_codes, i, model->pq_m,
                                                           model->pq_centroids, subspace_dim,
                                                           reconstructed_point);

                                for (int d = 0; d < model->n_features; d++) {
                                    reconstructed_data[i * model->n_features + d] = reconstructed_point[d];
                                }
                            }

                            // Add all points to HNSW index
                            for (int i = 0; i < model->n_samples; i++) {
                                model->original_space_index->addPoint(&reconstructed_data[i * model->n_features], i);
                            }
                        }
                    }
                }
                catch (const std::exception&) {
                    model->original_space_index = nullptr;
                    
                    send_warning_to_callback("Original space HNSW index loading failed - no index available");
                }
            } else {
                // Skip the zero-size placeholder written when original HNSW was not saved
                size_t zero_size;
                if (!endian_utils::read_value(file, zero_size)) {
                    throw std::runtime_error("Failed to read original HNSW placeholder size");
                }

                // KNN direct mode: rebuild original space index from training data
                if (model->force_exact_knn && !model->training_data.empty()) {
                    try {
                        send_warning_to_callback("KNN direct mode: rebuilding original space HNSW index from training data");

                        // Create original space HNSW index
                        static hnswlib::L2Space original_space(model->n_features);
                        model->original_space_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                            &original_space,
                            model->n_samples,
                            model->hnsw_m,
                            model->hnsw_ef_construction
                        );
                        model->original_space_index->setEf(model->hnsw_ef_search);

                        // Add all training data points to the index
                        for (int i = 0; i < model->n_samples; i++) {
                            const float* training_point = &model->training_data[static_cast<size_t>(i) * static_cast<size_t>(model->n_features)];
                            model->original_space_index->addPoint(training_point, static_cast<size_t>(i));
                        }

                        send_warning_to_callback("Original space HNSW index rebuilt from training data - KNN direct mode ready");
                    }
                    catch (const std::exception& e) {
                        send_warning_to_callback(("Failed to rebuild original space index: " + std::string(e.what())).c_str());
                        model->original_space_index = nullptr;
                    }
                }
            }

            // Load embedding space HNSW index (critical for AI inference)
            if (has_embedding_index) {
                try {
                    // Create embedding space HNSW index with saved parameters
                    // PACMAP uses direct L2 space for embeddings
                    static hnswlib::L2Space embedding_space(model->n_components);
                    model->embedding_space_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                        &embedding_space,
                        model->n_samples,
                        model->hnsw_m,
                        model->hnsw_ef_construction
                    );

                    model->embedding_space_index->setEf(model->hnsw_ef_search);

                    // Load embedding space HNSW data - PROPER LOADING FROM SAVED INDEX
                    try {
   send_warning_to_callback("Loading embedding space HNSW index from saved data");

                        // Load HNSW index directly from stream (massive performance improvement)
                        hnsw_utils::load_hnsw_from_stream_compressed(file, model->embedding_space_index.get(),
                                                                   &embedding_space);

   send_warning_to_callback("Embedding space HNSW index loaded successfully - AI inference ready");
                    }
                    catch (const std::exception&) {
   send_warning_to_callback("Failed to load saved embedding HNSW index, rebuilding from embeddings");

                        // Fallback: rebuild the index from embedding coordinates if loading fails
                        if (!model->embedding.empty() && model->n_samples > 0) {
                            // Add embedding points to HNSW index
                            for (int i = 0; i < model->n_samples; i++) {
                                model->embedding_space_index->addPoint(&model->embedding[static_cast<size_t>(i) * static_cast<size_t>(model->n_components)], static_cast<size_t>(i));
                            }
                            
                    send_warning_to_callback("Embedding space HNSW index rebuilt from embeddings");
                        }
                    }
                }
                catch (const std::exception&) {
                    model->embedding_space_index = nullptr;
                    
                    send_warning_to_callback("CRITICAL: Embedding space HNSW index loading failed - AI inference will not work");
                }
            }
            else if (model->always_save_embedding_data) {
                // NEW: Rebuild embedding HNSW index from saved embeddings when always_save_embedding_data is true
                 send_warning_to_callback("Rebuilding embedding space HNSW index from saved embeddings (always_save_embedding_data mode)");
                try {
                    // Create embedding space HNSW index with saved parameters
                    // PACMAP uses direct L2 space for embeddings
                    static hnswlib::L2Space embedding_space_rebuild(model->n_components);
                    model->embedding_space_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                        &embedding_space_rebuild,
                        model->n_samples,
                        model->hnsw_m,
                        model->hnsw_ef_construction
                    );

                    model->embedding_space_index->setEf(model->hnsw_ef_search);

                    // Rebuild the index from embedding coordinates
                    if (!model->embedding.empty() && model->n_samples > 0) {
                        // Add embedding points to HNSW index
                        for (int i = 0; i < model->n_samples; i++) {
                            model->embedding_space_index->addPoint(&model->embedding[static_cast<size_t>(i) * static_cast<size_t>(model->n_components)], static_cast<size_t>(i));
                        }
   send_warning_to_callback("Embedding space HNSW index rebuilt successfully from saved embeddings");
                    }
                }
                catch (const std::exception&) {
                    model->embedding_space_index = nullptr;
                    
                    send_warning_to_callback("CRITICAL: Failed to rebuild embedding space HNSW index from saved embeddings");
                }
            }
            else {
                // No HNSW indices saved - rebuild from quantized data if available
                if (model->use_quantization && !model->pq_codes.empty() && !model->pq_centroids.empty()) {
                    // WARNING: Reconstructing original space HNSW from lossy quantized data

                    send_warning_to_callback("Reconstructing original space HNSW from quantized data - accuracy may be reduced");
                    try {
                        // Create original space HNSW index
                        // PACMAP uses direct space interface based on metric
                        static hnswlib::L2Space original_space_recon(model->n_features);
                        model->original_space_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                            &original_space_recon,
                            model->n_samples,
                            model->hnsw_m,
                            model->hnsw_ef_construction
                        );
                        model->original_space_index->setEf(model->hnsw_ef_search);

                        // Reconstruct quantized data from PQ codes
                        std::vector<float> reconstructed_data(model->n_samples * model->n_features);
                        int subspace_dim = model->n_features / model->pq_m;

                        for (int i = 0; i < model->n_samples; i++) {
                            std::vector<float> reconstructed_point;
                            pq_utils::reconstruct_vector(model->pq_codes, i, model->pq_m,
                                                       model->pq_centroids, subspace_dim,
                                                       reconstructed_point);

                            // Copy to reconstructed data
                            for (int d = 0; d < model->n_features; d++) {
                                reconstructed_data[i * model->n_features + d] = reconstructed_point[d];
                            }
                        }

                        // Add all points to original space HNSW index
                        for (int i = 0; i < model->n_samples; i++) {
                            model->original_space_index->addPoint(&reconstructed_data[i * model->n_features], i);
                        }
                    }
                    catch (...) {
                        model->original_space_index = nullptr;
   send_warning_to_callback("Original space HNSW reconstruction failed");
                    }
                }

                // Embedding space HNSW cannot be reconstructed from quantized data
                // It must be rebuilt from embeddings if available
                if (!model->embedding.empty()) {

                    send_warning_to_callback("Rebuilding embedding space HNSW from saved embeddings");
                    try {
                        // Create embedding space HNSW index
                        // PACMAP uses direct L2 space for embeddings
                        static hnswlib::L2Space embedding_space_final(model->n_components);
                        model->embedding_space_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                            &embedding_space_final,
                            model->n_samples,
                            model->hnsw_m,
                            model->hnsw_ef_construction
                        );
                        model->embedding_space_index->setEf(model->hnsw_ef_search);

                        // Add all embedding points to embedding space HNSW index
                        for (int i = 0; i < model->n_samples; i++) {
                            const float* embedding_point = &model->embedding[static_cast<size_t>(i) * static_cast<size_t>(model->n_components)];
                            model->embedding_space_index->addPoint(embedding_point, i);
                        }
                    }
                    catch (...) {
                        model->embedding_space_index = nullptr;
   send_warning_to_callback("CRITICAL: Embedding space HNSW reconstruction failed - AI inference will not work");
                    }
                }
            }

            // Load model CRC32 values for integrity validation
            uint32_t saved_original_crc, saved_embedding_crc, saved_model_crc;
            if (!endian_utils::read_value(file, saved_original_crc) ||
                !endian_utils::read_value(file, saved_embedding_crc) ||
                !endian_utils::read_value(file, saved_model_crc)) {
                 send_warning_to_callback("Failed to read model CRC32 values - integrity validation disabled");
            } else {
                // Validate CRC32 values
                bool crc_valid = true;
                std::string crc_errors = "";

                // Validate original space CRC
                uint32_t computed_original_crc = 0;
                if (model->force_exact_knn && !model->training_data.empty()) {
                    computed_original_crc = crc_utils::compute_vector_crc32(model->training_data);
                } else if (model->original_space_index) {
                    uint32_t index_params_crc = crc_utils::compute_crc32(&model->n_samples, sizeof(int));
                    index_params_crc = crc_utils::combine_crc32(index_params_crc, crc_utils::compute_crc32(&model->n_features, sizeof(int)), sizeof(int));
                    index_params_crc = crc_utils::combine_crc32(index_params_crc, crc_utils::compute_crc32(&model->hnsw_m, sizeof(int)), sizeof(int));
                    computed_original_crc = index_params_crc;
                }

                if (computed_original_crc != saved_original_crc) {
                    crc_valid = false;
                    crc_errors += "Original space CRC mismatch; ";
                }

                // Validate embedding space CRC
                uint32_t computed_embedding_crc = 0;
                if (!model->embedding.empty()) {
                    computed_embedding_crc = crc_utils::compute_vector_crc32(model->embedding);
                }

                if (computed_embedding_crc != saved_embedding_crc) {
                    crc_valid = false;
                    crc_errors += "Embedding space CRC mismatch; ";
                }

                // Validate model version CRC
                uint32_t computed_model_crc = crc_utils::compute_crc32(&model->n_samples, sizeof(int));
                computed_model_crc = crc_utils::combine_crc32(computed_model_crc, crc_utils::compute_crc32(&model->n_features, sizeof(int)), sizeof(int));
                computed_model_crc = crc_utils::combine_crc32(computed_model_crc, crc_utils::compute_crc32(&model->n_components, sizeof(int)), sizeof(int));
                computed_model_crc = crc_utils::combine_crc32(computed_model_crc, crc_utils::compute_crc32(&model->n_neighbors, sizeof(int)), sizeof(int));
                computed_model_crc = crc_utils::combine_crc32(computed_model_crc, crc_utils::compute_crc32(&model->mn_ratio, sizeof(float)), sizeof(float));
                computed_model_crc = crc_utils::combine_crc32(computed_model_crc, crc_utils::compute_crc32(&model->fp_ratio, sizeof(float)), sizeof(float));

                if (computed_model_crc != saved_model_crc) {
                    crc_valid = false;
                    crc_errors += "Model parameters CRC mismatch; ";
                }

                if (!crc_valid) {
                    send_warning_to_callback(("CRC validation failed: " + crc_errors + " - data may be corrupted").c_str());
                } else {
                    send_warning_to_callback("CRC validation passed - model integrity verified");
                }

                // Store the loaded CRC values in the model
                model->original_space_crc = saved_original_crc;
                model->embedding_space_crc = saved_embedding_crc;
                model->model_version_crc = saved_model_crc;
            }

            model->is_fitted = true;
            file.close();

            return model;
        }
        catch (const std::exception& e) {
            send_error_to_callback(e.what());
            if (model) {
                delete model;
            }
            return nullptr;
        }
    }
}