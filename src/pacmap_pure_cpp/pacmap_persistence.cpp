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
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <cstdarg>

namespace persistence_utils {

    // Constants for safety limits
    static constexpr uint64_t MAX_FILE_SIZE = 10ULL * 1024 * 1024 * 1024; // 10 GB
    static constexpr uint64_t MAX_VECTOR_SIZE = 100 * 1024 * 1024; // 100M elements
    static constexpr size_t CHUNK_SIZE = 1024 * 1024; // 1MB chunks for I/O

    // Logging function (controlled by PACMAP_VERBOSE environment variable)
    static void log_debug(const char* format, ...) {
#ifdef PACMAP_VERBOSE
        va_list args;
        va_start(args, format);
        vprintf(format, args);
        printf("\n");
        va_end(args);
#else
        (void)format; // Suppress unused parameter warning
#endif
    }

    namespace endian_utils {
        bool is_little_endian() {
            uint16_t test = 0x1234;
            return *reinterpret_cast<uint8_t*>(&test) == 0x34;
        }

        template<typename T>
        void to_little_endian(T& value) {
            if (!is_little_endian()) {
                uint8_t* bytes = reinterpret_cast<uint8_t*>(&value);
                for (size_t i = 0; i < sizeof(T) / 2; ++i) {
                    std::swap(bytes[i], bytes[sizeof(T) - 1 - i]);
                }
            }
        }

        template<typename T>
        void from_little_endian(T& value) {
            to_little_endian(value);
        }

        template<typename T>
        void write_value(std::ostream& stream, uint32_t& crc, const T& value, const char* field_name = nullptr) {
            if (!stream.good()) {
                throw std::runtime_error(std::string("Stream not in a good state before writing ") + (field_name ? field_name : "value"));
            }
            std::streampos pos_before = stream.tellp();
            T temp = value;
            to_little_endian(temp);
            stream.write(reinterpret_cast<const char*>(&temp), sizeof(T));
            crc = crc_utils::update_crc32(crc, reinterpret_cast<const char*>(&temp), sizeof(T));
            if (!stream.good()) {
                throw std::runtime_error(std::string("Failed to write ") + (field_name ? field_name : "value"));
            }
            std::streampos pos_after = stream.tellp();
            if (field_name) {
                log_debug("[SAVE] Wrote %s: value=%llu, pos_before=%zd, pos_after=%zd, bytes_written=%zd",
                    field_name, static_cast<uint64_t>(value), static_cast<std::streamoff>(pos_before),
                    static_cast<std::streamoff>(pos_after), static_cast<std::streamoff>(pos_after - pos_before));
            }
        }

        template<typename T>
        bool read_value(std::istream& stream, T& value, const char* field_name = nullptr) {
            if (!stream.good()) {
                if (field_name) {
                    throw std::runtime_error(std::string("Stream not in a good state before reading ") + field_name);
                }
                return false;
            }
            std::streampos pos_before = stream.tellg();
            if (!stream.read(reinterpret_cast<char*>(&value), sizeof(T))) {
                if (field_name) {
                    throw std::runtime_error(std::string("Failed to read ") + field_name);
                }
                return false;
            }
            from_little_endian(value);
            std::streampos pos_after = stream.tellg();
            if (field_name) {
                log_debug("[LOAD] Read %s: value=%llu, pos_before=%zd, pos_after=%zd, bytes_read=%zd",
                    field_name, static_cast<uint64_t>(value), static_cast<std::streamoff>(pos_before),
                    static_cast<std::streamoff>(pos_after), static_cast<std::streamoff>(pos_after - pos_before));
            }
            return true;
        }
    }

    void dump_file_contents(const char* filename, size_t bytes_to_dump = 128) {
#ifdef PACMAP_VERBOSE
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            log_debug("[DEBUG ERROR] Cannot open file %s for dumping", filename);
            return;
        }
        std::vector<char> buffer(std::min(bytes_to_dump, MAX_FILE_SIZE));
        file.read(buffer.data(), buffer.size());
        std::streamsize bytes_read = file.gcount();
        log_debug("[FILE DUMP] First %lld bytes of %s:", bytes_read, filename);
        for (size_t i = 0; i < static_cast<size_t>(bytes_read); ++i) {
            printf("%02x ", static_cast<unsigned char>(buffer[i]));
            if ((i + 1) % 16 == 0) printf("\n");
        }
        printf("\n");
        file.close();
#else
        (void)filename;      // Suppress unused parameter warning
        (void)bytes_to_dump; // Suppress unused parameter warning
#endif
    }

    // Helper functions for modularity
    static void save_header(std::ostream& stream, uint32_t& crc, uint32_t& header_crc, uint32_t& header_size) {
        log_debug("[SAVE HEADER] Starting header save");
        std::streampos start_pos = stream.tellp();
        log_debug("[SAVE HEADER] Start position: %zd", static_cast<std::streamoff>(start_pos));

        const uint32_t magic = 0x50414D50; // "PAMP"
        endian_utils::write_value(stream, crc, magic, "magic");
        log_debug("[SAVE HEADER] After magic, position: %zd", static_cast<std::streamoff>(stream.tellp()));

        const uint32_t format_version = 1;
        endian_utils::write_value(stream, crc, format_version, "format_version");
        log_debug("[SAVE HEADER] After format_version, position: %zd", static_cast<std::streamoff>(stream.tellp()));

        std::stringstream header_buffer;
        header_buffer.write(reinterpret_cast<const char*>(&magic), sizeof(uint32_t));
        header_buffer.write(reinterpret_cast<const char*>(&format_version), sizeof(uint32_t));

        constexpr size_t MAX_VERSION_LEN = 15;
        constexpr size_t version_len = sizeof(PACMAP_WRAPPER_VERSION_STRING) - 1;
        static_assert(version_len <= MAX_VERSION_LEN, "Version string too long");
        char version_buffer[16] = { 0 };
        std::memcpy(version_buffer, PACMAP_WRAPPER_VERSION_STRING, version_len);
        stream.write(version_buffer, 16);
        crc = crc_utils::update_crc32(crc, version_buffer, 16);
        header_buffer.write(version_buffer, 16);
        log_debug("[SAVE HEADER] Wrote library_version: '%s' (len=%zu), position: %zd", PACMAP_WRAPPER_VERSION_STRING, version_len, static_cast<std::streamoff>(stream.tellp()));

        header_size = static_cast<uint32_t>(stream.tellp()) + sizeof(uint32_t);
        endian_utils::write_value(stream, crc, header_size, "header_size");
        header_buffer.write(reinterpret_cast<const char*>(&header_size), sizeof(uint32_t));
        log_debug("[SAVE HEADER] After header_size=%u, position: %zd", header_size, static_cast<std::streamoff>(stream.tellp()));

        header_crc = crc_utils::compute_crc32(header_buffer.str().data(), header_buffer.str().length());
        endian_utils::write_value(stream, crc, header_crc, "header_crc");
        log_debug("[SAVE HEADER] Computed header CRC32: %u from %zu bytes, final position: %zd", header_crc, header_buffer.str().length(), static_cast<std::streamoff>(stream.tellp()));
        log_debug("[SAVE HEADER] Header complete. Total header size: %u bytes", header_size);
    }

    static void save_parameters(std::ostream& stream, uint32_t& crc, const PacMapModel* model) {
        endian_utils::write_value(stream, crc, model->n_samples, "n_samples");
        endian_utils::write_value(stream, crc, model->n_features, "n_features");
        endian_utils::write_value(stream, crc, model->n_components, "n_components");
        endian_utils::write_value(stream, crc, model->n_neighbors, "n_neighbors");
        endian_utils::write_value(stream, crc, model->mn_ratio, "mn_ratio");
        endian_utils::write_value(stream, crc, model->fp_ratio, "fp_ratio");
        endian_utils::write_value(stream, crc, static_cast<int>(model->metric), "metric");
        endian_utils::write_value(stream, crc, model->learning_rate, "learning_rate");
        int force_exact_knn_int = model->force_exact_knn ? 1 : 0;
        endian_utils::write_value(stream, crc, force_exact_knn_int, "force_exact_knn");
        endian_utils::write_value(stream, crc, model->random_seed, "random_seed");
        endian_utils::write_value(stream, crc, model->hnsw_m, "hnsw_m");
        endian_utils::write_value(stream, crc, model->hnsw_ef_construction, "hnsw_ef_construction");
        endian_utils::write_value(stream, crc, model->hnsw_ef_search, "hnsw_ef_search");

        // Neighbor statistics
        endian_utils::write_value(stream, crc, model->mean_original_distance, "mean_original_distance");
        endian_utils::write_value(stream, crc, model->std_original_distance, "std_original_distance");
        endian_utils::write_value(stream, crc, model->min_original_distance, "min_original_distance");
        endian_utils::write_value(stream, crc, model->p95_original_distance, "p95_original_distance");
        endian_utils::write_value(stream, crc, model->p99_original_distance, "p99_original_distance");
        endian_utils::write_value(stream, crc, model->mild_original_outlier_threshold, "mild_original_outlier_threshold");
        endian_utils::write_value(stream, crc, model->extreme_original_outlier_threshold, "extreme_original_outlier_threshold");
        endian_utils::write_value(stream, crc, model->median_original_distance, "median_original_distance");
        endian_utils::write_value(stream, crc, model->exact_match_threshold, "exact_match_threshold");
        endian_utils::write_value(stream, crc, model->hnsw_recall_percentage, "hnsw_recall_percentage");

        // Save additional algorithm parameters
        endian_utils::write_value(stream, crc, model->initialization_std_dev, "initialization_std_dev");
        endian_utils::write_value(stream, crc, model->always_save_embedding_data, "always_save_embedding_data");
    }

    static void save_vector(std::ostream& stream, uint32_t& crc, const std::vector<float>& vec, const char* name, bool compress) {
        uint64_t vec_size = static_cast<uint64_t>(vec.size());
        endian_utils::write_value(stream, crc, vec_size, (std::string(name) + "_size").c_str());
        if (vec_size > MAX_VECTOR_SIZE) {
            throw std::runtime_error(std::string("Vector ") + name + " size exceeds limit: " + std::to_string(vec_size));
        }
        if (vec_size == 0) return;

        if (compress) {
            size_t uncompressed_bytes = vec_size * sizeof(float);
            int max_compressed_size = LZ4_compressBound(static_cast<int>(uncompressed_bytes));
            std::vector<char> compressed_data(max_compressed_size);
            int compressed_bytes = LZ4_compress_default(
                reinterpret_cast<const char*>(vec.data()),
                compressed_data.data(),
                static_cast<int>(uncompressed_bytes),
                max_compressed_size);
            if (compressed_bytes > 0) {
                uint64_t uncompressed_size = static_cast<uint64_t>(uncompressed_bytes);
                uint64_t comp_size = static_cast<uint64_t>(compressed_bytes);
                log_debug("[SAVE] Writing %s: uncompressed=%llu, compressed=%llu", name, uncompressed_size, comp_size);
                endian_utils::write_value(stream, crc, uncompressed_size, (std::string(name) + "_uncompressed_size").c_str());
                endian_utils::write_value(stream, crc, comp_size, (std::string(name) + "_compressed_size").c_str());
                stream.write(compressed_data.data(), compressed_bytes);
                crc = crc_utils::update_crc32(crc, compressed_data.data(), compressed_bytes);
            }
            else {
                uint64_t uncompressed_size = static_cast<uint64_t>(uncompressed_bytes);
                uint64_t comp_size = 0;
                log_debug("[SAVE] %s compression failed, writing uncompressed", name);
                endian_utils::write_value(stream, crc, uncompressed_size, (std::string(name) + "_uncompressed_size").c_str());
                endian_utils::write_value(stream, crc, comp_size, (std::string(name) + "_compressed_size").c_str());
                for (size_t i = 0; i < vec_size; i++) {
                    endian_utils::write_value(stream, crc, vec[i], (std::string(name) + "[" + std::to_string(i) + "]").c_str());
                }
            }
        }
        else {
            for (size_t i = 0; i < vec_size; i++) {
                endian_utils::write_value(stream, crc, vec[i], (std::string(name) + "[" + std::to_string(i) + "]").c_str());
            }
        }
    }

    static void save_vector(std::ostream& stream, uint32_t& crc, const std::vector<double>& vec, const char* name, bool compress) {
        uint64_t vec_size = static_cast<uint64_t>(vec.size());
        endian_utils::write_value(stream, crc, vec_size, (std::string(name) + "_size").c_str());
        if (vec_size > MAX_VECTOR_SIZE) {
            throw std::runtime_error(std::string("Vector ") + name + " size exceeds limit: " + std::to_string(vec_size));
        }
        if (vec_size == 0) return;

        if (compress) {
            size_t uncompressed_bytes = vec_size * sizeof(double);
            int max_compressed_size = LZ4_compressBound(static_cast<int>(uncompressed_bytes));
            std::vector<char> compressed_data(max_compressed_size);
            int compressed_bytes = LZ4_compress_default(
                reinterpret_cast<const char*>(vec.data()),
                compressed_data.data(),
                static_cast<int>(uncompressed_bytes),
                max_compressed_size);
            if (compressed_bytes > 0) {
                uint64_t uncompressed_size = static_cast<uint64_t>(uncompressed_bytes);
                uint64_t comp_size = static_cast<uint64_t>(compressed_bytes);
                log_debug("[SAVE] Writing %s (double): uncompressed=%llu, compressed=%llu", name, uncompressed_size, comp_size);
                endian_utils::write_value(stream, crc, uncompressed_size, (std::string(name) + "_uncompressed_size").c_str());
                endian_utils::write_value(stream, crc, comp_size, (std::string(name) + "_compressed_size").c_str());
                stream.write(compressed_data.data(), compressed_bytes);
                crc = crc_utils::update_crc32(crc, compressed_data.data(), compressed_bytes);
            }
            else {
                uint64_t uncompressed_size = static_cast<uint64_t>(uncompressed_bytes);
                uint64_t comp_size = 0;
                log_debug("[SAVE] %s compression failed, writing uncompressed", name);
                endian_utils::write_value(stream, crc, uncompressed_size, (std::string(name) + "_uncompressed_size").c_str());
                endian_utils::write_value(stream, crc, comp_size, (std::string(name) + "_compressed_size").c_str());
                for (size_t i = 0; i < vec_size; i++) {
                    endian_utils::write_value(stream, crc, vec[i], (std::string(name) + "[" + std::to_string(i) + "]").c_str());
                }
            }
        }
        else {
            for (size_t i = 0; i < vec_size; i++) {
                endian_utils::write_value(stream, crc, vec[i], (std::string(name) + "[" + std::to_string(i) + "]").c_str());
            }
        }
    }

    static void save_vector(std::ostream& stream, uint32_t& crc, const std::vector<int>& vec, const char* name, bool compress) {
        uint64_t vec_size = static_cast<uint64_t>(vec.size());
        endian_utils::write_value(stream, crc, vec_size, (std::string(name) + "_size").c_str());
        if (vec_size > MAX_VECTOR_SIZE) {
            throw std::runtime_error(std::string("Vector ") + name + " size exceeds limit: " + std::to_string(vec_size));
        }
        if (vec_size == 0) return;

        // Integer vectors are not compressed (they don't compress well with LZ4)
        for (size_t i = 0; i < vec_size; i++) {
            endian_utils::write_value(stream, crc, vec[i], (std::string(name) + "[" + std::to_string(i) + "]").c_str());
        }
    }

    static void save_knn_data(std::ostream& stream, uint32_t& crc, const PacMapModel* model) {
        bool needs_knn = !model->use_quantization;
        endian_utils::write_value(stream, crc, needs_knn, "needs_knn");
        if (needs_knn) {
            save_vector(stream, crc, model->nn_indices, "nn_indices", false);
            save_vector(stream, crc, model->nn_distances, "nn_distances", false);
            save_vector(stream, crc, model->nn_weights, "nn_weights", false);
        }
    }

    static void save_quantization_data(std::ostream& stream, uint32_t& crc, const PacMapModel* model) {
        endian_utils::write_value(stream, crc, model->use_quantization, "use_quantization");
        endian_utils::write_value(stream, crc, model->pq_m, "pq_m");
        if (model->use_quantization) {
            uint64_t pq_codes_size = static_cast<uint64_t>(model->pq_codes.size());
            endian_utils::write_value(stream, crc, pq_codes_size, "pq_codes_size");
            if (pq_codes_size > MAX_VECTOR_SIZE) {
                throw std::runtime_error("PQ codes size exceeds limit: " + std::to_string(pq_codes_size));
            }
            if (pq_codes_size > 0) {
                for (size_t i = 0; i < pq_codes_size; i += CHUNK_SIZE) {
                    size_t chunk = std::min(CHUNK_SIZE, pq_codes_size - i);
                    stream.write(reinterpret_cast<const char*>(model->pq_codes.data() + i), chunk * sizeof(uint8_t));
                    crc = crc_utils::update_crc32(crc, reinterpret_cast<const char*>(model->pq_codes.data() + i), chunk * sizeof(uint8_t));
                }
            }
            save_vector(stream, crc, model->pq_centroids, "pq_centroids", false);
        }
    }

    static void save_indices_and_training(std::ostream& stream, uint32_t& crc, const PacMapModel* model) {
        bool save_original_index = !model->force_exact_knn && model->original_space_index != nullptr && !model->use_quantization;
        bool save_embedding_index = model->embedding_space_index != nullptr && !model->always_save_embedding_data;
        bool save_training_data = model->force_exact_knn && !model->training_data.empty();

        endian_utils::write_value(stream, crc, save_original_index, "save_original_index");
        endian_utils::write_value(stream, crc, save_embedding_index, "save_embedding_index");
        endian_utils::write_value(stream, crc, save_training_data, "save_training_data");

        if (save_original_index) {
            try {
                // Use stringstream to capture HNSW data for CRC computation
                std::stringstream hnsw_buffer;
                hnsw_utils::save_hnsw_to_stream_compressed(hnsw_buffer, model->original_space_index.get());
                std::string hnsw_data_str = hnsw_buffer.str();
                stream.write(hnsw_data_str.data(), hnsw_data_str.size());
                crc = crc_utils::update_crc32(crc, hnsw_data_str.data(), hnsw_data_str.size());
            }
            catch (const std::exception& e) {
                uint64_t zero_size = 0;
                endian_utils::write_value(stream, crc, zero_size, "original_index_zero_size");
                log_debug("[SAVE WARNING] Failed to save original HNSW index: %s", e.what());
            }
        }
        else {
            uint64_t zero_size = 0;
            endian_utils::write_value(stream, crc, zero_size, "original_index_zero_size");
        }

        if (save_embedding_index) {
            try {
                // Use stringstream to capture HNSW data for CRC computation
                std::stringstream hnsw_buffer;
                hnsw_utils::save_hnsw_to_stream_compressed(hnsw_buffer, model->embedding_space_index.get());
                std::string hnsw_data_str = hnsw_buffer.str();
                stream.write(hnsw_data_str.data(), hnsw_data_str.size());
                crc = crc_utils::update_crc32(crc, hnsw_data_str.data(), hnsw_data_str.size());
            }
            catch (const std::exception& e) {
                uint64_t zero_size = 0;
                endian_utils::write_value(stream, crc, zero_size, "embedding_index_zero_size");
                log_debug("[SAVE WARNING] Failed to save embedding HNSW index: %s", e.what());
            }
        }
        else {
            uint64_t zero_size = 0;
            endian_utils::write_value(stream, crc, zero_size, "embedding_index_zero_size");
        }

        if (save_training_data) {
            save_vector(stream, crc, model->training_data, "training_data", true);
        }
        else {
            uint64_t zero_size = 0;
            endian_utils::write_value(stream, crc, zero_size, "training_data_zero_size");
        }
    }

    static void save_crc_values(std::ostream& stream, uint32_t& crc, const PacMapModel* model) {
        uint32_t original_space_crc = 0;
        if (model->force_exact_knn && !model->training_data.empty()) {
            original_space_crc = crc_utils::compute_vector_crc32(model->training_data);
        }
        else if (model->original_space_index) {
            uint32_t index_params_crc = crc_utils::compute_crc32(&model->n_samples, sizeof(int));
            index_params_crc = crc_utils::combine_crc32(index_params_crc, crc_utils::compute_crc32(&model->n_features, sizeof(int)), sizeof(int));
            index_params_crc = crc_utils::combine_crc32(index_params_crc, crc_utils::compute_crc32(&model->hnsw_m, sizeof(int)), sizeof(int));
            original_space_crc = index_params_crc;
        }
        endian_utils::write_value(stream, crc, original_space_crc, "original_space_crc");

        uint32_t embedding_space_crc = model->embedding.empty() ? 0 : crc_utils::compute_vector_crc32(model->embedding);
        endian_utils::write_value(stream, crc, embedding_space_crc, "embedding_space_crc");

        uint32_t params_crc = crc_utils::compute_crc32(&model->n_samples, sizeof(int));
        params_crc = crc_utils::combine_crc32(params_crc, crc_utils::compute_crc32(&model->n_features, sizeof(int)), sizeof(int));
        params_crc = crc_utils::combine_crc32(params_crc, crc_utils::compute_crc32(&model->n_components, sizeof(int)), sizeof(int));
        params_crc = crc_utils::combine_crc32(params_crc, crc_utils::compute_crc32(&model->n_neighbors, sizeof(int)), sizeof(int));
        params_crc = crc_utils::combine_crc32(params_crc, crc_utils::compute_crc32(&model->mn_ratio, sizeof(float)), sizeof(float));
        params_crc = crc_utils::combine_crc32(params_crc, crc_utils::compute_crc32(&model->fp_ratio, sizeof(float)), sizeof(float));
        endian_utils::write_value(stream, crc, params_crc, "model_version_crc");
    }

    int save_model(PacMapModel* model, const char* filename) {
        if (!model || !model->is_fitted || !filename) {
            return PACMAP_ERROR_INVALID_PARAMS;
        }
        if (model->n_samples > MAX_VECTOR_SIZE || model->n_features > MAX_VECTOR_SIZE || model->n_components > MAX_VECTOR_SIZE) {
            send_error_to_callback(("Model parameters exceed size limits: n_samples=" + std::to_string(model->n_samples) +
                ", n_features=" + std::to_string(model->n_features) + ", n_components=" + std::to_string(model->n_components)).c_str());
            return PACMAP_ERROR_INVALID_PARAMS;
        }

        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            send_error_to_callback(("Failed to open file for writing: " + std::string(filename)).c_str());
            return PACMAP_ERROR_FILE_IO;
        }

        try {
            uint32_t crc = crc_utils::init_crc32();
            uint32_t header_crc = 0, header_size = 0;

            save_header(file, crc, header_crc, header_size);
            save_parameters(file, crc, model);

            // Save normalization data
            bool has_normalization = !model->feature_means.empty() && !model->feature_stds.empty();
            endian_utils::write_value(file, crc, has_normalization, "has_normalization");
            if (has_normalization) {
                // Save min-max normalization parameters (v2.8.4+)
                endian_utils::write_value(file, crc, model->xmin, "xmin");
                endian_utils::write_value(file, crc, model->xmax, "xmax");

                for (int i = 0; i < model->n_features; i++) {
                    endian_utils::write_value(file, crc, model->feature_means[i], ("feature_means[" + std::to_string(i) + "]").c_str());
                }
                for (int i = 0; i < model->n_features; i++) {
                    endian_utils::write_value(file, crc, model->feature_stds[i], ("feature_stds[" + std::to_string(i) + "]").c_str());
                }
            }

            // Save phase iterators
            endian_utils::write_value(file, crc, model->phase1_iters, "phase1_iters");
            endian_utils::write_value(file, crc, model->phase2_iters, "phase2_iters");
            endian_utils::write_value(file, crc, model->phase3_iters, "phase3_iters");

            // Save embedding statistics
            endian_utils::write_value(file, crc, model->min_embedding_distance, "min_embedding_distance");
            endian_utils::write_value(file, crc, model->p95_embedding_distance, "p95_embedding_distance");
            endian_utils::write_value(file, crc, model->p99_embedding_distance, "p99_embedding_distance");
            endian_utils::write_value(file, crc, model->mild_embedding_outlier_threshold, "mild_embedding_outlier_threshold");
            endian_utils::write_value(file, crc, model->extreme_embedding_outlier_threshold, "extreme_embedding_outlier_threshold");
            endian_utils::write_value(file, crc, model->mean_embedding_distance, "mean_embedding_distance");
            endian_utils::write_value(file, crc, model->std_embedding_distance, "std_embedding_distance");

            // Save algorithm state
            endian_utils::write_value(file, crc, model->total_triplets, "total_triplets");
            endian_utils::write_value(file, crc, model->neighbor_triplets, "neighbor_triplets");
            endian_utils::write_value(file, crc, model->mid_near_triplets, "mid_near_triplets");
            endian_utils::write_value(file, crc, model->far_triplets, "far_triplets");

            // Save distance percentiles
            endian_utils::write_value(file, crc, model->p25_distance, "p25_distance");
            endian_utils::write_value(file, crc, model->p75_distance, "p75_distance");

            // Save Adam optimizer state
            save_vector(file, crc, model->adam_m, "adam_m", true);
            save_vector(file, crc, model->adam_v, "adam_v", true);
            endian_utils::write_value(file, crc, model->adam_beta1, "adam_beta1");
            endian_utils::write_value(file, crc, model->adam_beta2, "adam_beta2");
            endian_utils::write_value(file, crc, model->adam_eps, "adam_eps");

            // Save embedding save flag before saving embedding data
            bool save_embedding = !model->embedding.empty();
            endian_utils::write_value(file, crc, save_embedding, "save_embedding");
            if (save_embedding) {
                save_vector(file, crc, model->embedding, "embedding", true);
            }

            save_knn_data(file, crc, model);
            save_quantization_data(file, crc, model);
            save_indices_and_training(file, crc, model);
            save_crc_values(file, crc, model);

            endian_utils::write_value(file, crc, crc, "full_file_crc");
            log_debug("[SAVE] Computed full-file CRC32: %u", crc);

            file.close();
            log_debug("[SAVE] File saved: %s", filename);
            dump_file_contents(filename);
            return PACMAP_SUCCESS;
        }
        catch (const std::exception& e) {
            send_error_to_callback(("Failed to save model to file: " + std::string(filename) + " - " + e.what()).c_str());
            file.close();
            return PACMAP_ERROR_FILE_IO;
        }
    }

    static bool load_header(std::istream& stream, const char* filename, uint32_t& header_size) {
        log_debug("[LOAD HEADER] Starting header load from file: %s", filename);
        std::streampos initial_pos = stream.tellg();
        log_debug("[LOAD HEADER] Initial position: %zd", static_cast<std::streamoff>(initial_pos));

        uint32_t magic;
        if (!endian_utils::read_value(stream, magic, "magic")) {
            throw std::runtime_error("Failed to read magic number from file: " + std::string(filename));
        }
        log_debug("[LOAD HEADER] Read magic: 0x%08X, position: %zd", magic, static_cast<std::streamoff>(stream.tellg()));
        if (magic != 0x50414D50) {
            throw std::runtime_error("Invalid file format: magic number mismatch in file: " + std::string(filename));
        }

        uint32_t format_version;
        if (!endian_utils::read_value(stream, format_version, "format_version")) {
            throw std::runtime_error("Failed to read format version from file: " + std::string(filename));
        }
        log_debug("[LOAD HEADER] Read format_version: %u, position: %zd", format_version, static_cast<std::streamoff>(stream.tellg()));
        if (format_version != 1) {
            throw std::runtime_error("Unsupported format version: " + std::to_string(format_version) +
                " in file: " + std::string(filename));
        }

        char version[17] = { 0 };
        stream.read(version, 16);
        if (!stream.good()) {
            throw std::runtime_error("Failed to read library version from file: " + std::string(filename));
        }
        log_debug("[LOAD HEADER] Read version: '%s', position: %zd", version, static_cast<std::streamoff>(stream.tellg()));
        if (strcmp(version, PACMAP_WRAPPER_VERSION_STRING) != 0) {
            throw std::runtime_error("Library version mismatch: expected " + std::string(PACMAP_WRAPPER_VERSION_STRING) +
                ", got " + std::string(version) + " in file: " + std::string(filename));
        }

        if (!endian_utils::read_value(stream, header_size, "header_size")) {
            throw std::runtime_error("Failed to read header size from file: " + std::string(filename));
        }
        log_debug("[LOAD HEADER] Read header_size: %u, position: %zd", header_size, static_cast<std::streamoff>(stream.tellg()));

        std::streampos header_data_pos = stream.tellg();
        uint32_t saved_header_crc;
        if (!endian_utils::read_value(stream, saved_header_crc, "header_crc")) {
            throw std::runtime_error("Failed to read header CRC from file: " + std::string(filename));
        }
        log_debug("[LOAD HEADER] Read header_crc: %u, position before CRC validation: %zd", saved_header_crc, static_cast<std::streamoff>(stream.tellg()));

        // CRC validation: read entire header and validate
        std::streampos crc_validation_start = stream.tellg();
        stream.seekg(0, std::ios::beg);
        log_debug("[LOAD HEADER] Seeking to start for CRC validation: %zd", static_cast<std::streamoff>(stream.tellg()));

        std::vector<char> header_data(header_size);
        stream.read(header_data.data(), header_size);
        if (!stream.good() || stream.gcount() != static_cast<std::streamsize>(header_size)) {
            throw std::runtime_error("Failed to read header data for CRC validation");
        }
        log_debug("[LOAD HEADER] Read %zu header bytes for CRC validation, position: %zd", header_size, static_cast<std::streamoff>(stream.tellg()));

        uint32_t computed_header_crc = crc_utils::compute_crc32(header_data.data(), header_size);
        log_debug("[LOAD HEADER] CRC validation: computed=%u, saved=%u", computed_header_crc, saved_header_crc);
        if (computed_header_crc != saved_header_crc) {
            throw std::runtime_error("Header CRC mismatch: computed=" + std::to_string(computed_header_crc) +
                ", saved=" + std::to_string(saved_header_crc) + " in file: " + std::string(filename));
        }

        // CRITICAL: Restore stream position to after header_crc
        stream.seekg(header_data_pos + static_cast<std::streamoff>(sizeof(uint32_t)));
        log_debug("[LOAD HEADER] Restored position to after header_crc: %zd", static_cast<std::streamoff>(stream.tellg()));
        return true;
    }

    static void load_parameters(std::istream& stream, PacMapModel* model, const char* filename) {
        log_debug("[LOAD PARAMS] Starting parameter loading, position: %zd", static_cast<std::streamoff>(stream.tellg()));

        if (!endian_utils::read_value(stream, model->n_samples, "n_samples") ||
            !endian_utils::read_value(stream, model->n_features, "n_features") ||
            !endian_utils::read_value(stream, model->n_components, "n_components") ||
            !endian_utils::read_value(stream, model->n_neighbors, "n_neighbors") ||
            !endian_utils::read_value(stream, model->mn_ratio, "mn_ratio") ||
            !endian_utils::read_value(stream, model->fp_ratio, "fp_ratio")) {
            throw std::runtime_error("Failed to read model parameters from file: " + std::string(filename));
        }

        log_debug("[LOAD PARAMS] Read core parameters: n_samples=%d, n_features=%d, n_components=%d, n_neighbors=%d, mn_ratio=%.2f, fp_ratio=%.2f",
                model->n_samples, model->n_features, model->n_components, model->n_neighbors, model->mn_ratio, model->fp_ratio);
        log_debug("[LOAD PARAMS] Position after core params: %zd", static_cast<std::streamoff>(stream.tellg()));

        if (model->n_samples > MAX_VECTOR_SIZE || model->n_features > MAX_VECTOR_SIZE || model->n_components > MAX_VECTOR_SIZE) {
            throw std::runtime_error("Model parameters exceed size limits: n_samples=" + std::to_string(model->n_samples) +
                ", n_features=" + std::to_string(model->n_features) + ", n_components=" + std::to_string(model->n_components));
        }

        int metric_value;
        if (!endian_utils::read_value(stream, metric_value, "metric")) {
            throw std::runtime_error("Failed to read metric from file: " + std::string(filename));
        }
        model->metric = static_cast<PacMapMetric>(metric_value);

        if (!endian_utils::read_value(stream, model->learning_rate, "learning_rate")) {
            throw std::runtime_error("Failed to read additional model parameters from file: " + std::string(filename));
        }

        int force_exact_knn_int;
        if (!endian_utils::read_value(stream, force_exact_knn_int, "force_exact_knn") ||
            !endian_utils::read_value(stream, model->random_seed, "random_seed") ||
            !endian_utils::read_value(stream, model->hnsw_m, "hnsw_m") ||
            !endian_utils::read_value(stream, model->hnsw_ef_construction, "hnsw_ef_construction") ||
            !endian_utils::read_value(stream, model->hnsw_ef_search, "hnsw_ef_search")) {
            throw std::runtime_error("Failed to read additional model parameters from file: " + std::string(filename));
        }
        model->force_exact_knn = (force_exact_knn_int != 0);

        if (!endian_utils::read_value(stream, model->mean_original_distance, "mean_original_distance") ||
            !endian_utils::read_value(stream, model->std_original_distance, "std_original_distance") ||
            !endian_utils::read_value(stream, model->min_original_distance, "min_original_distance") ||
            !endian_utils::read_value(stream, model->p95_original_distance, "p95_original_distance") ||
            !endian_utils::read_value(stream, model->p99_original_distance, "p99_original_distance") ||
            !endian_utils::read_value(stream, model->mild_original_outlier_threshold, "mild_original_outlier_threshold") ||
            !endian_utils::read_value(stream, model->extreme_original_outlier_threshold, "extreme_original_outlier_threshold") ||
            !endian_utils::read_value(stream, model->median_original_distance, "median_original_distance") ||
            !endian_utils::read_value(stream, model->exact_match_threshold, "exact_match_threshold") ||
            !endian_utils::read_value(stream, model->hnsw_recall_percentage, "hnsw_recall_percentage")) {
            throw std::runtime_error("Failed to read neighbor statistics from file: " + std::string(filename));
        }

        // Load additional algorithm parameters
        if (!endian_utils::read_value(stream, model->initialization_std_dev, "initialization_std_dev") ||
            !endian_utils::read_value(stream, model->always_save_embedding_data, "always_save_embedding_data")) {
            throw std::runtime_error("Failed to read additional algorithm parameters from file: " + std::string(filename));
        }
    }

    static void load_vector(std::istream& stream, std::vector<float>& vec, const char* name, const char* filename, bool compressed) {
        uint64_t vec_size;
        if (!endian_utils::read_value(stream, vec_size, (std::string(name) + "_size").c_str())) {
            throw std::runtime_error("Failed to read " + std::string(name) + " size from file: " + std::string(filename));
        }
        if (vec_size > MAX_VECTOR_SIZE) {
            throw std::runtime_error(std::string(name) + " size exceeds limit: " + std::to_string(vec_size));
        }
        try {
            vec.resize(vec_size);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Failed to allocate " + std::string(name) + " vector: " + e.what());
        }
        if (vec_size == 0) return;

        if (compressed) {
            uint64_t uncompressed_size, compressed_size;
            if (!endian_utils::read_value(stream, uncompressed_size, (std::string(name) + "_uncompressed_size").c_str()) ||
                !endian_utils::read_value(stream, compressed_size, (std::string(name) + "_compressed_size").c_str())) {
                throw std::runtime_error("Failed to read " + std::string(name) + " compression headers from file: " + std::string(filename));
            }
            if (uncompressed_size > MAX_VECTOR_SIZE * sizeof(float) || compressed_size > MAX_VECTOR_SIZE) {
                throw std::runtime_error("Invalid " + std::string(name) + " compression sizes: uncompressed=" +
                    std::to_string(uncompressed_size) + ", compressed=" + std::to_string(compressed_size));
            }
            if (compressed_size > 0) {
                std::vector<char> compressed_data(compressed_size);
                for (size_t i = 0; i < compressed_size; i += CHUNK_SIZE) {
                    size_t chunk = std::min(CHUNK_SIZE, compressed_size - i);
                    stream.read(compressed_data.data() + i, chunk);
                    if (!stream.good() || stream.gcount() != static_cast<std::streamsize>(chunk)) {
                        throw std::runtime_error("Failed to read compressed " + std::string(name) + " data from file: " + std::string(filename));
                    }
                }
                int decompressed_size = LZ4_decompress_safe(
                    compressed_data.data(),
                    reinterpret_cast<char*>(vec.data()),
                    static_cast<int>(compressed_size),
                    static_cast<int>(uncompressed_size));
                if (decompressed_size != static_cast<int>(uncompressed_size)) {
                    throw std::runtime_error("LZ4 decompression failed for " + std::string(name) + ": got " +
                        std::to_string(decompressed_size) + ", expected " + std::to_string(uncompressed_size));
                }
            }
            else {
                for (size_t i = 0; i < vec_size; i++) {
                    if (!endian_utils::read_value(stream, vec[i], (std::string(name) + "[" + std::to_string(i) + "]").c_str())) {
                        throw std::runtime_error("Failed to read " + std::string(name) + " data from file: " + std::string(filename));
                    }
                }
            }
        }
        else {
            for (size_t i = 0; i < vec_size; i++) {
                if (!endian_utils::read_value(stream, vec[i], (std::string(name) + "[" + std::to_string(i) + "]").c_str())) {
                    throw std::runtime_error("Failed to read " + std::string(name) + " data from file: " + std::string(filename));
                }
            }
        }
    }

    static void load_vector(std::istream& stream, std::vector<double>& vec, const char* name, const char* filename, bool compressed) {
        uint64_t vec_size;
        if (!endian_utils::read_value(stream, vec_size, (std::string(name) + "_size").c_str())) {
            throw std::runtime_error("Failed to read " + std::string(name) + " size from file: " + std::string(filename));
        }
        if (vec_size > MAX_VECTOR_SIZE) {
            throw std::runtime_error(std::string(name) + " size exceeds limit: " + std::to_string(vec_size));
        }
        try {
            vec.resize(vec_size);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Failed to allocate " + std::string(name) + " vector: " + e.what());
        }
        if (vec_size == 0) return;

        if (compressed) {
            uint64_t uncompressed_size, compressed_size;
            if (!endian_utils::read_value(stream, uncompressed_size, (std::string(name) + "_uncompressed_size").c_str()) ||
                !endian_utils::read_value(stream, compressed_size, (std::string(name) + "_compressed_size").c_str())) {
                throw std::runtime_error("Failed to read " + std::string(name) + " compression headers from file: " + std::string(filename));
            }
            if (uncompressed_size > MAX_VECTOR_SIZE * sizeof(double) || compressed_size > MAX_VECTOR_SIZE) {
                throw std::runtime_error("Invalid " + std::string(name) + " compression sizes: uncompressed=" +
                    std::to_string(uncompressed_size) + ", compressed=" + std::to_string(compressed_size));
            }
            if (compressed_size > 0) {
                std::vector<char> compressed_data(compressed_size);
                for (size_t i = 0; i < compressed_size; i += CHUNK_SIZE) {
                    size_t chunk = std::min(CHUNK_SIZE, compressed_size - i);
                    stream.read(compressed_data.data() + i, chunk);
                    if (!stream.good() || stream.gcount() != static_cast<std::streamsize>(chunk)) {
                        throw std::runtime_error("Failed to read compressed " + std::string(name) + " data from file: " + std::string(filename));
                    }
                }
                int decompressed_size = LZ4_decompress_safe(
                    compressed_data.data(),
                    reinterpret_cast<char*>(vec.data()),
                    static_cast<int>(compressed_size),
                    static_cast<int>(uncompressed_size));
                if (decompressed_size != static_cast<int>(uncompressed_size)) {
                    throw std::runtime_error("LZ4 decompression failed for " + std::string(name) + ": got " +
                        std::to_string(decompressed_size) + ", expected " + std::to_string(uncompressed_size));
                }
            }
            else {
                for (size_t i = 0; i < vec_size; i++) {
                    if (!endian_utils::read_value(stream, vec[i], (std::string(name) + "[" + std::to_string(i) + "]").c_str())) {
                        throw std::runtime_error("Failed to read " + std::string(name) + " data from file: " + std::string(filename));
                    }
                }
            }
        }
        else {
            for (size_t i = 0; i < vec_size; i++) {
                if (!endian_utils::read_value(stream, vec[i], (std::string(name) + "[" + std::to_string(i) + "]").c_str())) {
                    throw std::runtime_error("Failed to read " + std::string(name) + " data from file: " + std::string(filename));
                }
            }
        }
    }

    static void load_vector(std::istream& stream, std::vector<int>& vec, const char* name, const char* filename, bool compressed) {
        uint64_t vec_size;
        if (!endian_utils::read_value(stream, vec_size, (std::string(name) + "_size").c_str())) {
            throw std::runtime_error("Failed to read " + std::string(name) + " size from file: " + std::string(filename));
        }
        if (vec_size > MAX_VECTOR_SIZE) {
            throw std::runtime_error(std::string(name) + " size exceeds limit: " + std::to_string(vec_size));
        }
        try {
            vec.resize(vec_size);
        }
        catch (const std::exception& e) {
            throw std::runtime_error("Failed to allocate " + std::string(name) + " vector: " + e.what());
        }
        if (vec_size == 0) return;

        // Integer vectors are not compressed (same as in save)
        for (size_t i = 0; i < vec_size; i++) {
            if (!endian_utils::read_value(stream, vec[i], (std::string(name) + "[" + std::to_string(i) + "]").c_str())) {
                throw std::runtime_error("Failed to read " + std::string(name) + " data from file: " + std::string(filename));
            }
        }
    }

    static void load_knn_data(std::istream& stream, PacMapModel* model, const char* filename) {
        bool needs_knn;
        if (!endian_utils::read_value(stream, needs_knn, "needs_knn")) {
            throw std::runtime_error("Failed to read k-NN flag from file: " + std::string(filename));
        }
        if (needs_knn) {
            load_vector(stream, model->nn_indices, "nn_indices", filename, false);
            load_vector(stream, model->nn_distances, "nn_distances", filename, false);
            load_vector(stream, model->nn_weights, "nn_weights", filename, false);
        }
    }

    static void load_quantization_data(std::istream& stream, PacMapModel* model, const char* filename) {
        if (!endian_utils::read_value(stream, model->use_quantization, "use_quantization") ||
            !endian_utils::read_value(stream, model->pq_m, "pq_m")) {
            throw std::runtime_error("Failed to read quantization parameters from file: " + std::string(filename));
        }
        if (model->use_quantization) {
            uint64_t pq_codes_size;
            if (!endian_utils::read_value(stream, pq_codes_size, "pq_codes_size")) {
                throw std::runtime_error("Failed to read PQ codes size from file: " + std::string(filename));
            }
            if (pq_codes_size > MAX_VECTOR_SIZE) {
                throw std::runtime_error("PQ codes size exceeds limit: " + std::to_string(pq_codes_size));
            }
            if (pq_codes_size > 0) {
                try {
                    model->pq_codes.resize(pq_codes_size);
                }
                catch (const std::exception& e) {
                    std::string error_msg = "Failed to allocate PQ codes vector: " + std::string(e.what());
                    throw std::runtime_error(error_msg);
                }
                for (size_t i = 0; i < pq_codes_size; i += CHUNK_SIZE) {
                    size_t chunk = std::min(CHUNK_SIZE, pq_codes_size - i);
                    stream.read(reinterpret_cast<char*>(model->pq_codes.data() + i), chunk * sizeof(uint8_t));
                    if (!stream.good()) {
                        throw std::runtime_error("Failed to read PQ codes from file: " + std::string(filename));
                    }
                }
            }
            load_vector(stream, model->pq_centroids, "pq_centroids", filename, false);
        }
    }

    static void load_indices_and_training(std::istream& stream, PacMapModel* model, const char* filename) {
        bool save_original_index, save_embedding_index, save_training_data;
        if (!endian_utils::read_value(stream, save_original_index, "save_original_index") ||
            !endian_utils::read_value(stream, save_embedding_index, "save_embedding_index") ||
            !endian_utils::read_value(stream, save_training_data, "save_training_data")) {
            throw std::runtime_error("Failed to read save flags from file: " + std::string(filename));
        }

        if (save_original_index) {
            try {
                //  Create persistent L2Space owned by the model (fixes AccessViolationException)
                model->original_space = std::make_unique<hnswlib::L2Space>(model->n_features);

                //  Construct the original space index with persistent metric space
                model->original_space_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                    model->original_space.get(),
                    model->n_samples,
                    model->hnsw_m,
                    model->hnsw_ef_construction,
                    model->random_seed
                );
                model->original_space_index->setEf(model->hnsw_ef_search);

                //  Load the compressed index from file with persistent space
                hnsw_utils::load_hnsw_from_stream_compressed(
                    stream,
                    model->original_space_index.get(),
                    model->original_space.get()
                );

                log_debug("[LOAD] Original space HNSW index successfully loaded.");
            }
            catch (const std::exception& e) {
                model->original_space_index = nullptr;
                model->original_space = nullptr;
                send_warning_to_callback(("Failed to load original space HNSW index: " + std::string(e.what())).c_str());
            }
        }
        else {
            uint64_t zero_size;
            if (!endian_utils::read_value(stream, zero_size, "original_index_zero_size")) {
                throw std::runtime_error("Failed to read original HNSW placeholder from file: " + std::string(filename));
            }
        }

        if (save_embedding_index) {
            try {
                //  OPTION A: Create persistent L2Space owned by the model (fixes AccessViolationException)
                model->embedding_space = std::make_unique<hnswlib::L2Space>(model->n_components);

                //  Construct the embedding index with persistent metric space
                model->embedding_space_index = std::make_unique<hnswlib::HierarchicalNSW<float>>(
                    model->embedding_space.get(),
                    model->n_samples,
                    model->hnsw_m,
                    model->hnsw_ef_construction
                );

                model->embedding_space_index->setEf(model->hnsw_ef_search);

                //  Load the compressed index from file with persistent space
                hnsw_utils::load_hnsw_from_stream_compressed(
                    stream,
                    model->embedding_space_index.get(),
                    model->embedding_space.get()
                );

                log_debug("[LOAD] Embedding space HNSW index loaded successfully - AI inference ready.");
                log_debug("[LOAD] Embedding space HNSW index successfully loaded.");
            }
            catch (const std::exception& e) {
                model->embedding_space_index = nullptr;
                model->embedding_space = nullptr;
                send_warning_to_callback(("Failed to load embedding space HNSW index: " + std::string(e.what())).c_str());
            }
        }
        else {
            uint64_t zero_size;
            if (!endian_utils::read_value(stream, zero_size, "embedding_index_zero_size")) {
                throw std::runtime_error("Failed to read embedding HNSW placeholder from file: " + std::string(filename));
            }
        }

        if (save_training_data) {
            load_vector(stream, model->training_data, "training_data", filename, true);
        }
        else {
            uint64_t zero_size;
            if (!endian_utils::read_value(stream, zero_size, "training_data_zero_size")) {
                throw std::runtime_error("Failed to read training data placeholder from file: " + std::string(filename));
            }
        }
    }

    static void load_crc_values(std::istream& stream, PacMapModel* model, const char* filename) {
        uint32_t saved_original_crc, saved_embedding_crc, saved_model_crc;
        if (!endian_utils::read_value(stream, saved_original_crc, "original_space_crc") ||
            !endian_utils::read_value(stream, saved_embedding_crc, "embedding_space_crc") ||
            !endian_utils::read_value(stream, saved_model_crc, "model_version_crc")) {
            send_warning_to_callback(("Failed to read CRC values from file: " + std::string(filename)).c_str());
            return;
        }

        bool crc_valid = true;
        std::string crc_errors;

        uint32_t computed_original_crc = 0;
        if (model->force_exact_knn && !model->training_data.empty()) {
            computed_original_crc = crc_utils::compute_vector_crc32(model->training_data);
        }
        else if (model->original_space_index) {
            uint32_t index_params_crc = crc_utils::compute_crc32(&model->n_samples, sizeof(int));
            index_params_crc = crc_utils::combine_crc32(index_params_crc, crc_utils::compute_crc32(&model->n_features, sizeof(int)), sizeof(int));
            index_params_crc = crc_utils::combine_crc32(index_params_crc, crc_utils::compute_crc32(&model->hnsw_m, sizeof(int)), sizeof(int));
            computed_original_crc = index_params_crc;
        }
        if (computed_original_crc != saved_original_crc) {
            crc_valid = false;
            crc_errors += "Original space CRC mismatch; ";
        }

        uint32_t computed_embedding_crc = model->embedding.empty() ? 0 : crc_utils::compute_vector_crc32(model->embedding);
        if (computed_embedding_crc != saved_embedding_crc) {
            crc_valid = false;
            crc_errors += "Embedding space CRC mismatch; ";
        }

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
            throw std::runtime_error("CRC validation failed: " + crc_errors + " in file: " + std::string(filename));
        }
        model->original_space_crc = saved_original_crc;
        model->embedding_space_crc = saved_embedding_crc;
        model->model_version_crc = saved_model_crc;
    }

    PacMapModel* load_model(const char* filename) {
        if (!filename) {
            return nullptr;
        }

        std::ifstream file(filename, std::ios::binary | std::ios::ate);
        if (!file.is_open()) {
            send_error_to_callback(("Failed to open file for reading: " + std::string(filename)).c_str());
            return nullptr;
        }

        std::streamsize file_size = file.tellg();
        if (file_size < static_cast<std::streamsize>(sizeof(uint32_t)) || file_size > static_cast<std::streamsize>(MAX_FILE_SIZE)) {
            send_error_to_callback(("Invalid file size: " + std::to_string(file_size) + " in file: " + std::string(filename)).c_str());
            file.close();
            return nullptr;
        }

        // Verify full file CRC incrementally
        file.seekg(0, std::ios::beg);
        uint32_t crc = crc_utils::init_crc32();
        std::vector<char> buffer(CHUNK_SIZE);
        std::streamsize bytes_remaining = file_size - sizeof(uint32_t);
        while (bytes_remaining > 0) {
            std::streamsize chunk = std::min(static_cast<std::streamsize>(CHUNK_SIZE), bytes_remaining);
            file.read(buffer.data(), chunk);
            if (!file.good() || file.gcount() != chunk) {
                send_error_to_callback(("Failed to read file data for CRC validation: " + std::string(filename)).c_str());
                file.close();
                return nullptr;
            }
            crc = crc_utils::update_crc32(crc, buffer.data(), static_cast<size_t>(chunk));
            bytes_remaining -= chunk;
        }
        uint32_t saved_file_crc;
        if (!endian_utils::read_value(file, saved_file_crc, "full_file_crc")) {
            send_error_to_callback(("Failed to read full file CRC: " + std::string(filename)).c_str());
            file.close();
            return nullptr;
        }
        if (crc != saved_file_crc) {
            send_error_to_callback(("Full file CRC mismatch: computed=" + std::to_string(crc) +
                ", saved=" + std::to_string(saved_file_crc) + " in file: " + std::string(filename)).c_str());
            file.close();
            return nullptr;
        }
        log_debug("[LOAD] Full file CRC32 validated: %u", crc);

        file.seekg(0, std::ios::beg);
        PacMapModel* model = nullptr;

        try {
            dump_file_contents(filename);
            model = new PacMapModel();
            if (!model) {
                throw std::runtime_error("Failed to allocate PacMapModel");
            }

            uint32_t header_size = 0;
            load_header(file, filename, header_size);
            load_parameters(file, model, filename);

            bool has_normalization;
            if (!endian_utils::read_value(file, has_normalization, "has_normalization")) {
                throw std::runtime_error("Failed to read normalization flag from file: " + std::string(filename));
            }
            if (has_normalization) {
                // Load min-max normalization parameters (v2.8.4+)
                if (!endian_utils::read_value(file, model->xmin, "xmin") ||
                    !endian_utils::read_value(file, model->xmax, "xmax")) {
                    throw std::runtime_error("Failed to read min-max normalization parameters from file: " + std::string(filename));
                }

                try {
                    model->feature_means.resize(model->n_features);
                    model->feature_stds.resize(model->n_features);
                }
                catch (const std::exception& e) {
                    throw std::runtime_error("Failed to allocate normalization vectors: " + std::string(e.what()));
                }
                for (int i = 0; i < model->n_features; i++) {
                    if (!endian_utils::read_value(file, model->feature_means[i], ("feature_means[" + std::to_string(i) + "]").c_str())) {
                        throw std::runtime_error("Failed to read feature means from file: " + std::string(filename));
                    }
                }
                for (int i = 0; i < model->n_features; i++) {
                    if (!endian_utils::read_value(file, model->feature_stds[i], ("feature_stds[" + std::to_string(i) + "]").c_str())) {
                        throw std::runtime_error("Failed to read feature stds from file: " + std::string(filename));
                    }
                }
                model->use_normalization = true;
            }

            if (!endian_utils::read_value(file, model->phase1_iters, "phase1_iters") ||
                !endian_utils::read_value(file, model->phase2_iters, "phase2_iters") ||
                !endian_utils::read_value(file, model->phase3_iters, "phase3_iters")) {
                throw std::runtime_error("Failed to read phase iterators from file: " + std::string(filename));
            }

            if (!endian_utils::read_value(file, model->min_embedding_distance, "min_embedding_distance") ||
                !endian_utils::read_value(file, model->p95_embedding_distance, "p95_embedding_distance") ||
                !endian_utils::read_value(file, model->p99_embedding_distance, "p99_embedding_distance") ||
                !endian_utils::read_value(file, model->mild_embedding_outlier_threshold, "mild_embedding_outlier_threshold") ||
                !endian_utils::read_value(file, model->extreme_embedding_outlier_threshold, "extreme_embedding_outlier_threshold") ||
                !endian_utils::read_value(file, model->mean_embedding_distance, "mean_embedding_distance") ||
                !endian_utils::read_value(file, model->std_embedding_distance, "std_embedding_distance")) {
                throw std::runtime_error("Failed to read embedding statistics from file: " + std::string(filename));
            }

            // Load algorithm state
            if (!endian_utils::read_value(file, model->total_triplets, "total_triplets") ||
                !endian_utils::read_value(file, model->neighbor_triplets, "neighbor_triplets") ||
                !endian_utils::read_value(file, model->mid_near_triplets, "mid_near_triplets") ||
                !endian_utils::read_value(file, model->far_triplets, "far_triplets")) {
                throw std::runtime_error("Failed to read algorithm state from file: " + std::string(filename));
            }

            // Load distance percentiles
            if (!endian_utils::read_value(file, model->p25_distance, "p25_distance") ||
                !endian_utils::read_value(file, model->p75_distance, "p75_distance")) {
                throw std::runtime_error("Failed to read distance percentiles from file: " + std::string(filename));
            }

            // Load Adam optimizer state
            log_debug("[LOAD] About to load adam_m vector");
            load_vector(file, model->adam_m, "adam_m", filename, true);
            log_debug("[LOAD] Successfully loaded adam_m vector, size=%zu", model->adam_m.size());

            log_debug("[LOAD] About to load adam_v vector");
            load_vector(file, model->adam_v, "adam_v", filename, true);
            log_debug("[LOAD] Successfully loaded adam_v vector, size=%zu", model->adam_v.size());

            log_debug("[LOAD] About to read Adam optimizer parameters");
            if (!endian_utils::read_value(file, model->adam_beta1, "adam_beta1") ||
                !endian_utils::read_value(file, model->adam_beta2, "adam_beta2") ||
                !endian_utils::read_value(file, model->adam_eps, "adam_eps")) {
                throw std::runtime_error("Failed to read Adam optimizer parameters from file: " + std::string(filename));
            }
            log_debug("[LOAD] Successfully loaded Adam optimizer parameters: beta1=%.6f, beta2=%.6f, eps=%.2e",
                      model->adam_beta1, model->adam_beta2, model->adam_eps);

            bool save_embedding;
            log_debug("[LOAD] About to read save_embedding flag");
            if (!endian_utils::read_value(file, save_embedding, "save_embedding")) {
                throw std::runtime_error("Failed to read embedding save flag from file: " + std::string(filename));
            }
            log_debug("[LOAD] Read save_embedding flag: value=%d", save_embedding);

            if (save_embedding) {
                log_debug("[LOAD] About to load embedding vector");
                load_vector(file, model->embedding, "embedding", filename, true);
                log_debug("[LOAD] Successfully loaded embedding vector, size=%zu", model->embedding.size());
            }

            log_debug("[LOAD] About to load k-NN data");
            load_knn_data(file, model, filename);
            log_debug("[LOAD] Successfully loaded k-NN data");
            load_quantization_data(file, model, filename);
            load_indices_and_training(file, model, filename);
            load_crc_values(file, model, filename);

            model->is_fitted = true;
            file.close();
            log_debug("[LOAD] Model loaded successfully: %s", filename);
            return model;
        }
        catch (const std::exception& e) {
            send_error_to_callback(("Failed to load model from file: " + std::string(filename) + " - " + e.what()).c_str());
            if (model) {
                delete model;
            }
            file.close();
            return nullptr;
        }
    }
}