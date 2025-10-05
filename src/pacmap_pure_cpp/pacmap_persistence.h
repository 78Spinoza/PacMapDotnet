#pragma once

#include "pacmap_model.h"
#include <string>
#include <vector>

// Model persistence functions
extern int save_pacmap_model(const PacMapModel* model, const std::string& filename);
extern int load_pacmap_model(PacMapModel* model, const std::string& filename);

// Legacy compatibility functions
extern int save_pacmap_legacy(const PacMapModel* model, const std::string& filename);
extern int load_pacmap_legacy(PacMapModel* model, const std::string& filename);

// Binary persistence with CRC32 validation
extern int save_pacmap_binary(const PacMapModel* model, const std::string& filename);
extern int load_pacmap_binary(PacMapModel* model, const std::string& filename);

// Text-based persistence for debugging
extern int save_pacmap_text(const PacMapModel* model, const std::string& filename);
extern int load_pacmap_text(PacMapModel* model, const std::string& filename);

// Model serialization components
extern int serialize_model_header(const PacMapModel* model, std::vector<char>& buffer);
extern int deserialize_model_header(PacMapModel* model, const char* buffer, size_t buffer_size);

extern int serialize_triplets(const std::vector<Triplet>& triplets, std::vector<char>& buffer);
extern int deserialize_triplets(std::vector<Triplet>& triplets, const char* buffer, size_t buffer_size);

extern int serialize_embedding(const float* embedding, int n_samples, int n_components,
                             std::vector<char>& buffer);
extern int deserialize_embedding(float* embedding, int n_samples, int n_components,
                               const char* buffer, size_t buffer_size);

extern int serialize_hnsw_index(const hnswlib::HierarchicalNSW<float>* index,
                              std::vector<char>& buffer);
extern int deserialize_hnsw_index(std::unique_ptr<hnswlib::HierarchicalNSW<float>>& index,
                                 const char* buffer, size_t buffer_size,
                                 int n_samples, int n_features);

// Compression and quantization
extern int compress_model_data(const PacMapModel* model, std::vector<char>& compressed_buffer);
extern int decompress_model_data(PacMapModel* model, const char* compressed_buffer,
                               size_t compressed_size);

extern void quantize_embedding_16bit(const float* embedding, int n_samples, int n_components,
                                    std::vector<uint16_t>& quantized_embedding);
extern void dequantize_embedding_16bit(const std::vector<uint16_t>& quantized_embedding,
                                      float* embedding, int n_samples, int n_components);

// Validation and integrity checking
extern bool validate_model_file(const std::string& filename);
extern uint32_t compute_model_crc32(const PacMapModel* model);
extern bool verify_model_crc32(const PacMapModel* model, uint32_t expected_crc);

extern int validate_loaded_model(const PacMapModel* model);
extern bool check_model_compatibility(const PacMapModel* model, int version);

// Migration utilities
extern int migrate_model_from_v1_to_v2(PacMapModel* model);
extern int migrate_model_from_legacy(PacMapModel* model);

// Persistence diagnostics
struct PersistenceDiagnostics {
    bool file_exists = false;
    bool file_readable = false;
    size_t file_size_bytes = 0;
    uint32_t computed_crc32 = 0;
    uint32_t stored_crc32 = 0;
    bool crc_valid = false;
    int model_version = 0;
    bool is_compatible = true;
    float load_time_ms = 0.0f;
    float save_time_ms = 0.0f;
    std::vector<std::string> validation_errors;
};

extern PersistenceDiagnostics analyze_model_file(const std::string& filename);
extern PersistenceDiagnostics run_save_with_diagnostics(const PacMapModel* model,
                                                       const std::string& filename);
extern PersistenceDiagnostics run_load_with_diagnostics(PacMapModel* model,
                                                       const std::string& filename);

// Advanced persistence features
extern int save_model_with_metadata(const PacMapModel* model, const std::string& filename,
                                   const std::string& metadata);
extern int load_model_with_metadata(PacMapModel* model, const std::string& filename,
                                   std::string& metadata);

extern void create_model_backup(const PacMapModel* model, const std::string& base_filename);
extern int restore_model_from_backup(PacMapModel* model, const std::string& backup_filename);

// Batch operations
extern int save_multiple_models(const std::vector<const PacMapModel*>& models,
                               const std::vector<std::string>& filenames);
extern int load_multiple_models(const std::vector<PacMapModel*>& models,
                               const std::vector<std::string>& filenames);

// Performance optimization
extern void optimize_for_fast_loading(PacMapModel* model);
extern void optimize_for_fast_saving(const PacMapModel* model);
extern void prefetch_model_data(const std::string& filename);

// Cross-platform compatibility
extern void ensure_platform_endianess(std::vector<char>& buffer);
extern bool check_platform_compatibility(const std::string& filename);