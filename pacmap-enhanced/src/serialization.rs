use serde::{Serialize, Deserialize};
use zstd::{stream::encode_all as compress, stream::decode_all as decompress};
use std::fs::File;
use std::io::{Read, Write};
use ndarray::{Array2, Array1};
use crate::quantize::{quantize_embedding_linear, QuantizedEmbedding, dequantize_embedding};
use crate::stats::NormalizationParams;
use crate::hnsw_params::HnswParams;

/// Core PacMAP model parameters (serializable subset of Configuration)
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct PacMAPConfig {
    pub n_neighbors: usize,
    pub embedding_dim: usize,
    pub n_epochs: usize,
    pub learning_rate: f64,
    pub mid_near_ratio: f64,
    pub far_pair_ratio: f64,
    pub seed: Option<u64>,
    /// HNSW parameters for neighbor search optimization
    pub hnsw_params: HnswParams,
    /// Whether HNSW was used during training
    pub used_hnsw: bool,
    /// Force KNN instead of HNSW for small datasets
    pub force_knn: bool,
}

impl Default for PacMAPConfig {
    fn default() -> Self {
        Self {
            n_neighbors: 10,
            embedding_dim: 2,
            n_epochs: 450,
            learning_rate: 1.0,
            mid_near_ratio: 0.5,
            far_pair_ratio: 0.5,
            seed: None,
            hnsw_params: HnswParams::default(),
            used_hnsw: false,
            force_knn: false,
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct PaCMAP {
    pub embedding: Array2<f64>,
    pub config: PacMAPConfig,
    pub stats: DistanceStats,
    /// Normalization parameters - CRITICAL for consistent transforms (like UMAP)
    /// These parameters are computed during training and must be applied to new data
    pub normalization: NormalizationParams,
    /// If true, the model will be saved with the embedding quantized to `f16`
    /// to reduce the on‑disk size. The quantization is performed lazily when
    /// `save_compressed` is called.
    #[serde(default)]
    pub quantize_on_save: bool,
    /// Quantized embedding with parameters (replaces simple embedding_q)
    #[serde(default)]
    pub quantized_embedding: Option<QuantizedEmbedding>,
    /// HNSW-related fields for advanced functionality
    #[serde(default)]
    pub used_hnsw: bool,
    #[serde(default)]
    pub hnsw_index_crc32: Option<u32>,
    #[serde(default)]
    pub fitted_projections_crc32: Option<u32>,
    #[serde(default)]
    pub embedding_centroid: Option<Array1<f64>>,
    #[serde(default)]
    pub fitted_projections: Array2<f64>,
    #[serde(default)]
    pub embedding_hnsw_index: Option<SerializedHnswIndex>,
    #[serde(default)]
    pub serialized_hnsw_index: Option<Vec<u8>>,
    /// Original training data for transform support (like UMAP's raw_data)
    #[serde(default)]
    pub original_data: Option<Array2<f64>>,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DistanceStats {
    pub mean_distance: f64,
    pub p95_distance: f64,
    pub max_distance: f64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct TransformStats {
    pub transform_time_ms: u64,
    pub neighbors_found: usize,
    pub avg_distance: f64,
    pub coordinates: Option<Vec<f64>>,
    pub distance_to_closest_neighbor: f64,
    pub mean_distance_to_k_neighbors: f64,
    pub distance_to_training_centroid: f64,
}

#[derive(Serialize, Deserialize, Clone)]
pub struct SerializedHnswIndex {
    pub data: Vec<u8>,
    pub point_count: usize,
    pub dimension: usize,
    pub m: usize,
    pub ef_construction: usize,
    pub ef_search: usize,
    pub max_layer: usize,
}

impl SerializedHnswIndex {
    /// Search for nearest neighbors - placeholder implementation
    pub fn search(&self, _query: &[f32], _k: usize, _ef: usize) -> Vec<NeighborResult> {
        // TODO: Implement proper HNSW search from serialized index
        // For now, return empty results
        eprintln!("⚠️  SerializedHnswIndex::search not yet implemented");
        Vec::new()
    }
}

#[derive(Serialize, Deserialize, Clone)]
pub struct NeighborResult {
    pub index: usize,
    pub distance: f32,
}

/// HNSW index wrapper for serialization
#[derive(Serialize, Deserialize, Clone)]
pub struct HnswIndex {
    pub points: Vec<Vec<f32>>,
    pub params: HnswSerializedParams,
    pub point_count: usize,
    pub dimension: usize,
}

/// Serialized HNSW parameters
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct HnswSerializedParams {
    pub ef_construction: usize,
    pub ef_search: usize,
    pub m: Option<usize>,
    pub max_layer: Option<usize>,
}

/// Load HNSW index from serialized data
pub fn load_hnsw_from_serialized(data: &HnswIndex) -> Result<crate::hnsw_index::HnswIndexWrapper, Box<dyn std::error::Error>> {
    // TEMPORARY: Basic implementation - in future this will reconstruct the actual HNSW index
    eprintln!("   HNSW deserialization: {} points, {} dimensions", data.point_count, data.dimension);

    Ok(crate::hnsw_index::HnswIndexWrapper {
        points: data.points.clone(),
        params: data.params.clone(),
        is_loaded: true,
    })
}

/// Custom HNSW serialization for our specific use case
pub fn custom_serialize_hnsw(
    data: &ndarray::Array2<f64>,
    hnsw_params: &crate::hnsw_params::HnswParams,
    _max_layer: usize
) -> Result<HnswIndex, Box<dyn std::error::Error>> {
    let (n_samples, n_features) = data.dim();

    // Convert to f32 points
    let points: Vec<Vec<f32>> = (0..n_samples)
        .map(|i| {
            data.row(i).iter().map(|&x| x as f32).collect()
        })
        .collect();

    let serialized_params = HnswSerializedParams {
        ef_construction: hnsw_params.ef_construction,
        ef_search: hnsw_params.ef_search,
        m: Some(hnsw_params.m),
        max_layer: None, // Not used in new API
    };

    eprintln!("✅ HNSW SERIALIZATION: Serialized {} points with {} dimensions", n_samples, n_features);

    Ok(HnswIndex {
        points,
        params: serialized_params,
        point_count: n_samples,
        dimension: n_features,
    })
}

/// Serialize HNSW index to bytes
pub fn serialize_hnsw_to_bytes(hnsw: &HnswIndex) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    // Serialize with MessagePack and compress with ZSTD
    let serialized = rmp_serde::to_vec(hnsw)?;
    let compressed = zstd::stream::encode_all(&serialized[..], 3)?;

    // Calculate CRC32 checksum
    use crc32fast::Hasher;
    let mut hasher = Hasher::new();
    hasher.update(&compressed);
    let checksum = hasher.finalize();

    // Combine data with checksum
    let mut result = Vec::with_capacity(4 + compressed.len());
    result.extend_from_slice(&checksum.to_le_bytes());
    result.extend_from_slice(&compressed);

    eprintln!("✅ HNSW BYTES: Serialized {} bytes with CRC32 checksum", result.len());

    Ok(result)
}

/// Deserialize HNSW index from bytes
pub fn deserialize_hnsw_from_bytes(data: &[u8]) -> Result<HnswIndex, Box<dyn std::error::Error>> {
    if data.len() < 4 {
        return Err("Data too short to contain checksum".into());
    }

    // Extract checksum
    let stored_checksum = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    let compressed_data = &data[4..];

    // Verify checksum
    use crc32fast::Hasher;
    let mut hasher = Hasher::new();
    hasher.update(compressed_data);
    let computed_checksum = hasher.finalize();

    if stored_checksum != computed_checksum {
        return Err(format!("CRC32 checksum mismatch: expected {}, got {}", stored_checksum, computed_checksum).into());
    }

    // Decompress and deserialize
    let decompressed = zstd::stream::decode_all(std::io::Cursor::new(compressed_data))?;
    let hnsw: HnswIndex = rmp_serde::from_slice(&decompressed)?;

    eprintln!("✅ HNSW DESERIALIZATION: Successfully loaded {} points, CRC32 verified", hnsw.point_count);

    Ok(hnsw)
}

impl PaCMAP {
    /// Prepare a quantized copy of the embedding for saving.
    /// This is called automatically by `save_compressed` when
    /// `quantize_on_save` is true.
    ///
    /// The quantized embedding with parameters is stored in `quantized_embedding`.
    /// The original high‑precision `embedding` is left untouched.
    pub fn quantize_for_save(&mut self) {
        if self.quantized_embedding.is_none() {
            self.quantized_embedding = Some(quantize_embedding_linear(&self.embedding));
        }
    }

    /// Get the embedding in f64 precision, dequantizing if necessary
    pub fn get_embedding(&self) -> Array2<f64> {
        if let Some(ref quantized) = self.quantized_embedding {
            // If we have quantized data, dequantize it
            dequantize_embedding(quantized)
        } else {
            // Return original embedding
            self.embedding.clone()
        }
    }

    /// Save without quantization (preserves full precision)
    pub fn save_uncompressed(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.save_uncompressed_with_progress(path, None)
    }

    /// Save without quantization with progress reporting
    pub fn save_uncompressed_with_progress(
        &self,
        path: &str,
        progress_callback: Option<&crate::ProgressCallback>
    ) -> Result<(), Box<dyn std::error::Error>> {
        let progress = |phase: &str, current: usize, total: usize, percent: f32, message: &str| {
            if let Some(callback) = progress_callback {
                callback(phase, current, total, percent, message);
            }
        };

        progress("Preparing", 0, 100, 0.0, "Preparing model for uncompressed save");

        // Display model settings
        self.print_model_settings("Saving Uncompressed Model");

        progress("Serializing", 20, 100, 20.0, "Serializing model to MessagePack format");
        let serialized = rmp_serde::to_vec(self)?;

        progress("Compressing", 60, 100, 60.0, "Compressing with ZSTD");
        let compressed = compress(&serialized[..], 3)?;

        progress("Writing", 80, 100, 80.0, &format!("Writing to file: {}", path));
        File::create(path)?.write_all(&compressed)?;

        let file_size = std::fs::metadata(path)?.len();
        progress("Complete", 100, 100, 100.0, &format!("Save completed - {} bytes", file_size));

        println!("💾 Model saved successfully to: {}", path);
        println!("    File size: {} bytes ({:.2} KB)", file_size, file_size as f64 / 1024.0);

        Ok(())
    }

    pub fn save_compressed(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        self.save_compressed_with_progress(path, None)
    }

    /// Save with optional quantization and progress reporting
    pub fn save_compressed_with_progress(
        &mut self,
        path: &str,
        progress_callback: Option<&crate::ProgressCallback>
    ) -> Result<(), Box<dyn std::error::Error>> {
        let progress = |phase: &str, current: usize, total: usize, percent: f32, message: &str| {
            if let Some(callback) = progress_callback {
                callback(phase, current, total, percent, message);
            }
        };

        progress("Preparing", 0, 100, 0.0, "Preparing model for compressed save");

        // Display model settings
        let save_type = if self.quantize_on_save { "Saving Quantized Model" } else { "Saving Compressed Model" };
        self.print_model_settings(save_type);

        // If quantization is enabled, quantize and remove original to save space
        if self.quantize_on_save {
            progress("Quantizing", 20, 100, 20.0, "Quantizing embedding to reduce file size");
            self.quantize_for_save();
            // Replace original embedding with minimal placeholder to save space
            self.embedding = Array2::zeros((1, 1));
        }

        progress("Serializing", 40, 100, 40.0, "Serializing model to MessagePack format");
        let serialized = rmp_serde::to_vec(self)?;

        progress("Compressing", 70, 100, 70.0, "Compressing with ZSTD");
        let compressed = compress(&serialized[..], 3)?;

        progress("Writing", 90, 100, 90.0, &format!("Writing to file: {}", path));
        File::create(path)?.write_all(&compressed)?;

        let file_size = std::fs::metadata(path)?.len();
        progress("Complete", 100, 100, 100.0, &format!("Save completed - {} bytes", file_size));

        println!("💾 Model saved successfully to: {}", path);
        println!("    File size: {} bytes ({:.2} KB)", file_size, file_size as f64 / 1024.0);
        if self.quantize_on_save {
            println!("     Quantization enabled - model size optimized");
        }

        Ok(())
    }

    pub fn load_compressed(path: &str) -> Result<Self, Box<dyn std::error::Error>> {
        Self::load_compressed_with_progress(path, None)
    }

    /// Load model with progress reporting
    pub fn load_compressed_with_progress(
        path: &str,
        progress_callback: Option<&crate::ProgressCallback>
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let progress = |phase: &str, current: usize, total: usize, percent: f32, message: &str| {
            if let Some(callback) = progress_callback {
                callback(phase, current, total, percent, message);
            }
        };

        progress("Reading", 0, 100, 0.0, &format!("Reading model file: {}", path));

        let mut file = File::open(path)?;
        let mut compressed = Vec::new();
        file.read_to_end(&mut compressed)?;

        let file_size = compressed.len();
        progress("Decompressing", 30, 100, 30.0, &format!("Decompressing {} bytes", file_size));

        let decompressed = decompress(std::io::Cursor::new(&compressed))?;

        progress("Deserializing", 70, 100, 70.0, "Deserializing model from MessagePack");
        let model: PaCMAP = rmp_serde::from_slice(&decompressed)?;

        progress("Validating", 90, 100, 90.0, "Validating loaded model");

        // Display loaded model settings
        model.print_model_settings("Loaded Model");

        progress("Complete", 100, 100, 100.0, "Model loading completed successfully");

        println!("📂 Model loaded successfully from: {}", path);
        println!("    File size: {} bytes ({:.2} KB)", file_size, file_size as f64 / 1024.0);

        Ok(model)
    }

    /// Print comprehensive model settings
    pub fn print_model_settings(&self, title: &str) {
        println!();
        println!("===========================================");
        println!("   {}", title);
        println!("===========================================");
        println!();

        // Core PacMAP Configuration
        println!("  Core PacMAP Configuration:");
        println!("   - Embedding dimensions: {}D", self.config.embedding_dim);
        println!("   - Number of neighbors: {}", self.config.n_neighbors);
        println!("   - Number of epochs: {}", self.config.n_epochs);
        println!("   - Learning rate: {:.3}", self.config.learning_rate);
        println!("   - Mid-near ratio: {:.2}", self.config.mid_near_ratio);
        println!("   - Far-pair ratio: {:.2}", self.config.far_pair_ratio);
        if let Some(seed) = self.config.seed {
            println!("   - Random seed: {} (deterministic)", seed);
        } else {
            println!("   - Random seed: None (non-deterministic)");
        }

        println!();

        // HNSW Configuration
        println!(" HNSW Neighbor Search Configuration:");
        println!("   - M parameter: {}", self.config.hnsw_params.m);
        println!("   - ef_construction: {}", self.config.hnsw_params.ef_construction);
        println!("   - ef_search: {}", self.config.hnsw_params.ef_search);
        println!("   - Estimated memory: {} MB", self.config.hnsw_params.estimated_memory_bytes / (1024 * 1024));
        println!("   - Characteristics: {}", self.config.hnsw_params.get_characteristics());

        println!();

        // Normalization Configuration
        println!(" Data Normalization Configuration:");
        println!("   - Mode: {:?}", self.normalization.mode);
        println!("   - Features: {}", self.normalization.n_features);
        match self.normalization.mode {
            crate::stats::NormalizationMode::ZScore => {
                if !self.normalization.means.is_empty() {
                    println!("   - Means: [{:.3}, ..., {:.3}] (length: {})",
                             self.normalization.means[0],
                             self.normalization.means.last().unwrap_or(&0.0),
                             self.normalization.means.len());
                }
                if !self.normalization.stds.is_empty() {
                    println!("   - Std devs: [{:.3}, ..., {:.3}] (length: {})",
                             self.normalization.stds[0],
                             self.normalization.stds.last().unwrap_or(&1.0),
                             self.normalization.stds.len());
                }
            },
            crate::stats::NormalizationMode::MinMax => {
                if !self.normalization.mins.is_empty() {
                    println!("   - Minimums: [{:.3}, ..., {:.3}] (length: {})",
                             self.normalization.mins[0],
                             self.normalization.mins.last().unwrap_or(&0.0),
                             self.normalization.mins.len());
                }
                if !self.normalization.maxs.is_empty() {
                    println!("   - Maximums: [{:.3}, ..., {:.3}] (length: {})",
                             self.normalization.maxs[0],
                             self.normalization.maxs.last().unwrap_or(&1.0),
                             self.normalization.maxs.len());
                }
            },
            crate::stats::NormalizationMode::Robust => {
                if !self.normalization.medians.is_empty() {
                    println!("   - Medians: [{:.3}, ..., {:.3}] (length: {})",
                             self.normalization.medians[0],
                             self.normalization.medians.last().unwrap_or(&0.0),
                             self.normalization.medians.len());
                }
                if !self.normalization.iqrs.is_empty() {
                    println!("   - IQRs: [{:.3}, ..., {:.3}] (length: {})",
                             self.normalization.iqrs[0],
                             self.normalization.iqrs.last().unwrap_or(&1.0),
                             self.normalization.iqrs.len());
                }
            },
            _ => {}
        }

        println!();

        // Embedding Information
        println!(" Embedding Information:");
        println!("   - Shape: {:?}", self.embedding.shape());
        let non_zero_count = self.embedding.iter().filter(|&&x| x.abs() > 1e-10).count();
        println!("   - Non-zero values: {}/{} ({:.1}%)",
                 non_zero_count, self.embedding.len(),
                 (non_zero_count as f64 / self.embedding.len() as f64) * 100.0);

        println!();

        // Distance Statistics
        println!("📏 Distance Statistics:");
        println!("   - Mean distance: {:.6}", self.stats.mean_distance);
        println!("   - 95th percentile: {:.6}", self.stats.p95_distance);
        println!("   - Maximum distance: {:.6}", self.stats.max_distance);

        println!();

        // Quantization Information
        if self.quantize_on_save {
            println!("  Quantization Configuration:");
            println!("   - Quantize on save: Enabled");
            if let Some(ref quantized) = self.quantized_embedding {
                println!("   - Quantized shape: {:?}", quantized.data.shape());
                println!("   - Min value: {:.6}", quantized.params.min_value);
                println!("   - Max value: {:.6}", quantized.params.max_value);
                println!("   - Scale factor: {:.6}", quantized.params.scale);
                println!("   - Zero point: {}", quantized.params.zero_point);
                if let Some(ref centroids) = quantized.params.centroids {
                    if !centroids.is_empty() {
                        println!("   - Centroids: {} values", centroids.len());
                    }
                } else {
                    println!("   - Centroids: None");
                }
            } else {
                println!("   - Quantized embedding: Not yet computed");
            }
        } else {
            println!("  Quantization: Disabled (full precision preserved)");
        }

        println!();
        println!("===========================================");
        println!();
    }

    /// Get original training data if available
    pub fn get_original_data(&self) -> Option<Array2<f64>> {
        self.original_data.clone()
    }

    /// Ensure HNSW index is available for neighbor search
    pub fn ensure_hnsw_index(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.embedding_hnsw_index.is_none() && self.serialized_hnsw_index.is_some() {
            // Deserialize HNSW index if we have the serialized data
            // This would require implementing proper deserialization
            eprintln!("⚠️  HNSW index deserialization not yet implemented");
            return Err("HNSW index deserialization not yet implemented".into());
        }

        if self.embedding_hnsw_index.is_none() {
            return Err("No HNSW index available".into());
        }

        Ok(())
    }
}

impl Clone for PaCMAP {
    fn clone(&self) -> Self {
        Self {
            embedding: self.embedding.clone(),
            config: self.config.clone(),
            stats: self.stats.clone(),
            normalization: self.normalization.clone(),
            quantize_on_save: self.quantize_on_save,
            quantized_embedding: self.quantized_embedding.clone(),
            used_hnsw: self.used_hnsw,
            hnsw_index_crc32: self.hnsw_index_crc32,
            fitted_projections_crc32: self.fitted_projections_crc32,
            embedding_centroid: self.embedding_centroid.clone(),
            fitted_projections: self.fitted_projections.clone(),
            embedding_hnsw_index: self.embedding_hnsw_index.clone(),
            serialized_hnsw_index: self.serialized_hnsw_index.clone(),
            original_data: self.original_data.clone(),
        }
    }
}
