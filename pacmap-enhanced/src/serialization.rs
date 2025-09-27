use serde::{Serialize, Deserialize};
use zstd::{stream::encode_all as compress, stream::decode_all as decompress};
use std::fs::File;
use std::io::{Read, Write};
use ndarray::Array2;
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
    pub min_dist: f64,
    pub mid_near_ratio: f64,
    pub far_pair_ratio: f64,
    pub seed: Option<u64>,
    /// HNSW parameters for neighbor search optimization
    pub hnsw_params: HnswParams,
}

impl Default for PacMAPConfig {
    fn default() -> Self {
        Self {
            n_neighbors: 10,
            embedding_dim: 2,
            n_epochs: 450,
            learning_rate: 1.0,
            min_dist: 0.1,
            mid_near_ratio: 0.5,
            far_pair_ratio: 0.5,
            seed: None,
            hnsw_params: HnswParams::default(),
        }
    }
}

#[derive(Serialize, Deserialize, Clone)]
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
}

#[derive(Serialize, Deserialize, Clone)]
pub struct DistanceStats {
    pub mean_distance: f64,
    pub p95_distance: f64,
    pub max_distance: f64,
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
        println!("   📊 File size: {} bytes ({:.2} KB)", file_size, file_size as f64 / 1024.0);

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
        println!("   📊 File size: {} bytes ({:.2} KB)", file_size, file_size as f64 / 1024.0);
        if self.quantize_on_save {
            println!("   🗜️  Quantization enabled - model size optimized");
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
        println!("   📊 File size: {} bytes ({:.2} KB)", file_size, file_size as f64 / 1024.0);

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
        println!("🎛️  Core PacMAP Configuration:");
        println!("   - Embedding dimensions: {}D", self.config.embedding_dim);
        println!("   - Number of neighbors: {}", self.config.n_neighbors);
        println!("   - Number of epochs: {}", self.config.n_epochs);
        println!("   - Learning rate: {:.3}", self.config.learning_rate);
        println!("   - Minimum distance: {:.3}", self.config.min_dist);
        println!("   - Mid-near ratio: {:.2}", self.config.mid_near_ratio);
        println!("   - Far-pair ratio: {:.2}", self.config.far_pair_ratio);
        if let Some(seed) = self.config.seed {
            println!("   - Random seed: {} (deterministic)", seed);
        } else {
            println!("   - Random seed: None (non-deterministic)");
        }

        println!();

        // HNSW Configuration
        println!("🔍 HNSW Neighbor Search Configuration:");
        println!("   - M parameter: {}", self.config.hnsw_params.m);
        println!("   - ef_construction: {}", self.config.hnsw_params.ef_construction);
        println!("   - ef_search: {}", self.config.hnsw_params.ef_search);
        println!("   - Estimated memory: {} MB", self.config.hnsw_params.estimated_memory_bytes / (1024 * 1024));
        println!("   - Characteristics: {}", self.config.hnsw_params.get_characteristics());

        println!();

        // Normalization Configuration
        println!("📊 Data Normalization Configuration:");
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
        println!("🎯 Embedding Information:");
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
            println!("🗜️  Quantization Configuration:");
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
            println!("🗜️  Quantization: Disabled (full precision preserved)");
        }

        println!();
        println!("===========================================");
        println!();
    }
}
