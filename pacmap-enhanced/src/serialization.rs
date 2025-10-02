use serde::{Serialize, Deserialize};
use zstd::{stream::encode_all as compress, stream::decode_all as decompress};
use std::fs::File;
use std::io::{Read, Write};
use ndarray::{Array2, Array1};
use crate::quantize::{quantize_embedding_linear, QuantizedEmbedding, dequantize_embedding};
use crate::stats::NormalizationParams;
use crate::hnsw_params::HnswParams;

#[cfg(feature = "use_hnsw")]
use hnsw_rs::hnsw::Hnsw;
#[cfg(feature = "use_hnsw")]
use hnsw_rs::dist::DistL2;

// Import is_verbose function from lib.rs
use crate::is_verbose;

// Define vprint macro locally since macros don't auto-import
macro_rules! vprint {
    ($($arg:tt)*) => {
        if is_verbose() {
            eprintln!($($arg)*);
        }
    };
}

/// A serializable container for all the data needed to rebuild an HNSW index.
/// This is a robust alternative to trying to serialize the Hnsw struct itself.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct SerializableHnswIndex {
    /// The original data points that were inserted into the index.
    pub data: Vec<Vec<f32>>,
    /// The HNSW parameters used to build the index.
    pub params: HnswParams,
    /// The number of layers in the original index.
    pub max_layer: usize,
}

/// Statistics calculated for a transformed data point to assess its relationship
/// to the original training data's embedding. Used for "No Man's Land" detection.
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct TransformStats {
    /// The low-dimensional coordinates of the transformed point.
    pub coordinates: Array1<f64>,

    /// The distance to the single closest neighbor in the training embedding.
    /// A primary indicator of outlierness.
    pub distance_to_closest_neighbor: f64,

    /// The mean distance to the k nearest neighbors in the training embedding.
    /// Provides a more stable measure of local density.
    pub mean_distance_to_k_neighbors: f64,

    /// The distance from the point to the centroid of the entire training embedding.
    /// A global measure of how far the point is from the "center" of the data.
    pub distance_to_training_centroid: f64,
}

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
    /// Whether HNSW was actually used during fitting (vs exact KNN)
    #[serde(default)]
    pub used_hnsw: bool,
    /// Force direct KNN instead of HNSW regardless of dataset size
    #[serde(default)]
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
            far_pair_ratio: 2.0,
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

    /// TRANSFORM SUPPORT: Original high-dimensional training data (quantized)
    /// Used for initial neighbor search when transforming new points
    #[serde(default)]
    pub original_data: Option<QuantizedEmbedding>,

    /// TRANSFORM SUPPORT: Fitted low-dimensional projections
    /// Used for final neighbor refinement in embedding space
    /// This is the same as embedding but stored separately for clarity
    #[serde(default)]
    pub fitted_projections: Option<Array2<f64>>,

    /// The centroid of the fitted embedding space.
    /// Used for global outlierness detection during transform ("No Man's Land" detection).
    #[serde(default)]
    pub embedding_centroid: Option<Array1<f64>>,

    /// The HNSW index for fast neighbor searches in the original data space.
    /// It is not serialized and will be rebuilt on load if needed.
    #[serde(skip)]
    #[cfg(feature = "use_hnsw")]
    pub hnsw_index: Option<hnsw_rs::hnsw::Hnsw<'static, f32, hnsw_rs::dist::DistL2>>,

    /// The HNSW index for fast neighbor searches in the embedding space.
    /// It is not serialized and will be rebuilt on load if needed.
    #[serde(skip)]
    #[cfg(feature = "use_hnsw")]
    pub embedding_hnsw_index: Option<hnsw_rs::hnsw::Hnsw<'static, f32, hnsw_rs::dist::DistL2>>,

    /// The raw bytes of a serialized HNSW index for original data.
    /// Used for massive datasets where rebuilding is too slow.
    #[serde(default)]
    pub serialized_hnsw_index: Option<Vec<u8>>,

    /// The raw bytes of a serialized HNSW index for embedding data.
    /// Used for massive datasets where rebuilding is too slow.
    #[serde(default)]
    pub serialized_embedding_hnsw_index: Option<Vec<u8>>,

    /// CRC32 checksum of the serialized HNSW index for original data.
    /// Used for integrity validation when loading.
    #[serde(default)]
    pub hnsw_index_crc32: Option<u32>,

    /// CRC32 checksum of the serialized HNSW index for embedding data.
    /// Used for integrity validation when loading.
    #[serde(default)]
    pub embedding_hnsw_index_crc32: Option<u32>,
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

    /// Store original training data and fitted projections for transform support
    pub fn store_transform_data(&mut self, original_data: &Array2<f64>, fitted_projections: &Array2<f64>) {
        // Store quantized original data for efficient storage
        self.original_data = Some(quantize_embedding_linear(original_data));

        // Store fitted projections (low-dim embeddings)
        self.fitted_projections = Some(fitted_projections.clone());
    }

    /// Get dequantized original training data for transforms
    pub fn get_original_data(&self) -> Option<Array2<f64>> {
        self.original_data.as_ref().map(|quantized| {
            dequantize_embedding(quantized)
        })
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

    /// A helper to ensure the HNSW index is available, rebuilding it if necessary
    /// (e.g., after loading from disk where it wasn't serialized).
    #[cfg(feature = "use_hnsw")]
    pub fn ensure_hnsw_index(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.hnsw_index.is_some() {
            return Ok(());
        }

        if let Some(ref serialized_bytes) = self.serialized_hnsw_index {
            // FAST PATH: Load from serialized index
            vprint!("🚀 Loading HNSW index from serialized data (fast path)");
            let serialized_index = deserialize_hnsw_from_bytes(serialized_bytes)?;

            // Build HNSW directly from serialized data
            let n_samples = serialized_index.data.len();
            let hnsw = hnsw_rs::hnsw::Hnsw::<f32, hnsw_rs::dist::DistL2>::new(
                serialized_index.params.m,
                n_samples,
                serialized_index.max_layer,
                serialized_index.params.ef_construction,
                hnsw_rs::dist::DistL2{}
            );

            // Insert all data points
            for (i, point) in serialized_index.data.iter().enumerate() {
                hnsw.insert((point, i));
            }

            self.hnsw_index = Some(hnsw);
        } else {
            // SLOW PATH: Rebuild from stored data for normal datasets
            vprint!("🔧 HNSW index not found, rebuilding from stored training data...");
            let original_data = self.get_original_data().ok_or("Cannot rebuild HNSW index: original training data not stored")?;
            let (n_samples, _) = original_data.dim();

            let hnsw_params = &self.config.hnsw_params;
            let points: Vec<Vec<f32>> = (0..n_samples)
                .map(|i| original_data.row(i).iter().map(|&x| x as f32).collect())
                .collect();

            let max_layer = ((n_samples as f32).ln() / (hnsw_params.m as f32).ln()).ceil() as usize + 1;
            let max_layer = max_layer.min(32).max(4);

            let hnsw = hnsw_rs::hnsw::Hnsw::<f32, hnsw_rs::dist::DistL2>::new(
                hnsw_params.m,
                n_samples,
                max_layer,
                hnsw_params.ef_construction,
                hnsw_rs::dist::DistL2{}
            );

            let data_with_id: Vec<(&[f32], usize)> = points.iter().enumerate().map(|(i, p)| (p.as_slice(), i)).collect();

            #[cfg(feature = "parallel")]
            {
                hnsw.parallel_insert(&data_with_id);
            }
            #[cfg(not(feature = "parallel"))]
            {
                for (point, i) in data_with_id {
                    hnsw.insert((&point.to_vec(), i));
                }
            }

            self.hnsw_index = Some(hnsw);
            vprint!("✅ HNSW index rebuilt successfully.");
        }
        Ok(())
    }

    /// A helper to ensure the embedding HNSW index is available, rebuilding it if necessary
    #[cfg(feature = "use_hnsw")]
    pub fn ensure_embedding_hnsw_index(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.embedding_hnsw_index.is_some() {
            return Ok(());
        }

        if let Some(ref serialized_bytes) = self.serialized_embedding_hnsw_index {
            // FAST PATH: Load from serialized index
            vprint!("🚀 Loading embedding HNSW index from serialized data (fast path)");
            let serialized_index = deserialize_hnsw_from_bytes(serialized_bytes)?;

            // Build HNSW directly from serialized data
            let n_samples = serialized_index.data.len();
            let hnsw = hnsw_rs::hnsw::Hnsw::<f32, hnsw_rs::dist::DistL2>::new(
                serialized_index.params.m,
                n_samples,
                serialized_index.max_layer,
                serialized_index.params.ef_construction,
                hnsw_rs::dist::DistL2{}
            );

            // Insert all data points
            for (i, point) in serialized_index.data.iter().enumerate() {
                hnsw.insert((point, i));
            }

            self.embedding_hnsw_index = Some(hnsw);
        } else {
            // SLOW PATH: Rebuild from fitted projections
            vprint!("🔧 Embedding HNSW index not found, rebuilding from fitted projections...");
            let fitted_projections = self.fitted_projections.as_ref().ok_or("Cannot rebuild embedding HNSW index: fitted projections not stored")?;
            let (n_samples, _) = fitted_projections.dim();

            let hnsw_params = &self.config.hnsw_params;
            let points: Vec<Vec<f32>> = (0..n_samples)
                .map(|i| fitted_projections.row(i).iter().map(|&x| x as f32).collect())
                .collect();

            let max_layer = ((n_samples as f32).ln() / (hnsw_params.m as f32).ln()).ceil() as usize + 1;
            let max_layer = max_layer.min(32).max(4);

            let hnsw = hnsw_rs::hnsw::Hnsw::<f32, hnsw_rs::dist::DistL2>::new(
                hnsw_params.m,
                n_samples,
                max_layer,
                hnsw_params.ef_construction,
                hnsw_rs::dist::DistL2{}
            );

            let data_with_id: Vec<(&[f32], usize)> = points.iter().enumerate().map(|(i, p)| (p.as_slice(), i)).collect();

            #[cfg(feature = "parallel")]
            {
                hnsw.parallel_insert(&data_with_id);
            }
            #[cfg(not(feature = "parallel"))]
            {
                for (point, i) in data_with_id {
                    hnsw.insert((&point.to_vec(), i));
                }
            }

            self.embedding_hnsw_index = Some(hnsw);
            vprint!("✅ Embedding HNSW index rebuilt successfully.");
        }
        Ok(())
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
            original_data: self.original_data.clone(),
            fitted_projections: self.fitted_projections.clone(),
            embedding_centroid: self.embedding_centroid.clone(),
            #[cfg(feature = "use_hnsw")]
            hnsw_index: None, // Don't clone HNSW indices - they'll be rebuilt on demand
            #[cfg(feature = "use_hnsw")]
            embedding_hnsw_index: None,
            serialized_hnsw_index: self.serialized_hnsw_index.clone(),
            serialized_embedding_hnsw_index: self.serialized_embedding_hnsw_index.clone(),
            hnsw_index_crc32: self.hnsw_index_crc32,
            embedding_hnsw_index_crc32: self.embedding_hnsw_index_crc32,
        }
    }
}

/// Create serialized HNSW index using binary MessagePack format (much faster than JSON)
#[cfg(feature = "use_hnsw")]
pub fn custom_serialize_hnsw(data: &Array2<f64>, hnsw_params: &HnswParams, max_layer: usize) -> Result<SerializableHnswIndex, Box<dyn std::error::Error>> {
    let data_f32: Vec<Vec<f32>> = data.rows()
        .into_iter()
        .map(|row| row.iter().map(|&x| x as f32).collect())
        .collect();

    let serializable = SerializableHnswIndex {
        data: data_f32,
        params: hnsw_params.clone(),
        max_layer,
    };

    Ok(serializable)
}

/// Serialize SerializableHnswIndex to binary bytes using MessagePack
pub fn serialize_hnsw_to_bytes(serializable_index: &SerializableHnswIndex) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let serialized_bytes = rmp_serde::to_vec(serializable_index)?;
    Ok(serialized_bytes)
}

/// Deserialize SerializableHnswIndex from binary bytes using MessagePack
pub fn deserialize_hnsw_from_bytes(serialized_bytes: &[u8]) -> Result<SerializableHnswIndex, Box<dyn std::error::Error>> {
    let serializable_index: SerializableHnswIndex = rmp_serde::from_slice(serialized_bytes)?;
    Ok(serializable_index)
}

/// Load HNSW from serialized index (used by transform functions)
#[cfg(feature = "use_hnsw")]
pub fn load_hnsw_from_serialized(serialized_index: &SerializableHnswIndex) -> Result<Hnsw<'_, f32, DistL2>, Box<dyn std::error::Error>> {
    let n_samples = serialized_index.data.len();

    let hnsw = Hnsw::<f32, DistL2>::new(
        serialized_index.params.m,
        n_samples,
        serialized_index.max_layer,
        serialized_index.params.ef_construction,
        DistL2{}
    );

    // Insert all data points
    for (i, point) in serialized_index.data.iter().enumerate() {
        hnsw.insert((point, i));
    }

    Ok(hnsw)
}
